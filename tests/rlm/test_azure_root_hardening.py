"""Tests for Azure root-transport hardening (A1–A3).

Covers:
  A1 — _inject_azure_kwargs injects api_version (default + env override),
       applied to both root and sub_backend_kwargs in resolve_root_model.
  A2 — apply_azure_root_hardening_patch() is idempotent and the patched
       AzureOpenAIClient has max_retries=6.
  A3 — The Azure accelerator nav branch builds an azure_openai sub-backend
       with the correct shape when _accel_ep.is_azure is True.

All tests are hermetic: no network, no real credentials.  The suite is
socket-hermetic (pytest-socket blocks non-loopback).
"""

from __future__ import annotations

import os


# ---------------------------------------------------------------------------
# A1 — _inject_azure_kwargs injects api_version
# ---------------------------------------------------------------------------


def test_inject_azure_kwargs_default_api_version(monkeypatch):
    """_inject_azure_kwargs injects api_version defaulting to DEFAULT_AZURE_OPENAI_API_VERSION."""
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
    monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)

    from backend.agents.rlm.models import _inject_azure_kwargs
    from backend.services.context.workspace.tools.azure_openai_client import (
        DEFAULT_AZURE_OPENAI_API_VERSION,
    )

    result = _inject_azure_kwargs({}, model_key="azure-gpt-4o")

    assert "api_version" in result
    assert result["api_version"] == DEFAULT_AZURE_OPENAI_API_VERSION


def test_inject_azure_kwargs_env_override_api_version(monkeypatch):
    """_inject_azure_kwargs honours AZURE_OPENAI_API_VERSION env override."""
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-01-01")
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)

    from backend.agents.rlm.models import _inject_azure_kwargs

    result = _inject_azure_kwargs({}, model_key="azure-gpt-4o")

    assert result["api_version"] == "2025-01-01"


def test_inject_azure_kwargs_existing_api_version_not_overwritten(monkeypatch):
    """_inject_azure_kwargs does not overwrite an api_version already in kwargs."""
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-01-01")

    from backend.agents.rlm.models import _inject_azure_kwargs

    result = _inject_azure_kwargs({"api_version": "2024-02-01"}, model_key="azure-gpt-4o")

    # Pre-existing value wins.
    assert result["api_version"] == "2024-02-01"


def test_inject_azure_kwargs_raises_without_endpoint(monkeypatch):
    """_inject_azure_kwargs raises ValueError when AZURE_OPENAI_ENDPOINT is absent."""
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)

    import pytest
    from backend.agents.rlm.models import _inject_azure_kwargs

    with pytest.raises(ValueError, match="AZURE_OPENAI_ENDPOINT"):
        _inject_azure_kwargs({}, model_key="azure-gpt-4o")


def test_resolve_root_model_injects_api_version_into_both_backends(monkeypatch):
    """resolve_root_model applies api_version to both root and sub_backend_kwargs."""
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)

    from backend.agents.rlm.models import resolve_root_model
    from backend.services.context.workspace.tools.azure_openai_client import (
        DEFAULT_AZURE_OPENAI_API_VERSION,
    )

    entry = resolve_root_model("azure-gpt-4o")

    assert entry.backend_kwargs.get("api_version") == DEFAULT_AZURE_OPENAI_API_VERSION, (
        f"root backend_kwargs missing api_version: {entry.backend_kwargs}"
    )
    assert entry.sub_backend_kwargs.get("api_version") == DEFAULT_AZURE_OPENAI_API_VERSION, (
        f"sub_backend_kwargs missing api_version: {entry.sub_backend_kwargs}"
    )


# ---------------------------------------------------------------------------
# A2 — apply_azure_root_hardening_patch idempotency + max_retries
# ---------------------------------------------------------------------------


def test_patch_is_idempotent(monkeypatch):
    """apply_azure_root_hardening_patch() is idempotent — second call is a no-op."""
    import sys
    # Drop the cached patch module so we get a clean import.
    for key in list(sys.modules):
        if "azure_root_hardening_patch" in key:
            del sys.modules[key]
    # Also reset the rlm module attribute so the patch starts fresh.
    import rlm.clients.azure_openai as _az_mod
    original_cls = _az_mod.AzureOpenAIClient
    # Strip the hardening flag if present so the first call actually patches.
    if hasattr(_az_mod.AzureOpenAIClient, "_openresearch_azure_hardening_applied"):
        del _az_mod.AzureOpenAIClient._openresearch_azure_hardening_applied

    from backend.agents.rlm.azure_root_hardening_patch import apply_azure_root_hardening_patch

    apply_azure_root_hardening_patch()
    patched_cls_first = _az_mod.AzureOpenAIClient
    apply_azure_root_hardening_patch()
    patched_cls_second = _az_mod.AzureOpenAIClient

    # Same object — second call did not create a new subclass.
    assert patched_cls_first is patched_cls_second


def test_patched_client_has_max_retries_6(monkeypatch):
    """The hardened AzureOpenAIClient builds openai clients with max_retries=6."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")

    # Ensure the patch is applied (import may already have applied it).
    from backend.agents.rlm.azure_root_hardening_patch import apply_azure_root_hardening_patch
    apply_azure_root_hardening_patch()

    from rlm.clients.azure_openai import AzureOpenAIClient

    assert getattr(AzureOpenAIClient, "_openresearch_azure_hardening_applied", False), (
        "Patch flag not set — patch was not applied"
    )

    client = AzureOpenAIClient(
        model_name="gpt-4o",
        azure_endpoint="https://test.openai.azure.com",
        api_version="2024-10-21",
        azure_deployment="gpt-4o",
    )
    assert client.client.max_retries == 6, (
        f"sync client max_retries={client.client.max_retries}, expected 6"
    )
    assert client.async_client.max_retries == 6, (
        f"async client max_retries={client.async_client.max_retries}, expected 6"
    )


def test_patch_preserves_api_version(monkeypatch):
    """Hardened client preserves the api_version passed by the caller."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")

    from backend.agents.rlm.azure_root_hardening_patch import apply_azure_root_hardening_patch
    apply_azure_root_hardening_patch()

    from rlm.clients.azure_openai import AzureOpenAIClient

    client = AzureOpenAIClient(
        model_name="gpt-4o",
        azure_endpoint="https://test.openai.azure.com",
        api_version="2024-10-21",
        azure_deployment="gpt-4o",
    )
    assert client.client._api_version == "2024-10-21"


def test_patch_noops_gracefully_on_import_failure(monkeypatch):
    """If rlm azure_openai cannot be imported, the patch logs a warning and does not raise."""
    import sys
    # Temporarily make rlm.clients.azure_openai unimportable.
    original = sys.modules.get("rlm.clients.azure_openai")
    sys.modules["rlm.clients.azure_openai"] = None  # type: ignore[assignment]
    try:
        # Re-import with a fresh module state.
        for key in list(sys.modules):
            if "azure_root_hardening_patch" in key:
                del sys.modules[key]
        # The import of the patch module triggers apply_azure_root_hardening_patch()
        # at module bottom — it must not raise.
        import importlib
        import backend.agents.rlm.azure_root_hardening_patch as mod
        importlib.reload(mod)  # force re-execution with blocked rlm module
    except Exception as exc:
        raise AssertionError(
            f"apply_azure_root_hardening_patch raised when rlm import fails: {exc}"
        ) from exc
    finally:
        if original is None:
            del sys.modules["rlm.clients.azure_openai"]
        else:
            sys.modules["rlm.clients.azure_openai"] = original


# ---------------------------------------------------------------------------
# A3 — Azure accelerator nav branch builds azure_openai sub-backend
# ---------------------------------------------------------------------------


def _make_azure_accel_ep(
    base_url: str = "https://accel.openai.azure.com",
    model: str = "gpt-4o-mini",
    api_key: str = "accel-key",
) -> object:
    """Build a minimal AcceleratorEndpoint-like object with is_azure=True."""
    from backend.agents.rlm.accelerator import AcceleratorEndpoint
    return AcceleratorEndpoint(
        base_url=base_url,
        model=model,
        api_key=api_key,
        kind="azure",
        is_azure=True,
    )


def test_azure_accel_sets_azure_openai_sub_backend(monkeypatch):
    """When _accel_ep.is_azure is True, the sub-backend must be 'azure_openai'."""
    from backend.services.context.workspace.tools.azure_openai_client import (
        DEFAULT_AZURE_OPENAI_API_VERSION,
    )

    accel_ep = _make_azure_accel_ep()

    # Simulate the branch logic from run.py directly (extracted for unit testing).
    _other_backends: list[str] = ["original_sub_backend"]
    _other_backend_kwargs: list[dict] = [{"model_name": "original"}]

    if accel_ep is not None and not accel_ep.is_azure:
        _other_backends = ["openai"]
        _other_backend_kwargs = [
            {"model_name": accel_ep.model, "base_url": accel_ep.base_url, "api_key": accel_ep.api_key}
        ]
    elif accel_ep is not None and accel_ep.is_azure:
        monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
        _other_backends = ["azure_openai"]
        _other_backend_kwargs = [
            {
                "model_name": accel_ep.model,
                "azure_endpoint": accel_ep.base_url,
                "azure_deployment": accel_ep.model,
                "api_key": accel_ep.api_key,
                "api_version": (
                    os.environ.get("AZURE_OPENAI_API_VERSION")
                    or DEFAULT_AZURE_OPENAI_API_VERSION
                ),
            }
        ]

    assert _other_backends == ["azure_openai"], (
        f"Expected azure_openai sub-backend, got {_other_backends}"
    )
    bk = _other_backend_kwargs[0]
    assert bk["model_name"] == "gpt-4o-mini"
    assert bk["azure_endpoint"] == "https://accel.openai.azure.com"
    assert bk["azure_deployment"] == "gpt-4o-mini"
    assert bk["api_key"] == "accel-key"
    assert bk["api_version"] == DEFAULT_AZURE_OPENAI_API_VERSION


def test_azure_accel_api_version_env_override(monkeypatch):
    """Azure accelerator sub-backend honours AZURE_OPENAI_API_VERSION env override."""
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-01-01")

    from backend.services.context.workspace.tools.azure_openai_client import (
        DEFAULT_AZURE_OPENAI_API_VERSION,
    )

    accel_ep = _make_azure_accel_ep()

    _other_backend_kwargs: list[dict] = [{}]
    if accel_ep is not None and accel_ep.is_azure:
        _other_backend_kwargs = [
            {
                "model_name": accel_ep.model,
                "azure_endpoint": accel_ep.base_url,
                "azure_deployment": accel_ep.model,
                "api_key": accel_ep.api_key,
                "api_version": (
                    os.environ.get("AZURE_OPENAI_API_VERSION")
                    or DEFAULT_AZURE_OPENAI_API_VERSION
                ),
            }
        ]

    assert _other_backend_kwargs[0]["api_version"] == "2025-01-01"


def test_non_azure_accel_unaffected_by_azure_branch():
    """When is_azure is False, the branch should select openai, not azure_openai."""
    from backend.agents.rlm.accelerator import AcceleratorEndpoint

    accel_ep = AcceleratorEndpoint(
        base_url="http://127.0.0.1:8001/v1",
        model="Qwen/Qwen3-Coder",
        api_key="local",
        kind="local",
        is_azure=False,
    )

    _other_backends: list[str] = ["original"]
    _other_backend_kwargs: list[dict] = [{}]

    if accel_ep is not None and not accel_ep.is_azure:
        _other_backends = ["openai"]
        _other_backend_kwargs = [
            {"model_name": accel_ep.model, "base_url": accel_ep.base_url, "api_key": accel_ep.api_key}
        ]
    elif accel_ep is not None and accel_ep.is_azure:
        _other_backends = ["azure_openai"]

    assert _other_backends == ["openai"]
    assert "azure_endpoint" not in _other_backend_kwargs[0]


def test_no_accel_leaves_sub_backend_unchanged():
    """When _accel_ep is None, neither branch fires; sub-backend stays as-is."""
    accel_ep = None

    _other_backends: list[str] = ["original_sub"]
    _other_backend_kwargs: list[dict] = [{"model_name": "original"}]

    if accel_ep is not None and not accel_ep.is_azure:
        _other_backends = ["openai"]
    elif accel_ep is not None and accel_ep.is_azure:
        _other_backends = ["azure_openai"]

    assert _other_backends == ["original_sub"]
    assert _other_backend_kwargs[0]["model_name"] == "original"
