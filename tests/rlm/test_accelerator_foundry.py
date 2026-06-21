"""G1: Azure AI Foundry navigation-accelerator tier. No network — all creds
come from env (monkeypatched) and no probe is performed for Foundry."""
from __future__ import annotations

import types
import pytest

from backend.agents.rlm.accelerator import (
    AcceleratorEndpoint,
    AcceleratorError,
    build_accelerator_client,
    resolve_accelerator,
)


def _set_foundry(monkeypatch, endpoint="https://x.services.ai.azure.com",
                 deployment="grok-4.3", key="foundry-key"):
    monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", endpoint)
    monkeypatch.setenv("AZURE_FOUNDRY_DEPLOYMENT", deployment)
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", key)


def _clear_foundry(monkeypatch):
    for k in ("AZURE_FOUNDRY_ENDPOINT", "AZURE_FOUNDRY_DEPLOYMENT", "AZURE_FOUNDRY_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    # Neutralise Settings/.env — the real .env may carry Foundry creds.
    monkeypatch.setattr("backend.config.get_settings", lambda *a, **k: types.SimpleNamespace())


@pytest.mark.parametrize("mode", ["azure-foundry", "foundry", "grok"])
def test_explicit_foundry_returns_endpoint_no_probe(monkeypatch, mode):
    _set_foundry(monkeypatch)
    ep = resolve_accelerator(mode)
    assert isinstance(ep, AcceleratorEndpoint)
    assert ep.base_url == "https://x.services.ai.azure.com/openai/v1"
    assert ep.model == "grok-4.3"
    assert ep.api_key == "foundry-key"
    assert ep.kind == "foundry"
    assert ep.is_azure is False


def test_explicit_foundry_raises_on_missing_creds(monkeypatch):
    _clear_foundry(monkeypatch)
    with pytest.raises(AcceleratorError, match="AZURE_FOUNDRY"):
        resolve_accelerator("foundry")


def test_explicit_foundry_raises_on_missing_deployment(monkeypatch):
    monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", "https://x.services.ai.azure.com")
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "foundry-key")
    monkeypatch.delenv("AZURE_FOUNDRY_DEPLOYMENT", raising=False)
    monkeypatch.setattr("backend.config.get_settings", lambda *a, **k: types.SimpleNamespace())
    with pytest.raises(AcceleratorError, match="AZURE_FOUNDRY_DEPLOYMENT"):
        resolve_accelerator("foundry")


def test_auto_picks_foundry_when_creds_present_and_no_gpu(monkeypatch):
    monkeypatch.setattr(
        "backend.services.runtime.gpu_resolution.host_supports_nvidia_gpu",
        lambda: False,
    )
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    _set_foundry(monkeypatch)
    ep = resolve_accelerator("auto")
    assert ep is not None
    assert ep.kind == "foundry"


def test_unset_foundry_off_is_unchanged(monkeypatch):
    _clear_foundry(monkeypatch)
    monkeypatch.setattr(
        "backend.services.runtime.gpu_resolution.host_supports_nvidia_gpu",
        lambda: False,
    )
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    assert resolve_accelerator("off") is None
    assert resolve_accelerator("auto") is None


def test_build_client_for_foundry_uses_openai_client(monkeypatch):
    from backend.services.context.workspace.tools.openai_client import OpenAILlmClient

    ep = AcceleratorEndpoint(
        base_url="https://x.services.ai.azure.com/openai/v1",
        model="grok-4.3",
        api_key="foundry-key",
        kind="foundry",
        is_azure=False,
    )
    client = build_accelerator_client(ep)
    assert isinstance(client, OpenAILlmClient)
    assert hasattr(client, "complete")
