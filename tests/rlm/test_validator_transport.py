"""build_validator_client — FAIL-CLOSED transport (spec 2026-06-20 §7.2/§7.4).

Unit tests only; every underlying SDK client is mocked, no network/LLM. The
validator is the deliberate inverse of ``build_grader_client``: where the grader
silently falls back to the caller's client on any misconfig, the validator
RAISES — a validator that quietly degrades to the executor-family client gives
zero independent grounding. Covers:

  * backend unset → passthrough (validator role not overridden);
  * backend set + missing creds → ValueError (fail-closed, the load-bearing case);
  * backend set + unknown backend → ValueError;
  * backend=azure + OPENRESEARCH_VALIDATOR_MODEL → the constructed client targets
    the OVERRIDE deployment, not AZURE_OPENAI_DEPLOYMENT (the distinct-deployment
    gap fix — the load-bearing weak-panel case);
  * backend=oauth → constructs without API creds (the funded transport);
  * the existing grader azure path is unchanged when no model override is given.
"""
from __future__ import annotations

import types

import pytest

from backend.agents.rlm.grader_transport import (
    build_transport_client,
    build_validator_client,
)


@pytest.fixture(autouse=True)
def _clear_validator_env(monkeypatch):
    """Every test starts with the validator env unset (hermetic)."""
    monkeypatch.delenv("OPENRESEARCH_VALIDATOR_BACKEND", raising=False)
    monkeypatch.delenv("OPENRESEARCH_VALIDATOR_MODEL", raising=False)


# ---------------------------------------------------------------------------
# 1. backend unset → passthrough (validator role not overridden).
# ---------------------------------------------------------------------------
def test_validator_passthrough_when_backend_unset():
    sentinel = object()
    client, label = build_validator_client(fallback_client=sentinel, fallback_label="exec")
    assert client is sentinel  # unchanged → caller resolves separation=unavailable
    assert label == "exec"


def test_validator_model_only_is_passthrough(monkeypatch):
    # A model override with no backend cannot pick a transport → passthrough
    # (NOT a raise — the validator was not actually requested).
    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_MODEL", "gpt-4.1")
    sentinel = object()
    client, label = build_validator_client(fallback_client=sentinel, fallback_label="exec")
    assert client is sentinel
    assert label == "exec"


# ---------------------------------------------------------------------------
# 2. FAIL-CLOSED — backend set but the transport can't build → ValueError.
# ---------------------------------------------------------------------------
def test_validator_build_fail_closed_missing_creds(monkeypatch):
    # The canonical load-bearing case: backend=azure with no AZURE_OPENAI_*.
    # build_transport_client falls back internally; build_validator_client must
    # convert that fallback into a raise rather than ride the executor client.
    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_BACKEND", "azure")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)
    with pytest.raises(ValueError):
        build_validator_client(fallback_client=object(), fallback_label="exec")


def test_validator_build_fail_closed_unknown_backend(monkeypatch):
    # An unknown backend also falls back inside build_transport_client → raise.
    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_BACKEND", "bananas")
    with pytest.raises(ValueError):
        build_validator_client(fallback_client=object(), fallback_label="exec")


def test_validator_build_fail_closed_construction_error(monkeypatch):
    # Force the azure client constructor to blow up → build_transport_client
    # fails soft to the fallback → build_validator_client must RAISE.
    import backend.services.context.workspace.tools.azure_openai_client as azc

    def _boom(**kwargs):
        raise RuntimeError("no SDK")

    monkeypatch.setattr(azc, "AzureOpenAILlmClient", _boom)
    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_BACKEND", "azure")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    with pytest.raises(ValueError):
        build_validator_client(fallback_client=object(), fallback_label="exec")


# ---------------------------------------------------------------------------
# 3. The azure distinct-deployment gap fix — the validator must hit deployment
# B (OPENRESEARCH_VALIDATOR_MODEL), NOT the executor's AZURE_OPENAI_DEPLOYMENT.
# ---------------------------------------------------------------------------
def _patch_fake_azure(monkeypatch):
    import backend.services.context.workspace.tools.azure_openai_client as azc

    class _FakeAzure:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(azc, "AzureOpenAILlmClient", _FakeAzure)
    return _FakeAzure


def test_azure_validator_uses_distinct_deployment(monkeypatch):
    fake = _patch_fake_azure(monkeypatch)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "execA")  # the executor's deployment
    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_BACKEND", "azure")
    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_MODEL", "valB")  # the validator's

    client, label = build_validator_client(fallback_client=object(), fallback_label="exec")
    assert isinstance(client, fake)
    # The deployment ROUTES the request on Azure → it must be the override valB.
    assert client.kwargs["azure_deployment"] == "valB"
    assert client.kwargs["azure_deployment"] != "execA"
    assert client.kwargs["model"] == "valB"
    assert label == "validator:azure:valB"


def test_transport_azure_deployment_override_through_build_transport_client(monkeypatch):
    # The fix lives in build_transport_client's azure branch; verify it directly:
    # a passed model overrides AZURE_OPENAI_DEPLOYMENT.
    fake = _patch_fake_azure(monkeypatch)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "execA")

    client, _ = build_transport_client(
        backend="azure",
        model="valB",
        fallback_client=object(),
        fallback_label="fb",
        role_label="validator",
    )
    assert isinstance(client, fake)
    assert client.kwargs["azure_deployment"] == "valB"


def test_transport_azure_no_override_keeps_env_deployment(monkeypatch):
    # Back-compat for the existing grader path: no model override → the
    # deployment still comes from AZURE_OPENAI_DEPLOYMENT (behaviour unchanged).
    fake = _patch_fake_azure(monkeypatch)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "execA")

    client, _ = build_transport_client(
        backend="azure",
        model=None,
        fallback_client=object(),
        fallback_label="fb",
        role_label="grader",
    )
    assert isinstance(client, fake)
    assert client.kwargs["azure_deployment"] == "execA"


# ---------------------------------------------------------------------------
# 4. backend=oauth — the funded Claude transport, constructs without API creds.
# ---------------------------------------------------------------------------
def test_validator_oauth_backend_constructs(monkeypatch):
    import backend.services.context.workspace.tools.rlm_query as _rq

    class _FakeOAuth:
        def __init__(self, model=None):
            self.model = model

    monkeypatch.setattr(_rq, "ClaudeLlmClient", _FakeOAuth)
    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_BACKEND", "oauth")
    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_MODEL", "claude-opus-4-8")

    client, label = build_validator_client(fallback_client=object(), fallback_label="exec")
    assert isinstance(client, _FakeOAuth)
    assert client.model == "claude-opus-4-8"
    assert label == "validator:oauth:claude-opus-4-8"


def test_validator_azure_foundry_missing_creds_fail_closed(monkeypatch):
    # Foundry with no creds falls back inside build_transport_client → raise.
    monkeypatch.setenv("OPENRESEARCH_VALIDATOR_BACKEND", "azure-foundry")
    monkeypatch.delenv("AZURE_FOUNDRY_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_FOUNDRY_DEPLOYMENT", raising=False)
    monkeypatch.setattr(
        "backend.config.get_settings", lambda *a, **k: types.SimpleNamespace()
    )
    with pytest.raises(ValueError):
        build_validator_client(fallback_client=object(), fallback_label="exec")
