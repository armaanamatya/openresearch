"""G5: models.py routes Foundry endpoint normalization through the SINGLE
canonical foundry_endpoint.resolve_foundry_credentials — no local re-impl.
Behaviour-preserving: kwargs byte-identical, fail-fast on missing
endpoint/deployment kept, api_key injected ONCE (upstream)."""
from __future__ import annotations

import pytest

import backend.agents.rlm.models as models
from backend.agents.rlm.models import resolve_root_model


def _set_foundry(monkeypatch,
                 endpoint="https://x.services.ai.azure.com/openai/v1/chat/completions",
                 deployment="grok-4.3", key="foundry-key"):
    monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", endpoint)
    monkeypatch.setenv("AZURE_FOUNDRY_DEPLOYMENT", deployment)
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", key)


def test_resolve_foundry_root_kwargs_exact(monkeypatch):
    _set_foundry(monkeypatch)
    entry = resolve_root_model("azure-foundry")
    assert entry.backend_kwargs == {
        "base_url": "https://x.services.ai.azure.com/openai/v1",
        "model_name": "grok-4.3",
        "api_key": "foundry-key",
    }
    assert entry.sub_backend_kwargs == {
        "base_url": "https://x.services.ai.azure.com/openai/v1",
        "model_name": "grok-4.3",
        "api_key": "foundry-key",
    }


def test_inject_foundry_kwargs_routes_through_canonical_resolver(monkeypatch):
    called = {"n": 0}

    def _fake_resolve():
        called["n"] += 1
        return ("https://canon.services.ai.azure.com/openai/v1", "canon-dep", "canon-key")

    monkeypatch.setattr(
        "backend.agents.runtime.foundry_endpoint.resolve_foundry_credentials",
        _fake_resolve,
    )
    out = models._inject_foundry_kwargs({"api_key": "preinjected"}, model_key="azure-foundry")
    assert called["n"] >= 1
    assert out["base_url"] == "https://canon.services.ai.azure.com/openai/v1"
    assert out["model_name"] == "canon-dep"
    assert out["api_key"] == "preinjected"  # NOT overwritten by resolver's key


def test_inject_foundry_fail_fast_on_missing(monkeypatch):
    monkeypatch.setattr(
        "backend.agents.runtime.foundry_endpoint.resolve_foundry_credentials",
        lambda: ("", "", ""),
    )
    with pytest.raises(ValueError, match="AZURE_FOUNDRY"):
        models._inject_foundry_kwargs({}, model_key="azure-foundry")


def test_local_normalize_helper_removed():
    assert not hasattr(models, "_normalize_foundry_base_url")
