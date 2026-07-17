# tests/agents/rlm/test_anthropic_foundry_patch.py
from unittest.mock import patch

from backend.agents.rlm import models
from backend.agents.rlm._anthropic_foundry_patch import (
    apply_anthropic_foundry_backend_patch,
    build_anthropic_foundry_client,
)


def test_registry_has_foundry_root_entries():
    assert "opus-foundry" in models.ROOT_MODELS
    assert models.ROOT_MODELS["opus-foundry"].backend_kwargs["model_name"] == "claude-opus-4-8"
    assert models.ROOT_MODELS["opus-foundry"].rlm_backend == "anthropic-foundry"
    assert models.ROOT_MODELS["sonnet-foundry"].backend_kwargs["model_name"] == "claude-sonnet-5"


def test_resolve_opus_foundry_alias(monkeypatch):
    # resolve_root_model fails closed when the backend's key is absent, so the
    # alias assertion needs a credential present. Inject a fake one explicitly
    # (same pattern as test_client_builder_targets_foundry below) rather than
    # relying on the developer's .env — that made this test pass locally and
    # fail on any clean checkout / CI runner.
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "k-fake-for-test")

    entry = models.resolve_root_model("opus-4-8")
    assert entry.key == "opus-foundry"


def test_client_builder_targets_foundry(monkeypatch):
    monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", "https://r.services.ai.azure.com/openai/v1")
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "k-7")
    with patch("anthropic.Anthropic") as mock_anthropic:
        build_anthropic_foundry_client({"model_name": "claude-opus-4-8"})
    _, kwargs = mock_anthropic.call_args
    assert kwargs["base_url"] == "https://r.services.ai.azure.com/anthropic"
    assert kwargs["api_key"] == "k-7"


def test_client_builder_fails_closed_on_missing_creds(monkeypatch):
    import pytest

    import backend.agents.runtime.foundry_anthropic as fa

    for v in ("AZURE_FOUNDRY_ENDPOINT", "AZURE_FOUNDRY_ANTHROPIC_ENDPOINT",
              "AZURE_FOUNDRY_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    # neutralize the .env-backed Settings fallback (this checkout carries live creds)
    monkeypatch.setattr(fa, "_env_or_settings", lambda *a, **k: "")
    with pytest.raises(ValueError, match="AZURE_FOUNDRY"):
        build_anthropic_foundry_client({"model_name": "claude-opus-4-8"})


def test_patch_is_idempotent():
    apply_anthropic_foundry_backend_patch()
    apply_anthropic_foundry_backend_patch()  # second call must not raise
