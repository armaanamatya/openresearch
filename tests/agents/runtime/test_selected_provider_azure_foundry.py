"""G3: selected_provider() lets generic sub-agents NAME azure / azure-foundry.
Both resolve to the "openai" ProviderName literal (Foundry rides the OpenAI
SDK), matching the validate_provider_credentials precedent. Existing
anthropic/openai resolution is unchanged."""
from __future__ import annotations

import pytest

from backend.agents.runtime.base import ProviderConfigurationError
from backend.agents.runtime.factory import selected_provider


@pytest.mark.parametrize("name", ["azure", "azure-openai", "azure_openai"])
def test_azure_resolves_to_openai(name):
    assert selected_provider(name) == "openai"


@pytest.mark.parametrize("name", ["azure-foundry", "foundry", "grok", "grok-4.3"])
def test_azure_foundry_resolves_to_openai(name):
    assert selected_provider(name) == "openai"


def test_anthropic_and_openai_unchanged():
    assert selected_provider("anthropic") == "anthropic"
    assert selected_provider("claude") == "anthropic"
    assert selected_provider("openai") == "openai"
    assert selected_provider("oai") == "openai"


def test_unknown_provider_still_raises():
    with pytest.raises(ProviderConfigurationError):
        selected_provider("bananas")


def test_unset_defaults_unchanged(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_LLM_PROVIDER", raising=False)
    monkeypatch.setattr(
        "backend.agents.runtime.factory.get_settings",
        lambda *a, **k: type("S", (), {"llm_provider": ""})(),
    )
    assert selected_provider() == "anthropic"
