"""Tests for _MODEL_ALIASES and the alias-resolution path in resolve_root_model.

Guards that lab UI dropdown values, Anthropic SDK model names, and common
nicknames all resolve to valid registry keys without changing behaviour for
inputs that already match a registry key directly.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_has_credentials(monkeypatch, value: bool = True) -> None:
    """Monkeypatch has_provider_credentials so credential checks don't
    depend on the real environment."""
    monkeypatch.setattr(
        "backend.agents.runtime.factory.has_provider_credentials",
        lambda provider=None: value,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLabUiDropdownAliases:
    """Lab UI sends 'sonnet' / 'opus' — both must resolve to claude-oauth."""

    def test_sonnet_alias_to_claude_oauth(self, monkeypatch):
        _patch_has_credentials(monkeypatch)
        from backend.agents.rlm.models import resolve_root_model

        entry = resolve_root_model("sonnet")
        assert entry.key == "claude-oauth"

    def test_opus_alias(self, monkeypatch):
        _patch_has_credentials(monkeypatch)
        from backend.agents.rlm.models import resolve_root_model

        entry = resolve_root_model("opus")
        assert entry.key == "claude-oauth"


class TestAnthropicSdkNameAliases:
    """Anthropic SDK model-string names must resolve via the alias map."""

    def test_claude_sonnet_4_6_alias(self, monkeypatch):
        """The live-failure case: 'claude-sonnet-4-6' must not raise."""
        _patch_has_credentials(monkeypatch)
        from backend.agents.rlm.models import resolve_root_model

        entry = resolve_root_model("claude-sonnet-4-6")
        assert entry.key == "claude-oauth"

    def test_claude_opus_4_7_alias(self, monkeypatch):
        _patch_has_credentials(monkeypatch)
        from backend.agents.rlm.models import resolve_root_model

        entry = resolve_root_model("claude-opus-4-7")
        assert entry.key == "claude-oauth"

    def test_haiku_alias(self, monkeypatch):
        _patch_has_credentials(monkeypatch)
        from backend.agents.rlm.models import resolve_root_model

        entry = resolve_root_model("claude-haiku-4-5-20251001")
        assert entry.key == "claude-oauth"


class TestRegistryKeyPassThrough:
    """Inputs that are already valid registry keys must still work unchanged."""

    def test_registry_key_pass_through(self, monkeypatch):
        _patch_has_credentials(monkeypatch)
        from backend.agents.rlm.models import resolve_root_model

        entry = resolve_root_model("claude-oauth")
        assert entry.key == "claude-oauth"

    def test_explicit_anthropic_key_still_resolves(self, monkeypatch):
        """When ANTHROPIC_API_KEY is set, resolve_root_model('claude') still works."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        # The 'claude' registry entry uses the anthropic (non-oauth) backend.
        # It should resolve without needing has_provider_credentials to return True.
        from backend.agents.rlm.models import ROOT_MODELS, resolve_root_model

        # Only run this test if 'claude' is in the registry.
        if "claude" not in ROOT_MODELS:
            pytest.skip("'claude' registry entry not present in this build")

        entry = resolve_root_model("claude")
        assert entry.key == "claude"


class TestUnknownNameError:
    """Unknown names must raise ValueError with both registry and alias hints."""

    def test_unknown_name_raises_with_helpful_hint(self, monkeypatch):
        _patch_has_credentials(monkeypatch)
        from backend.agents.rlm.models import resolve_root_model

        with pytest.raises(ValueError) as exc_info:
            resolve_root_model("totally-bogus-name")

        msg = str(exc_info.value)
        assert "Valid registry keys" in msg, f"Missing 'Valid registry keys' in: {msg}"
        assert "Aliases also accepted" in msg, f"Missing 'Aliases also accepted' in: {msg}"


class TestQwenAlias:
    """Qwen nickname aliases must resolve to registry keys."""

    def test_qwen_alias(self, monkeypatch):
        """'qwen3' alias resolves to qwen3-coder registry key."""
        from backend.agents.rlm.models import ROOT_MODELS, _MODEL_ALIASES

        # Verify the alias map entry is correct without triggering the credential
        # check (qwen3-coder requires OPENROUTER_API_KEY in the full resolve path).
        assert "qwen3" in _MODEL_ALIASES
        assert _MODEL_ALIASES["qwen3"] == "qwen3-coder"
        assert "qwen3-coder" in ROOT_MODELS
