"""Tests for the claude-oauth RootModel entry and resolve_root_model OAuth logic."""

from __future__ import annotations

import sys

import pytest


def _reload_models():
    """Re-import models.py so _build_registry() re-reads env vars."""
    for mod in list(sys.modules):
        if "backend.agents.rlm.models" in mod:
            del sys.modules[mod]
    import backend.agents.rlm.models as m
    return m


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestClaudeOauthInRegistry:
    """claude-oauth exists in ROOT_MODELS with the correct fields."""

    def test_claude_oauth_in_registry(self):
        from backend.agents.rlm.models import ROOT_MODELS

        assert "claude-oauth" in ROOT_MODELS
        entry = ROOT_MODELS["claude-oauth"]
        assert entry.rlm_backend == "anthropic-oauth"
        assert entry.api_key_env is None


class TestResolveClaudeOauthWithCredentials:
    """resolve_root_model('claude-oauth') succeeds when has_provider_credentials is True."""

    def test_resolve_claude_oauth_with_credentials(self, monkeypatch):
        monkeypatch.setattr(
            "backend.agents.runtime.factory.has_provider_credentials",
            lambda provider=None: True,
        )
        from backend.agents.rlm.models import resolve_root_model

        entry = resolve_root_model("claude-oauth")
        assert entry.key == "claude-oauth"
        assert entry.rlm_backend == "anthropic-oauth"


class TestResolveClaudeOauthWithoutCredentialsFailsLoud:
    """resolve_root_model('claude-oauth') raises ValueError when no creds are found."""

    def test_resolve_claude_oauth_without_credentials_fails_loud(self, monkeypatch):
        monkeypatch.setattr(
            "backend.agents.runtime.factory.has_provider_credentials",
            lambda provider=None: False,
        )
        from backend.agents.rlm.models import resolve_root_model

        with pytest.raises(ValueError) as exc_info:
            resolve_root_model("claude-oauth")

        msg = str(exc_info.value)
        assert "Claude credentials" in msg
        assert "claude login" in msg


class TestDefaultFallsBackToClaudeOauth:
    """When no API keys are set but OAuth creds are present, default picks claude-oauth."""

    def test_default_falls_back_to_claude_oauth_when_no_keys_but_oauth_present(
        self, monkeypatch
    ):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("FEATHERLESS_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OPENRESEARCH_RLM_ROOT_MODEL", raising=False)

        monkeypatch.setattr(
            "backend.agents.runtime.factory.has_provider_credentials",
            lambda provider=None: True,
        )

        mod = _reload_models()
        result = mod.resolve_root_model(None)
        assert result.key == "claude-oauth"


class TestDefaultFallsThroughToQwen3CoderWhenNoCredentials:
    """When all keys are unset and OAuth is False, default picks qwen3-coder which then fails."""

    def test_default_falls_through_to_qwen3coder_when_no_credentials_anywhere(
        self, monkeypatch
    ):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("FEATHERLESS_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OPENRESEARCH_RLM_ROOT_MODEL", raising=False)

        monkeypatch.setattr(
            "backend.agents.runtime.factory.has_provider_credentials",
            lambda provider=None: False,
        )

        mod = _reload_models()
        # qwen3-coder requires OPENROUTER_API_KEY which is unset — expect ValueError.
        with pytest.raises(ValueError):
            mod.resolve_root_model(None)
