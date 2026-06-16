"""Tests for the OPENRESEARCH_LLM_AUTH_STRATEGY production gate.

The three modes:
  auto       — accept any working credential (default)
  api_only   — fail fast unless a paid API key is present
  oauth_only — fail fast unless the Claude Code CLI is logged in
"""

from __future__ import annotations

import pytest

import backend.config as _config
from backend.agents.runtime import factory as _factory


@pytest.fixture(autouse=True)
def _isolate_settings_cache():
    _config._settings_cache = None
    try:
        yield
    finally:
        _config._settings_cache = None


def test_default_is_auto() -> None:
    s = _config.Settings()
    assert s.llm_auth_strategy == "auto"


def test_env_var_overrides(monkeypatch) -> None:
    monkeypatch.setenv("OPENRESEARCH_LLM_AUTH_STRATEGY", "api_only")
    s = _config.Settings()
    assert s.llm_auth_strategy == "api_only"


def test_invalid_strategy_rejected(monkeypatch) -> None:
    monkeypatch.setenv("OPENRESEARCH_LLM_AUTH_STRATEGY", "nonsense")
    with pytest.raises(Exception):
        _config.Settings()


def test_api_only_without_key_fails_fast(monkeypatch) -> None:
    monkeypatch.setenv("OPENRESEARCH_LLM_AUTH_STRATEGY", "api_only")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENRESEARCH_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(_factory, "_has_claude_subscription_oauth", lambda: True)
    from backend.agents.runtime.factory import (
        ProviderConfigurationError, validate_provider_credentials,
    )
    with pytest.raises(ProviderConfigurationError) as exc:
        validate_provider_credentials("anthropic")
    assert "api_only" in str(exc.value).lower()


def test_oauth_only_without_cli_fails_fast(monkeypatch) -> None:
    monkeypatch.setenv("OPENRESEARCH_LLM_AUTH_STRATEGY", "oauth_only")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-something")
    monkeypatch.setattr(_factory, "_has_claude_subscription_oauth", lambda: False)
    from backend.agents.runtime.factory import (
        ProviderConfigurationError, validate_provider_credentials,
    )
    with pytest.raises(ProviderConfigurationError) as exc:
        validate_provider_credentials("anthropic")
    assert "oauth_only" in str(exc.value).lower()


def test_auto_accepts_oauth_only(monkeypatch) -> None:
    monkeypatch.setenv("OPENRESEARCH_LLM_AUTH_STRATEGY", "auto")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENRESEARCH_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(_factory, "_has_claude_subscription_oauth", lambda: True)
    from backend.agents.runtime.factory import validate_provider_credentials
    assert validate_provider_credentials("anthropic") == "anthropic"


def test_auto_accepts_api_key_only(monkeypatch) -> None:
    monkeypatch.setenv("OPENRESEARCH_LLM_AUTH_STRATEGY", "auto")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-something")
    monkeypatch.setattr(_factory, "_has_claude_subscription_oauth", lambda: False)
    from backend.agents.runtime.factory import validate_provider_credentials
    assert validate_provider_credentials("anthropic") == "anthropic"


def test_api_only_passes_with_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENRESEARCH_LLM_AUTH_STRATEGY", "api_only")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-something")
    monkeypatch.setattr(_factory, "_has_claude_subscription_oauth", lambda: False)
    from backend.agents.runtime.factory import validate_provider_credentials
    assert validate_provider_credentials("anthropic") == "anthropic"
