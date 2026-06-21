"""Tests for validate_root_credentials in pre_flight_validator.

Pinned guarantees:
- Valid credential → (True, descriptive message)
- Definitive 401/403 → (False, actionable message naming the env var)
- Missing credential → (False, actionable message naming the env var)
- Network error / timeout → fail-open (True, message)
- Inconclusive HTTP status (e.g. 500) → fail-open (True, message)
- Unknown provider → fail-open (True, message)
- All tests use mocks — NO real network calls (suite is socket-hermetic).
"""

from __future__ import annotations

import urllib.error
import urllib.request
from unittest.mock import MagicMock, patch

from backend.agents.rlm.pre_flight_validator import validate_root_credentials


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _http_error(code: int, body: bytes = b"Unauthorized") -> urllib.error.HTTPError:
    """Build a minimal HTTPError for mocking."""
    resp = MagicMock()
    resp.read.return_value = body
    return urllib.error.HTTPError(
        url="https://example.com",
        code=code,
        msg=str(code),
        hdrs={},  # type: ignore[arg-type]
        fp=resp,
    )


def _url_open_ok(timeout=None):
    """Context manager mock that simulates a successful HTTP response."""
    ctx = MagicMock()
    ctx.__enter__ = lambda s: s
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------


class TestOpenAI:
    def test_valid_key_returns_true(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-valid-key")
        with patch("urllib.request.urlopen", return_value=_url_open_ok()):
            ok, msg = validate_root_credentials("openai", model="gpt-5")
        assert ok is True
        assert "accepted" in msg.lower() or "openai" in msg.lower()

    def test_401_returns_false_actionable(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-bad-key")
        with patch("urllib.request.urlopen", side_effect=_http_error(401, b'{"error": "invalid key"}')):
            ok, msg = validate_root_credentials("openai", model="gpt-5")
        assert ok is False
        assert "OPENAI_API_KEY" in msg
        assert "401" in msg
        assert "OPENRESEARCH_SKIP_CRED_PREFLIGHT" in msg

    def test_403_returns_false_actionable(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-forbidden")
        with patch("urllib.request.urlopen", side_effect=_http_error(403, b"Forbidden")):
            ok, msg = validate_root_credentials("openai")
        assert ok is False
        assert "OPENAI_API_KEY" in msg

    def test_missing_key_returns_false(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        ok, msg = validate_root_credentials("openai", model="gpt-5")
        assert ok is False
        assert "OPENAI_API_KEY" in msg

    def test_network_error_fail_open(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-network-blip")
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            ok, msg = validate_root_credentials("openai")
        assert ok is True
        assert "proceeding" in msg.lower() or "inconclusive" in msg.lower()

    def test_http_500_fail_open(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-server-error")
        with patch("urllib.request.urlopen", side_effect=_http_error(500, b"Internal Server Error")):
            ok, msg = validate_root_credentials("openai")
        assert ok is True
        assert "500" in msg or "proceeding" in msg.lower()


# ---------------------------------------------------------------------------
# Anthropic API key provider
# ---------------------------------------------------------------------------


class TestAnthropic:
    def test_valid_key_returns_true(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-valid-key")
        with patch("urllib.request.urlopen", return_value=_url_open_ok()):
            ok, msg = validate_root_credentials("anthropic", model="claude")
        assert ok is True
        assert "accepted" in msg.lower() or "anthropic" in msg.lower()

    def test_401_returns_false_actionable(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-bad")
        with patch("urllib.request.urlopen", side_effect=_http_error(401, b'{"error": "invalid_api_key"}')):
            ok, msg = validate_root_credentials("anthropic")
        assert ok is False
        assert "ANTHROPIC_API_KEY" in msg
        assert "401" in msg

    def test_missing_key_returns_false(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        ok, msg = validate_root_credentials("anthropic")
        assert ok is False
        assert "ANTHROPIC_API_KEY" in msg
        assert "claude-oauth" in msg.lower() or "oauth" in msg.lower()

    def test_network_error_fail_open(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-blip")
        with patch("urllib.request.urlopen", side_effect=OSError("timeout")):
            ok, msg = validate_root_credentials("anthropic")
        assert ok is True
        assert "proceeding" in msg.lower() or "inconclusive" in msg.lower()


# ---------------------------------------------------------------------------
# Anthropic OAuth provider
# ---------------------------------------------------------------------------


class TestAnthropicOAuth:
    def test_oauth_present_returns_true(self):
        with patch(
            "backend.agents.runtime.factory._has_claude_subscription_oauth",
            return_value=True,
        ):
            ok, msg = validate_root_credentials("anthropic-oauth", model="claude-oauth")
        assert ok is True
        assert "oauth" in msg.lower() or "subscription" in msg.lower()

    def test_oauth_absent_returns_false(self):
        with patch(
            "backend.agents.runtime.factory._has_claude_subscription_oauth",
            return_value=False,
        ):
            ok, msg = validate_root_credentials("anthropic-oauth")
        assert ok is False
        assert "claude login" in msg.lower() or "oauth" in msg.lower()
        assert "ANTHROPIC_API_KEY" in msg

    def test_claude_oauth_alias_works(self):
        """'claude-oauth' provider string is also recognised."""
        with patch(
            "backend.agents.runtime.factory._has_claude_subscription_oauth",
            return_value=True,
        ):
            ok, msg = validate_root_credentials("claude-oauth")
        assert ok is True

    def test_import_error_fail_open(self):
        with patch(
            "backend.agents.runtime.factory._has_claude_subscription_oauth",
            side_effect=ImportError("no module"),
        ):
            ok, msg = validate_root_credentials("anthropic-oauth")
        assert ok is True
        assert "proceeding" in msg.lower() or "inconclusive" in msg.lower()


# ---------------------------------------------------------------------------
# OpenRouter provider
# ---------------------------------------------------------------------------


class TestOpenRouter:
    def test_valid_key_returns_true(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-valid")
        with patch("urllib.request.urlopen", return_value=_url_open_ok()):
            ok, msg = validate_root_credentials("openrouter", model="qwen3-coder")
        assert ok is True

    def test_missing_key_returns_false(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        ok, msg = validate_root_credentials("openrouter")
        assert ok is False
        assert "OPENROUTER_API_KEY" in msg

    def test_401_returns_false(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-bad")
        with patch("urllib.request.urlopen", side_effect=_http_error(401)):
            ok, msg = validate_root_credentials("openrouter")
        assert ok is False
        assert "OPENROUTER_API_KEY" in msg


# ---------------------------------------------------------------------------
# Azure OpenAI provider
# ---------------------------------------------------------------------------


class TestAzureOpenAI:
    def test_missing_key_returns_false(self, monkeypatch):
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        ok, msg = validate_root_credentials("azure_openai")
        assert ok is False
        assert "AZURE_OPENAI_API_KEY" in msg

    def test_key_present_no_endpoint_fail_open(self, monkeypatch):
        """Key present but no endpoint — we can't probe, so fail-open."""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "some-key")
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        ok, msg = validate_root_credentials("azure_openai")
        assert ok is True
        assert "proceeding" in msg.lower() or "skipping" in msg.lower()

    def test_valid_endpoint_key_returns_true(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "valid-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://my-resource.openai.azure.com")
        with patch("urllib.request.urlopen", return_value=_url_open_ok()):
            ok, msg = validate_root_credentials("azure_openai")
        assert ok is True

    def test_401_returns_false(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "bad-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://my-resource.openai.azure.com")
        with patch("urllib.request.urlopen", side_effect=_http_error(401)):
            ok, msg = validate_root_credentials("azure_openai")
        assert ok is False
        assert "AZURE_OPENAI_API_KEY" in msg


# ---------------------------------------------------------------------------
# Azure AI Foundry provider
# ---------------------------------------------------------------------------


class TestAzureFoundry:
    def test_missing_key_returns_false(self, monkeypatch):
        monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
        ok, msg = validate_root_credentials("azure-foundry")
        assert ok is False
        assert "AZURE_FOUNDRY_API_KEY" in msg

    def test_key_present_no_endpoint_fail_open(self, monkeypatch):
        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "some-key")
        monkeypatch.delenv("AZURE_FOUNDRY_ENDPOINT", raising=False)
        ok, msg = validate_root_credentials("azure-foundry")
        assert ok is True

    def test_valid_credentials_returns_true(self, monkeypatch):
        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "valid-key")
        monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", "https://my-resource.services.ai.azure.com/openai/v1")
        with patch("urllib.request.urlopen", return_value=_url_open_ok()):
            ok, msg = validate_root_credentials("azure-foundry")
        assert ok is True

    def test_401_returns_false(self, monkeypatch):
        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "bad-key")
        monkeypatch.setenv("AZURE_FOUNDRY_ENDPOINT", "https://my-resource.services.ai.azure.com/openai/v1")
        with patch("urllib.request.urlopen", side_effect=_http_error(401)):
            ok, msg = validate_root_credentials("azure-foundry")
        assert ok is False
        assert "AZURE_FOUNDRY_API_KEY" in msg


# ---------------------------------------------------------------------------
# Featherless provider
# ---------------------------------------------------------------------------


class TestFeatherless:
    def test_missing_key_returns_false(self, monkeypatch):
        monkeypatch.delenv("FEATHERLESS_API_KEY", raising=False)
        ok, msg = validate_root_credentials("featherless")
        assert ok is False
        assert "FEATHERLESS_API_KEY" in msg

    def test_valid_key_returns_true(self, monkeypatch):
        monkeypatch.setenv("FEATHERLESS_API_KEY", "sk-fl-valid")
        with patch("urllib.request.urlopen", return_value=_url_open_ok()):
            ok, msg = validate_root_credentials("featherless")
        assert ok is True

    def test_401_returns_false(self, monkeypatch):
        monkeypatch.setenv("FEATHERLESS_API_KEY", "sk-fl-bad")
        with patch("urllib.request.urlopen", side_effect=_http_error(401)):
            ok, msg = validate_root_credentials("featherless")
        assert ok is False
        assert "FEATHERLESS_API_KEY" in msg


# ---------------------------------------------------------------------------
# Unknown / unrecognised providers — always fail-open
# ---------------------------------------------------------------------------


class TestUnknownProvider:
    def test_unknown_provider_fail_open(self):
        ok, msg = validate_root_credentials("some-future-provider")
        assert ok is True

    def test_empty_provider_fail_open(self):
        ok, msg = validate_root_credentials("")
        assert ok is True

    def test_none_like_provider_fail_open(self):
        ok, msg = validate_root_credentials("none")
        assert ok is True
