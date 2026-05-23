"""Tests for backend.agents.runtime.factory credential helpers.

Covers _has_azure_openai_credentials, has_provider_credentials (azure branch),
and validate_provider_credentials (azure branch).
"""

from __future__ import annotations

import pytest


class TestHasAzureOpenAICredentials:
    """_has_azure_openai_credentials requires BOTH key AND endpoint."""

    def test_both_set_returns_true(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
        from backend.agents.runtime.factory import _has_azure_openai_credentials
        assert _has_azure_openai_credentials() is True

    def test_missing_key_returns_false(self, monkeypatch):
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
        from backend.agents.runtime.factory import _has_azure_openai_credentials
        assert _has_azure_openai_credentials() is False

    def test_missing_endpoint_returns_false(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-key")
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        from backend.agents.runtime.factory import _has_azure_openai_credentials
        assert _has_azure_openai_credentials() is False

    def test_both_missing_returns_false(self, monkeypatch):
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        from backend.agents.runtime.factory import _has_azure_openai_credentials
        assert _has_azure_openai_credentials() is False


class TestHasProviderCredentialsAzure:
    """has_provider_credentials routes 'azure-openai' through _has_azure_openai_credentials."""

    def test_azure_openai_str_both_set(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com")
        from backend.agents.runtime.factory import has_provider_credentials
        assert has_provider_credentials("azure-openai") is True

    def test_azure_str_both_set(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com")
        from backend.agents.runtime.factory import has_provider_credentials
        assert has_provider_credentials("azure") is True

    def test_azure_underscore_str_both_set(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com")
        from backend.agents.runtime.factory import has_provider_credentials
        assert has_provider_credentials("azure_openai") is True

    def test_azure_openai_missing_endpoint_returns_false(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        from backend.agents.runtime.factory import has_provider_credentials
        assert has_provider_credentials("azure-openai") is False

    def test_azure_openai_missing_key_returns_false(self, monkeypatch):
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com")
        from backend.agents.runtime.factory import has_provider_credentials
        assert has_provider_credentials("azure-openai") is False


class TestValidateProviderCredentialsAzure:
    """validate_provider_credentials raises ProviderConfigurationError when Azure creds missing."""

    def test_both_set_does_not_raise(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com")
        from backend.agents.runtime.factory import validate_provider_credentials
        # Should not raise; returns 'openai' as the closest ProviderName.
        result = validate_provider_credentials("azure-openai")
        assert result == "openai"

    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com")
        from backend.agents.runtime.factory import validate_provider_credentials
        from backend.agents.runtime.base import ProviderConfigurationError
        with pytest.raises(ProviderConfigurationError):
            validate_provider_credentials("azure-openai")

    def test_missing_endpoint_raises(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        from backend.agents.runtime.factory import validate_provider_credentials
        from backend.agents.runtime.base import ProviderConfigurationError
        with pytest.raises(ProviderConfigurationError):
            validate_provider_credentials("azure-openai")
