"""Tests for _build_llm_client dispatch logic in backend.agents.rlm.run.

Covers all 6 dispatch branches and the two ValueError paths, plus the critical
regression test that a stale OPENAI_API_KEY in env does not override claude-oauth.
"""

from __future__ import annotations

import os

import pytest

from backend.agents.rlm.models import RootModel
from backend.agents.rlm.run import _build_llm_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_root_model(
    rlm_backend: str,
    backend_kwargs: dict | None = None,
    sub_backend_kwargs: dict | None = None,
    key: str = "test-model",
) -> RootModel:
    """Construct a RootModel directly (bypassing resolve_root_model's env checks)."""
    return RootModel(
        key=key,
        rlm_backend=rlm_backend,
        backend_kwargs=backend_kwargs or {},
        sub_backend_kwargs=sub_backend_kwargs or {},
    )


# ---------------------------------------------------------------------------
# Branch 1: anthropic-oauth → ClaudeLlmClient, label "claude-oauth"
# ---------------------------------------------------------------------------

class TestClaudeOAuth:

    def test_claude_oauth_routes_to_claude_client(self):
        from backend.services.context.workspace.tools.rlm_query import ClaudeLlmClient

        root = _make_root_model("anthropic-oauth", key="claude-oauth")
        client, label = _build_llm_client(None, root)

        assert isinstance(client, ClaudeLlmClient)
        assert label == "claude-oauth"

    # --- Critical regression test -------------------------------------------

    def test_stale_openai_key_in_env_does_not_override_claude_oauth(self, monkeypatch):
        """A stale OPENAI_API_KEY must never route claude-oauth primitives to OpenAI.

        This is the exact bug we fixed: the old heuristic "OPENAI_API_KEY in env
        → use OpenAI" misrouted every primitive when the key was present but stale,
        causing 401s and fail-soft empty results throughout the run.
        """
        from backend.services.context.workspace.tools.rlm_query import ClaudeLlmClient
        from backend.services.context.workspace.tools.openai_client import OpenAILlmClient

        monkeypatch.setenv("OPENAI_API_KEY", "sk-stale-bad-key")
        root = _make_root_model("anthropic-oauth", key="claude-oauth")
        client, label = _build_llm_client(None, root)

        assert isinstance(client, ClaudeLlmClient), (
            "claude-oauth must route to ClaudeLlmClient regardless of OPENAI_API_KEY in env"
        )
        assert not isinstance(client, OpenAILlmClient)
        assert label == "claude-oauth"


# ---------------------------------------------------------------------------
# Branch 2: openai + base_url → OpenAILlmClient mirroring custom endpoint
# ---------------------------------------------------------------------------

class TestFeatherless:

    def test_featherless_routes_to_openai_with_base_url(self):
        from backend.services.context.workspace.tools.openai_client import OpenAILlmClient

        root = _make_root_model(
            "openai",
            backend_kwargs={
                "base_url": "https://x.example/v1",
                "api_key": "sk-test",
                "model_name": "Qwen/Qwen3",
            },
            sub_backend_kwargs={"model_name": "Qwen/Qwen3"},
            key="qwen3-coder-featherless",
        )
        client, label = _build_llm_client(None, root)

        assert isinstance(client, OpenAILlmClient)
        # sub_backend_kwargs model_name takes precedence
        assert label == "Qwen/Qwen3"
        assert client._client.base_url is not None
        assert "x.example" in str(client._client.base_url)
        assert client._model == "Qwen/Qwen3"

    def test_featherless_missing_api_key_raises(self):
        root = _make_root_model(
            "openai",
            backend_kwargs={
                "base_url": "https://x/",
                "model_name": "X",
                # deliberately NO api_key
            },
            key="qwen3-coder-featherless",
        )
        with pytest.raises(ValueError, match="api_key was not resolved"):
            _build_llm_client(None, root)


# ---------------------------------------------------------------------------
# Branch 3: openrouter → OpenAILlmClient with openrouter base_url
# ---------------------------------------------------------------------------

class TestOpenRouter:

    def test_openrouter_routes_to_openai_with_openrouter_base_url(self):
        from backend.services.context.workspace.tools.openai_client import OpenAILlmClient

        root = _make_root_model(
            "openrouter",
            backend_kwargs={"api_key": "or-test", "model_name": "qwen/x"},
            sub_backend_kwargs={"model_name": "qwen/x"},
            key="qwen3-coder",
        )
        client, label = _build_llm_client(None, root)

        assert isinstance(client, OpenAILlmClient)
        assert "openrouter.ai" in str(client._client.base_url)
        assert label == "qwen/x"

    def test_openrouter_missing_api_key_raises(self):
        root = _make_root_model(
            "openrouter",
            backend_kwargs={"model_name": "qwen/x"},  # NO api_key
            key="qwen3-coder",
        )
        with pytest.raises(ValueError, match="api_key was not resolved"):
            _build_llm_client(None, root)


# ---------------------------------------------------------------------------
# Branch 4: anthropic (raw paid API) → ClaudeLlmClient
# ---------------------------------------------------------------------------

class TestAnthropicRaw:

    def test_anthropic_raw_routes_to_claude_client(self):
        from backend.services.context.workspace.tools.rlm_query import ClaudeLlmClient

        root = _make_root_model(
            "anthropic",
            backend_kwargs={"model_name": "claude-opus-4-7"},
            key="claude",
        )
        client, label = _build_llm_client(None, root)

        assert isinstance(client, ClaudeLlmClient)
        assert label == "claude"


# ---------------------------------------------------------------------------
# Branch 4.5: azure_openai → AzureOpenAILlmClient
# ---------------------------------------------------------------------------

class TestAzureOpenAI:

    def test_azure_routes_to_azure_client(self):
        from unittest.mock import patch, MagicMock
        from backend.services.context.workspace.tools.azure_openai_client import AzureOpenAILlmClient

        root = _make_root_model(
            "azure_openai",
            backend_kwargs={
                "model_name": "gpt-4o",
                "azure_deployment": "gpt-4o",
                "azure_endpoint": "https://myresource.openai.azure.com",
                "api_key": "fake-azure-key",
            },
            key="azure-gpt-4o",
        )
        with patch("openai.AzureOpenAI") as mock_azure:
            mock_azure.return_value = MagicMock()
            client, label = _build_llm_client(None, root)

        assert isinstance(client, AzureOpenAILlmClient)
        assert label == "gpt-4o"

    def test_azure_label_matches_model_name(self):
        from unittest.mock import patch, MagicMock

        root = _make_root_model(
            "azure_openai",
            backend_kwargs={
                "model_name": "gpt-4o",
                "azure_endpoint": "https://example.openai.azure.com",
                "api_key": "k",
            },
            key="azure-gpt-4o",
        )
        with patch("openai.AzureOpenAI") as mock_azure:
            mock_azure.return_value = MagicMock()
            _, label = _build_llm_client(None, root)

        assert label == "gpt-4o"

    def test_azure_missing_endpoint_raises(self):
        """_build_llm_client raises ValueError when azure_endpoint is not resolved."""
        root = _make_root_model(
            "azure_openai",
            backend_kwargs={
                "model_name": "gpt-4o",
                # deliberately NO azure_endpoint
                "api_key": "k",
            },
            key="azure-gpt-4o",
        )
        with pytest.raises(ValueError, match="azure_endpoint was not resolved"):
            _build_llm_client(None, root)

    def test_resolve_root_model_azure_alias(self, monkeypatch):
        """resolve_root_model('azure') returns the azure-gpt-4o registry entry."""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-azure-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://myresource.openai.azure.com")
        from backend.agents.rlm.models import resolve_root_model

        entry = resolve_root_model("azure")
        assert entry.key == "azure-gpt-4o"
        assert entry.rlm_backend == "azure_openai"
        assert "azure_endpoint" in entry.backend_kwargs


# ---------------------------------------------------------------------------
# Branch 6: plain openai (no base_url) → OpenAILlmClient
# ---------------------------------------------------------------------------

class TestPlainOpenAI:

    def test_openai_plain_routes_to_openai_client(self):
        from backend.services.context.workspace.tools.openai_client import OpenAILlmClient

        root = _make_root_model(
            "openai",
            backend_kwargs={"model_name": "gpt-5"},
            key="gpt-5",
        )
        client, label = _build_llm_client(None, root)

        assert isinstance(client, OpenAILlmClient)
        assert label == "gpt-4o-mini"
