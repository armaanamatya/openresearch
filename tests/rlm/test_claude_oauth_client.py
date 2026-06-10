"""Unit tests for ClaudeOauthClient.

All SDK calls are mocked — no real network or Claude credentials required.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _force_sdk_transport(monkeypatch):
    """These tests target the claude-agent-sdk path (prompt rendering, per-model
    client caching, exception propagation), which is now the *fallback*
    transport. The reliable CLI primary path is covered by
    tests/agents/rlm/test_claude_oauth_cli_transport.py."""
    monkeypatch.setenv("OPENRESEARCH_RLM_ROOT_TRANSPORT", "sdk")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCompletionPassesPromptThroughThreadIsolation:
    """test_completion_passes_prompt_through_thread_isolation"""

    def test_completion_passes_prompt_through_thread_isolation(self, monkeypatch):
        """ClaudeLlmClient.complete is called with system="" and user="hello";
        the returned text is propagated back to the caller unchanged."""
        from backend.agents.rlm.claude_oauth_client import ClaudeOauthClient
        from backend.services.context.workspace.tools.rlm_query import ClaudeLlmClient

        mock_complete = MagicMock(return_value="sdk-response")
        monkeypatch.setattr(ClaudeLlmClient, "complete", mock_complete)

        client = ClaudeOauthClient(model_name="claude-sonnet-4-6")
        result = client.completion("hello")

        assert result == "sdk-response"
        mock_complete.assert_called_once_with(system="", user="hello")


class TestCompletionRendersListPromptWithSystem:
    """List-of-messages prompt with a system role is rendered correctly."""

    def test_completion_renders_list_prompt_with_system(self, monkeypatch):
        from backend.agents.rlm.claude_oauth_client import ClaudeOauthClient
        from backend.services.context.workspace.tools.rlm_query import ClaudeLlmClient

        mock_complete = MagicMock(return_value="response")
        monkeypatch.setattr(ClaudeLlmClient, "complete", mock_complete)

        client = ClaudeOauthClient()
        prompt = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
        ]
        client.completion(prompt)

        mock_complete.assert_called_once()
        call_kwargs = mock_complete.call_args[1]
        assert call_kwargs["system"] == "S"
        assert call_kwargs["user"] == "U"


class TestCompletionCachesClientsPerModel:
    """ClaudeLlmClient instances are cached and reused per model."""

    def test_completion_caches_clients_per_model(self, monkeypatch):
        from backend.agents.rlm.claude_oauth_client import ClaudeOauthClient
        from backend.services.context.workspace.tools import rlm_query

        init_calls: list[dict] = []
        original_init = rlm_query.ClaudeLlmClient.__init__

        def _tracking_init(self, model=None, max_turns=1):
            init_calls.append({"model": model, "max_turns": max_turns})
            original_init(self, model=model, max_turns=max_turns)

        monkeypatch.setattr(rlm_query.ClaudeLlmClient, "__init__", _tracking_init)
        monkeypatch.setattr(rlm_query.ClaudeLlmClient, "complete", lambda self, **kw: "ok")

        client = ClaudeOauthClient(model_name="claude-sonnet-4-6")

        # Two calls with the same model — init should fire only once.
        client.completion("first call")
        client.completion("second call")
        assert len(init_calls) == 1

        # A call with a different model — init should fire a second time.
        client.completion("third call", model="claude-haiku-4-5")
        assert len(init_calls) == 2


class TestCompletionPropagatesExceptions:
    """Exceptions from ClaudeLlmClient.complete propagate out of completion()."""

    def test_completion_propagates_exceptions(self, monkeypatch):
        from backend.agents.rlm.claude_oauth_client import ClaudeOauthClient
        from backend.services.context.workspace.tools.rlm_query import ClaudeLlmClient

        monkeypatch.setattr(
            ClaudeLlmClient,
            "complete",
            MagicMock(side_effect=ValueError("synthetic")),
        )

        client = ClaudeOauthClient()
        with pytest.raises(ValueError, match="synthetic"):
            client.completion("will fail")


class TestAcompletionRoutesThroughToThread:
    """acompletion dispatches through asyncio.to_thread to completion."""

    def test_acompletion_routes_through_to_thread(self, monkeypatch):
        from backend.agents.rlm.claude_oauth_client import ClaudeOauthClient
        from backend.services.context.workspace.tools.rlm_query import ClaudeLlmClient

        monkeypatch.setattr(
            ClaudeLlmClient,
            "complete",
            MagicMock(return_value="async-response"),
        )

        client = ClaudeOauthClient()
        result = asyncio.run(client.acompletion("async test"))
        assert result == "async-response"


class TestNoApiKeyRequiredAtConstruction:
    """ClaudeOauthClient can be instantiated without any api_key argument."""

    def test_no_api_key_required_at_construction(self):
        from backend.agents.rlm.claude_oauth_client import ClaudeOauthClient

        # Should not raise — no api_key needed.
        client = ClaudeOauthClient(model_name="claude-sonnet-4-6")
        assert client.model_name == "claude-sonnet-4-6"
