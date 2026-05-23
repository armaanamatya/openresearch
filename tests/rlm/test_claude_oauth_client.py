"""Unit tests for ClaudeOauthClient.

All SDK calls are mocked — no real network or Claude credentials required.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCompletionPassesPromptThroughThreadIsolation:
    """test_completion_passes_prompt_through_thread_isolation"""

    def test_completion_passes_prompt_through_thread_isolation(self, monkeypatch):
        """_run_sdk_in_thread is called with the correct args; the returned text
        is propagated back to the caller unchanged."""
        from backend.agents.rlm.claude_oauth_client import ClaudeOauthClient

        monkeypatch.setattr(
            "backend.agents.rdr.agent._run_sdk_in_thread",
            lambda **kw: "sdk-response",
        )

        client = ClaudeOauthClient(model_name="claude-sonnet-4-6")

        captured: list[dict] = []

        def _fake_run(**kw):
            captured.append(kw)
            return "sdk-response"

        monkeypatch.setattr("backend.agents.rdr.agent._run_sdk_in_thread", _fake_run)

        result = client.completion("hello")

        assert result == "sdk-response"
        assert len(captured) == 1
        call = captured[0]
        assert call["prompt"] == "hello"
        assert call["provider"] == "anthropic"
        assert call["max_turns"] == 1


class TestCompletionRendersListPromptWithSystem:
    """List-of-messages prompt with a system role is rendered correctly."""

    def test_completion_renders_list_prompt_with_system(self, monkeypatch):
        from backend.agents.rlm.claude_oauth_client import ClaudeOauthClient

        captured: list[dict] = []

        def _fake_run(**kw):
            captured.append(kw)
            return "response"

        monkeypatch.setattr("backend.agents.rdr.agent._run_sdk_in_thread", _fake_run)

        client = ClaudeOauthClient()
        prompt = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
        ]
        client.completion(prompt)

        assert len(captured) == 1
        full_prompt = captured[0]["prompt"]
        assert "S" in full_prompt
        assert "U" in full_prompt


class TestAcompletionRoutesThroughToThread:
    """acompletion dispatches through asyncio.to_thread to completion."""

    def test_acompletion_routes_through_to_thread(self, monkeypatch):
        from backend.agents.rlm.claude_oauth_client import ClaudeOauthClient

        monkeypatch.setattr(
            "backend.agents.rdr.agent._run_sdk_in_thread",
            lambda **kw: "async-response",
        )

        client = ClaudeOauthClient()
        result = asyncio.run(client.acompletion("async test"))
        assert result == "async-response"


class TestTimeoutPropagatesAsTimeoutError:
    """A TimeoutError from _run_sdk_in_thread is re-raised by completion."""

    def test_timeout_propagates_as_timeouterror(self, monkeypatch):
        from backend.agents.rlm.claude_oauth_client import ClaudeOauthClient

        def _raise_timeout(**kw):
            raise TimeoutError("sdk timed out")

        monkeypatch.setattr("backend.agents.rdr.agent._run_sdk_in_thread", _raise_timeout)

        client = ClaudeOauthClient()
        with pytest.raises(TimeoutError):
            client.completion("will timeout")


class TestNoApiKeyRequiredAtConstruction:
    """ClaudeOauthClient can be instantiated without any api_key argument."""

    def test_no_api_key_required_at_construction(self):
        from backend.agents.rlm.claude_oauth_client import ClaudeOauthClient

        # Should not raise — no api_key needed.
        client = ClaudeOauthClient(model_name="claude-sonnet-4-6")
        assert client.model_name == "claude-sonnet-4-6"
