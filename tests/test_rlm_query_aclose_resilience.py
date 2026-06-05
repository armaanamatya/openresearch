"""Regression tests for the RLM-root SDK aclose-race resilience (2026-05-30).

Root cause of the prj_09047604e591d969 iteration-1 death: the root-model SDK
path (``ClaudeLlmClient.complete`` → ``_async_complete``) SWALLOWED a pre-result
aclose race and returned ``""``; the rlm root loop, parsing no ```repl block,
treated the empty turn as terminal and shipped a partial report. Sub-agent calls
had ``run_isolated``'s retry; the root did not.

Fix (each asserted below):
1. ``_async_complete`` RE-RAISES a pre-result aclose race (no salvageable text)
   so the isolation layer can retry; it still SALVAGES streamed text on a
   post-stream race.
2. ``complete()`` routes the coroutine through ``run_isolated`` → a pre-result
   aclose race retries with a fresh query and recovers; exhaustion returns "".
3. ``ClaudeOauthClient.completion`` converts a residual empty completion into a
   no-op ```repl turn so the root loop survives (opt-out via env).
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

_ACLOSE = "aclose(): asynchronous generator is already running"


@pytest.fixture(autouse=True)
def _force_sdk_transport(monkeypatch):
    """Section 3 (ClaudeOauthClient.completion empty→no-op) targets the
    claude-agent-sdk fallback path; force it so the CLI primary transport does
    not bypass the injected SDK mock. The CLI path's empty→fallback is covered by
    tests/agents/rlm/test_claude_oauth_cli_transport.py. Harmless to the
    ClaudeLlmClient-direct tests, which never read the transport env."""
    monkeypatch.setenv("REPROLAB_RLM_ROOT_TRANSPORT", "sdk")


def _make_client():
    from backend.services.context.workspace.tools.rlm_query import ClaudeLlmClient

    return ClaudeLlmClient()


def _make_query(items):
    """Return a fake ``claude_agent_sdk.query`` yielding ``items``.

    An item that is an ``Exception`` is RAISED at its position in the stream;
    any other item is yielded as an SDK event.
    """

    def fake_query(*, prompt, options):  # noqa: ANN001 - mirror SDK signature
        async def gen():
            for item in items:
                if isinstance(item, BaseException):
                    raise item
                yield item

        return gen()

    return fake_query


# --- 1. _async_complete re-raise / salvage -----------------------------------


def test_async_complete_reraises_pre_result_aclose_race():
    """A race before any text streamed must PROPAGATE (so complete() retries)."""
    client = _make_client()
    fake = _make_query([RuntimeError(_ACLOSE)])

    async def _run():
        with patch("claude_agent_sdk.query", fake):
            await client._async_complete(system="s", user="u")

    with pytest.raises(RuntimeError, match="aclose"):
        asyncio.run(_run())


def test_async_complete_salvages_post_stream_text_on_aclose():
    """A race AFTER assistant text streamed is swallowed; the text is returned."""
    from claude_agent_sdk import AssistantMessage, TextBlock

    client = _make_client()
    msg = AssistantMessage(content=[TextBlock(text="partial answer")], model="m")
    fake = _make_query([msg, RuntimeError(_ACLOSE)])

    async def _run():
        with patch("claude_agent_sdk.query", fake):
            return await client._async_complete(system="s", user="u")

    text, _usage = asyncio.run(_run())
    assert text == "partial answer"


def test_async_complete_non_aclose_error_still_salvages_empty():
    """A NON-aclose stream error with no text is swallowed (unchanged behavior)."""
    client = _make_client()
    fake = _make_query([ValueError("boom")])

    async def _run():
        with patch("claude_agent_sdk.query", fake):
            return await client._async_complete(system="s", user="u")

    text, _usage = asyncio.run(_run())
    assert text == ""  # salvage-over-crash: non-aclose errors are not re-raised


# --- 2. complete() retry via run_isolated ------------------------------------


def test_complete_retries_pre_result_aclose_and_recovers():
    """The exact bug: a pre-result aclose race now RETRIES and recovers text."""
    client = _make_client()
    calls = {"n": 0}

    async def flaky(*, system, user):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError(_ACLOSE)
        return ("recovered", {"input_tokens": 0})

    with patch.object(client, "_async_complete", flaky):
        result = client.complete(system="s", user="u")

    assert result == "recovered"
    assert calls["n"] == 2, "complete() must retry the pre-result aclose race once"


def test_complete_exhausted_aclose_returns_empty(monkeypatch):
    """When retries are exhausted, complete() returns "" (caller decides)."""
    monkeypatch.setenv("REPROLAB_RLM_ROOT_SDK_MAX_RETRIES", "1")
    client = _make_client()
    calls = {"n": 0}

    async def always_aclose(*, system, user):
        calls["n"] += 1
        raise RuntimeError(_ACLOSE)

    with patch.object(client, "_async_complete", always_aclose):
        result = client.complete(system="s", user="u")

    assert result == ""
    assert calls["n"] == 2, "max_retries=1 → initial attempt + exactly one retry"


# --- 3. ClaudeOauthClient empty → no-op turn ---------------------------------


def _oauth_client_with_inner(return_value: str):
    from backend.agents.rlm.claude_oauth_client import ClaudeOauthClient

    client = ClaudeOauthClient(model_name="claude-sonnet-4-6")
    inner = MagicMock()
    inner.complete.return_value = return_value
    client._claude_clients["claude-sonnet-4-6"] = inner
    return client


def test_completion_empty_falls_back_to_noop_repl_turn(monkeypatch):
    monkeypatch.delenv("REPROLAB_RLM_EMPTY_TURN_FALLBACK", raising=False)
    client = _oauth_client_with_inner("")
    out = client.completion("do something")
    assert "```repl" in out and "pass" in out, "empty completion must become a no-op turn"


def test_completion_empty_fallback_optout(monkeypatch):
    monkeypatch.setenv("REPROLAB_RLM_EMPTY_TURN_FALLBACK", "0")
    client = _oauth_client_with_inner("")
    assert client.completion("do something") == ""


def test_completion_passthrough_when_nonempty(monkeypatch):
    monkeypatch.delenv("REPROLAB_RLM_EMPTY_TURN_FALLBACK", raising=False)
    client = _oauth_client_with_inner("```repl\nprint(1)\n```")
    assert client.completion("x") == "```repl\nprint(1)\n```"
