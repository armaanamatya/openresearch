"""Phase 1 — ClaudeLlmClient read-idle timeout + typed sub-RLM failure + hardened kill.

Covers the FM-001/002/007 fix: a stalled SDK stream is bounded by a per-event
read-idle timeout (default 120s), surfaces a non-empty self-describing sentinel to
the root (not ""), salvages partial text when present, notifies a stall sink, and
kills wedged descendants fail-soft.
"""
from __future__ import annotations

import asyncio
import sys
import types
from typing import Any

import pytest


def _install_fake_sdk(monkeypatch, query_fn) -> None:
    """Monkeypatch sys.modules['claude_agent_sdk'] with stub message types + query()."""
    class TextBlock:
        def __init__(self, text: str):
            self.text = text

    class AssistantMessage:
        def __init__(self, content, usage=None):
            self.content = content
            self.usage = usage

    class ResultMessage:
        def __init__(self, result, usage=None):
            self.result = result
            self.usage = usage

    fake = types.ModuleType("claude_agent_sdk")
    fake.AssistantMessage = AssistantMessage
    fake.ResultMessage = ResultMessage
    fake.ClaudeAgentOptions = type(
        "ClaudeAgentOptions", (), {"__init__": lambda self, **k: None}
    )
    fake.query = query_fn
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake)


def test_read_idle_stall_returns_sentinel_not_empty(monkeypatch):
    from backend.services.context.workspace.tools import rlm_query as rq

    async def hanging_query(prompt: str, options: Any):
        # Never yields an event — a half-open socket / stalled stream.
        await asyncio.sleep(60)
        yield  # pragma: no cover — unreachable

    _install_fake_sdk(monkeypatch, hanging_query)
    monkeypatch.setattr(rq, "_bundled_claude_child_pids", lambda: set())

    client = rq.ClaudeLlmClient(model="claude-test", max_turns=1)
    result = client.complete(system="s", user="u", read_idle_s=0.2)

    assert result != ""  # NOT the old empty-string fallback
    assert rq.SUB_RLM_STALL_SENTINEL in result


def test_read_idle_salvages_partial_text(monkeypatch):
    from backend.services.context.workspace.tools import rlm_query as rq

    async def partial_then_hang(prompt, options):
        import claude_agent_sdk as sdk

        yield sdk.AssistantMessage(content=[type("B", (), {"text": "partial answer"})()])
        await asyncio.sleep(60)  # stall before a ResultMessage arrives

    _install_fake_sdk(monkeypatch, partial_then_hang)
    monkeypatch.setattr(rq, "_bundled_claude_child_pids", lambda: set())

    client = rq.ClaudeLlmClient(model="claude-test", max_turns=1)
    result = client.complete(system="s", user="u", read_idle_s=0.2)

    assert result == "partial answer"
    assert rq.SUB_RLM_STALL_SENTINEL not in result


def test_normal_completion_still_works(monkeypatch):
    """Read-idle path must not regress a normal, fast completion."""
    from backend.services.context.workspace.tools import rlm_query as rq

    async def good_query(prompt, options):
        import claude_agent_sdk as sdk

        yield sdk.AssistantMessage(
            content=[type("B", (), {"text": "hi"})()],
            usage={"input_tokens": 3, "output_tokens": 1},
        )
        yield sdk.ResultMessage(
            result="final answer",
            usage={
                "input_tokens": 3,
                "output_tokens": 1,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "reasoning_tokens": 0,
            },
        )

    _install_fake_sdk(monkeypatch, good_query)
    client = rq.ClaudeLlmClient(model="claude-test", max_turns=1)
    result = client.complete(system="s", user="u", read_idle_s=5.0)
    assert result == "final answer"
    assert client._last_usage["input_tokens"] == 3


def test_stall_event_sink_receives_event(monkeypatch):
    from backend.services.context.workspace.tools import rlm_query as rq

    async def hanging(prompt, options):
        await asyncio.sleep(60)
        yield  # pragma: no cover

    _install_fake_sdk(monkeypatch, hanging)
    monkeypatch.setattr(rq, "_bundled_claude_child_pids", lambda: set())

    sink: list[dict] = []
    client = rq.ClaudeLlmClient(model="m", max_turns=1, stall_event_sink=sink.append)
    client.complete(system="s", user="u", read_idle_s=0.2)

    assert len(sink) == 1
    assert sink[0]["event"] == "sub_rlm_stalled"
    assert sink[0]["idle_seconds"] >= 0.2 - 0.05


def test_kill_helpers_are_fail_soft(monkeypatch):
    from backend.services.context.workspace.tools import rlm_query as rq

    client = rq.ClaudeLlmClient(model="m", max_turns=1)
    # A non-existent pid reported as "new" — the SIGKILL attempt must swallow OSError.
    monkeypatch.setattr(rq, "_bundled_claude_child_pids", lambda: {999999})
    client._kill_wedged_children(set())  # must not raise
    client._notify_stall(1.0)  # sink None — must not raise


def test_kill_never_uses_killpg(monkeypatch):
    """REGRESSION: the bundled child shares OUR process group (SDK uses no setsid),
    so killpg(getpgid(child)) would SIGKILL the backend. We must SIGKILL per-pid only.
    """
    import os
    import signal
    from backend.services.context.workspace.tools import rlm_query as rq

    killed: list[tuple[int, int]] = []
    killpg_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(
        os, "killpg", lambda pgid, sig: killpg_calls.append((pgid, sig))
    )
    # One "new" child appears after the call.
    monkeypatch.setattr(rq, "_bundled_claude_child_pids", lambda: {424242})

    client = rq.ClaudeLlmClient(model="m", max_turns=1)
    client._kill_wedged_children(pre_pids=set())

    assert (424242, signal.SIGKILL) in killed, "the wedged child must be SIGKILL'd individually"
    assert killpg_calls == [], "killpg must NEVER be called (would kill the backend's own group)"
