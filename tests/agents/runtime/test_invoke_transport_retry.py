"""FM-001 (2026-05-30): bounded retry-with-backoff on transient bundled-CLI
transport failures in collect_agent_text.

The bundled-CLI OAuth transport degrades under rapid call volume and surfaces
a zero-text "success" / ConnectionRefused ProcessError. collect_agent_text must
retry that signature (with backoff) so a single transient wedge does not kill
implement_baseline (-> no commands.json -> no experiment -> verdict=failed),
while NON-transient errors must propagate immediately.

R1 – transient twice then success: run_agent called 3x, returns text, no raise.
R2 – transient on every attempt: raises after `attempts`, failed report written.
R3 – non-transient error: raises immediately, no retry.
R4 – retries disabled (RETRIES=0): one attempt only.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator

import pytest

from backend.agents.runtime.base import AgentRuntimeSpec, ProviderName, StreamEvent, StreamText, StreamUsage

_TRANSIENT = "Claude Code returned an error result: success"
_CONNREFUSED = "API Error: Unable to connect to API (ConnectionRefused)"
_NON_TRANSIENT = "Claude Code returned an error result: error_max_turns"


class FlakyRuntime:
    """Raises `exc` on the first `fail_times` run_agent calls, then yields `tail`."""

    def __init__(self, fail_times: int, exc: Exception, tail: list[StreamEvent],
                 provider: ProviderName = "anthropic") -> None:
        self.fail_times = fail_times
        self.exc = exc
        self.tail = tail
        self.calls = 0
        self._provider = provider

    @property
    def provider_name(self) -> ProviderName:
        return self._provider

    async def run_agent(self, *, agent: AgentRuntimeSpec, user_input: str) -> AsyncIterator[StreamEvent]:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc
            yield  # type: ignore[misc]  # unreachable — makes this an async generator
        for ev in self.tail:
            yield ev


def _fake_registry(agent_id: str, tmp_path: Path) -> dict:
    spec = AgentRuntimeSpec(
        name=agent_id, instructions="system", model="claude-sonnet-4-5",
        tools=(), working_directory=tmp_path,
    )

    class FakeEntry:
        def to_runtime_spec(self, provider, *, model_override=None, working_directory=None, max_turns=None, **_kwargs):
            # **_kwargs: the trunk's collect_agent_text passes newer kwargs
            # (e.g. blocked_terms from the RuntimeGuard line) this fake ignores.
            return spec

    return {agent_id: FakeEntry()}


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    # Zero base backoff so retries are instant in tests.
    monkeypatch.setenv("OPENRESEARCH_SUBAGENT_TRANSPORT_BACKOFF_S", "0")


def _run(monkeypatch, tmp_path, runtime, agent_id="baseline-implementation"):
    monkeypatch.setattr("backend.agents.runtime.invoke.AGENT_REGISTRY",
                        _fake_registry(agent_id, tmp_path))
    from backend.agents.runtime.invoke import collect_agent_text
    return asyncio.run(collect_agent_text(agent_id, "write code", project_dir=tmp_path, runtime=runtime))


def test_transient_then_success_retries(monkeypatch, tmp_path: Path) -> None:
    runtime = FlakyRuntime(fail_times=2, exc=Exception(_TRANSIENT),
                           tail=[StreamText("done"), StreamUsage(input_tokens=3, output_tokens=1)])
    text = _run(monkeypatch, tmp_path, runtime)
    assert runtime.calls == 3  # 2 transient failures + 1 success
    assert text == "done"


def test_connrefused_is_transient(monkeypatch, tmp_path: Path) -> None:
    runtime = FlakyRuntime(fail_times=1, exc=Exception(_CONNREFUSED), tail=[StreamText("ok")])
    text = _run(monkeypatch, tmp_path, runtime)
    assert runtime.calls == 2
    assert text == "ok"


def test_transient_exhausts_then_raises(monkeypatch, tmp_path: Path) -> None:
    # 3 attempts (default 2 retries + 1); all fail -> raises, failed report written.
    runtime = FlakyRuntime(fail_times=99, exc=Exception(_TRANSIENT), tail=[StreamText("never")])
    with pytest.raises(Exception, match="error result: success"):
        _run(monkeypatch, tmp_path, runtime)
    assert runtime.calls == 3
    # a failed worker report must have been written (worker_reports.jsonl, appended)
    reports = list(tmp_path.glob("**/worker_reports.jsonl"))
    assert reports, "expected worker_reports.jsonl to be written on final failure"
    rows = [json.loads(l) for l in reports[0].read_text().splitlines() if l.strip()]
    assert any(r.get("status") == "failed" for r in rows), "expected a failed worker report row"


def test_non_transient_does_not_retry(monkeypatch, tmp_path: Path) -> None:
    runtime = FlakyRuntime(fail_times=99, exc=Exception(_NON_TRANSIENT), tail=[StreamText("x")])
    with pytest.raises(Exception, match="error_max_turns"):
        _run(monkeypatch, tmp_path, runtime)
    assert runtime.calls == 1  # no retry on a non-transient error


def test_retries_disabled(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENRESEARCH_SUBAGENT_TRANSPORT_RETRIES", "0")
    runtime = FlakyRuntime(fail_times=99, exc=Exception(_TRANSIENT), tail=[StreamText("x")])
    with pytest.raises(Exception, match="error result: success"):
        _run(monkeypatch, tmp_path, runtime)
    assert runtime.calls == 1  # 0 retries => 1 attempt


# ---------------------------------------------------------------------------
# FM-001 silent-hang idle timeout (2026-07-07)
# ---------------------------------------------------------------------------

class HangingRuntime:
    """Hangs (awaits forever, yields NO event) on the first `hang_times` calls,
    then yields `tail`. Simulates the claude-agent-sdk generator wedging with no
    exception — the failure mode the idle watchdog must catch."""

    def __init__(self, hang_times: int, tail: list[StreamEvent],
                 provider: ProviderName = "anthropic") -> None:
        self.hang_times = hang_times
        self.tail = tail
        self.calls = 0
        self._provider = provider

    @property
    def provider_name(self) -> ProviderName:
        return self._provider

    async def run_agent(self, *, agent: AgentRuntimeSpec, user_input: str) -> AsyncIterator[StreamEvent]:
        self.calls += 1
        if self.calls <= self.hang_times:
            await asyncio.sleep(30)  # hang, emit nothing -> idle watchdog must fire
            yield  # type: ignore[misc]  # unreachable — makes this an async generator
        for ev in self.tail:
            yield ev


class SlowButActiveRuntime:
    """Streams an event every 0.1s for a total wall-clock LONGER than idle_s,
    but never idle for idle_s — the watchdog must NOT fire (no false cancel)."""

    def __init__(self, n: int = 6, provider: ProviderName = "anthropic") -> None:
        self.n = n
        self.calls = 0
        self._provider = provider

    @property
    def provider_name(self) -> ProviderName:
        return self._provider

    async def run_agent(self, *, agent: AgentRuntimeSpec, user_input: str) -> AsyncIterator[StreamEvent]:
        self.calls += 1
        for i in range(self.n):
            await asyncio.sleep(0.1)
            yield StreamText(f"t{i}")


def test_idle_timeout_hang_then_success(monkeypatch, tmp_path: Path) -> None:
    # A hung first attempt (no events) must trip the idle watchdog, be treated
    # as transient, and retry — the second attempt streams normally.
    monkeypatch.setenv("OPENRESEARCH_SUBAGENT_IDLE_TIMEOUT_S", "0.4")
    runtime = HangingRuntime(hang_times=1, tail=[StreamText("recovered")])
    text = _run(monkeypatch, tmp_path, runtime)
    assert runtime.calls == 2  # 1 hang (idle-timeout) + 1 success
    assert text == "recovered"


def test_idle_timeout_exhausts_then_raises(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENRESEARCH_SUBAGENT_IDLE_TIMEOUT_S", "0.3")
    runtime = HangingRuntime(hang_times=99, tail=[StreamText("never")])
    with pytest.raises(Exception, match="subagent_idle_timeout"):
        _run(monkeypatch, tmp_path, runtime)
    assert runtime.calls == 3  # default 2 retries + 1


def test_idle_timeout_does_not_fire_for_active_stream(monkeypatch, tmp_path: Path) -> None:
    # Events every 0.1s, total 0.6s > idle_s 0.4s, but never idle 0.4s -> no cancel.
    monkeypatch.setenv("OPENRESEARCH_SUBAGENT_IDLE_TIMEOUT_S", "0.4")
    runtime = SlowButActiveRuntime(n=6)
    text = _run(monkeypatch, tmp_path, runtime)
    assert runtime.calls == 1  # completed on the first attempt, no false timeout
    assert text == "t0\nt1\nt2\nt3\nt4\nt5"  # collect_agent_text joins with newlines


def test_idle_timeout_disabled_awaits_unbounded(monkeypatch, tmp_path: Path) -> None:
    # idle_s=0 disables the watchdog: a normal (non-hanging) call is byte-identical.
    monkeypatch.setenv("OPENRESEARCH_SUBAGENT_IDLE_TIMEOUT_S", "0")
    runtime = HangingRuntime(hang_times=0, tail=[StreamText("ok")])
    text = _run(monkeypatch, tmp_path, runtime)
    assert runtime.calls == 1
    assert text == "ok"
