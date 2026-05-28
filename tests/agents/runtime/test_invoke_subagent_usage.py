"""TDD tests for Phase 0: StreamUsage capture in collect_agent_text.

Tests inject a fake AgentRuntime that yields StreamUsage events directly,
bypassing the SDK adapter layer so we test exactly what collect_agent_text
does with those events (ledger write + SSE event).

T1 – StreamUsage tokens accumulate and are written to cost_ledger.jsonl.
T2 – cost_ledger row has correct agent_id, model, provider, token fields.
T3 – multiple StreamUsage events are summed.
T4 – no StreamUsage emitted → zero-token row, no error.
T5 – subagent_usage SSE event is appended to dashboard_events.jsonl.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncIterator

import pytest

from backend.agents.runtime.base import AgentRuntimeSpec, ProviderName, StreamEvent, StreamText, StreamUsage


class FakeRuntime:
    """Emits a fixed sequence of StreamEvents without hitting any SDK."""

    def __init__(self, events: list[StreamEvent], provider: ProviderName = "anthropic") -> None:
        self._events = events
        self._provider = provider

    @property
    def provider_name(self) -> ProviderName:
        return self._provider

    async def run_agent(self, *, agent: AgentRuntimeSpec, user_input: str) -> AsyncIterator[StreamEvent]:
        for ev in self._events:
            yield ev


def _fake_registry(agent_id: str, tmp_path: Path) -> dict:
    spec = AgentRuntimeSpec(
        name=agent_id,
        instructions="system",
        model="claude-sonnet-4-5",
        tools=(),
        working_directory=tmp_path,
    )

    class FakeEntry:
        def to_runtime_spec(self, provider, *, model_override=None, working_directory=None, max_turns=None):
            return spec

    return {agent_id: FakeEntry()}


# ---------------------------------------------------------------------------
# T1 – tokens written to cost_ledger.jsonl
# ---------------------------------------------------------------------------

def test_subagent_usage_written_to_cost_ledger(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("backend.agents.runtime.invoke.AGENT_REGISTRY",
                        _fake_registry("baseline-implementation", tmp_path))

    runtime = FakeRuntime([
        StreamText("implementing..."),
        StreamUsage(input_tokens=120, output_tokens=30,
                    cache_read_input_tokens=50, cache_creation_input_tokens=10),
    ])

    from backend.agents.runtime.invoke import collect_agent_text
    asyncio.run(collect_agent_text(
        "baseline-implementation", "write code",
        project_dir=tmp_path, runtime=runtime,
    ))

    ledger = tmp_path / "cost_ledger.jsonl"
    assert ledger.exists(), "cost_ledger.jsonl was not created"
    rows = [json.loads(l) for l in ledger.read_text().splitlines() if l.strip()]
    row = [r for r in rows if r.get("agent_id") == "baseline-implementation"][-1]
    assert row["tokens_in"] == 120
    assert row["tokens_out"] == 30
    assert row.get("cache_read_input_tokens") == 50


# ---------------------------------------------------------------------------
# T2 – ledger row has correct metadata fields
# ---------------------------------------------------------------------------

def test_subagent_usage_row_metadata(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("backend.agents.runtime.invoke.AGENT_REGISTRY",
                        _fake_registry("baseline-implementation", tmp_path))

    runtime = FakeRuntime([
        StreamUsage(input_tokens=5, output_tokens=2),
    ])

    from backend.agents.runtime.invoke import collect_agent_text
    asyncio.run(collect_agent_text(
        "baseline-implementation", "write code",
        project_dir=tmp_path, runtime=runtime,
    ))

    ledger = tmp_path / "cost_ledger.jsonl"
    rows = [json.loads(l) for l in ledger.read_text().splitlines() if l.strip()]
    row = [r for r in rows if r.get("agent_id") == "baseline-implementation"][-1]
    assert row["provider"] == "anthropic"
    assert "model" in row
    assert "timestamp" in row


# ---------------------------------------------------------------------------
# T3 – multiple StreamUsage events summed
# ---------------------------------------------------------------------------

def test_multiple_stream_usage_events_summed(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("backend.agents.runtime.invoke.AGENT_REGISTRY",
                        _fake_registry("baseline-implementation", tmp_path))

    runtime = FakeRuntime([
        StreamUsage(input_tokens=40, output_tokens=5),
        StreamUsage(input_tokens=60, output_tokens=10),
    ])

    from backend.agents.runtime.invoke import collect_agent_text
    asyncio.run(collect_agent_text(
        "baseline-implementation", "write code",
        project_dir=tmp_path, runtime=runtime,
    ))

    ledger = tmp_path / "cost_ledger.jsonl"
    rows = [json.loads(l) for l in ledger.read_text().splitlines() if l.strip()]
    row = [r for r in rows if r.get("agent_id") == "baseline-implementation"][-1]
    assert row["tokens_in"] == 100
    assert row["tokens_out"] == 15


# ---------------------------------------------------------------------------
# T4 – no StreamUsage → zero-token row, no error
# ---------------------------------------------------------------------------

def test_no_stream_usage_writes_zero_row(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("backend.agents.runtime.invoke.AGENT_REGISTRY",
                        _fake_registry("baseline-implementation", tmp_path))

    runtime = FakeRuntime([StreamText("done")])

    from backend.agents.runtime.invoke import collect_agent_text
    asyncio.run(collect_agent_text(
        "baseline-implementation", "write code",
        project_dir=tmp_path, runtime=runtime,
    ))

    ledger = tmp_path / "cost_ledger.jsonl"
    assert ledger.exists()
    rows = [json.loads(l) for l in ledger.read_text().splitlines() if l.strip()]
    row = [r for r in rows if r.get("agent_id") == "baseline-implementation"][-1]
    assert row["tokens_in"] == 0
    assert row["tokens_out"] == 0


# ---------------------------------------------------------------------------
# T5 – subagent_usage SSE event written to dashboard_events.jsonl
# ---------------------------------------------------------------------------

def test_subagent_usage_sse_event_emitted(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("backend.agents.runtime.invoke.AGENT_REGISTRY",
                        _fake_registry("baseline-implementation", tmp_path))

    runtime = FakeRuntime([
        StreamUsage(input_tokens=20, output_tokens=4),
    ])

    from backend.agents.runtime.invoke import collect_agent_text
    asyncio.run(collect_agent_text(
        "baseline-implementation", "write code",
        project_dir=tmp_path, runtime=runtime,
    ))

    events_file = tmp_path / "dashboard_events.jsonl"
    assert events_file.exists(), "dashboard_events.jsonl not created"
    events = [json.loads(l) for l in events_file.read_text().splitlines() if l.strip()]
    usage_events = [e for e in events if e.get("event") == "subagent_usage"]
    assert len(usage_events) >= 1
    data = usage_events[-1]["data"]
    assert data["agent_id"] == "baseline-implementation"
    assert data["input_tokens"] == 20
