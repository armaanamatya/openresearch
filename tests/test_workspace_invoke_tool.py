"""Tests for WorkspaceAppService.invoke_tool() — #47."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from backend.eventstore.sqlite_store import SqliteEventStore
from backend.messaging.envelope import AggregateId
from backend.messaging.event import _clear_registry_for_tests, register_event
from backend.schemas.citations import Citation
from backend.services.context.workspace.events import (
    ToolInvoked,
    VariableLoaded,
    WorkspaceClosed,
    WorkspaceCreated,
    WorkspaceReady,
)
from backend.services.context.workspace.model import Cited
from backend.services.context.workspace.service import (
    WorkspaceAppService,
    WorkspaceError,
)
from backend.services.context.workspace.tools.rlm_query import RlmQueryTool


# --- test doubles -----------------------------------------------------------


@dataclass
class _CountingLlm:
    calls: list[tuple[str, str]] = field(default_factory=list)

    def complete(self, *, system: str, user: str) -> str:
        self.calls.append((system, user))
        return f"[leaf-{len(self.calls)}]"


class _StubView:
    def __init__(self, name: str, value: Any, citations: tuple[Citation, ...]) -> None:
        self._name = name
        self._cited = Cited(value=value, citations=citations)

    def get(self, name: str):
        return self._cited if name == self._name else None

    def variable_names(self) -> set[str]:
        return {self._name}


class _FakeWorkspaceTool:
    """Minimal tool that returns a canned Cited[dict]."""

    name = "fake_tool"

    def __init__(self, answer: dict, citations: tuple[Citation, ...]) -> None:
        self._result = Cited(value=answer, citations=citations)
        self.call_count = 0

    def call(self, *, workspace_id: str, **kwargs: Any) -> Cited[dict]:
        self.call_count += 1
        return self._result


# --- fixtures ---------------------------------------------------------------


def _re_register_all() -> None:
    from backend.services.context.workspace.events import (
        CitationAttached,
        VariableEnriched,
        VariablePromoted,
    )

    for cls in (
        WorkspaceCreated,
        VariableLoaded,
        VariableEnriched,
        CitationAttached,
        ToolInvoked,
        VariablePromoted,
        WorkspaceReady,
        WorkspaceClosed,
    ):
        register_event(cls)


@pytest.fixture
def store(tmp_path: Path):
    _clear_registry_for_tests()
    _re_register_all()
    s = SqliteEventStore(f"sqlite:///{tmp_path}/events.db")
    yield s
    s.close()
    _clear_registry_for_tests()


def _make_ready_workspace(store, workspace_id: str = "ws_test") -> str:
    """Bootstrap a workspace to READY state directly via events."""
    from backend.messaging.envelope import make_envelope, new_correlation_id

    cid = new_correlation_id()
    agg_id = AggregateId(workspace_id)
    citation = Citation(
        source_id="s1", chunk_id="c1", quote="test", locator="test.pdf"
    )
    events = [
        WorkspaceCreated(
            workspace_id=workspace_id,
            project_id="proj_1",
            agent_name="default",
        ),
        VariableLoaded(
            workspace_id=workspace_id,
            variable_name="paper_text",
            value_payload={"text": "Short paper text for testing."},
            citations=(citation,),
        ),
        WorkspaceReady(workspace_id=workspace_id, variable_count=1),
    ]
    envelopes = [
        make_envelope(source="test", correlation_id=cid) for _ in events
    ]
    store.append(
        aggregate_id=agg_id,
        aggregate_type="workspace",
        events=events,
        expected_version=0,
        envelopes=envelopes,
    )
    return workspace_id


def _make_service(store) -> WorkspaceAppService:
    """Build a workspace service with a stub indexer (invoke_tool doesn't need it)."""

    class _StubIndexer:
        pass

    return WorkspaceAppService(store=store, indexer=_StubIndexer())  # type: ignore[arg-type]


# --- invoke_tool tests ------------------------------------------------------


def test_invoke_tool_returns_cited_result(store):
    wsid = _make_ready_workspace(store)
    svc = _make_service(store)
    citation = Citation(
        source_id="s1", chunk_id="c1", quote="evidence", locator="paper.pdf"
    )
    tool = _FakeWorkspaceTool(
        answer={"answer": "42", "llm_calls": 1}, citations=(citation,)
    )

    result = svc.invoke_tool(workspace_id=wsid, tool=tool, question="What is X?")

    assert result.value == {"answer": "42", "llm_calls": 1}
    assert len(result.citations) == 1
    assert tool.call_count == 1


def test_invoke_tool_emits_tool_invoked_event(store):
    wsid = _make_ready_workspace(store)
    svc = _make_service(store)
    citation = Citation(
        source_id="s1", chunk_id="c1", quote="evidence", locator="paper.pdf"
    )
    tool = _FakeWorkspaceTool(
        answer={"answer": "hello", "depth": 2}, citations=(citation,)
    )

    svc.invoke_tool(
        workspace_id=wsid, tool=tool, question="Q?", variable_name="paper_text"
    )

    # Read back events from the store
    stored = list(store.load(AggregateId(wsid)))
    tool_events = [e for e in stored if e.event_type == "tool_invoked"]
    assert len(tool_events) == 1

    payload = tool_events[0].payload
    assert payload["tool_name"] == "fake_tool"
    assert payload["arguments"]["question"] == "Q?"
    assert payload["arguments"]["variable_name"] == "paper_text"
    assert payload["result_payload"] == {"answer": "hello", "depth": 2}
    assert payload["duration_ms"] >= 0


def test_invoke_tool_records_duration(store):
    wsid = _make_ready_workspace(store)
    svc = _make_service(store)
    citation = Citation(
        source_id="s1", chunk_id="c1", quote="ev", locator="L"
    )
    tool = _FakeWorkspaceTool(answer={"a": 1}, citations=(citation,))

    svc.invoke_tool(workspace_id=wsid, tool=tool)

    stored = list(store.load(AggregateId(wsid)))
    tool_events = [e for e in stored if e.event_type == "tool_invoked"]
    assert tool_events[0].payload["duration_ms"] >= 0


def test_invoke_tool_rejects_new_workspace(store):
    """Cannot invoke tools on a workspace that hasn't been created."""
    svc = _make_service(store)
    citation = Citation(
        source_id="s1", chunk_id="c1", quote="ev", locator="L"
    )
    tool = _FakeWorkspaceTool(answer={"a": 1}, citations=(citation,))

    with pytest.raises(WorkspaceError, match="must be 'loading' or 'ready'"):
        svc.invoke_tool(workspace_id="ws_nonexistent", tool=tool)


def test_invoke_tool_with_rlm_query_tool(store):
    """Integration: invoke_tool works with an actual RlmQueryTool + StubLlm."""
    wsid = _make_ready_workspace(store)
    svc = _make_service(store)

    view = _StubView(
        "paper_text",
        "Short content for leaf path.",
        citations=(
            Citation(
                source_id="s1", chunk_id="c1", quote="Short content", locator="p.pdf"
            ),
        ),
    )
    llm = _CountingLlm()
    rlm_tool = RlmQueryTool(
        view_provider=lambda _wsid: view, llm_client=llm
    )

    result = svc.invoke_tool(
        workspace_id=wsid,
        tool=rlm_tool,
        question="What is this about?",
        variable_name="paper_text",
    )

    assert result.value["answer"] == "[leaf-1]"
    assert result.value["llm_calls"] == 1
    assert result.value["depth_reached"] == 0
    assert len(llm.calls) == 1

    # Verify event emitted
    stored = list(store.load(AggregateId(wsid)))
    tool_events = [e for e in stored if e.event_type == "tool_invoked"]
    assert len(tool_events) == 1
    assert tool_events[0].payload["tool_name"] == "rlm_query"


def test_invoke_tool_non_dict_result_wraps_in_raw(store):
    """When tool returns a non-dict value, it's wrapped as {"raw": value}."""
    wsid = _make_ready_workspace(store)
    svc = _make_service(store)
    citation = Citation(
        source_id="s1", chunk_id="c1", quote="ev", locator="L"
    )

    class _StringTool:
        name = "string_tool"

        def call(self, *, workspace_id: str, **kwargs) -> Cited[str]:
            return Cited(value="plain string", citations=(citation,))

    result = svc.invoke_tool(workspace_id=wsid, tool=_StringTool())

    assert result.value == "plain string"
    stored = list(store.load(AggregateId(wsid)))
    tool_events = [e for e in stored if e.event_type == "tool_invoked"]
    assert tool_events[0].payload["result_payload"] == {"raw": "plain string"}
