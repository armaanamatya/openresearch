"""Tests for RLM wiring in ReproLabOrchestrator — #50.

Verifies:
  - RlmQueryTool is constructed when workspace_service is provided
  - _rlm_query() returns Cited[dict] on success, None on no workspace / exhausted budget
  - _rlm_evidence_for_stage() returns dict of answers
  - RunBudget.rlm_calls_remaining persists through checkpoint
  - Budget decrement is correct
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from backend.agents.orchestrator import PipelineState, PipelineStage, ReproLabOrchestrator
from backend.agents.resilience import RunBudget
from backend.schemas.citations import Citation
from backend.services.context.workspace.model import Cited
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


class _StubWorkspaceService:
    """Minimal workspace service that supports materialize_view and invoke_tool."""

    def __init__(self, view: _StubView) -> None:
        self._view = view
        self.invoke_tool_calls: list[dict] = []

    def materialize_view(self, workspace_id: str) -> _StubView:
        return self._view

    def invoke_tool(self, *, workspace_id: str, tool: Any, **kwargs: Any) -> Any:
        result = tool.call(workspace_id=workspace_id, **kwargs)
        self.invoke_tool_calls.append({
            "workspace_id": workspace_id,
            "tool_name": tool.name,
            "kwargs": kwargs,
            "result": result,
        })
        return result


def _make_orchestrator(
    tmp_path: Path,
    *,
    workspace_service: Any = None,
    workspace_id: str | None = None,
    run_budget: RunBudget | None = None,
) -> ReproLabOrchestrator:
    """Build a minimal orchestrator for testing RLM wiring."""
    return ReproLabOrchestrator(
        project_id="test_proj",
        runs_root=tmp_path,
        workspace_service=workspace_service,
        workspace_id=workspace_id,
        run_budget=run_budget,
    )


def _make_stub_workspace(text: str = "Short paper content for testing."):
    citation = Citation(
        source_id="s1", chunk_id="c1", quote="Short paper", locator="paper.pdf"
    )
    view = _StubView("paper_text", text, citations=(citation,))
    return _StubWorkspaceService(view)


# --- RlmQueryTool construction -------------------------------------------


def test_rlm_tool_constructed_when_workspace_provided(tmp_path):
    ws = _make_stub_workspace()
    orch = _make_orchestrator(tmp_path, workspace_service=ws, workspace_id="ws_1")
    assert orch._rlm_tool is not None
    assert isinstance(orch._rlm_tool, RlmQueryTool)


def test_rlm_tool_none_without_workspace(tmp_path):
    orch = _make_orchestrator(tmp_path)
    assert orch._rlm_tool is None


# --- _rlm_query() ---------------------------------------------------------


def test_rlm_query_returns_cited_dict(tmp_path):
    ws = _make_stub_workspace()
    orch = _make_orchestrator(tmp_path, workspace_service=ws, workspace_id="ws_1")
    # Replace the LLM client with a counting stub
    orch._rlm_tool._llm = _CountingLlm()

    state = PipelineState(project_id="test_proj")
    result = orch._rlm_query(state, "What is this about?")

    assert result is not None
    assert isinstance(result, Cited)
    assert result.value["answer"] == "[leaf-1]"
    assert result.value["llm_calls"] == 1
    assert len(result.citations) == 1


def test_rlm_query_returns_none_without_workspace(tmp_path):
    orch = _make_orchestrator(tmp_path)
    state = PipelineState(project_id="test_proj")
    result = orch._rlm_query(state, "What is this about?")
    assert result is None


def test_rlm_query_returns_none_when_budget_exhausted(tmp_path):
    ws = _make_stub_workspace()
    orch = _make_orchestrator(tmp_path, workspace_service=ws, workspace_id="ws_1")
    orch._rlm_tool._llm = _CountingLlm()

    state = PipelineState(project_id="test_proj", rlm_calls_remaining=0)
    result = orch._rlm_query(state, "What is this about?")
    assert result is None


def test_rlm_query_decrements_budget(tmp_path):
    ws = _make_stub_workspace()
    orch = _make_orchestrator(tmp_path, workspace_service=ws, workspace_id="ws_1")
    orch._rlm_tool._llm = _CountingLlm()

    state = PipelineState(project_id="test_proj", rlm_calls_remaining=10)
    orch._rlm_query(state, "Q1?")
    assert state.rlm_calls_remaining == 9  # decremented by llm_calls=1

    orch._rlm_query(state, "Q2?")
    assert state.rlm_calls_remaining == 8


def test_rlm_query_routes_through_invoke_tool(tmp_path):
    ws = _make_stub_workspace()
    orch = _make_orchestrator(tmp_path, workspace_service=ws, workspace_id="ws_1")
    orch._rlm_tool._llm = _CountingLlm()

    state = PipelineState(project_id="test_proj")
    orch._rlm_query(state, "Test question?")

    assert len(ws.invoke_tool_calls) == 1
    call = ws.invoke_tool_calls[0]
    assert call["tool_name"] == "rlm_query"
    assert call["kwargs"]["question"] == "Test question?"
    assert call["kwargs"]["variable_name"] == "paper_text"


# --- _rlm_evidence_for_stage() -------------------------------------------


def test_rlm_evidence_for_stage_returns_answers(tmp_path):
    ws = _make_stub_workspace()
    orch = _make_orchestrator(tmp_path, workspace_service=ws, workspace_id="ws_1")
    orch._rlm_tool._llm = _CountingLlm()

    state = PipelineState(project_id="test_proj")
    evidence = orch._rlm_evidence_for_stage(state, ["Q1?", "Q2?", "Q3?"])

    assert len(evidence) == 3
    assert all("answer" in v for v in evidence.values())
    assert state.rlm_calls_remaining == 120 - 3  # 3 leaf calls


def test_rlm_evidence_empty_when_no_workspace(tmp_path):
    orch = _make_orchestrator(tmp_path)
    state = PipelineState(project_id="test_proj")
    evidence = orch._rlm_evidence_for_stage(state, ["Q1?", "Q2?"])
    assert evidence == {}


def test_rlm_evidence_stops_at_budget(tmp_path):
    ws = _make_stub_workspace()
    orch = _make_orchestrator(tmp_path, workspace_service=ws, workspace_id="ws_1")
    orch._rlm_tool._llm = _CountingLlm()

    state = PipelineState(project_id="test_proj", rlm_calls_remaining=2)
    evidence = orch._rlm_evidence_for_stage(state, ["Q1?", "Q2?", "Q3?", "Q4?"])

    # First 2 succeed (each uses 1 llm_call), rest return None
    assert len(evidence) == 2
    assert state.rlm_calls_remaining == 0


# --- RunBudget.rlm_calls_remaining ---------------------------------------


def test_run_budget_default_rlm_calls():
    budget = RunBudget()
    assert budget.rlm_calls_remaining == 120


def test_run_budget_custom_rlm_calls():
    budget = RunBudget(rlm_calls_remaining=50)
    assert budget.rlm_calls_remaining == 50


def test_fresh_state_inherits_budget_from_run_budget(tmp_path):
    ws = _make_stub_workspace()
    budget = RunBudget(rlm_calls_remaining=42)
    orch = _make_orchestrator(
        tmp_path, workspace_service=ws, workspace_id="ws_1", run_budget=budget
    )
    # Simulate what run() does for fresh state
    state = PipelineState(
        project_id="test_proj",
        rlm_calls_remaining=orch._run_budget.rlm_calls_remaining,
    )
    assert state.rlm_calls_remaining == 42


# --- Checkpoint round-trip ------------------------------------------------


def test_rlm_calls_remaining_persists_through_checkpoint(tmp_path):
    state = PipelineState(project_id="test_proj", rlm_calls_remaining=77)
    state.save_checkpoint(tmp_path)

    loaded = PipelineState.load_checkpoint(tmp_path, "test_proj")
    assert loaded is not None
    assert loaded.rlm_calls_remaining == 77


def test_checkpoint_without_rlm_field_defaults_to_120(tmp_path):
    """Old checkpoints without rlm_calls_remaining should default to 120."""
    state = PipelineState(project_id="test_proj")
    state.save_checkpoint(tmp_path)

    # Simulate old checkpoint: remove the field
    import json
    path = tmp_path / "test_proj" / "pipeline_state.json"
    data = json.loads(path.read_text())
    del data["rlm_calls_remaining"]
    path.write_text(json.dumps(data))

    loaded = PipelineState.load_checkpoint(tmp_path, "test_proj")
    assert loaded is not None
    assert loaded.rlm_calls_remaining == 120
