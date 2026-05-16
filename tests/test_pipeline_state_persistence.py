"""Invariant: pipeline stage transitions are always persisted to disk.

The Next.js lab UI bridge (`frontend/src/lib/demo/server-payload.ts`) reads
`runs/<id>/pipeline_state.json` to populate `payload.summary.stage`. If a stage
transition mutates `state.stage` in memory without writing the checkpoint, the
on-disk file goes stale and the UI workflow graph freezes at "1/12 agents
complete".

`PipelineState.advance_stage()` is the single sanctioned transition path. These
tests pin that contract: the helper persists atomically, and no module is
allowed to set `.stage` directly outside the two legitimate sites.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from backend.agents.orchestrator import PipelineStage, PipelineState

ORCHESTRATOR_SRC = Path(__file__).resolve().parents[1] / "backend" / "agents" / "orchestrator.py"
PIPELINE_SRC = Path(__file__).resolve().parents[1] / "backend" / "agents" / "pipeline.py"

# The only two places allowed to assign `.stage` directly:
#   - PipelineState.advance_stage  — the sanctioned setter (then persists)
#   - PipelineState.load_checkpoint — deserialization from disk (no persist)
_ALLOWED_STAGE_ASSIGNMENT_SCOPES = {"advance_stage", "load_checkpoint"}


def _bare_stage_assignments(source_path: Path) -> list[tuple[str, int]]:
    """Return (enclosing_function, lineno) for every `*.stage = ...` assignment."""
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    offenders: list[tuple[str, int]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self._scope: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._scope.append(node.name)
            self.generic_visit(node)
            self._scope.pop()

        visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

        def visit_Assign(self, node: ast.Assign) -> None:
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == "stage":
                    scope = self._scope[-1] if self._scope else "<module>"
                    if scope not in _ALLOWED_STAGE_ASSIGNMENT_SCOPES:
                        offenders.append((scope, node.lineno))
            self.generic_visit(node)

    Visitor().visit(tree)
    return offenders


@pytest.mark.parametrize("source_path", [ORCHESTRATOR_SRC, PIPELINE_SRC], ids=lambda p: p.name)
def test_no_bare_stage_assignment(source_path: Path) -> None:
    """Stage transitions must go through `advance_stage`, never `state.stage = X`."""
    offenders = _bare_stage_assignments(source_path)
    assert not offenders, (
        f"{source_path.name} has bare `.stage =` assignments outside "
        f"{sorted(_ALLOWED_STAGE_ASSIGNMENT_SCOPES)}: {offenders}. "
        f"Use `state.advance_stage(stage, runs_root)` so the checkpoint is persisted."
    )


def test_advance_stage_persists_checkpoint(tmp_path: Path) -> None:
    """advance_stage writes pipeline_state.json with the new stage."""
    state = PipelineState(project_id="prj_test")
    assert state.stage is PipelineStage.INGESTED

    written = state.advance_stage(PipelineStage.PAPER_UNDERSTOOD, tmp_path)

    assert written == tmp_path / "prj_test" / "pipeline_state.json"
    assert written.exists()
    on_disk = json.loads(written.read_text())
    assert on_disk["stage"] == "paper_understood"
    assert state.stage is PipelineStage.PAPER_UNDERSTOOD


def test_advance_stage_round_trips_through_load_checkpoint(tmp_path: Path) -> None:
    """A persisted transition is recoverable — checkpoint/resume stays intact."""
    state = PipelineState(project_id="prj_resume")
    state.advance_stage(PipelineStage.ENVIRONMENT_BUILT, tmp_path)

    resumed = PipelineState.load_checkpoint(tmp_path, "prj_resume")
    assert resumed is not None
    assert resumed.stage is PipelineStage.ENVIRONMENT_BUILT


def test_save_checkpoint_is_atomic(tmp_path: Path) -> None:
    """save_checkpoint leaves no .tmp turd and always yields valid JSON."""
    state = PipelineState(project_id="prj_atomic")
    path = state.advance_stage(PipelineStage.PLAN_CREATED, tmp_path)

    run_dir = tmp_path / "prj_atomic"
    assert not list(run_dir.glob("*.tmp")), "atomic write left a temp file behind"
    # File is complete, parseable JSON — never a truncated read.
    json.loads(path.read_text())
