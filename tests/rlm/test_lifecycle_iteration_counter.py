"""Lifecycle-primary must advance ctx.current_iteration per driven step so the
rubric_score event + ctx.latest_rubric_iteration + finalize's iteration count
are real (not 0). binding.wrap_primitive reads ctx.current_iteration."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from backend.agents.rlm.lifecycle_driver import drive_lifecycle_chain, run_lifecycle_primary


def _ctx(tmp_path):
    d = tmp_path / "proj"
    d.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(project_dir=d, remaining_s=lambda: None, current_iteration=0)


def _tool(ret):
    m = MagicMock()
    m.return_value = ret
    return m


def _tools():
    return {
        "understand_section": {"tool": _tool({"sections": []})},
        "detect_environment": {"tool": _tool({"env": "conda"})},
        "plan_reproduction": {"tool": _tool({"method_spec": "x"})},
        "implement_baseline": {"tool": _tool({"ok": True, "code_path": "/c"})},
        "run_experiment": {"tool": _tool({"success": True, "metrics": {"r": 0.7}})},
        "verify_against_rubric": {"tool": _tool({"overall_score": 0.9, "meets_target": True})},
    }


def test_full_backbone_advances_iteration_to_six(tmp_path):
    ctx = _ctx(tmp_path)
    drive_lifecycle_chain(
        tools=_tools(), ctx=ctx, paper_text="p", rubric_spec={"target_score": 0.7},
        start_stage="need_baseline", emit=lambda e: None,
    )
    assert ctx.current_iteration == 6


def test_run_primary_stamps_iterations_in_summary(tmp_path):
    ctx = _ctx(tmp_path)
    summary = run_lifecycle_primary(
        tools=_tools(), ctx=ctx, paper_text="p", rubric_spec={"target_score": 0.7},
        emit=lambda e: None, target_score=0.7, max_improve_iterations=0,
    )
    assert summary["iterations"] == ctx.current_iteration
    assert summary["iterations"] >= 6
