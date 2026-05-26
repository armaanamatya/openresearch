"""β3/β4: GET /leaderboard surfaces compute_adjusted_score + execution_mode columns.

Tests:
- Both rows have the new fields
- compute_adjusted_score falls back to overall_score for legacy runs
- execution_mode read from demo_status.json
- Default sort uses compute_adjusted_score
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.routes.leaderboard import aggregate_leaderboard


def _write_run(
    runs_root: Path,
    project_id: str,
    *,
    paper_id: str = "p",
    overall_score: float = 0.5,
    compute_adjusted_score: float | None = None,
    execution_mode: str | None = None,
    verdict: str = "partial",
) -> None:
    run_dir = runs_root / project_id
    run_dir.mkdir(parents=True, exist_ok=True)
    rubric: dict = {
        "overall_score": overall_score,
        "meets_target": False,
        "areas": [],
    }
    if compute_adjusted_score is not None:
        rubric["compute_adjusted_score"] = compute_adjusted_score
    (run_dir / "final_report.json").write_text(json.dumps({
        "paper": {"id": paper_id, "title": paper_id},
        "verdict": verdict,
        "reproduction_summary": "x",
        "baseline_metrics": {},
        "paper_claims": {},
        "rubric": rubric,
        "improvements": [],
        "primitive_trace": {},
        "cost": {"llm_usd": 1.0, "primitives": 0.0},
        "iterations": 1,
        "primitive_provider": "real",
        "degraded": False,
        "mode": "rlm",
        "models": {"planner": None, "executor": None, "verifier": None, "grader": None},
        "started_at": "2026-05-25T00:00:00+00:00",
        "completed_at": "2026-05-25T01:00:00+00:00",
    }))
    ds: dict = {"projectId": project_id, "status": "completed"}
    if execution_mode:
        ds["executionMode"] = execution_mode
    (run_dir / "demo_status.json").write_text(json.dumps(ds))


def test_leaderboard_has_compute_adjusted_score_column(tmp_path: Path):
    _write_run(tmp_path, "prj_a", paper_id="A",
               overall_score=0.7, compute_adjusted_score=0.7, execution_mode="max")
    rows = aggregate_leaderboard(tmp_path)
    assert len(rows) == 1
    assert rows[0].compute_adjusted_score == pytest.approx(0.7)
    assert rows[0].execution_mode == "max"


def test_leaderboard_has_execution_mode_column(tmp_path: Path):
    _write_run(tmp_path, "prj_b", paper_id="B",
               overall_score=0.4, compute_adjusted_score=0.85, execution_mode="efficient")
    rows = aggregate_leaderboard(tmp_path)
    assert rows[0].execution_mode == "efficient"
    assert rows[0].compute_adjusted_score == pytest.approx(0.85)


def test_leaderboard_compute_adjusted_falls_back_to_overall_score_for_legacy_runs(tmp_path: Path):
    """Old reports without compute_adjusted_score → fallback to overall_score."""
    _write_run(tmp_path, "prj_legacy", paper_id="L",
               overall_score=0.6, compute_adjusted_score=None)
    rows = aggregate_leaderboard(tmp_path)
    assert rows[0].compute_adjusted_score == pytest.approx(0.6)


def test_leaderboard_sorts_by_compute_adjusted_score_by_default(tmp_path: Path):
    """Default sort (score) uses compute_adjusted_score so efficient+max are comparable."""
    # Run A: max mode, raw=adjusted=0.7
    _write_run(tmp_path, "prj_a", paper_id="A",
               overall_score=0.7, compute_adjusted_score=0.7, execution_mode="max")
    # Run B: efficient mode, raw=0.4, adjusted=0.85 (floor-anchored)
    _write_run(tmp_path, "prj_b", paper_id="B",
               overall_score=0.4, compute_adjusted_score=0.85, execution_mode="efficient")

    rows = aggregate_leaderboard(tmp_path, order_by="score")
    assert len(rows) == 2
    # Run B should rank first because its compute_adjusted_score (0.85) > A's (0.7)
    assert rows[0].paper_id == "B"
    assert rows[1].paper_id == "A"


def test_leaderboard_execution_mode_none_for_old_runs_without_demo_status(tmp_path: Path):
    """Runs without demo_status.json get execution_mode=None (not an error)."""
    run_dir = tmp_path / "prj_old"
    run_dir.mkdir()
    (run_dir / "final_report.json").write_text(json.dumps({
        "paper": {"id": "x"},
        "verdict": "partial",
        "reproduction_summary": "",
        "baseline_metrics": {},
        "paper_claims": {},
        "rubric": {"overall_score": 0.5, "meets_target": False, "areas": []},
        "cost": {"llm_usd": 0.0},
        "iterations": 0,
        "mode": "rlm",
        "models": {},
        "started_at": None,
        "completed_at": None,
    }))
    rows = aggregate_leaderboard(tmp_path)
    assert rows[0].execution_mode is None
