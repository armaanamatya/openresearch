"""TDD tests for attempt-aware leaderboard aggregation.

Covers:
- extract_scores: nested overall; compute_adjusted-only; flat rubric_score;
  rubric is a list; empty dict.
- resolve_best_report_picks_best_attempt: top-level 0.188 + attempts/<ts> 0.488
  → returns the 0.488 report, picked_from_attempt=True, attempts_total=1.
- leaderboard_row_surfaces_best_attempt: aggregate_leaderboard over a tmp runs_root
  containing that project → row compute_adjusted_score == 0.488.
- leaderboard_status_terminalizes: demo_status status="running" + final_report
  present → row status == "completed".
- Existing leaderboard field schema now includes status + attempts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.services.runs.report_resolution import (
    ResolvedReport,
    extract_scores,
    normalized_score,
    resolve_best_report,
)
from backend.routes.leaderboard import LeaderboardRow, aggregate_leaderboard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_report(path: Path, overall: float | None, adjusted: float | None = None) -> None:
    rubric: dict = {"overall_score": overall, "meets_target": False, "areas": []}
    if adjusted is not None:
        rubric["compute_adjusted_score"] = adjusted
    path.write_text(json.dumps({
        "paper": {"id": "p1", "title": "Test"},
        "verdict": "partial",
        "reproduction_summary": "",
        "baseline_metrics": {},
        "paper_claims": {},
        "rubric": rubric,
        "improvements": [],
        "primitive_trace": {},
        "cost": {"llm_usd": 1.0, "primitives": 0.0},
        "iterations": 1,
        "mode": "rlm",
        "models": {"planner": None, "executor": None, "verifier": None, "grader": None},
        "started_at": "2026-05-31T00:00:00+00:00",
        "completed_at": "2026-05-31T01:00:00+00:00",
    }))


def _write_flat_report(path: Path, rubric_score: float) -> None:
    """Flat-schema (old rlm_oauth_smoke-style) where rubric_score is top-level."""
    path.write_text(json.dumps({
        "paper": {"id": "p1", "title": "Test"},
        "verdict": "partial",
        "rubric_score": rubric_score,
        "cost": {"llm_usd": 1.0},
        "iterations": 1,
        "mode": "rlm",
        "models": {},
        "started_at": "2026-05-31T00:00:00+00:00",
        "completed_at": "2026-05-31T01:00:00+00:00",
    }))


@pytest.fixture(autouse=True)
def _clear_leaderboard_cache():
    from backend.services.events import leaderboard_cache
    leaderboard_cache.clear()
    yield
    leaderboard_cache.clear()


# ---------------------------------------------------------------------------
# extract_scores
# ---------------------------------------------------------------------------


def test_extract_scores_nested_overall():
    report = {"rubric": {"overall_score": 0.42, "compute_adjusted_score": 0.55}}
    overall, adjusted = extract_scores(report)
    assert overall == pytest.approx(0.42)
    assert adjusted == pytest.approx(0.55)


def test_extract_scores_compute_adjusted_only_overall_none():
    """overall_score=None but compute_adjusted_score set — overall=None, adjusted=0.488."""
    report = {"rubric": {"overall_score": None, "compute_adjusted_score": 0.488}}
    overall, adjusted = extract_scores(report)
    assert overall is None
    assert adjusted == pytest.approx(0.488)


def test_extract_scores_flat_rubric_score_fallback():
    """Legacy flat schema: no rubric dict, top-level rubric_score."""
    report = {"rubric_score": 0.33}
    overall, adjusted = extract_scores(report)
    assert overall == pytest.approx(0.33)
    assert adjusted == pytest.approx(0.33)


def test_extract_scores_rubric_is_list_falls_back_to_flat():
    """rubric is a list (old legacy shape) → treated as absent, falls back to flat."""
    report = {"rubric": [{"area": "x"}], "rubric_score": 0.25}
    overall, adjusted = extract_scores(report)
    assert overall == pytest.approx(0.25)
    assert adjusted == pytest.approx(0.25)


def test_extract_scores_empty_dict():
    """Empty report — both scores are None."""
    overall, adjusted = extract_scores({})
    assert overall is None
    assert adjusted is None


# ---------------------------------------------------------------------------
# normalized_score
# ---------------------------------------------------------------------------


def test_normalized_score_prefers_adjusted():
    assert normalized_score(0.3, 0.7) == pytest.approx(0.7)


def test_normalized_score_falls_back_to_overall_when_adjusted_none():
    assert normalized_score(0.5, None) == pytest.approx(0.5)


def test_normalized_score_none_when_both_none():
    assert normalized_score(None, None) is None


# ---------------------------------------------------------------------------
# resolve_best_report
# ---------------------------------------------------------------------------


def test_resolve_best_report_picks_best_attempt(tmp_path: Path):
    """Best attempt (0.488) beats top-level (0.188)."""
    run_dir = tmp_path / "prj_test"
    run_dir.mkdir()

    # Top-level: overall=0.188, adjusted=0.188
    _write_report(run_dir / "final_report.json", overall=0.188, adjusted=0.188)

    # Attempt: overall=None, adjusted=0.488
    attempt_dir = run_dir / "attempts" / "20260531T164403-942683-f007fa"
    attempt_dir.mkdir(parents=True)
    _write_report(attempt_dir / "final_report.json", overall=None, adjusted=0.488)

    resolved = resolve_best_report(run_dir)

    assert resolved.report is not None
    _, adj = extract_scores(resolved.report)
    assert adj == pytest.approx(0.488)
    assert resolved.picked_from_attempt is True
    assert resolved.attempts_total == 1


def test_resolve_best_report_uses_top_level_when_it_wins(tmp_path: Path):
    run_dir = tmp_path / "prj_top_wins"
    run_dir.mkdir()

    _write_report(run_dir / "final_report.json", overall=0.9, adjusted=0.9)

    attempt_dir = run_dir / "attempts" / "20260531T000000-000000-aaaaaa"
    attempt_dir.mkdir(parents=True)
    _write_report(attempt_dir / "final_report.json", overall=0.1, adjusted=0.1)

    resolved = resolve_best_report(run_dir)
    assert resolved.picked_from_attempt is False
    _, adj = extract_scores(resolved.report)
    assert adj == pytest.approx(0.9)
    assert resolved.attempts_total == 1


def test_resolve_best_report_no_final_report_returns_empty(tmp_path: Path):
    run_dir = tmp_path / "prj_empty"
    run_dir.mkdir()
    resolved = resolve_best_report(run_dir)
    assert resolved.report is None
    assert resolved.report_path is None
    assert resolved.attempts_total == 0


def test_resolve_best_report_skips_corrupt_attempt_json(tmp_path: Path):
    run_dir = tmp_path / "prj_corrupt"
    run_dir.mkdir()

    _write_report(run_dir / "final_report.json", overall=0.3, adjusted=0.3)

    attempt_dir = run_dir / "attempts" / "20260531T000000-000000-corrupt"
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "final_report.json").write_text("{invalid json}")

    resolved = resolve_best_report(run_dir)
    assert resolved.report is not None
    _, adj = extract_scores(resolved.report)
    assert adj == pytest.approx(0.3)
    assert resolved.attempts_total == 1  # corrupt file still counts toward attempts_total


def test_resolve_best_report_multiple_attempts_picks_highest(tmp_path: Path):
    run_dir = tmp_path / "prj_multi"
    run_dir.mkdir()

    _write_report(run_dir / "final_report.json", overall=0.1, adjusted=0.1)

    for i, score in enumerate([0.3, 0.7, 0.5]):
        d = run_dir / "attempts" / f"2026053{i}T000000-000000-aaaaaa"
        d.mkdir(parents=True)
        _write_report(d / "final_report.json", overall=score, adjusted=score)

    resolved = resolve_best_report(run_dir)
    _, adj = extract_scores(resolved.report)
    assert adj == pytest.approx(0.7)
    assert resolved.attempts_total == 3
    assert resolved.picked_from_attempt is True


# ---------------------------------------------------------------------------
# aggregate_leaderboard: surfaces best attempt + new fields
# ---------------------------------------------------------------------------


def test_leaderboard_row_surfaces_best_attempt(tmp_path: Path):
    """aggregate_leaderboard picks the best attempt over the top-level report."""
    run_dir = tmp_path / "prj_09047604e591d969"
    run_dir.mkdir()

    _write_report(run_dir / "final_report.json", overall=0.188, adjusted=0.188)

    attempt_dir = run_dir / "attempts" / "20260531T164403-942683-f007fa"
    attempt_dir.mkdir(parents=True)
    _write_report(attempt_dir / "final_report.json", overall=None, adjusted=0.488)

    (run_dir / "demo_status.json").write_text(json.dumps({
        "projectId": "prj_09047604e591d969",
        "status": "completed",
    }))

    rows = aggregate_leaderboard(tmp_path, order_by="score")
    assert len(rows) == 1
    row = rows[0]
    assert row.project_id == "prj_09047604e591d969"
    assert row.compute_adjusted_score == pytest.approx(0.488)
    assert row.attempts == 1


def test_leaderboard_status_terminalizes(tmp_path: Path):
    """demo_status status='running' + final_report present → row status='completed'."""
    run_dir = tmp_path / "prj_stale_status"
    run_dir.mkdir()

    _write_report(run_dir / "final_report.json", overall=0.5, adjusted=0.5)
    (run_dir / "demo_status.json").write_text(json.dumps({
        "projectId": "prj_stale_status",
        "status": "running",
    }))

    rows = aggregate_leaderboard(tmp_path)
    assert len(rows) == 1
    assert rows[0].status == "completed"


def test_leaderboard_status_preserves_terminal_values(tmp_path: Path):
    """Terminal statuses (failed, stopped, killed, interrupted) are preserved as-is."""
    for status in ("failed", "stopped", "killed", "interrupted", "completed"):
        run_dir = tmp_path / f"prj_{status}"
        run_dir.mkdir(exist_ok=True)
        _write_report(run_dir / "final_report.json", overall=0.5)
        (run_dir / "demo_status.json").write_text(json.dumps({
            "projectId": f"prj_{status}", "status": status,
        }))

    rows = aggregate_leaderboard(tmp_path)
    status_by_id = {r.project_id: r.status for r in rows}
    assert status_by_id["prj_failed"] == "failed"
    assert status_by_id["prj_stopped"] == "stopped"
    assert status_by_id["prj_killed"] == "killed"
    assert status_by_id["prj_interrupted"] == "interrupted"
    assert status_by_id["prj_completed"] == "completed"


def test_leaderboard_status_defaults_to_completed_when_report_exists_no_status_file(tmp_path: Path):
    run_dir = tmp_path / "prj_no_status"
    run_dir.mkdir()
    _write_report(run_dir / "final_report.json", overall=0.5)

    rows = aggregate_leaderboard(tmp_path)
    assert len(rows) == 1
    assert rows[0].status == "completed"


def test_leaderboard_flat_rubric_score_schema(tmp_path: Path):
    """Old flat-schema runs (rubric_score top-level) score correctly."""
    run_dir = tmp_path / "rlm_oauth_smoke_test"
    run_dir.mkdir()
    _write_flat_report(run_dir / "final_report.json", rubric_score=0.33)
    (run_dir / "demo_status.json").write_text(json.dumps({
        "projectId": "rlm_oauth_smoke_test", "status": "completed",
    }))

    rows = aggregate_leaderboard(tmp_path)
    assert len(rows) == 1
    assert rows[0].overall_score == pytest.approx(0.33)
    assert rows[0].compute_adjusted_score == pytest.approx(0.33)


def test_leaderboard_row_schema_includes_status_and_attempts(tmp_path: Path):
    """LeaderboardRow must carry 'status' and 'attempts' fields (FE contract)."""
    run_dir = tmp_path / "prj_schema_check"
    run_dir.mkdir()
    _write_report(run_dir / "final_report.json", overall=0.5)
    (run_dir / "demo_status.json").write_text(json.dumps({
        "projectId": "prj_schema_check", "status": "completed",
    }))

    rows = aggregate_leaderboard(tmp_path)
    dumped = rows[0].model_dump()
    assert "status" in dumped, "LeaderboardRow missing 'status' field"
    assert "attempts" in dumped, "LeaderboardRow missing 'attempts' field"


def test_leaderboard_finished_at_orders_newest_first(tmp_path: Path):
    """order_by=finished_at returns most-recently-completed first; null completed_at last.

    Regression: the original sort was ascending (oldest-first), which buried the
    most recent run at the bottom of a 'recent runs' view.
    """
    def _mk(name: str, completed_at: str | None) -> None:
        d = tmp_path / name
        d.mkdir()
        report: dict = {
            "paper": {"id": "p", "title": "T"},
            "verdict": "partial",
            "rubric": {"overall_score": 0.5, "compute_adjusted_score": 0.5,
                       "meets_target": False, "areas": []},
            "cost": {"llm_usd": 1.0},
            "iterations": 1,
            "mode": "rlm",
            "models": {},
            "started_at": "2026-05-01T00:00:00+00:00",
        }
        if completed_at is not None:
            report["completed_at"] = completed_at
        (d / "final_report.json").write_text(json.dumps(report))
        (d / "demo_status.json").write_text(json.dumps({"projectId": name, "status": "completed"}))

    _mk("prj_old", "2026-05-23T09:00:00+00:00")
    _mk("prj_new", "2026-05-31T16:00:00+00:00")
    _mk("prj_mid", "2026-05-27T12:00:00+00:00")
    _mk("prj_nullts", None)  # no completed_at → must sort last

    rows = aggregate_leaderboard(tmp_path, order_by="finished_at")
    ids = [r.project_id for r in rows]
    assert ids[:3] == ["prj_new", "prj_mid", "prj_old"], ids
    assert ids[-1] == "prj_nullts", ids
