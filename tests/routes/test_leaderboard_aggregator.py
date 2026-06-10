"""Tests for the leaderboard aggregator.

Spec: 2026-05-23-rubric-climb-leaderboard §4.4–§4.5.
"""

import json
from pathlib import Path

import pytest

from backend.routes.leaderboard import (
    aggregate_leaderboard,
)


@pytest.fixture(autouse=True)
def _clear_leaderboard_cache():
    # leaderboard_cache._cache is a process-global keyed by project_id with
    # mtime-based invalidation. These tests reuse project_ids ("a"/"b") across
    # different tmp_paths; with coarse filesystem mtime the staleness check can
    # collide and return a STALE cross-test row, so a different test fails on
    # each run (timing-dependent flakiness). clear() before+after each test —
    # using the cache's own test-isolation hook — makes the suite deterministic.
    from backend.services.events import leaderboard_cache

    leaderboard_cache.clear()
    yield
    leaderboard_cache.clear()


def _write_run(
    runs_root: Path,
    project_id: str,
    *,
    paper_id: str,
    paper_title: str,
    overall_score: float,
    cost_usd: float,
    iterations: int,
    planner: str | None,
    executor: str | None,
    mode: str = "rlm",
    started_at: str = "2026-05-23T04:10:00+00:00",
    completed_at: str = "2026-05-23T04:15:00+00:00",
    meets_target: bool = False,
    degraded: bool = False,
    verdict: str = "partial",
) -> None:
    run_dir = runs_root / project_id
    run_dir.mkdir(parents=True, exist_ok=True)
    final = {
        "paper": {"id": paper_id, "title": paper_title},
        "verdict": verdict,
        "reproduction_summary": "x",
        "baseline_metrics": {},
        "paper_claims": {},
        "rubric": {
            "overall_score": overall_score,
            "meets_target": meets_target,
            "areas": [],
        },
        "improvements": [],
        "primitive_trace": {},
        "cost": {"llm_usd": cost_usd, "primitives": 0.0},
        "iterations": iterations,
        "primitive_provider": "real",
        "degraded": degraded,
        "mode": mode,
        "models": {
            "planner": planner,
            "executor": executor,
            "verifier": None,
            "grader": None,
        },
        "started_at": started_at,
        "completed_at": completed_at,
    }
    (run_dir / "final_report.json").write_text(json.dumps(final))
    (run_dir / "demo_status.json").write_text(json.dumps({
        "projectId": project_id, "status": "completed",
    }))


def test_aggregate_returns_empty_when_runs_dir_is_empty(tmp_path: Path):
    rows = aggregate_leaderboard(tmp_path)
    assert rows == []


def test_aggregate_returns_one_row_per_completed_run(tmp_path: Path):
    _write_run(
        tmp_path, "prj_a",
        paper_id="p1", paper_title="Paper One",
        overall_score=0.42, cost_usd=1.23, iterations=8,
        planner="gpt-5", executor="claude-sonnet-4-6",
    )
    _write_run(
        tmp_path, "prj_b",
        paper_id="p1", paper_title="Paper One",
        overall_score=0.71, cost_usd=2.50, iterations=12,
        planner="claude-opus-4-7", executor="claude-sonnet-4-6",
    )

    rows = aggregate_leaderboard(tmp_path)
    assert len(rows) == 2
    assert rows[0].project_id == "prj_b"
    assert rows[0].overall_score == pytest.approx(0.71)
    assert rows[1].project_id == "prj_a"


def test_aggregate_skips_runs_with_no_final_report(tmp_path: Path):
    _write_run(
        tmp_path, "prj_complete",
        paper_id="p1", paper_title="P", overall_score=0.5,
        cost_usd=1.0, iterations=5, planner=None, executor=None,
    )
    in_flight = tmp_path / "prj_in_flight"
    in_flight.mkdir()
    (in_flight / "demo_status.json").write_text("{}")

    rows = aggregate_leaderboard(tmp_path)
    assert [r.project_id for r in rows] == ["prj_complete"]


def test_aggregate_back_compat_legacy_run_missing_new_fields(tmp_path: Path):
    legacy = tmp_path / "prj_legacy"
    legacy.mkdir()
    (legacy / "final_report.json").write_text(json.dumps({
        "paper": {"id": "p1", "title": "Legacy"},
        "verdict": "failed",
        "reproduction_summary": "x",
        "baseline_metrics": {},
        "paper_claims": {},
        "rubric": {"overall_score": 0.0, "meets_target": False, "areas": []},
        "improvements": [],
        "primitive_trace": {},
        "cost": {"llm_usd": 0.0, "primitives": 0.0},
        "iterations": 0,
    }))
    (legacy / "demo_status.json").write_text("{}")

    rows = aggregate_leaderboard(tmp_path)
    assert len(rows) == 1
    assert rows[0].mode == "rlm"
    assert rows[0].models.planner is None
    assert rows[0].started_at is None


def test_aggregate_filter_by_paper(tmp_path: Path):
    _write_run(tmp_path, "a", paper_id="p1", paper_title="P1",
               overall_score=0.5, cost_usd=1, iterations=1,
               planner=None, executor=None)
    _write_run(tmp_path, "b", paper_id="p2", paper_title="P2",
               overall_score=0.6, cost_usd=1, iterations=1,
               planner=None, executor=None)

    rows = aggregate_leaderboard(tmp_path, paper="p1")
    assert [r.paper_id for r in rows] == ["p1"]


def test_aggregate_filter_by_mode(tmp_path: Path):
    _write_run(tmp_path, "a", paper_id="p1", paper_title="P",
               overall_score=0.5, cost_usd=1, iterations=1,
               planner=None, executor=None, mode="rlm")
    _write_run(tmp_path, "b", paper_id="p1", paper_title="P",
               overall_score=0.6, cost_usd=1, iterations=1,
               planner=None, executor=None, mode="rdr")

    rows = aggregate_leaderboard(tmp_path, mode="rdr")
    assert [r.project_id for r in rows] == ["b"]


def test_aggregate_order_by_cost(tmp_path: Path):
    _write_run(tmp_path, "a", paper_id="p1", paper_title="P",
               overall_score=0.5, cost_usd=3.0, iterations=1,
               planner=None, executor=None)
    _write_run(tmp_path, "b", paper_id="p1", paper_title="P",
               overall_score=0.6, cost_usd=1.0, iterations=1,
               planner=None, executor=None)

    rows = aggregate_leaderboard(tmp_path, order_by="cost")
    assert [r.project_id for r in rows] == ["b", "a"]


def test_leaderboard_row_schema_is_pinned(tmp_path: Path):
    """A LeaderboardRow MUST carry exactly the 14 documented fields.

    Future field additions / removals fail this test loudly — preventing
    silent wire-shape drift between frontend and backend.
    """
    _write_run(
        tmp_path, "prj_schema",
        paper_id="p1", paper_title="P",
        overall_score=0.5, cost_usd=1.0, iterations=4,
        planner="gpt-5", executor="claude-sonnet-4-6",
    )
    rows = aggregate_leaderboard(tmp_path)
    assert len(rows) == 1
    dumped = rows[0].model_dump()
    expected_keys = {
        "project_id", "paper_id", "paper_title", "title", "mode", "models",
        "overall_score", "compute_adjusted_score", "execution_mode",
        "meets_target", "degraded",
        "cost_usd", "iterations", "wall_clock_s",
        "sandbox", "started_at", "completed_at", "verdict",
        # β5: attempt-aware fields
        "status", "attempts",
    }
    assert set(dumped.keys()) == expected_keys, (
        f"LeaderboardRow shape drifted. "
        f"missing={expected_keys - set(dumped.keys())}, "
        f"unexpected={set(dumped.keys()) - expected_keys}"
    )
    # models nested object must carry exactly the 4 role keys.
    assert set(dumped["models"].keys()) == {"planner", "executor", "verifier", "grader"}


def test_aggregate_propagates_null_score_not_zero(tmp_path: Path):
    """A run that exited before scoring (rubric.overall_score is null) must
    surface as overall_score=None on the leaderboard row — not coerced to 0.0
    — so the UI can tell 'not scored' apart from 'scored zero'."""
    d = tmp_path / "prj_unscored"
    d.mkdir()
    (d / "final_report.json").write_text(json.dumps({
        "paper": {"id": "p1", "title": "P"},
        "verdict": "failed",
        "reproduction_summary": "x",
        "baseline_metrics": {},
        "paper_claims": {},
        "rubric": {"overall_score": None, "meets_target": None, "areas": []},
        "improvements": [],
        "primitive_trace": {},
        "cost": {"llm_usd": None, "primitives": 0.0},
        "iterations": 0,
    }))
    (d / "demo_status.json").write_text("{}")

    rows = aggregate_leaderboard(tmp_path)
    assert len(rows) == 1
    assert rows[0].overall_score is None
    # Sort key must put None scores at the bottom — adding a scored run
    # should keep the scored row first.
    _write_run(tmp_path, "scored", paper_id="p1", paper_title="P",
               overall_score=0.42, cost_usd=1.0, iterations=1,
               planner=None, executor=None)
    rows = aggregate_leaderboard(tmp_path, order_by="score")
    assert rows[0].project_id == "scored"
    assert rows[1].project_id == "prj_unscored"


def test_aggregate_computes_wall_clock_from_timestamps(tmp_path: Path):
    _write_run(tmp_path, "a", paper_id="p1", paper_title="P",
               overall_score=0.5, cost_usd=1, iterations=1,
               planner=None, executor=None,
               started_at="2026-05-23T04:10:00+00:00",
               completed_at="2026-05-23T04:15:00+00:00")
    rows = aggregate_leaderboard(tmp_path)
    assert rows[0].wall_clock_s == 300.0
