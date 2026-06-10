"""Tests for the leaderboard in-memory mtime cache.

TDD spec (task brief):
(a) Create two fake run dirs with final_report.json.
(b) Call aggregate_leaderboard — both rows are read from disk (cache cold).
(c) Modify one final_report.json, advancing its mtime.
(d) Call aggregate_leaderboard again.
(e) Assert:
    - the unchanged run's JSON was NOT re-read (cache hit)
    - the changed run's JSON WAS re-read (cache miss → refresh)

We spy on ``backend.services.events.leaderboard_cache._load_json`` to count
how many times the JSON parser is called without actually preventing the parse
(side_effect passes through to the real function).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

import backend.services.events.leaderboard_cache as _lc
from backend.routes.leaderboard import aggregate_leaderboard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_report(run_dir: Path, score: float) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "final_report.json").write_text(json.dumps({
        "paper": {"id": "p1", "title": "Test Paper"},
        "verdict": "partial",
        "reproduction_summary": "x",
        "baseline_metrics": {},
        "paper_claims": {},
        "rubric": {"overall_score": score, "meets_target": False, "areas": []},
        "improvements": [],
        "primitive_trace": {},
        "cost": {"llm_usd": 1.0, "primitives": 0.0},
        "iterations": 3,
        "mode": "rlm",
        "models": {"planner": "gpt-5", "executor": "claude-sonnet",
                   "verifier": None, "grader": None},
        "started_at": "2026-05-23T04:10:00+00:00",
        "completed_at": "2026-05-23T04:15:00+00:00",
    }))


@pytest.fixture(autouse=True)
def _clear_cache():
    """Ensure the module-level cache is empty before and after each test."""
    _lc.clear()
    yield
    _lc.clear()


# ---------------------------------------------------------------------------
# Core cache behaviour test (the TDD spec from the task brief)
# ---------------------------------------------------------------------------

def test_cache_hit_and_miss_on_mtime_change(tmp_path: Path):
    """Cache serves unchanged rows from memory; stale rows are re-read."""
    runs_root = tmp_path

    # (a) Create two fake run dirs.
    run_a = runs_root / "prj_a"
    run_b = runs_root / "prj_b"
    _write_report(run_a, score=0.42)
    _write_report(run_b, score=0.71)

    # (b) First call — cache is cold; _load_json must be called twice.
    real_load_json = _lc._load_json
    with patch.object(_lc, "_load_json", wraps=real_load_json) as spy:
        rows = aggregate_leaderboard(runs_root)
        assert len(rows) == 2
        first_call_count = spy.call_count
        assert first_call_count == 2, (
            f"Expected 2 JSON loads on cold cache, got {first_call_count}"
        )

    # (c) Modify prj_b's final_report.json — advance its mtime.
    # Sleep briefly to guarantee the OS mtime ticks forward.
    time.sleep(0.02)
    _write_report(run_b, score=0.88)  # different score → confirms refresh

    # (d) Second call with the spy active.
    with patch.object(_lc, "_load_json", wraps=real_load_json) as spy2:
        rows2 = aggregate_leaderboard(runs_root)
        second_call_count = spy2.call_count

    # (e) Assertions:
    # Only prj_b should have been re-read (1 load); prj_a is a cache hit (0 loads).
    assert second_call_count == 1, (
        f"Expected exactly 1 re-read for the stale run, got {second_call_count}"
    )
    # The refreshed row must carry the updated score.
    prj_b_row = next(r for r in rows2 if r.project_id == "prj_b")
    assert prj_b_row.overall_score == pytest.approx(0.88)
    # The cached row is unchanged.
    prj_a_row = next(r for r in rows2 if r.project_id == "prj_a")
    assert prj_a_row.overall_score == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Eviction: deleted run directories are expelled from the cache
# ---------------------------------------------------------------------------

def test_evict_missing_removes_deleted_run(tmp_path: Path):
    """A run removed from disk is expelled from the cache on the next request."""
    runs_root = tmp_path

    run_a = runs_root / "prj_a"
    run_b = runs_root / "prj_b"
    _write_report(run_a, score=0.5)
    _write_report(run_b, score=0.6)

    # Warm the cache.
    rows = aggregate_leaderboard(runs_root)
    assert len(rows) == 2
    assert "prj_a" in _lc._cache
    assert "prj_b" in _lc._cache

    # Delete prj_a from disk.
    import shutil
    shutil.rmtree(run_a)

    # Next aggregate call must evict the stale entry and return only prj_b.
    rows2 = aggregate_leaderboard(runs_root)
    assert len(rows2) == 1
    assert rows2[0].project_id == "prj_b"
    assert "prj_a" not in _lc._cache


# ---------------------------------------------------------------------------
# get_or_load directly
# ---------------------------------------------------------------------------

def test_get_or_load_returns_none_for_missing_file(tmp_path: Path):
    result = _lc.get_or_load("nonexistent", tmp_path / "nope.json")
    assert result is None


def test_get_or_load_caches_on_second_call(tmp_path: Path):
    fr = tmp_path / "final_report.json"
    fr.write_text(json.dumps({"key": "value"}))

    real_load_json = _lc._load_json
    with patch.object(_lc, "_load_json", wraps=real_load_json) as spy:
        d1 = _lc.get_or_load("prj_x", fr)
        d2 = _lc.get_or_load("prj_x", fr)  # same mtime → cache hit

    assert d1 == {"key": "value"}
    assert d2 == {"key": "value"}
    assert spy.call_count == 1, "second call must be served from cache (0 I/O)"


def test_get_or_load_invalidates_on_mtime_change(tmp_path: Path):
    fr = tmp_path / "final_report.json"
    fr.write_text(json.dumps({"v": 1}))

    _lc.get_or_load("prj_x", fr)  # warm cache

    time.sleep(0.02)
    fr.write_text(json.dumps({"v": 2}))  # mtime advances

    data = _lc.get_or_load("prj_x", fr)
    assert data == {"v": 2}


# ---------------------------------------------------------------------------
# clear() isolates tests
# ---------------------------------------------------------------------------

def test_clear_empties_cache(tmp_path: Path):
    fr = tmp_path / "final_report.json"
    fr.write_text(json.dumps({}))
    _lc.get_or_load("prj_x", fr)
    assert len(_lc._cache) == 1

    _lc.clear()
    assert len(_lc._cache) == 0
