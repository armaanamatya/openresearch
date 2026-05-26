"""Tests for backfill_timing.py — the historical timing.json populator.

Spec: docs/superpowers/specs/2026-05-25-three-source-budget-estimator-design.md §ε.1
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.services.pricing.backfill_timing import backfill


def _make_preserved_run(runs_root: Path, run_id: str, *, started_at: str, completed_at: str) -> Path:
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / ".preserved").write_text(
        json.dumps({"verdict": "partial", "schema_version": 1}),
        encoding="utf-8",
    )
    # minimal final_report.json
    (run_dir / "final_report.json").write_text(
        json.dumps({
            "verdict": "partial",
            "started_at": started_at,
            "completed_at": completed_at,
            "iterations": 3,
            "rubric": {"overall_score": 0.3},
        }),
        encoding="utf-8",
    )
    # minimal demo_status.json (fallback path)
    (run_dir / "demo_status.json").write_text(
        json.dumps({"startedAt": started_at, "completedAt": completed_at}),
        encoding="utf-8",
    )
    return run_dir


def test_backfill_writes_timing_for_preserved_runs(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    _make_preserved_run(
        runs_root, "run1",
        started_at="2026-05-25T10:00:00+00:00",
        completed_at="2026-05-25T11:00:00+00:00",
    )
    _make_preserved_run(
        runs_root, "run2",
        started_at="2026-05-25T12:00:00+00:00",
        completed_at="2026-05-25T14:00:00+00:00",
    )

    skipped, written, errors = backfill(runs_root)
    assert written == 2
    assert skipped == 0
    assert errors == 0

    # Verify the files were actually written
    for run_id in ("run1", "run2"):
        timing_path = runs_root / run_id / "timing.json"
        assert timing_path.exists(), f"timing.json missing for {run_id}"
        data = json.loads(timing_path.read_text(encoding="utf-8"))
        assert data["wall_clock_s"] > 0


def test_backfill_skips_runs_with_existing_timing(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    run_dir = _make_preserved_run(
        runs_root, "run_existing",
        started_at="2026-05-25T10:00:00+00:00",
        completed_at="2026-05-25T11:00:00+00:00",
    )
    # Pre-populate timing.json
    (run_dir / "timing.json").write_text(
        json.dumps({"schema_version": 1, "wall_clock_s": 999.0}),
        encoding="utf-8",
    )

    skipped, written, errors = backfill(runs_root)
    assert skipped == 1
    assert written == 0
    assert errors == 0


def test_backfill_skips_non_preserved_runs(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    # Non-preserved run (no .preserved marker)
    non_preserved = runs_root / "run_not_preserved"
    non_preserved.mkdir()
    (non_preserved / "final_report.json").write_text(
        json.dumps({"verdict": "failed", "started_at": "2026-05-25T10:00:00+00:00",
                    "completed_at": "2026-05-25T10:30:00+00:00", "iterations": 1, "rubric": {}}),
        encoding="utf-8",
    )

    skipped, written, errors = backfill(runs_root)
    assert written == 0
    assert skipped == 0  # Not skipped because of existing timing; just never visited
    assert not (non_preserved / "timing.json").exists()


def test_backfill_returns_zero_counts_for_empty_runs_root(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    skipped, written, errors = backfill(runs_root)
    assert (skipped, written, errors) == (0, 0, 0)


def test_backfill_wall_clock_computed_from_timestamps(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    _make_preserved_run(
        runs_root, "run_timed",
        started_at="2026-05-25T10:00:00+00:00",
        completed_at="2026-05-25T10:30:00+00:00",  # 1800s exactly
    )

    backfill(runs_root)

    data = json.loads((runs_root / "run_timed" / "timing.json").read_text(encoding="utf-8"))
    assert data["wall_clock_s"] == pytest.approx(1800.0, abs=5.0)
