"""Tests for timing.py — wall-clock capture from preserved run artifacts.

Spec: docs/superpowers/specs/2026-05-25-three-source-budget-estimator-design.md §timing
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from backend.services.pricing.timing import (
    TIMING_SCHEMA_VERSION,
    _extract_primitive_wall_clocks,
    load_preserved_timings,
    write_timing_json,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dashboard_event(
    primitive: str,
    status: str,
    timestamp: str,
) -> str:
    return json.dumps({
        "event": "primitive_call",
        "timestamp": timestamp,
        "primitive": primitive,
        "status": status,
        "args_summary": {},
        "result_summary": None,
        "iteration": None,
        "rubric_delta": None,
    })


def _write_final_report(run_dir: Path, started_at: str, completed_at: str, iterations: int = 4, rubric_score: float = 0.42) -> None:
    report = {
        "verdict": "partial",
        "started_at": started_at,
        "completed_at": completed_at,
        "iterations": iterations,
        "rubric": {"overall_score": rubric_score},
        "mode": "rlm",
        "models": {},
    }
    (run_dir / "final_report.json").write_text(json.dumps(report), encoding="utf-8")


def _write_gpu_plan(run_dir: Path, short_name: str = "rtx4090", gpu_count: int = 1, usd_per_hr: float = 0.34) -> None:
    rlm_state = run_dir / "rlm_state"
    rlm_state.mkdir(exist_ok=True)
    (rlm_state / "gpu_plan.json").write_text(
        json.dumps({"short_name": short_name, "gpu_count": gpu_count, "sku_usd_per_hr": usd_per_hr}),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# _extract_primitive_wall_clocks
# ---------------------------------------------------------------------------

def test_extract_primitive_wall_clocks_basic(tmp_path):
    dashboard = tmp_path / "dashboard_events.jsonl"
    lines = [
        _make_dashboard_event("understand_section", "start", "2026-05-25T10:00:00+00:00"),
        _make_dashboard_event("understand_section", "ok",    "2026-05-25T10:00:12+00:00"),
        _make_dashboard_event("implement_baseline", "start", "2026-05-25T10:00:20+00:00"),
        _make_dashboard_event("implement_baseline", "ok",    "2026-05-25T10:05:25+00:00"),
    ]
    dashboard.write_text("\n".join(lines), encoding="utf-8")

    wc, counts = _extract_primitive_wall_clocks(dashboard)
    assert "understand_section" in wc
    assert wc["understand_section"] == pytest.approx(12.0, abs=0.1)
    assert counts["understand_section"] == 1
    assert "implement_baseline" in wc
    assert wc["implement_baseline"] == pytest.approx(305.0, abs=1.0)
    assert counts["implement_baseline"] == 1


def test_extract_multiple_calls_accumulates(tmp_path):
    dashboard = tmp_path / "dashboard_events.jsonl"
    lines = [
        _make_dashboard_event("understand_section", "start", "2026-05-25T10:00:00+00:00"),
        _make_dashboard_event("understand_section", "ok",    "2026-05-25T10:00:10+00:00"),
        _make_dashboard_event("understand_section", "start", "2026-05-25T10:01:00+00:00"),
        _make_dashboard_event("understand_section", "ok",    "2026-05-25T10:01:25+00:00"),
    ]
    dashboard.write_text("\n".join(lines), encoding="utf-8")

    wc, counts = _extract_primitive_wall_clocks(dashboard)
    assert wc["understand_section"] == pytest.approx(35.0, abs=0.5)
    assert counts["understand_section"] == 2


def test_extract_missing_file_returns_empty(tmp_path):
    wc, counts = _extract_primitive_wall_clocks(tmp_path / "nonexistent.jsonl")
    assert wc == {}
    assert counts == {}


# ---------------------------------------------------------------------------
# write_timing_json
# ---------------------------------------------------------------------------

def test_write_timing_json_produces_correct_shape(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()

    _write_final_report(
        run_dir,
        started_at="2026-05-25T10:00:00+00:00",
        completed_at="2026-05-25T11:03:20+00:00",  # ~3800s = ~1.06h
        iterations=4,
        rubric_score=0.42,
    )
    _write_gpu_plan(run_dir, short_name="rtx4090", gpu_count=1, usd_per_hr=0.34)

    # Write minimal dashboard with one primitive call
    dashboard = run_dir / "dashboard_events.jsonl"
    lines = [
        _make_dashboard_event("understand_section", "start", "2026-05-25T10:05:00+00:00"),
        _make_dashboard_event("understand_section", "ok",    "2026-05-25T10:05:30+00:00"),
        _make_dashboard_event("run_experiment", "start",     "2026-05-25T10:10:00+00:00"),
        _make_dashboard_event("run_experiment", "ok",        "2026-05-25T10:50:00+00:00"),
    ]
    dashboard.write_text("\n".join(lines), encoding="utf-8")

    result = write_timing_json(run_dir)
    assert result is not None
    assert result.exists()

    data = json.loads(result.read_text(encoding="utf-8"))
    assert data["schema_version"] == TIMING_SCHEMA_VERSION
    assert data["wall_clock_s"] == pytest.approx(3800.0, abs=5.0)
    assert data["iterations"] == 4
    assert data["rubric_score"] == pytest.approx(0.42)
    assert data["gpu_type"] == "rtx4090"
    assert data["gpu_count"] == 1
    assert "understand_section" in data["primitive_wall_clock_s"]
    assert data["primitive_wall_clock_s"]["understand_section"] == pytest.approx(30.0, abs=1.0)
    assert "run_experiment" in data["primitive_wall_clock_s"]
    assert data["primitive_wall_clock_s"]["run_experiment"] == pytest.approx(2400.0, abs=5.0)
    # GPU hours from run_experiment wall-clock × gpu_count / 3600
    assert data["gpu_hours"] == pytest.approx(2400.0 / 3600.0, abs=0.05)
    assert "computed_at_utc" in data


def test_write_timing_json_falls_back_to_demo_status(tmp_path):
    """When final_report.json lacks started_at, fall back to demo_status.json."""
    run_dir = tmp_path / "run2"
    run_dir.mkdir()

    # final_report without timing fields
    (run_dir / "final_report.json").write_text(
        json.dumps({"verdict": "partial", "iterations": 2, "rubric": {}}),
        encoding="utf-8",
    )
    _write_gpu_plan(run_dir)

    # demo_status has the timing
    (run_dir / "demo_status.json").write_text(
        json.dumps({
            "startedAt": "2026-05-25T10:00:00+00:00",
            "completedAt": "2026-05-25T10:30:00+00:00",
        }),
        encoding="utf-8",
    )

    result = write_timing_json(run_dir)
    assert result is not None
    data = json.loads(result.read_text(encoding="utf-8"))
    assert data["wall_clock_s"] == pytest.approx(1800.0, abs=5.0)


def test_write_timing_json_atomic(tmp_path, monkeypatch):
    """Verify os.replace is called (atomic write)."""
    run_dir = tmp_path / "run3"
    run_dir.mkdir()
    _write_final_report(run_dir, "2026-05-25T10:00:00+00:00", "2026-05-25T10:10:00+00:00")

    replaces: list[str] = []
    real_replace = os.replace
    monkeypatch.setattr(os, "replace", lambda src, dst: (replaces.append(str(src)), real_replace(src, dst))[1])

    write_timing_json(run_dir)
    assert any(".tmp" in r for r in replaces)


# ---------------------------------------------------------------------------
# load_preserved_timings
# ---------------------------------------------------------------------------

def test_load_preserved_timings(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    # Preserved run WITH timing.json
    r1 = runs_root / "run_preserved"
    r1.mkdir()
    (r1 / ".preserved").write_text(json.dumps({"verdict": "partial"}), encoding="utf-8")
    (r1 / "timing.json").write_text(
        json.dumps({"schema_version": 1, "wall_clock_s": 3600.0, "gpu_type": "rtx4090", "iterations": 2}),
        encoding="utf-8",
    )

    # Preserved run WITHOUT timing.json → skipped
    r2 = runs_root / "run_no_timing"
    r2.mkdir()
    (r2 / ".preserved").write_text(json.dumps({"verdict": "partial"}), encoding="utf-8")

    # Non-preserved run → skipped
    r3 = runs_root / "run_not_preserved"
    r3.mkdir()
    (r3 / "timing.json").write_text(json.dumps({"wall_clock_s": 1000.0}), encoding="utf-8")

    timings = load_preserved_timings(runs_root)
    assert len(timings) == 1
    assert timings[0]["wall_clock_s"] == 3600.0
    assert timings[0]["_run_id"] == "run_preserved"


def test_load_preserved_timings_empty_dir(tmp_path):
    timings = load_preserved_timings(tmp_path / "nonexistent")
    assert timings == []
