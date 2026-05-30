"""Phase 7g — read-only run-replay / postmortem harness (Gap-3).

Turns the manual /runs audit into a repeatable tool: ingests the per-run JSONL and
flags the wedge shape (dangling sub_rlm_spawned + long event gap) and the FM-004
mismatch (success-ish verdict with empty metrics + no experiment evidence).
"""
from __future__ import annotations

import json

import pytest

from scripts.replay_run import analyze_run


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_flags_dangling_subcall_and_gap(tmp_path):
    run = tmp_path / "prj_wedge"
    run.mkdir()
    _write_jsonl(
        run / "dashboard_events.jsonl",
        [
            {"event": "sub_rlm_spawned", "timestamp": "2026-05-30T00:00:00+00:00", "depth": 1},
            {"event": "sub_rlm_complete", "timestamp": "2026-05-30T00:01:00+00:00", "depth": 1},
            # 76-min gap, then a spawn that never completes:
            {"event": "sub_rlm_spawned", "timestamp": "2026-05-30T01:17:00+00:00", "depth": 1},
        ],
    )
    (run / "demo_status.json").write_text(json.dumps({"status": "killed"}))

    report = analyze_run(run)
    assert report["dangling_subcalls"] == 1
    assert report["max_event_gap_seconds"] >= 76 * 60
    assert report["has_final_report"] is False
    assert any("dangling" in f.lower() for f in report["flags"])


def test_flags_evidence_verdict_mismatch(tmp_path):
    run = tmp_path / "pb_partial"
    run.mkdir()
    _write_jsonl(
        run / "dashboard_events.jsonl",
        [{"event": "run_complete", "timestamp": "2026-05-30T00:00:00+00:00"}],
    )
    (run / "final_report.json").write_text(
        json.dumps({"verdict": "partial", "baseline_metrics": {}})
    )
    # No experiment_runs.jsonl → no evidence.

    report = analyze_run(run)
    assert report["evidence_verdict_mismatch"] is True
    assert any("evidence" in f.lower() for f in report["flags"])


def test_clean_run_has_no_flags(tmp_path):
    run = tmp_path / "ok"
    run.mkdir()
    _write_jsonl(
        run / "dashboard_events.jsonl",
        [
            {"event": "sub_rlm_spawned", "timestamp": "2026-05-30T00:00:00+00:00", "depth": 1},
            {"event": "sub_rlm_complete", "timestamp": "2026-05-30T00:00:30+00:00", "depth": 1},
            {"event": "run_complete", "timestamp": "2026-05-30T00:01:00+00:00"},
        ],
    )
    _write_jsonl(run / "experiment_runs.jsonl", [{"success": True, "metrics": {"acc": 0.4}}])
    (run / "final_report.json").write_text(
        json.dumps({"verdict": "partial", "baseline_metrics": {"acc": 0.4}})
    )

    report = analyze_run(run)
    assert report["dangling_subcalls"] == 0
    assert report["evidence_verdict_mismatch"] is False
    assert report["flags"] == []


def test_missing_run_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        analyze_run(tmp_path / "does_not_exist")
