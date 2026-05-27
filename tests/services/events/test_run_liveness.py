"""Tests for PR-π Module B orphan-run liveness sweeps."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backend.services.events.run_liveness import _pid_alive, sweep_orphaned_runs


def _write_status(run_dir: Path, payload: dict) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "demo_status.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _old_iso(seconds: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()


def test_sweep_marks_orphan_with_dead_pid(tmp_path: Path) -> None:
    run_dir = tmp_path / "prj_X"
    _write_status(
        run_dir,
        {
            "projectId": "prj_X",
            "status": "running",
            "pid": 99999,
            "updatedAt": _old_iso(200),
            "startedAt": _old_iso(300),
            "runMode": "rlm",
            "paperId": "paper-x",
        },
    )

    reports = sweep_orphaned_runs(tmp_path, stale_after_s=120)

    assert [r.project_id for r in reports] == ["prj_X"]
    status = json.loads((run_dir / "demo_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "interrupted"
    assert status["error"] == "orphaned_stale_run"
    assert status["degraded"] is True
    report = json.loads((run_dir / "final_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "interrupted"
    assert report["reason"] == "orphaned"


def test_sweep_skips_orphan_with_live_pid(tmp_path: Path) -> None:
    run_dir = tmp_path / "prj_live"
    _write_status(
        run_dir,
        {"projectId": "prj_live", "status": "running", "pid": os.getpid(), "updatedAt": _old_iso(999)},
    )

    reports = sweep_orphaned_runs(tmp_path, stale_after_s=120)

    assert reports == []
    status = json.loads((run_dir / "demo_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "running"


def test_sweep_skips_recently_updated(tmp_path: Path) -> None:
    run_dir = tmp_path / "prj_recent"
    _write_status(
        run_dir,
        {"projectId": "prj_recent", "status": "running", "pid": 99999, "updatedAt": _old_iso(30)},
    )

    reports = sweep_orphaned_runs(tmp_path, stale_after_s=120)

    assert reports == []
    assert not (run_dir / "final_report.json").exists()


def test_sweep_idempotent(tmp_path: Path) -> None:
    run_dir = tmp_path / "prj_once"
    _write_status(
        run_dir,
        {"projectId": "prj_once", "status": "running", "pid": 99999, "updatedAt": _old_iso(500)},
    )

    first = sweep_orphaned_runs(tmp_path, stale_after_s=120)
    final_report = run_dir / "final_report.json"
    first_mtime = final_report.stat().st_mtime_ns
    second = sweep_orphaned_runs(tmp_path, stale_after_s=120)

    assert len(first) == 1
    assert second == []
    assert final_report.stat().st_mtime_ns == first_mtime


def test_sweep_writes_final_report_with_iter_count_from_jsonl(tmp_path: Path) -> None:
    run_dir = tmp_path / "prj_iters"
    state_dir = run_dir / "rlm_state"
    state_dir.mkdir(parents=True)
    (state_dir / "iterations.jsonl").write_text('{"i": 1}\n{"i": 2}\n\n{"i": 3}\n', encoding="utf-8")
    _write_status(
        run_dir,
        {"projectId": "prj_iters", "status": "running", "pid": 99999, "updatedAt": _old_iso(500)},
    )

    sweep_orphaned_runs(tmp_path, stale_after_s=120)

    final_report = json.loads((run_dir / "final_report.json").read_text(encoding="utf-8"))
    assert final_report["iterations"] == 3


def test_pid_alive_signal_0() -> None:
    assert _pid_alive(os.getpid()) is True
    assert _pid_alive(99999) is False


def test_sweep_appends_run_interrupted_and_warning_events(tmp_path: Path) -> None:
    run_dir = tmp_path / "prj_events"
    _write_status(
        run_dir,
        {"projectId": "prj_events", "status": "running", "pid": 99999, "updatedAt": _old_iso(500)},
    )

    sweep_orphaned_runs(tmp_path, stale_after_s=120)

    lines = (run_dir / "dashboard_events.jsonl").read_text(encoding="utf-8").splitlines()
    events = [json.loads(line)["event"] for line in lines]
    assert events == ["run_interrupted", "run_warning"]


def test_sweep_uses_mtime_when_updated_at_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "prj_mtime"
    status_path = _write_status(
        run_dir,
        {"projectId": "prj_mtime", "status": "running", "pid": 99999},
    )
    old = datetime.now(timezone.utc).timestamp() - 500
    os.utime(status_path, (old, old))

    reports = sweep_orphaned_runs(tmp_path, stale_after_s=120)

    assert len(reports) == 1
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["status"] == "interrupted"
