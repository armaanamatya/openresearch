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


# ---------------------------------------------------------------------------
# Audit 2026-06-09: cross pid-namespace / cross-user false-orphan fixes
# ---------------------------------------------------------------------------


def test_pid_alive_treats_eperm_as_alive(monkeypatch) -> None:
    """EPERM from os.kill(pid, 0) means the process EXISTS but belongs to
    another user — it must read as alive, not dead (a backend server running
    under a different OS user must not sweep a live CLI run)."""

    def _kill_eperm(_pid, _sig):
        raise PermissionError("Operation not permitted")

    monkeypatch.setattr("backend.services.events.run_liveness.os.kill", _kill_eperm)
    assert _pid_alive(12345) is True


def test_sweep_skips_pid_from_other_host(tmp_path: Path) -> None:
    """A pid minted on another host / pid namespace (e.g. a host-launched CLI
    run seen by the containerized backend via the bind-mounted runs/) cannot be
    probed with os.kill — liveness is UNKNOWN, so the sweep must skip it (same
    posture as the absent-pid case)."""
    run_dir = tmp_path / "prj_other_host"
    _write_status(
        run_dir,
        {
            "projectId": "prj_other_host",
            "status": "running",
            "pid": 99999,  # dead in OUR namespace — irrelevant, it isn't ours
            "pidHost": "some-other-machine",
            "updatedAt": _old_iso(500),
        },
    )

    reports = sweep_orphaned_runs(tmp_path, stale_after_s=120)

    assert reports == []
    status = json.loads((run_dir / "demo_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "running"
    assert not (run_dir / "final_report.json").exists()


def test_sweep_still_marks_orphan_when_pid_host_matches(tmp_path: Path) -> None:
    """Same-host snapshots (pidHost == our hostname) keep full sweep behavior."""
    import socket

    run_dir = tmp_path / "prj_same_host"
    _write_status(
        run_dir,
        {
            "projectId": "prj_same_host",
            "status": "running",
            "pid": 99999,
            "pidHost": socket.gethostname(),
            "updatedAt": _old_iso(500),
        },
    )

    reports = sweep_orphaned_runs(tmp_path, stale_after_s=120)

    assert [r.project_id for r in reports] == ["prj_same_host"]


# ---------------------------------------------------------------------------
# Salvage: stop_reason, verdict, final_report.md (2026-06-20)
# ---------------------------------------------------------------------------


def _write_rubric_score_event(run_dir: Path, score: float) -> None:
    """Append a rubric_score SSE event to dashboard_events.jsonl."""
    path = run_dir / "dashboard_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "rubric_score", "score": score, "target": 1.0}) + "\n")


def test_salvage_writes_stop_reason_and_verdict_with_score(tmp_path: Path) -> None:
    """Orphaned run with a rubric_score event → salvaged report has stop_reason+verdict=partial."""
    run_dir = tmp_path / "prj_salvage"
    _write_rubric_score_event(run_dir, 0.42)
    _write_status(
        run_dir,
        {"projectId": "prj_salvage", "status": "running", "pid": 99999, "updatedAt": _old_iso(500)},
    )

    sweep_orphaned_runs(tmp_path, stale_after_s=120)

    report = json.loads((run_dir / "final_report.json").read_text(encoding="utf-8"))
    assert report["stop_reason"] == {"kind": "orphaned", "detail": "run process disappeared (host suspend / SIGKILL / OOM)"}
    assert report["verdict"] == "partial"
    assert abs(report["rubric_score"] - 0.42) < 1e-6


def test_salvage_verdict_failed_when_no_score(tmp_path: Path) -> None:
    """Orphaned run with no rubric evidence → verdict=failed, stop_reason present."""
    run_dir = tmp_path / "prj_no_score"
    _write_status(
        run_dir,
        {"projectId": "prj_no_score", "status": "running", "pid": 99999, "updatedAt": _old_iso(500)},
    )

    sweep_orphaned_runs(tmp_path, stale_after_s=120)

    assert (run_dir / "demo_status.json").exists()
    report = json.loads((run_dir / "final_report.json").read_text(encoding="utf-8"))
    assert report["verdict"] == "failed"
    assert report["stop_reason"]["kind"] == "orphaned"
    assert report["rubric_score"] == 0.0


def test_salvage_does_not_overwrite_existing_final_report(tmp_path: Path) -> None:
    """A run that already has final_report.json must not be overwritten."""
    run_dir = tmp_path / "prj_has_report"
    existing = {"verdict": "reproduced", "rubric_score": 0.99, "custom": True}
    (run_dir).mkdir(parents=True)
    (run_dir / "final_report.json").write_text(json.dumps(existing), encoding="utf-8")
    _write_status(
        run_dir,
        {"projectId": "prj_has_report", "status": "running", "pid": 99999, "updatedAt": _old_iso(500)},
    )

    sweep_orphaned_runs(tmp_path, stale_after_s=120)

    report = json.loads((run_dir / "final_report.json").read_text(encoding="utf-8"))
    assert report["custom"] is True
    assert report["verdict"] == "reproduced"


def test_salvage_writes_md_companion(tmp_path: Path) -> None:
    """Orphaned run → final_report.md is written alongside final_report.json."""
    run_dir = tmp_path / "prj_md"
    _write_rubric_score_event(run_dir, 0.55)
    _write_status(
        run_dir,
        {"projectId": "prj_md", "status": "running", "pid": 99999, "updatedAt": _old_iso(500)},
    )

    sweep_orphaned_runs(tmp_path, stale_after_s=120)

    md_path = run_dir / "final_report.md"
    assert md_path.exists(), "final_report.md should be written for orphaned run"
    md_text = md_path.read_text(encoding="utf-8")
    assert "prj_md" in md_text
    assert "partial" in md_text
    assert "orphan" in md_text.lower()


def test_salvage_md_not_written_when_report_already_exists(tmp_path: Path) -> None:
    """When final_report.json exists, neither it nor final_report.md is overwritten."""
    run_dir = tmp_path / "prj_md_skip"
    (run_dir).mkdir(parents=True)
    (run_dir / "final_report.json").write_text(json.dumps({"verdict": "reproduced"}), encoding="utf-8")
    _write_status(
        run_dir,
        {"projectId": "prj_md_skip", "status": "running", "pid": 99999, "updatedAt": _old_iso(500)},
    )

    sweep_orphaned_runs(tmp_path, stale_after_s=120)

    assert not (run_dir / "final_report.md").exists()


def test_salvage_no_crash_on_no_evidence(tmp_path: Path) -> None:
    """Orphaned run with no evidence files → marked interrupted without crash."""
    run_dir = tmp_path / "prj_empty"
    _write_status(
        run_dir,
        {"projectId": "prj_empty", "status": "running", "pid": 99999, "updatedAt": _old_iso(500)},
    )

    reports = sweep_orphaned_runs(tmp_path, stale_after_s=120)

    assert len(reports) == 1
    assert reports[0].project_id == "prj_empty"
    status = json.loads((run_dir / "demo_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "interrupted"
    report = json.loads((run_dir / "final_report.json").read_text(encoding="utf-8"))
    assert report["rubric_score"] == 0.0
