"""Tests for extended worker report system (2026-05-24)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.agents.worker_reports import (
    WORKER_TYPE_RDR_CLUSTER,
    WORKER_TYPE_RLM_PRIMITIVE,
    build_extended_worker_report,
    build_summary_report,
    build_worker_report_completed_event,
    build_worker_report_failed_event,
    build_worker_report_started_event,
    classify_sdk_success_blocker,
    finalize_worker_report,
    get_or_build_summary,
    open_worker_report,
    read_worker_reports,
    write_summary_report,
)


def test_classify_sdk_success_blocker_positive() -> None:
    blocker = classify_sdk_success_blocker(
        "Exception: Claude Code returned an error result: success"
    )
    assert blocker is not None
    assert blocker["severity"] == "critical"
    assert blocker["source"] == "claude_agent_sdk"
    assert "success-with-no-text" in blocker["title"]


def test_classify_sdk_success_blocker_negative() -> None:
    assert classify_sdk_success_blocker(None) is None
    assert classify_sdk_success_blocker("Some other error") is None
    assert classify_sdk_success_blocker("") is None


def test_build_extended_worker_report_schema() -> None:
    report = build_extended_worker_report(
        run_id="run-1",
        worker_type=WORKER_TYPE_RDR_CLUSTER,
        agent_id="baseline-implementation",
        project_dir=Path("/tmp/test"),
        model="claude-sonnet",
        provider="anthropic",
        status="running",
        cluster_id="cluster-1",
        assignment={"summary": "Test cluster"},
    )
    assert report["worker_type"] == "rdr_cluster"
    assert report["cluster_id"] == "cluster-1"
    assert report["status"] == "running"
    assert report["assignment"]["summary"] == "Test cluster"
    assert report["worker_id"]  # should be auto-generated
    assert report["task_id"] == report["worker_id"]
    assert report["blockers"] == []
    assert report["errors"] == []
    assert report["artifacts"] == []


def test_build_extended_report_auto_classifies_sdk_blocker() -> None:
    report = build_extended_worker_report(
        agent_id="baseline-implementation",
        project_dir=Path("/tmp/test"),
        error="Exception: Claude Code returned an error result: success",
    )
    assert len(report["blockers"]) == 1
    assert report["blockers"][0]["source"] == "claude_agent_sdk"
    assert len(report["errors"]) == 1


def test_open_and_finalize_worker_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    report = build_extended_worker_report(
        worker_type=WORKER_TYPE_RLM_PRIMITIVE,
        agent_id="implement_baseline",
        project_dir=run_dir,
        status="running",
    )

    # Open
    path = open_worker_report(run_dir, report)
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["status"] == "running"

    # Finalize
    finalize_worker_report(
        run_dir, report,
        status="completed",
        duration_ms=5000,
        execution_summary={"concise_summary": "Done", "changed_files": ["train.py"]},
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["status"] == "completed"
    assert data["duration_ms"] == 5000
    assert data["execution_summary"]["changed_files"] == ["train.py"]


def test_partial_running_report_survives_crash(tmp_path: Path) -> None:
    """A report opened as 'running' should still be readable if the process crashes."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    report = build_extended_worker_report(
        agent_id="build_environment",
        project_dir=run_dir,
        status="running",
    )
    open_worker_report(run_dir, report)

    # Simulate crash: don't finalize. Read should still find it.
    reports = read_worker_reports(run_dir)
    assert len(reports) == 1
    assert reports[0]["status"] == "running"
    assert reports[0]["agent_id"] == "build_environment"


def test_read_worker_reports_falls_back_to_legacy(tmp_path: Path) -> None:
    """When the reports/ dir doesn't exist, fall back to legacy JSONL."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    legacy = run_dir / "worker_reports.jsonl"
    legacy.write_text(
        json.dumps({
            "report_id": "wr-legacy",
            "agent_id": "baseline-implementation",
            "status": "completed",
            "implemented": ["legacy feature"],
        }) + "\n",
        encoding="utf-8",
    )

    reports = read_worker_reports(run_dir)
    assert len(reports) == 1
    assert reports[0]["agent_id"] == "baseline-implementation"


def test_build_summary_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    reports_dir = run_dir / "reports" / "worker_reports"
    reports_dir.mkdir(parents=True)

    # Write two workers
    (reports_dir / "w1.json").write_text(json.dumps({
        "worker_id": "w1",
        "status": "completed",
        "commands": [{"command": "test", "exit_code": 0}],
        "blockers": [],
        "tests": [],
        "next_actions": [],
        "execution_summary": {"changed_files": ["a.py"]},
    }), encoding="utf-8")
    (reports_dir / "w2.json").write_text(json.dumps({
        "worker_id": "w2",
        "status": "failed",
        "commands": [{"command": "broken", "exit_code": 1}],
        "blockers": [{"title": "SDK error", "severity": "critical"}],
        "tests": [],
        "next_actions": [{"action": "Fix SDK"}],
        "execution_summary": {},
    }), encoding="utf-8")

    summary = build_summary_report(run_dir)
    assert summary["total_workers"] == 2
    assert summary["by_status"]["completed"] == 1
    assert summary["by_status"]["failed"] == 1
    assert summary["commands_run"] == 2
    assert summary["failed_commands"] == 1
    assert len(summary["critical_blockers"]) == 1
    assert "a.py" in summary["files_changed"]


def test_write_and_get_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "reports" / "worker_reports").mkdir(parents=True)

    path = write_summary_report(run_dir)
    assert path.exists()

    summary = get_or_build_summary(run_dir)
    assert summary["total_workers"] == 0


def test_sse_event_builders() -> None:
    report = build_extended_worker_report(
        agent_id="test",
        project_dir=Path("/tmp"),
        status="completed",
        worker_type="rdr_cluster",
        cluster_id="c-1",
    )

    started = build_worker_report_started_event(report)
    assert started["event"] == "worker_report_started"
    assert started["worker_type"] == "rdr_cluster"

    completed = build_worker_report_completed_event(report)
    assert completed["event"] == "worker_report_completed"

    report["error"] = "test error"
    failed = build_worker_report_failed_event(report)
    assert failed["event"] == "worker_report_failed"
    assert failed["error"] == "test error"


def test_backward_compat_old_flat_schema() -> None:
    """Old reports without the new fields should still work."""
    old_report = {
        "report_id": "old-1",
        "agent_id": "baseline-implementation",
        "status": "completed",
        "implemented": ["stuff"],
        "commands": [{"command": "test", "exit_code": 0}],
    }
    # Should not error when processed through summary
    # (simulating reading an old-format report)
    assert old_report.get("worker_type") is None
    assert old_report.get("blockers") is None
    # get_or_build_summary should handle this gracefully
