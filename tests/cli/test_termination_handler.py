"""STAB-3 / BUG-NEW-041: SIGTERM/SIGHUP handling.

The handler flips the active run's demo_status.json to status="killed" with a
killReason, and that terminal state is not overwritten by the graceful
stopped/failed helpers.
"""
import json
import signal

import pytest

from backend import cli


def _write_status(path, status):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": status, "projectId": "p"}), encoding="utf-8")


def test_mark_killed_writes_killed_with_reason(tmp_path):
    cli._mark_demo_status_killed(tmp_path, "p", kill_reason="Process terminated (SIGTERM)")
    data = json.loads((tmp_path / "p" / "demo_status.json").read_text())
    assert data["status"] == "killed"
    assert data["killReason"] == "Process terminated (SIGTERM)"
    assert "completedAt" in data


def test_mark_killed_does_not_overwrite_completed(tmp_path):
    _write_status(tmp_path / "p" / "demo_status.json", "completed")
    cli._mark_demo_status_killed(tmp_path, "p")
    assert json.loads((tmp_path / "p" / "demo_status.json").read_text())["status"] == "completed"


def test_stopped_and_failed_do_not_overwrite_killed(tmp_path):
    _write_status(tmp_path / "p" / "demo_status.json", "killed")
    cli._mark_demo_status_stopped(tmp_path, "p")
    assert json.loads((tmp_path / "p" / "demo_status.json").read_text())["status"] == "killed"
    cli._mark_demo_status_failed(tmp_path, "p")
    assert json.loads((tmp_path / "p" / "demo_status.json").read_text())["status"] == "killed"


def test_install_handlers_registers_sigterm_and_marks_active(tmp_path, monkeypatch):
    # point the active run at tmp, install handlers, restore SIGTERM after.
    prev = signal.getsignal(signal.SIGTERM)
    try:
        cli._set_active_project_id("p", tmp_path)
        _write_status(tmp_path / "p" / "demo_status.json", "running")
        cli._install_termination_handlers()
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        # invoke the handler directly but stub raise_signal so the test process
        # doesn't actually receive SIGINT.
        monkeypatch.setattr(signal, "raise_signal", lambda s: None)
        handler(signal.SIGTERM, None)
        data = json.loads((tmp_path / "p" / "demo_status.json").read_text())
        assert data["status"] == "killed"
        assert "SIGTERM" in data["killReason"]
    finally:
        signal.signal(signal.SIGTERM, prev)
        cli._set_active_project_id(None, None)
