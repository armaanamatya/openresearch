"""BUG-NEW-041: SIGTERM/SIGHUP termination handler flips demo_status.

The CLI subprocess installs a handler that catches SIGTERM/SIGHUP, writes
status="killed" to runs/<id>/demo_status.json, then converts the signal to
SIGINT so the existing KeyboardInterrupt graceful-exit path also runs.
Without this, `kill <pid>` (the default signal) leaves the run as "running"
forever in the lab UI — the phantom-state failure mode.

These tests cover the handler logic AND the preservation rules in
_mark_demo_status_stopped / _mark_demo_status_failed (both now refuse to
overwrite status="killed").
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.cli import (
    _ACTIVE_PROJECT_ID,
    _install_termination_handlers,
    _mark_demo_status_failed,
    _mark_demo_status_stopped,
    _set_active_project_id,
)


def _write_status(project_dir: Path, **fields) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "demo_status.json").write_text(json.dumps(fields), encoding="utf-8")


def _read_status(project_dir: Path) -> dict:
    return json.loads((project_dir / "demo_status.json").read_text(encoding="utf-8"))


def test_set_active_project_id_updates_module_holder() -> None:
    _set_active_project_id("prj_abc")
    assert _ACTIVE_PROJECT_ID[0] == "prj_abc"
    _set_active_project_id(None)
    assert _ACTIVE_PROJECT_ID[0] is None


@pytest.mark.parametrize("preserved_status", ["killed"])
def test_mark_stopped_preserves_killed(tmp_path: Path, preserved_status: str) -> None:
    """The SIGTERM handler writes status=killed and then converts to SIGINT.
    The KeyboardInterrupt path then calls _mark_demo_status_stopped; that
    helper must leave killed alone — otherwise we lose the signal context."""
    project_id = "prj_test"
    project_dir = tmp_path / project_id
    _write_status(
        project_dir,
        status=preserved_status,
        killedAt="2026-05-29T14:00:00Z",
        killReason="received signal 15",
    )

    _mark_demo_status_stopped(tmp_path, project_id, reason="Pipeline interrupted (Ctrl-C)")

    status = _read_status(project_dir)
    assert status["status"] == preserved_status, "killed must not be overwritten with stopped"
    assert status["killReason"] == "received signal 15"


def test_mark_failed_preserves_killed(tmp_path: Path) -> None:
    project_id = "prj_test"
    project_dir = tmp_path / project_id
    _write_status(project_dir, status="killed", killedAt="2026-05-29T14:00:00Z")

    _mark_demo_status_failed(tmp_path, project_id, reason="boom")

    status = _read_status(project_dir)
    assert status["status"] == "killed"


def test_install_termination_handlers_does_not_raise(tmp_path: Path) -> None:
    """Smoke test — installing handlers must not raise on any supported platform.
    SIGHUP is not available on Windows; the installer must swallow that."""
    _install_termination_handlers(tmp_path)


def test_handler_writes_killed_and_raises_sigint(tmp_path: Path) -> None:
    """Drive the registered SIGTERM handler directly and verify the side effects.

    We can't actually send SIGTERM to the test process without killing the
    test runner. Instead, install the handler, capture it, and call it
    synchronously the way the OS would.
    """
    import signal as _signal

    project_id = "prj_killtest"
    (tmp_path / project_id).mkdir()
    _write_status(
        tmp_path / project_id,
        status="running",
        projectId=project_id,
        startedAt="2026-05-29T13:00:00Z",
    )

    prior = _signal.getsignal(_signal.SIGTERM)
    try:
        _set_active_project_id(project_id)
        _install_termination_handlers(tmp_path)
        handler = _signal.getsignal(_signal.SIGTERM)
        assert callable(handler), "_install_termination_handlers must register a callable"

        with pytest.raises(KeyboardInterrupt):
            handler(_signal.SIGTERM, None)

        status = _read_status(tmp_path / project_id)
        assert status["status"] == "killed"
        assert status["killReason"] == f"received signal {int(_signal.SIGTERM)}"
        assert status["projectId"] == project_id  # preserved
        assert status["startedAt"] == "2026-05-29T13:00:00Z"  # preserved
        assert "killedAt" in status
        assert "completedAt" in status
    finally:
        _signal.signal(_signal.SIGTERM, prior)
        _set_active_project_id(None)


def test_handler_noop_when_no_active_project(tmp_path: Path) -> None:
    """If the holder has no project_id yet (SIGTERM during early ingest before
    register_project), the handler must still convert to SIGINT — it just
    can't flip any status file."""
    import signal as _signal

    prior = _signal.getsignal(_signal.SIGTERM)
    try:
        _set_active_project_id(None)
        _install_termination_handlers(tmp_path)
        handler = _signal.getsignal(_signal.SIGTERM)
        with pytest.raises(KeyboardInterrupt):
            handler(_signal.SIGTERM, None)
    finally:
        _signal.signal(_signal.SIGTERM, prior)
