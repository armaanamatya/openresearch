"""PR-ν.3 / P3 — ``_mark_demo_status_failed`` must be the zombie-status guard.

Invariants pinned here:
1. When demo_status.json shows status="running", the helper flips it to
   "failed" with a completedAt timestamp and the supplied reason.
2. When demo_status.json is already in a terminal state (completed / failed
   / stopped), the helper leaves it ALONE — a post-success cleanup crash
   must not relabel a successful run as failed.
3. The helper never raises (best-effort bookkeeping must not mask the
   original exception that triggered it).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.cli import _mark_demo_status_failed


def _write_status(project_dir: Path, **fields) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "demo_status.json").write_text(json.dumps(fields), encoding="utf-8")


def _read_status(project_dir: Path) -> dict:
    return json.loads((project_dir / "demo_status.json").read_text(encoding="utf-8"))


def test_flips_running_to_failed(tmp_path: Path) -> None:
    project_id = "prj_test"
    project_dir = tmp_path / project_id
    _write_status(project_dir, status="running", projectId=project_id, startedAt="2026-05-26T00:00:00Z")

    _mark_demo_status_failed(tmp_path, project_id, reason="boom")

    status = _read_status(project_dir)
    assert status["status"] == "failed"
    assert status["error"] == "boom"
    assert "completedAt" in status
    # Preserved fields survive the rewrite.
    assert status["projectId"] == project_id
    assert status["startedAt"] == "2026-05-26T00:00:00Z"


@pytest.mark.parametrize("existing_status", ["completed", "failed", "stopped"])
def test_skips_when_already_terminal(tmp_path: Path, existing_status: str) -> None:
    """A late crash in cleanup must NOT clobber a successful (or already-failed)
    run's terminal status — _finalize already wrote the canonical state."""
    project_id = "prj_test"
    project_dir = tmp_path / project_id
    _write_status(
        project_dir,
        status=existing_status,
        projectId=project_id,
        completedAt="2026-05-26T00:00:00Z",
        existingError=None,
    )

    _mark_demo_status_failed(tmp_path, project_id, reason="post-finalize crash")

    status = _read_status(project_dir)
    assert status["status"] == existing_status  # untouched
    assert "post-finalize crash" not in str(status.get("error") or "")


def test_silent_on_missing_file(tmp_path: Path) -> None:
    """No status file? Don't crash, just do nothing."""
    project_id = "prj_does_not_exist"
    # No file written.
    _mark_demo_status_failed(tmp_path, project_id, reason="x")
    assert not (tmp_path / project_id / "demo_status.json").exists()


def test_silent_on_corrupt_file(tmp_path: Path) -> None:
    """Corrupt existing JSON? Overwrite with a fresh failed payload."""
    project_id = "prj_corrupt"
    project_dir = tmp_path / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "demo_status.json").write_text("this is { not json", encoding="utf-8")

    _mark_demo_status_failed(tmp_path, project_id, reason="recovered")

    status = _read_status(project_dir)
    assert status["status"] == "failed"
    assert status["error"] == "recovered"


def test_helper_does_not_raise_on_oserror(tmp_path: Path, monkeypatch) -> None:
    """If the disk operation fails, the helper must swallow it — the caller
    is in an exception-handling path and cannot afford a secondary crash."""
    project_id = "prj_test"
    project_dir = tmp_path / project_id
    _write_status(project_dir, status="running")

    # Patch _atomic_write_json to raise OSError — simulates disk full.
    import backend.cli as cli_mod
    def _boom(*_a, **_kw):
        raise OSError("disk full")
    monkeypatch.setattr(cli_mod, "_atomic_write_json", _boom)

    # Must not raise.
    _mark_demo_status_failed(tmp_path, project_id, reason="x")
