"""Regression: terminal run statuses ``killed`` and ``interrupted`` must
round-trip through ``LiveRunState`` and the ``/runs`` endpoints.

BUG (found 2026-05-30 service validation): ``GET /runs/latest`` returned HTTP
500 on a live dev backend whenever any run dir carried
``demo_status.json::status="killed"``. Root cause: ``RunStatus = Literal[...]``
(``backend/services/events/live_runs.py``) listed only
``queued/running/stopped/completed/failed`` — but BUG-NEW-041's CLI SIGTERM
handler writes ``status="killed"`` and ``run_liveness.sweep_orphaned_runs``
writes ``status="interrupted"``. ``_load_run`` builds ``LiveRunState(**status)``,
so either value raised a pydantic ``ValidationError`` → 500, breaking the UI's
latest-run / auto-resume pointer.

The fix adds both terminal-signal states to ``RunStatus``. The
``{"queued", "running"}`` active-run guards already exclude them (both are
terminal), so the only effect is that they now parse instead of 500-ing.
"""

from __future__ import annotations

import json
import typing
from pathlib import Path

from starlette.testclient import TestClient

from backend.app import create_app
from backend.services.events.live_runs import (
    FileLiveRunService,
    LiveRunState,
    RunStatus,
)


def _write_status(runs_root: Path, project_id: str, status: dict) -> None:
    project_dir = runs_root / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "demo_status.json").write_text(json.dumps(status), encoding="utf-8")


def _client(runs_root: Path) -> TestClient:
    return TestClient(create_app(run_service=FileLiveRunService(runs_root=runs_root)))


def test_runs_latest_tolerates_killed_status(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _write_status(
        runs_root,
        "prj_killed",
        {
            "status": "killed",
            "projectId": "prj_killed",
            "outputDir": str(runs_root / "prj_killed"),
            "runMode": "rlm",
            "updatedAt": "2026-05-30T00:00:00Z",
            "killReason": "received signal 15",
        },
    )

    response = _client(runs_root).get("/runs/latest")

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "killed"


def test_runs_detail_tolerates_killed_status(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _write_status(
        runs_root,
        "prj_killed",
        {
            "status": "killed",
            "projectId": "prj_killed",
            "outputDir": str(runs_root / "prj_killed"),
            "runMode": "rlm",
            "updatedAt": "2026-05-30T00:00:00Z",
        },
    )

    response = _client(runs_root).get("/runs/prj_killed")

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "killed"


def test_runs_detail_tolerates_interrupted_status(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _write_status(
        runs_root,
        "prj_interrupted",
        {
            "status": "interrupted",
            "projectId": "prj_interrupted",
            "outputDir": str(runs_root / "prj_interrupted"),
            "runMode": "rlm",
            "updatedAt": "2026-05-30T00:00:00Z",
            "error": "orphaned_stale_run",
        },
    )

    response = _client(runs_root).get("/runs/prj_interrupted")

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "interrupted"


def test_runs_listing_tolerates_killed_and_interrupted(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _write_status(
        runs_root,
        "prj_killed",
        {"status": "killed", "projectId": "prj_killed", "updatedAt": "2026-05-30T03:00:00Z"},
    )
    _write_status(
        runs_root,
        "prj_interrupted",
        {"status": "interrupted", "projectId": "prj_interrupted", "updatedAt": "2026-05-30T02:00:00Z"},
    )
    _write_status(
        runs_root,
        "prj_done",
        {"status": "completed", "projectId": "prj_done", "updatedAt": "2026-05-30T01:00:00Z"},
    )

    response = _client(runs_root).get("/runs?limit=10")

    assert response.status_code == 200, response.text
    got = {r["projectId"]: r["status"] for r in response.json()}
    assert got == {
        "prj_killed": "killed",
        "prj_interrupted": "interrupted",
        "prj_done": "completed",
    }


def test_runstatus_literal_covers_terminal_signal_states() -> None:
    allowed = set(typing.get_args(RunStatus))
    # The CLI signal handler (BUG-NEW-041) writes "killed"; the orphan-run
    # liveness sweep writes "interrupted". Both must be representable.
    assert {"killed", "interrupted"} <= allowed
    for terminal in ("killed", "interrupted"):
        state = LiveRunState(
            projectId="p", outputDir="/x", runMode="rlm", status=terminal
        )
        assert state.status == terminal
