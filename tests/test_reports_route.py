"""Tests for GET /runs/{project_id}/reports route."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def populated_run(tmp_path: Path) -> tuple[Path, str]:
    """Create a run dir with worker reports."""
    runs_root = tmp_path / "runs"
    project_id = "prj_reports_test"
    run_dir = runs_root / project_id
    reports_dir = run_dir / "reports" / "worker_reports"
    reports_dir.mkdir(parents=True)

    (run_dir / "demo_status.json").write_text(json.dumps({
        "projectId": project_id,
        "status": "completed",
    }), encoding="utf-8")

    (reports_dir / "w1.json").write_text(json.dumps({
        "worker_id": "w1",
        "worker_type": "rdr_cluster",
        "agent_id": "baseline-implementation",
        "status": "completed",
        "blockers": [],
        "commands": [{"command": "test", "exit_code": 0}],
        "tests": [],
        "next_actions": [],
        "execution_summary": {},
    }), encoding="utf-8")

    (reports_dir / "w2.json").write_text(json.dumps({
        "worker_id": "w2",
        "worker_type": "rdr_cluster",
        "agent_id": "baseline-implementation",
        "status": "failed",
        "error": "Exception: Claude Code returned an error result: success",
        "blockers": [{"title": "SDK error", "severity": "critical", "source": "claude_agent_sdk"}],
        "commands": [],
        "tests": [],
        "next_actions": [],
        "execution_summary": {},
    }), encoding="utf-8")

    return runs_root, project_id


def test_reports_route_populated(populated_run: tuple[Path, str], monkeypatch: pytest.MonkeyPatch) -> None:
    runs_root, project_id = populated_run
    monkeypatch.setenv("OPENRESEARCH_RUNS_ROOT", str(runs_root))

    from backend.app import create_app
    app = create_app()
    client = TestClient(app)

    resp = client.get(f"/runs/{project_id}/reports")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["workers"]) == 2
    assert data["summary"]["total_workers"] == 2


def test_reports_route_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    monkeypatch.setenv("OPENRESEARCH_RUNS_ROOT", str(runs_root))

    from backend.app import create_app
    app = create_app()
    client = TestClient(app)

    resp = client.get("/runs/nonexistent/reports")
    assert resp.status_code == 200
    data = resp.json()
    assert data["workers"] == []
    assert data["summary"] == {}
