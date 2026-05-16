"""Tests for the GET /runs listing endpoint backed by FileLiveRunService.list_runs."""

from __future__ import annotations

import json
from pathlib import Path

from starlette.testclient import TestClient

from backend.app import create_app
from backend.services.events.live_runs import FileLiveRunService


def _write_status(runs_root: Path, project_id: str, status: dict) -> None:
    project_dir = runs_root / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "demo_status.json").write_text(json.dumps(status), encoding="utf-8")


def test_get_runs_returns_recents_sorted_by_updated_at(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _write_status(runs_root, "prj_old", {
        "status": "completed", "projectId": "prj_old",
        "updatedAt": "2026-01-01T00:00:00Z",
    })
    _write_status(runs_root, "prj_mid", {
        "status": "running", "projectId": "prj_mid",
        "updatedAt": "2026-02-01T00:00:00Z",
    })
    _write_status(runs_root, "prj_new", {
        "status": "completed", "projectId": "prj_new",
        "updatedAt": "2026-03-01T00:00:00Z",
    })

    service = FileLiveRunService(runs_root=runs_root)
    client = TestClient(create_app(run_service=service))

    response = client.get("/runs?limit=2")

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert [r["projectId"] for r in body] == ["prj_new", "prj_mid"]


def test_get_runs_filters_by_status(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _write_status(runs_root, "prj_a", {
        "status": "completed", "projectId": "prj_a",
        "updatedAt": "2026-03-01T00:00:00Z",
    })
    _write_status(runs_root, "prj_b", {
        "status": "failed", "projectId": "prj_b",
        "updatedAt": "2026-03-02T00:00:00Z",
    })

    service = FileLiveRunService(runs_root=runs_root)
    client = TestClient(create_app(run_service=service))

    response = client.get("/runs?limit=10&status=completed")

    assert response.status_code == 200
    assert [r["projectId"] for r in response.json()] == ["prj_a"]


def test_get_runs_returns_empty_list_when_runs_root_missing(tmp_path: Path) -> None:
    # runs_root does not exist; service should tolerate it.
    service = FileLiveRunService(runs_root=tmp_path / "does-not-exist")
    client = TestClient(create_app(run_service=service))

    response = client.get("/runs")

    assert response.status_code == 200
    assert response.json() == []


def test_get_runs_skips_directories_without_demo_status(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    (runs_root / "prj_no_status").mkdir(parents=True)
    _write_status(runs_root, "prj_with_status", {
        "status": "completed", "projectId": "prj_with_status",
        "updatedAt": "2026-01-01T00:00:00Z",
    })

    service = FileLiveRunService(runs_root=runs_root)
    client = TestClient(create_app(run_service=service))

    response = client.get("/runs")

    assert response.status_code == 200
    assert [r["projectId"] for r in response.json()] == ["prj_with_status"]


def test_get_runs_q_substring_match_on_source_label(tmp_path: Path) -> None:
    """q narrows results to runs whose sourceLabel contains the substring,
    case-insensitively. Runs without a sourceLabel are excluded."""
    runs_root = tmp_path / "runs"
    _write_status(runs_root, "prj_a", {
        "status": "completed", "projectId": "prj_a",
        "sourceLabel": "Diffusion Policy",
        "updatedAt": "2026-03-01T00:00:00Z",
    })
    _write_status(runs_root, "prj_b", {
        "status": "completed", "projectId": "prj_b",
        "sourceLabel": "ACT Transformer",
        "updatedAt": "2026-03-02T00:00:00Z",
    })
    _write_status(runs_root, "prj_c", {
        # no sourceLabel → must be excluded when q is provided
        "status": "completed", "projectId": "prj_c",
        "updatedAt": "2026-03-03T00:00:00Z",
    })

    service = FileLiveRunService(runs_root=runs_root)
    client = TestClient(create_app(run_service=service))

    response = client.get("/runs?limit=10&q=diffusion")

    assert response.status_code == 200
    assert [r["projectId"] for r in response.json()] == ["prj_a"]


def test_get_runs_q_is_case_insensitive(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _write_status(runs_root, "prj_a", {
        "status": "completed", "projectId": "prj_a",
        "sourceLabel": "ReproLab PPO Reproducibility Demo",
        "updatedAt": "2026-03-01T00:00:00Z",
    })

    service = FileLiveRunService(runs_root=runs_root)
    client = TestClient(create_app(run_service=service))

    response = client.get("/runs?q=PPO")
    assert response.status_code == 200
    assert [r["projectId"] for r in response.json()] == ["prj_a"]

    response = client.get("/runs?q=ppo")
    assert response.status_code == 200
    assert [r["projectId"] for r in response.json()] == ["prj_a"]


def test_get_runs_order_by_completed_at(tmp_path: Path) -> None:
    """order_by=completed_at sorts by completedAt desc; runs without a
    completedAt sort last (timestamp 0)."""
    runs_root = tmp_path / "runs"
    _write_status(runs_root, "prj_recent", {
        "status": "completed", "projectId": "prj_recent",
        "updatedAt": "2026-03-01T00:00:00Z",
        "completedAt": "2026-03-01T00:00:00Z",
    })
    _write_status(runs_root, "prj_older", {
        "status": "completed", "projectId": "prj_older",
        "updatedAt": "2026-03-02T00:00:00Z",  # newer updated_at
        "completedAt": "2026-01-15T00:00:00Z",  # but older completed_at
    })
    _write_status(runs_root, "prj_running", {
        # still running → no completedAt → sorts last under completed_at
        "status": "running", "projectId": "prj_running",
        "updatedAt": "2026-03-03T00:00:00Z",
    })

    service = FileLiveRunService(runs_root=runs_root)
    client = TestClient(create_app(run_service=service))

    response = client.get("/runs?limit=10&order_by=completed_at")

    assert response.status_code == 200
    assert [r["projectId"] for r in response.json()] == [
        "prj_recent", "prj_older", "prj_running",
    ]


def test_get_runs_order_by_defaults_to_updated_at(tmp_path: Path) -> None:
    """Unknown order_by value falls back to updated_at (no crash)."""
    runs_root = tmp_path / "runs"
    _write_status(runs_root, "prj_a", {
        "status": "completed", "projectId": "prj_a",
        "updatedAt": "2026-01-01T00:00:00Z",
    })
    _write_status(runs_root, "prj_b", {
        "status": "completed", "projectId": "prj_b",
        "updatedAt": "2026-03-01T00:00:00Z",
    })

    service = FileLiveRunService(runs_root=runs_root)
    client = TestClient(create_app(run_service=service))

    response = client.get("/runs?order_by=garbage")
    assert response.status_code == 200
    assert [r["projectId"] for r in response.json()] == ["prj_b", "prj_a"]


def test_get_runs_combines_status_q_and_order_by(tmp_path: Path) -> None:
    """Filters compose: status + q narrow the set, order_by sorts the result."""
    runs_root = tmp_path / "runs"
    _write_status(runs_root, "prj_match_recent", {
        "status": "completed", "projectId": "prj_match_recent",
        "sourceLabel": "Diffusion Policy v2",
        "updatedAt": "2026-03-01T00:00:00Z",
        "completedAt": "2026-03-01T00:00:00Z",
    })
    _write_status(runs_root, "prj_match_older", {
        "status": "completed", "projectId": "prj_match_older",
        "sourceLabel": "Diffusion Policy v1",
        "updatedAt": "2026-02-01T00:00:00Z",
        "completedAt": "2026-02-01T00:00:00Z",
    })
    _write_status(runs_root, "prj_wrong_status", {
        "status": "failed", "projectId": "prj_wrong_status",
        "sourceLabel": "Diffusion Policy failed",
        "updatedAt": "2026-04-01T00:00:00Z",
    })
    _write_status(runs_root, "prj_wrong_label", {
        "status": "completed", "projectId": "prj_wrong_label",
        "sourceLabel": "ACT Transformer",
        "updatedAt": "2026-04-02T00:00:00Z",
    })

    service = FileLiveRunService(runs_root=runs_root)
    client = TestClient(create_app(run_service=service))

    response = client.get(
        "/runs?limit=10&status=completed&q=diffusion&order_by=completed_at"
    )

    assert response.status_code == 200
    assert [r["projectId"] for r in response.json()] == [
        "prj_match_recent", "prj_match_older",
    ]
