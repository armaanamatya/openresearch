"""Tests for GET /runs/{project_id}/replay-events (UI timeline replay source)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _write_events(run_dir: Path, events: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "dashboard_events.jsonl").open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


@pytest.fixture
def populated_run(tmp_path: Path) -> tuple[Path, str]:
    runs_root = tmp_path / "runs"
    project_id = "prj_replay_test"
    events = [
        {"event": "repl_iteration", "timestamp": "2026-05-29T20:00:00+00:00", "iteration": 1},
        {"event": "primitive_call", "timestamp": "2026-05-29T20:00:05+00:00", "primitive": "understand_section"},
        {"event": "rubric_score", "timestamp": "2026-05-29T20:01:00+00:00", "data": {"overall_score": 0.2}},
        {"event": "run_complete", "timestamp": "2026-05-29T20:02:00+00:00"},
    ]
    _write_events(runs_root / project_id, events)
    return runs_root, project_id


def _client(monkeypatch: pytest.MonkeyPatch, runs_root: Path) -> TestClient:
    # Patch the route's runs-root resolver directly — hermetic, and avoids polluting
    # the cached get_settings() (whose runs_root would otherwise leak this tmp path
    # into sibling tests, e.g. the reports route test).
    monkeypatch.setattr("backend.routes.replay._runs_root", lambda: runs_root)
    from backend.app import create_app
    return TestClient(create_app())


def test_replay_returns_ordered_events_with_metadata(populated_run, monkeypatch) -> None:
    runs_root, project_id = populated_run
    client = _client(monkeypatch, runs_root)

    resp = client.get(f"/runs/{project_id}/replay-events")
    assert resp.status_code == 200
    data = resp.json()

    assert data["metadata"]["count"] == 4
    assert data["metadata"]["earliestTs"] == "2026-05-29T20:00:00+00:00"
    assert data["metadata"]["latestTs"] == "2026-05-29T20:02:00+00:00"
    # order preserved + events returned verbatim (frontend filters/folds them)
    assert [e["event"] for e in data["events"]] == [
        "repl_iteration", "primitive_call", "rubric_score", "run_complete",
    ]
    assert data["events"][2]["data"]["overall_score"] == 0.2


def test_replay_missing_run_returns_empty_200(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    client = _client(monkeypatch, runs_root)

    resp = client.get("/runs/does_not_exist/replay-events")
    assert resp.status_code == 200
    data = resp.json()
    assert data["events"] == []
    assert data["metadata"] == {"count": 0, "earliestTs": None, "latestTs": None}


def test_replay_tolerates_torn_final_line(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    project_id = "prj_torn"
    run_dir = runs_root / project_id
    run_dir.mkdir(parents=True)
    with (run_dir / "dashboard_events.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({"event": "repl_iteration", "timestamp": "t0"}) + "\n")
        f.write('{"event": "primitive_call", "timestamp": "t1"')  # torn (no closing brace/newline)
    client = _client(monkeypatch, runs_root)

    resp = client.get(f"/runs/{project_id}/replay-events")
    assert resp.status_code == 200
    data = resp.json()
    assert data["metadata"]["count"] == 1  # torn line skipped, not a 500
    assert data["events"][0]["event"] == "repl_iteration"


def test_replay_limit_keeps_recent_tail_in_order(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    project_id = "prj_limit"
    events = [{"event": "primitive_call", "timestamp": f"t{i:03d}", "i": i} for i in range(10)]
    _write_events(runs_root / project_id, events)
    client = _client(monkeypatch, runs_root)

    resp = client.get(f"/runs/{project_id}/replay-events?limit=3")
    assert resp.status_code == 200
    data = resp.json()
    assert data["metadata"]["count"] == 3
    assert [e["i"] for e in data["events"]] == [7, 8, 9]  # most-recent 3, order preserved
