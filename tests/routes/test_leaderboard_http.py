"""HTTP-level tests for the leaderboard endpoint.

Spec: 2026-05-23-rubric-climb-leaderboard §4.4, §3 #10.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _seed_run(runs_root: Path, project_id: str, score: float, paper_id="p1") -> None:
    d = runs_root / project_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "final_report.json").write_text(json.dumps({
        "paper": {"id": paper_id, "title": paper_id.upper()},
        "verdict": "partial",
        "reproduction_summary": "x",
        "baseline_metrics": {},
        "paper_claims": {},
        "rubric": {"overall_score": score, "meets_target": False, "areas": []},
        "improvements": [],
        "primitive_trace": {},
        "cost": {"llm_usd": 1.0, "primitives": 0.0},
        "iterations": 3,
        "mode": "rlm",
        "models": {"planner": "gpt-5", "executor": "claude-sonnet-4-6",
                   "verifier": None, "grader": None},
        "started_at": "2026-05-23T04:10:00+00:00",
        "completed_at": "2026-05-23T04:15:00+00:00",
    }))
    (d / "demo_status.json").write_text("{}")


def _reset_settings_cache():
    """Drop the cached Settings singleton so the next get_settings() rebuilds it."""
    import backend.config as _config
    _config._settings_cache = None


@pytest.fixture(autouse=True)
def _isolate_settings_cache():
    """Drop the cached Settings around every test in this module so changes to
    REPROLAB_RUNS_ROOT / REPROLAB_DEMO_SECRET take effect, and so we never leak
    a test-specific Settings object into other modules' tests.
    """
    _reset_settings_cache()
    try:
        yield
    finally:
        _reset_settings_cache()


def _fresh_app(monkeypatch, runs_root: Path, *, demo_secret: str = ""):
    monkeypatch.setenv("REPROLAB_RUNS_ROOT", str(runs_root))
    if demo_secret:
        monkeypatch.setenv("REPROLAB_DEMO_SECRET", demo_secret)
    else:
        monkeypatch.delenv("REPROLAB_DEMO_SECRET", raising=False)
    _reset_settings_cache()

    from backend.app import create_app
    return create_app()


@pytest.fixture
def app_with_runs(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    _seed_run(runs_root, "prj_a", 0.42)
    _seed_run(runs_root, "prj_b", 0.71)
    return _fresh_app(monkeypatch, runs_root)


def test_get_leaderboard_returns_ranked_rows(app_with_runs):
    client = TestClient(app_with_runs)
    r = client.get("/leaderboard")
    assert r.status_code == 200
    rows = r.json()
    assert [row["project_id"] for row in rows] == ["prj_b", "prj_a"]


def test_get_leaderboard_filters_by_paper(app_with_runs):
    client = TestClient(app_with_runs)
    r = client.get("/leaderboard", params={"paper": "p1"})
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_get_leaderboard_ignores_demo_secret_gate(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    _seed_run(runs_root, "prj_a", 0.42)

    app = _fresh_app(monkeypatch, runs_root, demo_secret="shh")
    client = TestClient(app)
    # No X-Demo-Secret header — must still return 200 (read-only endpoint).
    r = client.get("/leaderboard")
    assert r.status_code == 200
    assert len(r.json()) == 1
