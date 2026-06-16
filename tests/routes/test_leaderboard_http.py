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
    OPENRESEARCH_RUNS_ROOT / OPENRESEARCH_DEMO_SECRET take effect, and so we never leak
    a test-specific Settings object into other modules' tests.
    """
    _reset_settings_cache()
    try:
        yield
    finally:
        _reset_settings_cache()


def _fresh_app(monkeypatch, runs_root: Path, *, demo_secret: str = ""):
    monkeypatch.setenv("OPENRESEARCH_RUNS_ROOT", str(runs_root))
    if demo_secret:
        monkeypatch.setenv("OPENRESEARCH_DEMO_SECRET", demo_secret)
    else:
        monkeypatch.delenv("OPENRESEARCH_DEMO_SECRET", raising=False)
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


# 2026-05-23: legacy final_report shapes (rubric=list, models=list) used to
# crash the leaderboard with `'list' object has no attribute 'get'` because
# the route ran `data.get("rubric") or {}` — which keeps a non-empty list as
# the truthy value, then calls .get() on it. Pin the defensive coercion.


def _seed_legacy_listrubric_run(runs_root: Path, project_id: str) -> None:
    """Seed a run whose final_report.json has rubric as a list-of-areas instead
    of the {overall_score, meets_target, areas} dict. This was the on-disk
    shape of prj_verify_offline_report and similar legacy fixtures."""
    d = runs_root / project_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "final_report.json").write_text(json.dumps({
        "paper": {"id": "legacy_p", "title": "Legacy Paper"},
        "verdict": "partial",
        "rubric": [
            {"area": "paper_understanding", "score": 1.0, "weight": 0.15},
            {"area": "method_fidelity", "score": 0.3, "weight": 0.35},
        ],
        "cost": [1.0, 0.5],  # also list-shape — another legacy variant
        "models": ["claude-sonnet-4-6"],  # also list-shape
        "iterations": 1,
        "mode": "rlm",
    }))
    (d / "demo_status.json").write_text("{}")


def test_get_leaderboard_survives_legacy_list_shaped_rubric(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    _seed_run(runs_root, "prj_good", 0.5)
    _seed_legacy_listrubric_run(runs_root, "prj_legacy")
    app = _fresh_app(monkeypatch, runs_root)
    client = TestClient(app)
    r = client.get("/leaderboard")
    # MUST be 200 — a single malformed legacy row must not 500 the whole endpoint.
    assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text}"
    rows = r.json()
    # Both rows aggregated; legacy gets None score and is pushed to the bottom.
    ids = [row["project_id"] for row in rows]
    assert "prj_good" in ids
    assert "prj_legacy" in ids
    legacy = next(row for row in rows if row["project_id"] == "prj_legacy")
    # Legacy with non-dict rubric → score is None (cannot extract overall_score)
    assert legacy["overall_score"] is None
    # And the dict-shaped fields default to safe values
    assert legacy["models"] == {
        "planner": None, "executor": None, "verifier": None, "grader": None,
    }
    assert legacy["cost_usd"] is None
