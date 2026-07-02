"""Tests for the leaderboard campaign column (spec §12).

runs/<id>/campaign/campaign.json -> row["campaign"] = {attempts, terminal, spend}.
Absent/corrupt campaign.json -> the row has NO "campaign" key at all (not
merely a null-valued key) -- byte-identical to a non-campaign project. Covers
both the aggregate_leaderboard() row-building level and the HTTP JSON level,
since the "no key" contract only actually holds end-to-end if extra="allow"
(LeaderboardRow) survives FastAPI's response_model serialization -- this file
is the empirical proof of that, not just an assumption.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.routes.leaderboard import aggregate_leaderboard


def _write_report(run_dir: Path, score: float = 0.5) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "final_report.json").write_text(json.dumps({
        "paper": {"id": "p1", "title": "Test"},
        "verdict": "partial",
        "rubric": {"overall_score": score, "meets_target": False, "areas": []},
        "cost": {"llm_usd": 1.0},
        "iterations": 1,
        "mode": "rlm",
        "models": {},
        "started_at": "2026-07-01T00:00:00+00:00",
        "completed_at": "2026-07-01T01:00:00+00:00",
    }))
    (run_dir / "demo_status.json").write_text(json.dumps({"status": "completed"}))


def _write_campaign_json(
    run_dir: Path, *, next_attempt_n: int, terminal: dict | None, spent: dict
) -> None:
    campaign_dir = run_dir / "campaign"
    campaign_dir.mkdir(parents=True, exist_ok=True)
    (campaign_dir / "campaign.json").write_text(json.dumps({
        "project_id": run_dir.name,
        "paper_ref": "2605.15155",
        "state": "terminal" if terminal is not None else "attempt_loop",
        "next_attempt_n": next_attempt_n,
        "mode": "unattended",
        "driver": "cli",
        "budget": {"max_llm_usd": 10.0, "max_gpu_usd": 10.0, "max_gpu_hours": 5.0},
        "spent": spent,
        "scope_rung": 0,
        "in_flight": None,
        "understanding_sha256": None,
        "rubric_sha256": None,
        "steering_cursor": 0,
        "pending_approval": None,
        "warnings": [],
        "terminal": terminal,
        "created_at": 0.0,
        "updated_at": 0.0,
    }))


@pytest.fixture(autouse=True)
def _clear_leaderboard_cache():
    from backend.services.events import leaderboard_cache
    leaderboard_cache.clear()
    yield
    leaderboard_cache.clear()


# ---------------------------------------------------------------------------
# aggregate_leaderboard() level (row-building path)
# ---------------------------------------------------------------------------


def test_row_without_campaign_json_has_no_campaign_key(tmp_path: Path):
    run_dir = tmp_path / "prj_no_campaign"
    _write_report(run_dir)

    rows = aggregate_leaderboard(tmp_path)
    assert len(rows) == 1
    # extra="allow": the "campaign" kwarg is never passed for a non-campaign
    # row, so the attribute is genuinely UNSET (not merely None) -- direct
    # attribute access would raise AttributeError; getattr is the correct probe.
    assert getattr(rows[0], "campaign", None) is None
    assert "campaign" not in rows[0].model_dump()


def test_row_with_campaign_json_surfaces_attempts_terminal_spend(tmp_path: Path):
    run_dir = tmp_path / "prj_with_campaign"
    _write_report(run_dir)
    spent = {"llm_usd": 3.5, "gpu_usd": 1.2, "gpu_hours": 0.4, "wall_s": 900.0}
    _write_campaign_json(
        run_dir,
        next_attempt_n=4,
        terminal={
            "kind": "REPRODUCED", "rule": "r1", "stop_reason": None,
            "champion_attempt_n": 3, "spent": spent,
        },
        spent=spent,
    )

    rows = aggregate_leaderboard(tmp_path)
    assert len(rows) == 1
    campaign = rows[0].campaign
    assert campaign is not None
    assert campaign["attempts"] == 3
    assert campaign["terminal"] == "REPRODUCED"
    assert campaign["spend"] == spent


def test_row_with_in_progress_campaign_has_null_terminal(tmp_path: Path):
    run_dir = tmp_path / "prj_campaign_running"
    _write_report(run_dir)
    spent = {"llm_usd": 1.0, "gpu_usd": 0.0, "gpu_hours": 0.0, "wall_s": 300.0}
    _write_campaign_json(run_dir, next_attempt_n=2, terminal=None, spent=spent)

    rows = aggregate_leaderboard(tmp_path)
    campaign = rows[0].campaign
    assert campaign["attempts"] == 1
    assert campaign["terminal"] is None
    assert campaign["spend"] == spent


def test_row_with_fresh_campaign_floors_attempts_at_zero(tmp_path: Path):
    """next_attempt_n == 1 (no attempt launched yet) -> attempts == 0, never negative."""
    run_dir = tmp_path / "prj_campaign_fresh"
    _write_report(run_dir)
    zero_spend = {"llm_usd": 0.0, "gpu_usd": 0.0, "gpu_hours": 0.0, "wall_s": 0.0}
    _write_campaign_json(run_dir, next_attempt_n=1, terminal=None, spent=zero_spend)

    rows = aggregate_leaderboard(tmp_path)
    assert rows[0].campaign["attempts"] == 0


def test_row_with_invalid_json_campaign_file_has_no_campaign_key(tmp_path: Path):
    run_dir = tmp_path / "prj_corrupt_campaign"
    _write_report(run_dir)
    campaign_dir = run_dir / "campaign"
    campaign_dir.mkdir(parents=True)
    (campaign_dir / "campaign.json").write_text("{not valid json")

    rows = aggregate_leaderboard(tmp_path)  # must not raise
    assert len(rows) == 1
    assert getattr(rows[0], "campaign", None) is None
    assert "campaign" not in rows[0].model_dump()


def test_row_with_list_shaped_campaign_json_has_no_campaign_key(tmp_path: Path):
    """Valid JSON but the wrong top-level shape (a list, not an object)."""
    run_dir = tmp_path / "prj_listshaped_campaign"
    _write_report(run_dir)
    campaign_dir = run_dir / "campaign"
    campaign_dir.mkdir(parents=True)
    (campaign_dir / "campaign.json").write_text(json.dumps([1, 2, 3]))

    rows = aggregate_leaderboard(tmp_path)
    assert getattr(rows[0], "campaign", None) is None
    assert "campaign" not in rows[0].model_dump()


def test_row_with_campaign_json_missing_next_attempt_n_defaults_to_zero_attempts(tmp_path: Path):
    """A dict that parses but lacks the expected next_attempt_n key entirely
    (not merely wrong-typed) still defaults cleanly rather than raising, and
    IS surfaced (this is a shape gap, not corruption -- the dict itself is
    valid, so best-effort defaults apply rather than treating it as absent)."""
    run_dir = tmp_path / "prj_campaign_missing_field"
    _write_report(run_dir)
    campaign_dir = run_dir / "campaign"
    campaign_dir.mkdir(parents=True)
    (campaign_dir / "campaign.json").write_text(json.dumps({"foo": "bar"}))

    rows = aggregate_leaderboard(tmp_path)  # must not raise
    campaign = rows[0].campaign
    # next_attempt_n defaults to 1 -> attempts == 0; terminal/spend absent -> None.
    assert campaign == {"attempts": 0, "terminal": None, "spend": None}


def test_row_with_non_int_next_attempt_n_has_no_campaign_key(tmp_path: Path):
    run_dir = tmp_path / "prj_campaign_bad_type"
    _write_report(run_dir)
    campaign_dir = run_dir / "campaign"
    campaign_dir.mkdir(parents=True)
    (campaign_dir / "campaign.json").write_text(json.dumps({"next_attempt_n": "not-a-number"}))

    rows = aggregate_leaderboard(tmp_path)  # must not raise
    assert getattr(rows[0], "campaign", None) is None
    assert "campaign" not in rows[0].model_dump()


# ---------------------------------------------------------------------------
# HTTP level -- the actual wire contract
# ---------------------------------------------------------------------------


def _reset_settings_cache():
    import backend.config as _config
    _config._settings_cache = None


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    monkeypatch.setenv("OPENRESEARCH_RUNS_ROOT", str(runs_root))
    monkeypatch.delenv("OPENRESEARCH_DEMO_SECRET", raising=False)
    _reset_settings_cache()
    from backend.app import create_app
    client = TestClient(create_app())
    yield client, runs_root
    _reset_settings_cache()


def test_http_row_without_campaign_has_no_campaign_key(app_client):
    client, runs_root = app_client
    _write_report(runs_root / "prj_a")

    r = client.get("/leaderboard")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 1
    assert "campaign" not in rows[0]


def test_http_row_with_campaign_has_campaign_key(app_client):
    client, runs_root = app_client
    run_dir = runs_root / "prj_b"
    _write_report(run_dir)
    spent = {"llm_usd": 2.0, "gpu_usd": 0.5, "gpu_hours": 0.1, "wall_s": 120.0}
    _write_campaign_json(
        run_dir,
        next_attempt_n=3,
        terminal={
            "kind": "EXHAUSTED", "rule": "budget", "stop_reason": "max_attempts",
            "champion_attempt_n": None, "spent": spent,
        },
        spent=spent,
    )

    r = client.get("/leaderboard")
    assert r.status_code == 200
    rows = r.json()
    row = next(row for row in rows if row["project_id"] == "prj_b")
    assert row["campaign"] == {"attempts": 2, "terminal": "EXHAUSTED", "spend": spent}


def test_http_survives_corrupt_campaign_json_no_500(app_client):
    client, runs_root = app_client
    run_dir = runs_root / "prj_c"
    _write_report(run_dir)
    campaign_dir = run_dir / "campaign"
    campaign_dir.mkdir(parents=True)
    (campaign_dir / "campaign.json").write_text("{broken")

    r = client.get("/leaderboard")
    assert r.status_code == 200
    rows = r.json()
    row = next(row for row in rows if row["project_id"] == "prj_c")
    assert "campaign" not in row


def test_http_mixed_campaign_and_non_campaign_rows_are_independent(app_client):
    """A campaign project and a plain project side by side: the plain one's
    row must be completely unaffected (no key), proving per-row isolation."""
    client, runs_root = app_client
    _write_report(runs_root / "prj_plain")
    run_dir = runs_root / "prj_campaign"
    _write_report(run_dir)
    spent = {"llm_usd": 0.5, "gpu_usd": 0.0, "gpu_hours": 0.0, "wall_s": 60.0}
    _write_campaign_json(run_dir, next_attempt_n=1, terminal=None, spent=spent)

    r = client.get("/leaderboard")
    assert r.status_code == 200
    rows = {row["project_id"]: row for row in r.json()}
    assert "campaign" not in rows["prj_plain"]
    assert rows["prj_campaign"]["campaign"] == {"attempts": 0, "terminal": None, "spend": spent}
