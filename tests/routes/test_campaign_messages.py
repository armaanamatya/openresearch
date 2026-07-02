"""Tests for POST /runs/{project_id}/campaign/messages (campaign steering
channel, F13, spec §10.6). Mirrors tests/routes/test_messages.py's fixture
conventions (isolated OPENRESEARCH_RUNS_ROOT + a fresh settings cache per test).
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient


def _reset_settings_cache():
    import backend.config as _config
    _config._settings_cache = None


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch, tmp_path):
    """Each test gets an isolated runs_root and a fresh settings cache."""
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    monkeypatch.setenv("OPENRESEARCH_RUNS_ROOT", str(runs_root))
    monkeypatch.delenv("OPENRESEARCH_DEMO_SECRET", raising=False)
    _reset_settings_cache()
    yield runs_root
    _reset_settings_cache()


@pytest.fixture
def client(_isolate_settings):
    from backend.app import create_app
    return TestClient(create_app())


@pytest.fixture
def existing_run(_isolate_settings):
    """Create a minimal run directory that the endpoint will find. No
    campaign/ subdirectory — the route must create it on demand."""
    run_dir = _isolate_settings / "prj_test"
    run_dir.mkdir(parents=True)
    (run_dir / "demo_status.json").write_text("{}")
    return run_dir


# ---------------------------------------------------------------------------
# Success path — set_mode
# ---------------------------------------------------------------------------


def test_post_set_mode_returns_202_with_id(client, existing_run):
    r = client.post(
        "/runs/prj_test/campaign/messages",
        json={"op": "set_mode", "mode": "checkpoint"},
    )
    assert r.status_code == 202
    body = r.json()
    assert body["ok"] is True
    assert isinstance(body["id"], str) and body["id"]


def test_post_set_mode_row_carries_id_ts_op_mode(client, existing_run):
    r = client.post(
        "/runs/prj_test/campaign/messages",
        json={"op": "set_mode", "mode": "unattended"},
    )
    returned_id = r.json()["id"]

    lines = (existing_run / "campaign" / "user_messages.jsonl").read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["id"] == returned_id
    assert "ts" in entry
    assert entry["op"] == "set_mode"
    assert entry["mode"] == "unattended"
    assert "content" not in entry


# ---------------------------------------------------------------------------
# Success path — note
# ---------------------------------------------------------------------------


def test_post_note_returns_202_with_id(client, existing_run):
    r = client.post(
        "/runs/prj_test/campaign/messages",
        json={"op": "note", "content": "try width=3 next attempt"},
    )
    assert r.status_code == 202
    body = r.json()
    assert body["ok"] is True
    assert isinstance(body["id"], str) and body["id"]


def test_post_note_row_carries_id_ts_op_content(client, existing_run):
    r = client.post(
        "/runs/prj_test/campaign/messages",
        json={"op": "note", "content": "prefer champion lineage"},
    )
    returned_id = r.json()["id"]

    lines = (existing_run / "campaign" / "user_messages.jsonl").read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["id"] == returned_id
    assert "ts" in entry
    assert entry["op"] == "note"
    assert entry["content"] == "prefer champion lineage"
    assert "mode" not in entry


def test_multiple_messages_accumulate_with_distinct_ids(client, existing_run):
    ops = [
        {"op": "note", "content": "message 0"},
        {"op": "set_mode", "mode": "checkpoint"},
        {"op": "note", "content": "message 2"},
    ]
    ids = []
    for payload in ops:
        r = client.post("/runs/prj_test/campaign/messages", json=payload)
        assert r.status_code == 202
        ids.append(r.json()["id"])

    assert len(set(ids)) == 3, "message ids must be distinct"
    lines = (existing_run / "campaign" / "user_messages.jsonl").read_text().splitlines()
    assert len(lines) == 3


# ---------------------------------------------------------------------------
# Campaign dir auto-creation
# ---------------------------------------------------------------------------


def test_creates_campaign_dir_when_missing(client, existing_run):
    assert not (existing_run / "campaign").exists()
    r = client.post(
        "/runs/prj_test/campaign/messages",
        json={"op": "note", "content": "hello"},
    )
    assert r.status_code == 202
    assert (existing_run / "campaign").is_dir()
    assert (existing_run / "campaign" / "user_messages.jsonl").is_file()


def test_never_touches_campaign_json(client, existing_run):
    """The route only ever writes user_messages.jsonl + the dashboard mirror;
    campaign.json (the CampaignLedger's own state snapshot) is untouched."""
    r = client.post(
        "/runs/prj_test/campaign/messages",
        json={"op": "note", "content": "hello"},
    )
    assert r.status_code == 202
    assert not (existing_run / "campaign" / "campaign.json").exists()


# ---------------------------------------------------------------------------
# Dashboard mirror
# ---------------------------------------------------------------------------


def test_mirrors_dashboard_event(client, existing_run):
    r = client.post(
        "/runs/prj_test/campaign/messages",
        json={"op": "note", "content": "focus on lineage"},
    )
    returned_id = r.json()["id"]

    lines = (existing_run / "dashboard_events.jsonl").read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["event"] == "campaign_user_message"
    assert "timestamp" in entry
    assert entry["op"] == "note"
    assert entry["content"] == "focus on lineage"
    assert entry["id"] == returned_id


def test_dashboard_mirror_does_not_touch_run_level_user_messages(client, existing_run):
    """The campaign channel is isolated from the run-level user_messages.jsonl
    (spec F13's whole point — the top-level file is archived per attempt)."""
    client.post(
        "/runs/prj_test/campaign/messages",
        json={"op": "note", "content": "hi"},
    )
    assert not (existing_run / "user_messages.jsonl").exists()


# ---------------------------------------------------------------------------
# Validation: set_mode without mode -> 400
# ---------------------------------------------------------------------------


def test_set_mode_without_mode_returns_400(client, existing_run):
    r = client.post(
        "/runs/prj_test/campaign/messages",
        json={"op": "set_mode"},
    )
    assert r.status_code == 400


def test_set_mode_without_mode_does_not_write_a_row(client, existing_run):
    client.post("/runs/prj_test/campaign/messages", json={"op": "set_mode"})
    messages_path = existing_run / "campaign" / "user_messages.jsonl"
    assert not messages_path.exists() or messages_path.read_text() == ""


# ---------------------------------------------------------------------------
# Validation: note with blank content -> 400
# ---------------------------------------------------------------------------


def test_note_empty_content_returns_400(client, existing_run):
    r = client.post(
        "/runs/prj_test/campaign/messages",
        json={"op": "note", "content": ""},
    )
    assert r.status_code == 400


def test_note_whitespace_only_content_returns_400(client, existing_run):
    r = client.post(
        "/runs/prj_test/campaign/messages",
        json={"op": "note", "content": "   "},
    )
    assert r.status_code == 400


def test_note_missing_content_field_returns_400(client, existing_run):
    """content defaults to "" when omitted -> still rejected as blank."""
    r = client.post(
        "/runs/prj_test/campaign/messages",
        json={"op": "note"},
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Validation: unknown op -> 422 (pydantic Literal rejection)
# ---------------------------------------------------------------------------


def test_unknown_op_returns_422(client, existing_run):
    r = client.post(
        "/runs/prj_test/campaign/messages",
        json={"op": "bogus_op", "content": "x"},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Unknown project_id -> 404
# ---------------------------------------------------------------------------


def test_unknown_project_returns_404(client):
    r = client.post(
        "/runs/nonexistent_project/campaign/messages",
        json={"op": "note", "content": "hello"},
    )
    assert r.status_code == 404


def test_unknown_project_set_mode_returns_404(client):
    r = client.post(
        "/runs/nonexistent_project/campaign/messages",
        json={"op": "set_mode", "mode": "checkpoint"},
    )
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Demo gate
# ---------------------------------------------------------------------------


def test_rejects_missing_demo_secret(monkeypatch, _isolate_settings, existing_run):
    monkeypatch.setenv("OPENRESEARCH_DEMO_SECRET", "topsecret")
    _reset_settings_cache()
    from backend.app import create_app

    client = TestClient(create_app())
    r = client.post(
        "/runs/prj_test/campaign/messages",
        json={"op": "note", "content": "focus on methods"},
    )
    assert r.status_code == 401

    ok = client.post(
        "/runs/prj_test/campaign/messages",
        json={"op": "note", "content": "focus on methods"},
        headers={"X-Demo-Secret": "topsecret"},
    )
    assert ok.status_code == 202
