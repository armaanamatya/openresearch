"""Tests for POST /runs/{project_id}/messages (chat-steering endpoint)."""

from __future__ import annotations

import json
from pathlib import Path

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
    monkeypatch.setenv("REPROLAB_RUNS_ROOT", str(runs_root))
    monkeypatch.delenv("REPROLAB_DEMO_SECRET", raising=False)
    _reset_settings_cache()
    yield runs_root
    _reset_settings_cache()


@pytest.fixture
def client(_isolate_settings):
    from backend.app import create_app
    return TestClient(create_app())


@pytest.fixture
def existing_run(_isolate_settings):
    """Create a minimal run directory that the endpoint will find."""
    run_dir = _isolate_settings / "prj_test"
    run_dir.mkdir(parents=True)
    (run_dir / "demo_status.json").write_text("{}")
    return run_dir


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------

def test_post_message_valid_returns_202(client, existing_run):
    r = client.post(
        "/runs/prj_test/messages",
        json={"role": "user", "content": "focus on the training recipe section"},
    )
    assert r.status_code == 202
    assert r.json() == {"ok": True}


def test_post_message_appends_to_user_messages_jsonl(client, existing_run):
    client.post(
        "/runs/prj_test/messages",
        json={"role": "user", "content": "try batch size 128"},
    )
    lines = (existing_run / "user_messages.jsonl").read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["role"] == "user"
    assert entry["content"] == "try batch size 128"
    assert "ts" in entry


def test_post_message_appends_to_dashboard_events_jsonl(client, existing_run):
    client.post(
        "/runs/prj_test/messages",
        json={"role": "user", "content": "hello"},
    )
    lines = (existing_run / "dashboard_events.jsonl").read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["event"] == "user_message"
    assert entry["content"] == "hello"


def test_multiple_messages_accumulate(client, existing_run):
    for i in range(3):
        r = client.post(
            "/runs/prj_test/messages",
            json={"role": "user", "content": f"message {i}"},
        )
        assert r.status_code == 202
    lines = (existing_run / "user_messages.jsonl").read_text().splitlines()
    assert len(lines) == 3


# ---------------------------------------------------------------------------
# Validation: empty content → 400
# ---------------------------------------------------------------------------

def test_post_message_empty_content_returns_400(client, existing_run):
    r = client.post(
        "/runs/prj_test/messages",
        json={"role": "user", "content": ""},
    )
    assert r.status_code == 400


def test_post_message_whitespace_only_content_returns_400(client, existing_run):
    r = client.post(
        "/runs/prj_test/messages",
        json={"role": "user", "content": "   "},
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Unknown project_id → 404
# ---------------------------------------------------------------------------

def test_post_message_unknown_project_returns_404(client):
    r = client.post(
        "/runs/nonexistent_project/messages",
        json={"role": "user", "content": "hello"},
    )
    assert r.status_code == 404
