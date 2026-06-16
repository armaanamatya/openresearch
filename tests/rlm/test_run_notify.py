"""Tests for the opt-in terminal run-notification hook (run_notify)."""

from __future__ import annotations

import json
from pathlib import Path

from backend.agents.rlm import run_notify


class _FakeResp:
    def __init__(self, status: int = 200) -> None:
        self.status = status

    def __enter__(self) -> "_FakeResp":
        return self

    def __exit__(self, *_a: object) -> bool:
        return False


def _write_report(pdir: Path, **fields: object) -> None:
    (pdir / "final_report.json").write_text(json.dumps(fields))


def test_noop_when_no_webhook(tmp_path, monkeypatch):
    monkeypatch.delenv(run_notify.WEBHOOK_ENV, raising=False)

    def _boom(*_a, **_k):
        raise AssertionError("must not POST when no webhook is configured")

    monkeypatch.setattr(run_notify.urllib.request, "urlopen", _boom)
    assert run_notify.notify_run_terminal(tmp_path) is False


def test_slack_payload_uses_text(tmp_path, monkeypatch):
    _write_report(
        tmp_path,
        run_id="prj_x",
        verdict="reproduced",
        rubric={"overall_score": 0.73, "target_score": 0.70, "meets_target": True},
    )
    captured: dict[str, object] = {}

    def _fake(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())
        return _FakeResp(200)

    monkeypatch.setattr(run_notify.urllib.request, "urlopen", _fake)
    ok = run_notify.notify_run_terminal(
        tmp_path, webhook_url="https://hooks.slack.com/services/T/B/X"
    )
    assert ok is True
    body = captured["body"]
    assert "text" in body and "content" not in body
    assert "prj_x" in body["text"]
    assert "0.730" in body["text"]  # score formatted to 3dp
    assert "✅" in body["text"]


def test_discord_payload_uses_content(tmp_path, monkeypatch):
    _write_report(tmp_path, run_id="prj_y", verdict="failed", rubric={"overall_score": 0.1})
    captured: dict[str, object] = {}

    def _fake(req, timeout=None):
        captured["body"] = json.loads(req.data.decode())
        return _FakeResp(204)

    monkeypatch.setattr(run_notify.urllib.request, "urlopen", _fake)
    ok = run_notify.notify_run_terminal(
        tmp_path, webhook_url="https://discord.com/api/webhooks/1/abc"
    )
    assert ok is True
    body = captured["body"]
    assert "content" in body and "text" not in body
    assert "❌" in body["content"]


def test_failsoft_on_network_error(tmp_path, monkeypatch):
    _write_report(tmp_path, run_id="prj_z", verdict="reproduced", rubric={"overall_score": 0.5})

    def _raise(*_a, **_k):
        raise OSError("connection refused")

    monkeypatch.setattr(run_notify.urllib.request, "urlopen", _raise)
    # Must swallow the error and return False, never propagate.
    assert run_notify.notify_run_terminal(tmp_path, webhook_url="https://hooks.slack.com/x") is False


def test_includes_gcs_prefix_when_bucket_set(tmp_path, monkeypatch):
    _write_report(tmp_path, run_id="prj_g", verdict="reproduced", rubric={"overall_score": 0.6})
    monkeypatch.setenv(run_notify._BUCKET_ENV, "my-bucket")
    captured: dict[str, object] = {}

    def _fake(req, timeout=None):
        captured["body"] = json.loads(req.data.decode())
        return _FakeResp(200)

    monkeypatch.setattr(run_notify.urllib.request, "urlopen", _fake)
    run_notify.notify_run_terminal(tmp_path, webhook_url="https://hooks.slack.com/x")
    assert "gs://my-bucket/runs/prj_g/" in captured["body"]["text"]


def test_env_webhook_is_used(tmp_path, monkeypatch):
    _write_report(tmp_path, run_id="prj_e", verdict="reproduced", rubric={"overall_score": 0.6})
    monkeypatch.setenv(run_notify.WEBHOOK_ENV, "https://hooks.slack.com/env")
    captured: dict[str, object] = {}

    def _fake(req, timeout=None):
        captured["url"] = req.full_url
        return _FakeResp(200)

    monkeypatch.setattr(run_notify.urllib.request, "urlopen", _fake)
    assert run_notify.notify_run_terminal(tmp_path) is True
    assert captured["url"] == "https://hooks.slack.com/env"
