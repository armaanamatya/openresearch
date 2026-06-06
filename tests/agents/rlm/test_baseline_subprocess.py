"""Isolated-subprocess baseline runner (2026-05-30).

The baseline SDK call runs in a fresh `multiprocessing` child so the
claude-agent-sdk `aclose()` async-gen race crashes only the child and never
poisons the reproduction process (the 2026-05-30 SDAR run failed 8/8 iterations
in-process). Deterministic — the child body is exercised in-process with
run_with_sdk mocked; no real SDK, no real spawn.
"""
from __future__ import annotations

import queue
from types import SimpleNamespace

import backend.agents.baseline_implementation as bi
from backend.agents.rlm.baseline_runner import run_baseline_in_child
from backend.agents.rlm.primitives import _baseline_subprocess_enabled


def test_toggle_default_off(monkeypatch):
    # Default OFF (opt-in): the in-process path is the default the unit suite mocks.
    monkeypatch.delenv("OPENRESEARCH_BASELINE_SUBPROCESS", raising=False)
    assert _baseline_subprocess_enabled() is False


def test_toggle_off(monkeypatch):
    for v in ("0", "false", "no", "FALSE", "No"):
        monkeypatch.setenv("OPENRESEARCH_BASELINE_SUBPROCESS", v)
        assert _baseline_subprocess_enabled() is False


def test_toggle_on_for_other_values(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_BASELINE_SUBPROCESS", "1")
    assert _baseline_subprocess_enabled() is True


def test_child_ok_puts_three_fields_and_heartbeats(monkeypatch, tmp_path):
    async def fake_run_with_sdk(*_a, on_event=None, **_k):
        if on_event:
            on_event()  # touches the heartbeat file
        return SimpleNamespace(
            commands_to_run=["python train.py"], diff_summary="d", assumptions_applied=["A1"],
        )
    monkeypatch.setattr(bi, "run_with_sdk", fake_run_with_sdk)
    q: queue.Queue = queue.Queue()
    hb = tmp_path / "hb"
    run_baseline_in_child(q, str(hb), "prj", str(tmp_path), {"a": 1}, {"b": 2}, None, None, {})
    out = q.get_nowait()
    assert out == {
        "ok": True,
        "commands_to_run": ["python train.py"],
        "diff_summary": "d",
        "assumptions_applied": ["A1"],
    }
    assert hb.exists()  # liveness heartbeat was touched


def test_child_failure_is_reported_not_raised(monkeypatch, tmp_path):
    # The aclose race (or any failure) must be captured on the queue, never raised
    # out of the child — so the parent classifies it as repairable and retries.
    async def boom(*_a, **_k):
        raise RuntimeError("aclose(): asynchronous generator is already running")
    monkeypatch.setattr(bi, "run_with_sdk", boom)
    q: queue.Queue = queue.Queue()
    run_baseline_in_child(q, str(tmp_path / "hb"), "prj", str(tmp_path), 1, 2, 3, 4, {})
    out = q.get_nowait()
    assert out["ok"] is False
    assert "aclose()" in out["error"]
    assert "traceback" in out


def test_child_missing_fields_default_safely(monkeypatch, tmp_path):
    async def sparse(*_a, on_event=None, **_k):
        return SimpleNamespace()  # no commands_to_run / diff_summary / assumptions_applied
    monkeypatch.setattr(bi, "run_with_sdk", sparse)
    q: queue.Queue = queue.Queue()
    run_baseline_in_child(q, str(tmp_path / "hb"), "prj", str(tmp_path), 1, 2, 3, 4, {})
    out = q.get_nowait()
    assert out == {"ok": True, "commands_to_run": [], "diff_summary": "", "assumptions_applied": []}
