"""2026-05-30: the wall-clock watchdog must be sleep-robust.

The prior threading.Timer waited on a MONOTONIC clock that pauses during macOS
system sleep, so a closed lid stretched a 2h deadline to ~5h. The watchdog now
polls real wall-clock time(); these tests verify it fires when the absolute
deadline has passed, stays armed otherwise, cancels cleanly, and no-ops with no
ceiling.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import backend.agents.rlm.run as run


def _arm(deadline_s, tmp_path):
    return run._arm_watchdog(
        deadline_s,
        project_dir=tmp_path,
        emit=lambda e: None,
        iteration_count=lambda: 0,
    )


def test_none_deadline_returns_none(tmp_path: Path) -> None:
    assert _arm(None, tmp_path) is None


def test_fires_when_deadline_passed(monkeypatch, tmp_path: Path) -> None:
    fired = threading.Event()
    monkeypatch.setattr(run.os, "_exit", lambda code: fired.set())
    monkeypatch.setattr(run, "_WATCHDOG_GRACE_S", 0.0)
    monkeypatch.setattr(run, "_WATCHDOG_POLL_S", 0.05)
    # deadline_s=0 + grace 0 => fire_at == now => first poll fires
    handle = _arm(0.0, tmp_path)
    try:
        assert fired.wait(2.0), "watchdog did not fire within 2s of a passed deadline"
    finally:
        handle.cancel()
    # the honest partial report was written
    assert (tmp_path / "final_report.json").exists()


def test_cancel_prevents_fire(monkeypatch, tmp_path: Path) -> None:
    fired = threading.Event()
    monkeypatch.setattr(run.os, "_exit", lambda code: fired.set())
    monkeypatch.setattr(run, "_WATCHDOG_GRACE_S", 0.0)
    monkeypatch.setattr(run, "_WATCHDOG_POLL_S", 0.05)
    handle = _arm(100.0, tmp_path)  # far future
    handle.cancel()
    assert not fired.wait(0.5), "watchdog fired despite cancel + unreached deadline"
