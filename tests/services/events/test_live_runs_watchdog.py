"""Tests for the stderr aclose-deadlock watchdog in live_runs.

The watchdog (`_stderr_watchdog`) runs as an asyncio task tied to the
subprocess lifecycle. These tests exercise it in isolation using a fake
stderr file and a fake pid that never exists (so the watchdog exits cleanly).
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ACLOSE_LINE = (
    "RuntimeError: aclose(): asynchronous generator is already running\n"
)
_NONEXISTENT_PID = 99999999  # never a real pid


def _write_aclose_lines(path: Path, count: int) -> None:
    """Append `count` aclose-pattern lines to the given path."""
    with path.open("a", encoding="utf-8") as fh:
        for _ in range(count):
            fh.write(_ACLOSE_LINE)


# ---------------------------------------------------------------------------
# Core detection test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_watchdog_flags_degraded_after_threshold(tmp_path):
    """3+ aclose lines within 30 s → demo_status.json degraded + run_warning event."""
    from backend.services.events.live_runs import _stderr_watchdog

    # Seed demo_status.json with a minimal running status.
    status_path = tmp_path / "demo_status.json"
    status_path.write_text(
        json.dumps({"projectId": "test-prj", "status": "running"}), encoding="utf-8"
    )

    # Write 3 aclose lines to stderr before launching the watchdog.
    stderr_path = tmp_path / "runner.stderr.log"
    _write_aclose_lines(stderr_path, 3)

    import backend.services.events.live_runs as lr_module

    original_interval = lr_module._WATCHDOG_POLL_INTERVAL
    original_pid_exists = lr_module._pid_exists

    # Poll fast; fake pid alive for the first 3 checks (enough to process the
    # file), then gone so the watchdog exits cleanly.
    call_count = [0]

    def _fake_pid(pid):
        call_count[0] += 1
        return call_count[0] <= 3

    lr_module._WATCHDOG_POLL_INTERVAL = 0.05
    lr_module._pid_exists = _fake_pid
    try:
        task = asyncio.create_task(
            _stderr_watchdog("test-prj", tmp_path, _NONEXISTENT_PID)
        )
        await asyncio.sleep(0.5)
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    finally:
        lr_module._WATCHDOG_POLL_INTERVAL = original_interval
        lr_module._pid_exists = original_pid_exists

    # demo_status.json must have degraded=True.
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status.get("degraded") is True, f"degraded flag missing: {status}"
    assert status.get("degraded_reason") == "SDK aclose loop detected"

    # dashboard_events.jsonl must contain a run_warning event.
    events_path = tmp_path / "dashboard_events.jsonl"
    assert events_path.exists(), "dashboard_events.jsonl not created"
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    warning_events = [e for e in events if e.get("event") == "run_warning"]
    assert len(warning_events) == 1, f"expected 1 run_warning, got {warning_events}"
    ev = warning_events[0]
    assert ev["level"] == "warn"
    assert ev["code"] == "sdk_aclose_loop"
    assert "aclose" in ev["message"]


# ---------------------------------------------------------------------------
# Below-threshold: no flag
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_watchdog_no_flag_below_threshold(tmp_path):
    """2 aclose lines (< 3 threshold) must NOT flag the run."""
    from backend.services.events.live_runs import _stderr_watchdog

    status_path = tmp_path / "demo_status.json"
    status_path.write_text(
        json.dumps({"projectId": "test-prj", "status": "running"}), encoding="utf-8"
    )

    stderr_path = tmp_path / "runner.stderr.log"
    _write_aclose_lines(stderr_path, 2)

    import backend.services.events.live_runs as lr_module

    original_interval = lr_module._WATCHDOG_POLL_INTERVAL
    original_pid_exists = lr_module._pid_exists

    call_count = [0]

    def _fake_pid(pid):
        call_count[0] += 1
        return call_count[0] <= 3

    lr_module._WATCHDOG_POLL_INTERVAL = 0.05
    lr_module._pid_exists = _fake_pid
    try:
        task = asyncio.create_task(
            _stderr_watchdog("test-prj", tmp_path, _NONEXISTENT_PID)
        )
        await asyncio.sleep(0.4)
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    finally:
        lr_module._WATCHDOG_POLL_INTERVAL = original_interval
        lr_module._pid_exists = original_pid_exists

    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert "degraded" not in status, f"should not be degraded: {status}"
    events_path = tmp_path / "dashboard_events.jsonl"
    assert not events_path.exists(), "no events file should be created below threshold"


# ---------------------------------------------------------------------------
# Flag-once: repeated cycles don't append duplicate warnings
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_watchdog_flags_only_once(tmp_path):
    """Multiple poll cycles after crossing threshold must emit exactly 1 warning."""
    from backend.services.events.live_runs import _stderr_watchdog

    status_path = tmp_path / "demo_status.json"
    status_path.write_text(
        json.dumps({"projectId": "test-prj", "status": "running"}), encoding="utf-8"
    )

    # Write 5 aclose lines (well above threshold).
    stderr_path = tmp_path / "runner.stderr.log"
    _write_aclose_lines(stderr_path, 5)

    import backend.services.events.live_runs as lr_module

    original_interval = lr_module._WATCHDOG_POLL_INTERVAL
    # Keep the fake pid alive for the first two poll cycles to allow multiple
    # cycles past the threshold, then let it exit.  We override _pid_exists
    # via monkeypatching the module-level function.
    lr_module._WATCHDOG_POLL_INTERVAL = 0.05

    # Patch _pid_exists to return True for the first 4 calls, then False.
    call_count = [0]
    original_pid_exists = lr_module._pid_exists

    def _fake_pid_exists(pid):
        call_count[0] += 1
        return call_count[0] <= 4

    lr_module._pid_exists = _fake_pid_exists
    try:
        task = asyncio.create_task(
            _stderr_watchdog("test-prj", tmp_path, _NONEXISTENT_PID)
        )
        await asyncio.sleep(0.5)
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    finally:
        lr_module._WATCHDOG_POLL_INTERVAL = original_interval
        lr_module._pid_exists = original_pid_exists

    events_path = tmp_path / "dashboard_events.jsonl"
    assert events_path.exists()
    warnings = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("event") == "run_warning"
    ]
    assert len(warnings) == 1, f"expected exactly 1 run_warning, got {len(warnings)}"
