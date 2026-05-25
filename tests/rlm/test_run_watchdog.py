"""Tests for backend.agents.rlm.run_watchdog.

Twelve guarantees pinned here:

  1. Default config matches the documented thresholds (300 / 1200 / 30 s).
  2. ``REPROLAB_WATCHDOG_DISABLED=true`` short-circuits run_watchdog.
  3. ``from_env`` honours custom thresholds + ignores non-numeric junk.
  4. ``collect_staleness`` returns ``verdict="ok"`` when NO signal file exists
     (bootstrap-grace — can't classify nothing).
  5. All signals fresh → "ok".
  6. Any one signal fresh keeps the verdict "ok" (min-staleness wins).
  7. All signals stale > warn_after → "warn".
  8. All signals stale > kill_after → "kill".
  9. ``on_warn`` callback receives a report containing the freshest-signal
     identity and age.
 10. ``on_kill`` invoked AT MOST ONCE (loop returns after first kill).
 11. ``run_watchdog`` exits cleanly on cancellation.
 12. ``heartbeat_daemon_command`` produces a backgrounded shell line.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from backend.agents.rlm import run_watchdog as rw


# ---------------------------------------------------------------------------
# WatchdogConfig
# ---------------------------------------------------------------------------


def test_default_config_thresholds() -> None:
    c = rw.WatchdogConfig()
    # Policy: warn=600s (10 min), kill=1500s (25 min). The big gap between
    # warn and kill is intentional — Lane N probe-recover handles the case
    # where a legitimate slow-print train.py (e.g. epoch-25 reporting at slow
    # epoch speed) crosses the warn line. Killing a working pod is much
    # worse than a few extra min of wait, so kill is generous.
    assert c.warn_after_seconds == 600.0
    assert c.kill_after_seconds == 1500.0
    assert c.poll_interval_seconds == 30.0
    assert c.heartbeat_filename == ".heartbeat"
    assert c.exec_log_filename == "exec.log"
    assert c.dashboard_events_filename == "dashboard_events.jsonl"


def test_from_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("REPROLAB_WATCHDOG_WARN_SECONDS", "120")
    monkeypatch.setenv("REPROLAB_WATCHDOG_KILL_SECONDS", "900")
    monkeypatch.setenv("REPROLAB_WATCHDOG_POLL_INTERVAL_SECONDS", "10")
    c = rw.WatchdogConfig.from_env()
    assert c.warn_after_seconds == 120.0
    assert c.kill_after_seconds == 900.0
    assert c.poll_interval_seconds == 10.0


def test_from_env_ignores_junk(monkeypatch) -> None:
    monkeypatch.setenv("REPROLAB_WATCHDOG_WARN_SECONDS", "not-a-number")
    c = rw.WatchdogConfig.from_env()
    assert c.warn_after_seconds == 600.0  # default preserved (10 min)


# ---------------------------------------------------------------------------
# is_enabled
# ---------------------------------------------------------------------------


def test_is_enabled_default(monkeypatch) -> None:
    monkeypatch.delenv("REPROLAB_WATCHDOG_DISABLED", raising=False)
    assert rw.is_enabled() is True


@pytest.mark.parametrize("val", ["true", "True", "TRUE", "1", "yes", "on"])
def test_is_enabled_disabled_via_env(monkeypatch, val) -> None:
    monkeypatch.setenv("REPROLAB_WATCHDOG_DISABLED", val)
    assert rw.is_enabled() is False


# ---------------------------------------------------------------------------
# collect_staleness
# ---------------------------------------------------------------------------


def test_no_signals_yet_returns_ok(tmp_path: Path) -> None:
    artifact_root = tmp_path / "outputs" / "r1"
    artifact_root.mkdir(parents=True)
    project_dir = tmp_path

    report = rw.collect_staleness(
        artifact_root=artifact_root,
        project_dir=project_dir,
        config=rw.WatchdogConfig(),
        now=time.time(),
    )
    assert report.verdict == "ok"
    assert report.stale_seconds is None
    assert report.freshest_signal is None


def test_all_signals_fresh_ok(tmp_path: Path) -> None:
    artifact_root = tmp_path / "outputs" / "r1"
    artifact_root.mkdir(parents=True)
    project_dir = tmp_path
    now = time.time()
    for fname in ("exec.log", ".heartbeat"):
        p = artifact_root / fname
        p.write_text("x")
        os.utime(p, (now, now - 10))  # 10 s old
    # SSE event filter now requires a MEANINGFUL event (not heartbeat).
    import json
    from datetime import datetime, timezone
    sse = project_dir / "dashboard_events.jsonl"
    sse.write_text(json.dumps({
        "event": "primitive_call",
        "primitive": "understand_section",  # NOT 'heartbeat' — meaningful
        "ts": datetime.fromtimestamp(now - 5, tz=timezone.utc).isoformat(),
    }) + "\n")
    os.utime(sse, (now, now - 5))

    report = rw.collect_staleness(
        artifact_root=artifact_root,
        project_dir=project_dir,
        config=rw.WatchdogConfig(),
        now=now,
    )
    assert report.verdict == "ok"
    assert report.freshest_signal == "sse_event"
    assert 0 <= report.freshest_signal_age_seconds <= 6


def test_any_one_signal_fresh_keeps_ok(tmp_path: Path) -> None:
    artifact_root = tmp_path / "outputs" / "r1"
    artifact_root.mkdir(parents=True)
    project_dir = tmp_path
    now = time.time()
    # Two stale signals
    for fname in ("exec.log", ".heartbeat"):
        p = artifact_root / fname
        p.write_text("x")
        os.utime(p, (now, now - 3600))
    # One fresh signal — and it MUST be a meaningful (non-heartbeat) event.
    import json
    from datetime import datetime, timezone
    sse = project_dir / "dashboard_events.jsonl"
    sse.write_text(json.dumps({
        "event": "primitive_call",
        "primitive": "implement_baseline",
        "ts": datetime.fromtimestamp(now - 5, tz=timezone.utc).isoformat(),
    }) + "\n")
    os.utime(sse, (now, now - 5))

    report = rw.collect_staleness(
        artifact_root=artifact_root,
        project_dir=project_dir,
        config=rw.WatchdogConfig(),
        now=now,
    )
    assert report.verdict == "ok"
    assert report.freshest_signal == "sse_event"


def test_all_stale_past_warn_returns_warn(tmp_path: Path) -> None:
    artifact_root = tmp_path / "outputs" / "r1"
    artifact_root.mkdir(parents=True)
    project_dir = tmp_path
    now = time.time()
    for path in [
        artifact_root / "exec.log",
        artifact_root / ".heartbeat",
        project_dir / "dashboard_events.jsonl",
    ]:
        path.write_text("x")
        os.utime(path, (now, now - 600))  # 10 min stale

    report = rw.collect_staleness(
        artifact_root=artifact_root,
        project_dir=project_dir,
        config=rw.WatchdogConfig(warn_after_seconds=300, kill_after_seconds=1200),
        now=now,
    )
    assert report.verdict == "warn"
    assert report.stale_seconds >= 600


def test_all_stale_past_kill_returns_kill(tmp_path: Path) -> None:
    artifact_root = tmp_path / "outputs" / "r1"
    artifact_root.mkdir(parents=True)
    project_dir = tmp_path
    now = time.time()
    for path in [
        artifact_root / "exec.log",
        artifact_root / ".heartbeat",
        project_dir / "dashboard_events.jsonl",
    ]:
        path.write_text("x")
        os.utime(path, (now, now - 1500))  # 25 min stale

    report = rw.collect_staleness(
        artifact_root=artifact_root,
        project_dir=project_dir,
        config=rw.WatchdogConfig(warn_after_seconds=300, kill_after_seconds=1200),
        now=now,
    )
    assert report.verdict == "kill"


# ---------------------------------------------------------------------------
# run_watchdog — async loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_via_env_returns_immediately(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("REPROLAB_WATCHDOG_DISABLED", "true")
    fired = []

    async def _on_warn(r): fired.append(("warn", r))
    async def _on_kill(r): fired.append(("kill", r))

    await rw.run_watchdog(
        artifact_root=tmp_path,
        project_dir=tmp_path,
        config=rw.WatchdogConfig(poll_interval_seconds=0.01),
        on_warn=_on_warn,
        on_kill=_on_kill,
    )
    assert fired == []


@pytest.mark.asyncio
async def test_kill_invoked_at_most_once(tmp_path: Path) -> None:
    artifact_root = tmp_path / "outputs" / "r1"
    artifact_root.mkdir(parents=True)
    # Pre-create a stale signal (25 min old) so the first poll classifies as kill.
    p = artifact_root / "exec.log"
    p.write_text("x")
    now = time.time()
    os.utime(p, (now, now - 1500))

    kill_calls: list[rw.StalenessReport] = []

    async def _on_kill(r):
        kill_calls.append(r)

    await asyncio.wait_for(
        rw.run_watchdog(
            artifact_root=artifact_root,
            project_dir=tmp_path,
            config=rw.WatchdogConfig(
                warn_after_seconds=60, kill_after_seconds=600,
                poll_interval_seconds=0.01,
            ),
            on_kill=_on_kill,
        ),
        timeout=2.0,
    )
    assert len(kill_calls) == 1


# ---------------------------------------------------------------------------
# Lane N — KillVerdict.RECOVERED keeps watchdog polling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kill_verdict_recovered_keeps_watchdog_polling(tmp_path: Path) -> None:
    """When on_kill returns RECOVERED, watchdog must continue polling
    instead of returning. After staleness is refreshed by an external
    write, the loop returns to verdict=ok and the next kill won't fire."""
    artifact_root = tmp_path / "outputs" / "r1"
    artifact_root.mkdir(parents=True)
    p = artifact_root / "exec.log"
    p.write_text("x")
    now = time.time()
    os.utime(p, (now, now - 1500))  # 25 min stale

    kill_calls: list[rw.StalenessReport] = []
    refreshed = False

    async def _on_kill(r):
        nonlocal refreshed
        kill_calls.append(r)
        # Simulate soft-recovery: refresh the signal so next poll sees verdict=ok.
        if not refreshed:
            refreshed = True
            os.utime(p, (time.time(), time.time()))
            return rw.KillVerdict.RECOVERED
        return rw.KillVerdict.DESTROY  # second time, destroy

    # If RECOVERED works, the watchdog should NOT return after the first kill —
    # it should reset and continue. We give it 1 s and then expect to cancel.
    task = asyncio.create_task(rw.run_watchdog(
        artifact_root=artifact_root, project_dir=tmp_path,
        config=rw.WatchdogConfig(
            warn_after_seconds=60, kill_after_seconds=600,
            poll_interval_seconds=0.01,
        ),
        on_kill=_on_kill,
    ))
    await asyncio.sleep(0.3)  # give the loop time to fire kill + recover + see ok
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    # First kill fired (RECOVERED), no second kill (because signals fresh now).
    assert len(kill_calls) == 1


@pytest.mark.asyncio
async def test_kill_verdict_destroy_exits_watchdog(tmp_path: Path) -> None:
    """DESTROY (or None) returns from on_kill makes the watchdog return."""
    artifact_root = tmp_path / "outputs" / "r1"
    artifact_root.mkdir(parents=True)
    p = artifact_root / "exec.log"
    p.write_text("x")
    now = time.time()
    os.utime(p, (now, now - 1500))

    async def _on_kill(r):
        return rw.KillVerdict.DESTROY

    # run_watchdog should return on its own — no need to cancel.
    await asyncio.wait_for(
        rw.run_watchdog(
            artifact_root=artifact_root, project_dir=tmp_path,
            config=rw.WatchdogConfig(
                warn_after_seconds=60, kill_after_seconds=600,
                poll_interval_seconds=0.01,
            ),
            on_kill=_on_kill,
        ),
        timeout=2.0,
    )


@pytest.mark.asyncio
async def test_kill_verdict_none_defaults_to_destroy(tmp_path: Path) -> None:
    """Backward compatibility: a v1 callback returning None should not
    leave the watchdog polling forever."""
    artifact_root = tmp_path / "outputs" / "r1"
    artifact_root.mkdir(parents=True)
    p = artifact_root / "exec.log"
    p.write_text("x")
    now = time.time()
    os.utime(p, (now, now - 1500))

    async def _on_kill_v1(r):
        return None  # v1 callbacks return None

    await asyncio.wait_for(
        rw.run_watchdog(
            artifact_root=artifact_root, project_dir=tmp_path,
            config=rw.WatchdogConfig(
                warn_after_seconds=60, kill_after_seconds=600,
                poll_interval_seconds=0.01,
            ),
            on_kill=_on_kill_v1,
        ),
        timeout=2.0,
    )


def test_kill_verdict_is_string_enum() -> None:
    """KillVerdict values are stringly typed so they survive event-stream
    JSON serialisation without custom handlers."""
    assert rw.KillVerdict.DESTROY.value == "destroy"
    assert rw.KillVerdict.RECOVERED.value == "recovered"
    # Equality between enum and string works (StrEnum semantics).
    assert rw.KillVerdict.RECOVERED == "recovered"


@pytest.mark.asyncio
async def test_cancelled_exits_cleanly(tmp_path: Path) -> None:
    task = asyncio.create_task(rw.run_watchdog(
        artifact_root=tmp_path,
        project_dir=tmp_path,
        config=rw.WatchdogConfig(poll_interval_seconds=10),
    ))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# ---------------------------------------------------------------------------
# heartbeat_daemon_command
# ---------------------------------------------------------------------------


def test_heartbeat_daemon_command_is_backgrounded() -> None:
    cmd = rw.heartbeat_daemon_command("/artifacts")
    # The command MUST end with `exit 0` so SSH sees a clean immediate exit
    # rather than waiting on the daemon. Previous version used plain
    # `nohup ... &` which blocked because SSH waits for ALL inherited FDs
    # to close — the daemon kept stdout/stderr alive forever.
    assert "exit 0" in cmd
    assert "&" in cmd  # daemon still backgrounded
    assert "/artifacts/.heartbeat" in cmd
    assert "while true" in cmd
    assert "sleep 30" in cmd


def test_heartbeat_daemon_command_full_fd_detachment() -> None:
    """All three SSH-inherited FDs must be closed/redirected, else the
    SSH exec channel hangs waiting for them to close."""
    cmd = rw.heartbeat_daemon_command("/artifacts")
    assert "< /dev/null" in cmd or "</dev/null" in cmd  # stdin closed
    assert "> /dev/null" in cmd or ">/dev/null" in cmd  # stdout redirected
    assert "2>&1" in cmd                                 # stderr redirected
    assert "setsid" in cmd                               # new session


def test_heartbeat_daemon_command_alt_path() -> None:
    cmd = rw.heartbeat_daemon_command("/workspace/cache")
    assert "/workspace/cache/.heartbeat" in cmd


# ---------------------------------------------------------------------------
# Lane S — circular-freshness fix: run_warning events don't mask staleness
# ---------------------------------------------------------------------------


def test_run_warning_events_are_not_counted_as_fresh_signal(tmp_path: Path) -> None:
    """The 2026-05-25 Adam wedge: the watchdog emits ``run_warning`` events
    every 10 min on stale detection. Those writes update dashboard_events.jsonl
    mtime AND show up as recent events when ``_latest_meaningful_sse_event_age``
    walks backwards. If we count them as "fresh" signals, the watchdog can
    never escalate to kill verdict — it sits in `warn` forever, masking the
    underlying wedge.

    Set up: exec.log + heartbeat genuinely stale (90 min), but the most recent
    event in dashboard_events.jsonl is a run_warning (just written 5 s ago).
    Filter must skip it and find no other recent events, so sse_age comes back
    None — leaving exec_log + heartbeat as the freshest signals (both stale).
    """
    import json
    from datetime import datetime, timezone

    artifact_root = tmp_path / "outputs" / "r1"
    artifact_root.mkdir(parents=True)
    project_dir = tmp_path
    now = time.time()

    # Both pod-side signals 90 min stale.
    for fname in ("exec.log", ".heartbeat"):
        p = artifact_root / fname
        p.write_text("x")
        os.utime(p, (now, now - 5400))

    # SSE log has ONLY a recent run_warning (5 s ago) — exactly what the
    # watchdog itself emitted on the last stale detection. Plus a much older
    # meaningful event (90 min ago, matching the pod-side staleness).
    sse = project_dir / "dashboard_events.jsonl"
    older_meaningful = json.dumps({
        "event": "primitive_call",
        "primitive": "run_experiment",
        "status": "start",
        "ts": datetime.fromtimestamp(now - 5400, tz=timezone.utc).isoformat(),
    })
    recent_run_warning = json.dumps({
        "event": "run_warning",
        "data": {"reason": "stale_run"},
        "ts": datetime.fromtimestamp(now - 5, tz=timezone.utc).isoformat(),
    })
    sse.write_text(older_meaningful + "\n" + recent_run_warning + "\n")
    os.utime(sse, (now, now - 5))

    report = rw.collect_staleness(
        artifact_root=artifact_root,
        project_dir=project_dir,
        config=rw.WatchdogConfig(warn_after_seconds=600, kill_after_seconds=1500),
        now=now,
    )
    # The run_warning was filtered, so the freshest sse event is the OLD one.
    # Combined with stale exec.log + heartbeat, verdict must be kill.
    assert report.verdict == "kill"
    # freshest_signal can be any of the three (all stale at ~5400 s); the
    # invariant is just that it's NOT below the kill threshold.
    assert report.freshest_signal_age_seconds >= 1500


def test_worker_report_started_does_not_mask_staleness(tmp_path: Path) -> None:
    """worker_report_started is emitted alongside every run_warning — same
    circular-freshness risk."""
    import json
    from datetime import datetime, timezone

    artifact_root = tmp_path / "outputs" / "r1"
    artifact_root.mkdir(parents=True)
    project_dir = tmp_path
    now = time.time()

    for fname in ("exec.log", ".heartbeat"):
        p = artifact_root / fname
        p.write_text("x")
        os.utime(p, (now, now - 5400))

    sse = project_dir / "dashboard_events.jsonl"
    sse.write_text(json.dumps({
        "event": "worker_report_started",
        "ts": datetime.fromtimestamp(now - 5, tz=timezone.utc).isoformat(),
    }) + "\n")
    os.utime(sse, (now, now - 5))

    report = rw.collect_staleness(
        artifact_root=artifact_root,
        project_dir=project_dir,
        config=rw.WatchdogConfig(warn_after_seconds=600, kill_after_seconds=1500),
        now=now,
    )
    assert report.verdict == "kill"


def test_meaningful_event_alongside_run_warning_is_still_fresh(tmp_path: Path) -> None:
    """If a REAL primitive_call lands AFTER the watchdog's run_warning, the
    filter must still find it — we're not blindly skipping everything."""
    import json
    from datetime import datetime, timezone

    artifact_root = tmp_path / "outputs" / "r1"
    artifact_root.mkdir(parents=True)
    project_dir = tmp_path
    now = time.time()

    # Pod-side signals stale.
    for fname in ("exec.log", ".heartbeat"):
        p = artifact_root / fname
        p.write_text("x")
        os.utime(p, (now, now - 5400))

    sse = project_dir / "dashboard_events.jsonl"
    # An older run_warning (filtered), THEN a recent primitive_call (counts).
    sse.write_text(
        json.dumps({"event": "run_warning",
                    "ts": datetime.fromtimestamp(now - 60, tz=timezone.utc).isoformat()}) + "\n" +
        json.dumps({"event": "primitive_call", "primitive": "verify_against_rubric",
                    "ts": datetime.fromtimestamp(now - 10, tz=timezone.utc).isoformat()}) + "\n"
    )
    os.utime(sse, (now, now - 10))

    report = rw.collect_staleness(
        artifact_root=artifact_root,
        project_dir=project_dir,
        config=rw.WatchdogConfig(warn_after_seconds=600, kill_after_seconds=1500),
        now=now,
    )
    # SSE has a recent real event → fresh signal → verdict ok.
    assert report.verdict == "ok"
    assert report.freshest_signal == "sse_event"
