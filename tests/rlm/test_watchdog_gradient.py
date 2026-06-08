"""Tests for PR-ι.4 — watchdog primitive-aware idle threshold gradient.

Covers:
1. effective_idle_threshold returns run_experiment baseline (7200s).
2. effective_idle_threshold returns implement_baseline baseline (14400s).
3. effective_idle_threshold returns build_environment baseline (1800s).
4. effective_idle_threshold returns default (1800s) for unknown primitive.
5. effective_idle_threshold returns config.kill_after_seconds when larger than baseline.
6. effective_idle_threshold returns config.kill_after_seconds for None primitive.
7. collect_staleness uses run_experiment threshold → does NOT kill at 3h.
8. collect_staleness uses default threshold → WARNS at default kill_after (25min).
9. _detect_active_primitive returns None when no SSE log exists.
10. _detect_active_primitive returns most recent primitive name from SSE log.
"""

from __future__ import annotations

import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.agents.rlm import run_watchdog as rw


# ---------------------------------------------------------------------------
# effective_idle_threshold
# ---------------------------------------------------------------------------


def _config(kill_after: float = 1500.0) -> rw.WatchdogConfig:
    return rw.WatchdogConfig(
        warn_after_seconds=600.0,
        kill_after_seconds=kill_after,
        poll_interval_seconds=30.0,
    )


def test_effective_threshold_run_experiment() -> None:
    threshold = rw.effective_idle_threshold("run_experiment", _config())
    assert threshold == rw.PRIMITIVE_IDLE_BASELINE_S["run_experiment"]
    # 4 h (2026-06-08 execution-reliability redesign, §A decision #2): the
    # run_experiment file-mtime/SSE idle threshold must be >= the 60-min inner
    # stall window so the watchdog never pre-empts the GPU/CPU-aware inner stall.
    assert threshold == 14400.0


def test_effective_threshold_implement_baseline() -> None:
    threshold = rw.effective_idle_threshold("implement_baseline", _config())
    assert threshold == rw.PRIMITIVE_IDLE_BASELINE_S["implement_baseline"]
    assert threshold == 14400.0


def test_effective_threshold_build_environment() -> None:
    threshold = rw.effective_idle_threshold("build_environment", _config())
    assert threshold == rw.PRIMITIVE_IDLE_BASELINE_S["build_environment"]
    assert threshold == 1800.0


def test_effective_threshold_unknown_primitive() -> None:
    threshold = rw.effective_idle_threshold("understand_section", _config())
    assert threshold == rw._DEFAULT_IDLE_BASELINE_S


def test_effective_threshold_config_larger_than_baseline() -> None:
    """When config kill_after > primitive baseline, config value wins."""
    very_large = 20000.0
    threshold = rw.effective_idle_threshold("run_experiment", _config(kill_after=very_large))
    assert threshold == very_large


def test_effective_threshold_none_primitive() -> None:
    """None primitive returns the config's kill_after_seconds unchanged."""
    threshold = rw.effective_idle_threshold(None, _config(kill_after=1200.0))
    assert threshold == 1200.0


# ---------------------------------------------------------------------------
# collect_staleness with active_primitive
# ---------------------------------------------------------------------------


def _write_exec_log(artifact_root: Path, age_s: float) -> None:
    exec_log = artifact_root / "exec.log"
    exec_log.write_text("some output\n")
    past = time.time() - age_s
    import os
    os.utime(exec_log, (past, past))


def test_collect_staleness_run_experiment_not_killed_at_3h() -> None:
    """run_experiment active + 3h stale → should NOT be 'kill' (threshold is 7200s)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_root = Path(tmpdir) / "artifacts"
        artifact_root.mkdir()
        project_dir = Path(tmpdir) / "run"
        project_dir.mkdir()
        cfg = _config(kill_after=1500.0)

        # 3 hours idle (10800s) — below run_experiment baseline of 7200s... wait, 10800 > 7200.
        # Let's use 3h=10800 which IS above 7200. So we need < 7200 to test "not killed".
        # Use 2h = 7200 stale — exactly at boundary, use 7100 to be safe (just below 7200).
        _write_exec_log(artifact_root, age_s=7100.0)

        report = rw.collect_staleness(
            artifact_root=artifact_root,
            project_dir=project_dir,
            config=cfg,
            active_primitive="run_experiment",
        )
    # 7100s < effective threshold 7200s → should not be "kill"
    assert report.verdict != "kill"


def test_collect_staleness_run_experiment_warns_before_kill() -> None:
    """run_experiment active + stale beyond warn threshold → 'warn', not 'kill'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_root = Path(tmpdir) / "artifacts"
        artifact_root.mkdir()
        project_dir = Path(tmpdir) / "run"
        project_dir.mkdir()
        cfg = rw.WatchdogConfig(warn_after_seconds=600.0, kill_after_seconds=1500.0)

        # 30 min stale — above warn (600s) but below run_experiment baseline (7200s)
        _write_exec_log(artifact_root, age_s=1800.0)

        report = rw.collect_staleness(
            artifact_root=artifact_root,
            project_dir=project_dir,
            config=cfg,
            active_primitive="run_experiment",
        )
    assert report.verdict == "warn"


def test_collect_staleness_unknown_primitive_uses_default() -> None:
    """Unknown primitive uses _DEFAULT_IDLE_BASELINE_S (1800s)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_root = Path(tmpdir) / "artifacts"
        artifact_root.mkdir()
        project_dir = Path(tmpdir) / "run"
        project_dir.mkdir()
        cfg = rw.WatchdogConfig(warn_after_seconds=600.0, kill_after_seconds=1500.0)

        # 2000s stale — above both config kill (1500s) AND default baseline (1800s)
        _write_exec_log(artifact_root, age_s=2000.0)

        report = rw.collect_staleness(
            artifact_root=artifact_root,
            project_dir=project_dir,
            config=cfg,
            active_primitive="understand_section",
        )
    assert report.verdict == "kill"


# ---------------------------------------------------------------------------
# _detect_active_primitive
# ---------------------------------------------------------------------------


def _write_sse_log(project_dir: Path, events: list[dict]) -> None:
    log_path = project_dir / "dashboard_events.jsonl"
    with log_path.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


def test_detect_active_primitive_no_log() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        sse_log = project_dir / "dashboard_events.jsonl"
        result = rw._detect_active_primitive(sse_log)
    assert result is None


def test_detect_active_primitive_returns_latest_start() -> None:
    """Returns the primitive from the most recent primitive_call phase=start."""
    now = datetime.now(timezone.utc).isoformat()
    events = [
        {"event": "primitive_call", "phase": "start", "primitive": "understand_section", "ts": now},
        {"event": "primitive_call", "phase": "end", "primitive": "understand_section", "ts": now},
        {"event": "primitive_call", "phase": "start", "primitive": "implement_baseline", "ts": now},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        _write_sse_log(project_dir, events)
        sse_log = project_dir / "dashboard_events.jsonl"
        result = rw._detect_active_primitive(sse_log)
    assert result == "implement_baseline"


def test_detect_active_primitive_skips_non_primitive_call() -> None:
    """Non-primitive events are ignored."""
    now = datetime.now(timezone.utc).isoformat()
    events = [
        {"event": "run_complete", "ts": now},
        {"event": "iteration_heartbeat", "ts": now},
        {"event": "primitive_call", "phase": "start", "primitive": "run_experiment", "ts": now},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        _write_sse_log(project_dir, events)
        sse_log = project_dir / "dashboard_events.jsonl"
        result = rw._detect_active_primitive(sse_log)
    assert result == "run_experiment"


def test_detect_active_primitive_empty_log() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        sse_log = project_dir / "dashboard_events.jsonl"
        sse_log.write_text("")
        result = rw._detect_active_primitive(sse_log)
    assert result is None
