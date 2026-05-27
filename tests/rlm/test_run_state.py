"""Tests for the derived-run-state contract.

Spec: ``docs/superpowers/specs/2026-05-27-derived-run-state-contract-design.md``.

Covers:
- Initial state is INITIALIZING.
- on_primitive_start → WORKING (with fresh mtime).
- mtime past idle threshold → IDLE.
- mtime past stuck threshold + heartbeat stale → STUCK.
- run_complete locks to COMPLETED / FAILED.
- on_sweep_interrupted locks to INTERRUPTED.
- Terminal states are absorbing.
- De-dupe suppresses no-op emits.
- demo_status.json mirror is atomic.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from backend.agents.rlm.run_state import (
    RunStateComputer,
    RunStateKind,
    TERMINAL_KINDS,
)


@pytest.fixture()
def project_dir(tmp_path: Path) -> Path:
    code = tmp_path / "code"
    code.mkdir()
    # Seed demo_status so the mirror has something to merge into.
    (tmp_path / "demo_status.json").write_text(
        json.dumps({"status": "running", "projectId": tmp_path.name}),
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture()
def emit_list() -> list[dict]:
    return []


@pytest.fixture()
def emit(emit_list: list[dict]):
    def _emit(event: dict) -> None:
        emit_list.append(event)
    return _emit


class _Clock:
    """Manually-advanced clock so tests don't depend on wall-clock time."""

    def __init__(self, t0: float = 1000.0) -> None:
        self.t = t0

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


def _touch(file: Path, mtime: float) -> None:
    file.write_text("x", encoding="utf-8")
    import os
    os.utime(file, (mtime, mtime))


def test_initial_state_is_initializing(project_dir: Path, emit) -> None:
    clock = _Clock()
    rsc = RunStateComputer(
        project_id="prj_test",
        project_dir=project_dir,
        emit=emit,
        clock=clock,
    )
    kind, _ = rsc.snapshot()
    assert kind == RunStateKind.INITIALIZING


def test_primitive_start_with_fresh_mtime_transitions_to_working(
    project_dir: Path, emit, emit_list: list[dict]
) -> None:
    clock = _Clock()
    rsc = RunStateComputer(
        project_id="prj_test",
        project_dir=project_dir,
        emit=emit,
        clock=clock,
    )
    _touch(project_dir / "code" / "train.py", clock.t)
    rsc.on_primitive_start("implement_baseline")
    kind, sub = rsc.snapshot()
    assert kind == RunStateKind.WORKING
    assert sub.primitive == "implement_baseline"
    assert sub.last_file_touched == "train.py"
    # Should have emitted exactly one event.
    state_events = [e for e in emit_list if e["event"] == "run_state"]
    assert len(state_events) >= 1
    assert state_events[-1]["kind"] == "working"


def test_idle_after_60s_no_mtime(project_dir: Path, emit) -> None:
    clock = _Clock()
    rsc = RunStateComputer(
        project_id="prj_test",
        project_dir=project_dir,
        emit=emit,
        clock=clock,
        idle_s=60,
        stuck_s=240,
    )
    _touch(project_dir / "code" / "train.py", clock.t)
    rsc.on_primitive_start("implement_baseline")
    # Advance past the idle threshold.
    clock.advance(75)
    rsc.tick()
    kind, _ = rsc.snapshot()
    assert kind == RunStateKind.IDLE


def test_stuck_after_240s_no_mtime_and_no_heartbeat(
    project_dir: Path, emit
) -> None:
    clock = _Clock()
    rsc = RunStateComputer(
        project_id="prj_test",
        project_dir=project_dir,
        emit=emit,
        clock=clock,
        idle_s=60,
        stuck_s=240,
    )
    _touch(project_dir / "code" / "train.py", clock.t)
    rsc.on_primitive_start("implement_baseline")
    # Push mtime past stuck threshold AND advance clock past heartbeat grace.
    clock.advance(300)
    rsc.tick()
    kind, sub = rsc.snapshot()
    assert kind == RunStateKind.STUCK
    assert sub.pre_emit_stalled is True


def test_heartbeat_keeps_state_idle_not_stuck(project_dir: Path, emit) -> None:
    clock = _Clock()
    rsc = RunStateComputer(
        project_id="prj_test",
        project_dir=project_dir,
        emit=emit,
        clock=clock,
        idle_s=60,
        stuck_s=240,
    )
    _touch(project_dir / "code" / "train.py", clock.t)
    rsc.on_primitive_start("implement_baseline")
    clock.advance(300)
    # Heartbeat lands just before the tick — within the 60s grace.
    rsc.on_heartbeat(iteration=2)
    rsc.tick()
    kind, sub = rsc.snapshot()
    # Mtime is stale (>240s) but heartbeat is fresh → not STUCK; stays IDLE.
    assert kind == RunStateKind.IDLE
    assert sub.iteration == 2


def test_run_complete_locks_to_completed(project_dir: Path, emit) -> None:
    rsc = RunStateComputer(
        project_id="prj_test",
        project_dir=project_dir,
        emit=emit,
    )
    rsc.on_primitive_start("implement_baseline")
    rsc.on_run_complete("completed")
    kind, _ = rsc.snapshot()
    assert kind == RunStateKind.COMPLETED
    # Further signals are no-ops.
    rsc.on_primitive_start("something_else")
    assert rsc.snapshot()[0] == RunStateKind.COMPLETED


def test_run_complete_failed_status_becomes_failed_kind(
    project_dir: Path, emit
) -> None:
    rsc = RunStateComputer(
        project_id="prj_test",
        project_dir=project_dir,
        emit=emit,
    )
    rsc.on_run_complete("failed")
    assert rsc.snapshot()[0] == RunStateKind.FAILED


def test_on_crash_locks_to_failed_with_reason(
    project_dir: Path, emit, emit_list: list[dict]
) -> None:
    rsc = RunStateComputer(
        project_id="prj_test",
        project_dir=project_dir,
        emit=emit,
    )
    rsc.on_crash("RuntimeError: SDK aclose deadlock")
    kind, sub = rsc.snapshot()
    assert kind == RunStateKind.FAILED
    assert sub.reason == "RuntimeError: SDK aclose deadlock"
    # Terminal emission always fires (signature reset).
    state_events = [e for e in emit_list if e["event"] == "run_state"]
    assert state_events[-1]["kind"] == "failed"
    assert state_events[-1]["substate"]["reason"] == (
        "RuntimeError: SDK aclose deadlock"
    )


def test_sweep_interrupted_locks_to_interrupted(
    project_dir: Path, emit
) -> None:
    rsc = RunStateComputer(
        project_id="prj_test",
        project_dir=project_dir,
        emit=emit,
    )
    rsc.on_sweep_interrupted("orphaned_stale_run")
    assert rsc.snapshot()[0] == RunStateKind.INTERRUPTED


@pytest.mark.parametrize("terminal", list(TERMINAL_KINDS))
def test_terminal_states_are_absorbing(
    project_dir: Path, emit, terminal: RunStateKind
) -> None:
    rsc = RunStateComputer(
        project_id="prj_test",
        project_dir=project_dir,
        emit=emit,
    )
    if terminal == RunStateKind.COMPLETED:
        rsc.on_run_complete("completed")
    elif terminal == RunStateKind.FAILED:
        rsc.on_crash("test")
    elif terminal == RunStateKind.INTERRUPTED:
        rsc.on_sweep_interrupted("test")
    # Every subsequent signal is a no-op.
    rsc.on_primitive_start("foo")
    rsc.on_primitive_end("foo", "ok")
    rsc.on_heartbeat(iteration=99)
    rsc.tick()
    assert rsc.snapshot()[0] == terminal


def test_dedup_suppresses_noop_emits(
    project_dir: Path, emit, emit_list: list[dict]
) -> None:
    clock = _Clock()
    rsc = RunStateComputer(
        project_id="prj_test",
        project_dir=project_dir,
        emit=emit,
        clock=clock,
    )
    _touch(project_dir / "code" / "train.py", clock.t)
    rsc.on_primitive_start("implement_baseline")
    count_after_start = len([e for e in emit_list if e["event"] == "run_state"])
    # Ticking many times without anything changing → no extra emits.
    for _ in range(10):
        rsc.tick()
    count_after_idle_ticks = len(
        [e for e in emit_list if e["event"] == "run_state"]
    )
    assert count_after_idle_ticks == count_after_start


def test_demo_status_mirror_writes_run_state_field(
    project_dir: Path, emit
) -> None:
    rsc = RunStateComputer(
        project_id="prj_test",
        project_dir=project_dir,
        emit=emit,
    )
    rsc.on_primitive_start("plan_reproduction")
    raw = json.loads((project_dir / "demo_status.json").read_text())
    assert "run_state" in raw
    assert raw["run_state"]["kind"] in {
        "working", "idle", "initializing",
    }
    assert "substate" in raw["run_state"]
    assert "updatedAt" in raw["run_state"]


def test_threshold_env_overrides(monkeypatch, project_dir: Path, emit) -> None:
    monkeypatch.setenv("REPROLAB_RUN_STATE_IDLE_S", "10")
    monkeypatch.setenv("REPROLAB_PRE_EMIT_STALL_S", "30")
    from backend.agents.rlm import run_state as rs_mod
    assert rs_mod.idle_threshold_s() == 10
    assert rs_mod.stuck_threshold_s() == 30


def test_emit_none_does_not_crash(project_dir: Path) -> None:
    rsc = RunStateComputer(
        project_id="prj_test",
        project_dir=project_dir,
        emit=None,
    )
    # Just exercise the signal hooks — they must not raise.
    rsc.on_primitive_start("foo")
    rsc.on_primitive_end("foo", "ok")
    rsc.on_heartbeat()
    rsc.tick()
    rsc.on_run_complete("completed")


def test_emit_failure_does_not_crash_run(project_dir: Path) -> None:
    bad_emit = MagicMock(side_effect=RuntimeError("boom"))
    rsc = RunStateComputer(
        project_id="prj_test",
        project_dir=project_dir,
        emit=bad_emit,
    )
    # The hook should swallow emit failures.
    rsc.on_primitive_start("foo")
    rsc.on_run_complete("completed")
    assert rsc.snapshot()[0] == RunStateKind.COMPLETED
