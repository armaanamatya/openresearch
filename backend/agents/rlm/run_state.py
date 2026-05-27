"""Derived run-state contract — workflow-polish liveness.

Spec: ``docs/superpowers/specs/2026-05-27-derived-run-state-contract-design.md``.

The :class:`RunStateComputer` owns the run's derived liveness state machine.
It subscribes to three already-existing signals (``primitive_call`` lifecycle,
``iteration_heartbeat``, ``code/`` mtime) and emits a single ``run_state`` SSE
event on each transition. The same payload is mirrored into
``demo_status.json::run_state`` so the leaderboard and CLI tail consume the
same field as the lab UI.

Terminal states (``COMPLETED``, ``FAILED``, ``INTERRUPTED``) are absorbing —
once entered, all further ticks and signal hooks are no-ops.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Threshold defaults (env-overridable)
# ---------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


_IDLE_S_DEFAULT = 60
_STUCK_S_DEFAULT = 240  # matches _PRE_EMIT_STALL_S in primitives.py
_HEARTBEAT_GRACE_S = 60


def idle_threshold_s() -> int:
    return _env_int("REPROLAB_RUN_STATE_IDLE_S", _IDLE_S_DEFAULT)


def stuck_threshold_s() -> int:
    # Same env var as primitives.py's pre-emit stall — they must move together.
    return _env_int("REPROLAB_PRE_EMIT_STALL_S", _STUCK_S_DEFAULT)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class RunStateKind(str, Enum):
    """Derived high-level state of a run.

    Members are ``str``-based so :class:`json.dumps` serialises them as plain
    strings (no ``.value`` needed at call sites).
    """

    INITIALIZING = "initializing"
    WORKING = "working"
    IDLE = "idle"
    STUCK = "stuck"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"


TERMINAL_KINDS: frozenset[RunStateKind] = frozenset({
    RunStateKind.COMPLETED,
    RunStateKind.FAILED,
    RunStateKind.INTERRUPTED,
})


@dataclass(frozen=True)
class RunStateSubstate:
    """Per-tick context surfaced alongside the high-level ``kind``.

    ``last_file_touched`` is a *basename only*; absolute paths and corpus
    content never enter this dataclass.
    """

    primitive: str | None = None
    seconds_active: int = 0
    seconds_since_event: int = 0
    last_file_touched: str | None = None
    iteration: int = 0
    pre_emit_stalled: bool = False
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Computer
# ---------------------------------------------------------------------------


@dataclass
class _State:
    kind: RunStateKind = RunStateKind.INITIALIZING
    active_primitive: str | None = None
    primitive_started_at: float | None = None
    last_event_at: float = field(default_factory=time.time)
    iteration: int = 0
    reason: str | None = None
    locked: bool = False


class RunStateComputer:
    """Compute and emit derived ``run_state`` transitions.

    Thread-safe: every public method acquires the same lock so the periodic
    tick thread, the binding-side primitive hooks, and the crash try/finally
    cannot interleave a stale read.
    """

    def __init__(
        self,
        *,
        project_id: str,
        project_dir: Path,
        emit: Callable[[dict[str, Any]], None] | None,
        clock: Callable[[], float] = time.time,
        idle_s: int | None = None,
        stuck_s: int | None = None,
    ) -> None:
        self._project_id = project_id
        self._project_dir = project_dir
        self._emit = emit
        self._clock = clock
        self._idle_s = idle_s if idle_s is not None else idle_threshold_s()
        self._stuck_s = stuck_s if stuck_s is not None else stuck_threshold_s()
        self._lock = threading.Lock()
        self._state = _State(last_event_at=clock())
        self._last_emit_signature: tuple | None = None

    # ------------------------------------------------------------------
    # Public signal hooks
    # ------------------------------------------------------------------

    def on_primitive_start(self, primitive: str) -> None:
        with self._lock:
            if self._state.locked:
                return
            self._state.active_primitive = primitive
            self._state.primitive_started_at = self._clock()
            self._state.last_event_at = self._clock()
        self.tick()

    def on_primitive_end(self, primitive: str, outcome: str) -> None:
        with self._lock:
            if self._state.locked:
                return
            # Clearing the primitive lets the next tick fall to IDLE.
            if self._state.active_primitive == primitive:
                self._state.active_primitive = None
                self._state.primitive_started_at = None
            self._state.last_event_at = self._clock()
        self.tick()

    def on_heartbeat(self, iteration: int | None = None) -> None:
        with self._lock:
            if self._state.locked:
                return
            self._state.last_event_at = self._clock()
            if iteration is not None and iteration > self._state.iteration:
                self._state.iteration = iteration
        self.tick()

    def on_run_complete(self, status: str) -> None:
        kind = (
            RunStateKind.COMPLETED
            if status == "completed"
            else RunStateKind.FAILED
        )
        self._transition_to_terminal(kind, reason=status if kind == RunStateKind.FAILED else None)

    def on_crash(self, reason: str) -> None:
        self._transition_to_terminal(RunStateKind.FAILED, reason=reason)

    def on_sweep_interrupted(self, reason: str) -> None:
        self._transition_to_terminal(RunStateKind.INTERRUPTED, reason=reason)

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Recompute and emit if anything changed."""
        with self._lock:
            if self._state.locked:
                return
            kind, substate = self._compute_locked()
            self._maybe_emit_locked(kind, substate)

    def snapshot(self) -> tuple[RunStateKind, RunStateSubstate]:
        with self._lock:
            if self._state.locked:
                # Terminal state — return the stored kind verbatim with a
                # minimal substate that carries the reason.
                substate = RunStateSubstate(
                    primitive=None,
                    seconds_active=0,
                    seconds_since_event=0,
                    last_file_touched=self._latest_file_basename_locked(),
                    iteration=self._state.iteration,
                    pre_emit_stalled=False,
                    reason=self._state.reason,
                )
                return self._state.kind, substate
            kind, substate = self._compute_locked()
            return kind, substate

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_locked(self) -> tuple[RunStateKind, RunStateSubstate]:
        """Return the *would-emit* (kind, substate) without side effects."""
        s = self._state
        now = self._clock()
        mtime_s = self._latest_mtime_age_locked(now)
        last_event_age = int(now - s.last_event_at)
        seconds_active = (
            int(now - s.primitive_started_at)
            if s.primitive_started_at is not None
            else 0
        )
        last_file = self._latest_file_basename_locked()
        pre_emit_stalled = (
            s.active_primitive is not None
            and mtime_s is not None
            and mtime_s > self._stuck_s
        )

        # State derivation
        if s.kind == RunStateKind.INITIALIZING and s.active_primitive is None:
            kind = RunStateKind.INITIALIZING
        else:
            # An in-flight primitive with fresh mtime → WORKING.
            # No primitive in flight collapses to IDLE (between calls) unless
            # mtime is fresh (the just-completed primitive wrote files).
            fresh = mtime_s is not None and mtime_s <= self._idle_s
            if fresh:
                kind = RunStateKind.WORKING
            elif s.active_primitive is not None:
                if (
                    mtime_s is not None
                    and mtime_s > self._stuck_s
                    and last_event_age > _HEARTBEAT_GRACE_S
                ):
                    kind = RunStateKind.STUCK
                else:
                    kind = RunStateKind.IDLE
            else:
                kind = RunStateKind.IDLE

        substate = RunStateSubstate(
            primitive=s.active_primitive,
            seconds_active=seconds_active,
            seconds_since_event=last_event_age,
            last_file_touched=last_file,
            iteration=s.iteration,
            pre_emit_stalled=pre_emit_stalled,
            reason=s.reason,
        )
        return kind, substate

    def _maybe_emit_locked(
        self, kind: RunStateKind, substate: RunStateSubstate
    ) -> None:
        # De-dupe: significant changes only. We coarsen seconds_since_event /
        # seconds_active to 5-second buckets so a 5Hz tick doesn't flood the
        # stream.
        signature = (
            kind.value,
            substate.primitive,
            substate.seconds_active // 5,
            substate.seconds_since_event // 5,
            substate.last_file_touched,
            substate.iteration,
            substate.pre_emit_stalled,
            substate.reason,
        )
        if signature == self._last_emit_signature:
            return
        self._last_emit_signature = signature
        self._state.kind = kind
        self._emit_locked(kind, substate)

    def _emit_locked(self, kind: RunStateKind, substate: RunStateSubstate) -> None:
        # The event payload is corpus-free by construction (last_file_touched is
        # a basename, primitive is a known short string).
        if self._emit is None:
            return
        try:
            from backend.agents.rlm.sse_bridge import build_run_state_event

            event = build_run_state_event(
                run_id=self._project_id,
                kind=kind.value,
                substate=substate.to_dict(),
            )
            self._emit(event)
        except Exception:  # noqa: BLE001 — never crash the run
            logger.exception("RunStateComputer: emit failed")
        # Mirror into demo_status.json::run_state.
        try:
            self._mirror_demo_status_locked(kind, substate)
        except Exception:  # noqa: BLE001
            logger.exception("RunStateComputer: demo_status mirror failed")

    def _mirror_demo_status_locked(
        self, kind: RunStateKind, substate: RunStateSubstate
    ) -> None:
        path = self._project_dir / "demo_status.json"
        if not path.exists():
            return
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(existing, dict):
            return
        existing["run_state"] = {
            "kind": kind.value,
            "substate": substate.to_dict(),
            "updatedAt": datetime.now(timezone.utc).isoformat(),
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def _latest_mtime_age_locked(self, now: float) -> int | None:
        code_dir = self._project_dir / "code"
        if not code_dir.exists():
            return None
        latest = 0.0
        try:
            for f in code_dir.iterdir():
                if f.is_file():
                    try:
                        mt = f.stat().st_mtime
                    except OSError:
                        continue
                    if mt > latest:
                        latest = mt
        except OSError:
            return None
        if latest == 0.0:
            return None
        return max(0, int(now - latest))

    def _latest_file_basename_locked(self) -> str | None:
        code_dir = self._project_dir / "code"
        if not code_dir.exists():
            return None
        latest_file: Path | None = None
        latest_mt: float = 0.0
        try:
            for f in code_dir.iterdir():
                if f.is_file():
                    try:
                        mt = f.stat().st_mtime
                    except OSError:
                        continue
                    if mt > latest_mt:
                        latest_mt = mt
                        latest_file = f
        except OSError:
            return None
        return latest_file.name if latest_file is not None else None

    def _transition_to_terminal(
        self, kind: RunStateKind, *, reason: str | None
    ) -> None:
        with self._lock:
            if self._state.locked:
                return
            self._state = replace(
                self._state,
                kind=kind,
                active_primitive=None,
                primitive_started_at=None,
                last_event_at=self._clock(),
                reason=reason,
                locked=True,
            )
            # Force the dedupe signature to None so this emit always fires.
            self._last_emit_signature = None
            substate = RunStateSubstate(
                primitive=None,
                seconds_active=0,
                seconds_since_event=0,
                last_file_touched=self._latest_file_basename_locked(),
                iteration=self._state.iteration,
                pre_emit_stalled=False,
                reason=reason,
            )
            self._emit_locked(kind, substate)


# ---------------------------------------------------------------------------
# Periodic tick daemon
# ---------------------------------------------------------------------------


def start_periodic_ticker(
    computer: RunStateComputer,
    *,
    interval_s: float = 5.0,
    stop_event: threading.Event | None = None,
) -> threading.Thread:
    """Start a daemon thread that ``tick()``s the computer every interval.

    Stops on the supplied event (or never, if absent). The thread is daemon so
    it does not block process exit.
    """
    stopper = stop_event or threading.Event()

    def _loop() -> None:
        while not stopper.wait(max(0.1, interval_s)):
            try:
                computer.tick()
            except Exception:  # noqa: BLE001
                logger.exception("RunStateComputer: periodic tick failed")

    thread = threading.Thread(
        target=_loop,
        name=f"run-state-tick-{computer._project_id}",  # noqa: SLF001 — internal label
        daemon=True,
    )
    thread.start()
    return thread


__all__ = [
    "RunStateComputer",
    "RunStateKind",
    "RunStateSubstate",
    "TERMINAL_KINDS",
    "idle_threshold_s",
    "start_periodic_ticker",
    "stuck_threshold_s",
]
