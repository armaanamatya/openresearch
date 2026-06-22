"""Deterministic two-signal stall guard for REMOTE (RunPod) experiment execution.

WHY this module exists
----------------------
The ``local`` sandbox already kills a wedged training process EARLY, the moment
two independent liveness signals go quiet (``local_process._monitor`` →
``exec_stalled``). The ``runpod`` remote exec path historically had NO such
liveness detection (see ``runpod_backend._warn_no_remote_stall_detection_once``):
a remote NCCL deadlock, a stuck dataset download, or an idle GPU burns the FULL
per-command hard timeout before it is reaped. On a paid GPU that is direct,
avoidable waste.

This helper mirrors the ``local_process`` two-signal discipline for the remote
SSH stream, but stays transport-agnostic: it reads two abstract signal sources
(an output-progress counter and a checkpoint/metrics mtime) via injected
callables so it can be driven hermetically in tests with a fake clock and fake
readers — no SSH, no pod, no GPU.

Design invariants (all enforced here, see ``RemoteStallGuard``):

* **Default OFF.** ``OPENRESEARCH_REMOTE_STALL_GUARD`` must be truthy
  (``1``/``true``/``yes``/``on``) to arm the guard. When unarmed, ``observe``
  always returns ``False`` and the remote exec path is byte-for-byte unchanged.
* **Two independent signals required.** A kill verdict needs BOTH the output
  signal AND the checkpoint signal to have been stale for at least the stall
  window. A single quiet signal (e.g. a long compute step that emits no log
  lines but is still checkpointing, or vice-versa) must NEVER trigger a kill.
* **Fail-soft.** Any error in the bookkeeping (a reader raising, a bad clock)
  degrades to "not stalled" — the guard must never kill a healthy run or raise
  into the exec loop. On error the guard resets liveness so it cannot
  accumulate a false stall across a transient reader failure.
* **Grounded evidence.** When a stall is declared the guard exposes an
  ``evidence`` packet (which signals were stale, and for how long) so the
  caller can persist it and the failure classification is grounded in fact,
  not a bare timeout.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

__all__ = [
    "remote_stall_guard_enabled",
    "RemoteStallGuard",
    "StallEvidence",
]

_TRUTHY = {"1", "true", "yes", "on"}


def remote_stall_guard_enabled(env: dict[str, str] | None = None) -> bool:
    """Return whether the remote stall guard is armed for this process.

    Gated behind ``OPENRESEARCH_REMOTE_STALL_GUARD``; default OFF so the runpod
    remote exec path is byte-for-byte unchanged unless an operator opts in.
    Fail-soft: any error reading the env resolves to OFF.
    """
    try:
        src = os.environ if env is None else env
        raw = (src.get("OPENRESEARCH_REMOTE_STALL_GUARD", "") or "").strip().lower()
        return raw in _TRUTHY
    except Exception:  # noqa: BLE001 — never let env parsing break exec
        return False


@dataclass(frozen=True)
class StallEvidence:
    """Grounded evidence packet for a declared remote stall.

    Persisted alongside the kill so the ``exec_stalled`` classification names the
    two specific signals that were stale and for how long, rather than asserting
    a hang from a bare timeout.
    """

    stall_window_s: float
    output_stale_s: float
    checkpoint_stale_s: float
    output_signal_stale: bool
    checkpoint_signal_stale: bool
    last_output_count: int
    last_checkpoint_mtime: float

    def as_dict(self) -> dict[str, object]:
        return {
            "stall_window_s": self.stall_window_s,
            "output_stale_s": round(self.output_stale_s, 3),
            "checkpoint_stale_s": round(self.checkpoint_stale_s, 3),
            "output_signal_stale": self.output_signal_stale,
            "checkpoint_signal_stale": self.checkpoint_signal_stale,
            "last_output_count": self.last_output_count,
            "last_checkpoint_mtime": self.last_checkpoint_mtime,
            "signals_required": 2,
        }


@dataclass
class RemoteStallGuard:
    """Two-signal stall detector for a single remote exec command.

    Construct once per command, then call :meth:`observe` on each monitor poll.
    ``observe`` returns ``True`` exactly once, when BOTH signals have been
    continuously stale for at least ``stall_window_s`` — at which point the
    caller should terminate the remote process and classify it ``exec_stalled``,
    persisting :attr:`evidence`.

    Signals are read via injected zero-arg callables so the detector is fully
    hermetic in tests:

    * ``output_count_reader() -> int`` — a monotonically non-decreasing count of
      output units seen so far (lines streamed, or log-file byte length). A bump
      means the remote process is still talking.
    * ``checkpoint_mtime_reader() -> float`` — the newest mtime of any
      checkpoint/metrics artifact (0.0 if none yet). A bump means the remote
      process is still writing progress.

    The clock is injected (``monotonic``) so tests advance time deterministically
    without sleeping.
    """

    stall_window_s: float
    output_count_reader: Callable[[], int]
    checkpoint_mtime_reader: Callable[[], float]
    monotonic: Callable[[], float]
    enabled: bool = True

    # internal bookkeeping (not constructor args)
    _started: bool = field(default=False, init=False)
    _last_output_count: int = field(default=0, init=False)
    _last_checkpoint_mtime: float = field(default=0.0, init=False)
    _last_output_live_at: float = field(default=0.0, init=False)
    _last_checkpoint_live_at: float = field(default=0.0, init=False)
    _fired: bool = field(default=False, init=False)
    evidence: StallEvidence | None = field(default=None, init=False)

    def _now(self) -> float | None:
        try:
            return float(self.monotonic())
        except Exception:  # noqa: BLE001
            return None

    def _start(self, now: float) -> None:
        # Seed the baselines from the first successful read so an initial value
        # is never mistaken for staleness.
        self._last_output_count = int(self.output_count_reader())
        self._last_checkpoint_mtime = float(self.checkpoint_mtime_reader())
        self._last_output_live_at = now
        self._last_checkpoint_live_at = now
        self._started = True

    def observe(self) -> bool:
        """Sample both signals once; return ``True`` iff a stall is now declared.

        Idempotent after firing (returns ``True`` again without re-evaluating).
        Disabled / non-positive window / any bookkeeping error → ``False`` and
        liveness is reset so a transient fault cannot manufacture a stall.
        """
        if not self.enabled:
            return False
        if self.stall_window_s is None or self.stall_window_s <= 0:
            return False
        if self._fired:
            return True

        now = self._now()
        if now is None:
            return False

        try:
            if not self._started:
                self._start(now)
                return False

            out = int(self.output_count_reader())
            if out > self._last_output_count:
                self._last_output_live_at = now
            self._last_output_count = out

            ckpt = float(self.checkpoint_mtime_reader())
            if ckpt > self._last_checkpoint_mtime:
                self._last_checkpoint_live_at = now
            self._last_checkpoint_mtime = ckpt
        except Exception:  # noqa: BLE001 — fail-soft: reset liveness, never fire
            self._last_output_live_at = now
            self._last_checkpoint_live_at = now
            return False

        output_stale_s = now - self._last_output_live_at
        checkpoint_stale_s = now - self._last_checkpoint_live_at
        output_signal_stale = output_stale_s >= self.stall_window_s
        checkpoint_signal_stale = checkpoint_stale_s >= self.stall_window_s

        # TWO independent signals required — a single quiet signal is normal
        # (a long no-log compute step still checkpoints; a checkpoint-free warmup
        # still logs). Only when BOTH have been stale past the window is it a hang.
        if output_signal_stale and checkpoint_signal_stale:
            self._fired = True
            self.evidence = StallEvidence(
                stall_window_s=self.stall_window_s,
                output_stale_s=output_stale_s,
                checkpoint_stale_s=checkpoint_stale_s,
                output_signal_stale=True,
                checkpoint_signal_stale=True,
                last_output_count=self._last_output_count,
                last_checkpoint_mtime=self._last_checkpoint_mtime,
            )
            return True
        return False
