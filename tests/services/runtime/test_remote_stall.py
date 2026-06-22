"""Hermetic tests for the deterministic remote (RunPod) two-signal stall guard.

No SSH, no pod, no network, no GPU, no real time: the guard is driven with an
injected fake clock and fake signal readers so every scenario is deterministic.

Scenarios (per the feature spec):
 1. OFF (flag unset / disabled) → guard never fires.
 2. ON + BOTH signals stale past the window → fires ``exec_stalled`` with a
    grounded evidence packet.
 3. ON + only ONE signal stale → does NOT fire (two-signal discipline).
 4. ON + output keeps growing → does NOT fire.
 5. Bookkeeping error (a reader raises) → fail-soft (no fire, no raise).
"""

from __future__ import annotations

import pytest

from backend.services.runtime.remote_stall import (
    RemoteStallGuard,
    remote_stall_guard_enabled,
)


class _FakeClock:
    """Deterministic monotonic clock advanced explicitly by the test."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _guard(clock, output_reader, ckpt_reader, *, window=100.0, enabled=True):
    return RemoteStallGuard(
        stall_window_s=window,
        output_count_reader=output_reader,
        checkpoint_mtime_reader=ckpt_reader,
        monotonic=clock,
        enabled=enabled,
    )


# --- (1) flag gating --------------------------------------------------------


def test_flag_default_off(monkeypatch):
    """Unset env → disabled (byte-for-byte legacy behaviour)."""
    monkeypatch.delenv("OPENRESEARCH_REMOTE_STALL_GUARD", raising=False)
    assert remote_stall_guard_enabled() is False


@pytest.mark.parametrize("val", ["1", "true", "yes", "on", "TRUE", "Yes"])
def test_flag_truthy_values_enable(monkeypatch, val):
    monkeypatch.setenv("OPENRESEARCH_REMOTE_STALL_GUARD", val)
    assert remote_stall_guard_enabled() is True


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "", "maybe"])
def test_flag_non_truthy_values_disable(monkeypatch, val):
    monkeypatch.setenv("OPENRESEARCH_REMOTE_STALL_GUARD", val)
    assert remote_stall_guard_enabled() is False


def test_disabled_guard_never_fires():
    """(1) A guard constructed disabled never fires regardless of staleness."""
    clock = _FakeClock()
    g = _guard(clock, lambda: 0, lambda: 0.0, enabled=False)
    g.observe()  # seed
    clock.advance(10_000.0)
    assert g.observe() is False
    assert g.evidence is None


def test_zero_window_disables():
    """A non-positive window disables the guard (mirror of local ``0`` semantics)."""
    clock = _FakeClock()
    g = _guard(clock, lambda: 0, lambda: 0.0, window=0.0)
    clock.advance(10_000.0)
    assert g.observe() is False


# --- (2) both signals stale → fire ------------------------------------------


def test_both_signals_stale_fires_with_evidence():
    """(2) Output frozen AND checkpoint frozen past the window → exec_stalled."""
    clock = _FakeClock()
    # Both readers return a constant after the seed → both go stale.
    g = _guard(clock, lambda: 5, lambda: 1000.0, window=100.0)

    assert g.observe() is False  # seed at t=0
    clock.advance(50.0)
    assert g.observe() is False  # within window
    clock.advance(60.0)  # now 110s of staleness on both > 100s window
    assert g.observe() is True

    ev = g.evidence
    assert ev is not None
    assert ev.output_signal_stale is True
    assert ev.checkpoint_signal_stale is True
    assert ev.stall_window_s == 100.0
    assert ev.output_stale_s >= 100.0
    assert ev.checkpoint_stale_s >= 100.0

    packet = ev.as_dict()
    assert packet["signals_required"] == 2
    assert packet["output_signal_stale"] is True
    assert packet["checkpoint_signal_stale"] is True


def test_fire_is_idempotent():
    """Once fired, observe keeps returning True without re-evaluating."""
    clock = _FakeClock()
    g = _guard(clock, lambda: 5, lambda: 1000.0, window=100.0)
    g.observe()
    clock.advance(200.0)
    assert g.observe() is True
    first = g.evidence
    assert g.observe() is True
    assert g.evidence is first  # not recomputed


# --- (3) only one signal stale → no fire ------------------------------------


def test_only_output_stale_does_not_fire():
    """(3) Output frozen but checkpoints keep advancing → NOT a stall."""
    clock = _FakeClock()
    ckpt = {"m": 1000.0}
    g = _guard(clock, lambda: 5, lambda: ckpt["m"], window=100.0)

    assert g.observe() is False  # seed
    for _ in range(20):
        clock.advance(30.0)
        ckpt["m"] += 1.0  # checkpoint stays live
        assert g.observe() is False


def test_only_checkpoint_stale_does_not_fire():
    """(3, mirror) Checkpoints frozen but output keeps streaming → NOT a stall."""
    clock = _FakeClock()
    lines = {"n": 0}
    g = _guard(clock, lambda: lines["n"], lambda: 1000.0, window=100.0)

    assert g.observe() is False  # seed
    for _ in range(20):
        clock.advance(30.0)
        lines["n"] += 1  # output stays live
        assert g.observe() is False


# --- (4) output keeps growing → no fire -------------------------------------


def test_output_growing_never_fires():
    """(4) Both signals advancing every poll → never declared stalled."""
    clock = _FakeClock()
    lines = {"n": 0}
    ckpt = {"m": 0.0}
    g = _guard(clock, lambda: lines["n"], lambda: ckpt["m"], window=100.0)

    assert g.observe() is False
    for _ in range(50):
        clock.advance(40.0)
        lines["n"] += 3
        ckpt["m"] += 5.0
        assert g.observe() is False
    assert g.evidence is None


# --- (5) bookkeeping error → fail-soft ---------------------------------------


def test_reader_error_is_fail_soft():
    """(5) A reader raising must NOT fire and must NOT propagate."""
    clock = _FakeClock()

    calls = {"n": 0}

    def flaky_output() -> int:
        calls["n"] += 1
        if calls["n"] >= 2:  # raise after the seed read
            raise RuntimeError("transient probe failure")
        return 0

    g = _guard(clock, flaky_output, lambda: 1000.0, window=100.0)
    assert g.observe() is False  # seed
    clock.advance(10_000.0)
    # Reader raises on this sample → fail-soft: no fire, liveness reset.
    assert g.observe() is False
    assert g.evidence is None


def test_clock_error_is_fail_soft():
    """A broken clock degrades to not-stalled rather than raising."""

    def bad_clock() -> float:
        raise RuntimeError("clock exploded")

    g = _guard(bad_clock, lambda: 0, lambda: 0.0, window=100.0)
    assert g.observe() is False
    assert g.evidence is None


def test_reader_recovery_after_transient_error_does_not_false_fire():
    """A transient reader error resets liveness; a later real stall still fires."""
    clock = _FakeClock()
    state = {"fail": False}

    def flaky_output() -> int:
        if state["fail"]:
            raise RuntimeError("blip")
        return 7  # constant → would go stale if reads succeed

    g = _guard(clock, flaky_output, lambda: 1000.0, window=100.0)
    assert g.observe() is False  # seed t=0

    # A transient failure mid-run resets liveness (no accumulated stall).
    clock.advance(80.0)
    state["fail"] = True
    assert g.observe() is False
    state["fail"] = False

    # Liveness was reset to t=80; a constant-output stall must take a FULL
    # window from there, not from t=0.
    clock.advance(50.0)  # t=130, only 50s since reset
    assert g.observe() is False
    clock.advance(60.0)  # t=190, 110s since reset > window
    assert g.observe() is True
    assert g.evidence is not None
