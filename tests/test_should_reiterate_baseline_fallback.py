"""Unit tests for the widened ``_should_reiterate`` predicate (Option D, Q2).

Before Option D, ``_should_reiterate`` consulted only ``improved_verification``
and returned False whenever improved was None. The orchestrator worked around
that by always running one unconditional ``run_improvements + run_gate_3``
pair before the loop, so improved was populated by the time the predicate
ran.

After Option D, the unconditional pre-loop pair is deleted and the predicate
becomes the sole gate for whether the loop fires at all. To keep that
behavior coherent on the first iteration, the predicate widens: it now
seeds from ``baseline_verification`` when ``improved_verification`` is None.

These tests pin the new signature ``(improved, baseline, iteration, max)``
and the four invariants that follow from it:

  1. Both None → False (no rubric signal to drive iteration).
  2. Baseline below target with no improved → True (seed from baseline).
  3. Baseline meets target with no improved → False (don't waste a round).
  4. Improved overrides baseline once present.

See ``docs/design/option-d-q1q2-refactor.md`` Task 1.
"""

from __future__ import annotations

from backend.agents.orchestrator import _should_reiterate
from backend.agents.schemas import RubricVerification


def _verif(score: float, target: float) -> RubricVerification:
    return RubricVerification(
        overall_score=score,
        target_score=target,
        meets_target=score >= target,
        rubric_source="generated",
    )


def test_both_none_returns_false():
    """No rubric signal — neither verification present. Loop must not fire."""
    assert _should_reiterate(None, None, 0, 5) is False


def test_baseline_below_target_with_no_improved_returns_true():
    """First iteration: improved is None, baseline below target — fire the loop."""
    assert _should_reiterate(None, _verif(0.3, 0.7), 0, 2) is True


def test_baseline_meets_target_with_no_improved_returns_false():
    """Baseline already at/above target — skip improvements entirely."""
    assert _should_reiterate(None, _verif(0.8, 0.7), 0, 2) is False


def test_improved_overrides_baseline_when_present():
    """Once improved exists, it is authoritative — baseline is ignored."""
    # Improved met target → stop (even if baseline didn't).
    assert _should_reiterate(_verif(0.9, 0.7), _verif(0.3, 0.7), 1, 2) is False
    # Improved below target → continue (even if baseline met).
    assert _should_reiterate(_verif(0.3, 0.7), _verif(0.9, 0.7), 1, 2) is True


def test_cap_bounds_termination():
    """Iteration cap reached → False, regardless of verification state."""
    assert _should_reiterate(None, _verif(0.3, 0.7), 2, 2) is False
    assert _should_reiterate(_verif(0.3, 0.7), _verif(0.3, 0.7), 2, 2) is False
    # Cap of 0 disables the loop entirely.
    assert _should_reiterate(None, _verif(0.3, 0.7), 0, 0) is False
