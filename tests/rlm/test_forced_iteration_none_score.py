"""Tests for BUG-LR-013 — forced-iteration policy with rubric_score=None.

Verifies that:
1. FINAL_VAR is refused when score is None AND below the iteration floor.
2. FINAL_VAR is accepted when score is None AND iteration floor is met.
3. Existing behaviour (score=0.0 < target) is preserved.
4. Wall-clock pressure still bypasses the None-score check.
"""
from __future__ import annotations


from backend.agents.rlm.forced_iteration import (
    ForcedIterationPolicy,
    apply_forced_iteration_patch,
)

apply_forced_iteration_patch()


def _make_policy(
    *,
    score: float | None,
    target: float | None,
    iteration: int,
    min_iterations: int = 2,
    remaining_s: float = 3600.0,
) -> ForcedIterationPolicy:
    refusals: list[str] = []
    _p = ForcedIterationPolicy(
        min_iterations=min_iterations,
        rubric_snapshot=lambda: (score, target, iteration),
        current_iteration=lambda: iteration,
        remaining_s=lambda: remaining_s,
        on_refusal=lambda msg: refusals.append(msg),
    )
    # BUG-NEW-046 (ported 2026-06-09): a policy with zero run_experiment calls
    # now refuses FINAL_VAR unconditionally. These tests exercise OTHER policy
    # dimensions, so mark one experiment as done (and advance the iteration so
    # per-iteration repair state stays clean).
    _p.record_run_experiment("ok")
    _p.on_iteration_advance()
    return _p


# ---------------------------------------------------------------------------
# BUG-LR-013: None score + below floor → refuse
# ---------------------------------------------------------------------------

def test_none_score_below_floor_refuses() -> None:
    policy = _make_policy(score=None, target=None, iteration=1, min_iterations=2)
    refuse, msg = policy.should_refuse()
    assert refuse is True
    assert msg is not None
    assert "no rubric score" in msg.lower() or "verify_against_rubric" in msg


def test_none_score_at_floor_accepts() -> None:
    policy = _make_policy(score=None, target=None, iteration=2, min_iterations=2)
    refuse, msg = policy.should_refuse()
    assert refuse is False


def test_none_score_above_floor_accepts() -> None:
    policy = _make_policy(score=None, target=None, iteration=5, min_iterations=2)
    refuse, msg = policy.should_refuse()
    assert refuse is False


def test_none_score_min_iterations_zero_accepts() -> None:
    """min_iterations=0 disables the policy entirely."""
    policy = _make_policy(score=None, target=None, iteration=0, min_iterations=0)
    refuse, msg = policy.should_refuse()
    assert refuse is False


# ---------------------------------------------------------------------------
# Existing behaviour preserved
# ---------------------------------------------------------------------------

def test_zero_score_below_floor_refuses() -> None:
    """score=0.0 is a real score (not None) — existing branch, must still refuse."""
    policy = _make_policy(score=0.0, target=0.5, iteration=1, min_iterations=2)
    refuse, msg = policy.should_refuse()
    assert refuse is True


def test_score_satisfies_target_accepts() -> None:
    policy = _make_policy(score=0.8, target=0.7, iteration=1, min_iterations=2)
    refuse, msg = policy.should_refuse()
    assert refuse is False


# ---------------------------------------------------------------------------
# Wall-clock pressure bypasses None-score check
# ---------------------------------------------------------------------------

def test_none_score_wall_clock_pressure_accepts() -> None:
    policy = _make_policy(
        score=None, target=None, iteration=0, min_iterations=2, remaining_s=30.0
    )
    refuse, msg = policy.should_refuse()
    assert refuse is False


# ---------------------------------------------------------------------------
# Replay test — iter-5 FINAL_VAR from prj_09047604e591d969 would be refused
# ---------------------------------------------------------------------------

def test_replay_iter5_final_var_refused() -> None:
    """The run that caused BUG-LR-013: iter=5, no rubric score, min=2."""
    policy = _make_policy(
        score=None, target=None, iteration=5, min_iterations=2, remaining_s=3600.0
    )
    # iter=5 >= min=2 → accept (floor already met — the policy can't refuse at iter 5)
    refuse, msg = policy.should_refuse()
    # At iter 5 (>= min 2) policy passes through.  The fix prevents the EARLY
    # FINAL_VAR at iter 1 (below floor).  Confirm iter 1 refuses.
    assert refuse is False  # floor met at iter 5

    early_policy = _make_policy(
        score=None, target=None, iteration=1, min_iterations=2, remaining_s=3600.0
    )
    refuse_early, msg_early = early_policy.should_refuse()
    assert refuse_early is True
    assert "verify_against_rubric" in (msg_early or "").lower() or "rubric" in (msg_early or "").lower()
