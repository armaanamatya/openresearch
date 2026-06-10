"""Tests for the PR-α followup repair-iteration floor in ForcedIterationPolicy.

The repair-iteration floor refuses FINAL_VAR when the last run_experiment
returned a repairable outcome AND fewer than REPROLAB_MIN_REPAIR_ITERATIONS
repair attempts have been made.

Scenarios tested:

1. record_repair_attempt once + should_refuse → (True, "forced_repair_iteration")
2. After N=MIN_REPAIR calls → (False, None)
3. remaining_s=30 (below floor) → (False, None) regardless of repair state
4. No record_repair_attempt called → default behavior unchanged
5. REPROLAB_MIN_REPAIR_ITERATIONS=0 → repair policy disabled entirely
6. min_iterations=0 (rubric policy disabled) but repair still fires
7. No rubric data yet but repair fires
8. on_repair_refusal callback invoked with correct SSE code when set
9. Repair check fires independently of rubric score gap
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest

from backend.agents.rlm.forced_iteration import (
    ForcedIterationPolicy,
    apply_forced_iteration_patch,
    forced_iteration_policy,
)

# Ensure the patch is installed once for end-to-end checks.
apply_forced_iteration_patch()


def _make_policy(
    *,
    score: float | None,
    target: float | None,
    iteration: int,
    min_iterations: int = 2,
    remaining_s: float | None = 3600.0,
    on_repair_refusal: Any = None,
) -> ForcedIterationPolicy:
    refusals: list[str] = []
    _p = ForcedIterationPolicy(
        min_iterations=min_iterations,
        rubric_snapshot=lambda: (score, target, iteration),
        current_iteration=lambda: iteration,
        remaining_s=lambda: remaining_s,
        on_refusal=lambda msg: refusals.append(msg),
        on_repair_refusal=on_repair_refusal,
    )
    # BUG-NEW-046 (ported 2026-06-09): a policy with zero run_experiment calls
    # now refuses FINAL_VAR unconditionally. These tests exercise OTHER policy
    # dimensions, so mark one experiment as done (and advance the iteration so
    # per-iteration repair state stays clean).
    _p.record_run_experiment("ok")
    _p.on_iteration_advance()
    return _p


# -----------------------------------------------------------------------
# 1. record_repair_attempt once → should_refuse returns True
# -----------------------------------------------------------------------

def test_one_repair_attempt_refuses_final_var() -> None:
    """Single repairable outcome recorded; repair floor default=2 → refuse."""
    policy = _make_policy(score=0.0, target=0.6, iteration=2, min_iterations=2)
    policy.record_repair_attempt("preflight_blocked")

    refuse, msg = policy.should_refuse()

    assert refuse is True
    assert msg is not None
    assert "repairable outcome" in msg
    assert "preflight_blocked" in msg
    assert "1/2 repair iterations" in msg
    assert "implement_baseline" in msg
    assert "run_experiment" in msg


# -----------------------------------------------------------------------
# 2. After MIN_REPAIR calls → should_refuse returns False
# -----------------------------------------------------------------------

def test_after_min_repair_calls_accepts() -> None:
    """Once repair_iter_count reaches MIN, FINAL_VAR is accepted."""
    with patch.dict(os.environ, {"REPROLAB_MIN_REPAIR_ITERATIONS": "2"}):
        policy = _make_policy(score=0.0, target=0.6, iteration=2, min_iterations=2)
        policy.record_repair_attempt("preflight_blocked")
        policy.record_repair_attempt("preflight_blocked")

        refuse, msg = policy.should_refuse()

    assert refuse is False
    assert msg is None


def test_exactly_min_repair_calls_accepts() -> None:
    """repair_iter_count == MIN_REPAIR → accept (floor reached)."""
    with patch.dict(os.environ, {"REPROLAB_MIN_REPAIR_ITERATIONS": "3"}):
        policy = _make_policy(score=0.0, target=0.6, iteration=2, min_iterations=2)
        for _ in range(3):
            policy.record_repair_attempt("code_error")

        refuse, _ = policy.should_refuse()

    assert refuse is False


# -----------------------------------------------------------------------
# 3. Wall-clock bypass at 30s → accept regardless of repair state
# -----------------------------------------------------------------------

def test_wall_clock_floor_bypasses_repair_check() -> None:
    """remaining_s=30 → wall-clock floor fires before repair check → accept."""
    policy = _make_policy(
        score=0.0, target=0.6, iteration=2, min_iterations=2, remaining_s=30.0,
    )
    policy.record_repair_attempt("preflight_blocked")

    refuse, msg = policy.should_refuse()

    assert refuse is False
    assert msg is None


def test_wall_clock_exactly_60s_bypasses_repair_check() -> None:
    """remaining_s == 60.0 is at the floor → bypass."""
    policy = _make_policy(
        score=0.0, target=0.6, iteration=2, min_iterations=2, remaining_s=60.0,
    )
    policy.record_repair_attempt("preflight_blocked")

    refuse, _ = policy.should_refuse()

    assert refuse is False


def test_wall_clock_just_above_floor_repair_refuses() -> None:
    """remaining_s=61 → above floor → repair check applies."""
    with patch.dict(os.environ, {"REPROLAB_MIN_REPAIR_ITERATIONS": "2"}):
        policy = _make_policy(
            score=0.0, target=0.6, iteration=2, min_iterations=2, remaining_s=61.0,
        )
        policy.record_repair_attempt("preflight_blocked")

        refuse, _ = policy.should_refuse()

    assert refuse is True


# -----------------------------------------------------------------------
# 4. No record_repair_attempt → existing behavior unchanged
# -----------------------------------------------------------------------

def test_no_repair_attempt_default_behavior_accepts_at_floor() -> None:
    """No repair attempt recorded; iteration floor met → accept (unchanged behavior)."""
    policy = _make_policy(score=0.3, target=0.6, iteration=2, min_iterations=2)
    # No record_repair_attempt call.

    refuse, _ = policy.should_refuse()

    # Lane O: honest_candidate_outcomes is None by default → skip that check.
    # No repair recorded → accept.
    assert refuse is False


def test_no_repair_attempt_below_floor_still_refuses_rubric() -> None:
    """No repair attempt; below rubric floor → existing rubric refusal still fires."""
    policy = _make_policy(score=0.3, target=0.6, iteration=1, min_iterations=2)

    refuse, msg = policy.should_refuse()

    assert refuse is True
    assert msg is not None
    assert "min_rubric_iterations" in msg


# -----------------------------------------------------------------------
# 5. REPROLAB_MIN_REPAIR_ITERATIONS=0 disables repair policy
# -----------------------------------------------------------------------

def test_min_repair_zero_disables_repair_policy() -> None:
    """REPROLAB_MIN_REPAIR_ITERATIONS=0 → repair policy disabled; FINAL_VAR accepted."""
    with patch.dict(os.environ, {"REPROLAB_MIN_REPAIR_ITERATIONS": "0"}):
        policy = _make_policy(score=0.0, target=0.6, iteration=2, min_iterations=2)
        policy.record_repair_attempt("preflight_blocked")

        refuse, _ = policy.should_refuse()

    assert refuse is False


# -----------------------------------------------------------------------
# 6. Rubric policy disabled (min_iterations=0) but repair still fires
# -----------------------------------------------------------------------

def test_repair_fires_even_when_rubric_policy_disabled() -> None:
    """min_iterations=0 disables rubric checks; repair floor still enforced."""
    with patch.dict(os.environ, {"REPROLAB_MIN_REPAIR_ITERATIONS": "2"}):
        policy = _make_policy(
            score=0.0, target=0.6, iteration=2, min_iterations=0,  # rubric disabled
        )
        policy.record_repair_attempt("preflight_blocked")

        refuse, msg = policy.should_refuse()

    assert refuse is True
    assert msg is not None
    assert "repairable outcome" in msg


# -----------------------------------------------------------------------
# 7. No rubric data yet but repair fires
# -----------------------------------------------------------------------

def test_repair_fires_when_no_rubric_data() -> None:
    """No verify_against_rubric yet (score=None) but repair attempt recorded → refuse."""
    with patch.dict(os.environ, {"REPROLAB_MIN_REPAIR_ITERATIONS": "2"}):
        policy = _make_policy(
            score=None, target=None, iteration=1, min_iterations=2,
        )
        policy.record_repair_attempt("preflight_blocked")

        refuse, msg = policy.should_refuse()

    assert refuse is True
    assert msg is not None
    assert "repairable outcome" in msg


# -----------------------------------------------------------------------
# 8. on_repair_refusal callback — SSE event routing
# -----------------------------------------------------------------------

def test_on_repair_refusal_callback_receives_message() -> None:
    """When on_repair_refusal is set, it receives the refusal message."""
    repair_msgs: list[str] = []
    policy = _make_policy(
        score=0.0, target=0.6, iteration=2, min_iterations=2,
        on_repair_refusal=lambda m: repair_msgs.append(m),
    )
    policy.record_repair_attempt("preflight_blocked")

    refuse, msg = policy.should_refuse()
    assert refuse is True

    # Simulate interceptor routing
    assert policy._pending_refusal_code == "forced_repair_iteration"


def test_pending_refusal_code_is_forced_repair_iteration() -> None:
    """should_refuse sets _pending_refusal_code='forced_repair_iteration' on repair refusal."""
    with patch.dict(os.environ, {"REPROLAB_MIN_REPAIR_ITERATIONS": "2"}):
        policy = _make_policy(score=0.0, target=0.6, iteration=2, min_iterations=2)
        policy.record_repair_attempt("preflight_blocked")

        refuse, _ = policy.should_refuse()

    assert refuse is True
    assert policy._pending_refusal_code == "forced_repair_iteration"


def test_pending_refusal_code_is_forced_iteration_on_rubric_refusal() -> None:
    """Rubric-floor refusal sets _pending_refusal_code='forced_iteration' (default)."""
    with patch.dict(os.environ, {"REPROLAB_MIN_REPAIR_ITERATIONS": "2"}):
        policy = _make_policy(score=0.1, target=0.9, iteration=1, min_iterations=2)
        # No repair attempt.

        refuse, _ = policy.should_refuse()

    assert refuse is True
    # _pending_refusal_code is the default "forced_iteration" — not mutated
    assert policy._pending_refusal_code == "forced_iteration"


# -----------------------------------------------------------------------
# 9. Repair check fires even when rubric score is satisfied
#    (e.g. a stale high score from a previous candidate; last experiment failed)
#    Actually — by spec: score >= target → accept (step 3 wins). The repair
#    floor does NOT fire when the rubric is satisfied. Verify this.
# -----------------------------------------------------------------------

def test_repair_does_not_fire_when_rubric_satisfied() -> None:
    """Score >= target → rubric is satisfied; repair floor does NOT block."""
    with patch.dict(os.environ, {"REPROLAB_MIN_REPAIR_ITERATIONS": "2"}):
        policy = _make_policy(
            score=0.9, target=0.6, iteration=2, min_iterations=2,
        )
        policy.record_repair_attempt("preflight_blocked")

        refuse, _ = policy.should_refuse()

    # Rubric satisfied → accept (step 3 short-circuits before repair check)
    assert refuse is False


# -----------------------------------------------------------------------
# End-to-end: patched LocalREPL._final_var returns block string on repair refusal
# -----------------------------------------------------------------------

def test_patched_final_var_blocks_on_repair_refusal() -> None:
    """Patched _final_var returns the 'Variable not found / FINAL_VAR' shape
    when the repair policy refuses."""
    from rlm.utils.parsing import find_final_answer
    from rlm.environments.local_repl import LocalREPL

    repl = LocalREPL()
    repl.locals["report"] = "{'score': 0.0}"

    with patch.dict(os.environ, {"REPROLAB_MIN_REPAIR_ITERATIONS": "2"}):
        policy = _make_policy(score=0.0, target=0.6, iteration=2, min_iterations=2)
        policy.record_repair_attempt("preflight_blocked")

        with forced_iteration_policy(policy):
            out = repl._final_var("report")

    assert "Variable '" in out
    assert "' not found" in out
    assert "FINAL_VAR" in out
    assert "repairable outcome" in out

    class _FakeEnv:
        def execute_code(self, code: str) -> Any:
            class _R:
                stdout = out
            return _R()

    assert find_final_answer("FINAL_VAR(report)", environment=_FakeEnv()) is None


def test_patched_final_var_accepts_after_repair_floor_met() -> None:
    """Patched _final_var passes through when repair floor is satisfied."""
    from rlm.environments.local_repl import LocalREPL

    repl = LocalREPL()
    repl.locals["report"] = "{'score': 0.0}"

    with patch.dict(os.environ, {"REPROLAB_MIN_REPAIR_ITERATIONS": "2"}):
        policy = _make_policy(score=0.0, target=0.6, iteration=2, min_iterations=2)
        policy.record_repair_attempt("preflight_blocked")
        policy.record_repair_attempt("preflight_blocked")

        with forced_iteration_policy(policy):
            out = repl._final_var("report")

    assert out == "{'score': 0.0}"
