"""Tests for PR-ι.1 — per-run iteration budget cap.

Covers:
1. Budget cap accepts FINAL_VAR when current_iteration >= max_rlm_iterations.
2. Budget cap does NOT refuse when current_iteration < max_rlm_iterations.
3. Budget cap overrides the rubric-floor refusal (score below target).
4. Budget cap overrides the repair-floor refusal.
5. Budget cap emits the correct SSE code (iteration_budget_exceeded).
6. Wall-clock floor takes priority over budget cap when nearly timed out.
7. max_rlm_iterations=None disables the cap (env var absent).
8. Cap read from env var when not set on policy object.
9. Cap set to 0 on object is ignored (0 means "disabled").
10. Budget cap fires correctly at exactly N-1 and N iterations.
"""

from __future__ import annotations

import os

import pytest

from backend.agents.rlm.forced_iteration import (
    ForcedIterationPolicy,
    apply_forced_iteration_patch,
    forced_iteration_policy,
)

apply_forced_iteration_patch()


def _make_policy(
    *,
    score: float | None = 0.2,
    target: float | None = 0.7,
    iteration: int = 3,
    min_iterations: int = 2,
    remaining_s: float | None = 3600.0,
    max_rlm_iterations: int | None = None,
    refusals: list[str] | None = None,
    budget_msgs: list[str] | None = None,
) -> ForcedIterationPolicy:
    captured: list[str] = refusals if refusals is not None else []
    budget_captured: list[str] = budget_msgs if budget_msgs is not None else []
    _p = ForcedIterationPolicy(
        min_iterations=min_iterations,
        rubric_snapshot=lambda: (score, target, iteration),
        current_iteration=lambda: iteration,
        remaining_s=lambda: remaining_s,
        on_refusal=lambda msg: captured.append(msg),
        max_rlm_iterations=max_rlm_iterations,
        on_budget_exceeded=lambda msg: budget_captured.append(msg),
    )
    # BUG-NEW-046 (ported 2026-06-09): a policy with zero run_experiment calls
    # now refuses FINAL_VAR unconditionally. These tests exercise OTHER policy
    # dimensions, so mark one experiment as done (and advance the iteration so
    # per-iteration repair state stays clean).
    _p.record_run_experiment("ok")
    _p.on_iteration_advance()
    return _p


# --- 1. Budget cap accepts at limit ---

def test_budget_cap_accepts_at_max_iterations() -> None:
    """When current_iteration >= max_rlm_iterations, FINAL_VAR is accepted."""
    policy = _make_policy(iteration=5, max_rlm_iterations=5)
    refuse, msg = policy.should_refuse()
    assert refuse is False
    assert msg is None


def test_budget_cap_accepts_past_max_iterations() -> None:
    """When current_iteration > max_rlm_iterations, FINAL_VAR is accepted."""
    policy = _make_policy(iteration=7, max_rlm_iterations=5)
    refuse, msg = policy.should_refuse()
    assert refuse is False
    assert msg is None


# --- 2. Budget cap does not activate below limit ---

def test_budget_cap_does_not_activate_below_max() -> None:
    """When current_iteration < max_rlm_iterations, normal policy applies (refuses)."""
    policy = _make_policy(
        iteration=2,
        min_iterations=4,
        max_rlm_iterations=10,
        score=0.2,
        target=0.7,
    )
    refuse, msg = policy.should_refuse()
    # rubric-floor refusal should fire because iter(2) < min_iterations(4)
    assert refuse is True
    assert msg is not None


# --- 3. Budget cap overrides rubric-floor refusal ---

def test_budget_cap_overrides_rubric_floor() -> None:
    """Budget cap accepts FINAL_VAR even when rubric score is below target."""
    policy = _make_policy(
        iteration=5,
        min_iterations=2,
        max_rlm_iterations=5,
        score=0.1,   # well below target
        target=0.7,
    )
    refuse, msg = policy.should_refuse()
    assert refuse is False


# --- 4. Budget cap overrides repair-floor refusal ---

def test_budget_cap_overrides_repair_floor() -> None:
    """Budget cap accepts even when repair floor would refuse."""
    policy = _make_policy(
        iteration=5,
        min_iterations=2,
        max_rlm_iterations=5,
        score=0.1,
        target=0.7,
    )
    policy.record_repair_attempt("code_bug")
    # repair floor wants > 2 repair attempts; budget cap wins
    refuse, msg = policy.should_refuse()
    assert refuse is False


# --- 5. Budget cap emits the correct SSE code ---

def test_budget_cap_emits_iteration_budget_exceeded() -> None:
    """on_budget_exceeded is called (not on_refusal) when budget is exhausted."""
    budget_msgs: list[str] = []
    refusals: list[str] = []
    policy = _make_policy(
        iteration=5,
        max_rlm_iterations=5,
        refusals=refusals,
        budget_msgs=budget_msgs,
    )
    refuse, msg = policy.should_refuse()
    assert refuse is False
    # The budget callback should be called with the exceeded message.
    assert len(budget_msgs) == 1
    assert "iteration_budget_exceeded" in budget_msgs[0].lower() or "budget" in budget_msgs[0].lower()
    # The normal on_refusal callback should NOT be called.
    assert len(refusals) == 0


# --- 6. Wall-clock floor takes priority ---

def test_wall_clock_floor_beats_budget_cap() -> None:
    """Wall-clock floor (< 60s remaining) bypasses even budget-cap logic."""
    policy = _make_policy(
        iteration=5,
        max_rlm_iterations=5,
        remaining_s=30.0,  # below _WALL_CLOCK_FLOOR_S = 60.0
    )
    budget_msgs: list[str] = []
    policy.on_budget_exceeded = lambda msg: budget_msgs.append(msg)
    refuse, msg = policy.should_refuse()
    # Wall-clock floor accepts without invoking budget callback.
    assert refuse is False
    # Wall-clock check fires FIRST, so budget callback should not be called.
    assert len(budget_msgs) == 0


# --- 7. max_rlm_iterations=None disables cap ---

def test_none_max_iterations_disables_cap(monkeypatch) -> None:
    """When max_rlm_iterations is None and env var absent, cap is disabled."""
    monkeypatch.delenv("REPROLAB_MAX_RLM_ITERATIONS", raising=False)
    policy = _make_policy(
        iteration=100,
        max_rlm_iterations=None,
        min_iterations=2,
        score=0.9,
        target=0.7,
    )
    # Score satisfies target → accepted (not budget cap)
    refuse, msg = policy.should_refuse()
    assert refuse is False


# --- 8. Cap read from env var ---

def test_cap_read_from_env_var(monkeypatch) -> None:
    """When max_rlm_iterations is None on the object, env var is consulted."""
    monkeypatch.setenv("REPROLAB_MAX_RLM_ITERATIONS", "3")
    policy = _make_policy(
        iteration=3,
        max_rlm_iterations=None,  # should fall back to env var
        score=0.1,
        target=0.7,
    )
    refuse, msg = policy.should_refuse()
    assert refuse is False  # budget cap accepts at iteration=3 == max=3


def test_env_var_not_active_below_limit(monkeypatch) -> None:
    """Env-var cap does not fire when current_iteration < env-var cap."""
    monkeypatch.setenv("REPROLAB_MAX_RLM_ITERATIONS", "10")
    policy = _make_policy(
        iteration=2,
        max_rlm_iterations=None,
        min_iterations=4,  # rubric floor will refuse
        score=0.1,
        target=0.7,
    )
    refuse, msg = policy.should_refuse()
    # rubric floor refuses; budget cap inactive
    assert refuse is True


# --- 9. Cap=0 on object is disabled ---

def test_zero_max_iterations_disables_cap(monkeypatch) -> None:
    """max_rlm_iterations=0 disables the cap (0 means 'no cap')."""
    monkeypatch.delenv("REPROLAB_MAX_RLM_ITERATIONS", raising=False)
    policy = _make_policy(
        iteration=0,
        max_rlm_iterations=0,
        score=0.9,
        target=0.7,
    )
    # Score satisfies target → accepted; no budget cap interference.
    refuse, msg = policy.should_refuse()
    assert refuse is False


# --- 10. Boundary: exactly N-1 vs N ---

def test_boundary_n_minus_one_does_not_cap() -> None:
    """At iteration N-1, the budget cap does not yet fire."""
    policy = _make_policy(
        iteration=4,   # N-1 of 5
        max_rlm_iterations=5,
        min_iterations=2,
        score=0.9,
        target=0.7,
    )
    # Score above target → accepted (not due to budget cap)
    refuse, msg = policy.should_refuse()
    assert refuse is False


def test_boundary_n_fires_cap() -> None:
    """At iteration N, the budget cap fires and accepts FINAL_VAR."""
    budget_msgs: list[str] = []
    policy = _make_policy(
        iteration=5,   # exactly N
        max_rlm_iterations=5,
        score=0.1,
        target=0.7,
        budget_msgs=budget_msgs,
    )
    refuse, msg = policy.should_refuse()
    assert refuse is False
    assert len(budget_msgs) == 1
