"""Tests for backend.agents.rlm.forced_iteration — Lane H.

The policy is tested at two levels:

1. ``ForcedIterationPolicy.should_refuse`` in isolation — pure logic, no
   rlm patching. Covers every branch of the decision tree.
2. End-to-end through the patched ``LocalRepl._final_var`` — verifies the
   interceptor actually returns a "no final answer" string in the form
   ``rlm.utils.parsing.find_final_answer`` recognizes as "keep going".
"""

from __future__ import annotations

from typing import Any

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
    refusals: list[str] | None = None,
) -> ForcedIterationPolicy:
    """Factory for a policy whose rubric snapshot is fixed."""
    captured: list[str] = refusals if refusals is not None else []
    return ForcedIterationPolicy(
        min_iterations=min_iterations,
        rubric_snapshot=lambda: (score, target, iteration),
        current_iteration=lambda: iteration,
        remaining_s=lambda: remaining_s,
        on_refusal=lambda msg: captured.append(msg),
    )


# --- ForcedIterationPolicy.should_refuse — direct unit tests ---


def test_score_above_target_accepts() -> None:
    """Score satisfies target → accept FINAL_VAR."""
    policy = _make_policy(score=0.85, target=0.7, iteration=1, min_iterations=2)
    refuse, msg = policy.should_refuse()
    assert refuse is False
    assert msg is None


def test_score_equal_to_target_accepts() -> None:
    """Score == target counts as satisfied → accept."""
    policy = _make_policy(score=0.7, target=0.7, iteration=1, min_iterations=2)
    refuse, _msg = policy.should_refuse()
    assert refuse is False


def test_below_target_below_min_iterations_refuses() -> None:
    """Below target AND below floor → refuse with an actionable message."""
    refusals: list[str] = []
    policy = _make_policy(
        score=0.3, target=0.7, iteration=1, min_iterations=2,
        refusals=refusals,
    )
    refuse, msg = policy.should_refuse()
    assert refuse is True
    assert msg is not None
    # Message must include the key numbers + a "what to do next" instruction.
    assert "0.300" in msg or "0.30" in msg or "score=0.3" in msg
    assert "0.700" in msg or "0.70" in msg or "target_score=0.7" in msg
    assert "propose_improvements" in msg
    assert "implement_baseline" in msg


def test_below_target_above_min_iterations_accepts() -> None:
    """Below target but already hit the floor → accept (best-effort exit)."""
    policy = _make_policy(score=0.3, target=0.7, iteration=2, min_iterations=2)
    refuse, _msg = policy.should_refuse()
    assert refuse is False


def test_min_iterations_zero_disables_policy() -> None:
    """min_iterations=0 → policy disabled; any FINAL_VAR accepted."""
    policy = _make_policy(score=0.1, target=0.9, iteration=0, min_iterations=0)
    refuse, _msg = policy.should_refuse()
    assert refuse is False


def test_no_rubric_data_yet_refuses_below_floor() -> None:
    """BUG-LR-013: no verify_against_rubric call + below iteration floor → refuse.

    A model that hasn't scored at all has done strictly less work than one that
    scored 0.0 — so the floor should block it too.
    """
    policy = _make_policy(score=None, target=None, iteration=1, min_iterations=2)
    refuse, msg = policy.should_refuse()
    assert refuse is True
    assert msg is not None
    # Message should guide the model to call verify_against_rubric.
    assert "rubric" in msg.lower() or "verify" in msg.lower()


def test_no_rubric_data_accepts_when_floor_met() -> None:
    """score=None but floor already met → accept (policy can't force work retroactively)."""
    policy = _make_policy(score=None, target=None, iteration=2, min_iterations=2)
    refuse, _msg = policy.should_refuse()
    assert refuse is False


def test_wall_clock_floor_bypasses_policy() -> None:
    """Less than 60s remaining → accept (better partial than no report)."""
    # Score is below target AND below the iteration floor — would normally refuse.
    policy = _make_policy(
        score=0.1, target=0.9, iteration=0, min_iterations=2,
        remaining_s=30.0,  # below the 60s floor
    )
    refuse, _msg = policy.should_refuse()
    assert refuse is False


def test_wall_clock_floor_at_exactly_60s_bypasses() -> None:
    """remaining_s == 60.0 hits the floor — bypass."""
    policy = _make_policy(
        score=0.1, target=0.9, iteration=0, min_iterations=2,
        remaining_s=60.0,
    )
    refuse, _msg = policy.should_refuse()
    assert refuse is False


def test_wall_clock_just_above_floor_still_refuses() -> None:
    """remaining_s > 60 → policy applies normally."""
    policy = _make_policy(
        score=0.1, target=0.9, iteration=0, min_iterations=2,
        remaining_s=120.0,
    )
    refuse, _msg = policy.should_refuse()
    assert refuse is True


def test_no_deadline_means_no_wall_clock_bypass() -> None:
    """remaining_s=None → no deadline configured; policy applies normally."""
    policy = _make_policy(
        score=0.1, target=0.9, iteration=0, min_iterations=2,
        remaining_s=None,
    )
    refuse, _msg = policy.should_refuse()
    assert refuse is True


def test_refusal_count_caps_at_max() -> None:
    """A stubborn root that keeps calling FINAL_VAR still terminates eventually."""
    policy = _make_policy(score=0.1, target=0.9, iteration=0, min_iterations=2)
    # Drive refusal_count up to the bound.
    from backend.agents.rlm.forced_iteration import _MAX_REFUSALS_PER_RUN
    policy.refusal_count = _MAX_REFUSALS_PER_RUN
    refuse, _msg = policy.should_refuse()
    assert refuse is False  # capped — root can finally ship


# --- End-to-end via the patched LocalRepl._final_var ---


def _make_local_repl() -> Any:
    """Construct a bare LocalREPL with no LM handler.

    The interceptor lives on ``LocalREPL._final_var``; we don't need a real
    LLM client to exercise it, only an instance whose ``_final_var`` we can
    call.
    """
    from rlm.environments.local_repl import LocalREPL
    repl = LocalREPL()
    return repl


def test_patched_final_var_passes_through_when_no_policy() -> None:
    """No policy on the stack → original _final_var behavior."""
    repl = _make_local_repl()
    # Seed a variable so FINAL_VAR finds it.
    repl.locals["answer"] = "42"
    out = repl._final_var("answer")
    assert out == "42"
    assert repl._last_final_answer == "42"


def test_patched_final_var_blocks_when_policy_refuses() -> None:
    """Active policy refuses → returns a "variable not found / FINAL_VAR" string
    in the exact shape rlm.utils.parsing.find_final_answer treats as no answer.
    """
    from rlm.utils.parsing import find_final_answer

    repl = _make_local_repl()
    repl.locals["report"] = "{'score': 0.1}"
    captured: list[str] = []
    policy = _make_policy(
        score=0.1, target=0.9, iteration=0, min_iterations=2,
        refusals=captured,
    )

    with forced_iteration_policy(policy):
        out = repl._final_var("report")

    # The interceptor must have:
    #   1. invoked the on_refusal callback (run_warning surface),
    #   2. returned a string containing the three substrings find_final_answer
    #      treats as "keep going",
    #   3. NOT set _last_final_answer (so rlm's per-block consumer doesn't pick
    #      it up as a final answer).
    assert len(captured) == 1
    assert "Variable '" in out
    assert "' not found" in out
    assert "FINAL_VAR" in out
    assert repl._last_final_answer is None

    # And: find_final_answer must agree that this is NOT a final answer.
    # Simulate the rlm code path: feed the interpreter a string that begins
    # with FINAL_VAR(report) and ask it to resolve.
    # The fallback path: when env.execute_code returns our error string,
    # find_final_answer should return None.
    class _FakeEnv:
        def execute_code(self, code: str) -> Any:
            class _R:
                stdout = out
            return _R()
    assert find_final_answer("FINAL_VAR(report)", environment=_FakeEnv()) is None


def test_patched_final_var_accepts_when_score_above_target() -> None:
    """Score >= target → policy accepts even if iteration floor not hit."""
    repl = _make_local_repl()
    repl.locals["report"] = "{'score': 0.95}"
    policy = _make_policy(score=0.95, target=0.7, iteration=1, min_iterations=2)

    with forced_iteration_policy(policy):
        out = repl._final_var("report")

    assert out == "{'score': 0.95}"
    assert repl._last_final_answer == "{'score': 0.95}"


def test_patched_final_var_accepts_under_wall_clock_floor() -> None:
    """Wall-clock <= 60s → accept the partial result."""
    repl = _make_local_repl()
    repl.locals["report"] = "{'score': 0.1}"
    policy = _make_policy(
        score=0.1, target=0.9, iteration=0, min_iterations=2,
        remaining_s=30.0,  # below floor
    )

    with forced_iteration_policy(policy):
        out = repl._final_var("report")

    assert out == "{'score': 0.1}"
    assert repl._last_final_answer == "{'score': 0.1}"


def test_policy_disabled_via_min_iterations_zero() -> None:
    """min_iterations=0 → policy disabled; FINAL_VAR always accepted."""
    repl = _make_local_repl()
    repl.locals["report"] = "{'score': 0.1}"
    policy = _make_policy(
        score=0.1, target=0.9, iteration=0, min_iterations=0,
    )

    with forced_iteration_policy(policy):
        out = repl._final_var("report")

    assert out == "{'score': 0.1}"


def test_policy_stack_pops_on_exit() -> None:
    """Exiting the context manager removes the policy; outside the block,
    _final_var falls back to the original behavior.
    """
    repl = _make_local_repl()
    repl.locals["answer"] = "outside"
    policy = _make_policy(score=0.1, target=0.9, iteration=0, min_iterations=2)

    with forced_iteration_policy(policy):
        # Inside the block — would refuse, but we only want to verify the
        # stack semantics; check that the policy is in effect.
        out_inside = repl._final_var("answer")
        assert "Variable '" in out_inside  # blocked

    # Outside the block — back to original behavior.
    out_outside = repl._final_var("answer")
    assert out_outside == "outside"


def test_on_refusal_exception_does_not_break_interceptor() -> None:
    """A raise inside on_refusal must not propagate — interceptor still returns."""
    repl = _make_local_repl()
    repl.locals["report"] = "{}"

    def _bad_callback(msg: str) -> None:
        raise RuntimeError("emit broke")

    policy = ForcedIterationPolicy(
        min_iterations=2,
        rubric_snapshot=lambda: (0.1, 0.9, 0),
        current_iteration=lambda: 0,
        remaining_s=lambda: 3600.0,
        on_refusal=_bad_callback,
    )

    with forced_iteration_policy(policy):
        out = repl._final_var("report")  # must not raise

    assert "Variable '" in out
    assert "FINAL_VAR" in out


# ---------------------------------------------------------------------------
# Lane O — refuse FINAL_VAR when iteration floor is met but no candidate was
# honestly tested (the 2026-05-25 Adam blanket-decline regression)
# ---------------------------------------------------------------------------


def test_lane_o_blanket_decline_refuses_final_var() -> None:
    """Iteration floor reached + score < target + no honest outcomes → refuse.

    The 2026-05-25 Adam regression: agent reached iter 2 (= min_iterations),
    called propose_improvements, then blanket-declined all 3 candidates in
    a for-loop without running any, then FINAL_VAR'd with overall=0/0.6.
    The new check must refuse this exact shape.
    """
    refusals: list[str] = []
    policy = ForcedIterationPolicy(
        min_iterations=2,
        rubric_snapshot=lambda: (0.0, 0.6, 2),
        current_iteration=lambda: 2,
        remaining_s=lambda: 3600.0,
        on_refusal=lambda m: refusals.append(m),
        honest_candidate_outcomes=lambda: 0,  # ← all 3 declined, zero honest
    )
    refuse, msg = policy.should_refuse()
    assert refuse is True
    assert msg is not None
    assert "honestly tested" in msg
    assert "blanket-decl" in msg.lower() or "observer bias" in msg.lower()
    # Pick one + run it guidance
    assert "implement_baseline" in msg
    assert "record_candidate_outcome" in msg


def test_lane_o_at_least_one_tested_allows_final_var() -> None:
    """Iteration floor met + at least one honest outcome → accept.
    Even if rubric is below target — the agent honestly tried."""
    policy = ForcedIterationPolicy(
        min_iterations=2,
        rubric_snapshot=lambda: (0.3, 0.6, 2),
        current_iteration=lambda: 2,
        remaining_s=lambda: 3600.0,
        on_refusal=lambda m: None,
        honest_candidate_outcomes=lambda: 1,  # one candidate honestly tested
    )
    refuse, _msg = policy.should_refuse()
    assert refuse is False


def test_lane_o_marginal_counts_as_honest() -> None:
    """'marginal' is a truthful outcome — counts toward honest tested."""
    policy = ForcedIterationPolicy(
        min_iterations=2,
        rubric_snapshot=lambda: (0.3, 0.6, 2),
        current_iteration=lambda: 2,
        remaining_s=lambda: 3600.0,
        on_refusal=lambda m: None,
        honest_candidate_outcomes=lambda: 1,
    )
    refuse, _msg = policy.should_refuse()
    assert refuse is False


def test_lane_o_below_floor_uses_floor_message_not_blanket_decline() -> None:
    """Below iteration floor — older check (#4) wins; lane O check doesn't fire."""
    policy = ForcedIterationPolicy(
        min_iterations=2,
        rubric_snapshot=lambda: (0.3, 0.6, 1),
        current_iteration=lambda: 1,  # below floor
        remaining_s=lambda: 3600.0,
        on_refusal=lambda m: None,
        honest_candidate_outcomes=lambda: 0,
    )
    refuse, msg = policy.should_refuse()
    assert refuse is True
    assert msg is not None
    # Should be the OLDER message, not the Lane O one.
    assert "min_rubric_iterations" in msg


def test_lane_o_callable_unset_means_disabled() -> None:
    """When honest_candidate_outcomes=None, the lane O check doesn't fire —
    back-compat with v1 callers that didn't supply this field."""
    policy = ForcedIterationPolicy(
        min_iterations=2,
        rubric_snapshot=lambda: (0.0, 0.6, 2),
        current_iteration=lambda: 2,
        remaining_s=lambda: 3600.0,
        on_refusal=lambda m: None,
        honest_candidate_outcomes=None,  # disabled
    )
    refuse, _msg = policy.should_refuse()
    assert refuse is False


def test_lane_o_callable_raising_is_failsoft() -> None:
    """If the honest_candidate_outcomes callable raises, treat as 0
    (refuse for safety — the agent shouldn't shortcut on a buggy callable)."""
    def _bad_counter() -> int:
        raise RuntimeError("dashboard_events.jsonl read failed")
    policy = ForcedIterationPolicy(
        min_iterations=2,
        rubric_snapshot=lambda: (0.0, 0.6, 2),
        current_iteration=lambda: 2,
        remaining_s=lambda: 3600.0,
        on_refusal=lambda m: None,
        honest_candidate_outcomes=_bad_counter,
    )
    refuse, _msg = policy.should_refuse()
    # 0 from fail-soft → refuse, same as if no candidates tested.
    assert refuse is True


def test_lane_o_score_above_target_skips_check() -> None:
    """When rubric is satisfied, Lane O doesn't fire even if 0 honest outcomes."""
    policy = ForcedIterationPolicy(
        min_iterations=2,
        rubric_snapshot=lambda: (0.85, 0.6, 2),  # SATISFIED
        current_iteration=lambda: 2,
        remaining_s=lambda: 3600.0,
        on_refusal=lambda m: None,
        honest_candidate_outcomes=lambda: 0,
    )
    refuse, _msg = policy.should_refuse()
    assert refuse is False


# --- thread-locality regression (2026-05-31) ---------------------------------
# The policy stack is thread-local and the FINAL_VAR interceptor runs on the
# thread that runs rlm.completion. run.py dispatches completion via
# asyncio.to_thread (a SEPARATE worker thread), so the policy MUST be entered
# INSIDE the to_thread callable. Entering it on the asyncio loop thread (as
# run.py did until 2026-05-31) left _current_policy() empty on the worker
# thread → the entire premature-exit guard silently no-op'd.


def test_policy_invisible_across_to_thread_when_entered_on_loop_thread() -> None:
    """ROOT-CAUSE DOC: a policy pushed on the loop thread is NOT visible to a
    function dispatched via asyncio.to_thread (worker thread)."""
    import asyncio

    from backend.agents.rlm.forced_iteration import _current_policy

    policy = _make_policy(score=0.0, target=0.6, iteration=1)

    async def _run() -> Any:
        with forced_iteration_policy(policy):
            return await asyncio.to_thread(_current_policy)

    assert asyncio.run(_run()) is None  # the bug: worker sees no policy


def test_policy_visible_across_to_thread_when_entered_inside_worker() -> None:
    """THE FIX: entering the context manager INSIDE the to_thread callable makes
    the policy visible on the worker thread (where LocalREPL._final_var runs)."""
    import asyncio

    from backend.agents.rlm.forced_iteration import _current_policy

    policy = _make_policy(score=0.0, target=0.6, iteration=1)

    def _worker() -> Any:
        with forced_iteration_policy(policy):
            return _current_policy()

    async def _run() -> Any:
        return await asyncio.to_thread(_worker)

    assert asyncio.run(_run()) is policy
