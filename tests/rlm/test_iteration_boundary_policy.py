"""ForcedIterationPolicy must refuse FINAL_VAR when the same iteration
contains TWO run_experiment calls with the second returning repairable/
partial_evidence/fatal — the 0.305 Adam anti-pattern."""
import pytest
from unittest.mock import MagicMock
from backend.agents.rlm.forced_iteration import ForcedIterationPolicy


def _policy(**overrides):
    defaults = dict(
        target_score=0.6,
        min_iterations=2,
        max_rlm_iterations=10,
        run_id="test-run",
        ctx=MagicMock(remaining_s=MagicMock(return_value=99999)),
    )
    defaults.update(overrides)
    return ForcedIterationPolicy(**defaults)


def test_refuses_final_var_when_two_run_experiments_with_repairable_latter():
    p = _policy()
    # Simulate: first run_experiment ok, second returned repairable
    p.record_run_experiment(outcome="ok")
    p.record_run_experiment(outcome="repairable")
    decision = p.should_refuse_final_var(current_score=0.8, iteration_count=1)
    assert decision.refuse is True
    assert "two run_experiment" in decision.reason.lower()


@pytest.mark.parametrize("second_outcome", ["repairable", "partial_evidence", "fatal"])
def test_refuses_on_any_failure_outcome_of_latter_experiment(second_outcome):
    p = _policy()
    p.record_run_experiment(outcome="ok")
    p.record_run_experiment(outcome=second_outcome)
    decision = p.should_refuse_final_var(current_score=0.9, iteration_count=1)
    assert decision.refuse is True


def test_allows_final_var_when_only_one_run_experiment_in_iteration():
    p = _policy()
    p.record_run_experiment(outcome="repairable")
    decision = p.should_refuse_final_var(current_score=0.8, iteration_count=2)
    assert decision.refuse is False


def test_allows_final_var_when_both_run_experiments_ok():
    p = _policy()
    p.record_run_experiment(outcome="ok")
    p.record_run_experiment(outcome="ok")
    decision = p.should_refuse_final_var(current_score=0.9, iteration_count=2)
    assert decision.refuse is False


def test_iteration_boundary_history_resets_on_iteration_advance():
    p = _policy()
    p.record_run_experiment(outcome="ok")
    p.record_run_experiment(outcome="repairable")
    p.on_iteration_advance()  # turn boundary
    # In a fresh iteration, history clean
    decision = p.should_refuse_final_var(current_score=0.9, iteration_count=2)
    assert decision.refuse is False


def test_logger_resets_two_experiment_tracker_at_real_turn_boundary():
    """F-06: on_iteration_advance must fire at each real REPL turn boundary
    (ReproLabRLMLogger.log), not only inside a FINAL_VAR refusal path.

    Without it, one failing run_experiment in two DIFFERENT iterations
    accumulates to len>=2 and falsely refuses the next legitimate FINAL_VAR —
    the same 'guard wired but silently misfiring across the boundary' class as
    the thread-local bug fixed in 6990d56.
    """
    from rlm.core.types import RLMIteration

    from backend.agents.rlm.sse_bridge import ReproLabRLMLogger

    p = _policy()

    class _Ctx:
        current_iteration = 0
        _forced_iteration_policy = p

    ctx = _Ctx()
    logger = ReproLabRLMLogger(emit=lambda _e: None, checkpointer=MagicMock(), ctx=ctx)

    def _iter() -> RLMIteration:
        return RLMIteration(
            prompt={"role": "user", "content": "reproduce the paper"},
            response="reasoning",
            code_blocks=[],
            final_answer=None,
            iteration_time=1.0,
        )

    # Three real turns, each with exactly ONE failing run_experiment — never
    # two-in-one-turn, so the guard must NOT refuse the FINAL_VAR in turn 3.
    p.record_run_experiment(outcome="repairable")
    logger.log(_iter())  # turn 1 boundary → reset
    p.record_run_experiment(outcome="repairable")
    logger.log(_iter())  # turn 2 boundary → reset
    p.record_run_experiment(outcome="repairable")  # turn 3, one experiment so far

    decision = p.should_refuse_final_var(current_score=0.9, iteration_count=3)
    assert decision.refuse is False, (
        "a single run_experiment per turn must not trip the two-in-one-turn guard"
    )
    # And ctx.current_iteration was still advanced (the seam didn't break logging).
    assert ctx.current_iteration == 2


def _iter_obj():
    from rlm.core.types import RLMIteration

    return RLMIteration(
        prompt={"role": "user", "content": "reproduce the paper"},
        response="reasoning",
        code_blocks=[],
        final_answer=None,
        iteration_time=1.0,
    )


def test_logger_resets_stack_top_policy_when_ctx_attr_unset():
    """F6 (2026-06-16): the turn-boundary reset must act on the SAME policy the
    FINAL_VAR interceptor reads — the thread-local stack top
    (forced_iteration._current_policy()) — not ctx._forced_iteration_policy.

    Here the interceptor's policy lives ONLY on the thread-local stack and
    ctx._forced_iteration_policy is unset. Pre-fix the reset keyed off the ctx
    attr alone, so it never fired and a single failing run_experiment per turn
    accumulated across turns → false two-in-one-turn refusal of a legit FINAL_VAR.
    """
    from backend.agents.rlm.forced_iteration import forced_iteration_policy
    from backend.agents.rlm.sse_bridge import ReproLabRLMLogger

    p = _policy()

    class _Ctx:
        current_iteration = 0
        # NOTE: no _forced_iteration_policy attribute at all (the divergence).

    ctx = _Ctx()
    logger = ReproLabRLMLogger(emit=lambda _e: None, checkpointer=MagicMock(), ctx=ctx)

    # The policy is on the thread-local stack (what the interceptor consults),
    # NOT on ctx. Each turn has exactly one failing run_experiment.
    with forced_iteration_policy(p):
        p.record_run_experiment(outcome="repairable")
        logger.log(_iter_obj())  # turn 1 boundary → must reset the STACK-TOP policy
        p.record_run_experiment(outcome="repairable")
        logger.log(_iter_obj())  # turn 2 boundary → reset again
        p.record_run_experiment(outcome="repairable")  # turn 3, one experiment so far

        decision = p.should_refuse_final_var(current_score=0.9, iteration_count=3)

    assert decision.refuse is False, (
        "stack-top policy must be reset at each turn boundary even when "
        "ctx._forced_iteration_policy is unset (F6 divergence)"
    )
    assert ctx.current_iteration == 2  # the ctx counter seam still advanced


def test_logger_resets_stack_top_not_a_stale_ctx_policy():
    """When ctx carries a DIFFERENT (stale) policy than the stack top, the reset
    must hit the stack-top one (what the interceptor reads), leaving the stale
    ctx policy untouched."""
    from backend.agents.rlm.forced_iteration import forced_iteration_policy
    from backend.agents.rlm.sse_bridge import ReproLabRLMLogger

    interceptor_policy = _policy()   # the one on the thread-local stack
    stale_ctx_policy = _policy()     # a different object hanging off ctx

    class _Ctx:
        current_iteration = 0

    ctx = _Ctx()
    ctx._forced_iteration_policy = stale_ctx_policy
    logger = ReproLabRLMLogger(emit=lambda _e: None, checkpointer=MagicMock(), ctx=ctx)

    with forced_iteration_policy(interceptor_policy):
        # Both policies record one failing experiment this "turn".
        interceptor_policy.record_run_experiment(outcome="repairable")
        stale_ctx_policy.record_run_experiment(outcome="repairable")
        logger.log(_iter_obj())  # boundary → resets the STACK-TOP policy only

        # The interceptor's policy was reset (its per-turn tracker is empty).
        assert interceptor_policy._experiments_in_iteration == []
        # The stale ctx policy was NOT advanced by this reset.
        assert stale_ctx_policy._experiments_in_iteration == ["repairable"]
