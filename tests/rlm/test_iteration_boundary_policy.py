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
