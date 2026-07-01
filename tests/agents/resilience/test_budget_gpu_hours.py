from __future__ import annotations


import pytest

from backend.agents.resilience.budget import RunBudget, BudgetExhausted


def test_gpu_hours_within_budget_passes():
    RunBudget(max_gpu_hours=10.0).check_gpu_hours(gpu_hours_used=4.0, agent_id="root")   # no raise


def test_gpu_hours_over_budget_raises():
    with pytest.raises(BudgetExhausted):
        RunBudget(max_gpu_hours=2.0).check_gpu_hours(gpu_hours_used=2.5, agent_id="root")


def test_gpu_hours_none_is_unbounded():
    RunBudget().check_gpu_hours(gpu_hours_used=1_000.0, agent_id="root")                 # no raise


def test_gpu_hours_zero_disables():
    RunBudget(max_gpu_hours=0.0).check_gpu_hours(gpu_hours_used=1_000.0, agent_id="root")  # 0 == disabled
