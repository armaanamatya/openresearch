from __future__ import annotations


import pytest

from backend.agents.resilience.budget import RunBudget
from backend.agents.resilience.failures import BudgetExhausted


def test_check_run_gpu_usd_passes_when_under_cap():
    budget = RunBudget(max_run_gpu_usd=5.0)
    budget.check_run_gpu_usd(cumulative_pod_usd=2.0, agent_id="run_experiment")


def test_check_run_gpu_usd_raises_when_at_or_above_cap():
    budget = RunBudget(max_run_gpu_usd=5.0)
    with pytest.raises(BudgetExhausted) as exc:
        budget.check_run_gpu_usd(cumulative_pod_usd=5.0, agent_id="run_experiment")
    assert "5.0" in str(exc.value) or "pod" in str(exc.value).lower()


def test_check_run_gpu_usd_noop_when_cap_none():
    budget = RunBudget(max_run_gpu_usd=None)
    budget.check_run_gpu_usd(cumulative_pod_usd=1_000_000.0, agent_id="x")


def test_check_run_gpu_usd_noop_when_cap_zero():
    budget = RunBudget(max_run_gpu_usd=0.0)
    budget.check_run_gpu_usd(cumulative_pod_usd=1_000_000.0, agent_id="x")
