"""Tests for the max_pod_seconds field on RunBudget."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from backend.agents.resilience.budget import RunBudget
from backend.agents.resilience.cost import RunCostLedger
from backend.agents.resilience.failures import BudgetExhausted


def _frozen_now() -> datetime:
    return datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)


def test_run_budget_accepts_max_pod_seconds_field():
    budget = RunBudget(max_pod_seconds=600.0)
    assert budget.max_pod_seconds == 600.0


def test_run_budget_defaults_max_pod_seconds_to_none():
    budget = RunBudget()
    assert budget.max_pod_seconds is None


def test_check_pod_seconds_raises_when_pod_started_at_exceeds_cap():
    budget = RunBudget(max_pod_seconds=60.0)
    pod_started_at = _frozen_now() - timedelta(seconds=61)
    with pytest.raises(BudgetExhausted) as exc:
        budget.check_pod_seconds(
            pod_started_at=pod_started_at,
            agent_id="experiment-runner",
            now=_frozen_now(),
        )
    assert "61.0s" in str(exc.value)
    assert ">= 60.0s" in str(exc.value)
    assert exc.value.elapsed_seconds == pytest.approx(61.0)


def test_check_pod_seconds_raises_at_exact_boundary():
    budget = RunBudget(max_pod_seconds=60.0)
    pod_started_at = _frozen_now() - timedelta(seconds=60)
    with pytest.raises(BudgetExhausted):
        budget.check_pod_seconds(
            pod_started_at=pod_started_at,
            agent_id="experiment-runner",
            now=_frozen_now(),
        )


def test_check_pod_seconds_noop_when_under_cap():
    budget = RunBudget(max_pod_seconds=600.0)
    pod_started_at = _frozen_now() - timedelta(seconds=10)
    budget.check_pod_seconds(
        pod_started_at=pod_started_at,
        agent_id="experiment-runner",
        now=_frozen_now(),
    )  # must not raise


def test_check_pod_seconds_noop_when_cap_is_none():
    budget = RunBudget(max_pod_seconds=None)
    pod_started_at = _frozen_now() - timedelta(seconds=99_999)
    budget.check_pod_seconds(
        pod_started_at=pod_started_at,
        agent_id="experiment-runner",
        now=_frozen_now(),
    )  # must not raise


def test_check_pod_seconds_noop_when_pod_started_at_is_none():
    budget = RunBudget(max_pod_seconds=60.0)
    budget.check_pod_seconds(
        pod_started_at=None,
        agent_id="experiment-runner",
        now=_frozen_now(),
    )  # must not raise — no pod, no enforcement
