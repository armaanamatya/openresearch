"""Pod-time budget enforcement in RunpodBackend.exec()."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from backend.agents.resilience.budget import RunBudget
from backend.agents.resilience.failures import BudgetExhausted
from backend.services.runtime.interface import Sandbox, SandboxConfig
from backend.services.runtime.runpod_backend import RunpodBackend


def _frozen_now() -> datetime:
    return datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)


def _make_sandbox(*, created_at: datetime, tmp_path: Path) -> Sandbox:
    config = SandboxConfig(
        project_id="proj",
        run_id="run",
        project_root=tmp_path,
    )
    return Sandbox(
        sandbox_id="test-pod",
        name="test-pod",
        image="test-image",
        config=config,
        created_at=created_at,
    )


def test_runpod_backend_accepts_run_budget_in_constructor():
    budget = RunBudget(max_pod_seconds=60.0)
    backend = RunpodBackend(api_key="dummy", run_budget=budget)
    assert backend._run_budget is budget


def test_runpod_backend_defaults_run_budget_to_none():
    backend = RunpodBackend(api_key="dummy")
    assert backend._run_budget is None


@pytest.mark.asyncio
async def test_exec_raises_budget_exhausted_when_pod_time_exceeded(tmp_path):
    """When sandbox.created_at is older than max_pod_seconds, exec() raises BudgetExhausted."""
    budget = RunBudget(max_pod_seconds=60.0)
    backend = RunpodBackend(api_key="dummy", run_budget=budget)
    backend._owned_pod_ids = {"test-pod"}
    backend.destroy = AsyncMock()  # capture destroy call
    sandbox = _make_sandbox(
        created_at=_frozen_now() - timedelta(seconds=120),
        tmp_path=tmp_path,
    )

    # Use unittest.mock.patch to freeze datetime.now in budget.py for this test.
    with pytest.raises(BudgetExhausted) as exc:
        with patch("backend.agents.resilience.budget.datetime") as mock_dt:
            mock_dt.now.return_value = _frozen_now()
            await backend.exec(sandbox, "echo hello", timeout=30)

    assert "120" in str(exc.value) or "pod-time" in str(exc.value).lower()


@pytest.mark.asyncio
async def test_exec_forces_destroy_on_budget_exhaustion(tmp_path):
    """When budget exhausts, destroy() is called before BudgetExhausted propagates."""
    budget = RunBudget(max_pod_seconds=60.0)
    backend = RunpodBackend(api_key="dummy", run_budget=budget)
    backend._owned_pod_ids = {"test-pod"}
    backend.destroy = AsyncMock()
    sandbox = _make_sandbox(
        created_at=_frozen_now() - timedelta(seconds=120),
        tmp_path=tmp_path,
    )

    with patch("backend.agents.resilience.budget.datetime") as mock_dt:
        mock_dt.now.return_value = _frozen_now()
        with pytest.raises(BudgetExhausted):
            await backend.exec(sandbox, "echo hello", timeout=30)

    backend.destroy.assert_awaited_once_with(sandbox)


@pytest.mark.asyncio
async def test_exec_destroy_failure_does_not_swallow_budget_exhausted(tmp_path):
    """If destroy() itself raises, the BudgetExhausted still propagates (best-effort destroy)."""
    budget = RunBudget(max_pod_seconds=60.0)
    backend = RunpodBackend(api_key="dummy", run_budget=budget)
    backend._owned_pod_ids = {"test-pod"}
    backend.destroy = AsyncMock(side_effect=RuntimeError("destroy failed"))
    sandbox = _make_sandbox(
        created_at=_frozen_now() - timedelta(seconds=120),
        tmp_path=tmp_path,
    )

    with patch("backend.agents.resilience.budget.datetime") as mock_dt:
        mock_dt.now.return_value = _frozen_now()
        with pytest.raises(BudgetExhausted):
            await backend.exec(sandbox, "echo hello", timeout=30)


@pytest.mark.asyncio
async def test_exec_does_not_check_budget_when_none_configured(tmp_path):
    """exec() with no run_budget set must not raise BudgetExhausted even on ancient pods."""
    backend = RunpodBackend(api_key="dummy")  # no run_budget
    backend._owned_pod_ids = {"test-pod"}
    sandbox = _make_sandbox(
        created_at=_frozen_now() - timedelta(seconds=99_999),
        tmp_path=tmp_path,
    )
    # We expect exec() to proceed past the budget check and fail later on
    # the actual SSH attempt — which is fine; we're only asserting the
    # budget check itself doesn't trip.
    async def fake_ssh(pod_id):
        raise RuntimeError("ssh not mocked")  # any non-BudgetExhausted error
    backend._ssh = fake_ssh  # type: ignore[assignment]

    with pytest.raises(Exception) as exc:
        await backend.exec(sandbox, "echo hello", timeout=30)
    # The error must NOT be BudgetExhausted
    assert not isinstance(exc.value, BudgetExhausted)
