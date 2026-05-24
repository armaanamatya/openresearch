"""Tests for RunpodBackend.exec calling check_run_gpu_usd when gpu_plan is set."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from backend.agents.resilience.budget import RunBudget
from backend.agents.resilience.failures import BudgetExhausted
from backend.agents.schemas import GpuPlan, GpuRequirements
from backend.services.runtime.interface import Sandbox, SandboxConfig
from backend.services.runtime.runpod_backend import RunpodBackend


def _frozen_now() -> datetime:
    return datetime(2026, 5, 23, 12, 0, 0, tzinfo=timezone.utc)


def _make_gpu_plan(total_usd_per_hr: float = 1.89) -> GpuPlan:
    return GpuPlan(
        runpod_id="NVIDIA GeForce RTX 4090",
        short_name="rtx4090",
        vram_gb=24,
        gpu_count=1,
        cloud_type="COMMUNITY",
        sku_usd_per_hr=total_usd_per_hr,
        total_usd_per_hr=total_usd_per_hr,
        container_disk_gb=50,
        volume_gb=20,
        source="paper",
        requirements=GpuRequirements(
            estimated_vram_gb=24,
            paper_gpu_string=None,
            paper_gpu_count=None,
            reasoning="test",
            confidence=0.9,
        ),
        resolved_at="2026-05-23T12:00:00+00:00",
    )


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


def test_exec_calls_check_run_gpu_usd_when_gpu_plan_set(tmp_path):
    """RunpodBackend.exec must invoke check_run_gpu_usd when gpu_plan is set."""
    gpu_plan = _make_gpu_plan(total_usd_per_hr=1.89)
    # RunBudget is frozen; patch at the class level so the call is intercepted.
    calls: list[dict] = []

    def _spy(self, *, cumulative_pod_usd: float, agent_id: str) -> None:
        calls.append({"cumulative_pod_usd": cumulative_pod_usd, "agent_id": agent_id})

    with patch.object(RunBudget, "check_run_gpu_usd", _spy):
        budget = RunBudget(max_run_gpu_usd=100.0)
        backend = RunpodBackend(api_key="dummy", run_budget=budget, gpu_plan=gpu_plan)
        sandbox = _make_sandbox(created_at=_frozen_now(), tmp_path=tmp_path)
        import asyncio
        with pytest.raises(Exception):  # SSH will fail — that's expected
            asyncio.run(backend.exec(sandbox, "echo test", timeout=5))

    assert len(calls) == 1, f"check_run_gpu_usd called {len(calls)} times, expected 1"
    assert "cumulative_pod_usd" in calls[0]
    assert calls[0]["agent_id"] == "experiment-runner"


@pytest.mark.asyncio
async def test_exec_raises_budget_exhausted_when_gpu_usd_exceeded(tmp_path):
    """When cumulative pod USD >= max_run_gpu_usd, exec raises BudgetExhausted."""
    # Pod has been running for 10 hours at $1.89/hr = $18.90 cumulative.
    budget = RunBudget(max_run_gpu_usd=5.0)
    gpu_plan = _make_gpu_plan(total_usd_per_hr=1.89)
    backend = RunpodBackend(api_key="dummy", run_budget=budget, gpu_plan=gpu_plan)
    backend._owned_pod_ids = {"test-pod"}
    backend.destroy = AsyncMock()

    # Sandbox created 10 hours ago.
    old_created_at = _frozen_now() - timedelta(hours=10)
    sandbox = _make_sandbox(created_at=old_created_at, tmp_path=tmp_path)

    with patch("backend.services.runtime.runpod_backend.datetime") as mock_dt:
        mock_dt.now.return_value = _frozen_now()
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        # Also patch budget's datetime so check_run_gpu_usd sees consistent time.
        with pytest.raises(BudgetExhausted) as exc:
            await backend.exec(sandbox, "echo hello", timeout=30)

    assert "pod-USD" in str(exc.value) or "budget" in str(exc.value).lower()


def test_exec_skips_gpu_usd_check_when_no_gpu_plan(tmp_path):
    """When gpu_plan is None, check_run_gpu_usd must NOT be called."""
    calls: list[dict] = []

    def _spy(self, *, cumulative_pod_usd: float, agent_id: str) -> None:
        calls.append({"cumulative_pod_usd": cumulative_pod_usd, "agent_id": agent_id})

    with patch.object(RunBudget, "check_run_gpu_usd", _spy):
        budget = RunBudget(max_run_gpu_usd=5.0)
        backend = RunpodBackend(api_key="dummy", run_budget=budget)
        # gpu_plan is not set — gpu_plan defaults to None.
        assert backend.gpu_plan is None
        sandbox = _make_sandbox(created_at=_frozen_now(), tmp_path=tmp_path)
        import asyncio
        with pytest.raises(Exception):  # SSH will fail
            asyncio.run(backend.exec(sandbox, "echo test", timeout=5))

    assert len(calls) == 0, "check_run_gpu_usd must not be called when gpu_plan is None"


def test_exec_skips_gpu_usd_check_when_no_run_budget(tmp_path):
    """When run_budget is None, the whole budget block is skipped."""
    gpu_plan = _make_gpu_plan()
    backend = RunpodBackend(api_key="dummy", gpu_plan=gpu_plan)
    assert backend._run_budget is None

    sandbox = _make_sandbox(created_at=_frozen_now(), tmp_path=tmp_path)
    import asyncio
    # Should not raise BudgetExhausted (will raise SSH error instead).
    with pytest.raises(Exception) as exc:
        asyncio.run(backend.exec(sandbox, "echo test", timeout=5))
    assert "BudgetExhausted" not in type(exc.value).__name__
