"""Periodic scheduler runs sweep_stale_pods at the configured interval.
Fail-soft on exceptions; disabled when OPENRESEARCH_POD_SWEEP_ENABLED=false or
OPENRESEARCH_RUNPOD_API_KEY is unset."""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import MagicMock, patch


@pytest.mark.asyncio
async def test_scheduler_calls_sweep_at_interval(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_RUNPOD_API_KEY", "dummy")
    monkeypatch.setenv("OPENRESEARCH_POD_SWEEP_INTERVAL_S", "0.05")  # 50ms for fast test
    sweep_calls: list = []
    fake_sweep = MagicMock(side_effect=lambda **kw: sweep_calls.append(kw) or MagicMock())
    with patch("backend.services.runtime.pod_sweep_scheduler.sweep_stale_pods", fake_sweep):
        from backend.services.runtime.pod_sweep_scheduler import PodSweepScheduler
        sched = PodSweepScheduler()
        await sched.start()
        await asyncio.sleep(0.2)
        await sched.stop()
    assert len(sweep_calls) >= 2, f"expected >=2 sweeps, got {len(sweep_calls)}"


@pytest.mark.asyncio
async def test_scheduler_disabled_via_env(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_POD_SWEEP_ENABLED", "false")
    monkeypatch.setenv("OPENRESEARCH_RUNPOD_API_KEY", "dummy")
    fake_sweep = MagicMock(return_value=MagicMock())
    with patch("backend.services.runtime.pod_sweep_scheduler.sweep_stale_pods", fake_sweep):
        from backend.services.runtime.pod_sweep_scheduler import PodSweepScheduler
        sched = PodSweepScheduler()
        await sched.start()
        await asyncio.sleep(0.1)
        await sched.stop()
    fake_sweep.assert_not_called()


@pytest.mark.asyncio
async def test_scheduler_fail_soft_on_sweep_exception(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_RUNPOD_API_KEY", "dummy")
    monkeypatch.setenv("OPENRESEARCH_POD_SWEEP_INTERVAL_S", "0.05")
    calls: list = []

    def _flaky_sweep(**kw):
        calls.append(kw)
        if len(calls) == 1:
            raise RuntimeError("network blip")
        return MagicMock()

    with patch("backend.services.runtime.pod_sweep_scheduler.sweep_stale_pods", _flaky_sweep):
        from backend.services.runtime.pod_sweep_scheduler import PodSweepScheduler
        sched = PodSweepScheduler()
        await sched.start()
        await asyncio.sleep(0.2)
        await sched.stop()
    assert len(calls) >= 2, "scheduler stopped after exception (should have continued)"
