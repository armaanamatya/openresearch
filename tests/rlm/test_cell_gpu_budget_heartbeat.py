"""Pure accrual math for the mid-cell GPU-$ heartbeat."""

from __future__ import annotations

import pytest

from backend.agents.rlm.k8s_job_cell_runner import (
    _accrued_gpu_usd,
    _over_gpu_budget,
)


def test_accrued_usd_scales_with_time_and_gpus():
    assert _accrued_gpu_usd(elapsed_s=3600, usd_per_hr_per_gpu=3.93, gpu_count=2) == pytest.approx(7.86)


def test_over_budget_true_at_or_above_cap():
    assert _over_gpu_budget(accrued=40.0, cap=40.0) is True
    assert _over_gpu_budget(accrued=41.0, cap=40.0) is True


def test_over_budget_false_below_cap_or_no_cap():
    assert _over_gpu_budget(accrued=39.9, cap=40.0) is False
    assert _over_gpu_budget(accrued=100.0, cap=None) is False
    assert _over_gpu_budget(accrued=100.0, cap=0) is False


# ---------------------------------------------------------------------------
# Integration: the heartbeat actually TERMINATES the watch loop (not a no-op).
# ---------------------------------------------------------------------------

from backend.agents.rlm import k8s_job_cell_runner as _runner  # noqa: E402


class _FakeMonotonic:
    """Deterministic monotonic clock: each read advances by ``step`` seconds.

    ``_watch_job`` reads ``time.monotonic()`` at least twice before the first
    heartbeat evaluation (job_deadline, job_started, then ``now`` at the loop
    top). A per-read advance makes ``now - job_started`` grow each poll so the
    accrued GPU-$ crosses the cap within a couple of iterations — no real sleep.
    """

    def __init__(self, step: float) -> None:
        self._t = 0.0
        self._step = step

    def __call__(self) -> float:
        v = self._t
        self._t += self._step
        return v


class _NeverTerminalJob:
    """A Job status that never reaches a terminal condition: one active pod."""

    class _Status:
        conditions: list = []
        succeeded = 0
        failed = 0
        active = 1

    status = _Status()


class _FakeBatch:
    def read_namespaced_job_status(self, job_name, namespace):
        return _NeverTerminalJob()


class _FakeCore:
    def list_namespaced_pod(self, namespace, label_selector=None):
        # No pods → _collect_pod_info returns (None, None, "") without raising.
        class _Pods:
            items: list = []

        return _Pods()


class _FakeK8s:
    batch = _FakeBatch()
    core = _FakeCore()


def test_heartbeat_terminates_when_gpu_cap_breached(monkeypatch):
    """A wedged, never-terminal cell whose accrued GPU-$ crosses the cap returns
    ``gpu_budget_exceeded`` — proving the guard FIRES (not a silent no-op)."""
    # Advance the clock 600s per read; at $100/hr/GPU x 2 GPUs a couple of polls
    # accrue > $40, tripping the cap. Also stub sleep + poll interval so no wall time.
    monkeypatch.setattr(_runner.time, "monotonic", _FakeMonotonic(step=600.0))
    monkeypatch.setattr(_runner.time, "sleep", lambda _s: None)
    monkeypatch.setattr(_runner, "_cloud_setting", lambda *a, **k: 0.0)

    result = _runner._watch_job(
        k8s=_FakeK8s(),
        job_name="job-wedged",
        namespace="ns",
        overall_deadline=None,
        active_deadline_seconds=86400,  # far in the future → deadline never trips
        pending_timeout_s=99999.0,
        gpu_budget_cap=40.0,
        gpu_usd_per_hr_per_gpu=100.0,
        gpu_count=2,
    )

    assert result["status"] == "gpu_budget_exceeded"


def test_heartbeat_is_noop_without_a_cap(monkeypatch):
    """No cap (rate=0) → the heartbeat never fires; the loop reaches the per-cell
    deadline instead, exactly as before this change."""
    monkeypatch.setattr(_runner.time, "monotonic", _FakeMonotonic(step=600.0))
    monkeypatch.setattr(_runner.time, "sleep", lambda _s: None)
    monkeypatch.setattr(_runner, "_cloud_setting", lambda *a, **k: 0.0)

    result = _runner._watch_job(
        k8s=_FakeK8s(),
        job_name="job-nocap",
        namespace="ns",
        overall_deadline=None,
        active_deadline_seconds=1,  # trips the deadline quickly with the fake clock
        pending_timeout_s=99999.0,
        gpu_budget_cap=None,
        gpu_usd_per_hr_per_gpu=0.0,
        gpu_count=2,
    )

    assert result["status"] == "deadline"


def test_gpu_budget_exceeded_maps_to_terminal_nonretryable_error():
    """``_map_status`` treats ``gpu_budget_exceeded`` as a hard STATUS_ERROR (never
    STATUS_OOM_FAILED, so the escalation loop won't re-submit) with the
    ``budget_exhausted:`` terminal-stop prefix."""
    status, error, retries = _runner._map_status(
        _runner._watch_result("gpu_budget_exceeded", log="pod log tail"),
        cell_id="c0",
        blob_status=None,
    )
    assert status == _runner.STATUS_ERROR
    assert error is not None and error.startswith("budget_exhausted:")
