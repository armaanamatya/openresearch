"""Unit tests for the backend-agnostic GPU capacity descriptor (spec 2026-05-31).

All nvidia-smi access is mocked via ``monkeypatch.setattr(gc._alloc, ...)`` —
no real GPUs are required.  Mirrors the mocking style of
``tests/agents/rlm/test_gpu_cell_runner.py``.
"""

from __future__ import annotations

import types

import pytest

from backend.services.runtime import gpu_capacity as gc
from backend.services.runtime.local_gpu_allocator import GpuDevice


def _dev(index: int, uuid: str, total_mb: int, used_mb: int = 0, ext: int = 0) -> GpuDevice:
    return GpuDevice(
        index=index,
        uuid=uuid,
        memory_total_mb=total_mb,
        memory_used_mb=used_mb,
        ext_proc_count=ext,
        proc_pids=(),
    )


def _ctx(**kw):
    return types.SimpleNamespace(**kw)


# --- local: leased path -----------------------------------------------------

def test_local_leased_ids_report_count_and_min_vram(monkeypatch):
    devs = [_dev(0, "GPU-a", 24564), _dev(1, "GPU-b", 24564)]
    monkeypatch.setattr(gc._alloc, "discover_gpus", lambda: devs)
    cap = gc.describe_capacity(_ctx(sandbox_mode="local", gpu_device_ids=["GPU-a", "GPU-b"]))
    assert cap.backend_kind == "local"
    assert cap.num_gpus == 2
    assert cap.free_gpu_ids == ("GPU-a", "GPU-b")
    assert cap.per_gpu_vram_gb == pytest.approx(24564 / 1024)
    assert cap.can_escalate is False
    assert not cap.is_empty
    assert cap.detail["leased"] is True


def test_local_per_gpu_is_min_across_mixed_sizes(monkeypatch):
    devs = [_dev(0, "GPU-a", 24564), _dev(1, "GPU-b", 49152)]
    monkeypatch.setattr(gc._alloc, "discover_gpus", lambda: devs)
    cap = gc.describe_capacity(_ctx(sandbox_mode="local", gpu_device_ids=["GPU-a", "GPU-b"]))
    assert cap.per_gpu_vram_gb == pytest.approx(24564 / 1024)  # binding = smallest card
    assert cap.total_vram_gb == pytest.approx((24564 + 49152) / 1024)


def test_local_leased_as_csv_string(monkeypatch):
    devs = [_dev(0, "GPU-a", 24564), _dev(1, "GPU-b", 24564)]
    monkeypatch.setattr(gc._alloc, "discover_gpus", lambda: devs)
    cap = gc.describe_capacity(_ctx(sandbox_mode="local", gpu_device_ids="GPU-a,GPU-b"))
    assert cap.num_gpus == 2
    assert cap.free_gpu_ids == ("GPU-a", "GPU-b")


def test_local_lease_without_discovery_trusts_count_with_override(monkeypatch):
    monkeypatch.setattr(gc._alloc, "discover_gpus", lambda: [])  # smi failed
    cap = gc.describe_capacity(
        _ctx(sandbox_mode="local", gpu_device_ids=["GPU-a", "GPU-b"], vram_override=24.0)
    )
    assert cap.num_gpus == 2
    assert cap.per_gpu_vram_gb == pytest.approx(24.0)


# --- local: no-lease (planning) path ---------------------------------------

def test_local_no_lease_uses_free_devices(monkeypatch):
    devs = [_dev(0, "GPU-a", 24564), _dev(1, "GPU-b", 24564, used_mb=20000, ext=1)]
    monkeypatch.setattr(gc._alloc, "discover_gpus", lambda: devs)
    monkeypatch.setattr(gc._alloc, "free_devices", lambda d, **k: [devs[0]])
    cap = gc.describe_capacity(_ctx(sandbox_mode="local", gpu_device_ids=None))
    assert cap.num_gpus == 1
    assert cap.free_gpu_ids == ("GPU-a",)
    assert cap.detail["leased"] is False


def test_local_no_gpus_is_empty(monkeypatch):
    monkeypatch.setattr(gc._alloc, "discover_gpus", lambda: [])
    cap = gc.describe_capacity(_ctx(sandbox_mode="local", gpu_device_ids=None))
    assert cap.is_empty
    assert cap.num_gpus == 0


def test_discover_failure_is_safe(monkeypatch):
    def _boom():
        raise RuntimeError("nvidia-smi exploded")

    monkeypatch.setattr(gc._alloc, "discover_gpus", _boom)
    cap = gc.describe_capacity(_ctx(sandbox_mode="local", gpu_device_ids=None))
    assert cap.is_empty  # safe default, no crash


# --- fits() / headroom ------------------------------------------------------

def test_fits_applies_headroom():
    cap = gc.GpuCapacity("local", 2, 24.0, ("a", "b"), can_escalate=False)
    assert cap.fits(18.0, headroom=1.25)        # 22.5 <= 24
    assert not cap.fits(20.0, headroom=1.25)    # 25.0 > 24


def test_fits_unknown_capacity_never_blocks():
    cap = gc.GpuCapacity("local", 1, 0.0, ("a",), can_escalate=False)
    assert cap.fits(999.0)  # unknown per-gpu vram -> don't block


# --- cloud / azure ----------------------------------------------------------

def test_runpod_from_plan_can_escalate():
    plan = {"vram_gb": 80, "gpu_count": 1, "short_name": "h100_80"}
    cap = gc.describe_capacity(_ctx(sandbox_mode="runpod", gpu_plan=plan, gpu_device_ids=None))
    assert cap.backend_kind == "runpod"
    assert cap.can_escalate is True
    assert cap.per_gpu_vram_gb == 80
    assert cap.num_gpus == 1
    assert cap.detail["sku"] == "h100_80"


def test_runpod_defaults_when_plan_missing():
    cap = gc.describe_capacity(_ctx(sandbox_mode="runpod", gpu_plan=None, gpu_device_ids=None))
    assert cap.backend_kind == "runpod"
    assert cap.num_gpus == 1
    assert cap.can_escalate is True


def test_azure_returns_gpu_capacity_not_stub():
    # _describe_azure is now implemented (spec 2026-06-03-azure-aks-gpu-backend-design.md).
    # It must return a GpuCapacity rather than raising NotImplementedError.
    cap = gc.describe_capacity(_ctx(sandbox_mode="azure"))
    assert cap.backend_kind == "azure"
    assert cap.can_escalate is False
    assert cap.num_gpus >= 1
