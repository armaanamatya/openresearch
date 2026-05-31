"""Tests for backend.services.runtime.local_gpu_allocator.

Covers:
  1. free_devices exclusion logic (memory threshold; ext_proc_count).
  2. LocalGpuAllocator.capacity(count=2) with a tmp state file.
  3. acquire returns a GpuLease with the correct count; two leases are
     disjoint; acquire returns None when not enough GPUs are available.
  4. acquire is idempotent (same lease_id returns the same indices).
  5. release frees GPUs so capacity rises and a re-acquire succeeds.
  6. reclaim_stale reaps leases whose PID is dead.
  7. Two LocalGpuAllocator instances on the same state file do not
     double-allocate (exercises the flock read-modify-write cycle).

``discover_gpus`` is always monkeypatched at the module level so no
real nvidia-smi is invoked.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

import backend.services.runtime.local_gpu_allocator as _alloc_module
from backend.services.runtime.local_gpu_allocator import (
    GpuDevice,
    GpuLease,
    LocalGpuAllocator,
    free_devices,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_devices(*specs: tuple[int, str, int, int, int]) -> list[GpuDevice]:
    """Build a list of GpuDevice objects from (index, uuid, total, used, proc_count)."""
    return [
        GpuDevice(
            index=spec[0],
            uuid=spec[1],
            memory_total_mb=spec[2],
            memory_used_mb=spec[3],
            ext_proc_count=spec[4],
        )
        for spec in specs
    ]


def _dead_pid() -> int:
    """Return a PID that is guaranteed to not exist.

    Scans from a high number downward until os.kill(pid, 0) raises
    ProcessLookupError, or returns 999999 as a safe fallback.
    """
    for candidate in range(999999, 900000, -1):
        try:
            os.kill(candidate, 0)
        except ProcessLookupError:
            return candidate
        except PermissionError:
            continue  # process exists but owned by another user; skip
    return 999999  # fallback — virtually guaranteed to be dead


# ---------------------------------------------------------------------------
# free_devices
# ---------------------------------------------------------------------------


def test_free_devices_excludes_high_memory_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    """A device with memory_used_mb >= threshold must be excluded."""
    devices = _make_devices(
        (0, "GPU-a", 24000, 1024, 0),   # used == threshold → excluded (< check)
        (1, "GPU-b", 24000, 512, 0),    # under threshold → included
        (2, "GPU-c", 24000, 2048, 0),   # over threshold → excluded
    )
    result = free_devices(devices, free_mem_threshold_mb=1024)
    uuids = {d.uuid for d in result}
    assert "GPU-b" in uuids
    assert "GPU-a" not in uuids
    assert "GPU-c" not in uuids


def test_free_devices_excludes_devices_with_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A device with ext_proc_count > 0 must be excluded regardless of memory."""
    devices = _make_devices(
        (0, "GPU-busy", 24000, 100, 3),    # has processes → excluded
        (1, "GPU-idle", 24000, 100, 0),    # no processes → included
    )
    result = free_devices(devices, free_mem_threshold_mb=1024)
    uuids = {d.uuid for d in result}
    assert "GPU-idle" in uuids
    assert "GPU-busy" not in uuids


def test_free_devices_includes_only_fully_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only devices with zero processes AND memory below threshold are returned."""
    devices = _make_devices(
        (0, "GPU-clean", 24000, 50, 0),    # both OK → included
        (1, "GPU-mem-busy", 24000, 1500, 0),  # memory over threshold → excluded
        (2, "GPU-proc-busy", 24000, 50, 1),   # process running → excluded
        (3, "GPU-both-bad", 24000, 1500, 2),  # both bad → excluded
    )
    result = free_devices(devices, free_mem_threshold_mb=1024)
    assert len(result) == 1
    assert result[0].uuid == "GPU-clean"


def test_free_devices_calls_discover_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """free_devices calls module-level discover_gpus when devices=None."""
    called = {"n": 0}
    fixed_devices = _make_devices((0, "GPU-z", 24000, 10, 0))

    def _fake_discover() -> list[GpuDevice]:
        called["n"] += 1
        return fixed_devices

    monkeypatch.setattr(_alloc_module, "discover_gpus", _fake_discover)
    result = free_devices(None, free_mem_threshold_mb=1024)
    assert called["n"] == 1
    assert result[0].uuid == "GPU-z"


# ---------------------------------------------------------------------------
# LocalGpuAllocator.capacity
# ---------------------------------------------------------------------------


def test_capacity_with_two_free_gpus(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """capacity(count=2) with 4 free GPUs should return 2 (floor(4/2))."""
    devices = _make_devices(
        (0, "GPU-0", 24000, 10, 0),
        (1, "GPU-1", 24000, 10, 0),
        (2, "GPU-2", 24000, 10, 0),
        (3, "GPU-3", 24000, 10, 0),
    )
    monkeypatch.setattr(_alloc_module, "discover_gpus", lambda: devices)
    state_file = tmp_path / "leases.json"
    allocator = LocalGpuAllocator(state_file)
    assert allocator.capacity(count=2) == 2


def test_capacity_zero_when_insufficient(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """capacity(count=4) with only 3 free GPUs returns 0."""
    devices = _make_devices(
        (0, "GPU-0", 24000, 10, 0),
        (1, "GPU-1", 24000, 10, 0),
        (2, "GPU-2", 24000, 10, 0),
    )
    monkeypatch.setattr(_alloc_module, "discover_gpus", lambda: devices)
    state_file = tmp_path / "leases.json"
    allocator = LocalGpuAllocator(state_file)
    assert allocator.capacity(count=4) == 0


# ---------------------------------------------------------------------------
# acquire
# ---------------------------------------------------------------------------


def test_acquire_returns_lease_with_correct_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """acquire(count=2) returns a GpuLease with exactly 2 gpu_indices."""
    devices = _make_devices(
        (0, "GPU-0", 24000, 10, 0),
        (1, "GPU-1", 24000, 10, 0),
        (2, "GPU-2", 24000, 10, 0),
    )
    monkeypatch.setattr(_alloc_module, "discover_gpus", lambda: devices)
    allocator = LocalGpuAllocator(tmp_path / "leases.json")
    lease = allocator.acquire("run-1", os.getpid(), count=2)
    assert lease is not None
    assert isinstance(lease, GpuLease)
    assert len(lease.gpu_indices) == 2
    assert len(lease.gpu_uuids) == 2


def test_acquire_two_leases_are_disjoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two consecutive acquires must not share any GPU UUID."""
    devices = _make_devices(
        (0, "GPU-0", 24000, 10, 0),
        (1, "GPU-1", 24000, 10, 0),
        (2, "GPU-2", 24000, 10, 0),
        (3, "GPU-3", 24000, 10, 0),
    )
    monkeypatch.setattr(_alloc_module, "discover_gpus", lambda: devices)
    allocator = LocalGpuAllocator(tmp_path / "leases.json")
    lease_a = allocator.acquire("run-a", os.getpid(), count=2)
    lease_b = allocator.acquire("run-b", os.getpid(), count=2)
    assert lease_a is not None
    assert lease_b is not None
    # No UUID overlap
    assert set(lease_a.gpu_uuids).isdisjoint(set(lease_b.gpu_uuids))
    # No index overlap
    assert set(lease_a.gpu_indices).isdisjoint(set(lease_b.gpu_indices))


def test_acquire_returns_none_when_insufficient(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """acquire returns None (no partial allocation) when fewer than count GPUs are free."""
    devices = _make_devices(
        (0, "GPU-0", 24000, 10, 0),
    )
    monkeypatch.setattr(_alloc_module, "discover_gpus", lambda: devices)
    allocator = LocalGpuAllocator(tmp_path / "leases.json")
    result = allocator.acquire("run-1", os.getpid(), count=2)
    assert result is None


def test_acquire_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Calling acquire twice with the same lease_id returns the same gpu_indices."""
    devices = _make_devices(
        (0, "GPU-0", 24000, 10, 0),
        (1, "GPU-1", 24000, 10, 0),
    )
    monkeypatch.setattr(_alloc_module, "discover_gpus", lambda: devices)
    allocator = LocalGpuAllocator(tmp_path / "leases.json")
    lease_first = allocator.acquire("run-idem", os.getpid(), count=1)
    lease_second = allocator.acquire("run-idem", os.getpid(), count=1)
    assert lease_first is not None
    assert lease_second is not None
    assert lease_first.gpu_indices == lease_second.gpu_indices
    assert lease_first.gpu_uuids == lease_second.gpu_uuids


# ---------------------------------------------------------------------------
# release
# ---------------------------------------------------------------------------


def test_release_frees_gpus_for_reuse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """After release, capacity rises and the freed GPU can be re-acquired."""
    devices = _make_devices(
        (0, "GPU-only", 24000, 10, 0),
    )
    monkeypatch.setattr(_alloc_module, "discover_gpus", lambda: devices)
    state_file = tmp_path / "leases.json"
    allocator = LocalGpuAllocator(state_file)

    # Exhaust available GPU
    lease = allocator.acquire("run-1", os.getpid(), count=1)
    assert lease is not None

    # Capacity should now be 0
    assert allocator.capacity(count=1) == 0

    # Release
    allocator.release("run-1")

    # Capacity should be 1 again
    assert allocator.capacity(count=1) == 1

    # Re-acquire should succeed
    new_lease = allocator.acquire("run-2", os.getpid(), count=1)
    assert new_lease is not None
    assert new_lease.gpu_uuids[0] == "GPU-only"


# ---------------------------------------------------------------------------
# reclaim_stale
# ---------------------------------------------------------------------------


def test_reclaim_stale_reaps_dead_pid_lease(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """reclaim_stale removes leases whose PID is dead and returns their IDs."""
    devices = _make_devices(
        (0, "GPU-0", 24000, 10, 0),
    )
    monkeypatch.setattr(_alloc_module, "discover_gpus", lambda: devices)

    # Seed the state file with a lease for a definitely-dead PID
    dead_pid = _dead_pid()
    state_file = tmp_path / "leases.json"
    initial_state = {
        "stale-run": {
            "pid": dead_pid,
            "gpu_uuids": ["GPU-0"],
            "gpu_indices": [0],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    }
    state_file.write_text(json.dumps(initial_state), encoding="utf-8")

    allocator = LocalGpuAllocator(state_file)
    reclaimed = allocator.reclaim_stale()

    assert "stale-run" in reclaimed
    # State file must no longer contain the stale lease
    remaining = json.loads(state_file.read_text(encoding="utf-8"))
    assert "stale-run" not in remaining


def test_reclaim_stale_keeps_live_pid_lease(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """reclaim_stale must NOT remove a lease whose PID is still alive."""
    devices = _make_devices(
        (0, "GPU-0", 24000, 10, 0),
    )
    monkeypatch.setattr(_alloc_module, "discover_gpus", lambda: devices)

    live_pid = os.getpid()
    state_file = tmp_path / "leases.json"
    initial_state = {
        "live-run": {
            "pid": live_pid,
            "gpu_uuids": ["GPU-0"],
            "gpu_indices": [0],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    }
    state_file.write_text(json.dumps(initial_state), encoding="utf-8")

    allocator = LocalGpuAllocator(state_file)
    reclaimed = allocator.reclaim_stale()

    assert "live-run" not in reclaimed
    remaining = json.loads(state_file.read_text(encoding="utf-8"))
    assert "live-run" in remaining


# ---------------------------------------------------------------------------
# Concurrency safety: two allocators on the same state file
# ---------------------------------------------------------------------------


def test_two_allocators_no_double_allocation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two LocalGpuAllocator instances sharing one state file must not
    double-allocate the same GPU UUID.

    Each instance acquires one of two available GPUs.  After both acquires,
    the union covers exactly the two GPU UUIDs and there is no overlap.
    """
    devices = _make_devices(
        (0, "GPU-0", 24000, 10, 0),
        (1, "GPU-1", 24000, 10, 0),
    )
    monkeypatch.setattr(_alloc_module, "discover_gpus", lambda: devices)

    state_file = tmp_path / "shared_leases.json"
    alloc_a = LocalGpuAllocator(state_file)
    alloc_b = LocalGpuAllocator(state_file)

    lease_a = alloc_a.acquire("run-alpha", os.getpid(), count=1)
    lease_b = alloc_b.acquire("run-beta", os.getpid(), count=1)

    assert lease_a is not None, "First allocator should acquire a GPU"
    assert lease_b is not None, "Second allocator should acquire a different GPU"

    # Strict disjointness
    assert set(lease_a.gpu_uuids).isdisjoint(set(lease_b.gpu_uuids)), (
        "Two allocators on the same state file must never share a GPU UUID"
    )
    # Together they cover both GPUs
    all_uuids = set(lease_a.gpu_uuids) | set(lease_b.gpu_uuids)
    assert all_uuids == {"GPU-0", "GPU-1"}
