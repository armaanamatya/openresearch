"""GPU reservation + the allocator's own-holder discount (2026-05-30).

Reserving free cards on a shared host parks a tiny CUDA holder so other users
skip them; the allocator must still let OUR runs lease a reserved card (via the
holder PIDs). Deterministic — no real GPUs, holders mocked.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from backend.services.runtime import gpu_reservation as res_mod
from backend.services.runtime import local_gpu_allocator as alloc_mod
from backend.services.runtime.gpu_reservation import GpuReservation, GpuReservationManager
from backend.services.runtime.local_gpu_allocator import GpuDevice, free_devices


def _dev(index, uuid, used=1, proc_pids=()):
    return GpuDevice(
        index=index, uuid=uuid, memory_total_mb=24000, memory_used_mb=used,
        ext_proc_count=len(proc_pids), proc_pids=tuple(proc_pids),
    )


# ── free_devices own-holder discount ─────────────────────────────────────────

def test_free_idle_card_only():
    devs = [_dev(0, "GPU-a"), _dev(1, "GPU-b", used=5000)]  # b has heavyweight mem
    assert [d.index for d in free_devices(devs)] == [0]


def test_foreign_proc_excluded_but_own_holder_free():
    devs = [_dev(0, "GPU-a", proc_pids=(999,))]
    assert free_devices(devs) == []                      # foreign process
    assert free_devices(devs, own_pids={999}) == [devs[0]]  # our holder → free for us


def test_mixed_own_and_foreign_excluded():
    devs = [_dev(0, "GPU-a", proc_pids=(111, 222))]
    assert free_devices(devs, own_pids={111}) == []      # 222 is foreign


def test_unknown_occupancy_excluded():
    # ext_proc_count says occupied but no PID captured → unattributable → foreign.
    d = GpuDevice(index=0, uuid="GPU-a", memory_total_mb=24000, memory_used_mb=1,
                  ext_proc_count=1, proc_pids=())
    assert free_devices([d]) == []
    assert free_devices([d], own_pids={123}) == []


def test_discover_captures_pids(monkeypatch):
    def fake_smi(*args, **_kw):
        if args[0].startswith("--query-gpu"):
            return "0, GPU-a, 24000, 1\n1, GPU-b, 24000, 1\n"
        return "GPU-a, 111\nGPU-a, 222\n"  # compute-apps: two procs on GPU-a
    monkeypatch.setattr(alloc_mod, "_run_nvidia_smi", fake_smi)
    devs = {d.uuid: d for d in alloc_mod.discover_gpus()}
    assert devs["GPU-a"].proc_pids == (111, 222) and devs["GPU-a"].ext_proc_count == 2
    assert devs["GPU-b"].proc_pids == () and devs["GPU-b"].ext_proc_count == 0


# ── reservation registry: reap / release / holder_pids ───────────────────────

def _write(reg, entries):
    reg.write_text(json.dumps(entries), encoding="utf-8")


def test_list_reaps_dead_pid(tmp_path, monkeypatch):
    reg = tmp_path / "r.json"
    _write(reg, {"GPU-a": {"index": 0, "pid": 999999, "hold_mib": 256, "reserved_at": "", "expires_at": ""}})
    monkeypatch.setattr(res_mod, "_is_pid_alive", lambda p: False)
    mgr = GpuReservationManager(reg)
    assert mgr.list_reservations() == []
    assert json.loads(reg.read_text()) == {}  # persisted


def test_list_reaps_expired_and_terminates(tmp_path, monkeypatch):
    reg = tmp_path / "r.json"
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    _write(reg, {"GPU-a": {"index": 0, "pid": 1234, "hold_mib": 256, "reserved_at": "", "expires_at": past}})
    monkeypatch.setattr(res_mod, "_is_pid_alive", lambda p: True)
    killed = []
    monkeypatch.setattr(res_mod, "_terminate", lambda p: killed.append(p))
    mgr = GpuReservationManager(reg)
    assert mgr.list_reservations() == []
    assert killed == [1234]


def test_holder_pids(tmp_path, monkeypatch):
    reg = tmp_path / "r.json"
    _write(reg, {"GPU-a": {"index": 0, "pid": 1234, "hold_mib": 256, "reserved_at": "", "expires_at": ""}})
    monkeypatch.setattr(res_mod, "_is_pid_alive", lambda p: True)
    assert GpuReservationManager(reg).holder_pids() == frozenset({1234})


def test_release_specific(tmp_path, monkeypatch):
    reg = tmp_path / "r.json"
    _write(reg, {
        "GPU-a": {"index": 0, "pid": 111, "hold_mib": 256, "reserved_at": "", "expires_at": ""},
        "GPU-b": {"index": 1, "pid": 222, "hold_mib": 256, "reserved_at": "", "expires_at": ""},
    })
    killed = []
    monkeypatch.setattr(res_mod, "_terminate", lambda p: killed.append(p))
    mgr = GpuReservationManager(reg)
    assert mgr.release(uuids=["GPU-a"]) == ["GPU-a"]
    assert killed == [111]
    assert set(json.loads(reg.read_text())) == {"GPU-b"}


# ── reserve: target selection (holder spawn mocked) ──────────────────────────

def test_reserve_count_picks_first_free(tmp_path, monkeypatch):
    reg = tmp_path / "r.json"
    devs = [_dev(4, "GPU-e"), _dev(5, "GPU-f"), _dev(6, "GPU-g")]
    monkeypatch.setattr(res_mod, "discover_gpus", lambda: devs)
    monkeypatch.setattr(res_mod, "free_devices", lambda d, **_kw: list(d))

    def fake_spawn(self, dev, mib, ttl, now):
        return GpuReservation(dev.uuid, dev.index, 1000 + dev.index, mib, now.isoformat(), "")
    monkeypatch.setattr(GpuReservationManager, "_spawn_holder", fake_spawn)

    created = GpuReservationManager(reg).reserve(count=2)
    assert [r.index for r in created] == [4, 5]
    assert set(json.loads(reg.read_text())) == {"GPU-e", "GPU-f"}


def test_reserve_requires_exactly_one_selector(tmp_path):
    mgr = GpuReservationManager(tmp_path / "r.json")
    with pytest.raises(ValueError):
        mgr.reserve(count=2, all_free=True)
    with pytest.raises(ValueError):
        mgr.reserve()


def test_reserve_skips_already_reserved(tmp_path, monkeypatch):
    reg = tmp_path / "r.json"
    _write(reg, {"GPU-e": {"index": 4, "pid": 1004, "hold_mib": 256, "reserved_at": "", "expires_at": ""}})
    monkeypatch.setattr(res_mod, "_is_pid_alive", lambda p: True)
    devs = [_dev(4, "GPU-e", proc_pids=(1004,)), _dev(5, "GPU-f")]
    monkeypatch.setattr(res_mod, "discover_gpus", lambda: devs)
    monkeypatch.setattr(res_mod, "free_devices",
                        lambda d, **kw: free_devices(d, own_pids=kw.get("own_pids", ())))

    def fake_spawn(self, dev, mib, ttl, now):
        return GpuReservation(dev.uuid, dev.index, 2000 + dev.index, mib, now.isoformat(), "")
    monkeypatch.setattr(GpuReservationManager, "_spawn_holder", fake_spawn)

    created = GpuReservationManager(reg).reserve(all_free=True)
    assert [r.index for r in created] == [5]  # GPU-e already held → only GPU-f added


def test_reserve_excludes_allocator_leased(tmp_path, monkeypatch):
    reg = tmp_path / "r.json"
    # the allocator's lease file (sibling) holds GPU-e via another run
    (tmp_path / ".gpu_leases.json").write_text(json.dumps({
        "run-1": {"pid": 5, "gpu_uuids": ["GPU-e"], "gpu_indices": [4], "created_at": ""},
    }), encoding="utf-8")
    devs = [_dev(4, "GPU-e"), _dev(5, "GPU-f")]
    monkeypatch.setattr(res_mod, "discover_gpus", lambda: devs)
    monkeypatch.setattr(res_mod, "free_devices", lambda d, **_kw: list(d))

    def fake_spawn(self, dev, mib, ttl, now):
        return GpuReservation(dev.uuid, dev.index, 3000 + dev.index, mib, now.isoformat(), "")
    monkeypatch.setattr(GpuReservationManager, "_spawn_holder", fake_spawn)

    created = GpuReservationManager(reg).reserve(all_free=True)
    assert [r.index for r in created] == [5]  # GPU-e is leased → excluded from reservation
