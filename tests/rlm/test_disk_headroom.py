"""Disk-headroom preflight (2026-06-15): abort a run before any GPU work when the
runs root is near-full, so it can't hang/orphan on checkpoint writes (the SDAR
disk-100% incident: cells hung holding VRAM, no final report). Paper-agnostic.
"""

from __future__ import annotations

import pytest

from backend.agents.rlm import run as run_mod


def _fake_statvfs(free_gb: float):
    class _S:
        f_frsize = 4096
        f_bavail = int(free_gb * 1024 ** 3 / 4096)  # non-root available blocks
    return lambda _p: _S()


def test_disabled_when_min_gb_zero(tmp_path):
    assert run_mod._assert_disk_headroom(tmp_path, min_gb=0) is None
    assert run_mod._assert_disk_headroom(tmp_path, min_gb=-1) is None


def test_aborts_when_critically_low(tmp_path, monkeypatch):
    monkeypatch.setattr(run_mod.os, "statvfs", _fake_statvfs(2.0))  # 2 GB free
    with pytest.raises(RuntimeError, match="below the .* floor|GB free"):
        run_mod._assert_disk_headroom(tmp_path, min_gb=10)


def test_warns_when_thin_but_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(run_mod.os, "statvfs", _fake_statvfs(15.0))  # 15 GB, floor 10 → <2x
    reason = run_mod._assert_disk_headroom(tmp_path, min_gb=10)
    assert reason and "low disk headroom" in reason


def test_passes_with_ample_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(run_mod.os, "statvfs", _fake_statvfs(500.0))  # 500 GB free
    assert run_mod._assert_disk_headroom(tmp_path, min_gb=10) is None


def test_failsoft_on_stat_error(tmp_path, monkeypatch):
    def boom(_p):
        raise OSError("cannot stat mount")
    monkeypatch.setattr(run_mod.os, "statvfs", boom)
    # A bookkeeping failure must never block the run.
    assert run_mod._assert_disk_headroom(tmp_path, min_gb=10) is None
