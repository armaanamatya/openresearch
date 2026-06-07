"""Unit tests for EnvCacheManager (Part B, 2026-06-01).

The downloader / server launcher / health probe / clock are all INJECTED, so the
full lifecycle — idempotent download, ref-counted WebShop server, crash-safe
shared state, stale-PID reclaim, and failure→verified-Exclusion — is exercised
WITHOUT a real multi-GB download, a real server, or the network. The real
``alfworld-download`` / WebShop bring-up are integration-pending (design §5);
these tests pin the orchestration + the fairness fallback.
"""
from __future__ import annotations

import signal
from pathlib import Path

import pytest

from backend.agents.rlm import exclusion as X
from backend.services.runtime import env_cache as EC
from backend.services.runtime.env_cache import EnvCacheManager


class _Downloader:
    def __init__(self, fail: bool = False):
        self.calls = 0
        self.fail = fail

    def __call__(self, cache_dir: Path) -> None:
        self.calls += 1
        if self.fail:
            raise RuntimeError("alfworld-download exit 1")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)  # simulate fetched data


def _fake_clock(step: float = 1.0):
    t = [0.0]

    def clock() -> float:
        v = t[0]
        t[0] += step
        return v

    return clock


# --- ALFWorld: idempotent download + failure fallback ----------------------

def test_alfworld_downloads_once_then_cache_hits(tmp_path: Path):
    dl = _Downloader()
    m = EnvCacheManager(tmp_path, downloader=dl)
    r1 = m.ensure_alfworld()
    assert r1.ok and r1.data_path and dl.calls == 1 and r1.detail == "downloaded"
    r2 = m.ensure_alfworld()
    assert r2.ok and dl.calls == 1 and r2.detail == "cache hit"   # not re-downloaded
    assert r2.as_env_vars().get("ALFWORLD_DATA") == r1.data_path


def test_alfworld_failure_returns_verified_exclusion(tmp_path: Path):
    m = EnvCacheManager(tmp_path, downloader=_Downloader(fail=True))
    r = m.ensure_alfworld()
    assert not r.ok and r.exclusion is not None
    assert r.exclusion.kind == X.KIND_ENV_SETUP_FAILED
    assert r.exclusion.verified is True
    assert r.exclusion.axis == X.AXIS_ENVIRONMENT and r.exclusion.item == "ALFWorld"
    assert r.as_env_vars() == {}


def test_failed_setup_exclusion_routes_through_build_scope_block(tmp_path: Path):
    # The fairness payoff: a failed env becomes a VERIFIED exclusion that the
    # scorer drops from num+denom (self-sufficient since the Part A review fix).
    r = EnvCacheManager(tmp_path, downloader=_Downloader(fail=True)).ensure_alfworld()
    scope = X.build_scope_block([r.exclusion])
    assert scope["environments_skipped"] == ["ALFWorld"]
    assert {e["item"] for e in scope["exclusions"]} == {"ALFWorld"}


def test_shared_state_persists_across_manager_instances(tmp_path: Path):
    # Crash-safety: state is on disk + fcntl-guarded, so a fresh manager (e.g. a
    # later run/cell) sees the cache hit instead of re-downloading.
    dl = _Downloader()
    EnvCacheManager(tmp_path, downloader=dl).ensure_alfworld()
    r = EnvCacheManager(tmp_path, downloader=dl).ensure_alfworld()
    assert r.ok and r.detail == "cache hit" and dl.calls == 1


# --- WebShop: ref-counted server lifecycle ----------------------------------

def test_webshop_one_server_for_many_leases_then_torn_down(tmp_path: Path, monkeypatch):
    launches: list[int] = []
    kills: list[tuple[int, int]] = []
    monkeypatch.setattr(EC, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(EC.os, "kill", lambda pid, sig: kills.append((pid, sig)))

    def launcher(cache_dir, port):
        launches.append(port)
        return 424242

    m = EnvCacheManager(tmp_path, server_launcher=launcher, probe=lambda url: True)
    a = m.acquire_webshop()
    assert a.ok and a.base_url and len(launches) == 1 and a.as_env_vars().get("WEBSHOP_URL")
    b = m.acquire_webshop()
    assert b.ok and len(launches) == 1            # reuse the running server, no relaunch
    m.release_webshop()
    assert kills == []                            # one lease remains → server stays up
    m.release_webshop()
    assert kills and kills[-1][1] == signal.SIGTERM   # last lease → server stopped


def test_webshop_not_ready_fails_and_kills_server(tmp_path: Path, monkeypatch):
    kills: list[tuple[int, int]] = []
    monkeypatch.setattr(EC, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(EC.os, "kill", lambda pid, sig: kills.append((pid, sig)))
    monkeypatch.setattr(EC.time, "sleep", lambda *_a, **_k: None)
    m = EnvCacheManager(
        tmp_path, server_launcher=lambda c, p: 999, probe=lambda url: False,
        server_ready_timeout_s=0.5, clock=_fake_clock(),
    )
    r = m.acquire_webshop()
    assert not r.ok and r.exclusion.kind == X.KIND_ENV_SETUP_FAILED and r.exclusion.verified
    assert kills and kills[-1][1] == signal.SIGTERM   # unready server was reaped


def test_webshop_stale_pid_is_reclaimed_and_relaunched(tmp_path: Path, monkeypatch):
    alive = {424242: True}
    monkeypatch.setattr(EC, "_pid_alive", lambda pid: alive.get(pid, False))
    monkeypatch.setattr(EC.os, "kill", lambda pid, sig: None)
    launches: list[int] = []

    def launcher(c, p):
        launches.append(1)
        return 424242

    m = EnvCacheManager(tmp_path, server_launcher=launcher, probe=lambda url: True)
    m.acquire_webshop()
    assert len(launches) == 1
    alive[424242] = False             # the server process crashed out-of-band
    m.release_webshop()               # refcount→0; pid not alive → running cleared
    alive[424242] = True
    m.acquire_webshop()               # sees not-running → relaunches
    assert len(launches) == 2


def test_release_without_acquire_is_safe(tmp_path: Path):
    EnvCacheManager(tmp_path).release_webshop()   # no state yet → no-op, no raise


# --- routing + env-var resolution -------------------------------------------

def test_setup_routes_by_name(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(EC, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(EC.os, "kill", lambda pid, sig: None)
    dl = _Downloader()
    m = EnvCacheManager(tmp_path, downloader=dl, server_launcher=lambda c, p: 7, probe=lambda u: True)
    assert m.setup("ALFWorld").ok and dl.calls == 1
    assert m.setup("alf-world").detail == "cache hit"        # case/format-insensitive
    assert m.setup("WebShop").base_url
    sq = m.setup("Search-QA")
    assert sq.ok and sq.data_path is None and sq.base_url is None   # nothing to provision


def test_default_cache_dir_honours_override(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("REPROLAB_ENV_CACHE_DIR", str(tmp_path / "envs"))
    assert EC.default_cache_dir() == (tmp_path / "envs").resolve()


# --- provision_scope: whole-scope provisioning + fairness fallback ----------

def test_provision_scope_mixes_success_and_verified_failure(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(EC, "_pid_alive", lambda pid: True)
    releases: list[int] = []
    monkeypatch.setattr(EC.os, "kill", lambda pid, sig: releases.append(pid))
    # ALFWorld download FAILS; WebShop succeeds; Search-QA needs nothing.
    m = EnvCacheManager(
        tmp_path, downloader=_Downloader(fail=True),
        server_launcher=lambda c, p: 555, probe=lambda u: True,
    )
    res = EC.provision_scope(["ALFWorld", "WebShop", "Search-QA"], m)
    assert "WEBSHOP_URL" in res.env_vars and "ALFWORLD_DATA" not in res.env_vars
    assert [e.item for e in res.exclusions] == ["ALFWorld"]
    assert res.exclusions[0].kind == X.KIND_ENV_SETUP_FAILED and res.exclusions[0].verified
    # The verified failure folds into scope as an exclusion (excluded, not zeroed).
    scope = X.build_scope_block(res.exclusions)
    assert scope["environments_skipped"] == ["ALFWorld"]
    # release drops exactly the one WebShop lease acquired.
    res.release()
    assert len(releases) == 1


def test_provision_scope_all_ok_no_exclusions(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(EC, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(EC.os, "kill", lambda pid, sig: None)
    m = EnvCacheManager(tmp_path, downloader=_Downloader(),
                        server_launcher=lambda c, p: 5, probe=lambda u: True)
    res = EC.provision_scope(["ALFWorld", "Search-QA"], m)
    assert res.exclusions == [] and "ALFWORLD_DATA" in res.env_vars
    res.release()  # no webshop lease → safe no-op
