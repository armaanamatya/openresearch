"""C2 — mid-run orphan-resource guard (2026-06-16).

REPROLAB_ORPHAN_GUARD default OFF → kill_orphans is a no-op (byte-for-byte today);
ON → every registered experiment process group is SIGKILLed on abandonment.
"""

from __future__ import annotations

import signal

import pytest

from backend.agents.rlm import orphan_guard as og


@pytest.fixture(autouse=True)
def _clear(monkeypatch):
    monkeypatch.delenv("REPROLAB_ORPHAN_GUARD", raising=False)
    with og._lock:
        og._active_pgids.clear()


def test_off_state_kill_is_noop(monkeypatch):
    og.register(12345)
    killed: list[int] = []
    monkeypatch.setattr(og.os, "killpg", lambda pid, sig: killed.append(pid))
    assert og.kill_orphans() == 0  # flag off → no kill
    assert killed == []
    # registry untouched when off (a later on-call can still act)
    assert 12345 in og._active_pgids


def test_on_state_kills_registered_groups(monkeypatch):
    monkeypatch.setenv("REPROLAB_ORPHAN_GUARD", "1")
    og.register(111)
    og.register(222)
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(og.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    assert og.kill_orphans() == 2
    assert sorted(p for p, _ in killed) == [111, 222]
    assert all(s == signal.SIGKILL for _, s in killed)
    assert og._active_pgids == set()  # drained after kill


def test_deregister_prevents_kill(monkeypatch):
    monkeypatch.setenv("REPROLAB_ORPHAN_GUARD", "1")
    og.register(111)
    og.register(222)
    og.deregister(111)
    killed: list[int] = []
    monkeypatch.setattr(og.os, "killpg", lambda pid, sig: killed.append(pid))
    assert og.kill_orphans() == 1
    assert killed == [222]


def test_dead_pgid_is_failsoft(monkeypatch):
    monkeypatch.setenv("REPROLAB_ORPHAN_GUARD", "1")
    og.register(999)

    def _raise(pid, sig):
        raise ProcessLookupError()

    monkeypatch.setattr(og.os, "killpg", _raise)
    assert og.kill_orphans() == 0  # dead pgid no-ops, never raises
