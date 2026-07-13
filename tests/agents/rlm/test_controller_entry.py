"""In-Pod controller heartbeat loop (WS3).

The heartbeat is what makes ``sweep_orphaned_controllers`` safe: a live
controller renews its lease on a cadence, so an expired lease reliably means
the controller is gone. Tested here with a fake lease + injected clock/sleep —
no threads, no time.time().
"""

from __future__ import annotations

from backend.agents.rlm import controller_entry as ce


class _FakeLease:
    def __init__(self, renew_results):
        self._renew_results = list(renew_results)
        self.renew_calls = 0

    def renew(self, token, now_epoch):
        self.renew_calls += 1
        return self._renew_results.pop(0)


def test_heartbeat_renews_until_stopped():
    lease = _FakeLease(renew_results=["t1", "t2", "t3"])

    def is_running():
        return lease.renew_calls < 2   # stop once we have renewed twice

    lost = []
    ce.heartbeat_loop(
        lease=lease, token="t0", interval_s=60,
        sleep=lambda s: None, clock=lambda: 1000.0,
        is_running=is_running, on_lost=lost.append,
    )
    assert lease.renew_calls == 2
    assert lost == []   # stopped cleanly, never lost the lease


def test_heartbeat_stops_and_signals_when_superseded():
    lease = _FakeLease(renew_results=[None])   # first renew fails → superseded
    lost = []
    ce.heartbeat_loop(
        lease=lease, token="t0", interval_s=60,
        sleep=lambda s: None, clock=lambda: 1000.0,
        is_running=lambda: True, on_lost=lambda: lost.append("lost"),
    )
    assert lease.renew_calls == 1
    assert lost == ["lost"]   # a superseded controller must be told to stop
