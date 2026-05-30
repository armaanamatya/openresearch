"""Phase 2 — dangling sub_rlm_spawned detector emits sub_rlm_stalled.

A depth>=1 sub-call that opens (on_start) and never completes within the stall
window is the exact shape of the SDAR wedge (a sub_rlm_spawned with no matching
sub_rlm_complete for 76 min). The SubcallTracker, polled by a daemon thread,
emits a single sub_rlm_stalled per dangling sub-call so a wedge surfaces in <2 min.
"""
from __future__ import annotations

import time


def test_tracker_emits_stalled_for_open_subcall():
    from backend.agents.rlm.sse_bridge import SubcallTracker

    emitted: list[dict] = []
    tracker = SubcallTracker(emit=emitted.append, stall_after_s=0.2)

    tracker.on_start(depth=1, model="claude-haiku-4-5", prompt_preview="navigate §3")
    time.sleep(0.25)
    tracker.check_once()  # one synchronous poll
    assert any(e["event"] == "sub_rlm_stalled" for e in emitted)
    assert emitted[-1]["depth"] == 1
    assert emitted[-1]["model"] == "claude-haiku-4-5"
    assert emitted[-1]["idle_seconds"] >= 0.2

    # A second poll must NOT double-emit for the same still-open sub-call.
    emitted.clear()
    tracker.check_once()
    assert emitted == []


def test_tracker_no_stall_when_completed():
    from backend.agents.rlm.sse_bridge import SubcallTracker

    emitted: list[dict] = []
    tracker = SubcallTracker(emit=emitted.append, stall_after_s=0.2)
    tracker.on_start(depth=1, model="m", prompt_preview="x")
    tracker.on_complete(depth=1, model="m", duration=0.01, error=None)
    time.sleep(0.25)
    tracker.check_once()
    assert emitted == []


def test_tracker_emit_failure_is_fail_soft():
    from backend.agents.rlm.sse_bridge import SubcallTracker

    def boom(_event: dict) -> None:
        raise RuntimeError("dashboard write failed")

    tracker = SubcallTracker(emit=boom, stall_after_s=0.05)
    tracker.on_start(depth=1, model="m", prompt_preview="x")
    time.sleep(0.1)
    tracker.check_once()  # must not raise — observability never blocks


def test_build_sub_rlm_stalled_event_shape():
    from backend.agents.rlm.sse_bridge import build_sub_rlm_stalled_event

    ev = build_sub_rlm_stalled_event(depth=2, model="m", idle_seconds=130.0)
    assert ev["event"] == "sub_rlm_stalled"
    assert ev["depth"] == 2
    assert ev["model"] == "m"
    assert ev["idle_seconds"] == 130.0
    assert "timestamp" in ev
