"""Tests for the heartbeat() primitive."""

from __future__ import annotations

import json


def test_heartbeat_returns_alive_and_note(make_context, tmp_path):
    from backend.agents.rlm.primitives import heartbeat

    ctx = make_context(tmp_path)
    result = heartbeat("about to implement_baseline", ctx=ctx)
    assert result["alive"] is True
    assert result["note"] == "about to implement_baseline"
    assert isinstance(result["counter"], int)
    assert result["counter"] >= 1


def test_heartbeat_empty_note_ok(make_context, tmp_path):
    from backend.agents.rlm.primitives import heartbeat

    ctx = make_context(tmp_path)
    result = heartbeat(ctx=ctx)
    assert result["alive"] is True
    assert result["note"] == ""


def test_heartbeat_counter_increments(make_context, tmp_path):
    from backend.agents.rlm.primitives import heartbeat

    ctx = make_context(tmp_path)
    r1 = heartbeat("first", ctx=ctx)
    r2 = heartbeat("second", ctx=ctx)
    assert r2["counter"] == r1["counter"] + 1


def test_heartbeat_emits_iteration_heartbeat_event(make_context, tmp_path):
    from backend.agents.rlm.primitives import heartbeat

    ctx = make_context(tmp_path)
    result = heartbeat("about to run_experiment", ctx=ctx)

    dashboard_path = ctx.project_dir / "dashboard_events.jsonl"
    assert dashboard_path.exists()

    events = [
        json.loads(line)
        for line in dashboard_path.read_text().splitlines()
        if line.strip()
    ]
    heartbeat_events = [e for e in events if e.get("event") == "iteration_heartbeat"]
    assert len(heartbeat_events) >= 1

    ev = heartbeat_events[-1]
    assert ev["event"] == "iteration_heartbeat"
    assert ev["note"] == "about to run_experiment"
    assert ev["counter"] == result["counter"]
    assert "timestamp" in ev
    assert "iteration" in ev  # may be None if current_iteration not set


def test_heartbeat_event_shape_verbatim(make_context, tmp_path):
    """Verify the exact iteration_heartbeat event payload shape."""
    from backend.agents.rlm.primitives import heartbeat

    ctx = make_context(tmp_path)
    ctx.current_iteration = 3  # simulate being inside iteration 3
    heartbeat("about to rlm_query", ctx=ctx)

    events = [
        json.loads(line)
        for line in (ctx.project_dir / "dashboard_events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    ev = next(e for e in events if e.get("event") == "iteration_heartbeat")

    # All five required keys must be present
    assert set(ev.keys()) >= {"event", "timestamp", "iteration", "counter", "note"}
    assert ev["event"] == "iteration_heartbeat"
    assert ev["iteration"] == 3
    assert ev["note"] == "about to rlm_query"
    assert isinstance(ev["counter"], int)
    assert isinstance(ev["timestamp"], str)
