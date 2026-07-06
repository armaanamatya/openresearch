"""Tests for the 4 spec-phase SSE event builders (corpus-free control events).

Task 7: ``spec_generation_started`` / ``spec_generated`` / ``spec_validation_started``
/ ``spec_validated`` — emitted around the spec-validator hook (T8) and folded by
the frontend reducers (T12). The event NAMES are a cross-task contract; the
payloads must stay corpus-free (ids/counts/enums only, never paper text).
"""

from __future__ import annotations

from backend.agents.rlm.sse_bridge import (
    build_spec_generated_event,
    build_spec_generation_started_event,
    build_spec_validated_event,
    build_spec_validation_started_event,
)


def test_spec_generation_started_event_shape():
    ev = build_spec_generation_started_event()
    assert ev["event"] == "spec_generation_started"
    assert "timestamp" in ev
    assert set(ev.keys()) == {"event", "timestamp"}


def test_spec_generated_event_shape():
    ev = build_spec_generated_event(leaf_count=7)
    assert ev["event"] == "spec_generated"
    assert ev["leaf_count"] == 7
    assert "timestamp" in ev
    assert set(ev.keys()) == {"event", "timestamp", "leaf_count"}


def test_spec_validation_started_event_shape():
    ev = build_spec_validation_started_event(validator_model="grok-4.3")
    assert ev["event"] == "spec_validation_started"
    assert ev["validator_model"] == "grok-4.3"
    assert "timestamp" in ev
    assert set(ev.keys()) == {"event", "timestamp", "validator_model"}


def test_spec_validated_event_shape():
    ev = build_spec_validated_event(verdict="flagged", flagged_leaves=["L2"])
    assert ev["event"] == "spec_validated"
    assert ev["verdict"] == "flagged"
    assert ev["flagged_leaves"] == ["L2"]
    assert "timestamp" in ev
    assert set(ev.keys()) == {"event", "timestamp", "verdict", "flagged_leaves"}


def test_spec_validated_event_defensively_copies_flagged_leaves():
    """Mutating the caller's list afterward must not change the emitted event."""
    leaves = ["L1", "L2"]
    ev = build_spec_validated_event(verdict="ok", flagged_leaves=leaves)
    leaves.append("L3")
    assert ev["flagged_leaves"] == ["L1", "L2"]
