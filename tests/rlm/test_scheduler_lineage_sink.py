"""Hermetic controller EventStore adapter tests for scheduler runtime facts."""
from __future__ import annotations

from types import SimpleNamespace

from backend.agents.rlm.scheduler_lineage_sink import BranchTreeEventSink
from backend.eventstore.interface import ConcurrencyError


class _Store:
    def __init__(self, *, conflict_once: bool = False) -> None:
        self.events: list[SimpleNamespace] = []
        self.conflict_once = conflict_once
        self.append_calls = 0

    def load(self, _aggregate_id):
        return iter(self.events)

    def get_aggregate_version(self, _aggregate_id):
        return len(self.events)

    def append(self, *, aggregate_id, aggregate_type, events, expected_version, envelopes):
        self.append_calls += 1
        if self.conflict_once:
            self.conflict_once = False
            raise ConcurrencyError(str(aggregate_id), expected_version, expected_version + 1)
        assert aggregate_type == "branch_tree"
        assert expected_version == len(self.events)
        for event in events:
            self.events.append(SimpleNamespace(event_type=event.event_type, payload=event.model_dump(mode="json")))


def test_maps_runtime_facts_to_registered_branch_lineage_events_and_deduplicates():
    store = _Store()
    sink = BranchTreeEventSink(store, "campaign-1")
    payload = {
        "branch_id": "b1", "branch_type": "faithful", "parent_branch_id": None,
        "rung": 0, "hypothesis_fingerprint": "existing-f10",
    }

    sink("branch_spawned", payload)
    sink("branch_spawned", payload)
    sink("dedup_hit", {"hypothesis_fingerprint": "existing-f10", "existing_branch_id": "b1"})

    assert [item.event_type for item in store.events] == ["branch_spawned", "dedup_hit"]
    assert store.events[0].payload["hypothesis_fingerprint"] == "existing-f10"


def test_retries_optimistic_concurrency_without_duplicate_event():
    store = _Store(conflict_once=True)
    sink = BranchTreeEventSink(store, "campaign-1")

    sink("branch_promoted", {
        "branch_id": "b1", "from_rung": 0, "to_rung": 1, "cohort_ids": ("b1", "b2"),
        "receipt_sha256": "a" * 64, "authority_audit_sha256": "b" * 64,
        "decision_evidence_sha256": "c" * 64,
    })

    assert store.append_calls == 2
    assert [item.event_type for item in store.events] == ["branch_promoted"]
