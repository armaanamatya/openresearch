"""Controller-owned EventStore adapter for durable scheduler-tree facts.

``SchedulerTreeRuntime`` calls this adapter only *after* its local write-ahead
projection has committed.  It maps the runtime's narrow event vocabulary onto
the existing branch-lineage ``DomainEvent`` schema, retries ordinary optimistic
concurrency, and treats an already-stored equal payload as an idempotent
delivery rather than creating a second history fact.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from backend.agents.rlm.branch_lineage import (
    BranchPromoted,
    BranchRevived,
    BranchSpawned,
    BranchTrueKilled,
    DedupHit,
    FrozenPoolEviction,
    RungClimbed,
    branch_tree_aggregate_id,
)
from backend.eventstore.interface import ConcurrencyError, EventStore
from backend.messaging.envelope import AggregateId, CorrelationId, EventEnvelope, new_event_id
from backend.messaging.event import DomainEvent


class BranchTreeEventSink:
    """Append scheduler facts to ``branch-tree:<campaign_id>`` idempotently."""

    def __init__(self, store: EventStore, campaign_id: str, *, retries: int = 3) -> None:
        if not campaign_id:
            raise ValueError("campaign_id is required")
        if type(retries) is not int or retries < 1:
            raise ValueError("retries must be a positive integer")
        self.store = store
        self.campaign_id = campaign_id
        self.retries = retries
        self.aggregate_id = AggregateId(branch_tree_aggregate_id(campaign_id))

    def __call__(self, kind: str, payload: Mapping[str, Any]) -> None:
        event = _to_domain_event(kind, payload)
        expected_payload = event.model_dump(mode="json")
        for _ in range(self.retries):
            stored = tuple(self.store.load(self.aggregate_id))
            if any(item.event_type == event.event_type and item.payload == expected_payload for item in stored):
                return
            try:
                self.store.append(
                    aggregate_id=self.aggregate_id,
                    aggregate_type="branch_tree",
                    events=[event],
                    expected_version=self.store.get_aggregate_version(self.aggregate_id),
                    envelopes=[EventEnvelope(
                        event_id=new_event_id(),
                        correlation_id=CorrelationId(self.campaign_id),
                        source="agents.rlm.scheduler_lineage_sink",
                    )],
                )
                return
            except ConcurrencyError:
                continue
        raise RuntimeError("scheduler branch-tree event append exhausted concurrency retries")


def _to_domain_event(kind: str, payload: Mapping[str, Any]) -> DomainEvent:
    """Validate runtime facts by constructing the registered event class."""
    data = dict(payload)
    if kind == "branch_spawned":
        return BranchSpawned(**data)
    if kind == "rung_climbed":
        return RungClimbed(**data)
    if kind == "branch_promoted":
        return BranchPromoted(**data)
    if kind == "frozen_pool_eviction":
        return FrozenPoolEviction(**data)
    if kind == "branch_revived":
        return BranchRevived(**data)
    if kind == "branch_true_killed":
        return BranchTrueKilled(**data)
    if kind == "dedup_hit":
        return DedupHit(**data)
    raise ValueError(f"unknown scheduler lineage event {kind!r}")
