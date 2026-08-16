"""Event-sourced lineage for the reproduction scheduler's branch tree.

New ``DomainEvent`` types recording every scheduler transition on a dedicated
``branch-tree:<campaign_id>`` aggregate (reusing the existing SQLite CQRS event
store + optimistic concurrency). These are the persistence backbone for:

- **backtracking** — resume a subtree from any recorded rung/checkpoint,
- the **revivable frozen pool** — ``FrozenPoolEviction`` + ``BranchRevived``,
- **anti-repetition dedup** — ``DedupHit`` records a collision against the
  *existing* novelty fingerprint (campaign F10 machinery); this module does NOT
  compute a new fingerprint.

Design: ``docs/history/specs/2026-07-18-dynamic-reproduction-scheduler-design.md`` §7.1.
With ``OPENRESEARCH_SCHEDULER_TREE`` enabled, the serial campaign emits only a
durable root ``BranchSpawned`` after its write-ahead launch row has recorded the
existing F10 fingerprint.  The remaining transition types (including `DedupHit`,
whose rejected candidate lacks a durable branch owner) stay inert until their real
checkpoint/receipt/branch-queue owners exist; a shadow advisory is never an event
source for them.
"""
from __future__ import annotations

from typing import ClassVar

from backend.agents.rlm.asha_scheduler import BranchType
from backend.messaging.event import DomainEvent, register_event


def branch_tree_aggregate_id(campaign_id: str) -> str:
    """The event-store aggregate stream for a campaign's branch tree."""
    return f"branch-tree:{campaign_id}"


@register_event
class BranchSpawned(DomainEvent):
    """A new typed branch was created (root has ``parent_branch_id=None``)."""

    event_type: ClassVar[str] = "branch_spawned"
    schema_version: ClassVar[int] = 1

    branch_id: str
    branch_type: BranchType = "faithful"
    parent_branch_id: str | None = None
    rung: int = 0
    hypothesis_fingerprint: str


@register_event
class RungClimbed(DomainEvent):
    """A branch trained the step-slice for one rung and was measured."""

    event_type: ClassVar[str] = "rung_climbed"
    schema_version: ClassVar[int] = 1

    branch_id: str
    rung: int
    step_budget: int
    measured_score: float | None = None
    gpu_usd: float = 0.0
    resume_from_ckpt: str | None = None


@register_event
class BranchPromoted(DomainEvent):
    """A branch survived the rung's halving and climbs to the next rung."""

    event_type: ClassVar[str] = "branch_promoted"
    schema_version: ClassVar[int] = 1

    branch_id: str
    from_rung: int
    to_rung: int
    cohort_ids: tuple[str, ...] = ()
    # Authority actions are bound to both the verified rung receipt and the
    # exact durable audit payload.  Defaults preserve the historical event
    # schema for shadow-only / pre-authority streams.
    receipt_sha256: str | None = None
    authority_audit_sha256: str | None = None
    decision_evidence_sha256: str | None = None


@register_event
class FrozenPoolEviction(DomainEvent):
    """An underperforming-but-faithful branch was checkpointed + evicted to the
    revivable frozen pool (reversible — the mainline eviction path, not a kill)."""

    event_type: ClassVar[str] = "frozen_pool_eviction"
    schema_version: ClassVar[int] = 1

    branch_id: str
    rung: int
    ckpt_uri: str
    reason: str
    receipt_sha256: str | None = None
    authority_audit_sha256: str | None = None
    decision_evidence_sha256: str | None = None


@register_event
class BranchRevived(DomainEvent):
    """A frozen branch was resumed from its checkpoint."""

    event_type: ClassVar[str] = "branch_revived"
    schema_version: ClassVar[int] = 1

    branch_id: str
    from_ckpt: str


@register_event
class BranchTrueKilled(DomainEvent):
    """A branch was true-deleted for provable breakage (the only irreversible
    kill; ``termination_cause`` is a structured breakage code, never free text)."""

    event_type: ClassVar[str] = "branch_true_killed"
    schema_version: ClassVar[int] = 1

    branch_id: str
    termination_cause: str
    # The measured fidelity rung is mandatory for authority proof but optional
    # for historical shadow streams; the gate rejects an applied kill when it
    # is absent/mismatched.
    rung: int | None = None
    receipt_sha256: str | None = None
    authority_audit_sha256: str | None = None
    decision_evidence_sha256: str | None = None


@register_event
class DedupHit(DomainEvent):
    """A would-be branch collided with one the tree already explored — recorded
    against the existing novelty fingerprint (campaign F10), not a new one."""

    event_type: ClassVar[str] = "dedup_hit"
    schema_version: ClassVar[int] = 1

    hypothesis_fingerprint: str
    existing_branch_id: str
