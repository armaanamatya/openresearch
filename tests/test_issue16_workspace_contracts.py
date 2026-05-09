"""Issue #16 — RLM context workspace, citation tracking, and context-enrichment events.

The Workspace service is not implemented yet, but this file exercises the
foundation contracts #16 will rely on:

  - The citation invariant survives event construction AND projection-side
    deserialization (StoredEvent.into) — defense in depth (spec §5.6).
  - The DomainEventBus delivers context-enrichment events to listeners,
    which is what the dashboard bridge will consume.
  - The Scope enum carries the three values that PRD §1078-1082 requires
    and that the existing BlackboardService stores as strings.
  - Variable promotion semantics (scope transitions) are representable.

When WorkspaceAppService lands, integration tests join this file.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
from pydantic import ValidationError

from backend.messaging.bus import DomainEventBus
from backend.messaging.envelope import AggregateId, make_envelope
from backend.messaging.event import (
    DomainEvent,
    StoredEvent,
    _clear_registry_for_tests,
    register_event,
)
from backend.schemas.citations import Citation, NonEmptyCitations
from backend.schemas.scope import Scope


@pytest.fixture(autouse=True)
def _clean_registry():
    _clear_registry_for_tests()
    yield
    _clear_registry_for_tests()


# --- Citation invariant: defense in depth -----------------------------------


def test_variable_enriched_with_no_citations_fails_construction():
    """An agent's claim must carry evidence at the moment it becomes an event.
    This is the first defense layer — the Pydantic constructor."""

    @register_event
    class VariableEnrichedLike(DomainEvent):
        event_type: ClassVar[str] = "variable_enriched_like"
        schema_version: ClassVar[int] = 1
        workspace_id: str
        variable_name: str
        value_payload: dict
        citations: NonEmptyCitations

    with pytest.raises(ValidationError):
        VariableEnrichedLike(
            workspace_id="ws_1",
            variable_name="env_python_version",
            value_payload={"version": "3.11"},
            citations=(),  # type: ignore[arg-type]
        )


def test_stored_event_validates_citations_on_into():
    """Second defense layer: a hand-rolled dict bypassing the constructor
    is caught when the projection materializes it via StoredEvent.into().
    Stored payloads with empty citations must NOT produce a typed event."""

    @register_event
    class VariableEnrichedLike(DomainEvent):
        event_type: ClassVar[str] = "variable_enriched_like_2"
        schema_version: ClassVar[int] = 1
        workspace_id: str
        variable_name: str
        value_payload: dict
        citations: NonEmptyCitations

    env = make_envelope(source="test")
    stored = StoredEvent(
        global_position=1,
        aggregate_id=AggregateId("ws_1"),
        aggregate_type="workspace",
        aggregate_version=1,
        event_type="variable_enriched_like_2",
        schema_version=1,
        # Adversarial payload: simulating a hand-rolled dict that bypassed
        # the Pydantic constructor (e.g., direct SQL insert by a careless
        # tool). The into() re-validation must catch this.
        payload={
            "workspace_id": "ws_1",
            "variable_name": "x",
            "value_payload": {},
            "citations": [],
        },
        envelope=env,
        occurred_at=env.occurred_at,
    )
    with pytest.raises(ValidationError):
        stored.into(VariableEnrichedLike)


def test_stored_event_with_proper_citations_materializes_correctly():
    """The happy path: a well-formed event flows from store to typed."""

    @register_event
    class CitationAttachedLike(DomainEvent):
        event_type: ClassVar[str] = "citation_attached_like"
        schema_version: ClassVar[int] = 1
        workspace_id: str
        decision_id: str
        citations: NonEmptyCitations

    env = make_envelope(source="test")
    stored = StoredEvent(
        global_position=1,
        aggregate_id=AggregateId("ws_1"),
        aggregate_type="workspace",
        aggregate_version=1,
        event_type="citation_attached_like",
        schema_version=1,
        payload={
            "workspace_id": "ws_1",
            "decision_id": "dec_42",
            "citations": [
                {
                    "source_id": "src_paper",
                    "chunk_id": None,
                    "quote": "we use γ=0.99",
                    "locator": "PPO §4.2",
                    "confidence": 1.0,
                }
            ],
        },
        envelope=env,
        occurred_at=env.occurred_at,
    )
    typed = stored.into(CitationAttachedLike)
    assert isinstance(typed, CitationAttachedLike)
    assert typed.decision_id == "dec_42"
    assert len(typed.citations) == 1
    assert typed.citations[0].locator == "PPO §4.2"


# --- Context-enrichment event delivery via DomainEventBus -------------------


def test_dashboard_bridge_receives_context_enrichment_events():
    """The DomainEventBus is what the dashboard bridge subscribes to to
    translate our domain events into the existing EventPayload broadcast.
    Verify a context-enrichment event reaches the listener."""

    @register_event
    class ContextEnriched(DomainEvent):
        event_type: ClassVar[str] = "context_enriched"
        schema_version: ClassVar[int] = 1
        workspace_id: str
        variable_name: str
        citations: NonEmptyCitations

    bus = DomainEventBus()
    received: list[StoredEvent] = []
    bus.subscribe(received.append)

    env = make_envelope(source="context.workspace.service")
    payload_event = ContextEnriched(
        workspace_id="ws_1",
        variable_name="env_python_version",
        citations=(Citation(source_id="src_repo", quote="3.11", locator="reqs.txt:1"),),
    )
    stored = StoredEvent(
        global_position=1,
        aggregate_id=AggregateId("ws_1"),
        aggregate_type="workspace",
        aggregate_version=1,
        event_type="context_enriched",
        schema_version=1,
        payload=payload_event.model_dump(),
        envelope=env,
        occurred_at=env.occurred_at,
    )
    bus.emit(stored)

    assert len(received) == 1
    assert received[0].event_type == "context_enriched"
    typed = received[0].into(ContextEnriched)
    assert typed.variable_name == "env_python_version"


# --- Scope enum: matches blackboard service strings -------------------------


def test_scope_enum_values_match_blackboard_strings():
    """BlackboardService stores scope as a raw string; our Scope enum must
    serialize to those exact values for cross-team interop (spec §15.13)."""
    assert Scope.private_to_parent.value == "private_to_parent"
    assert Scope.branch_shared.value == "branch_shared"
    assert Scope.global_verified.value == "global_verified"


def test_scope_promotion_path_is_representable():
    """Variable promotion (#16's VariablePromoted event) goes
    private_to_parent → branch_shared → global_verified. Verify
    each transition has a typed Scope on each end."""

    @register_event
    class VariablePromotedLike(DomainEvent):
        event_type: ClassVar[str] = "variable_promoted_like"
        schema_version: ClassVar[int] = 1
        workspace_id: str
        variable_name: str
        from_scope: Scope
        to_scope: Scope

    promo = VariablePromotedLike(
        workspace_id="ws_1",
        variable_name="env_python_version",
        from_scope=Scope.branch_shared,
        to_scope=Scope.global_verified,
    )
    assert promo.from_scope == Scope.branch_shared
    assert promo.to_scope == Scope.global_verified
    # Round-trip through dict (what storage looks like) preserves the values.
    dumped = promo.model_dump()
    assert dumped["from_scope"] == "branch_shared"
    assert dumped["to_scope"] == "global_verified"


def test_scope_string_is_distinct_from_int_or_other_enum():
    """Defensive: the Scope enum is a str-enum, so comparison to int fails
    as expected."""
    assert Scope.global_verified != 1
    assert Scope.global_verified != "GLOBAL_VERIFIED"  # different casing
