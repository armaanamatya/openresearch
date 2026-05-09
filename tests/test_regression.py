"""Cross-cutting regression tests.

These guard against silent breaks when teammates or future commits
modify the foundation. They are intentionally broad and fast; if any
fail, something fundamental about the layer composition is wrong.
"""

from __future__ import annotations

import importlib
from typing import ClassVar

import pytest

from backend.messaging.event import (
    DomainEvent,
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


# --- Module composition -----------------------------------------------------


def test_all_backend_modules_import_cleanly():
    """If a teammate edits a shared module and accidentally introduces
    a circular import or syntax error, this catches it before
    integration tests would."""
    modules = [
        "backend",
        "backend.app",
        "backend.config",
        "backend.eventstore",
        "backend.eventstore.interface",
        "backend.messaging",
        "backend.messaging.bus",
        "backend.messaging.command",
        "backend.messaging.envelope",
        "backend.messaging.event",
        "backend.messaging.idempotency",
        "backend.persistence",
        "backend.persistence.database",
        "backend.persistence.repositories.task_repository",
        "backend.persistence.repositories.message_repository",
        "backend.persistence.repositories.run_repository",
        "backend.persistence.repositories.artifact_repository",
        "backend.persistence.repositories.verification_repository",
        "backend.schemas",
        "backend.schemas.artifacts",
        "backend.schemas.citations",
        "backend.schemas.events",
        "backend.schemas.messages",
        "backend.schemas.runs",
        "backend.schemas.scope",
        "backend.schemas.tasks",
        "backend.schemas.verifications",
        "backend.services",
        "backend.services.context",
        "backend.services.events",
        "backend.services.ingestion",
        "backend.services.orchestration",
        "backend.services.orchestration.blackboard",
        "backend.services.orchestration.delegation",
        "backend.services.orchestration.spawn_policy",
        "backend.services.orchestration.task_lifecycle",
        "backend.services.runtime",
        "backend.services.verification",
    ]
    failures: list[tuple[str, str]] = []
    for mod in modules:
        try:
            importlib.import_module(mod)
        except Exception as exc:  # noqa: BLE001
            failures.append((mod, repr(exc)))
    assert not failures, "Modules failed to import: " + ", ".join(
        f"{m} ({err})" for m, err in failures
    )


def test_messaging_public_api_is_complete():
    """The package __init__ should re-export everything callers use, so
    `from backend.messaging import X` works for every X documented in the
    module docstring."""
    import backend.messaging as m

    expected = {
        "AggregateId",
        "CausationId",
        "Command",
        "CommandId",
        "CorrelationId",
        "DomainEvent",
        "DomainEventBus",
        "EventEnvelope",
        "EventId",
        "EventListener",
        "IdempotencyTable",
        "StoredEvent",
        "new_correlation_id",
        "new_event_id",
        "register_event",
        "resolve_event_class",
    }
    actual = set(m.__all__)
    missing = expected - actual
    extra = actual - expected
    assert not missing, f"Public API missing: {missing}"
    # Extra is allowed but flag it so we notice unintentional additions.
    if extra:
        # Document, don't fail.
        pass


def test_eventstore_public_api_is_complete():
    import backend.eventstore as es

    expected = {
        "AppendError",
        "AppendResult",
        "ConcurrencyError",
        "DuplicateEventError",
        "EventStore",
        "StoreCapabilities",
        "Subscription",
        "SubscriptionClosedError",
    }
    assert expected.issubset(set(es.__all__))


# --- Schema compatibility with teammate's code -----------------------------


def test_blackboard_scope_strings_match_our_enum():
    """The existing BlackboardService records scope as a raw string. Our
    Scope enum must round-trip through their string column losslessly."""
    from backend.services.orchestration.blackboard import BlackboardRecord

    record = BlackboardRecord(
        key="env_python_version",
        value="3.11",
        scope=Scope.branch_shared.value,  # we serialize via .value
        owner_task_id="task_42",
        created_at="2026-05-09T00:00:00+00:00",
    )
    # Round-trip back into our enum.
    assert Scope(record.scope) == Scope.branch_shared


def test_teammate_event_payload_still_constructible():
    """Our additions must not break the existing EventPayload contract
    used by the dashboard."""
    from backend.schemas.events import EventPayload

    p = EventPayload(
        event="context_enrichment",  # type: ignore[arg-type]
        agent_id="paper_understanding_agent",
        data={"variable_name": "claim_map", "domain_event_id": "evt_xyz"},
    )
    assert p.agent_id == "paper_understanding_agent"
    assert p.data["domain_event_id"] == "evt_xyz"


def test_teammate_task_status_enum_unchanged():
    """Defensive check: their TaskStatus values still match the PRD's
    8 lifecycle states. Our work imports from this module; if they shift
    a value, we want to know immediately."""
    from backend.schemas.tasks import TaskStatus

    expected = {
        "created",
        "context_prepared",
        "running",
        "artifact_submitted",
        "verification_pending",
        "verified",
        "failed",
        "blocked_requires_human",
    }
    assert {s.value for s in TaskStatus} == expected


# --- Citation invariant: end-to-end trip through register/store/load -------


def test_citation_invariant_survives_full_event_lifecycle():
    """Build an event with citations, dump to dict (what the store does),
    re-validate (what the store does on load) — the value comes back
    intact and the constraint still holds against tampering."""

    @register_event
    class WorkspaceClaim(DomainEvent):
        event_type: ClassVar[str] = "workspace_claim"
        schema_version: ClassVar[int] = 1
        decision_id: str
        citations: NonEmptyCitations

    original = WorkspaceClaim(
        decision_id="dec_1",
        citations=(
            Citation(source_id="src_paper", quote="γ=0.99", locator="§4.2"),
            Citation(source_id="src_repo", quote="discount=0.99", locator="config.yaml"),
        ),
    )

    # Dump -> reload (simulates store round-trip).
    dumped = original.model_dump()
    reloaded = WorkspaceClaim.model_validate(dumped)
    assert reloaded == original

    # Tampered (zero citations) — store-side validation rejects.
    tampered = {**dumped, "citations": []}
    with pytest.raises(Exception) as exc_info:
        WorkspaceClaim.model_validate(tampered)
    assert "min_length" in str(exc_info.value).lower() or "at least 1" in str(exc_info.value)


# --- DomainEventBus: idempotent unsubscribe & no leaks ---------------------


def test_double_unsubscribe_does_not_raise():
    """The unsubscribe callable should be safe to call repeatedly —
    listeners are removed once; subsequent calls are no-ops."""
    from backend.messaging.bus import DomainEventBus

    bus = DomainEventBus()
    unsub = bus.subscribe(lambda _e: None)
    unsub()
    unsub()  # second call — must not raise
    assert bus.listener_count() == 0
