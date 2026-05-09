"""Domain event base class + the (event_type, schema_version) registry.

Subclasses set a class-level `event_type` and `schema_version`. The
registry is populated by `@register_event` so the event store can
look up the correct Pydantic class to validate any payload it loads
back from disk (catching a hand-rolled-dict backdoor — spec §5.6).

Registry key is `(event_type, schema_version)` so multiple shape
versions of the same event_type can coexist during migrations.
Upcasters bridge older versions forward at read time.

`DomainEvent.model_construct` is overridden to raise. The bypass
exists in tests via `BaseModel.model_construct.__func__(cls, ...)`
which is intentionally awkward — production code that calls it is
caught by reviewers and by ruff.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar, Self

from pydantic import BaseModel, ConfigDict

from backend.messaging.envelope import (
    AggregateId,
    EventEnvelope,
    EventId,
)


class InvariantBypassError(Exception):
    """Raised when code attempts `model_construct` on a DomainEvent or
    other invariant-bearing model. Use the validated constructor instead."""


class DomainEvent(BaseModel):
    """Base class for every domain event in our system.

    Subclasses MUST set:
      - event_type: ClassVar[str]   - e.g. "project_created"
      - schema_version: ClassVar[int] - 1 on first introduction; bump
        when payload shape changes (paired with an upcaster registration)

    Subclasses MUST be frozen (model_config frozen=True). Domain events
    are immutable facts.
    """

    model_config = ConfigDict(frozen=True)

    # Subclasses override.
    event_type: ClassVar[str] = ""
    schema_version: ClassVar[int] = 1

    @classmethod
    def model_construct(cls, _fields_set: set[str] | None = None, **values: Any) -> Self:
        """Banned: `model_construct` bypasses Pydantic validation, which
        would let an event payload sidestep invariants like NonEmptyCitations.

        Use `cls(...)` for validated construction or `cls.model_validate(...)`
        for validated deserialization. Tests that genuinely need to bypass
        (rare) call `BaseModel.model_construct.__func__(cls, ...)` which is
        intentionally awkward."""
        raise InvariantBypassError(
            f"{cls.__name__}.model_construct is banned to preserve event "
            f"payload invariants (e.g. NonEmptyCitations). Use {cls.__name__}(...) "
            f"or {cls.__name__}.model_validate(...) instead."
        )


class StoredEvent(BaseModel):
    """An event as it exists in the event store: payload + envelope + position info.

    Yielded by EventStore.load() / load_global() / Subscription.
    """

    model_config = ConfigDict(frozen=True)

    global_position: int
    aggregate_id: AggregateId
    aggregate_type: str
    aggregate_version: int
    event_type: str
    schema_version: int
    payload: dict[str, Any]
    """Validated against the registered DomainEvent subclass on load."""
    envelope: EventEnvelope
    occurred_at: datetime

    def into(self, cls: type[DomainEvent]) -> DomainEvent:
        """Return the payload deserialized into a typed DomainEvent.

        Re-validates via Pydantic — empty-citation backdoors etc. raise here.
        Verifies BOTH event_type and schema_version match `cls`. Mismatched
        schema_version means the caller should consult the upcaster registry
        first rather than load directly into a newer class.
        """
        if cls.event_type != self.event_type:
            raise ValueError(
                f"Cannot deserialize event_type={self.event_type!r} into "
                f"{cls.__name__} (event_type={cls.event_type!r})"
            )
        if cls.schema_version != self.schema_version:
            raise ValueError(
                f"Schema version mismatch deserializing event_type={self.event_type!r}: "
                f"stored={self.schema_version}, target {cls.__name__} v{cls.schema_version}. "
                f"Upcast first."
            )
        return cls.model_validate(self.payload)


# --- Registry --------------------------------------------------------------


_REGISTRY: dict[tuple[str, int], type[DomainEvent]] = {}
"""Keyed by (event_type, schema_version). Multiple shape versions of the
same event_type may coexist during migrations; upcasters bridge them."""


class EventTypeAlreadyRegistered(Exception):
    """Raised when two classes claim the same (event_type, schema_version)."""


def register_event(cls: type[DomainEvent]) -> type[DomainEvent]:
    """Decorator: register a DomainEvent subclass by (event_type, schema_version).

    Used by the event store to look up validators on load. Idempotent
    when the same class is re-registered. Raises if a *different* class
    claims an existing (type, version) pair.

    Example:
        @register_event
        class ProjectCreated(DomainEvent):
            event_type: ClassVar[str] = "project_created"
            schema_version: ClassVar[int] = 1
            project_id: str
            source: dict
    """
    if not cls.event_type:
        raise ValueError(
            f"{cls.__name__} must set a non-empty `event_type` class attribute"
        )
    if cls.schema_version < 1:
        raise ValueError(
            f"{cls.__name__}.schema_version must be >= 1, got {cls.schema_version}"
        )
    key = (cls.event_type, cls.schema_version)
    existing = _REGISTRY.get(key)
    if existing is not None and existing is not cls:
        raise EventTypeAlreadyRegistered(
            f"({cls.event_type!r}, v{cls.schema_version}) already registered to "
            f"{existing.__module__}.{existing.__name__}; cannot reassign to "
            f"{cls.__module__}.{cls.__name__}"
        )
    _REGISTRY[key] = cls
    return cls


def resolve_event_class(event_type: str, schema_version: int = 1) -> type[DomainEvent]:
    """Look up the registered DomainEvent subclass for (event_type, schema_version).

    `schema_version` defaults to 1 for backwards compat with single-version events.
    Raises KeyError if no match — indicating either a never-registered class
    or an upcaster/rename rule is missing for this version.
    """
    try:
        return _REGISTRY[(event_type, schema_version)]
    except KeyError as exc:
        raise KeyError(
            f"No DomainEvent class registered for "
            f"event_type={event_type!r} schema_version={schema_version}. "
            f"Either register it or add an upcaster/rename rule."
        ) from exc


def registered_event_types() -> list[tuple[str, int]]:
    """Sorted list of currently-registered (event_type, schema_version) pairs."""
    return sorted(_REGISTRY.keys())


def _clear_registry_for_tests() -> None:
    """Test-only helper to reset the registry between test cases."""
    _REGISTRY.clear()


__all__ = [
    "DomainEvent",
    "EventTypeAlreadyRegistered",
    "InvariantBypassError",
    "StoredEvent",
    "register_event",
    "registered_event_types",
    "resolve_event_class",
]
