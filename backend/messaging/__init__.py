"""Messaging primitives: events, commands, envelope, idempotency, bus.

These types form the contract every aggregate, application service, and
coordinator speaks. They are intentionally infrastructure-free: an
event/command instance does not know about SQLite, listeners, or HTTP.
The event store and bus consume these types.
"""

from backend.messaging.bus import DomainEventBus, EventListener, ListenerErrorHook
from backend.messaging.command import Command, CommandId
from backend.messaging.envelope import (
    AggregateId,
    CausationId,
    CorrelationId,
    EventEnvelope,
    EventId,
    new_event_id,
    new_correlation_id,
)
from backend.messaging.event import (
    DomainEvent,
    InvariantBypassError,
    StoredEvent,
    register_event,
    resolve_event_class,
)
from backend.messaging.idempotency import DuplicateCommandError, IdempotencyTable

__all__ = [
    "AggregateId",
    "CausationId",
    "Command",
    "CommandId",
    "CorrelationId",
    "DomainEvent",
    "DomainEventBus",
    "DuplicateCommandError",
    "EventEnvelope",
    "EventId",
    "EventListener",
    "IdempotencyTable",
    "InvariantBypassError",
    "ListenerErrorHook",
    "StoredEvent",
    "new_correlation_id",
    "new_event_id",
    "register_event",
    "resolve_event_class",
]
