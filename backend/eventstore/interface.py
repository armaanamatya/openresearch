"""EventStore + Subscription + StoreCapabilities Protocols.

Implementations:
  - `backend.eventstore.sqlite_store.SqliteEventStore` — production default.
  - `backend.eventstore.jsonl_store.JsonlEventStore` — debug / ops dump.
  - Future: EventStoreDBStore, PostgresEventStore, NatsEventStore (fan-out).

The Protocol is **fully sync** to match teammate's persistence layer
(see spec §15.13). Concurrency is achieved by SQLite's WAL mode + per-
aggregate locks. Threading drives parallel projections and coordinators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Protocol, Sequence

from backend.messaging.envelope import AggregateId, EventEnvelope, EventId
from backend.messaging.event import DomainEvent, StoredEvent


# --- Errors ---------------------------------------------------------------


class AppendError(Exception):
    """Base for any failure to append events to the store."""


class ConcurrencyError(AppendError):
    """Raised when expected_version != current aggregate_version.

    `actual_version` reports what the store currently holds so the caller
    can reload the aggregate and retry.
    """

    def __init__(self, aggregate_id: str, expected: int, actual: int) -> None:
        super().__init__(
            f"Concurrency conflict on aggregate {aggregate_id!r}: "
            f"expected version {expected}, found {actual}"
        )
        self.aggregate_id = aggregate_id
        self.expected_version = expected
        self.actual_version = actual


class DuplicateEventError(AppendError):
    """Raised when an `event_id` collides with an existing row.

    A duplicate (same envelope.event_id) is treated as a no-op idempotent
    re-emit only when ALL events in the append batch are duplicates;
    a partial duplicate is an error.
    """


class SubscriptionClosedError(Exception):
    """Raised when a subscription is consumed after `close()`."""


# --- Capabilities ---------------------------------------------------------


@dataclass(frozen=True)
class StoreCapabilities:
    """Capabilities a backend reports about itself.

    Used by callers to choose code paths (e.g., a Postgres backend
    might offer transactional outbox; SQLite uses an in-process bus).
    """

    supports_persistent_subscriptions: bool
    supports_stream_categories: bool
    optimistic_concurrency: bool
    max_event_payload_bytes: int


# --- Append result --------------------------------------------------------


@dataclass(frozen=True)
class AppendResult:
    """Outcome of a successful append."""

    new_aggregate_version: int
    written_event_ids: tuple[EventId, ...]
    written_global_positions: tuple[int, ...]


# --- Subscription ---------------------------------------------------------


class Subscription(Protocol):
    """Persistent subscription with checkpoint, ack/nack, and lease.

    Iterating yields events in `global_position` order from the
    subscription's last checkpoint. `ack()` advances the checkpoint;
    `nack()` returns the event for redelivery after a delay.

    Subscriptions are durable across process restarts: the same
    `(subscription_name, consumer_id)` pair resumes from the last
    ack'd position.

    Ack ordering contract:
      Acks MUST be in `global_position` order. Calling `ack(event_n)`
      without first having ack'd every event up to `event_n - 1` is
      undefined behavior — the implementation MAY raise or silently
      advance, depending on backend. The recommended consumer pattern
      is "iterate, handle one at a time, ack after each, on failure
      nack and stop." Concurrency-tolerant designs use multiple
      consumer_ids on the same subscription_name (each gets its own
      checkpoint) rather than parallel ack on a single subscription.
    """

    @property
    def name(self) -> str: ...

    @property
    def consumer_id(self) -> str:
        """Identifies this consumer of the subscription. Multiple
        consumers on one subscription_name (different consumer_ids)
        get independent checkpoints — used for fan-out / sharding.
        Single-consumer setups use a stable consumer_id like the
        process name or "default"."""

    @property
    def position(self) -> int:
        """Last ack'd global_position for this consumer. -1 means
        never ack'd anything."""

    def __iter__(self) -> Iterator[StoredEvent]: ...

    def ack(self, event: StoredEvent) -> None:
        """Mark `event` as successfully handled. Advances the checkpoint
        if `event.global_position` > current position. Acks must arrive
        in global_position order; see the class docstring."""

    def nack(self, event: StoredEvent, *, retry_after_seconds: float) -> None:
        """Return the event for redelivery. The subscription holds it
        back from this consumer (and only this consumer) until
        `retry_after_seconds` has elapsed."""

    def renew_lease(self, ttl_seconds: float = 30.0) -> None:
        """For long-running handlers: extend the lease that prevents the
        store from reassigning the subscription to another worker.
        `ttl_seconds` is how long the lease should be valid from now."""

    def close(self) -> None:
        """Release the subscription. Subsequent calls raise SubscriptionClosedError."""


# --- EventStore -----------------------------------------------------------


class EventStore(Protocol):
    """Append-only event store.

    Concrete implementations may use SQLite, Postgres, EventStoreDB,
    NATS, or anything else. Callers code against this Protocol.
    """

    @property
    def capabilities(self) -> StoreCapabilities: ...

    def append(
        self,
        aggregate_id: AggregateId,
        aggregate_type: str,
        events: Sequence[DomainEvent],
        expected_version: int,
        envelopes: Sequence[EventEnvelope],
    ) -> AppendResult:
        """Append a batch of events to the aggregate's stream.

        Atomically:
          1. Verify current aggregate_version == expected_version.
          2. Re-validate every event against its registered Pydantic
             class (catches model_construct/dict-bypass backdoors).
          3. Reject if any event_id collides AND not all are duplicates.
          4. Insert all events at consecutive aggregate_versions.

        Args:
          aggregate_id: stream identity (one stream per aggregate).
          aggregate_type: short name for indexing/filtering ("project").
          events: domain events to append. Must equal len(envelopes).
          expected_version: caller's view of the aggregate's current
            version. Use 0 for a fresh aggregate.
          envelopes: per-event metadata (event_id, correlation_id, ...).
            Order matches `events`.

        Raises:
          ConcurrencyError if expected_version is stale.
          DuplicateEventError if some-but-not-all events are duplicates.
          AppendError on payload validation failure.
        """

    def load(
        self,
        aggregate_id: AggregateId,
        from_version: int = 0,
    ) -> Iterator[StoredEvent]:
        """Yield this aggregate's events in version order, starting at
        from_version. Each yielded payload is re-validated against its
        registered Pydantic class."""

    def load_global(
        self,
        from_position: int = 0,
        to_position: int | None = None,
        types: Iterable[str] | None = None,
        batch_size: int = 1000,
    ) -> Iterator[StoredEvent]:
        """Yield events across all aggregates in global_position order.

        Used for projection rebuild and audit replays. `types` filters
        by event_type when set."""

    def subscribe(
        self,
        subscription_name: str,
        *,
        consumer_id: str = "default",
        types: Iterable[str] | None = None,
        from_position: int | None = None,
    ) -> Subscription:
        """Open (or resume) a durable subscription.

        `(subscription_name, consumer_id)` together uniquely identify a
        consumer position. The same pair across process restarts resumes
        from the last ack'd position. Using different `consumer_id`s on
        the same `subscription_name` gives each consumer its own
        checkpoint (used for fan-out: a projection and a coordinator can
        share a subscription_name but never block each other).

        `from_position` overrides the resumed checkpoint for this open;
        useful for explicit re-replay during projection rebuilds. Pass
        `from_position=0` to start from the beginning of the log."""

    def get_aggregate_version(self, aggregate_id: AggregateId) -> int:
        """Current version of the aggregate. 0 if no events yet."""


__all__ = [
    "AppendError",
    "AppendResult",
    "ConcurrencyError",
    "DuplicateEventError",
    "EventStore",
    "StoreCapabilities",
    "Subscription",
    "SubscriptionClosedError",
]
