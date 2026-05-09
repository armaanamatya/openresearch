"""Event envelope and ID newtypes.

The envelope carries the metadata every stored event needs:
- A unique event_id for idempotent appends.
- A correlation_id pinned to a single user-facing request, propagated
  through every event the request causes.
- A causation_id pointing at the parent event in the chain (or None
  for the root command).
- The wall-clock time the event occurred (UTC).
- The producing module ("ingestion.intake.service", "context.workspace.service").
- A schema_version so upcasters can migrate older events forward.

Newtype IDs are typed strings — `mypy --strict` distinguishes a
ProjectId from a TaskId at compile time even though both are strings
at runtime.
"""

from __future__ import annotations

import secrets
import threading
import time
from datetime import datetime, timezone
from typing import NewType

from pydantic import BaseModel, ConfigDict, Field

EventId = NewType("EventId", str)
CorrelationId = NewType("CorrelationId", str)
CausationId = NewType("CausationId", str)
AggregateId = NewType("AggregateId", str)


# --- ID generation ---------------------------------------------------------
#
# Crockford-base32 ULIDs: 48-bit ms timestamp + 80-bit randomness, sortable.
# Inline to avoid a new dependency. Same-ms monotonic: when two ULIDs are
# generated within the same millisecond, the random tail is incremented
# rather than re-randomized, so they sort in generation order. Clock
# regressions are pinned to the last seen timestamp (NTP backsteps,
# leap-second smearing).

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_RAND_MASK = (1 << 80) - 1
_TS_MASK = (1 << 48) - 1

_ulid_lock = threading.Lock()
_last_ulid_ts: int = 0
_last_ulid_rnd: int = 0


def _ulid() -> str:
    """Produce a 26-character Crockford-base32 ULID, monotonic same-ms.

    First 10 chars: 48-bit timestamp (ms since epoch).
    Last 16 chars: 80-bit randomness, incremented on same-ms collision.
    """
    global _last_ulid_ts, _last_ulid_rnd
    with _ulid_lock:
        ts_ms = int(time.time() * 1000) & _TS_MASK
        if ts_ms < _last_ulid_ts:
            # Clock regressed. Pin to last seen ts so monotonicity holds.
            ts_ms = _last_ulid_ts
        if ts_ms == _last_ulid_ts:
            rnd = (_last_ulid_rnd + 1) & _RAND_MASK
            if rnd == 0:
                # Carry into ts_ms only if the 80-bit space wraps in one ms
                # (impossibly rare in practice; defend anyway).
                ts_ms = (ts_ms + 1) & _TS_MASK
                rnd = int.from_bytes(secrets.token_bytes(10), "big")
        else:
            rnd = int.from_bytes(secrets.token_bytes(10), "big")
        _last_ulid_ts = ts_ms
        _last_ulid_rnd = rnd

    ts_chars = []
    ts = ts_ms
    for _ in range(10):
        ts_chars.append(_CROCKFORD[ts & 0x1F])
        ts >>= 5
    ts_str = "".join(reversed(ts_chars))

    rnd_chars = []
    r = rnd
    for _ in range(16):
        rnd_chars.append(_CROCKFORD[r & 0x1F])
        r >>= 5
    rnd_str = "".join(reversed(rnd_chars))

    return ts_str + rnd_str


def new_event_id() -> EventId:
    return EventId(f"evt_{_ulid()}")


def new_correlation_id() -> CorrelationId:
    return CorrelationId(f"cor_{_ulid()}")


def new_causation_id(from_event: EventId) -> CausationId:
    """Causation IDs are event IDs of the parent event in the chain."""
    return CausationId(from_event)


# --- Envelope --------------------------------------------------------------


class EventEnvelope(BaseModel):
    """Metadata that wraps every domain event in storage.

    The envelope and the payload are stored separately in the event store:
    payload_json holds the domain event, metadata_json holds a serialized
    EventEnvelope. This keeps payload schemas pure.
    """

    model_config = ConfigDict(frozen=True)

    event_id: EventId
    correlation_id: CorrelationId
    causation_id: CausationId | None = None
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: str
    """Producing module path, e.g. 'ingestion.intake.service'."""
    schema_version: int = 1


def make_envelope(
    *,
    source: str,
    correlation_id: CorrelationId | None = None,
    causation_id: CausationId | None = None,
    schema_version: int = 1,
) -> EventEnvelope:
    """Convenience constructor: fills in event_id and occurred_at."""
    return EventEnvelope(
        event_id=new_event_id(),
        correlation_id=correlation_id or new_correlation_id(),
        causation_id=causation_id,
        source=source,
        schema_version=schema_version,
        occurred_at=datetime.now(timezone.utc),
    )


__all__ = [
    "AggregateId",
    "CausationId",
    "CorrelationId",
    "EventEnvelope",
    "EventId",
    "make_envelope",
    "new_causation_id",
    "new_correlation_id",
    "new_event_id",
]
