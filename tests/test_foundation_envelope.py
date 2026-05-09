"""Tests for backend.messaging.envelope."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.messaging.envelope import (
    EventEnvelope,
    make_envelope,
    new_correlation_id,
    new_event_id,
)


def test_event_id_is_prefixed_and_sortable():
    a = new_event_id()
    b = new_event_id()
    assert a.startswith("evt_")
    assert b.startswith("evt_")
    assert len(a) == len("evt_") + 26  # ULID is 26 chars
    # ULIDs are lexicographically sortable; b is generated after a.
    assert a < b or a == b  # extremely close-in-time is tolerated


def test_correlation_id_prefix():
    cid = new_correlation_id()
    assert cid.startswith("cor_")


def test_make_envelope_fills_defaults():
    env = make_envelope(source="ingestion.intake")
    assert env.source == "ingestion.intake"
    assert env.causation_id is None
    assert env.event_id.startswith("evt_")
    assert env.correlation_id.startswith("cor_")
    assert env.schema_version == 1


def test_envelope_is_frozen():
    env = make_envelope(source="x")
    with pytest.raises(ValidationError):
        env.source = "y"  # type: ignore[misc]


def test_envelope_accepts_explicit_correlation_for_chaining():
    cid = new_correlation_id()
    e1 = make_envelope(source="a", correlation_id=cid)
    e2 = make_envelope(source="b", correlation_id=cid, causation_id=e1.event_id)  # type: ignore[arg-type]
    assert e1.correlation_id == e2.correlation_id == cid
    assert e2.causation_id == e1.event_id


def test_event_id_is_monotonic_in_same_millisecond():
    """A burst of IDs generated in the same ms must still sort in
    generation order. Without same-ms monotonicity the global event
    log can present reordered concurrent events under high write rates."""
    burst = [new_event_id() for _ in range(2000)]
    # Each id is unique.
    assert len(set(burst)) == len(burst)
    # And lexicographically sorted in generation order.
    assert burst == sorted(burst)


def test_event_id_monotonic_across_thread_concurrency():
    """Same property under contention: 4 threads each generate 500 ids
    in parallel; the union must still be unique. Order across threads
    is not required, but no two threads ever observe the same id."""
    import threading

    out: list[str] = []
    out_lock = threading.Lock()

    def worker() -> None:
        local = [new_event_id() for _ in range(500)]
        with out_lock:
            out.extend(local)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(set(out)) == len(out) == 2000
