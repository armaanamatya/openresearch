"""Tests for SqliteSubscription — durable iteration, ack/nack/lease,
multiple consumer_id positions on one subscription_name.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from backend.eventstore.interface import SubscriptionClosedError
from backend.eventstore.sqlite_store import SqliteEventStore
from backend.eventstore.subscription import SqliteSubscription
from backend.messaging.envelope import AggregateId, make_envelope
from backend.messaging.event import (
    DomainEvent,
    _clear_registry_for_tests,
    register_event,
)


@pytest.fixture
def store(tmp_path):
    _clear_registry_for_tests()
    s = SqliteEventStore(f"sqlite:///{tmp_path}/sub.db")
    yield s
    s.close()
    _clear_registry_for_tests()


def _register():
    @register_event
    class Tick(DomainEvent):
        event_type: ClassVar[str] = "tick"
        schema_version: ClassVar[int] = 1
        n: int

    return Tick


def _seed(store, n):
    Tick = _register()
    agg = AggregateId("ticker")
    for i in range(n):
        store.append(agg, "ticker", [Tick(n=i)], expected_version=i,
                     envelopes=[make_envelope(source="seeder")])


def _exit_at_tail_sub(store, name="s", consumer="default", from_pos=None):
    return SqliteSubscription(
        store=store, subscription_name=name, consumer_id=consumer,
        from_position=from_pos, tail_behavior="exit_at_tail",
    )


# --- Iteration --------------------------------------------------------------


def test_subscription_yields_all_events_from_zero(store):
    _seed(store, 5)
    sub = _exit_at_tail_sub(store)
    events = list(sub)
    assert len(events) == 5
    assert [e.payload["n"] for e in events] == [0, 1, 2, 3, 4]


def test_subscription_starts_at_explicit_from_position(store):
    _seed(store, 5)
    # Only events with global_position >= 3 (i.e., the last 3 of 5).
    sub = _exit_at_tail_sub(store, from_pos=3)
    events = list(sub)
    assert [e.global_position for e in events] == [3, 4, 5]


# --- Ack and durable position ----------------------------------------------


def test_ack_advances_persistent_checkpoint(store):
    _seed(store, 5)
    sub1 = _exit_at_tail_sub(store, name="durable", consumer="worker_a")
    iterator = iter(sub1)
    first = next(iterator)
    second = next(iterator)
    sub1.ack(first)
    sub1.ack(second)
    sub1.close()

    # Open a new subscription with the same (name, consumer) — must
    # resume from the last ack.
    sub2 = _exit_at_tail_sub(store, name="durable", consumer="worker_a")
    remaining = list(sub2)
    # We acked positions 1 and 2; the next event is at position 3.
    assert [e.global_position for e in remaining] == [3, 4, 5]


def test_position_property_reflects_ack(store):
    _seed(store, 3)
    sub = _exit_at_tail_sub(store, name="pos", consumer="c1")
    assert sub.position == -1
    iterator = iter(sub)
    e = next(iterator)
    sub.ack(e)
    assert sub.position == e.global_position


# --- Multiple consumer_ids on one subscription_name ------------------------


def test_two_consumers_have_independent_checkpoints(store):
    _seed(store, 3)

    sub_a = _exit_at_tail_sub(store, name="fanout", consumer="proj_a")
    list_a = list(sub_a)
    for e in list_a:
        sub_a.ack(e)
    sub_a.close()

    # New consumer on the same subscription_name starts from -1.
    sub_b = _exit_at_tail_sub(store, name="fanout", consumer="coord_b")
    list_b = list(sub_b)
    assert len(list_b) == 3  # consumer B sees the full log too


# --- Nack and redelivery ---------------------------------------------------


def test_nack_schedules_redelivery_with_delay(store):
    import time

    _seed(store, 1)
    sub = _exit_at_tail_sub(store, name="retry", consumer="c1")
    iterator = iter(sub)
    e = next(iterator)
    # Nack with very short delay so the test runs fast.
    sub.nack(e, retry_after_seconds=0.05)
    sub.close()

    # Before the delay elapses, the event should not redeliver.
    sub_again_quick = _exit_at_tail_sub(store, name="retry", consumer="c1")
    immediate = list(sub_again_quick)
    assert immediate == [], "Nacked event should not redeliver before delay"
    sub_again_quick.close()

    time.sleep(0.1)
    sub_again_after = _exit_at_tail_sub(store, name="retry", consumer="c1")
    redelivered = list(sub_again_after)
    assert len(redelivered) == 1
    assert redelivered[0].payload["n"] == 0


def test_ack_clears_pending_redelivery(store):
    """If an event was nacked then ack'd before its redelivery time,
    it must not redeliver."""
    import time

    _seed(store, 1)
    sub = _exit_at_tail_sub(store, name="retry2", consumer="c1")
    iterator = iter(sub)
    e = next(iterator)
    sub.nack(e, retry_after_seconds=0.5)
    # Now the consumer reconsiders and acks (e.g., a retry on the
    # underlying handler succeeded).
    sub.ack(e)
    sub.close()

    time.sleep(0.6)
    sub_again = _exit_at_tail_sub(store, name="retry2", consumer="c1")
    after = list(sub_again)
    assert after == [], "Ack must clear pending redelivery"


# --- Lease + close ---------------------------------------------------------


def test_renew_lease_does_not_raise_on_open_subscription(store):
    _register()
    sub = _exit_at_tail_sub(store, name="lease", consumer="c1")
    sub.renew_lease(ttl_seconds=10.0)
    sub.close()


def test_subscription_methods_raise_after_close(store):
    _register()
    sub = _exit_at_tail_sub(store, name="closed", consumer="c1")
    sub.close()

    with pytest.raises(SubscriptionClosedError):
        _ = sub.position
    with pytest.raises(SubscriptionClosedError):
        sub.renew_lease()


# --- Ordering during streaming append --------------------------------------


def test_subscription_ignores_events_below_filter_type(store):
    """Type-filtered subscription only yields its types and skips the rest,
    advancing global_position past skipped events on ack."""
    Tick = _register()

    @register_event
    class Tock(DomainEvent):
        event_type: ClassVar[str] = "tock"
        schema_version: ClassVar[int] = 1
        m: int

    agg = AggregateId("mixed")
    store.append(agg, "ticker", [Tick(n=0)], expected_version=0,
                 envelopes=[make_envelope(source="t")])
    store.append(agg, "ticker", [Tock(m=99)], expected_version=1,
                 envelopes=[make_envelope(source="t")])
    store.append(agg, "ticker", [Tick(n=2)], expected_version=2,
                 envelopes=[make_envelope(source="t")])

    sub = SqliteSubscription(
        store=store, subscription_name="filtered", consumer_id="c1",
        types=("tick",), tail_behavior="exit_at_tail",
    )
    received = list(sub)
    assert [e.event_type for e in received] == ["tick", "tick"]
    assert [e.payload["n"] for e in received] == [0, 2]
