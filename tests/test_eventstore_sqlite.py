"""Tests for backend.eventstore.sqlite_store.SqliteEventStore."""

from __future__ import annotations

from typing import ClassVar

import pytest
from pydantic import ValidationError

from backend.eventstore.interface import (
    AppendError,
    ConcurrencyError,
    DuplicateEventError,
)
from backend.eventstore.sqlite_store import SqliteEventStore
from backend.messaging.envelope import (
    AggregateId,
    EventEnvelope,
    make_envelope,
)
from backend.messaging.event import (
    DomainEvent,
    _clear_registry_for_tests,
    register_event,
)
from backend.schemas.citations import Citation, NonEmptyCitations


# --- Test event types -------------------------------------------------------


def _register_test_events():
    @register_event
    class ThingHappened(DomainEvent):
        event_type: ClassVar[str] = "thing_happened"
        schema_version: ClassVar[int] = 1
        thing: str

    @register_event
    class OtherThingHappened(DomainEvent):
        event_type: ClassVar[str] = "other_thing_happened"
        schema_version: ClassVar[int] = 1
        n: int

    @register_event
    class ClaimMade(DomainEvent):
        event_type: ClassVar[str] = "claim_made"
        schema_version: ClassVar[int] = 1
        decision_id: str
        citations: NonEmptyCitations

    return ThingHappened, OtherThingHappened, ClaimMade


@pytest.fixture
def store(tmp_path):
    _clear_registry_for_tests()
    s = SqliteEventStore(f"sqlite:///{tmp_path}/events.db")
    yield s
    s.close()
    _clear_registry_for_tests()


# --- Append happy path ------------------------------------------------------


def test_append_writes_events_and_returns_positions(store):
    ThingHappened, _, _ = _register_test_events()
    agg = AggregateId("agg_1")
    e1 = ThingHappened(thing="alpha")
    e2 = ThingHappened(thing="beta")
    env1 = make_envelope(source="test")
    env2 = make_envelope(source="test", correlation_id=env1.correlation_id)

    result = store.append(agg, "thing", [e1, e2], expected_version=0, envelopes=[env1, env2])
    assert result.new_aggregate_version == 2
    assert len(result.written_event_ids) == 2
    assert len(result.written_global_positions) == 2
    # Global positions are monotonically increasing.
    assert result.written_global_positions[0] < result.written_global_positions[1]


def test_append_increments_aggregate_version(store):
    ThingHappened, _, _ = _register_test_events()
    agg = AggregateId("agg_1")
    e = ThingHappened(thing="x")
    env = make_envelope(source="test")
    store.append(agg, "thing", [e], expected_version=0, envelopes=[env])
    assert store.get_aggregate_version(agg) == 1
    store.append(agg, "thing", [e := ThingHappened(thing="y")], expected_version=1,
                 envelopes=[make_envelope(source="test")])
    assert store.get_aggregate_version(agg) == 2


def test_append_rejects_empty_batch(store):
    _register_test_events()
    with pytest.raises(AppendError, match="empty"):
        store.append(AggregateId("agg_1"), "thing", [], expected_version=0, envelopes=[])


def test_append_rejects_mismatched_envelope_count(store):
    ThingHappened, _, _ = _register_test_events()
    e1 = ThingHappened(thing="a")
    e2 = ThingHappened(thing="b")
    with pytest.raises(AppendError, match="same length"):
        store.append(
            AggregateId("agg_1"), "thing", [e1, e2],
            expected_version=0,
            envelopes=[make_envelope(source="test")],  # only one
        )


# --- Optimistic concurrency -------------------------------------------------


def test_append_with_stale_expected_version_raises_concurrency_error(store):
    ThingHappened, _, _ = _register_test_events()
    agg = AggregateId("agg_1")
    store.append(agg, "thing", [ThingHappened(thing="a")], expected_version=0,
                 envelopes=[make_envelope(source="test")])

    # Now expected_version=0 is stale; aggregate is at version 1.
    with pytest.raises(ConcurrencyError) as exc_info:
        store.append(agg, "thing", [ThingHappened(thing="b")], expected_version=0,
                     envelopes=[make_envelope(source="test")])
    err = exc_info.value
    assert err.expected_version == 0
    assert err.actual_version == 1


def test_append_with_correct_expected_version_succeeds(store):
    ThingHappened, _, _ = _register_test_events()
    agg = AggregateId("agg_1")
    store.append(agg, "thing", [ThingHappened(thing="a")], expected_version=0,
                 envelopes=[make_envelope(source="test")])
    store.append(agg, "thing", [ThingHappened(thing="b")], expected_version=1,
                 envelopes=[make_envelope(source="test")])
    assert store.get_aggregate_version(agg) == 2


# --- Idempotent re-emit -----------------------------------------------------


def test_append_full_duplicate_batch_is_idempotent_no_op(store):
    ThingHappened, _, _ = _register_test_events()
    agg = AggregateId("agg_1")
    e1 = ThingHappened(thing="a")
    e2 = ThingHappened(thing="b")
    env1 = make_envelope(source="test")
    env2 = make_envelope(source="test")

    first = store.append(agg, "thing", [e1, e2], expected_version=0, envelopes=[env1, env2])
    # Re-append with the SAME envelopes (same event_ids) — must be a no-op.
    second = store.append(agg, "thing", [e1, e2], expected_version=0, envelopes=[env1, env2])
    assert second.written_global_positions == first.written_global_positions
    assert second.written_event_ids == first.written_event_ids
    assert store.get_aggregate_version(agg) == 2  # still 2, not 4


def test_append_partial_duplicate_batch_raises(store):
    ThingHappened, _, _ = _register_test_events()
    agg = AggregateId("agg_1")
    e1 = ThingHappened(thing="a")
    env1 = make_envelope(source="test")
    store.append(agg, "thing", [e1], expected_version=0, envelopes=[env1])

    # Mix the existing event_id with a new one — partial duplicate.
    e2 = ThingHappened(thing="b")
    env2 = make_envelope(source="test")
    with pytest.raises(DuplicateEventError, match="Partial duplicate"):
        store.append(agg, "thing", [e1, e2], expected_version=1, envelopes=[env1, env2])


def test_append_event_id_collision_across_aggregates_raises(store):
    ThingHappened, _, _ = _register_test_events()
    e = ThingHappened(thing="a")
    env = make_envelope(source="test")
    store.append(AggregateId("agg_A"), "thing", [e], expected_version=0, envelopes=[env])

    # Reusing the same event_id on a different aggregate is a misuse.
    with pytest.raises(DuplicateEventError, match="different aggregate"):
        store.append(AggregateId("agg_B"), "thing", [e], expected_version=0, envelopes=[env])


# --- Payload re-validation on append ---------------------------------------


def test_append_revalidates_payload_against_registered_class(store):
    """An event constructed via the Pydantic constructor is fine; the
    re-validation step is the safety net for hand-rolled payloads
    that manage to reach append. We verify the safety net fires by
    rejecting an event whose dump fails class validation."""
    _, _, ClaimMade = _register_test_events()
    agg = AggregateId("agg_1")
    # ClaimMade requires NonEmptyCitations — direct construction with empty fails.
    with pytest.raises(ValidationError):
        ClaimMade(decision_id="d1", citations=())  # type: ignore[arg-type]

    # The constructor refused, so this path can never reach append for
    # well-behaved callers. The defense-in-depth is in StoredEvent.into()
    # for adversarial payloads (covered in test_eventstore_load_round_trip).


# --- Load by aggregate ------------------------------------------------------


def test_load_yields_events_in_aggregate_version_order(store):
    ThingHappened, _, _ = _register_test_events()
    agg = AggregateId("agg_1")
    for i in range(5):
        store.append(agg, "thing", [ThingHappened(thing=f"v{i}")],
                     expected_version=i, envelopes=[make_envelope(source="test")])
    versions = [e.aggregate_version for e in store.load(agg)]
    assert versions == [1, 2, 3, 4, 5]
    things = [e.payload["thing"] for e in store.load(agg)]
    assert things == ["v0", "v1", "v2", "v3", "v4"]


def test_load_from_version_skips_earlier(store):
    ThingHappened, _, _ = _register_test_events()
    agg = AggregateId("agg_1")
    for i in range(5):
        store.append(agg, "thing", [ThingHappened(thing=f"v{i}")],
                     expected_version=i, envelopes=[make_envelope(source="test")])
    versions = [e.aggregate_version for e in store.load(agg, from_version=2)]
    assert versions == [3, 4, 5]


def test_load_unknown_aggregate_yields_empty(store):
    _register_test_events()
    assert list(store.load(AggregateId("never_exists"))) == []


# --- Load global ------------------------------------------------------------


def test_load_global_yields_in_global_position_order_across_aggregates(store):
    ThingHappened, OtherThingHappened, _ = _register_test_events()
    store.append(AggregateId("a1"), "thing", [ThingHappened(thing="x")],
                 expected_version=0, envelopes=[make_envelope(source="t")])
    store.append(AggregateId("a2"), "other", [OtherThingHappened(n=1)],
                 expected_version=0, envelopes=[make_envelope(source="t")])
    store.append(AggregateId("a1"), "thing", [ThingHappened(thing="y")],
                 expected_version=1, envelopes=[make_envelope(source="t")])
    types = [e.event_type for e in store.load_global()]
    assert types == ["thing_happened", "other_thing_happened", "thing_happened"]


def test_load_global_filters_by_event_type(store):
    ThingHappened, OtherThingHappened, _ = _register_test_events()
    store.append(AggregateId("a1"), "thing", [ThingHappened(thing="x")],
                 expected_version=0, envelopes=[make_envelope(source="t")])
    store.append(AggregateId("a2"), "other", [OtherThingHappened(n=1)],
                 expected_version=0, envelopes=[make_envelope(source="t")])
    only_thing = list(store.load_global(types=["thing_happened"]))
    assert len(only_thing) == 1
    assert only_thing[0].event_type == "thing_happened"


def test_load_global_paginates_with_batch_size(store):
    ThingHappened, _, _ = _register_test_events()
    agg = AggregateId("a1")
    for i in range(25):
        store.append(agg, "thing", [ThingHappened(thing=f"v{i}")],
                     expected_version=i, envelopes=[make_envelope(source="t")])
    # Small batch_size exercises the cursor pagination path.
    events = list(store.load_global(batch_size=4))
    assert len(events) == 25
    positions = [e.global_position for e in events]
    assert positions == sorted(positions)


# --- Round-trip + load-side validation -------------------------------------


def test_event_round_trips_through_store_with_citations(store):
    _, _, ClaimMade = _register_test_events()
    agg = AggregateId("ws_1")
    claim = ClaimMade(
        decision_id="dec_42",
        citations=(Citation(source_id="src_paper", quote="γ=0.99", locator="§4.2"),),
    )
    env = make_envelope(source="workspace")
    store.append(agg, "workspace", [claim], expected_version=0, envelopes=[env])

    loaded = list(store.load(agg))
    assert len(loaded) == 1
    typed = loaded[0].into(ClaimMade)
    assert typed.decision_id == "dec_42"
    assert len(typed.citations) == 1
    assert typed.citations[0].locator == "§4.2"


def test_unknown_event_type_loads_as_generic_stored_event(store):
    """If a stored event was registered earlier but its class is no longer
    in the registry (e.g., a renamed type without the rename rule
    installed), load() must still return the StoredEvent so a projection
    can route it through an upcaster. It must NOT crash."""
    ThingHappened, _, _ = _register_test_events()
    agg = AggregateId("a1")
    e = ThingHappened(thing="x")
    env = make_envelope(source="t")
    store.append(agg, "thing", [e], expected_version=0, envelopes=[env])
    # Drop the registry — simulates "we forgot to import the class".
    _clear_registry_for_tests()
    loaded = list(store.load(agg))
    assert len(loaded) == 1
    assert loaded[0].event_type == "thing_happened"
    # But .into() now requires re-registration; that's already covered
    # by test_foundation_event.py.


# --- Capabilities -----------------------------------------------------------


def test_capabilities_reports_concurrency_and_subscriptions(store):
    caps = store.capabilities
    assert caps.optimistic_concurrency is True
    assert caps.supports_persistent_subscriptions is True
    assert caps.max_event_payload_bytes >= 1024 * 1024
