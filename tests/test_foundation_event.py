"""Tests for backend.messaging.event."""

from __future__ import annotations

from typing import ClassVar

import pytest
from pydantic import ValidationError

from backend.messaging.envelope import AggregateId, make_envelope
from backend.messaging.event import (
    DomainEvent,
    EventTypeAlreadyRegistered,
    InvariantBypassError,
    StoredEvent,
    _clear_registry_for_tests,
    register_event,
    registered_event_types,
    resolve_event_class,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    _clear_registry_for_tests()
    yield
    _clear_registry_for_tests()


def test_register_event_records_class_under_type_and_version():
    @register_event
    class TestThingHappened(DomainEvent):
        event_type: ClassVar[str] = "test_thing_happened"
        schema_version: ClassVar[int] = 1
        thing_id: str

    assert resolve_event_class("test_thing_happened", schema_version=1) is TestThingHappened
    assert ("test_thing_happened", 1) in registered_event_types()


def test_register_event_rejects_blank_event_type():
    class Bad(DomainEvent):
        event_type: ClassVar[str] = ""

    with pytest.raises(ValueError):
        register_event(Bad)


def test_register_event_rejects_zero_or_negative_schema_version():
    class Zero(DomainEvent):
        event_type: ClassVar[str] = "zero_v"
        schema_version: ClassVar[int] = 0

    with pytest.raises(ValueError):
        register_event(Zero)


def test_register_event_rejects_collision_within_same_version():
    @register_event
    class A(DomainEvent):
        event_type: ClassVar[str] = "collide"
        schema_version: ClassVar[int] = 1

    class B(DomainEvent):
        event_type: ClassVar[str] = "collide"
        schema_version: ClassVar[int] = 1

    with pytest.raises(EventTypeAlreadyRegistered):
        register_event(B)


def test_register_event_allows_same_type_at_different_versions():
    """Multiple shape versions of the same event_type coexist; upcasters
    bridge them at read time."""

    @register_event
    class V1(DomainEvent):
        event_type: ClassVar[str] = "evolved"
        schema_version: ClassVar[int] = 1
        old_field: str

    @register_event
    class V2(DomainEvent):
        event_type: ClassVar[str] = "evolved"
        schema_version: ClassVar[int] = 2
        new_field: str

    assert resolve_event_class("evolved", schema_version=1) is V1
    assert resolve_event_class("evolved", schema_version=2) is V2


def test_register_event_is_idempotent_for_same_class():
    @register_event
    class Same(DomainEvent):
        event_type: ClassVar[str] = "same"
        schema_version: ClassVar[int] = 1

    register_event(Same)
    assert resolve_event_class("same", schema_version=1) is Same


def test_resolve_unknown_combination_raises_keyerror():
    with pytest.raises(KeyError):
        resolve_event_class("nonexistent")
    with pytest.raises(KeyError):
        resolve_event_class("nonexistent", schema_version=2)


def test_domain_event_is_frozen():
    @register_event
    class Frozen(DomainEvent):
        event_type: ClassVar[str] = "frozen_event"
        schema_version: ClassVar[int] = 1
        x: int

    e = Frozen(x=1)
    with pytest.raises(ValidationError):
        e.x = 2  # type: ignore[misc]


def test_domain_event_model_construct_is_banned():
    """Calling model_construct on a DomainEvent subclass raises
    InvariantBypassError. This is the language-level enforcement that
    makes the citation invariant unbypassable from production code."""

    @register_event
    class Banned(DomainEvent):
        event_type: ClassVar[str] = "banned_construct"
        schema_version: ClassVar[int] = 1
        x: int

    with pytest.raises(InvariantBypassError):
        Banned.model_construct(x=1)


def test_stored_event_into_validates_payload():
    @register_event
    class Demo(DomainEvent):
        event_type: ClassVar[str] = "demo_event"
        schema_version: ClassVar[int] = 1
        n: int

    env = make_envelope(source="test")
    stored = StoredEvent(
        global_position=1,
        aggregate_id=AggregateId("agg_1"),
        aggregate_type="demo",
        aggregate_version=1,
        event_type="demo_event",
        schema_version=1,
        payload={"n": 42},
        envelope=env,
        occurred_at=env.occurred_at,
    )
    typed = stored.into(Demo)
    assert isinstance(typed, Demo)
    assert typed.n == 42


def test_stored_event_into_rejects_wrong_class():
    @register_event
    class A(DomainEvent):
        event_type: ClassVar[str] = "type_a"
        schema_version: ClassVar[int] = 1

    @register_event
    class B(DomainEvent):
        event_type: ClassVar[str] = "type_b"
        schema_version: ClassVar[int] = 1

    env = make_envelope(source="test")
    stored = StoredEvent(
        global_position=1,
        aggregate_id=AggregateId("agg_1"),
        aggregate_type="x",
        aggregate_version=1,
        event_type="type_a",
        schema_version=1,
        payload={},
        envelope=env,
        occurred_at=env.occurred_at,
    )
    with pytest.raises(ValueError, match="Cannot deserialize"):
        stored.into(B)


def test_stored_event_into_rejects_schema_version_mismatch():
    """Loading a v1-stored payload directly into a v2 class is forbidden;
    callers must consult the upcaster registry first."""

    @register_event
    class V1(DomainEvent):
        event_type: ClassVar[str] = "version_aware"
        schema_version: ClassVar[int] = 1
        x: int

    @register_event
    class V2(DomainEvent):
        event_type: ClassVar[str] = "version_aware"
        schema_version: ClassVar[int] = 2
        x: int
        y: int

    env = make_envelope(source="test")
    stored_v1 = StoredEvent(
        global_position=1,
        aggregate_id=AggregateId("agg_1"),
        aggregate_type="x",
        aggregate_version=1,
        event_type="version_aware",
        schema_version=1,
        payload={"x": 1},
        envelope=env,
        occurred_at=env.occurred_at,
    )
    # Loading v1 stored as v1 is fine.
    typed_v1 = stored_v1.into(V1)
    assert typed_v1.x == 1
    # Loading v1 stored directly into V2 must raise — go through upcaster.
    with pytest.raises(ValueError, match="Schema version mismatch"):
        stored_v1.into(V2)
