"""Tests for backend.messaging.bus."""

from __future__ import annotations

import threading
from typing import ClassVar

from backend.messaging.bus import DomainEventBus
from backend.messaging.envelope import AggregateId, make_envelope
from backend.messaging.event import (
    DomainEvent,
    StoredEvent,
    _clear_registry_for_tests,
    register_event,
)


def _stored_event() -> StoredEvent:
    @register_event
    class Toy(DomainEvent):
        event_type: ClassVar[str] = "toy_emitted"
        schema_version: ClassVar[int] = 1

    env = make_envelope(source="test")
    return StoredEvent(
        global_position=1,
        aggregate_id=AggregateId("agg_x"),
        aggregate_type="toy",
        aggregate_version=1,
        event_type="toy_emitted",
        schema_version=1,
        payload={},
        envelope=env,
        occurred_at=env.occurred_at,
    )


def setup_function() -> None:
    _clear_registry_for_tests()


def teardown_function() -> None:
    _clear_registry_for_tests()


def test_emit_calls_each_listener_once():
    bus = DomainEventBus()
    received: list[StoredEvent] = []

    bus.subscribe(received.append)
    bus.emit(_stored_event())

    assert len(received) == 1
    assert received[0].event_type == "toy_emitted"


def test_unsubscribe_stops_delivery():
    bus = DomainEventBus()
    received: list[StoredEvent] = []
    unsub = bus.subscribe(received.append)
    unsub()

    bus.emit(_stored_event())
    assert received == []


def test_listener_exception_does_not_block_others():
    bus = DomainEventBus()
    received_b: list[StoredEvent] = []

    def crashing(_e: StoredEvent) -> None:
        raise RuntimeError("boom")

    bus.subscribe(crashing)
    bus.subscribe(received_b.append)
    bus.emit(_stored_event())

    assert len(received_b) == 1


def test_on_listener_error_hook_is_invoked_with_context():
    """The bus surfaces listener failures via the on_listener_error
    hook so production can log + meter them. Without the hook,
    failures are isolated but invisible."""
    captured: list[tuple[BaseException, object, StoredEvent]] = []

    def hook(exc: BaseException, listener: object, event: StoredEvent) -> None:
        captured.append((exc, listener, event))

    bus = DomainEventBus(on_listener_error=hook)
    err = RuntimeError("boom")

    def crashing(_e: StoredEvent) -> None:
        raise err

    bus.subscribe(crashing)
    bus.emit(_stored_event())

    assert len(captured) == 1
    seen_exc, seen_listener, seen_event = captured[0]
    assert seen_exc is err
    assert seen_listener is crashing
    assert seen_event.event_type == "toy_emitted"


def test_listener_error_hook_failure_does_not_break_bus():
    """Defensive: even a buggy on_listener_error hook must not break
    delivery to other listeners."""
    other_received: list[StoredEvent] = []

    def buggy_hook(_exc: BaseException, _l: object, _e: StoredEvent) -> None:
        raise RuntimeError("hook itself crashed")

    bus = DomainEventBus(on_listener_error=buggy_hook)
    bus.subscribe(lambda _e: (_ for _ in ()).throw(RuntimeError("listener crashed")))
    bus.subscribe(other_received.append)

    bus.emit(_stored_event())  # must not raise
    assert len(other_received) == 1


def test_subscribe_is_thread_safe():
    bus = DomainEventBus()
    received: list[StoredEvent] = []
    lock = threading.Lock()

    def safe_append(e: StoredEvent) -> None:
        with lock:
            received.append(e)

    threads = [threading.Thread(target=lambda: bus.subscribe(safe_append)) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert bus.listener_count() == 20
    bus.emit(_stored_event())
    assert len(received) == 20
