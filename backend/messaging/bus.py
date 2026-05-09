"""Thread-safe in-process domain event bus.

The bus is an emit/listen surface for projections, coordinators, and
the EventPayloadBridge that translates our domain events into the
dashboard's EventPayload notifications. The event store is the
source of truth; the bus is a *broadcast* channel for derived state.

Listeners run synchronously on the emit thread by default. Long-running
listeners should hand work to their own queue/thread; the bus does not
back-pressure or buffer.

Listener exceptions are isolated (one bad listener does not kill the
others), but the failure is *observable* via the optional
`on_listener_error` hook supplied at construction. Production wires
this to structured logging + a Prometheus counter; tests assert on it.
"""

from __future__ import annotations

import threading
from typing import Callable

from backend.messaging.event import StoredEvent

EventListener = Callable[[StoredEvent], None]
ListenerErrorHook = Callable[[BaseException, EventListener, StoredEvent], None]


class DomainEventBus:
    """Listener-based broadcast. Thread-safe registration and emit.

    A single global instance lives in the app DI; tests construct
    their own.

    Args:
      on_listener_error: optional hook called when a listener raises.
        Receives `(exception, listener, event)`. The hook MUST NOT raise
        — if it does, the bus catches and discards to preserve listener
        isolation. Default is a no-op (audit-grade callers should always
        supply one in production).
    """

    def __init__(self, on_listener_error: ListenerErrorHook | None = None) -> None:
        self._lock = threading.RLock()
        self._listeners: list[EventListener] = []
        self._on_listener_error = on_listener_error

    def subscribe(self, listener: EventListener) -> Callable[[], None]:
        """Register a listener. Returns an unsubscribe callable."""
        with self._lock:
            self._listeners.append(listener)

        def _unsubscribe() -> None:
            with self._lock:
                try:
                    self._listeners.remove(listener)
                except ValueError:
                    pass

        return _unsubscribe

    def emit(self, event: StoredEvent) -> None:
        """Synchronously call every listener with the event.

        Exceptions from listeners are isolated: one bad listener does
        not kill the others. The failure surfaces via the
        `on_listener_error` hook (if supplied at construction).
        """
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(event)
            except BaseException as exc:  # noqa: BLE001 — bus must isolate listener faults
                if self._on_listener_error is not None:
                    try:
                        self._on_listener_error(exc, listener, event)
                    except BaseException:  # noqa: BLE001 — don't let the hook break the bus
                        pass
                continue

    def listener_count(self) -> int:
        with self._lock:
            return len(self._listeners)


__all__ = ["DomainEventBus", "EventListener", "ListenerErrorHook"]
