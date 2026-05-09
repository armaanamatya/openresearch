"""SQLite-backed persistent Subscription.

Polls the event_store_events table from the last ack'd position for a
given (subscription_name, consumer_id). Iterating yields events one at
a time; ack advances the durable checkpoint; nack schedules redelivery
after a delay.

Two consumer_ids on one subscription_name maintain independent
positions — used for fan-out (a projection and a coordinator can share
a stream filter without blocking each other).

The poll loop uses a short sleep when the log is at the cursor tip;
this is fine for our scale (low-thousands of events/sec). A push model
(SQLite update_hook + condition variable) is a future optimization.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Iterator

from backend.eventstore.interface import (
    Subscription,
    SubscriptionClosedError,
)
from backend.messaging.event import StoredEvent

if TYPE_CHECKING:
    from backend.eventstore.sqlite_store import SqliteEventStore


_DEFAULT_POLL_INTERVAL_SECONDS = 0.25
_DEFAULT_LEASE_TTL_SECONDS = 30.0


class SqliteSubscription(Subscription):
    """Concrete Subscription against SqliteEventStore.

    Iteration is *bounded* by `tail_behavior`:
      - "block" (default): keep polling forever until close().
      - "exit_at_tail": stop when there are no more events.

    Tests use "exit_at_tail"; production coordinators use "block".
    """

    def __init__(
        self,
        store: "SqliteEventStore",
        subscription_name: str,
        consumer_id: str = "default",
        types: tuple[str, ...] | None = None,
        from_position: int | None = None,
        tail_behavior: str = "block",
        poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
    ) -> None:
        self._store = store
        self._name = subscription_name
        self._consumer_id = consumer_id
        self._types = types
        self._tail_behavior = tail_behavior
        self._poll_interval = poll_interval_seconds
        self._closed = False
        self._ensure_checkpoint_row(from_position)

    # --- Properties --------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def consumer_id(self) -> str:
        return self._consumer_id

    @property
    def position(self) -> int:
        if self._closed:
            raise SubscriptionClosedError(
                f"Subscription {self._name!r}/{self._consumer_id!r} is closed"
            )
        row = self._store._conn().execute(
            """
            SELECT last_position FROM event_store_subscription_checkpoints
            WHERE subscription_name = ? AND consumer_id = ?
            """,
            (self._name, self._consumer_id),
        ).fetchone()
        return int(row["last_position"]) if row else -1

    # --- Iteration ---------------------------------------------------------

    def __iter__(self) -> Iterator[StoredEvent]:
        return self._iter()

    def _iter(self) -> Iterator[StoredEvent]:
        next_position = self.position + 1
        while not self._closed:
            # Pull a small batch from the log starting just after the
            # last ack, plus any redeliveries whose `available_at` has passed.
            redeliveries = self._claim_due_redeliveries()
            if redeliveries:
                for ev in redeliveries:
                    next_position = max(next_position, ev.global_position + 1)
                    yield ev
                continue

            sql = (
                "SELECT * FROM event_store_events WHERE global_position >= ? "
            )
            params: list[object] = [next_position]
            if self._types:
                placeholders = ",".join("?" for _ in self._types)
                sql += f"AND event_type IN ({placeholders}) "
                params.extend(self._types)
            # Skip events currently held back for redelivery for this consumer.
            sql += (
                "AND global_position NOT IN ("
                "SELECT global_position FROM event_store_subscription_redelivery "
                "WHERE subscription_name = ? AND consumer_id = ?"
                ") "
                "ORDER BY global_position LIMIT 100"
            )
            params.extend([self._name, self._consumer_id])

            rows = self._store._conn().execute(sql, tuple(params)).fetchall()
            if not rows:
                if self._tail_behavior == "exit_at_tail":
                    return
                time.sleep(self._poll_interval)
                continue

            for row in rows:
                if self._closed:
                    return
                next_position = int(row["global_position"]) + 1
                yield self._store._row_to_stored(row)

    # --- Ack / Nack / Lease ------------------------------------------------

    def ack(self, event: StoredEvent) -> None:
        self._guard_open()
        # Advance only forward (acks can technically arrive in order;
        # if a caller acks a later position before earlier, we accept it
        # but the contract in the Protocol says "in order"). We do NOT
        # back up the checkpoint.
        now_iso = datetime.now(timezone.utc).isoformat()
        conn = self._store._conn()
        conn.execute(
            """
            UPDATE event_store_subscription_checkpoints
            SET last_position = MAX(last_position, ?), last_ack_at = ?
            WHERE subscription_name = ? AND consumer_id = ?
            """,
            (event.global_position, now_iso, self._name, self._consumer_id),
        )
        # If the event was scheduled for redelivery, drop it.
        conn.execute(
            """
            DELETE FROM event_store_subscription_redelivery
            WHERE subscription_name = ? AND consumer_id = ? AND global_position = ?
            """,
            (self._name, self._consumer_id, event.global_position),
        )
        conn.commit()

    def nack(self, event: StoredEvent, *, retry_after_seconds: float) -> None:
        self._guard_open()
        avail = (
            datetime.now(timezone.utc) + timedelta(seconds=retry_after_seconds)
        ).isoformat()
        conn = self._store._conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO event_store_subscription_redelivery
                (subscription_name, consumer_id, global_position, available_at)
            VALUES (?, ?, ?, ?)
            """,
            (self._name, self._consumer_id, event.global_position, avail),
        )
        conn.commit()

    def renew_lease(self, ttl_seconds: float = _DEFAULT_LEASE_TTL_SECONDS) -> None:
        self._guard_open()
        expires = (
            datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
        ).isoformat()
        conn = self._store._conn()
        conn.execute(
            """
            UPDATE event_store_subscription_checkpoints
            SET lease_expires_at = ?
            WHERE subscription_name = ? AND consumer_id = ?
            """,
            (expires, self._name, self._consumer_id),
        )
        conn.commit()

    def close(self) -> None:
        self._closed = True

    # --- Internal ----------------------------------------------------------

    def _guard_open(self) -> None:
        if self._closed:
            raise SubscriptionClosedError(
                f"Subscription {self._name!r}/{self._consumer_id!r} is closed"
            )

    def _ensure_checkpoint_row(self, from_position: int | None) -> None:
        conn = self._store._conn()
        existing = conn.execute(
            """
            SELECT last_position FROM event_store_subscription_checkpoints
            WHERE subscription_name = ? AND consumer_id = ?
            """,
            (self._name, self._consumer_id),
        ).fetchone()
        if existing is None:
            initial = -1 if from_position is None else from_position - 1
            conn.execute(
                """
                INSERT INTO event_store_subscription_checkpoints
                    (subscription_name, consumer_id, last_position, last_ack_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    self._name,
                    self._consumer_id,
                    initial,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
        elif from_position is not None:
            # Explicit reposition (used by projection rebuilds).
            conn.execute(
                """
                UPDATE event_store_subscription_checkpoints
                SET last_position = ?
                WHERE subscription_name = ? AND consumer_id = ?
                """,
                (from_position - 1, self._name, self._consumer_id),
            )
            conn.commit()

    def _claim_due_redeliveries(self) -> list[StoredEvent]:
        """Return events whose redelivery time has arrived. Pops them
        from the redelivery table so the iterator cycles through and
        the consumer can ack/nack again."""
        now_iso = datetime.now(timezone.utc).isoformat()
        conn = self._store._conn()
        rows = conn.execute(
            """
            SELECT global_position FROM event_store_subscription_redelivery
            WHERE subscription_name = ? AND consumer_id = ? AND available_at <= ?
            ORDER BY global_position
            LIMIT 100
            """,
            (self._name, self._consumer_id, now_iso),
        ).fetchall()
        if not rows:
            return []
        positions = [r["global_position"] for r in rows]
        placeholders = ",".join("?" for _ in positions)
        event_rows = conn.execute(
            f"SELECT * FROM event_store_events WHERE global_position IN ({placeholders}) "
            f"ORDER BY global_position",
            tuple(positions),
        ).fetchall()
        # Drop them from the redelivery table — caller will ack/nack again
        # if the handler still fails.
        conn.execute(
            f"DELETE FROM event_store_subscription_redelivery "
            f"WHERE subscription_name = ? AND consumer_id = ? "
            f"AND global_position IN ({placeholders})",
            (self._name, self._consumer_id, *positions),
        )
        conn.commit()
        return [self._store._row_to_stored(r) for r in event_rows]


__all__ = ["SqliteSubscription"]
