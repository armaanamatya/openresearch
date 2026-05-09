"""SQLite-backed EventStore implementation.

Production default for the event-sourcing layer. Lives alongside teammate's
sync sqlite3-based persistence (`backend.persistence.database.Database`)
and writes to the same file via the configured `database_url`. Tables are
prefixed `event_store_` so they cannot collide with teammate's CRUD
schemas.

Threading model (spec §15.13): per-thread connections via
`threading.local()`. Each thread that reads or writes opens its own
sqlite3.Connection on first access; WAL mode allows N readers + 1 writer
concurrently. Writes serialize on SQLite's writer lock; we add no
Python-level lock.

Append semantics:
  - Optimistic concurrency keyed on (aggregate_id, aggregate_version).
    Stale `expected_version` raises ConcurrencyError.
  - Every payload is re-validated against its registered DomainEvent
    class on the way in (catches dict-bypass / model_construct
    backdoors before they reach storage).
  - Whole-batch duplicate event_id detection: if every event_id in
    the batch already exists with matching content, the call is an
    idempotent no-op returning the original positions. Partial
    duplicates (some-but-not-all) raise DuplicateEventError.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Iterable, Iterator, Sequence

from backend.eventstore.interface import (
    AppendError,
    AppendResult,
    ConcurrencyError,
    DuplicateEventError,
    EventStore,
    StoreCapabilities,
    Subscription,
)
from backend.messaging.envelope import (
    AggregateId,
    EventEnvelope,
    EventId,
)
from backend.messaging.event import DomainEvent, StoredEvent, resolve_event_class


_DDL = """
CREATE TABLE IF NOT EXISTS event_store_events (
    global_position INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    aggregate_id TEXT NOT NULL,
    aggregate_type TEXT NOT NULL,
    aggregate_version INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    UNIQUE (aggregate_id, aggregate_version)
);
CREATE INDEX IF NOT EXISTS idx_event_store_events_aggregate
    ON event_store_events(aggregate_id, aggregate_version);
CREATE INDEX IF NOT EXISTS idx_event_store_events_type
    ON event_store_events(event_type);
CREATE INDEX IF NOT EXISTS idx_event_store_events_occurred_at
    ON event_store_events(occurred_at);

CREATE TABLE IF NOT EXISTS event_store_subscription_checkpoints (
    subscription_name TEXT NOT NULL,
    consumer_id TEXT NOT NULL,
    last_position INTEGER NOT NULL DEFAULT -1,
    leased_by TEXT,
    lease_expires_at TEXT,
    last_ack_at TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (subscription_name, consumer_id)
);

CREATE TABLE IF NOT EXISTS event_store_subscription_redelivery (
    subscription_name TEXT NOT NULL,
    consumer_id TEXT NOT NULL,
    global_position INTEGER NOT NULL,
    available_at TEXT NOT NULL,
    PRIMARY KEY (subscription_name, consumer_id, global_position)
);
"""


# Capabilities reported by this store.
_CAPABILITIES = StoreCapabilities(
    supports_persistent_subscriptions=True,
    supports_stream_categories=True,
    optimistic_concurrency=True,
    max_event_payload_bytes=2 * 1024 * 1024,  # 2 MB; large blobs go to the blob store
)


def _extract_path(database_url: str) -> str:
    """Mirror teammate's `Database._path` extraction so we point at the
    same file when given the same `REPROLAB_DATABASE_URL`."""
    return database_url.replace("sqlite:///", "")


def _new_connection(path: str) -> sqlite3.Connection:
    """Open a fresh connection with our PRAGMAs."""
    conn = sqlite3.connect(path, isolation_level="DEFERRED")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


class SqliteEventStore(EventStore):
    """Concrete EventStore on SQLite (WAL mode).

    Construct once per database file. Multiple threads share one
    SqliteEventStore instance; each thread internally gets its own
    sqlite3.Connection via threading.local.
    """

    def __init__(self, database_url: str) -> None:
        self._path = _extract_path(database_url)
        self._local = threading.local()
        self._all_conns: list[sqlite3.Connection] = []
        self._all_conns_lock = threading.Lock()
        # Bootstrap schema once on construction using a throwaway conn.
        boot = _new_connection(self._path)
        try:
            boot.executescript(_DDL)
            boot.commit()
        finally:
            boot.close()

    # --- Lifecycle ---------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = _new_connection(self._path)
            self._local.conn = c
            with self._all_conns_lock:
                self._all_conns.append(c)
        return c

    def close(self) -> None:
        """Close every per-thread connection. Test/teardown helper —
        production keeps the store alive for the process lifetime."""
        with self._all_conns_lock:
            for c in self._all_conns:
                try:
                    c.close()
                except sqlite3.Error:
                    pass
            self._all_conns.clear()
        self._local = threading.local()

    # --- Capabilities ------------------------------------------------------

    @property
    def capabilities(self) -> StoreCapabilities:
        return _CAPABILITIES

    # --- Append ------------------------------------------------------------

    def append(
        self,
        aggregate_id: AggregateId,
        aggregate_type: str,
        events: Sequence[DomainEvent],
        expected_version: int,
        envelopes: Sequence[EventEnvelope],
    ) -> AppendResult:
        if len(events) == 0:
            raise AppendError("Cannot append an empty event batch.")
        if len(events) != len(envelopes):
            raise AppendError(
                f"events ({len(events)}) and envelopes ({len(envelopes)}) "
                f"must be the same length."
            )

        # Re-validate every event payload against its registered class
        # before any disk write. Catches dict-bypass / model_construct
        # backdoors at the storage boundary.
        for ev in events:
            cls = type(ev)
            if not cls.event_type:
                raise AppendError(
                    f"DomainEvent class {cls.__name__} has empty event_type."
                )
            cls.model_validate(ev.model_dump())

        conn = self._conn()
        try:
            conn.execute("BEGIN")

            # Check for whole-batch duplicate event_ids: every id already
            # in store with matching aggregate (idempotent re-emit) -> no-op.
            event_ids = [env.event_id for env in envelopes]
            placeholders = ",".join("?" for _ in event_ids)
            existing_rows = conn.execute(
                f"""
                SELECT event_id, global_position, aggregate_id, aggregate_version
                FROM event_store_events WHERE event_id IN ({placeholders})
                """,
                tuple(event_ids),
            ).fetchall()
            existing_ids = {r["event_id"] for r in existing_rows}
            if existing_ids:
                if len(existing_ids) == len(event_ids):
                    # Fully duplicate. All must belong to this aggregate; if
                    # not, that's a real misuse (event_id collision across
                    # aggregates).
                    for r in existing_rows:
                        if r["aggregate_id"] != aggregate_id:
                            conn.execute("ROLLBACK")
                            raise DuplicateEventError(
                                f"event_id {r['event_id']!r} already exists on a "
                                f"different aggregate ({r['aggregate_id']!r})"
                            )
                    conn.execute("ROLLBACK")  # nothing changed
                    by_id = {r["event_id"]: r for r in existing_rows}
                    return AppendResult(
                        new_aggregate_version=max(
                            r["aggregate_version"] for r in existing_rows
                        ),
                        written_event_ids=tuple(EventId(eid) for eid in event_ids),
                        written_global_positions=tuple(
                            by_id[eid]["global_position"] for eid in event_ids
                        ),
                    )
                # Partial duplicate.
                conn.execute("ROLLBACK")
                raise DuplicateEventError(
                    f"Partial duplicate batch: {sorted(existing_ids)} already in store "
                    f"while {sorted(set(event_ids) - existing_ids)} are new. "
                    f"Either the whole batch is a re-emit or none of it is."
                )

            # Optimistic concurrency check.
            row = conn.execute(
                "SELECT MAX(aggregate_version) AS mv FROM event_store_events "
                "WHERE aggregate_id = ?",
                (aggregate_id,),
            ).fetchone()
            current_version = row["mv"] if row and row["mv"] is not None else 0
            if current_version != expected_version:
                conn.execute("ROLLBACK")
                raise ConcurrencyError(
                    aggregate_id=aggregate_id,
                    expected=expected_version,
                    actual=current_version,
                )

            written_event_ids: list[EventId] = []
            written_positions: list[int] = []
            new_version = expected_version
            for ev, env in zip(events, envelopes):
                new_version += 1
                payload_json = ev.model_dump_json()
                metadata_json = env.model_dump_json()
                cur = conn.execute(
                    """
                    INSERT INTO event_store_events
                        (event_id, aggregate_id, aggregate_type, aggregate_version,
                         event_type, schema_version, payload_json, metadata_json,
                         occurred_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        env.event_id,
                        aggregate_id,
                        aggregate_type,
                        new_version,
                        type(ev).event_type,
                        type(ev).schema_version,
                        payload_json,
                        metadata_json,
                        env.occurred_at.isoformat(),
                    ),
                )
                gp = cur.lastrowid
                if gp is None:
                    conn.execute("ROLLBACK")
                    raise AppendError("INSERT returned no global_position.")
                written_event_ids.append(env.event_id)
                written_positions.append(int(gp))

            conn.execute("COMMIT")
            return AppendResult(
                new_aggregate_version=new_version,
                written_event_ids=tuple(written_event_ids),
                written_global_positions=tuple(written_positions),
            )
        except (ConcurrencyError, DuplicateEventError, AppendError):
            raise
        except sqlite3.Error as exc:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise AppendError(f"SQLite error during append: {exc}") from exc

    # --- Load --------------------------------------------------------------

    def load(
        self,
        aggregate_id: AggregateId,
        from_version: int = 0,
    ) -> Iterator[StoredEvent]:
        rows = self._conn().execute(
            """
            SELECT * FROM event_store_events
            WHERE aggregate_id = ? AND aggregate_version > ?
            ORDER BY aggregate_version
            """,
            (aggregate_id, from_version),
        ).fetchall()
        for r in rows:
            yield self._row_to_stored(r)

    def load_global(
        self,
        from_position: int = 0,
        to_position: int | None = None,
        types: Iterable[str] | None = None,
        batch_size: int = 1000,
    ) -> Iterator[StoredEvent]:
        type_list = list(types) if types is not None else None
        # Page through with explicit cursor on global_position for
        # streaming-friendly behavior (no all-rows-in-memory load).
        cursor_pos = from_position
        while True:
            sql = (
                "SELECT * FROM event_store_events "
                "WHERE global_position >= ? "
            )
            params: list[object] = [cursor_pos]
            if to_position is not None:
                sql += "AND global_position <= ? "
                params.append(to_position)
            if type_list:
                placeholders = ",".join("?" for _ in type_list)
                sql += f"AND event_type IN ({placeholders}) "
                params.extend(type_list)
            sql += "ORDER BY global_position LIMIT ?"
            params.append(batch_size)

            rows = self._conn().execute(sql, tuple(params)).fetchall()
            if not rows:
                break
            for r in rows:
                yield self._row_to_stored(r)
            last = rows[-1]["global_position"]
            cursor_pos = last + 1
            if len(rows) < batch_size:
                break

    def get_aggregate_version(self, aggregate_id: AggregateId) -> int:
        row = self._conn().execute(
            "SELECT MAX(aggregate_version) AS mv FROM event_store_events "
            "WHERE aggregate_id = ?",
            (aggregate_id,),
        ).fetchone()
        return int(row["mv"]) if row and row["mv"] is not None else 0

    # --- Subscribe ---------------------------------------------------------

    def subscribe(
        self,
        subscription_name: str,
        *,
        consumer_id: str = "default",
        types: Iterable[str] | None = None,
        from_position: int | None = None,
    ) -> Subscription:
        # Concrete Subscription lives in subscription.py to keep this file
        # focused. Imported lazily to avoid a circular at module-load time.
        from backend.eventstore.subscription import SqliteSubscription

        return SqliteSubscription(
            store=self,
            subscription_name=subscription_name,
            consumer_id=consumer_id,
            types=tuple(types) if types is not None else None,
            from_position=from_position,
        )

    # --- Internal: row decoding -------------------------------------------

    @staticmethod
    def _row_to_stored(row: sqlite3.Row) -> StoredEvent:
        payload = json.loads(row["payload_json"])
        envelope_dict = json.loads(row["metadata_json"])
        envelope = EventEnvelope.model_validate(envelope_dict)

        # Re-validate the payload against its registered class on the way
        # OUT of storage too. Closes the StoredEvent.into() defense-in-depth
        # loop (spec §5.6) at the deserialization seam.
        try:
            cls = resolve_event_class(row["event_type"], schema_version=row["schema_version"])
        except KeyError:
            # No class registered for this (type, version). The event is
            # still loadable as a generic StoredEvent; callers must invoke
            # an upcaster before .into(). We do not raise here because
            # projections may want to forward it through an upcaster pipeline.
            cls = None
        if cls is not None:
            cls.model_validate(payload)

        return StoredEvent(
            global_position=int(row["global_position"]),
            aggregate_id=AggregateId(row["aggregate_id"]),
            aggregate_type=row["aggregate_type"],
            aggregate_version=int(row["aggregate_version"]),
            event_type=row["event_type"],
            schema_version=int(row["schema_version"]),
            payload=payload,
            envelope=envelope,
            occurred_at=datetime.fromisoformat(row["occurred_at"]),
        )


__all__ = ["SqliteEventStore"]
