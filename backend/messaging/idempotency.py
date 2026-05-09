"""Bounded command-idempotency table.

Spec §4.6 / §8.2: every command application service consults this
table before performing IO. A duplicate command_id returns the
previously recorded result event ids without re-executing.

Bounded retention prevents unbounded growth: rows expire after
`default_retention`. A periodic cleanup job purges expired rows.

Recording semantics:
- Same (aggregate_id, command_id) with the *same* result_event_ids:
  idempotent no-op.
- Same (aggregate_id, command_id) with *different* result_event_ids:
  raises `DuplicateCommandError`. Indicates a real bug — the calling
  application service should have hit the `lookup()` short-circuit
  before re-executing IO.
- Expired prior row: replaced by the new record.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Iterable

from backend.messaging.command import CommandId
from backend.messaging.envelope import AggregateId, EventId
from backend.persistence.database import Database


_DDL = """
CREATE TABLE IF NOT EXISTS event_store_command_idempotency (
    aggregate_id TEXT NOT NULL,
    command_id TEXT NOT NULL,
    result_event_ids_json TEXT NOT NULL,
    handled_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    PRIMARY KEY (aggregate_id, command_id)
);
CREATE INDEX IF NOT EXISTS idx_event_store_idempotency_expires
    ON event_store_command_idempotency(expires_at);
"""


class DuplicateCommandError(Exception):
    """Raised when `record()` is asked to write a different result for an
    already-recorded (aggregate_id, command_id) whose row is still valid.

    This indicates the caller skipped `lookup()` before re-executing IO.
    Carries `existing` and `incoming` for diagnostics.
    """

    def __init__(
        self,
        aggregate_id: str,
        command_id: str,
        existing: tuple[EventId, ...],
        incoming: tuple[EventId, ...],
    ) -> None:
        super().__init__(
            f"Command (aggregate_id={aggregate_id!r}, command_id={command_id!r}) "
            f"already recorded with result {list(existing)} but caller is now "
            f"trying to record {list(incoming)}. Use lookup() to short-circuit "
            f"before re-executing IO."
        )
        self.aggregate_id = aggregate_id
        self.command_id = command_id
        self.existing = existing
        self.incoming = incoming


class IdempotencyTable:
    """Stores `(aggregate_id, command_id) -> [event_id, ...]` with bounded retention."""

    def __init__(
        self,
        db: Database,
        default_retention: timedelta = timedelta(days=30),
    ) -> None:
        self._db = db
        self._retention = default_retention
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._db.connection.executescript(_DDL)
        self._db.connection.commit()

    def lookup(
        self,
        aggregate_id: AggregateId,
        command_id: CommandId,
    ) -> tuple[EventId, ...] | None:
        """Return previously recorded result event ids for this command, or None.

        Treats expired rows as missing — they are still in the DB (cleanup
        job removes them later) but no longer authoritative for idempotency.
        """
        row = self._db.connection.execute(
            """
            SELECT result_event_ids_json, expires_at
            FROM event_store_command_idempotency
            WHERE aggregate_id = ? AND command_id = ?
            """,
            (aggregate_id, command_id),
        ).fetchone()
        if row is None:
            return None
        expires_at = datetime.fromisoformat(row["expires_at"])
        if expires_at < datetime.now(timezone.utc):
            return None
        ids = json.loads(row["result_event_ids_json"])
        return tuple(EventId(s) for s in ids)

    def record(
        self,
        aggregate_id: AggregateId,
        command_id: CommandId,
        result_event_ids: Iterable[EventId],
    ) -> None:
        """Record a command's result event ids.

        Caller is responsible for committing — `record` does not commit
        so it can join the caller's atomic block (event store append +
        idempotency record in one transaction).

        Behavior on (aggregate_id, command_id) collision:
          - Existing row is expired: REPLACE it.
          - Existing row matches `result_event_ids` exactly: no-op
            (idempotent re-record).
          - Existing row differs: raise `DuplicateCommandError`.
        """
        incoming = tuple(result_event_ids)
        existing = self.lookup(aggregate_id, command_id)
        if existing is not None:
            if existing == incoming:
                # Idempotent re-record with same result; no-op.
                return
            raise DuplicateCommandError(
                aggregate_id=aggregate_id,
                command_id=command_id,
                existing=existing,
                incoming=incoming,
            )

        now = datetime.now(timezone.utc)
        expires = now + self._retention
        # INSERT OR REPLACE handles the expired-row case (lookup returned
        # None for it, but the row still exists physically until the
        # cleanup job runs). Without OR REPLACE the PK conflict would raise.
        self._db.connection.execute(
            """
            INSERT OR REPLACE INTO event_store_command_idempotency
                (aggregate_id, command_id, result_event_ids_json, handled_at, expires_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                aggregate_id,
                command_id,
                json.dumps(list(incoming)),
                now.isoformat(),
                expires.isoformat(),
            ),
        )

    def purge_expired(self) -> int:
        """Delete expired rows. Returns the count purged."""
        now_iso = datetime.now(timezone.utc).isoformat()
        cur = self._db.connection.execute(
            "DELETE FROM event_store_command_idempotency WHERE expires_at < ?",
            (now_iso,),
        )
        self._db.connection.commit()
        return cur.rowcount or 0


__all__ = ["DuplicateCommandError", "IdempotencyTable"]
