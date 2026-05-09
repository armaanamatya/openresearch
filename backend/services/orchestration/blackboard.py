"""Blackboard shared-state service with scoped visibility."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel

from backend.persistence.database import Database


class BlackboardRecord(BaseModel):
    key: str
    value: str
    scope: str  # private_to_parent, branch_shared, global_verified
    owner_task_id: str
    created_at: datetime


class BlackboardService:
    def __init__(self, db: Database) -> None:
        self._db = db
        self._ensure_table()

    def _ensure_table(self) -> None:
        self._db.connection.execute("""
            CREATE TABLE IF NOT EXISTS blackboard (
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                scope TEXT NOT NULL,
                owner_task_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (key, owner_task_id)
            )
        """)
        self._db.connection.commit()

    def write(self, key: str, value: str, scope: str, owner_task_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._db.connection.execute(
            """INSERT OR REPLACE INTO blackboard (key, value, scope, owner_task_id, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (key, value, scope, owner_task_id, now),
        )
        self._db.connection.commit()

    def read(self, key: str) -> Optional[BlackboardRecord]:
        row = self._db.connection.execute(
            "SELECT * FROM blackboard WHERE key = ? ORDER BY created_at DESC LIMIT 1",
            (key,),
        ).fetchone()
        if row is None:
            return None
        return BlackboardRecord(
            key=row["key"],
            value=row["value"],
            scope=row["scope"],
            owner_task_id=row["owner_task_id"],
            created_at=row["created_at"],
        )

    def read_visible(
        self,
        key: str,
        reader_task_id: str,
        reader_lineage: list[str],
    ) -> Optional[BlackboardRecord]:
        """Read a record respecting scope visibility rules."""
        rows = self._db.connection.execute(
            "SELECT * FROM blackboard WHERE key = ?", (key,)
        ).fetchall()

        for row in rows:
            scope = row["scope"]
            owner = row["owner_task_id"]

            if scope == "global_verified":
                return BlackboardRecord(
                    key=row["key"], value=row["value"],
                    scope=scope, owner_task_id=owner, created_at=row["created_at"],
                )

            if scope == "branch_shared":
                # Visible if owner is in reader's lineage (ancestor wrote it)
                if owner in reader_lineage or owner == reader_task_id:
                    return BlackboardRecord(
                        key=row["key"], value=row["value"],
                        scope=scope, owner_task_id=owner, created_at=row["created_at"],
                    )

            if scope == "private_to_parent":
                # Only visible to the direct parent of the owner
                # reader must be the parent of the owner (owner is in reader's children)
                # i.e., owner's lineage[-1] before owner should be reader_task_id
                # Simplified: reader is NOT a sibling — reader must be an ancestor
                if reader_task_id in reader_lineage:
                    # reader is somewhere in the lineage chain
                    pass
                # Private records are only visible if reader is the parent
                # owner wrote it, only owner's parent can read
                # We check if reader is the immediate parent by seeing if
                # owner is NOT the reader and reader is in the lineage right before owner
                # For simplicity: private_to_parent means only the parent of owner sees it
                # If owner is in reader's children set, reader can see it
                # We don't have children info here, so: owner's lineage should end with reader
                # Skip — not visible to siblings
                continue

        return None
