"""Delegation tree tracking for parent/child task relationships."""

from __future__ import annotations

from typing import Any

from backend.persistence.database import Database


class DelegationService:
    def __init__(self, db: Database) -> None:
        self._db = db
        self._ensure_table()

    def _ensure_table(self) -> None:
        self._db.connection.execute("""
            CREATE TABLE IF NOT EXISTS delegation_tree (
                parent_id TEXT NOT NULL,
                child_id TEXT NOT NULL,
                PRIMARY KEY (parent_id, child_id)
            )
        """)
        self._db.connection.commit()

    def register(self, parent_id: str, child_id: str) -> None:
        self._db.connection.execute(
            "INSERT OR IGNORE INTO delegation_tree (parent_id, child_id) VALUES (?, ?)",
            (parent_id, child_id),
        )
        self._db.connection.commit()

    def get_children(self, parent_id: str) -> list[str]:
        rows = self._db.connection.execute(
            "SELECT child_id FROM delegation_tree WHERE parent_id = ?", (parent_id,)
        ).fetchall()
        return [r["child_id"] for r in rows]

    def get_parent(self, task_id: str) -> str | None:
        row = self._db.connection.execute(
            "SELECT parent_id FROM delegation_tree WHERE child_id = ?", (task_id,)
        ).fetchone()
        return row["parent_id"] if row else None

    def get_lineage(self, task_id: str) -> list[str]:
        """Return full lineage from root to task_id (inclusive)."""
        lineage = [task_id]
        current = task_id
        while True:
            parent = self.get_parent(current)
            if parent is None:
                break
            lineage.insert(0, parent)
            current = parent
        return lineage

    def get_depth(self, task_id: str) -> int:
        """Return depth of task in delegation tree (root = 0)."""
        return len(self.get_lineage(task_id)) - 1

    def get_tree(self, root_id: str) -> dict[str, Any]:
        """Return full tree as nested dict (for dashboard serialization)."""
        children = self.get_children(root_id)
        return {
            "task_id": root_id,
            "children": [self.get_tree(c) for c in children],
        }
