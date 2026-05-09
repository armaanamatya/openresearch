"""Repository for runs."""

from __future__ import annotations

from backend.persistence.database import Database
from backend.schemas.runs import Run


class RunRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def create(self, run: Run) -> None:
        self._db.connection.execute(
            """INSERT INTO runs
               (run_id, run_type, status, task_id, parent_run_id, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                run.run_id,
                run.run_type.value if hasattr(run.run_type, "value") else run.run_type,
                run.status,
                run.task_id,
                run.parent_run_id,
                run.created_at.isoformat(),
                run.updated_at.isoformat(),
            ),
        )
        self._db.connection.commit()

    def get_by_id(self, run_id: str) -> Run | None:
        row = self._db.connection.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return Run(
            run_id=row["run_id"],
            run_type=row["run_type"],
            status=row["status"],
            task_id=row["task_id"],
            parent_run_id=row["parent_run_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
