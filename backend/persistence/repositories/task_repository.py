"""Repository for agent_tasks."""

from __future__ import annotations

from backend.persistence.database import Database
from backend.schemas.tasks import AgentTask


class TaskRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def create(self, task: AgentTask) -> None:
        self._db.connection.execute(
            """INSERT INTO agent_tasks
               (task_id, agent_type, status, parent_task_id, failure_substatus, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                task.task_id,
                task.agent_type,
                task.status.value if hasattr(task.status, "value") else task.status,
                task.parent_task_id,
                task.failure_substatus.value if task.failure_substatus else None,
                task.created_at.isoformat(),
                task.updated_at.isoformat(),
            ),
        )
        self._db.connection.commit()

    def get_by_id(self, task_id: str) -> AgentTask | None:
        row = self._db.connection.execute(
            "SELECT * FROM agent_tasks WHERE task_id = ?", (task_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_task(row)

    def update_status(self, task_id: str, status: str, failure_substatus: str | None = None) -> None:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        self._db.connection.execute(
            "UPDATE agent_tasks SET status = ?, failure_substatus = ?, updated_at = ? WHERE task_id = ?",
            (status, failure_substatus, now, task_id),
        )
        self._db.connection.commit()

    def list_by_status(self, status: str) -> list[AgentTask]:
        rows = self._db.connection.execute(
            "SELECT * FROM agent_tasks WHERE status = ?", (status,)
        ).fetchall()
        return [self._row_to_task(r) for r in rows]

    def list_by_parent(self, parent_task_id: str) -> list[AgentTask]:
        rows = self._db.connection.execute(
            "SELECT * FROM agent_tasks WHERE parent_task_id = ?", (parent_task_id,)
        ).fetchall()
        return [self._row_to_task(r) for r in rows]

    @staticmethod
    def _row_to_task(row) -> AgentTask:
        return AgentTask(
            task_id=row["task_id"],
            agent_type=row["agent_type"],
            status=row["status"],
            parent_task_id=row["parent_task_id"],
            failure_substatus=row["failure_substatus"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
