"""Agent task lifecycle service with state machine enforcement."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from pydantic import BaseModel

from backend.persistence.database import Database
from backend.persistence.repositories.task_repository import TaskRepository
from backend.schemas.events import EventPayload
from backend.schemas.tasks import AgentTask, TaskStatus


class InvalidTransitionError(Exception):
    pass


class TransitionRecord(BaseModel):
    task_id: str
    from_status: str
    to_status: str
    timestamp: datetime


# Valid transitions: from_status -> set of allowed to_statuses
_VALID_TRANSITIONS: dict[str, set[str]] = {
    "created": {"context_prepared"},
    "context_prepared": {"running"},
    "running": {"artifact_submitted", "failed", "blocked_requires_human"},
    "artifact_submitted": {"verification_pending", "failed"},
    "verification_pending": {"verified", "failed", "blocked_requires_human"},
    # Terminal states — no outgoing transitions
    "verified": set(),
    "failed": set(),
    "blocked_requires_human": set(),
}


class TaskLifecycleService:
    def __init__(self, db: Database) -> None:
        self._db = db
        self._repo = TaskRepository(db)
        self._listeners: list[Callable[[EventPayload], Any]] = []
        self._history: dict[str, list[TransitionRecord]] = {}

    def on_event(self, callback: Callable[[EventPayload], Any]) -> None:
        self._listeners.append(callback)

    def _emit(self, event: EventPayload) -> None:
        for listener in self._listeners:
            listener(event)

    def create_task(
        self,
        agent_type: str,
        parent_task_id: str | None = None,
    ) -> AgentTask:
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        task = AgentTask(
            task_id=task_id,
            agent_type=agent_type,
            status="created",
            parent_task_id=parent_task_id,
        )
        self._repo.create(task)

        # Record initial history
        now = datetime.now(timezone.utc)
        self._history.setdefault(task_id, []).append(
            TransitionRecord(
                task_id=task_id,
                from_status="",
                to_status="created",
                timestamp=now,
            )
        )

        self._emit(EventPayload(
            event="agent_started",
            agent_id=agent_type,
            data={"task_id": task_id, "status": "created"},
        ))
        return task

    def transition(
        self,
        task_id: str,
        to_status: str,
        failure_substatus: str | None = None,
    ) -> AgentTask:
        task = self._repo.get_by_id(task_id)
        if task is None:
            raise ValueError(f"Task {task_id} not found")

        current = task.status.value if hasattr(task.status, "value") else task.status
        allowed = _VALID_TRANSITIONS.get(current, set())

        if to_status not in allowed:
            raise InvalidTransitionError(
                f"Cannot transition from '{current}' to '{to_status}'. "
                f"Allowed: {allowed}"
            )

        self._repo.update_status(task_id, to_status, failure_substatus)
        updated = self._repo.get_by_id(task_id)

        # Record history
        now = datetime.now(timezone.utc)
        self._history.setdefault(task_id, []).append(
            TransitionRecord(
                task_id=task_id,
                from_status=current,
                to_status=to_status,
                timestamp=now,
            )
        )

        self._emit(EventPayload(
            event="task_status_changed",
            agent_id=updated.agent_type,
            data={
                "task_id": task_id,
                "old_status": current,
                "new_status": to_status,
            },
        ))
        return updated

    def get_task(self, task_id: str) -> AgentTask | None:
        return self._repo.get_by_id(task_id)

    def get_transition_history(self, task_id: str) -> list[TransitionRecord]:
        return self._history.get(task_id, [])
