"""WorkspaceAggregate — pure state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from backend.messaging.event import DomainEvent
from backend.services.context.workspace.events import (
    CitationAttached,
    ToolInvoked,
    VariableEnriched,
    VariableLoaded,
    VariablePromoted,
    WorkspaceClosed,
    WorkspaceCreated,
    WorkspaceReady,
)


class WorkspaceState(str, Enum):
    NEW = "new"
    LOADING = "loading"
    READY = "ready"
    CLOSED = "closed"


class InvalidWorkspaceTransition(Exception):
    def __init__(self, state: WorkspaceState, attempted: str) -> None:
        super().__init__(
            f"Invalid command {attempted!r} for WorkspaceAggregate in "
            f"state {state.value!r}"
        )
        self.state = state
        self.attempted = attempted


@dataclass
class WorkspaceAggregate:
    workspace_id: str = ""
    project_id: str = ""
    agent_name: str = ""
    state: WorkspaceState = WorkspaceState.NEW
    variable_count: int = 0
    version: int = 0

    @classmethod
    def empty(cls, workspace_id: str) -> "WorkspaceAggregate":
        return cls(workspace_id=workspace_id, state=WorkspaceState.NEW, version=0)

    def handle_create(
        self,
        *,
        project_id: str,
        agent_name: str,
        parent_workspace_id: str | None = None,
        branch_id: str | None = None,
        task_id: str | None = None,
    ) -> Sequence[DomainEvent]:
        if self.state is not WorkspaceState.NEW:
            raise InvalidWorkspaceTransition(self.state, "create")
        return [
            WorkspaceCreated(
                workspace_id=self.workspace_id,
                project_id=project_id,
                agent_name=agent_name,
                parent_workspace_id=parent_workspace_id,
                branch_id=branch_id,
                task_id=task_id,
            )
        ]

    def apply(self, event: DomainEvent) -> None:
        if isinstance(event, WorkspaceCreated):
            self.state = WorkspaceState.LOADING
            self.project_id = event.project_id
            self.agent_name = event.agent_name
        elif isinstance(event, (VariableLoaded, VariableEnriched)):
            self.variable_count += 1
        elif isinstance(event, CitationAttached):
            pass  # decisions recorded; no aggregate state change
        elif isinstance(event, ToolInvoked):
            pass  # call history recorded by projection, not aggregate
        elif isinstance(event, VariablePromoted):
            pass  # scope change recorded; projection tracks per-variable scope
        elif isinstance(event, WorkspaceReady):
            self.state = WorkspaceState.READY
        elif isinstance(event, WorkspaceClosed):
            self.state = WorkspaceState.CLOSED
        else:
            raise TypeError(
                f"WorkspaceAggregate cannot apply event of type {type(event).__name__}"
            )
        self.version += 1

    def apply_all(self, events: Sequence[DomainEvent]) -> None:
        for ev in events:
            self.apply(ev)


__all__ = [
    "InvalidWorkspaceTransition",
    "WorkspaceAggregate",
    "WorkspaceState",
]
