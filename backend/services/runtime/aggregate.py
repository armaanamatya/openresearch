"""Sandbox aggregate state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from backend.messaging.event import DomainEvent
from backend.services.runtime.events import (
    CommandExecuted,
    CommandFailed,
    SandboxCreated,
    SandboxDestroyed,
    SandboxFailed,
    SandboxRequested,
)
from backend.services.runtime.interface import ExecResult, SandboxConfig, SandboxRuntimeError


class SandboxState(str, Enum):
    NEW = "new"
    REQUESTED = "requested"
    CREATED = "created"
    RUNNING = "running"
    DESTROYED = "destroyed"
    FAILED = "failed"


class InvalidSandboxTransition(Exception):
    def __init__(self, state: SandboxState, attempted: str) -> None:
        super().__init__(
            f"Invalid command {attempted!r} for SandboxAggregate in state {state.value!r}"
        )
        self.state = state
        self.attempted = attempted


@dataclass
class SandboxAggregate:
    project_id: str
    run_id: str
    state: SandboxState = SandboxState.NEW
    sandbox_id: str = ""
    version: int = 0

    @classmethod
    def empty(cls, project_id: str, run_id: str) -> "SandboxAggregate":
        return cls(project_id=project_id, run_id=run_id)

    def handle_request(self, config: SandboxConfig) -> Sequence[DomainEvent]:
        if self.state is not SandboxState.NEW:
            raise InvalidSandboxTransition(self.state, "request")
        return [
            SandboxRequested(
                project_id=self.project_id,
                run_id=self.run_id,
                config=config.model_dump(mode="json"),
            )
        ]

    def handle_created(self, sandbox_id: str, image: str) -> Sequence[DomainEvent]:
        if self.state is not SandboxState.REQUESTED:
            raise InvalidSandboxTransition(self.state, "created")
        return [
            SandboxCreated(
                project_id=self.project_id,
                run_id=self.run_id,
                sandbox_id=sandbox_id,
                image=image,
            )
        ]

    def handle_failed(self, exc: SandboxRuntimeError) -> Sequence[DomainEvent]:
        if self.state is SandboxState.DESTROYED:
            raise InvalidSandboxTransition(self.state, "failed")
        return [
            SandboxFailed(
                project_id=self.project_id,
                run_id=self.run_id,
                cause_kind=exc.cause_kind.value,
                cause_message=str(exc),
                retryable=exc.retryable,
            )
        ]

    def handle_command_result(self, result: ExecResult) -> Sequence[DomainEvent]:
        if self.state not in (SandboxState.CREATED, SandboxState.RUNNING):
            raise InvalidSandboxTransition(self.state, "command_result")
        if result.succeeded:
            return [
                CommandExecuted(
                    project_id=self.project_id,
                    run_id=self.run_id,
                    sandbox_id=self.sandbox_id,
                    command=result.command,
                    exit_code=result.exit_code or 0,
                    duration_seconds=result.duration_seconds,
                )
            ]
        cause = result.cause_kind.value if result.cause_kind else "command_failed"
        message = result.stderr or result.stdout or f"Command exited {result.exit_code}"
        return [
            CommandFailed(
                project_id=self.project_id,
                run_id=self.run_id,
                sandbox_id=self.sandbox_id,
                command=result.command,
                exit_code=result.exit_code,
                duration_seconds=result.duration_seconds,
                cause_kind=cause,
                cause_message=message,
                retryable=result.timed_out,
            )
        ]

    def handle_destroyed(self) -> Sequence[DomainEvent]:
        if self.state is SandboxState.DESTROYED:
            raise InvalidSandboxTransition(self.state, "destroyed")
        return [
            SandboxDestroyed(
                project_id=self.project_id,
                run_id=self.run_id,
                sandbox_id=self.sandbox_id,
            )
        ]

    def apply(self, event: DomainEvent) -> None:
        if isinstance(event, SandboxRequested):
            self.state = SandboxState.REQUESTED
        elif isinstance(event, SandboxCreated):
            self.state = SandboxState.CREATED
            self.sandbox_id = event.sandbox_id
        elif isinstance(event, CommandExecuted):
            self.state = SandboxState.RUNNING
        elif isinstance(event, CommandFailed):
            self.state = SandboxState.RUNNING
        elif isinstance(event, SandboxDestroyed):
            self.state = SandboxState.DESTROYED
        elif isinstance(event, SandboxFailed):
            self.state = SandboxState.FAILED
        else:
            raise TypeError(
                f"SandboxAggregate cannot apply event of type {type(event).__name__}"
            )
        self.version += 1

    def apply_all(self, events: Sequence[DomainEvent]) -> None:
        for event in events:
            self.apply(event)


__all__ = [
    "InvalidSandboxTransition",
    "SandboxAggregate",
    "SandboxState",
]
