"""Runtime domain events."""

from __future__ import annotations

from typing import Any, ClassVar

from backend.messaging.event import DomainEvent, register_event


@register_event
class SandboxRequested(DomainEvent):
    event_type: ClassVar[str] = "sandbox_requested"
    schema_version: ClassVar[int] = 1

    project_id: str
    run_id: str
    config: dict[str, Any]


@register_event
class SandboxCreated(DomainEvent):
    event_type: ClassVar[str] = "sandbox_created"
    schema_version: ClassVar[int] = 1

    project_id: str
    run_id: str
    sandbox_id: str
    image: str


@register_event
class SandboxFailed(DomainEvent):
    event_type: ClassVar[str] = "sandbox_failed"
    schema_version: ClassVar[int] = 1

    project_id: str
    run_id: str
    cause_kind: str
    cause_message: str
    retryable: bool


@register_event
class CommandExecuted(DomainEvent):
    event_type: ClassVar[str] = "command_executed"
    schema_version: ClassVar[int] = 1

    project_id: str
    run_id: str
    sandbox_id: str
    command: str
    exit_code: int
    duration_seconds: float


@register_event
class CommandFailed(DomainEvent):
    event_type: ClassVar[str] = "command_failed"
    schema_version: ClassVar[int] = 1

    project_id: str
    run_id: str
    sandbox_id: str
    command: str
    exit_code: int | None
    duration_seconds: float
    cause_kind: str
    cause_message: str
    retryable: bool


@register_event
class SandboxDestroyed(DomainEvent):
    event_type: ClassVar[str] = "sandbox_destroyed"
    schema_version: ClassVar[int] = 1

    project_id: str
    run_id: str
    sandbox_id: str


__all__ = [
    "CommandExecuted",
    "CommandFailed",
    "SandboxCreated",
    "SandboxDestroyed",
    "SandboxFailed",
    "SandboxRequested",
]
