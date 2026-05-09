"""Discovery aggregate state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from backend.messaging.event import DomainEvent
from backend.services.ingestion.discovery.events import (
    ArtifactDiscovered,
    DiscoveryCompleted,
    DiscoveryFailed,
    DiscoveryStarted,
)


class DiscoveryState(str, Enum):
    PENDING = "pending"
    DISCOVERING = "discovering"
    COMPLETED = "completed"
    FAILED = "failed"


class InvalidDiscoveryTransition(Exception):
    def __init__(self, state: DiscoveryState, attempted: str) -> None:
        super().__init__(
            f"Invalid command {attempted!r} for DiscoveryAggregate in "
            f"state {state.value!r}"
        )
        self.state = state
        self.attempted = attempted


@dataclass
class DiscoveryAggregate:
    project_id: str = ""
    state: DiscoveryState = DiscoveryState.PENDING
    artifact_count: int = 0
    version: int = 0

    @classmethod
    def empty(cls, project_id: str) -> "DiscoveryAggregate":
        return cls(project_id=project_id)

    def handle_start(self, adapter_names: tuple[str, ...]) -> Sequence[DomainEvent]:
        if self.state is DiscoveryState.DISCOVERING:
            raise InvalidDiscoveryTransition(self.state, "start")
        if self.state is DiscoveryState.COMPLETED:
            raise InvalidDiscoveryTransition(self.state, "start")
        return [DiscoveryStarted(project_id=self.project_id, adapter_names=adapter_names)]

    def apply(self, event: DomainEvent) -> None:
        if isinstance(event, DiscoveryStarted):
            self.state = DiscoveryState.DISCOVERING
        elif isinstance(event, ArtifactDiscovered):
            self.artifact_count += 1
        elif isinstance(event, DiscoveryCompleted):
            self.state = DiscoveryState.COMPLETED
            self.artifact_count = event.artifact_count
        elif isinstance(event, DiscoveryFailed):
            self.state = DiscoveryState.FAILED
        else:
            raise TypeError(
                f"DiscoveryAggregate cannot apply event of type {type(event).__name__}"
            )
        self.version += 1

    def apply_all(self, events: Sequence[DomainEvent]) -> None:
        for event in events:
            self.apply(event)


__all__ = [
    "DiscoveryAggregate",
    "DiscoveryState",
    "InvalidDiscoveryTransition",
]
