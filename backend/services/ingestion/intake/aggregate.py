"""Project aggregate — state-machine ONLY. No IO.

The aggregate enforces transition validity and emits the events that
record state changes. It does NOT call HTTP, read files, or invoke
parsers. That's the application service's job (see service.py).

Replay: state is rebuilt by feeding events one-by-one to `apply()`.
Test rules:
  - Aggregate methods that produce events are pure functions of
    (current_state, command).
  - `apply()` is idempotent in the sense that applying an already-applied
    event again advances state correctly (we re-apply during load from
    the event store).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from backend.messaging.event import DomainEvent
from backend.services.ingestion.intake.events import (
    PaperFetched,
    PaperFetchFailed,
    ProjectCreated,
)
from backend.services.ingestion.intake.sources import PaperSource


class ProjectState(str, Enum):
    NEW = "new"
    REGISTERED = "registered"
    FETCHED = "fetched"


class InvalidStateTransition(Exception):
    """Raised when a command is invalid for the aggregate's current state."""

    def __init__(self, state: ProjectState, attempted: str) -> None:
        super().__init__(
            f"Invalid command {attempted!r} for ProjectAggregate in state {state.value!r}"
        )
        self.state = state
        self.attempted = attempted


@dataclass
class ProjectAggregate:
    """The Project aggregate root.

    `version` is the count of applied events (== aggregate_version in
    the event store). `state` and `source` are the materialized current
    view. Construction starts at version=0, state=NEW.
    """

    project_id: str = ""
    state: ProjectState = ProjectState.NEW
    source: PaperSource | None = None
    version: int = 0

    @classmethod
    def empty(cls, project_id: str) -> "ProjectAggregate":
        """Make a fresh aggregate at version 0. Caller should then either
        apply replayed events or call `handle_register` to advance it."""
        return cls(project_id=project_id, state=ProjectState.NEW, version=0)

    # --- Command handlers (pure: state, command -> events) ----------------

    def handle_register(self, source: PaperSource) -> Sequence[DomainEvent]:
        if self.state is not ProjectState.NEW:
            raise InvalidStateTransition(self.state, "register")
        return [ProjectCreated(project_id=self.project_id, source=source)]

    def handle_fetch(
        self,
        *,
        raw_paper_path: str,
        pdf_sha256: str,
        pdf_size_bytes: int,
        fetched_via: str,
    ) -> Sequence[DomainEvent]:
        if self.state is not ProjectState.REGISTERED:
            raise InvalidStateTransition(self.state, "fetch")
        return [
            PaperFetched(
                project_id=self.project_id,
                raw_paper_path=raw_paper_path,
                pdf_sha256=pdf_sha256,
                pdf_size_bytes=pdf_size_bytes,
                fetched_via=fetched_via,
            )
        ]

    def handle_fetch_failure(
        self,
        *,
        cause_kind: str,
        cause_message: str,
        retryable: bool,
    ) -> Sequence[DomainEvent]:
        if self.state is not ProjectState.REGISTERED:
            raise InvalidStateTransition(self.state, "fetch_failure")
        return [
            PaperFetchFailed(
                project_id=self.project_id,
                cause_kind=cause_kind,
                cause_message=cause_message,
                retryable=retryable,
            )
        ]

    # --- Apply (event -> state mutation; replay-safe) ---------------------

    def apply(self, event: DomainEvent) -> None:
        if isinstance(event, ProjectCreated):
            self.state = ProjectState.REGISTERED
            self.source = event.source
            self.version += 1
        elif isinstance(event, PaperFetched):
            self.state = ProjectState.FETCHED
            self.version += 1
        elif isinstance(event, PaperFetchFailed):
            # Retry stays in REGISTERED.
            self.version += 1
        else:
            raise TypeError(
                f"ProjectAggregate cannot apply event of type {type(event).__name__}"
            )

    def apply_all(self, events: Sequence[DomainEvent]) -> None:
        for ev in events:
            self.apply(ev)


__all__ = ["InvalidStateTransition", "ProjectAggregate", "ProjectState"]
