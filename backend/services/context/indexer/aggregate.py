"""IndexAggregate — pure state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from backend.messaging.event import DomainEvent
from backend.services.context.indexer.events import (
    ChunkCreated,
    IndexingCompleted,
    IndexingFailed,
    IndexingStarted,
    SourceRegistered,
)


class IndexState(str, Enum):
    PENDING = "pending"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"


class InvalidIndexTransition(Exception):
    def __init__(self, state: IndexState, attempted: str) -> None:
        super().__init__(
            f"Invalid command {attempted!r} for IndexAggregate in state {state.value!r}"
        )
        self.state = state
        self.attempted = attempted


@dataclass
class IndexAggregate:
    project_id: str = ""
    state: IndexState = IndexState.PENDING
    chunker_name: str = ""
    chunker_version: str = ""
    source_count: int = 0
    chunk_count: int = 0
    version: int = 0

    @classmethod
    def empty(cls, project_id: str) -> "IndexAggregate":
        return cls(project_id=project_id, state=IndexState.PENDING, version=0)

    def handle_start(
        self, chunker_name: str, chunker_version: str
    ) -> Sequence[DomainEvent]:
        if self.state in (IndexState.INDEXING, IndexState.INDEXED):
            raise InvalidIndexTransition(self.state, "start")
        return [
            IndexingStarted(
                project_id=self.project_id,
                chunker_name=chunker_name,
                chunker_version=chunker_version,
            )
        ]

    def apply(self, event: DomainEvent) -> None:
        if isinstance(event, IndexingStarted):
            self.state = IndexState.INDEXING
            self.chunker_name = event.chunker_name
            self.chunker_version = event.chunker_version
        elif isinstance(event, SourceRegistered):
            self.source_count += 1
        elif isinstance(event, ChunkCreated):
            self.chunk_count += 1
        elif isinstance(event, IndexingCompleted):
            self.state = IndexState.INDEXED
        elif isinstance(event, IndexingFailed):
            self.state = IndexState.FAILED
        else:
            raise TypeError(
                f"IndexAggregate cannot apply event of type {type(event).__name__}"
            )
        self.version += 1

    def apply_all(self, events: Sequence[DomainEvent]) -> None:
        for ev in events:
            self.apply(ev)


__all__ = ["IndexAggregate", "IndexState", "InvalidIndexTransition"]
