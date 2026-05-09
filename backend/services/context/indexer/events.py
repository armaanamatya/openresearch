"""Indexer domain events (#15)."""

from __future__ import annotations

from typing import ClassVar

from backend.messaging.event import DomainEvent, register_event
from backend.services.context.indexer.model import Chunk, SourceRef


@register_event
class IndexingStarted(DomainEvent):
    event_type: ClassVar[str] = "indexing_started"
    schema_version: ClassVar[int] = 1
    project_id: str
    chunker_name: str
    chunker_version: str


@register_event
class SourceRegistered(DomainEvent):
    event_type: ClassVar[str] = "source_registered"
    schema_version: ClassVar[int] = 1
    project_id: str
    source: SourceRef


@register_event
class ChunkCreated(DomainEvent):
    event_type: ClassVar[str] = "chunk_created"
    schema_version: ClassVar[int] = 1
    project_id: str
    chunk: Chunk


@register_event
class IndexingCompleted(DomainEvent):
    event_type: ClassVar[str] = "indexing_completed"
    schema_version: ClassVar[int] = 1
    project_id: str
    source_count: int
    chunk_count: int
    chunker_name: str
    chunker_version: str


@register_event
class IndexingFailed(DomainEvent):
    event_type: ClassVar[str] = "indexing_failed"
    schema_version: ClassVar[int] = 1
    project_id: str
    cause_kind: str
    cause_message: str
    retryable: bool


__all__ = [
    "ChunkCreated",
    "IndexingCompleted",
    "IndexingFailed",
    "IndexingStarted",
    "SourceRegistered",
]
