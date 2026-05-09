"""SourcesProjection — in-memory read model rebuilt from index events.

The projection subscribes to `index` events and maintains:
  - sources_by_id: SourceId -> SourceRef
  - chunks_by_id: ChunkId -> Chunk
  - chunks_by_source: SourceId -> tuple[Chunk, ...]

For the slice this lives in-memory; SQLite-backed projection comes
later when concurrency / persistence matters.
"""

from __future__ import annotations

from typing import Iterable

from backend.services.context.indexer.model import Chunk, SourceRef


class SourcesProjection:
    """In-memory projection. Apply events to mutate; query by id."""

    def __init__(self) -> None:
        self._sources: dict[str, SourceRef] = {}
        self._chunks: dict[str, Chunk] = {}
        self._chunks_by_source: dict[str, list[Chunk]] = {}

    def apply_source(self, source: SourceRef) -> None:
        self._sources[source.id] = source

    def apply_chunk(self, chunk: Chunk) -> None:
        self._chunks[chunk.id] = chunk
        self._chunks_by_source.setdefault(chunk.source_id, []).append(chunk)

    def get_source(self, source_id: str) -> SourceRef | None:
        return self._sources.get(source_id)

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def chunks_for_source(self, source_id: str) -> tuple[Chunk, ...]:
        return tuple(self._chunks_by_source.get(source_id, ()))

    def list_chunks(self, project_id: str | None = None) -> tuple[Chunk, ...]:
        if project_id is None:
            return tuple(self._chunks.values())
        return tuple(c for c in self._chunks.values() if c.project_id == project_id)

    def list_sources(self, project_id: str | None = None) -> tuple[SourceRef, ...]:
        if project_id is None:
            return tuple(self._sources.values())
        return tuple(s for s in self._sources.values() if s.project_id == project_id)

    @property
    def source_count(self) -> int:
        return len(self._sources)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)


__all__ = ["SourcesProjection"]
