"""Context indexer (#15): turns parsed sections into a registry of
SourceRefs and Chunks. Owns content-addressed identity composition
keyed on chunker version so re-indexing the same parsed paper is
idempotent."""

from backend.services.context.indexer.aggregate import (
    IndexAggregate,
    IndexState,
    InvalidIndexTransition,
)
from backend.services.context.indexer.chunkers.section import SectionChunker
from backend.services.context.indexer.events import (
    ChunkCreated,
    IndexingCompleted,
    IndexingFailed,
    IndexingStarted,
    SourceRegistered,
)
from backend.services.context.indexer.model import (
    Chunk,
    ChunkType,
    SourceKind,
    SourceRef,
    chunk_id_for,
    source_id_for,
)
from backend.services.context.indexer.projections import SourcesProjection
from backend.services.context.indexer.service import (
    IndexerAppService,
    IndexerError,
    StartIndexing,
)

__all__ = [
    "Chunk",
    "ChunkCreated",
    "ChunkType",
    "IndexAggregate",
    "IndexState",
    "IndexerAppService",
    "IndexerError",
    "IndexingCompleted",
    "IndexingFailed",
    "IndexingStarted",
    "InvalidIndexTransition",
    "SectionChunker",
    "SourceKind",
    "SourceRef",
    "SourceRegistered",
    "SourcesProjection",
    "StartIndexing",
    "chunk_id_for",
    "source_id_for",
]
