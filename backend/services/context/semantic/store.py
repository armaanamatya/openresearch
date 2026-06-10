"""EmbeddingStore — protocol and Chroma implementation for vector search.

The protocol decouples the SemanticSearchTool from any specific vector DB.
ChromaEmbeddingStore wraps chromadb for in-process embedding + retrieval
using the default all-MiniLM-L6-v2 model (no API key required).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, Sequence

from backend.services.context.indexer.model import Chunk

logger = logging.getLogger(__name__)


class EmbeddingStoreError(Exception):
    pass


@dataclass(frozen=True)
class SearchResult:
    """A single vector search hit."""

    chunk_id: str
    source_id: str
    score: float
    text: str


class EmbeddingStore(Protocol):
    """Protocol for vector stores used by SemanticSearchTool."""

    def add_chunks(self, chunks: Sequence[Chunk]) -> int:
        """Embed and store chunks. Returns count of chunks added."""
        ...

    def query(self, text: str, *, top_k: int = 5) -> list[SearchResult]:
        """Return the top_k most similar chunks to the query text."""
        ...

    @property
    def count(self) -> int:
        """Number of chunks currently stored."""
        ...


def _chroma_available() -> bool:
    try:
        import chromadb  # noqa: F401
        return True
    except ImportError:
        return False


class ChromaEmbeddingStore:
    """Chroma-backed embedding store.

    Runs fully in-process using the default embedding function
    (all-MiniLM-L6-v2 via ONNX). No server or API key needed.

    Args:
        collection_name: Name of the Chroma collection.
        persist_directory: If provided, persist to disk. Otherwise in-memory.
    """

    def __init__(
        self,
        collection_name: str = "reprolab_chunks",
        persist_directory: str | None = None,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise EmbeddingStoreError(
                "chromadb is not installed. Install with: "
                "pip install reprolab-backend[semantic]"
            ) from exc

        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaEmbeddingStore initialized: collection=%s, count=%d",
            collection_name,
            self._collection.count(),
        )

    def add_chunks(self, chunks: Sequence[Chunk]) -> int:
        if not chunks:
            return 0

        # Chroma deduplicates by ID, so re-adding is safe.
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for chunk in chunks:
            if not chunk.text.strip():
                continue
            ids.append(chunk.id)
            documents.append(chunk.text)
            metadatas.append({
                "source_id": chunk.source_id,
                "project_id": chunk.project_id,
                "chunk_type": chunk.chunk_type.value,
            })

        if not ids:
            return 0

        # Chroma has a batch size limit; process in batches of 5000.
        batch_size = 5000
        added = 0
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_docs = documents[i : i + batch_size]
            batch_meta = metadatas[i : i + batch_size]
            self._collection.upsert(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta,
            )
            added += len(batch_ids)

        logger.info("Added %d chunks to Chroma (total: %d)", added, self._collection.count())
        return added

    def query(self, text: str, *, top_k: int = 5) -> list[SearchResult]:
        if not text.strip():
            return []
        if self._collection.count() == 0:
            return []

        # Chroma returns distances (lower = more similar for cosine).
        # Convert to similarity scores: score = 1 - distance.
        actual_k = min(top_k, self._collection.count())
        results = self._collection.query(
            query_texts=[text],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"],
        )

        hits: list[SearchResult] = []
        if not results["ids"] or not results["ids"][0]:
            return hits

        for chunk_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = max(0.0, 1.0 - dist)
            hits.append(SearchResult(
                chunk_id=chunk_id,
                source_id=meta.get("source_id", ""),
                score=round(score, 6),
                text=doc,
            ))

        return hits

    @property
    def count(self) -> int:
        return self._collection.count()


def try_create_chroma_store(
    collection_name: str = "reprolab_chunks",
    persist_directory: str | None = None,
) -> ChromaEmbeddingStore | None:
    """Try to create a ChromaEmbeddingStore; return None if chromadb is not installed."""
    if not _chroma_available():
        logger.info("chromadb not available; semantic search will use BM25 fallback")
        return None
    try:
        return ChromaEmbeddingStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
    except Exception:
        logger.warning("Failed to create ChromaEmbeddingStore", exc_info=True)
        return None


__all__ = [
    "ChromaEmbeddingStore",
    "EmbeddingStore",
    "EmbeddingStoreError",
    "SearchResult",
    "try_create_chroma_store",
]
