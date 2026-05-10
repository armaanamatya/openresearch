"""Semantic Index — embedding-based vector search over indexed chunks."""

from backend.services.context.semantic.store import (
    EmbeddingStore,
    EmbeddingStoreError,
)

__all__ = [
    "EmbeddingStore",
    "EmbeddingStoreError",
]
