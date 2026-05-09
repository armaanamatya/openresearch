"""Deterministic local semantic-search fallback over indexed chunks.

This is not an embedding service; it is a production-safe lexical ranker
that gives agents a real retrieval primitive before Chroma is connected.
Every result is returned with evidence-grade citations.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from backend.schemas.citations import Citation
from backend.services.context.indexer.model import Chunk
from backend.services.context.indexer.projections import SourcesProjection
from backend.services.context.workspace.model import Cited
from backend.services.context.workspace.tools.interface import WorkspaceToolError


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_QUOTE_TRUNCATE = 240


class SemanticSearchTool:
    """Rank indexed chunks with a small BM25-style lexical scorer."""

    name = "semantic_search"

    def __init__(self, projection: SourcesProjection) -> None:
        self._proj = projection

    def call(
        self,
        *,
        workspace_id: str,
        query: str,
        project_id: str | None = None,
        limit: int = 5,
    ) -> Cited[dict[str, Any]]:
        del workspace_id
        if limit < 1:
            raise WorkspaceToolError("semantic_search limit must be >= 1")
        query_terms = _tokenize(query)
        if not query_terms:
            raise WorkspaceToolError("semantic_search query must contain searchable text")

        chunks = self._proj.list_chunks(project_id)
        if not chunks:
            raise WorkspaceToolError("semantic_search has no indexed chunks to search")

        ranked = _rank_chunks(chunks, query_terms)
        top = ranked[:limit]
        if not top:
            raise WorkspaceToolError("semantic_search found no matching chunks")

        results: list[dict[str, Any]] = []
        citations: list[Citation] = []
        for score, chunk in top:
            source = self._proj.get_source(chunk.source_id)
            locator = source.locator if source is not None else chunk.source_id
            quote = chunk.text[:_QUOTE_TRUNCATE]
            results.append(
                {
                    "source_id": chunk.source_id,
                    "chunk_id": chunk.id,
                    "score": round(score, 6),
                    "locator": locator,
                    "excerpt": quote,
                }
            )
            citations.append(
                Citation(
                    source_id=chunk.source_id,
                    chunk_id=chunk.id,
                    quote=quote,
                    locator=locator,
                    confidence=min(1.0, max(0.2, score / 5.0)),
                )
            )

        return Cited(
            value={"query": query, "results": results},
            citations=tuple(citations),
        )


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _rank_chunks(chunks: tuple[Chunk, ...], query_terms: list[str]) -> list[tuple[float, Chunk]]:
    term_docs: Counter[str] = Counter()
    chunk_terms: dict[str, Counter[str]] = {}
    chunk_lengths: dict[str, int] = {}

    for chunk in chunks:
        counts = Counter(_tokenize(chunk.text))
        chunk_terms[chunk.id] = counts
        chunk_lengths[chunk.id] = sum(counts.values())
        for term in counts:
            term_docs[term] += 1

    avg_len = sum(chunk_lengths.values()) / max(1, len(chunks))
    query_counts = Counter(query_terms)
    scored: list[tuple[float, Chunk]] = []
    for chunk in chunks:
        counts = chunk_terms[chunk.id]
        score = 0.0
        length = chunk_lengths[chunk.id] or 1
        for term, query_weight in query_counts.items():
            freq = counts.get(term, 0)
            if freq == 0:
                continue
            idf = math.log(1 + (len(chunks) - term_docs[term] + 0.5) / (term_docs[term] + 0.5))
            denom = freq + 1.2 * (1 - 0.75 + 0.75 * length / avg_len)
            score += query_weight * idf * (freq * 2.2 / denom)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda item: (-item[0], item[1].source_id, item[1].id))
    return scored


__all__ = ["SemanticSearchTool"]
