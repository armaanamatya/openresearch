"""SectionChunker — one chunk per section.

Sorts incoming sections by (depth, char_offset, section_id) before
chunking so the emitted chunks are deterministic regardless of the
parser's emit order. ChunkIds are content-addressed and include the
chunker name + version.
"""

from __future__ import annotations

from typing import Iterable, List

from backend.services.context.indexer.model import (
    Chunk,
    ChunkType,
    SourceRef,
    chunk_id_for,
)
from backend.services.ingestion.parser.model import Section


class SectionChunker:
    """One chunk per parsed section."""

    name = "section"
    version = "1"

    def chunk(
        self,
        *,
        sources_by_upstream: dict[str, SourceRef],
        sections: Iterable[Section],
    ) -> List[Chunk]:
        # Materialize so we can sort.
        sections_list = list(sections)
        sections_list.sort(key=lambda s: (s.depth, s.char_offset, s.id))

        chunks: List[Chunk] = []
        for section in sections_list:
            src = sources_by_upstream.get(section.id)
            if src is None:
                continue
            span = (section.char_offset, section.char_offset + len(section.text))
            cid = chunk_id_for(
                source_id=src.id,
                chunker_name=self.name,
                chunker_version=self.version,
                text=section.text,
                span=span,
                chunk_type=ChunkType.section,
            )
            chunks.append(
                Chunk(
                    id=cid,
                    source_id=src.id,
                    project_id=section.project_id,
                    text=section.text,
                    span=span,
                    chunk_type=ChunkType.section,
                    parent_chunk_id=None,
                )
            )
        return chunks


__all__ = ["SectionChunker"]
