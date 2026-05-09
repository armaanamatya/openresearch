"""Source / Chunk model + content-addressed ID composition.

Per Codex feedback (2026-05-09): chunk IDs include the chunker name and
version so a different chunker over the same text produces different
ids — no silent reuse on chunker upgrade.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import NewType

from pydantic import BaseModel, ConfigDict

SourceId = NewType("SourceId", str)
ChunkId = NewType("ChunkId", str)


class SourceKind(str, Enum):
    paper_section = "paper_section"
    paper_reference = "paper_reference"
    repository = "repository"
    dataset = "dataset"
    issue = "issue"
    discussion = "discussion"
    # Future: repo_code_file, model_card, provenance_artifact (#19 cross-link).


class ChunkType(str, Enum):
    section = "section"
    paragraph = "paragraph"
    artifact_metadata = "artifact_metadata"


def source_id_for(*, project_id: str, kind: SourceKind, upstream_id: str) -> SourceId:
    h = hashlib.sha256()
    h.update(f"source:{project_id}:{kind.value}:".encode())
    h.update(upstream_id.encode())
    return SourceId(f"src_{h.hexdigest()[:16]}")


def chunk_id_for(
    *,
    source_id: str,
    chunker_name: str,
    chunker_version: str,
    text: str,
    span: tuple[int, int],
    chunk_type: ChunkType = ChunkType.section,
    normalization: str = "v1",
) -> ChunkId:
    h = hashlib.sha256()
    h.update(f"chunk:{source_id}:{chunker_name}:{chunker_version}:".encode())
    h.update(f"{chunk_type.value}:{span[0]}:{span[1]}:{normalization}:".encode())
    h.update(text.encode())
    return ChunkId(f"chk_{h.hexdigest()[:24]}")


class SourceRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    project_id: str
    kind: SourceKind
    locator: str
    """Human-readable: 'PPO §3.2', 'github.com/openai/baselines/README.md'."""
    upstream_id: str
    """Identity from the producing layer (Section.id, Reference.id, ...)."""


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    source_id: str
    project_id: str
    text: str
    span: tuple[int, int]
    chunk_type: ChunkType
    parent_chunk_id: str | None = None


__all__ = [
    "Chunk",
    "ChunkId",
    "ChunkType",
    "SourceId",
    "SourceKind",
    "SourceRef",
    "chunk_id_for",
    "source_id_for",
]
