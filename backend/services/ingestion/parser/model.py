"""Parsed-paper artifact model (#13).

Section, Reference, Figure are frozen Pydantic models so they round-trip
cleanly through event payloads and the event store. IDs are
content-addressed: same paper text + same parser version → same IDs.
"""

from __future__ import annotations

import hashlib
from typing import NewType

from pydantic import BaseModel, ConfigDict

SectionId = NewType("SectionId", str)
ReferenceId = NewType("ReferenceId", str)
FigureId = NewType("FigureId", str)


def section_id_for(
    project_id: str,
    depth: int,
    char_offset: int,
    title: str,
    text: str,
) -> SectionId:
    """Deterministic SectionId.

    Composition includes (depth, char_offset) to distinguish sections
    that share a title within the same paper (e.g. multiple "Methods"
    subsections). Text content goes into the hash so a re-parse of the
    same PDF gives the same id only when the parser produces the same
    text.
    """
    h = hashlib.sha256()
    h.update(f"section:{project_id}:{depth}:{char_offset}:".encode())
    h.update(title.encode())
    h.update(b":")
    h.update(text.encode())
    return SectionId(f"sec_{h.hexdigest()[:16]}")


def reference_id_for(project_id: str, raw_text: str) -> ReferenceId:
    h = hashlib.sha256()
    h.update(f"reference:{project_id}:".encode())
    h.update(raw_text.encode())
    return ReferenceId(f"ref_{h.hexdigest()[:16]}")


def figure_id_for(project_id: str, page: int, caption: str) -> FigureId:
    h = hashlib.sha256()
    h.update(f"figure:{project_id}:{page}:".encode())
    h.update(caption.encode())
    return FigureId(f"fig_{h.hexdigest()[:16]}")


class Section(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str  # SectionId; using str for Pydantic JSON-friendliness
    project_id: str
    title: str
    text: str
    char_offset: int
    """Absolute offset in the concatenated full-text blob."""
    parent_id: str | None = None
    depth: int = 0


class Reference(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    project_id: str
    raw_text: str
    arxiv_id: str | None = None
    doi: str | None = None
    title: str | None = None


class Figure(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    project_id: str
    caption: str
    page: int
    description: str | None = None


__all__ = [
    "Figure",
    "FigureId",
    "Reference",
    "ReferenceId",
    "Section",
    "SectionId",
    "figure_id_for",
    "reference_id_for",
    "section_id_for",
]
