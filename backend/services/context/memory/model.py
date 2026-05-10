"""Cross-project memory models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


MemoryKind = Literal[
    "environment_recipe",
    "baseline_result",
    "failure_mode",
    "improvement_result",
    "dataset_note",
    "verification_caveat",
]


class MemoryRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    kind: MemoryKind
    source_project_id: str
    paper_id: str = ""
    title: str
    summary: str
    tags: tuple[str, ...] = ()
    metrics: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: tuple[str, ...] = ()
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: datetime


class MemorySearchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    record: MemoryRecord
    score: float
    matched_terms: tuple[str, ...] = ()


__all__ = ["MemoryKind", "MemoryRecord", "MemorySearchResult"]
