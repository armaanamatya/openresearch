"""Models for multi-paper comparative studies."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PaperRunSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    project_id: str
    paper_id: str = ""
    paper_title: str
    method_name: str = ""
    dataset: str = ""
    split: str = ""
    metric_name: str = ""
    metric_value: float | None = None
    status: str = ""
    reduced_run: bool = False
    assumptions: tuple[str, ...] = ()
    artifact_refs: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


class ComparableGroup(BaseModel):
    model_config = ConfigDict(frozen=True)

    group_key: str
    dataset: str
    split: str
    metric_name: str
    runs: tuple[PaperRunSummary, ...]
    best_project_id: str | None = None
    best_metric_value: float | None = None
    notes: tuple[str, ...] = ()


class IncomparableRun(BaseModel):
    model_config = ConfigDict(frozen=True)

    project_id: str
    reason: str


class ComparisonReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    comparison_id: str
    created_at: datetime
    status: Literal["complete", "partial"]
    groups: tuple[ComparableGroup, ...] = ()
    incomparable_runs: tuple[IncomparableRun, ...] = ()
    shared_assumptions: tuple[str, ...] = ()
    recommendations: tuple[str, ...] = ()


__all__ = [
    "ComparableGroup",
    "ComparisonReport",
    "IncomparableRun",
    "PaperRunSummary",
]
