"""Phase 2 research workspace facade models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from backend.services.approval import ApprovalRequest
from backend.services.comparison import ComparisonReport
from backend.services.context.memory import MemoryRecord
from backend.services.datasets import DatasetCacheEntry
from backend.services.diagnostics import FailureEvent
from backend.services.scoring import ReproducibilityScore


class KnowledgeGraphStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    node_count: int = 0
    edge_count: int = 0
    function_count: int = 0
    class_count: int = 0
    module_count: int = 0


class ResearchWorkspaceSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    project_id: str
    generated_at: datetime
    graph: KnowledgeGraphStats
    memory_records: tuple[MemoryRecord, ...] = ()
    datasets: tuple[DatasetCacheEntry, ...] = ()
    pending_approvals: tuple[ApprovalRequest, ...] = ()
    recent_failures: tuple[FailureEvent, ...] = ()
    comparison_reports: tuple[ComparisonReport, ...] = ()
    reproducibility_score: ReproducibilityScore | None = None
    recommendations: tuple[str, ...] = ()


__all__ = ["KnowledgeGraphStats", "ResearchWorkspaceSummary"]
