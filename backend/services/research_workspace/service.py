"""Facade for the Phase 2 research workspace."""

from __future__ import annotations

from datetime import datetime, timezone

from backend.persistence.database import Database
from backend.services.approval import ApprovalService
from backend.services.comparison import MultiPaperComparisonService
from backend.services.context.graph import KnowledgeGraphService
from backend.services.context.memory import CrossProjectMemoryService
from backend.services.datasets import DatasetCacheService
from backend.services.diagnostics import FailureDiagnosisService
from backend.services.research_workspace.model import (
    KnowledgeGraphStats,
    ResearchWorkspaceSummary,
)
from backend.services.scoring import ReproducibilityScore


class ResearchWorkspaceService:
    """Read model spanning Phase 2's graph, memory, cache, approvals, and failures."""

    def __init__(self, db: Database, *, dataset_cache_root: str = "datasets") -> None:
        self._db = db
        self._graph = KnowledgeGraphService(db)
        self._memory = CrossProjectMemoryService(db)
        self._datasets = DatasetCacheService(db, cache_root=dataset_cache_root)
        self._approvals = ApprovalService(db)
        self._diagnostics = FailureDiagnosisService(db)
        self._comparisons = MultiPaperComparisonService(db)

    def summarize_project(
        self,
        project_id: str,
        *,
        memory_query: str = "",
        comparison_limit: int = 5,
        reproducibility_score: ReproducibilityScore | None = None,
    ) -> ResearchWorkspaceSummary:
        graph = self._graph_stats(project_id)
        memory_records = (
            tuple(hit.record for hit in self._memory.search(memory_query, limit=5))
            if memory_query
            else self._memory.list_records(source_project_id=project_id)
        )
        datasets = self._datasets.list_entries(source_project_id=project_id)
        pending = self._approvals.list_requests(project_id=project_id, state="pending")
        failures = self._diagnostics.list_events(project_id=project_id)[:10]
        comparisons = self._comparisons.list_reports(limit=comparison_limit)
        return ResearchWorkspaceSummary(
            project_id=project_id,
            generated_at=datetime.now(timezone.utc),
            graph=graph,
            memory_records=memory_records,
            datasets=datasets,
            pending_approvals=pending,
            recent_failures=failures,
            comparison_reports=comparisons,
            reproducibility_score=reproducibility_score,
            recommendations=_recommendations(graph, memory_records, datasets, pending, failures),
        )

    def _graph_stats(self, project_id: str) -> KnowledgeGraphStats:
        nodes = self._graph.list_nodes(project_id=project_id)
        edges = self._graph.list_edges(project_id=project_id)
        return KnowledgeGraphStats(
            node_count=len(nodes),
            edge_count=len(edges),
            function_count=sum(1 for node in nodes if node.kind in {"function", "method"}),
            class_count=sum(1 for node in nodes if node.kind == "class"),
            module_count=sum(1 for node in nodes if node.kind == "module"),
        )


def _recommendations(
    graph: KnowledgeGraphStats,
    memory_records: tuple,
    datasets: tuple,
    pending: tuple,
    failures: tuple,
) -> tuple[str, ...]:
    out: list[str] = []
    if graph.node_count == 0:
        out.append("Build the AST knowledge graph for the baseline repository.")
    if not memory_records:
        out.append("Promote verified environment recipes or failure modes into cross-project memory.")
    if any(dataset.status != "available" for dataset in datasets):
        out.append("Resolve blocked or failed dataset cache entries before long runs.")
    if pending:
        out.append("Resolve pending human approvals before executing risky actions.")
    if failures:
        out.append("Review recent failure diagnoses before launching new improvement paths.")
    if not out:
        out.append("Workspace is ready for comparison or improvement planning.")
    return tuple(out)


__all__ = ["ResearchWorkspaceService"]
