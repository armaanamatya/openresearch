"""FastAPI application factory."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from backend import __version__
from backend.config import get_settings
from backend.persistence.database import Database
from backend.services.approval import ApprovalAction, ApprovalService, ApprovalState
from backend.services.context.graph import KnowledgeGraphService
from backend.services.context.memory import CrossProjectMemoryService, MemoryKind
from backend.services.datasets import DatasetCacheService
from backend.services.diagnostics import FailureDiagnosisService
from backend.services.research_workspace import ResearchWorkspaceService


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="ReproLab Agent",
        version=__version__,
        debug=settings.debug,
    )

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/phase2/projects/{project_id}/summary")
    async def phase2_summary(project_id: str, memory_query: str = "") -> dict:
        db = _database(settings.database_url)
        try:
            summary = ResearchWorkspaceService(db).summarize_project(
                project_id,
                memory_query=memory_query,
            )
            return summary.model_dump(mode="json")
        finally:
            db.close()

    @app.get("/phase2/projects/{project_id}/graph")
    async def phase2_graph_query(
        project_id: str,
        entity_type: str = "function",
        calls: str | None = None,
        imports: str | None = None,
        name: str | None = None,
        path_contains: str | None = None,
    ) -> dict:
        db = _database(settings.database_url)
        try:
            result = KnowledgeGraphService(db).query(
                entity_type,
                project_id=project_id,
                calls=calls,
                imports=imports,
                name=name,
                path_contains=path_contains,
            )
            return result.model_dump(mode="json")
        finally:
            db.close()

    @app.get("/phase2/memory/search")
    async def phase2_memory_search(
        query: str,
        kind: MemoryKind | None = None,
        limit: int = 5,
    ) -> dict:
        db = _database(settings.database_url)
        try:
            results = CrossProjectMemoryService(db).search(query, kind=kind, limit=limit)
            return {"results": [item.model_dump(mode="json") for item in results]}
        finally:
            db.close()

    @app.post("/phase2/approvals/evaluate")
    async def phase2_approval_evaluate(request: ApprovalEvaluateRequest) -> dict:
        db = _database(settings.database_url)
        try:
            service = ApprovalService(db)
            evaluation = service.evaluate(
                action=request.action,
                dataset_size_gb=request.dataset_size_gb,
                runtime_minutes=request.runtime_minutes,
                gpu_cost_usd=request.gpu_cost_usd,
                repo_trust_level=request.repo_trust_level,
                license_state=request.license_state,
                network_stage=request.network_stage,
                assumption_risk=request.assumption_risk,
                external_data=request.external_data,
                metadata=request.metadata,
            )
            approval = service.request_if_needed(
                project_id=request.project_id,
                label=request.label or request.action.replace("_", " ").title(),
                evaluation=evaluation,
            )
            return {
                "evaluation": evaluation.model_dump(mode="json"),
                "approval": approval.model_dump(mode="json") if approval else None,
            }
        finally:
            db.close()

    @app.post("/phase2/approvals/{approval_id}/resolve")
    async def phase2_approval_resolve(
        approval_id: str,
        request: ApprovalResolveRequest,
    ) -> dict:
        db = _database(settings.database_url)
        try:
            resolved = ApprovalService(db).resolve(
                approval_id,
                state=request.state,
                resolved_by=request.resolved_by,
                note=request.note,
            )
            return resolved.model_dump(mode="json")
        finally:
            db.close()

    @app.post("/phase2/datasets/plan")
    async def phase2_dataset_plan(request: DatasetPlanRequest) -> dict:
        db = _database(settings.database_url)
        try:
            entry = DatasetCacheService(db).plan(
                name=request.name,
                source_url=request.source_url,
                version=request.version,
                checksum=request.checksum,
                size_bytes=request.size_bytes,
                source_project_id=request.project_id,
                metadata=request.metadata,
            )
            return entry.model_dump(mode="json")
        finally:
            db.close()

    @app.post("/phase2/failures/diagnose")
    async def phase2_failure_diagnose(request: FailureDiagnoseRequest) -> dict:
        db = _database(settings.database_url)
        try:
            event = FailureDiagnosisService(db).diagnose(
                project_id=request.project_id,
                stage=request.stage,
                command=request.command,
                exit_code=request.exit_code,
                stdout=request.stdout,
                stderr=request.stderr,
                timed_out=request.timed_out,
                cause_kind=request.cause_kind,
                artifact_refs=tuple(request.artifact_refs),
            )
            return event.model_dump(mode="json")
        finally:
            db.close()

    return app


class ApprovalEvaluateRequest(BaseModel):
    project_id: str
    action: ApprovalAction
    label: str = ""
    dataset_size_gb: float | None = None
    runtime_minutes: float | None = None
    gpu_cost_usd: float | None = None
    repo_trust_level: str = ""
    license_state: str = ""
    network_stage: str = ""
    assumption_risk: str = ""
    external_data: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ApprovalResolveRequest(BaseModel):
    state: ApprovalState
    resolved_by: str = ""
    note: str = ""


class DatasetPlanRequest(BaseModel):
    project_id: str
    name: str
    source_url: str = ""
    version: str = ""
    checksum: str = ""
    size_bytes: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FailureDiagnoseRequest(BaseModel):
    project_id: str
    stage: str
    command: str = ""
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    cause_kind: str = ""
    artifact_refs: list[str] = Field(default_factory=list)


def _database(database_url: str) -> Database:
    db = Database(database_url)
    db.initialize()
    return db
