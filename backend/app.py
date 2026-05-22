"""FastAPI application factory."""

from __future__ import annotations

import hmac
import re

from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend import __version__
from backend.agents.topology import PipelineTopology, default_topology
from backend.config import get_settings
from backend.persistence.database import Database
from backend.services.approval import ApprovalAction, ApprovalService, ApprovalState
from backend.services.context.graph import KnowledgeGraphService
from backend.services.context.memory import CrossProjectMemoryService, MemoryKind
from backend.services.datasets import DatasetCacheService
from backend.services.diagnostics import FailureDiagnosisService
from backend.services.events.live_runs import FileLiveRunService, StartRunRequest
from backend.services.research_workspace import ResearchWorkspaceService


def _enforce_demo_gate(provided_secret: str | None, configured_secret: str) -> None:
    """Require a matching X-Demo-Secret header on the run-start endpoints.

    When ``configured_secret`` is empty the gate is disabled (local dev).
    When set, the caller must present a matching secret; a mismatch or a
    missing secret raises 403. The comparison is constant-time.
    """
    if not configured_secret:
        return
    if not provided_secret or not hmac.compare_digest(provided_secret, configured_secret):
        raise HTTPException(status_code=403, detail="A valid demo access secret is required.")


def create_app(*, run_service: Any | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    import os as _os
    import sys as _sys
    from pathlib import Path as _Path
    settings = get_settings()
    # Tier 2a — install pipeline.log + pipeline.jsonl on the root logger when
    # REPROLAB_LOG_DIR / REPROLAB_RUNS_ROOT is set. No-op otherwise.
    from backend.observability.run_logging import configure_root_logger
    configure_root_logger()

    # Resolve runs_root with a direct os.environ.get fallback. The Settings
    # singleton (`_settings_cache`) is per-process; under uvicorn --reload on
    # Windows the reloader and the worker are SEPARATE processes and the
    # worker's get_settings() apparently returned None earlier despite the
    # env var being baked into the cmd shim. Reading os.environ directly
    # here removes Settings as a possible failure point, and we still benefit
    # from Settings when env is unset (e.g. tests).
    env_runs_root = _os.environ.get("REPROLAB_RUNS_ROOT")
    effective_runs_root = settings.runs_root
    if effective_runs_root is None and env_runs_root:
        effective_runs_root = _Path(env_runs_root)

    # Diagnostic + marker. The print goes to backend.log; the marker file
    # captures the FULL story per-process so we can compare reloader vs
    # worker. backend.log only seems to capture stdout from one of them.
    print(
        f"[reprolab] runs_root: settings={settings.runs_root!r} "
        f"env={env_runs_root!r} effective={effective_runs_root!r} "
        f"pid={_os.getpid()} cwd={_os.getcwd()!r}",
        flush=True,
    )
    try:
        marker_root = (
            effective_runs_root if effective_runs_root else _Path("logs") / "_no_runs_root"
        )
        marker_root.mkdir(parents=True, exist_ok=True)
        (marker_root / f"_create_app_pid{_os.getpid()}.txt").write_text(
            f"settings.runs_root={settings.runs_root!r}\n"
            f"env REPROLAB_RUNS_ROOT={env_runs_root!r}\n"
            f"effective={effective_runs_root!r}\n"
            f"cwd={_os.getcwd()}\n"
            f"argv={_sys.argv}\n",
            encoding="utf-8",
        )
    except Exception:
        pass

    # Honor REPROLAB_RUNS_ROOT so dev.ps1 / dev.sh actually colocate pipeline
    # workspaces with the launch's server logs. When unset, FileLiveRunService
    # falls back to <repo>/runs as before.
    service = run_service or FileLiveRunService(runs_root=effective_runs_root)

    app = FastAPI(
        title="ReproLab Agent",
        version=__version__,
        debug=settings.debug,
    )

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "version": __version__}

    # ------------------------------------------------------------------ #
    # Live-run API + SSE event stream (origin/main)
    # ------------------------------------------------------------------ #

    @app.post("/runs", status_code=202)
    async def start_run(request: StartRunRequest, x_demo_secret: str | None = Header(default=None)):
        _enforce_demo_gate(x_demo_secret, settings.demo_secret)
        return await service.start_run(request)

    @app.post("/runs/{project_id}/resume", status_code=202)
    async def resume_run(
        project_id: str,
        request: ResumeRunRequest | None = None,
        x_demo_secret: str | None = Header(default=None),
    ):
        """Re-spawn the orchestrator subprocess for an existing project.

        The orchestrator's resume-from-checkpoint logic picks up at the
        last completed stage. ``request_overrides`` (optional body) lets
        callers bump e.g. executionMode=max to push past a wall-clock
        failure without losing the work already done.
        """
        _enforce_demo_gate(x_demo_secret, settings.demo_secret)
        overrides: dict[str, Any] | None = None
        if request is not None:
            overrides = request.model_dump(exclude_none=True) or None
        state = await service.resume_run(project_id, request_overrides=overrides)
        if state is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return state

    @app.post("/runs/arxiv", status_code=202)
    async def start_arxiv_run(request: StartArxivRunRequest, x_demo_secret: str | None = Header(default=None)):
        """Fetch a paper from a URL (arXiv/openreview/etc) and start a run.

        Server-side fetch sidesteps browser CORS and the multipart upload
        gymnastics that the file path requires. The bytes are handed to the
        same ``start_uploaded_run`` service as a real upload — no second code
        path to keep in sync.
        """
        _enforce_demo_gate(x_demo_secret, settings.demo_secret)
        normalized_url = (request.url or "").strip()
        if not normalized_url:
            raise HTTPException(status_code=400, detail="An arXiv (or other paper) URL is required.")
        if not re.match(r"^https?://", normalized_url, re.IGNORECASE):
            raise HTTPException(status_code=400, detail="URL must start with http:// or https://.")
        # arxiv.org/abs/1234.5678 → arxiv.org/pdf/1234.5678 so the response is
        # the PDF rather than the HTML abstract page.
        fetch_url = re.sub(r"^(https?://arxiv\.org)/abs/", r"\1/pdf/", normalized_url, flags=re.IGNORECASE)
        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=httpx.Timeout(30.0, connect=10.0),
                headers={"user-agent": "ReproLab/0.1 (+https://github.com/anthropics/openresearch)"},
            ) as client:
                response = await client.get(fetch_url)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Could not fetch paper from {fetch_url!r}: {exc}") from exc
        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"Upstream returned HTTP {response.status_code} for {fetch_url!r}.",
            )
        content = response.content
        if not content:
            raise HTTPException(status_code=502, detail="Upstream returned an empty body.")
        if len(content) > 50 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Fetched paper exceeds 50 MB limit.")
        # Validate it's actually a PDF — content-type or magic bytes. arxiv
        # serves application/pdf; we also accept the %PDF- header as a
        # secondary check because some mirrors advertise octet-stream.
        looks_like_pdf = content[:5] == b"%PDF-" or "pdf" in (response.headers.get("content-type") or "").lower()
        if not looks_like_pdf:
            raise HTTPException(
                status_code=415,
                detail="Fetched content does not appear to be a PDF (no %PDF- header and content-type is not pdf-ish).",
            )
        # Derive a stable, safe filename from the URL's last path segment.
        # `2512.24601` → `arxiv_2512_24601.pdf` so the manifest renders nicely.
        last_segment = re.split(r"[/?#]", normalized_url.rstrip("/"))[-1] or "paper"
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", last_segment).strip("._-") or "paper"
        if "arxiv.org" in normalized_url.lower() and not safe_stem.lower().startswith("arxiv"):
            safe_stem = f"arxiv_{safe_stem}"
        if not safe_stem.lower().endswith(".pdf"):
            safe_stem = f"{safe_stem}.pdf"
        run_request = StartRunRequest(
            mode=request.mode or "rlm",
            provider=request.provider or "anthropic",
            verificationProvider=request.verificationProvider,
            executionMode=request.executionMode or "efficient",
            sandbox=request.sandbox or settings.default_sandbox,
            gpuMode=request.gpuMode or "auto",
            model=request.model or "sonnet",
        )
        return await service.start_uploaded_run(
            run_request,
            file_name=safe_stem,
            content=content,
        )

    @app.post("/runs/upload", status_code=202)
    async def start_uploaded_run(request: Request, x_demo_secret: str | None = Header(default=None)):
        _enforce_demo_gate(x_demo_secret, settings.demo_secret)
        form = await request.form()
        paper = form.get("paper")
        if paper is None or not hasattr(paper, "read"):
            raise HTTPException(status_code=400, detail="Upload a PDF before starting a lab run.")
        # Normalize the reported filename to a bare basename. A client may
        # send a path-qualified name — Windows browsers/tools can include
        # `C:\\...\\file.pdf` or backslash separators — and downstream code
        # (_stage_upload) treats it as a POSIX path. Strip both separators
        # so staging is platform-agnostic regardless of the upload source.
        raw_name = str(getattr(paper, "filename", "") or "paper.pdf")
        file_name = raw_name.replace("\\", "/").rsplit("/", 1)[-1].strip() or "paper.pdf"
        if not file_name.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")
        content = await paper.read()
        if not content:
            raise HTTPException(status_code=400, detail="Upload a PDF before starting a lab run.")
        run_request = StartRunRequest(
            mode=_form_value(form, "mode", "rlm"),
            provider=_form_value(form, "provider", "anthropic"),
            verificationProvider=_optional_form_value(form, "verificationProvider"),
            executionMode=_form_value(form, "executionMode", "efficient"),
            sandbox=_form_value(form, "sandbox", settings.default_sandbox),
            gpuMode=_form_value(form, "gpuMode", "auto"),
            model=_form_value(form, "model", "sonnet"),
        )
        return await service.start_uploaded_run(
            run_request,
            file_name=file_name,
            content=content,
        )

    @app.get("/runs")
    async def list_runs(
        limit: int = 10,
        status: str | None = None,
        q: str | None = None,
        order_by: str = "updated_at",
    ) -> list[dict]:
        return await service.list_runs(
            limit=limit,
            status=status,
            q=q,
            order_by=order_by,
        )

    @app.get("/runs/latest")
    async def latest_run(
        mode: str | None = None,
        provider: str | None = None,
        executionMode: str | None = None,
        sandbox: str | None = None,
        verificationProvider: str | None = None,
        gpuMode: str | None = None,
    ):
        state = await service.latest_run(
            mode=mode,
            provider=provider,
            execution_mode=executionMode,
            sandbox=sandbox,
            verification_provider=verificationProvider,
            gpu_mode=gpuMode,
        )
        if state is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return state

    @app.get("/runs/{project_id}")
    async def get_run(project_id: str):
        state = await service.get_run(project_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return state

    @app.get("/runs/{project_id}/source-pdf")
    async def get_source_pdf(project_id: str):
        getter = getattr(service, "get_source_pdf_path", None)
        if not callable(getter):
            raise HTTPException(status_code=404, detail="Source PDF not found")
        path = await getter(project_id)
        if path is None:
            raise HTTPException(status_code=404, detail="Source PDF not found")
        return FileResponse(
            path,
            media_type="application/pdf",
            filename="paper.pdf",
            content_disposition_type="inline",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/runs/{project_id}/final-report")
    async def get_final_report(project_id: str):
        getter = getattr(service, "get_final_report_path", None)
        if not callable(getter):
            raise HTTPException(status_code=404, detail="Final report not found")
        path = await getter(project_id)
        if path is None:
            raise HTTPException(status_code=404, detail="Final report not found")
        return FileResponse(
            path,
            media_type="text/markdown; charset=utf-8",
            filename="final_benchmark_report.md",
            content_disposition_type="inline",
            headers={"Cache-Control": "no-store"},
        )

    @app.delete("/runs/{project_id}")
    async def stop_run(project_id: str):
        state = await service.stop_run(project_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return state

    @app.get("/runs/{project_id}/events")
    async def stream_run_events(project_id: str):
        return StreamingResponse(
            service.stream_events(project_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ------------------------------------------------------------------ #
    # Pipeline topology (canonical graph metadata for the frontend)
    # ------------------------------------------------------------------ #

    @app.get("/pipeline/topology", response_model=PipelineTopology)
    async def pipeline_topology() -> PipelineTopology:
        return default_topology()

    # ------------------------------------------------------------------ #
    # Models (LLM choices surfaced in the upload-view dropdown)
    # ------------------------------------------------------------------ #

    @app.get("/models")
    async def list_models() -> list[dict[str, str]]:
        return [
            {"id": "sonnet", "label": "Sonnet", "provider": "anthropic"},
            {"id": "opus", "label": "Opus", "provider": "anthropic"},
        ]

    # ------------------------------------------------------------------ #
    # Phase 2 workspace services (HEAD)
    # ------------------------------------------------------------------ #

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
            approval_service = ApprovalService(db)
            evaluation = approval_service.evaluate(
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
            approval = approval_service.request_if_needed(
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


# --------------------------------------------------------------------------- #
# Phase 2 request models (HEAD)
# --------------------------------------------------------------------------- #

class ResumeRunRequest(BaseModel):
    """Body for ``POST /runs/{project_id}/resume``.

    All fields optional — when None, the resumed run inherits the
    original run's config (read from demo_status.json). Set fields
    override that inheritance per-key.
    """

    mode: str | None = None
    provider: str | None = None
    verificationProvider: str | None = None
    executionMode: str | None = None
    sandbox: str | None = None
    gpuMode: str | None = None
    model: str | None = None


class StartArxivRunRequest(BaseModel):
    """Body for ``POST /runs/arxiv``.

    All run-config fields are optional so the client only has to send the URL.
    Defaults are resolved server-side to mirror what the multipart upload path
    provides (mode=rlm, provider=anthropic, sandbox=<settings default>, …).
    """

    url: str = ""
    mode: str | None = None
    provider: str | None = None
    verificationProvider: str | None = None
    executionMode: str | None = None
    sandbox: str | None = None
    gpuMode: str | None = None
    model: str | None = None


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


# --------------------------------------------------------------------------- #
# Form helpers (origin/main, used by /runs/upload)
# --------------------------------------------------------------------------- #

def _form_value(form: Any, key: str, default: str) -> str:
    value = form.get(key)
    return str(value) if value not in (None, "") else default


def _optional_form_value(form: Any, key: str) -> str | None:
    value = form.get(key)
    if value in (None, "", "same"):
        return None
    return str(value)
