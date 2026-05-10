"""FastAPI application factory."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from backend import __version__
from backend.config import get_settings
from backend.services.events.live_runs import FileLiveRunService, StartRunRequest


def create_app(*, run_service: Any | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    service = run_service or FileLiveRunService()

    app = FastAPI(
        title="ReproLab Agent",
        version=__version__,
        debug=settings.debug,
    )

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.post("/runs", status_code=202)
    async def start_run(request: StartRunRequest):
        return await service.start_run(request)

    @app.post("/runs/upload", status_code=202)
    async def start_uploaded_run(request: Request):
        form = await request.form()
        paper = form.get("paper")
        if paper is None or not hasattr(paper, "read"):
            raise HTTPException(status_code=400, detail="Upload a PDF before starting a lab run.")
        file_name = str(getattr(paper, "filename", "") or "paper.pdf")
        if not file_name.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")
        content = await paper.read()
        if not content:
            raise HTTPException(status_code=400, detail="Upload a PDF before starting a lab run.")
        run_request = StartRunRequest(
            mode=_form_value(form, "mode", "offline"),
            provider=_form_value(form, "provider", "anthropic"),
            verificationProvider=_optional_form_value(form, "verificationProvider"),
            executionMode=_form_value(form, "executionMode", "efficient"),
            sandbox=_form_value(form, "sandbox", "auto"),
            gpuMode=_form_value(form, "gpuMode", "auto"),
        )
        return await service.start_uploaded_run(
            run_request,
            file_name=file_name,
            content=content,
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

    return app


def _form_value(form: Any, key: str, default: str) -> str:
    value = form.get(key)
    return str(value) if value not in (None, "") else default


def _optional_form_value(form: Any, key: str) -> str | None:
    value = form.get(key)
    if value in (None, "", "same"):
        return None
    return str(value)
