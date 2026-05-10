from __future__ import annotations

import json
from collections.abc import AsyncIterator

from starlette.testclient import TestClient

from backend.app import create_app
from backend.services.events.live_runs import (
    LiveRunState,
    StartRunRequest,
    sse_event,
)


class FakeRunService:
    def __init__(self) -> None:
        self.started: StartRunRequest | None = None
        self.stopped_project_id: str | None = None
        self.state = LiveRunState(
            projectId="prj_api",
            outputDir="runs/prj_api",
            runMode="sdk",
            llmProvider="anthropic",
            status="queued",
            payload=None,
            log="",
        )

    async def start_run(self, request: StartRunRequest) -> LiveRunState:
        self.started = request
        return self.state

    async def start_uploaded_run(
        self,
        request: StartRunRequest,
        *,
        file_name: str,
        content: bytes,
    ) -> LiveRunState:
        self.started = request
        self.state.sourceKind = "uploaded_pdf"
        self.state.sourceLabel = file_name
        return self.state

    async def get_run(self, project_id: str) -> LiveRunState | None:
        if project_id != self.state.projectId:
            return None
        return self.state

    async def latest_run(
        self,
        *,
        mode: str | None = None,
        provider: str | None = None,
        execution_mode: str | None = None,
        sandbox: str | None = None,
        verification_provider: str | None = None,
        gpu_mode: str | None = None,
    ) -> LiveRunState | None:
        return self.state

    async def stop_run(self, project_id: str) -> LiveRunState | None:
        self.stopped_project_id = project_id
        self.state.status = "stopped"
        return self.state

    async def stream_events(self, project_id: str) -> AsyncIterator[str]:
        yield sse_event("run_state", self.state.model_dump(mode="json"))
        yield sse_event("agent_log", {"projectId": project_id, "text": "hello"})


def test_fastapi_can_start_and_fetch_runs_through_backend_api() -> None:
    service = FakeRunService()
    client = TestClient(create_app(run_service=service))

    response = client.post(
        "/runs",
        json={
            "mode": "sdk",
            "provider": "anthropic",
            "executionMode": "efficient",
            "sandbox": "docker",
            "gpuMode": "prefer",
        },
    )

    assert response.status_code == 202
    assert response.json()["projectId"] == "prj_api"
    assert service.started is not None
    assert service.started.gpuMode == "prefer"

    fetched = client.get("/runs/prj_api")
    assert fetched.status_code == 200
    assert fetched.json()["status"] == "queued"


def test_fastapi_upload_route_starts_uploaded_pdf_run() -> None:
    service = FakeRunService()
    client = TestClient(create_app(run_service=service))

    response = client.post(
        "/runs/upload",
        data={"mode": "sdk", "provider": "anthropic"},
        files={"paper": ("paper.pdf", b"%PDF-demo", "application/pdf")},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["sourceKind"] == "uploaded_pdf"
    assert body["sourceLabel"] == "paper.pdf"


def test_fastapi_can_stop_run_and_stream_sse() -> None:
    service = FakeRunService()
    client = TestClient(create_app(run_service=service))

    stopped = client.delete("/runs/prj_api")
    assert stopped.status_code == 200
    assert stopped.json()["status"] == "stopped"
    assert service.stopped_project_id == "prj_api"

    with client.stream("GET", "/runs/prj_api/events") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        text = response.read().decode()

    assert "event: run_state" in text
    assert json.dumps("prj_api") in text

