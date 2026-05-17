from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

from starlette.testclient import TestClient

from backend.app import create_app
from backend.services.events.live_runs import (
    LiveRunState,
    StartRunRequest,
    sse_event,
)


class FakeRunService:
    def __init__(
        self,
        source_pdf_path: Path | None = None,
        final_report_path: Path | None = None,
    ) -> None:
        self.started: StartRunRequest | None = None
        self.stopped_project_id: str | None = None
        self.source_pdf_path = source_pdf_path
        self.final_report_path = final_report_path
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

    async def get_source_pdf_path(self, project_id: str) -> Path | None:
        if project_id != self.state.projectId:
            return None
        return self.source_pdf_path

    async def get_final_report_path(self, project_id: str) -> Path | None:
        if project_id != self.state.projectId:
            return None
        return self.final_report_path

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


def test_fastapi_upload_route_normalizes_path_qualified_filenames() -> None:
    """A path-qualified filename (Windows `C:\\...` or `\\`-separated) is
    reduced to its basename so downstream POSIX path handling is safe."""
    for raw, expected in [
        ("C:\\Users\\14698\\Downloads\\2402.02868v3.pdf", "2402.02868v3.pdf"),
        ("/home/abheekp/Downloads/paper.pdf", "paper.pdf"),
        ("subdir\\nested\\report.PDF", "report.PDF"),
        ("plain.pdf", "plain.pdf"),
    ]:
        service = FakeRunService()
        client = TestClient(create_app(run_service=service))
        response = client.post(
            "/runs/upload",
            data={"mode": "sdk", "provider": "anthropic"},
            files={"paper": (raw, b"%PDF-demo", "application/pdf")},
        )
        assert response.status_code == 202, f"{raw!r} -> {response.status_code}"
        assert response.json()["sourceLabel"] == expected, raw


def test_fastapi_serves_stored_source_pdf(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-demo\n")
    service = FakeRunService(source_pdf_path=pdf)
    client = TestClient(create_app(run_service=service))

    response = client.get("/runs/prj_api/source-pdf")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/pdf")
    assert response.content.startswith(b"%PDF-demo")


def test_fastapi_serves_final_report_markdown(tmp_path: Path) -> None:
    report = tmp_path / "final_benchmark_report.md"
    report.write_text("# Final Benchmark Report\n\nready\n", encoding="utf-8")
    service = FakeRunService(final_report_path=report)
    client = TestClient(create_app(run_service=service))

    response = client.get("/runs/prj_api/final-report")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert "Final Benchmark Report" in response.text


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


# ---------------------------------------------------------------------------
# Task 1 — model choice field
# ---------------------------------------------------------------------------

def test_start_run_request_defaults_model_to_sonnet() -> None:
    from backend.services.events.live_runs import StartRunRequest

    req = StartRunRequest()
    assert req.model == "sonnet"


def test_start_run_request_accepts_opus() -> None:
    from backend.services.events.live_runs import StartRunRequest

    req = StartRunRequest(model="opus")
    assert req.model == "opus"


def test_python_script_contains_model_id_and_config_key() -> None:
    from pathlib import Path

    from backend.services.events.live_runs import StartRunRequest, _python_script

    req = StartRunRequest(mode="sdk", model="opus")
    script = _python_script(req, project_id="p", runs_root=Path("/tmp/r"), uploaded_paper=None)
    assert "claude-opus-4-7" in script
    assert 'model=config["model"]' in script


# ---------------------------------------------------------------------------
# Task B — _read_log reads stdout + stderr
# ---------------------------------------------------------------------------

def test_read_log_combines_stdout_and_stderr(tmp_path: Path) -> None:
    from backend.services.events.live_runs import FileLiveRunService

    project_id = "prj_test_logs"
    run_dir = tmp_path / project_id
    run_dir.mkdir(parents=True)

    (run_dir / "runner.stdout.log").write_text("agent output line\n", encoding="utf-8")
    (run_dir / "runner.stderr.log").write_text("runner error line\n", encoding="utf-8")

    service = FileLiveRunService(runs_root=tmp_path)
    result = service._read_log(project_id)

    assert "agent output line" in result
    assert "runner error line" in result


def test_read_log_returns_empty_when_neither_file_exists(tmp_path: Path) -> None:
    from backend.services.events.live_runs import FileLiveRunService

    service = FileLiveRunService(runs_root=tmp_path)
    assert service._read_log("prj_no_logs") == ""


def test_read_log_returns_stdout_only_when_stderr_missing(tmp_path: Path) -> None:
    from backend.services.events.live_runs import FileLiveRunService

    project_id = "prj_stdout_only"
    run_dir = tmp_path / project_id
    run_dir.mkdir(parents=True)
    (run_dir / "runner.stdout.log").write_text("only stdout\n", encoding="utf-8")

    service = FileLiveRunService(runs_root=tmp_path)
    result = service._read_log(project_id)

    assert "only stdout" in result
    assert "stderr" not in result


def test_read_log_returns_stderr_only_when_stdout_missing(tmp_path: Path) -> None:
    from backend.services.events.live_runs import FileLiveRunService

    project_id = "prj_stderr_only"
    run_dir = tmp_path / project_id
    run_dir.mkdir(parents=True)
    (run_dir / "runner.stderr.log").write_text("only stderr\n", encoding="utf-8")

    service = FileLiveRunService(runs_root=tmp_path)
    result = service._read_log(project_id)

    assert "only stderr" in result


def test_read_log_tail_cap_applied_to_combined(tmp_path: Path) -> None:
    from backend.services.events.live_runs import FileLiveRunService

    project_id = "prj_cap_test"
    run_dir = tmp_path / project_id
    run_dir.mkdir(parents=True)

    (run_dir / "runner.stdout.log").write_text("A" * 200, encoding="utf-8")
    (run_dir / "runner.stderr.log").write_text("B" * 200, encoding="utf-8")

    service = FileLiveRunService(runs_root=tmp_path)
    result = service._read_log(project_id, max_chars=100)

    # Each stream is tail-capped to its own half of the budget, so a
    # noisy stderr cannot evict stdout entirely.
    assert "A" in result and "B" in result
    assert result.count("A") == 50 and result.count("B") == 50
    assert len(result) <= 100 + len("\n--- runner.stderr.log ---\n")


# ---------------------------------------------------------------------------
# /runs/arxiv — server-side paper fetch + reuse of the upload pipeline.
# Previously the UI's arxiv input was wired to the demo-fixture handler and
# every URL paste silently became a CartPole-PPO fixture run; the new
# endpoint fetches the URL with httpx and routes the bytes through the
# existing start_uploaded_run path so the rest of the pipeline is unchanged.
# ---------------------------------------------------------------------------

import pytest


class _FakeHttpxResponse:
    def __init__(self, status_code: int, content: bytes, content_type: str = "application/pdf") -> None:
        self.status_code = status_code
        self.content = content
        self.headers = {"content-type": content_type}


class _FakeHttpxClient:
    def __init__(self, response: _FakeHttpxResponse, *, raise_on_get: Exception | None = None) -> None:
        self._response = response
        self._raise = raise_on_get
        self.requested_urls: list[str] = []

    async def __aenter__(self) -> "_FakeHttpxClient":
        return self

    async def __aexit__(self, *_exc) -> None:
        return None

    async def get(self, url: str) -> _FakeHttpxResponse:
        self.requested_urls.append(url)
        if self._raise is not None:
            raise self._raise
        return self._response


def _patch_httpx(monkeypatch, response, *, raise_on_get=None):
    fake = _FakeHttpxClient(response, raise_on_get=raise_on_get)
    def _factory(*_a, **_kw):
        return fake
    import backend.app as app_module
    monkeypatch.setattr(app_module.httpx, "AsyncClient", _factory)
    return fake


def test_runs_arxiv_fetches_and_starts_upload(monkeypatch) -> None:
    fake_response = _FakeHttpxResponse(200, b"%PDF-1.7 fake bytes")
    fake_client = _patch_httpx(monkeypatch, fake_response)
    service = FakeRunService()
    client = TestClient(create_app(run_service=service))

    response = client.post(
        "/runs/arxiv",
        json={"url": "https://arxiv.org/abs/2512.24601", "mode": "sdk", "provider": "anthropic"},
    )

    assert response.status_code == 202, response.text
    body = response.json()
    assert body["sourceKind"] == "uploaded_pdf"
    # arxiv.org/abs/N gets rewritten to /pdf/N so we receive the PDF, not the
    # abstract page. The derived filename includes an "arxiv_" prefix.
    assert "/pdf/" in fake_client.requested_urls[0]
    assert body["sourceLabel"].startswith("arxiv_")
    assert body["sourceLabel"].endswith(".pdf")


def test_runs_arxiv_rejects_missing_url() -> None:
    service = FakeRunService()
    client = TestClient(create_app(run_service=service))
    response = client.post("/runs/arxiv", json={"url": ""})
    assert response.status_code == 400


def test_runs_arxiv_rejects_non_http_scheme() -> None:
    service = FakeRunService()
    client = TestClient(create_app(run_service=service))
    response = client.post("/runs/arxiv", json={"url": "ftp://example.com/paper.pdf"})
    assert response.status_code == 400


def test_runs_arxiv_rejects_non_pdf_content(monkeypatch) -> None:
    _patch_httpx(monkeypatch, _FakeHttpxResponse(200, b"<html>not a pdf</html>", "text/html"))
    service = FakeRunService()
    client = TestClient(create_app(run_service=service))
    response = client.post("/runs/arxiv", json={"url": "https://example.com/page.html"})
    assert response.status_code == 415


def test_runs_arxiv_passes_through_upstream_failure(monkeypatch) -> None:
    _patch_httpx(monkeypatch, _FakeHttpxResponse(404, b""))
    service = FakeRunService()
    client = TestClient(create_app(run_service=service))
    response = client.post("/runs/arxiv", json={"url": "https://arxiv.org/abs/nope"})
    assert response.status_code == 502


def test_runs_arxiv_accepts_pdf_with_octet_stream_content_type(monkeypatch) -> None:
    # Some mirrors serve PDFs with application/octet-stream; the magic-byte
    # check (%PDF- header) is what makes the validation tolerant.
    _patch_httpx(
        monkeypatch,
        _FakeHttpxResponse(200, b"%PDF-1.4 bytes", content_type="application/octet-stream"),
    )
    service = FakeRunService()
    client = TestClient(create_app(run_service=service))
    response = client.post("/runs/arxiv", json={"url": "https://example.com/paper"})
    assert response.status_code == 202
