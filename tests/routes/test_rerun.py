"""Tests for POST /runs/{project_id}/rerun."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _reset_settings_cache():
    import backend.config as _config
    _config._settings_cache = None


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch, tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    monkeypatch.setenv("REPROLAB_RUNS_ROOT", str(runs_root))
    monkeypatch.delenv("REPROLAB_DEMO_SECRET", raising=False)
    _reset_settings_cache()
    yield runs_root
    _reset_settings_cache()


@pytest.fixture
def client(_isolate_settings):
    from backend.app import create_app
    return TestClient(create_app())


def _seed_run_with_pdf(runs_root: Path, project_id: str, pdf_content: bytes = b"%PDF-1.4 test") -> Path:
    """Seed a run directory with a demo_status.json pointing to a real PDF."""
    run_dir = runs_root / project_id
    run_dir.mkdir(parents=True)

    # Write the staged PDF (raw_paper.pdf is what runPath points to)
    pdf_path = run_dir / "raw_paper.pdf"
    pdf_path.write_bytes(pdf_content)

    status = {
        "projectId": project_id,
        "status": "failed",
        "runMode": "rlm",
        "llmProvider": "anthropic",
        "executionMode": "efficient",
        "sandboxMode": "local",
        "gpuMode": "auto",
        "model": "sonnet",
        "sourceKind": "uploaded_pdf",
        "sourcePdf": {
            "fileName": "mytest.pdf",
            "title": "My Test Paper",
            "sizeBytes": len(pdf_content),
            "sha256": "abc",
            "pageCount": 1,
            "runPath": str(pdf_path),
            "codePath": str(run_dir / "code" / "paper.pdf"),
        },
    }
    (run_dir / "demo_status.json").write_text(json.dumps(status))
    return pdf_path


# ---------------------------------------------------------------------------
# Happy path — PDF source exists → new run created
# ---------------------------------------------------------------------------

def test_rerun_happy_path_returns_new_project(client, _isolate_settings, monkeypatch):
    """Rerunning a failed PDF run creates a fresh project_id and returns its state."""
    # Patch _start_python_run to avoid actually spawning a subprocess.
    from backend.services.events import live_runs as _lr
    original_start = _lr.FileLiveRunService._start_python_run

    created_ids: list[str] = []

    async def _fake_start(self, request, *, project_id, uploaded_paper):
        created_ids.append(project_id)
        run_dir = self.runs_root / project_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "demo_status.json").write_text(json.dumps({
            "projectId": project_id,
            "status": "queued",
            "runMode": "rlm",
            "llmProvider": "anthropic",
            "executionMode": "efficient",
            "sandboxMode": "local",
            "gpuMode": "auto",
            "model": "sonnet",
            "sourceKind": "uploaded_pdf",
            "outputDir": str(run_dir),
        }))
        return (await self.get_run(project_id)) or _lr.LiveRunState(
            projectId=project_id,
            outputDir=str(run_dir),
            runMode="rlm",
            status="queued",
            payload=None,
            log="",
        )

    monkeypatch.setattr(_lr.FileLiveRunService, "_start_python_run", _fake_start)

    old_project_id = "prj_old_failed"
    _seed_run_with_pdf(_isolate_settings, old_project_id)

    r = client.post(f"/runs/{old_project_id}/rerun")
    assert r.status_code == 202, r.text
    body = r.json()
    assert "projectId" in body
    new_id = body["projectId"]
    # A fresh project id must be different from the old one.
    assert new_id != old_project_id
    assert len(created_ids) == 1
    assert created_ids[0] == new_id


# ---------------------------------------------------------------------------
# 404 — project does not exist
# ---------------------------------------------------------------------------

def test_rerun_unknown_project_returns_404(client):
    r = client.post("/runs/nonexistent_project_xyz/rerun")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# 422 — source PDF is gone from disk
# ---------------------------------------------------------------------------

def test_rerun_missing_pdf_returns_422(client, _isolate_settings):
    """When the PDF at sourcePdf.runPath is deleted, rerun returns 422."""
    run_dir = _isolate_settings / "prj_nopdf"
    run_dir.mkdir()
    gone_path = _isolate_settings / "gone.pdf"
    status = {
        "projectId": "prj_nopdf",
        "status": "failed",
        "runMode": "rlm",
        "sourceKind": "uploaded_pdf",
        "sourcePdf": {
            "fileName": "gone.pdf",
            "runPath": str(gone_path),
        },
    }
    (run_dir / "demo_status.json").write_text(json.dumps(status))

    # Ensure the file does NOT exist
    assert not gone_path.exists()

    r = client.post("/runs/prj_nopdf/rerun")
    assert r.status_code == 422
