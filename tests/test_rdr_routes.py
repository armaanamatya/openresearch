"""Tests for rdr-specific REST endpoints added to backend/app.py.

Tests:
- GET /runs/<id>/clusters  — 200 with valid project_id; 404 missing; 200+empty partial
- GET /runs/<id>/repair-iterations — same variants
- GET /runs/<id>/leaf-scores — 200 with report; 404 missing
- Corpus-redaction: paper_full_text key stripped from cluster response
- POST /runs with mode='rdr' accepted by StartRunRequest schema
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from backend.app import create_app
from backend.services.events.live_runs import (
    LiveRunState,
    StartRunRequest,
    sse_event,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class FakeRunService:
    """Minimal stub that satisfies the interface used by create_app."""

    def __init__(self) -> None:
        self.state = LiveRunState(
            projectId="prj_rdr_test",
            outputDir="runs/prj_rdr_test",
            runMode="rdr",
            status="running",
            payload=None,
            log="",
        )

    async def start_run(self, request: StartRunRequest) -> LiveRunState:
        return self.state

    async def start_uploaded_run(self, request, *, file_name, content) -> LiveRunState:
        return self.state

    async def get_run(self, project_id: str) -> LiveRunState | None:
        if project_id == self.state.projectId:
            return self.state
        return None

    async def get_source_pdf_path(self, project_id: str) -> Path | None:
        return None

    async def get_final_report_path(self, project_id: str) -> Path | None:
        return None

    async def latest_run(self, **kwargs) -> LiveRunState | None:
        return self.state

    async def stop_run(self, project_id: str) -> LiveRunState | None:
        if project_id == self.state.projectId:
            self.state.status = "stopped"
            return self.state
        return None

    async def stream_events(self, project_id: str):
        yield sse_event("run_state", self.state.model_dump(mode="json"))

    async def list_runs(self, **kwargs):
        return []

    async def resume_run(self, project_id, **kwargs):
        return self.state


@pytest.fixture()
def rdr_run_dir(tmp_path: Path) -> Path:
    """Write a minimal rdr run directory with cluster + repair checkpoints."""
    run_dir = tmp_path / "pb_seqnn_12345"
    iterations_dir = run_dir / "iterations"
    iterations_dir.mkdir(parents=True)

    # Cluster checkpoints
    cluster_a = "aaaabbbb-cccc-dddd-eeee-ffffaaaabbbb"
    cluster_b = "11112222-3333-4444-5555-666677778888"
    (iterations_dir / f"cluster_0_{cluster_a}.json").write_text(
        json.dumps({
            "cluster_id": cluster_a,
            "cluster_title": "Implement the model",
            "leaf_ids": ["leaf-001", "leaf-002"],
            "failed": False,
            "file_count": 4,
            "paper_full_text": "SHOULD BE REDACTED",
        }),
        encoding="utf-8",
    )
    (iterations_dir / f"cluster_1_{cluster_b}.json").write_text(
        json.dumps({
            "cluster_id": cluster_b,
            "cluster_title": "Run experiments",
            "leaf_ids": ["leaf-003"],
            "failed": True,
            "file_count": 0,
        }),
        encoding="utf-8",
    )

    # Repair checkpoint for cluster_b, pass 1
    (iterations_dir / f"repair_1_cluster_{cluster_b}.json").write_text(
        json.dumps({
            "cluster_id": cluster_b,
            "cluster_title": "Run experiments",
            "leaf_ids": ["leaf-003"],
            "failed": False,
            "file_count": 3,
            "repair_pass": 1,
        }),
        encoding="utf-8",
    )

    # final_report.json with leaf scores
    report = {
        "rubric": {
            "overall_score": 0.456,
            "leaf_scores": [
                {"id": "leaf-001", "score": 0.8, "justification": "Good implementation."},
                {"id": "leaf-002", "score": 0.7, "justification": "A" * 2000},  # truncation test
                {"id": "leaf-003", "score": 0.1, "justification": "Missing results."},
            ],
        }
    }
    (run_dir / "final_report.json").write_text(json.dumps(report), encoding="utf-8")

    return tmp_path


@pytest.fixture()
def client_with_run_dir(rdr_run_dir: Path, monkeypatch) -> TestClient:
    """TestClient wired to serve from tmp run dir via _runs_root monkeypatching."""
    monkeypatch.setattr("backend.app._runs_root", lambda: rdr_run_dir)
    service = FakeRunService()
    app = create_app(run_service=service)
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /runs/<id>/clusters
# ---------------------------------------------------------------------------

def test_clusters_200_with_valid_id(client_with_run_dir: TestClient) -> None:
    resp = client_with_run_dir.get("/runs/pb_seqnn_12345/clusters")
    assert resp.status_code == 200
    body = resp.json()
    assert body["project_id"] == "pb_seqnn_12345"
    clusters = body["clusters"]
    assert len(clusters) == 2
    # First cluster (index 0)
    c0 = clusters[0]
    assert c0["index"] == 0
    assert c0["title"] == "Implement the model"
    assert c0["leaf_ids"] == ["leaf-001", "leaf-002"]
    assert c0["failed"] is False
    assert c0["file_count"] == 4
    assert c0["repair_history"] == []


def test_clusters_404_with_missing_id(client_with_run_dir: TestClient) -> None:
    resp = client_with_run_dir.get("/runs/nonexistent_project/clusters")
    assert resp.status_code == 404


def test_clusters_200_empty_when_no_iterations_dir(tmp_path: Path, monkeypatch) -> None:
    """A run dir without an iterations/ subdir returns 200 + empty clusters."""
    run_dir = tmp_path / "pb_empty_run"
    run_dir.mkdir()
    monkeypatch.setattr("backend.app._runs_root", lambda: tmp_path)
    service = FakeRunService()
    app = create_app(run_service=service)
    client = TestClient(app)
    resp = client.get("/runs/pb_empty_run/clusters")
    assert resp.status_code == 200
    assert resp.json()["clusters"] == []


def test_clusters_corpus_redaction(client_with_run_dir: TestClient) -> None:
    """paper_full_text key must not appear in any cluster record."""
    resp = client_with_run_dir.get("/runs/pb_seqnn_12345/clusters")
    assert resp.status_code == 200
    body_text = resp.text
    # The fixture embeds "SHOULD BE REDACTED" in paper_full_text
    assert "SHOULD BE REDACTED" not in body_text
    assert "paper_full_text" not in body_text


def test_clusters_repair_history_populated(client_with_run_dir: TestClient) -> None:
    """Cluster with a repair checkpoint has a populated repair_history."""
    resp = client_with_run_dir.get("/runs/pb_seqnn_12345/clusters")
    assert resp.status_code == 200
    clusters = resp.json()["clusters"]
    # cluster_b (index 1) has a repair
    c1 = clusters[1]
    assert len(c1["repair_history"]) == 1
    assert c1["repair_history"][0]["pass"] == 1
    assert c1["repair_history"][0]["failed"] is False
    assert c1["repair_history"][0]["file_count"] == 3


# ---------------------------------------------------------------------------
# GET /runs/<id>/repair-iterations
# ---------------------------------------------------------------------------

def test_repair_iterations_200(client_with_run_dir: TestClient) -> None:
    resp = client_with_run_dir.get("/runs/pb_seqnn_12345/repair-iterations")
    assert resp.status_code == 200
    body = resp.json()
    assert body["project_id"] == "pb_seqnn_12345"
    assert len(body["passes"]) == 1
    p = body["passes"][0]
    assert p["pass"] == 1
    assert p["cluster_count"] == 1
    assert p["failed_count"] == 0  # repair pass had failed=False


def test_repair_iterations_404(client_with_run_dir: TestClient) -> None:
    resp = client_with_run_dir.get("/runs/nosuchrun/repair-iterations")
    assert resp.status_code == 404


def test_repair_iterations_empty_no_iterations_dir(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "pb_no_iter"
    run_dir.mkdir()
    monkeypatch.setattr("backend.app._runs_root", lambda: tmp_path)
    service = FakeRunService()
    app = create_app(run_service=service)
    client = TestClient(app)
    resp = client.get("/runs/pb_no_iter/repair-iterations")
    assert resp.status_code == 200
    assert resp.json()["passes"] == []


# ---------------------------------------------------------------------------
# GET /runs/<id>/leaf-scores
# ---------------------------------------------------------------------------

def test_leaf_scores_200(client_with_run_dir: TestClient) -> None:
    resp = client_with_run_dir.get("/runs/pb_seqnn_12345/leaf-scores")
    assert resp.status_code == 200
    body = resp.json()
    assert body["project_id"] == "pb_seqnn_12345"
    assert abs(body["overall_score"] - 0.456) < 1e-6
    scores = body["leaf_scores"]
    assert len(scores) == 3
    assert scores[0]["id"] == "leaf-001"
    assert abs(scores[0]["score"] - 0.8) < 1e-6
    assert scores[0]["justification"] == "Good implementation."


def test_leaf_scores_justification_truncated(client_with_run_dir: TestClient) -> None:
    """Justification longer than 1000 chars is truncated."""
    resp = client_with_run_dir.get("/runs/pb_seqnn_12345/leaf-scores")
    body = resp.json()
    # leaf-002 has justification of 2000 'A's — should be capped
    leaf2 = next(s for s in body["leaf_scores"] if s["id"] == "leaf-002")
    assert len(leaf2["justification"]) <= 1001  # 1000 + possible ellipsis char


def test_leaf_scores_404_no_run(client_with_run_dir: TestClient) -> None:
    resp = client_with_run_dir.get("/runs/phantom/leaf-scores")
    assert resp.status_code == 404


def test_leaf_scores_404_no_report(tmp_path: Path, monkeypatch) -> None:
    """Run dir exists but no final_report.json → 404."""
    run_dir = tmp_path / "pb_no_report"
    run_dir.mkdir()
    monkeypatch.setattr("backend.app._runs_root", lambda: tmp_path)
    service = FakeRunService()
    app = create_app(run_service=service)
    client = TestClient(app)
    resp = client.get("/runs/pb_no_report/leaf-scores")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /runs — mode='rdr' accepted, paper_id threaded through
# ---------------------------------------------------------------------------

def test_start_run_request_accepts_rdr_mode() -> None:
    """StartRunRequest validates mode='rdr' without raising."""
    req = StartRunRequest(mode="rdr", paper_id="sequential-neural-score-estimation")
    assert req.mode == "rdr"
    assert req.paper_id == "sequential-neural-score-estimation"


def test_start_run_request_paper_id_none_by_default() -> None:
    # The mode literal narrowed from {offline,sdk,rlm,rdr} to {rlm,rdr} when
    # the offline/sdk paths were removed; the test's intent — that paper_id
    # defaults to None when omitted — holds for the surviving rlm value.
    req = StartRunRequest(mode="rlm")
    assert req.paper_id is None


def test_post_runs_rdr_mode_accepted(client_with_run_dir: TestClient) -> None:
    """POST /runs with mode='rdr' returns 202 (service stub accepts it)."""
    resp = client_with_run_dir.post(
        "/runs",
        json={"mode": "rdr", "paper_id": "my-bundle"},
    )
    assert resp.status_code == 202
