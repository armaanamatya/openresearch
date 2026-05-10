"""Phase 2 research-workspace services."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.persistence.database import Database


@pytest.fixture
def db(tmp_path: Path):
    database = Database(f"sqlite:///{tmp_path / 'phase2_workspace.db'}")
    database.initialize()
    yield database
    database.close()


def test_approval_policy_creates_and_resolves_checkpoint(db):
    from backend.services.approval import ApprovalService

    service = ApprovalService(db)
    evaluation = service.evaluate(
        action="dataset_download",
        dataset_size_gb=82,
        metadata={"dataset": "large-benchmark"},
    )

    assert evaluation.requires_approval is True
    request = service.request_if_needed(
        project_id="prj_approval",
        label="Approve dataset download",
        evaluation=evaluation,
    )
    assert request is not None
    assert request.state == "pending"

    resolved = service.resolve(
        request.approval_id,
        state="approved",
        resolved_by="tester",
        note="cache is pre-approved",
    )
    assert resolved.state == "approved"
    assert service.list_requests(project_id="prj_approval", state="pending") == ()


def test_dataset_cache_tracks_reusable_available_dataset(db, tmp_path: Path):
    from backend.services.datasets import DatasetCacheService

    cache = DatasetCacheService(db, cache_root=tmp_path / "datasets")
    planned = cache.plan(
        name="CIFAR-10",
        source_url="https://www.cs.toronto.edu/~kriz/cifar.html",
        version="official",
        size_bytes=170 * 1024 * 1024,
        source_project_id="prj_mixmatch",
    )

    assert planned.status == "planned"
    available = cache.mark_available(planned.dataset_id, local_path=tmp_path / "datasets/cifar10")
    assert available.status == "available"
    assert available.size_gb > 0

    reusable = cache.find_reusable(name="cifar-10", version="official")
    assert reusable is not None
    assert reusable.dataset_id == planned.dataset_id


def test_failure_diagnosis_classifies_and_promotes_to_memory(db):
    from backend.services.context.memory import CrossProjectMemoryService
    from backend.services.diagnostics import FailureDiagnosisService

    diagnostics = FailureDiagnosisService(db)
    event = diagnostics.diagnose(
        project_id="prj_fail",
        stage="environment",
        command="pip install old-package==0.1",
        stderr="ERROR: No matching distribution found for old-package==0.1",
        artifact_refs=("runs/prj_fail/baseline/logs/build.log",),
    )

    assert event.kind == "failed_dependency_resolution"
    assert event.retryable is True

    memory_id = diagnostics.promote_to_memory(CrossProjectMemoryService(db), event)
    hits = CrossProjectMemoryService(db).search("dependency resolution old package")
    assert hits
    assert hits[0].record.id == memory_id


def test_reproducibility_scoring_uses_dynamic_thresholds_and_assumption_risk():
    from backend.agents.schemas import Assumption, GateStatus, VerificationReport, VerifierScore
    from backend.services.scoring import ReproducibilityScoringService

    report = VerificationReport(
        gate="gate_2",
        status=GateStatus.verified_with_caveats,
        verifier_scores=[
            VerifierScore(
                verifier_name="method_fidelity",
                score=0.91,
                evidence_refs=["paper#method", "code#train", "diff.patch"],
                severity="medium",
            ),
            VerifierScore(
                verifier_name="environment_execution",
                score=0.86,
                evidence_refs=["Dockerfile", "commands.log"],
                severity="medium",
            ),
            VerifierScore(
                verifier_name="data_metrics",
                score=0.8,
                evidence_refs=["metrics.json"],
                severity="high",
                mismatches=["single seed only"],
            ),
            VerifierScore(
                verifier_name="artifact_diff",
                score=0.95,
                evidence_refs=["diff.patch", "plots/reward.png", "provenance.json"],
                severity="low",
            ),
        ],
    )
    score = ReproducibilityScoringService().score(
        report,
        assumptions=[
            Assumption(
                assumption_id="A001",
                detail="Adam epsilon inferred",
                chosen_value="1e-5",
                risk="high",
            )
        ],
    )

    assert score.assumption_risk == "high"
    assert score.composite <= 84
    assert any(item.verdict in {"verified", "caveated"} for item in score.dynamic_thresholds)
    assert score.blocking_issues == ("single seed only",)


def test_research_workspace_summary_combines_phase2_state(db, tmp_path: Path):
    from backend.services.approval import ApprovalService
    from backend.services.comparison import MultiPaperComparisonService, PaperRunSummary
    from backend.services.context.graph import KnowledgeGraphService
    from backend.services.context.memory import CrossProjectMemoryService
    from backend.services.datasets import DatasetCacheService
    from backend.services.diagnostics import FailureDiagnosisService
    from backend.services.research_workspace import ResearchWorkspaceService

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "train.py").write_text("def train():\n    return 1\n")
    KnowledgeGraphService(db).ingest_python_repo(project_id="prj_workspace", repo_root=repo)
    CrossProjectMemoryService(db).remember_environment_recipe(
        source_project_id="prj_workspace",
        title="CPU CartPole environment",
        summary="torch and gymnasium pins work",
        packages={"torch": "2.2.0"},
    )
    DatasetCacheService(db).plan(name="CartPole-v1", source_project_id="prj_workspace")
    approval_eval = ApprovalService(db).evaluate(action="long_run", runtime_minutes=90)
    ApprovalService(db).request_if_needed(
        project_id="prj_workspace",
        label="Approve long run",
        evaluation=approval_eval,
    )
    FailureDiagnosisService(db).diagnose(
        project_id="prj_workspace",
        stage="training",
        command="python train.py",
        stderr="CUDA out of memory",
    )
    MultiPaperComparisonService(db).compare(
        [
            PaperRunSummary(
                project_id="prj_workspace",
                paper_title="PPO",
                dataset="CartPole-v1",
                split="eval",
                metric_name="mean_reward",
                metric_value=475,
                status="verified",
            )
        ]
    )

    summary = ResearchWorkspaceService(db).summarize_project("prj_workspace")
    assert summary.graph.node_count >= 2
    assert summary.memory_records
    assert summary.datasets
    assert summary.pending_approvals
    assert summary.recent_failures[0].kind == "out_of_memory"
    assert summary.comparison_reports
    assert any("Resolve pending human approvals" in item for item in summary.recommendations)


def test_phase2_fastapi_endpoints(monkeypatch, tmp_path: Path):
    from starlette.testclient import TestClient

    monkeypatch.setenv("REPROLAB_DATABASE_URL", f"sqlite:///{tmp_path / 'api.db'}")
    from backend.config import get_settings

    get_settings(_force_reload=True)

    from backend.app import create_app

    client = TestClient(create_app())
    dataset = client.post(
        "/phase2/datasets/plan",
        json={
            "project_id": "prj_api",
            "name": "CIFAR-10",
            "version": "official",
            "size_bytes": 170 * 1024 * 1024,
        },
    )
    assert dataset.status_code == 200
    assert dataset.json()["status"] == "planned"

    approval = client.post(
        "/phase2/approvals/evaluate",
        json={
            "project_id": "prj_api",
            "action": "sandbox_network",
            "network_stage": "run",
        },
    )
    assert approval.status_code == 200
    assert approval.json()["approval"]["state"] == "pending"

    failure = client.post(
        "/phase2/failures/diagnose",
        json={
            "project_id": "prj_api",
            "stage": "training",
            "command": "python train.py",
            "timed_out": True,
        },
    )
    assert failure.status_code == 200
    assert failure.json()["kind"] == "timeout"

    summary = client.get("/phase2/projects/prj_api/summary")
    assert summary.status_code == 200
    body = summary.json()
    assert body["project_id"] == "prj_api"
    assert len(body["datasets"]) == 1
    assert len(body["pending_approvals"]) == 1
    assert len(body["recent_failures"]) == 1
