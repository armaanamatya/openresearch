"""Tests for Issue #27: Verification Team."""

from pathlib import Path
import json
import pytest

from backend.agents.verification import (
    run_gate_offline,
    run_improvement_gate_offline,
    verify_method_fidelity,
    verify_environment,
    verify_data_metrics,
    verify_artifacts,
)
from backend.agents.schemas import (
    Ambiguity,
    BaselineResult,
    ExperimentArtifacts,
    GateStatus,
    MetricSpec,
    PaperClaimMap,
    PathResult,
    RiskLevel,
)


def _claim_map():
    return PaperClaimMap(
        core_contribution="PPO",
        metrics=[MetricSpec(name="mean_reward", definition="Mean over 100 eps")],
        ambiguities=[
            Ambiguity(assumption_id="A001", detail="test", risk=RiskLevel.high),
            Ambiguity(assumption_id="A002", detail="test", risk=RiskLevel.medium),
        ],
    )


def _baseline(tmp_path: Path):
    df_path = tmp_path / "Dockerfile"
    df_path.write_text("FROM python:3.11-slim\nRUN pip install torch==2.2.0")
    return BaselineResult(
        mode="implement_from_paper",
        code_path=str(tmp_path / "code"),
        dockerfile_path=str(df_path),
        commands_to_run=["python train.py"],
        assumptions_applied=["A001", "A002"],
    )


def _success_artifacts(tmp_path: Path):
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    (baseline_dir / "logs").mkdir()
    (baseline_dir / "logs" / "run.log").write_text("OK")
    (baseline_dir / "commands.log").write_text(
        json.dumps({
            "command": "python train.py",
            "phase": "experiment_runner",
            "status": "succeeded",
            "started_at": "2026-05-09T00:00:00+00:00",
            "finished_at": "2026-05-09T00:00:01+00:00",
            "duration_seconds": 1.0,
            "exit_code": 0,
        })
        + "\n"
    )
    (baseline_dir / "provenance.json").write_text(
        json.dumps({"success": True, "command_results": [{"exit_code": 0}]})
    )
    return ExperimentArtifacts(
        metrics={"mean_reward": 487.3},
        plots=[str(baseline_dir / "plot.png")],
        log_path=str(baseline_dir / "logs" / "run.log"),
        commands_log_path=str(baseline_dir / "commands.log"),
        provenance_path=str(baseline_dir / "provenance.json"),
        success=True,
    )


class TestMethodFidelityVerifier:
    def test_all_assumptions_applied(self, tmp_path: Path):
        score = verify_method_fidelity(_claim_map(), _baseline(tmp_path))
        assert score.score >= 0.8
        assert any("All" in f for f in score.findings)

    def test_missing_assumptions_flagged(self, tmp_path: Path):
        bl = BaselineResult(mode="adapt", assumptions_applied=["A001"])
        score = verify_method_fidelity(_claim_map(), bl)
        assert len(score.mismatches) >= 1

    def test_code_dir_checked(self, tmp_path: Path):
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        (code_dir / "train.py").write_text("# PPO")
        score = verify_method_fidelity(_claim_map(), _baseline(tmp_path), code_dir)
        assert any("train.py" in f for f in score.findings)


class TestEnvironmentVerifier:
    def test_valid_environment(self, tmp_path: Path):
        score = verify_environment(_baseline(tmp_path), _success_artifacts(tmp_path))
        assert score.score >= 0.8
        assert any("Dockerfile" in f for f in score.findings)

    def test_missing_dockerfile(self, tmp_path: Path):
        bl = BaselineResult(mode="adapt", dockerfile_path="/nonexistent")
        arts = ExperimentArtifacts(success=True)
        score = verify_environment(bl, arts)
        assert len(score.mismatches) >= 1

    def test_malformed_command_log_is_flagged(self, tmp_path: Path):
        arts = _success_artifacts(tmp_path)
        Path(arts.commands_log_path).write_text("python train.py\n")
        score = verify_environment(_baseline(tmp_path), arts)
        assert any("structured JSONL" in mismatch for mismatch in score.mismatches)

    def test_failed_command_log_is_high_severity(self, tmp_path: Path):
        arts = _success_artifacts(tmp_path)
        Path(arts.commands_log_path).write_text(
            json.dumps({
                "command": "python train.py",
                "status": "failed",
                "exit_code": 1,
            })
            + "\n"
        )
        score = verify_environment(_baseline(tmp_path), arts)
        assert score.severity == "high"


class TestDataMetricsVerifier:
    def test_success_with_metrics(self, tmp_path: Path):
        score = verify_data_metrics(_claim_map(), _success_artifacts(tmp_path))
        assert score.score >= 0.5
        assert any("Metrics present" in f for f in score.findings)

    def test_failure_flagged(self):
        arts = ExperimentArtifacts(success=False, error_message="Crashed")
        score = verify_data_metrics(_claim_map(), arts)
        assert len(score.mismatches) >= 1


class TestArtifactVerifier:
    def test_all_artifacts_present(self, tmp_path: Path):
        score = verify_artifacts(_success_artifacts(tmp_path))
        assert score.score >= 0.8

    def test_missing_artifacts(self):
        arts = ExperimentArtifacts(success=True, metrics={})
        score = verify_artifacts(arts)
        assert score.score < 0.5


class TestGateOffline:
    def test_gate_2_passes_on_success(self, tmp_path: Path):
        report = run_gate_offline(
            "gate_2", _claim_map(), _baseline(tmp_path), _success_artifacts(tmp_path),
        )
        assert report.status in (GateStatus.verified, GateStatus.verified_with_caveats)
        assert len(report.verifier_scores) == 4
        assert all(score.evidence_refs for score in report.verifier_scores)

    def test_gate_2_fails_on_failure(self, tmp_path: Path):
        arts = ExperimentArtifacts(success=False, error_message="Crashed")
        bl = BaselineResult(mode="adapt")
        report = run_gate_offline("gate_2", _claim_map(), bl, arts)
        assert report.status in (GateStatus.failed_reproduction, GateStatus.partial_reproduction)

    def test_gate_has_decision_log(self, tmp_path: Path):
        report = run_gate_offline(
            "gate_2", _claim_map(), _baseline(tmp_path), _success_artifacts(tmp_path),
        )
        assert report.decision_log_entry != ""
        assert "gate_2" in report.decision_log_entry


class TestImprovementGate:
    def test_passes_with_successful_paths(self):
        paths = [
            PathResult(path_id="p1", hypothesis="h1", success=True, metrics={"reward": 500}),
            PathResult(path_id="p2", hypothesis="h2", success=True, metrics={"reward": 490}),
        ]
        report = run_improvement_gate_offline(paths, _claim_map(), {"mean_reward": 487})
        assert report.status == GateStatus.verified

    def test_partial_with_mixed_results(self):
        paths = [
            PathResult(path_id="p1", hypothesis="h1", success=True, metrics={"reward": 500}),
            PathResult(path_id="p2", hypothesis="h2", success=False, failure_notes="failed"),
            PathResult(path_id="p3", hypothesis="h3", success=False, failure_notes="failed"),
        ]
        report = run_improvement_gate_offline(paths, _claim_map(), {"mean_reward": 487})
        assert report.status == GateStatus.partial_reproduction
