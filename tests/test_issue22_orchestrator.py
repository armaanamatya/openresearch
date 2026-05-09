"""Tests for Issue #22: Claude Agent SDK integration + root orchestrator.

Validates:
- Agent registry contains all 13 PRD agents with correct tool permissions
- Agent definitions can be converted to SDK format
- Structured output schemas validate correctly
- Pipeline state checkpoint/resume works
- Pipeline stage ordering is correct
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.agents.registry import AGENT_REGISTRY, AgentSpec, get_agent_definitions
from backend.agents.schemas import (
    AgentOutput,
    Ambiguity,
    Assumption,
    BaselineResult,
    EnvironmentSpec,
    ExperimentArtifacts,
    GateDecision,
    GateStatus,
    ImprovementHypothesis,
    MetricSpec,
    PaperClaimMap,
    PathResult,
    ReproductionContract,
    ResearchMap,
    RiskLevel,
    TrainingRecipe,
    VerificationReport,
    VerifierScore,
)
from backend.agents.orchestrator import PipelineStage, PipelineState, ReproLabOrchestrator


# -----------------------------------------------------------------------
# Registry tests
# -----------------------------------------------------------------------

EXPECTED_AGENTS = [
    "paper-understanding",
    "artifact-discovery",
    "environment-detective",
    "reproduction-planner",
    "baseline-implementation",
    "experiment-runner",
    "method-fidelity-verifier",
    "environment-verifier",
    "data-metrics-verifier",
    "artifact-diff-verifier",
    "supervisor-verifier",
    "improvement-orchestrator",
    "improvement-path",
]


def test_registry_has_all_13_agents():
    """PRD specifies exactly 13 agents."""
    assert len(AGENT_REGISTRY) == 13
    for name in EXPECTED_AGENTS:
        assert name in AGENT_REGISTRY, f"Missing agent: {name}"


def test_registry_entries_are_agent_specs():
    for name, spec in AGENT_REGISTRY.items():
        assert isinstance(spec, AgentSpec), f"{name} is not AgentSpec"
        assert spec.agent_id == name
        assert spec.role in ("builder", "verifier", "supervisor", "improvement")
        assert len(spec.description) > 10
        assert len(spec.prompt) > 50


def test_builder_agents_have_correct_tools():
    """Builder agents should have file/code tools."""
    assert "Read" in AGENT_REGISTRY["paper-understanding"].tools
    assert "Bash" in AGENT_REGISTRY["paper-understanding"].tools
    assert "Write" in AGENT_REGISTRY["baseline-implementation"].tools
    assert "Edit" in AGENT_REGISTRY["baseline-implementation"].tools


def test_verifier_agents_are_read_only():
    """Verifier agents should NOT have Write or Edit tools."""
    for name in ["method-fidelity-verifier", "environment-verifier",
                 "data-metrics-verifier", "artifact-diff-verifier"]:
        spec = AGENT_REGISTRY[name]
        assert "Write" not in spec.tools, f"{name} should not have Write"
        assert "Edit" not in spec.tools, f"{name} should not have Edit"


def test_supervisor_has_agent_tool():
    """Supervisor needs Agent tool to spawn verifier subagents."""
    assert "Agent" in AGENT_REGISTRY["supervisor-verifier"].tools
    assert AGENT_REGISTRY["supervisor-verifier"].spawn_permissions is True


def test_improvement_orchestrator_has_agent_tool():
    """Improvement orchestrator spawns path agents."""
    assert "Agent" in AGENT_REGISTRY["improvement-orchestrator"].tools
    assert AGENT_REGISTRY["improvement-orchestrator"].spawn_permissions is True


def test_get_agent_definitions_returns_sdk_format():
    """get_agent_definitions() should produce claude-agent-sdk AgentDefinition objects."""
    defs = get_agent_definitions()
    assert len(defs) == 13
    for name, defn in defs.items():
        assert hasattr(defn, "description")
        assert hasattr(defn, "prompt")
        assert len(defn.description) > 10
        assert len(defn.prompt) > 50


# -----------------------------------------------------------------------
# Schema tests
# -----------------------------------------------------------------------

def test_paper_claim_map_validates():
    pcm = PaperClaimMap(
        core_contribution="Proximal Policy Optimization",
        claims=[{"method": "PPO", "dataset": "CartPole-v1", "metric": "mean_reward", "expected_result": ">=475"}],
        datasets=[{"name": "CartPole-v1", "source": "Gymnasium", "download_method": "bundled"}],
        metrics=[MetricSpec(name="mean_reward", definition="Mean over 100 episodes", target_value="475")],
        model_architecture="2-layer MLP with 64 hidden units",
        training_recipe=TrainingRecipe(optimizer="Adam", learning_rate="3e-4"),
        evaluation_protocol="100 episodes, report mean reward",
        ambiguities=[
            Ambiguity(
                assumption_id="A001",
                detail="Adam epsilon not specified",
                chosen_value="1e-5",
                risk=RiskLevel.high,
            )
        ],
    )
    assert len(pcm.ambiguities) == 1
    assert pcm.ambiguities[0].assumption_id == "A001"
    # Round-trip JSON
    data = json.loads(pcm.model_dump_json())
    pcm2 = PaperClaimMap(**data)
    assert pcm2.core_contribution == pcm.core_contribution


def test_environment_spec_validates():
    spec = EnvironmentSpec(
        dockerfile="FROM python:3.11-slim\nRUN pip install torch==2.2.0",
        python_version="3.11",
        framework="pytorch",
        framework_version="2.2.0",
        pip_packages={"torch": "2.2.0", "gymnasium": "0.29.1"},
        assumptions=[
            Assumption(
                assumption_id="ENV001",
                detail="PyTorch version",
                chosen_value="2.2.0",
                risk=RiskLevel.low,
            )
        ],
    )
    assert spec.python_version == "3.11"
    assert len(spec.assumptions) == 1


def test_gate_decision_validates():
    gd = GateDecision(gate="gate_1", passed=True, status=GateStatus.verified)
    assert gd.passed is True
    gd2 = GateDecision(gate="gate_2", passed=False, status=GateStatus.failed_reproduction)
    assert gd2.passed is False


def test_verification_report_validates():
    vr = VerificationReport(
        gate="gate_2",
        status=GateStatus.verified_with_caveats,
        verifier_scores=[
            VerifierScore(verifier_name="method_fidelity", score=0.85, findings=["OK"]),
            VerifierScore(verifier_name="environment", score=0.9),
        ],
        reasoning="All verifiers agree.",
        decision_log_entry="Gate 2 passed with caveats.",
    )
    assert len(vr.verifier_scores) == 2
    assert vr.status == GateStatus.verified_with_caveats


def test_improvement_hypothesis_validates():
    h = ImprovementHypothesis(
        path_id="path_1",
        hypothesis="Reduce entropy coefficient",
        rationale="Premature convergence observed",
        expected_outcome="Mean reward +15",
        risk=RiskLevel.low,
    )
    assert h.path_id == "path_1"


def test_research_map_validates():
    rm = ResearchMap(
        baseline_summary="PPO reproduces with mean reward 485",
        promising_directions=["Lower entropy coefficient"],
        dead_ends=["Separate actor-critic networks"],
        next_experiments=["GAE lambda sweep"],
    )
    assert len(rm.promising_directions) == 1


def test_agent_output_envelope():
    out = AgentOutput(
        agent_id="paper-understanding",
        status="completed",
        structured_outputs={"claim_map": {"core_contribution": "PPO"}},
        summary="Extracted 5 claims and 8 ambiguities.",
    )
    assert out.agent_id == "paper-understanding"


# -----------------------------------------------------------------------
# Pipeline state checkpoint tests
# -----------------------------------------------------------------------

def test_pipeline_state_checkpoint_roundtrip(tmp_path: Path):
    """Checkpoint save and load should preserve all state."""
    state = PipelineState(project_id="prj_test123")
    state.stage = PipelineStage.GATE_1_PASSED
    state.paper_claim_map = PaperClaimMap(core_contribution="test")
    state.environment_spec = EnvironmentSpec(dockerfile="FROM python:3.11-slim")
    state.assumption_ledger = [{"id": "A001", "detail": "test"}]
    state.decision_log = ["Gate 1 passed"]

    # Save
    cp_path = state.save_checkpoint(tmp_path)
    assert cp_path.exists()

    # Load
    loaded = PipelineState.load_checkpoint(tmp_path, "prj_test123")
    assert loaded is not None
    assert loaded.stage == PipelineStage.GATE_1_PASSED
    assert loaded.paper_claim_map is not None
    assert loaded.paper_claim_map.core_contribution == "test"
    assert loaded.environment_spec is not None
    assert loaded.environment_spec.dockerfile == "FROM python:3.11-slim"
    assert len(loaded.assumption_ledger) == 1
    assert loaded.decision_log == ["Gate 1 passed"]


def test_pipeline_state_no_checkpoint(tmp_path: Path):
    """Loading from non-existent checkpoint returns None."""
    result = PipelineState.load_checkpoint(tmp_path, "prj_nonexistent")
    assert result is None


def test_pipeline_state_full_roundtrip(tmp_path: Path):
    """Full state with all fields survives checkpoint."""
    state = PipelineState(project_id="prj_full")
    state.stage = PipelineStage.COMPLETE
    state.paper_claim_map = PaperClaimMap(core_contribution="PPO")
    state.environment_spec = EnvironmentSpec(dockerfile="FROM python:3.11")
    state.reproduction_contract = ReproductionContract(
        reproduction_definition="Same algo, same data"
    )
    state.baseline_result = BaselineResult(mode="adapt", code_path="/code")
    state.experiment_artifacts = ExperimentArtifacts(
        metrics={"reward": 485}, success=True
    )
    state.gate_1 = GateDecision(gate="gate_1", passed=True, status=GateStatus.verified)
    state.gate_2 = GateDecision(gate="gate_2", passed=True, status=GateStatus.verified_with_caveats)
    state.gate_3 = GateDecision(gate="gate_3", passed=True, status=GateStatus.verified)
    state.improvement_hypotheses = [
        ImprovementHypothesis(
            path_id="path_1",
            hypothesis="h",
            rationale="r",
            expected_outcome="o",
        )
    ]
    state.path_results = [
        PathResult(path_id="path_1", hypothesis="h", success=True)
    ]
    state.research_map = ResearchMap(baseline_summary="Good")

    state.save_checkpoint(tmp_path)
    loaded = PipelineState.load_checkpoint(tmp_path, "prj_full")
    assert loaded is not None
    assert loaded.stage == PipelineStage.COMPLETE
    assert loaded.gate_2.status == GateStatus.verified_with_caveats
    assert len(loaded.improvement_hypotheses) == 1
    assert loaded.path_results[0].success is True
    assert loaded.research_map.baseline_summary == "Good"


# -----------------------------------------------------------------------
# Pipeline stage ordering
# -----------------------------------------------------------------------

def test_pipeline_stages_are_ordered():
    """Stages must be in correct sequential order."""
    stages = list(PipelineStage)
    expected_order = [
        "ingested",
        "paper_understood",
        "artifacts_discovered",
        "environment_built",
        "plan_created",
        "gate_1_passed",
        "baseline_implemented",
        "baseline_run",
        "gate_2_passed",
        "improvements_selected",
        "improvements_run",
        "gate_3_passed",
        "research_map_generated",
        "complete",
    ]
    assert [s.value for s in stages] == expected_order


def test_reproduction_contract_normalizes_expected_outputs():
    orchestrator = ReproLabOrchestrator(
        project_id="prj_norm",
        runs_root=Path("runs"),
    )

    normalized = orchestrator._normalize_reproduction_contract(
        {
            "reproduction_definition": "same algo",
            "smoke_test_plan": "run smoke",
            "full_run_plan": "run full",
            "evaluation_plan": "score reward",
            "expected_outputs": [
                {"path": "metrics.json", "description": "metrics artifact"},
                {"name": "plots/reward_curve.png"},
            ],
        }
    )

    assert normalized["expected_outputs"] == [
        "metrics.json",
        "plots/reward_curve.png",
    ]
