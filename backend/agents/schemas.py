"""Structured output schemas for all ReproLab agents.

Every agent returns an AgentOutput envelope containing typed structured_outputs.
These Pydantic models define the contract between agents.
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------

class RiskLevel(str, enum.Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class Ambiguity(BaseModel):
    """A detail missing or under-specified in the paper."""
    assumption_id: str = Field(description="e.g. A001")
    detail: str
    chosen_value: str | None = None
    evidence: list[str] = Field(default_factory=list)
    risk: RiskLevel = RiskLevel.medium


class Assumption(BaseModel):
    """A concrete assumption logged in the assumption ledger."""
    assumption_id: str
    detail: str
    chosen_value: str
    evidence: list[str] = Field(default_factory=list)
    risk: RiskLevel = RiskLevel.medium
    verified_by: str | None = None


class MetricSpec(BaseModel):
    name: str
    definition: str
    target_value: str | None = None
    source_section: str | None = None


class DatasetRequirement(BaseModel):
    name: str
    source: str = ""
    download_method: str = ""
    size_estimate: str = ""
    notes: str = ""


class TrainingRecipe(BaseModel):
    optimizer: str = ""
    learning_rate: str = ""
    batch_size: str = ""
    epochs_or_steps: str = ""
    scheduler: str = ""
    other_hparams: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Paper Understanding Agent (#23)
# ---------------------------------------------------------------------------

class PaperClaimMap(BaseModel):
    """Canonical extraction of the paper's claims, methods, and gaps."""
    model_config = {"extra": "ignore"}
    core_contribution: str
    claims: list[dict[str, str]] = Field(
        default_factory=list,
        description="Each claim: method, dataset, metric, expected_result",
    )
    datasets: list[DatasetRequirement] = Field(default_factory=list)
    metrics: list[MetricSpec] = Field(default_factory=list)
    model_architecture: str = ""
    training_recipe: TrainingRecipe = Field(default_factory=TrainingRecipe)
    evaluation_protocol: str = ""
    hardware_clues: list[str] = Field(default_factory=list)
    ambiguities: list[Ambiguity] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Environment Detective (#24)
# ---------------------------------------------------------------------------

class EnvironmentSpec(BaseModel):
    """Dockerfile and dependency specification."""
    model_config = {"extra": "ignore"}
    dockerfile: str = Field(default="", description="Full Dockerfile content")
    python_version: str = ""
    framework: str = ""
    framework_version: str | dict[str, str] = ""
    system_packages: list[str] = Field(default_factory=list)
    pip_packages: dict[str, Any] = Field(
        default_factory=dict,
        description="package_name -> pinned_version (or nested env dicts)",
    )
    assumptions: list[Assumption] = Field(default_factory=list)
    compatibility_notes: str | list[str] = ""
    extra: dict[str, Any] = Field(default_factory=dict, description="Overflow for LLM-generated fields")


# ---------------------------------------------------------------------------
# Reproduction Planner
# ---------------------------------------------------------------------------

class ReproductionContract(BaseModel):
    """Defines what counts as reproduction for this paper."""
    model_config = {"extra": "ignore"}
    reproduction_definition: str = ""
    smoke_test_plan: str = ""
    full_run_plan: str = ""
    expected_outputs: list[str] = Field(default_factory=list)
    dataset_plan: str = ""
    evaluation_plan: str = ""
    verification_checklist: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Baseline Implementation (#25)
# ---------------------------------------------------------------------------

class BaselineResult(BaseModel):
    """Output of the Baseline Implementation Agent."""
    model_config = {"extra": "ignore"}
    mode: str = Field(default="", description="'adapt' or 'implement_from_paper'")
    code_path: str = ""
    dockerfile_path: str = ""
    diff_summary: str = ""
    commands_to_run: list[str] = Field(default_factory=list)
    assumptions_applied: list[str] = Field(
        default_factory=list,
        description="Assumption IDs (e.g. A001) applied during implementation",
    )


# ---------------------------------------------------------------------------
# Experiment Runner (#26)
# ---------------------------------------------------------------------------

class ExperimentArtifacts(BaseModel):
    """Hard artifacts produced by the experiment runner."""
    model_config = {"extra": "ignore"}
    metrics: dict[str, Any] = Field(default_factory=dict)
    plots: list[str] = Field(default_factory=list, description="Plot file paths")
    log_path: str = ""
    commands_log_path: str = ""
    provenance_path: str = ""
    success: bool = False
    error_message: str = ""


# ---------------------------------------------------------------------------
# Verification (#27)
# ---------------------------------------------------------------------------

class GateStatus(str, enum.Enum):
    verified = "verified"
    verified_with_caveats = "verified_with_caveats"
    partial_reproduction = "partial_reproduction"
    failed_reproduction = "failed_reproduction"
    blocked_requires_human = "blocked_requires_human"
    invalid_claim = "invalid_claim"


class VerifierScore(BaseModel):
    model_config = {"extra": "ignore"}
    verifier_name: str = ""
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    findings: list[str] = Field(default_factory=list)
    mismatches: list[str] = Field(default_factory=list)
    severity: str = ""


class VerificationReport(BaseModel):
    """Output of the Supervisor Verification Agent."""
    model_config = {"extra": "ignore"}
    gate: str = Field(default="", description="gate_1, gate_2, or gate_3")
    status: GateStatus
    verifier_scores: list[VerifierScore] = Field(default_factory=list)
    reasoning: str = ""
    decision_log_entry: str = ""


class GateDecision(BaseModel):
    """Simplified gate pass/fail for orchestrator consumption."""
    gate: str
    passed: bool
    status: GateStatus
    blocking_issues: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Improvement (#28)
# ---------------------------------------------------------------------------

class ImprovementHypothesis(BaseModel):
    """A single improvement path hypothesis."""
    model_config = {"extra": "ignore"}
    path_id: str
    hypothesis: str
    rationale: str
    expected_outcome: str
    compute_estimate: str = ""
    risk: RiskLevel = RiskLevel.medium


class PathResult(BaseModel):
    """Output of one Improvement Path Agent."""
    model_config = {"extra": "ignore"}
    path_id: str = ""
    hypothesis: str
    diff_summary: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    plots: list[str] = Field(default_factory=list)
    commands: list[str] = Field(default_factory=list)
    failure_notes: str = ""
    recommendation: str = ""
    success: bool = False


# ---------------------------------------------------------------------------
# Research Map (#29)
# ---------------------------------------------------------------------------

class ResearchMap(BaseModel):
    """Final synthesis produced by the Supervisor after all paths verified."""
    model_config = {"extra": "ignore"}
    baseline_summary: str = ""
    promising_directions: list[str] = Field(default_factory=list)
    dead_ends: list[str] = Field(default_factory=list)
    inconclusive: list[str] = Field(default_factory=list)
    next_experiments: list[str] = Field(default_factory=list)
    overall_reproducibility_assessment: str = ""


# ---------------------------------------------------------------------------
# Generic agent output envelope
# ---------------------------------------------------------------------------

class AgentOutput(BaseModel):
    """Standard envelope every agent returns to the orchestrator."""
    agent_id: str
    status: str = "completed"
    structured_outputs: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    exploration_log: dict[str, Any] = Field(default_factory=dict)
