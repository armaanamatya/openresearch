"""Structured output schemas for all ReproLab agents.

Every agent returns an AgentOutput envelope containing typed structured_outputs.
These Pydantic models define the contract between agents.
"""

from __future__ import annotations

import enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------

class RiskLevel(str, enum.Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


def _coerce_risk_level(value: Any) -> Any:
    # LLMs sometimes emit "low — scripts already…" or "low: rationale" instead
    # of the bare enum value. Take the first token before whitespace, em/en-dash,
    # or colon, lowercased. Non-strings pass through to Pydantic's normal handling.
    if not isinstance(value, str):
        return value
    token = value
    for sep in ("—", "–", ":", " ", "\t", "\n"):
        token = token.split(sep, 1)[0]
    return token.strip().lower()


RiskLevelField = Annotated[RiskLevel, BeforeValidator(_coerce_risk_level)]


class Ambiguity(BaseModel):
    """A detail missing or under-specified in the paper."""
    assumption_id: str = Field(description="e.g. A001")
    detail: str
    chosen_value: str | None = None
    evidence: list[str] = Field(default_factory=list)
    risk: RiskLevelField = RiskLevel.medium


class Assumption(BaseModel):
    """A concrete assumption logged in the assumption ledger."""
    assumption_id: str
    detail: str
    chosen_value: str
    evidence: list[str] = Field(default_factory=list)
    risk: RiskLevelField = RiskLevel.medium
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
    evidence_refs: list[str] = Field(default_factory=list)
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
    risk: RiskLevelField = RiskLevel.medium
    expected_value_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="LLM-estimated probability of success (0-1). Used for adaptive batch selection.",
    )
    category: str = Field(
        default="",
        description="Hypothesis category (e.g. hyperparameter, architecture, data, regularization). Used to diversify batches.",
    )


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


class ImprovementRound(BaseModel):
    """Records one round of parallel improvement paths."""
    round_number: int
    baseline_path_id: str | None = Field(
        default=None,
        description="path_id of the winning path used as baseline for this round (None for round 1 = original baseline)",
    )
    baseline_metrics: dict[str, Any] = Field(default_factory=dict)
    hypotheses: list[ImprovementHypothesis] = Field(default_factory=list)
    path_results: list[PathResult] = Field(default_factory=list)
    best_path_id: str | None = None
    best_metrics: dict[str, Any] = Field(default_factory=dict)
    improvement_pct: float | None = Field(
        default=None,
        description="Best path improvement % over this round's baseline",
    )
    converged: bool = False


class CompositionAttempt(BaseModel):
    """One attempt at composing multiple winning paths."""
    attempt_id: str = Field(description="e.g. compose_all, compose_p1_p2")
    composed_path_ids: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    improvement_pct_vs_baseline: float | None = None
    improvement_pct_vs_best_individual: float | None = None
    success: bool = False
    diff_summary: str = ""
    failure_notes: str = ""


class CompositionPhase(BaseModel):
    """Records the full composition phase: combine winners, ablate if needed."""
    winning_path_ids: list[str] = Field(default_factory=list)
    full_composition: CompositionAttempt | None = None
    ablation_attempts: list[CompositionAttempt] = Field(default_factory=list)
    best_composition: CompositionAttempt | None = None
    strategy_used: str = Field(
        default="",
        description="full_only (combo worked), greedy_ablation (combo failed, searched subsets), skipped (< 2 winners)",
    )


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
# Rubric Verifier (Track 3)
# ---------------------------------------------------------------------------

class RubricAreaScore(BaseModel):
    """One area of the rubric-verifier's PaperBench-style assessment.

    Distinct from ``RubricArea`` (the heuristic, artifact-derived rubric in the
    final report): this is the LLM rubric-verifier's *judged* score for one
    weighted area, including the weak points that feed improvement selection.
    """
    model_config = {"extra": "ignore"}
    area: str
    weight: float = Field(default=0.0, ge=0.0, le=1.0, description="Contribution to the weighted overall score")
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    justification: str = ""
    weak_points: list[str] = Field(
        default_factory=list,
        description="Concrete, actionable gaps lowering this area — consumed by improvement-orchestrator",
    )


class RubricVerification(BaseModel):
    """Structured output of the rubric-verifier agent for one checkpoint.

    ``overall_score`` / ``meets_target`` are computed deterministically by
    ``from_areas`` from the per-area score+weight the LLM supplies — they are
    never taken on trust from the model.
    """
    model_config = {"extra": "ignore"}
    areas: list[RubricAreaScore] = Field(default_factory=list)
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Weight-normalized aggregate of area scores")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Verifier's self-rated confidence in the assessment")
    rubric_source: Literal["paperbench_bundle", "generated"] = "generated"
    target_score: float = Field(default=0.0, ge=0.0, le=1.0, description="PaperBench-equivalent target this run is graded against")
    meets_target: bool = False
    verified_at: str = Field(default="", description="ISO-8601 UTC timestamp of the verification")

    @classmethod
    def from_areas(
        cls,
        areas: list[RubricAreaScore],
        *,
        rubric_source: Literal["paperbench_bundle", "generated"],
        target_score: float,
        confidence: float = 0.0,
        verified_at: str = "",
    ) -> RubricVerification:
        """Build a verification with ``overall_score`` computed, not trusted.

        ``overall_score`` is the weight-normalized mean of area scores (a plain
        mean when no weights are set); ``meets_target`` is derived from it. The
        LLM supplies only per-area ``score`` and ``weight``.
        """
        total_weight = sum(area.weight for area in areas)
        if total_weight > 0:
            overall = sum(area.score * area.weight for area in areas) / total_weight
        elif areas:
            overall = sum(area.score for area in areas) / len(areas)
        else:
            overall = 0.0
        return cls(
            areas=areas,
            overall_score=overall,
            confidence=confidence,
            rubric_source=rubric_source,
            target_score=target_score,
            meets_target=overall >= target_score,
            verified_at=verified_at,
        )


# ---------------------------------------------------------------------------
# Final Report (#30)
# ---------------------------------------------------------------------------

class MetricDelta(BaseModel):
    """Delta comparison for a single metric across paper -> baseline -> improvement."""
    metric_name: str
    paper_target: str | float | None = None
    baseline_value: float | None = None
    best_improved_value: float | None = None
    best_path_id: str | None = Field(default=None, description="Which path achieved the best value")
    delta_vs_paper: float | None = Field(default=None, description="best_improved - paper_target (or baseline - paper_target if no improvement)")
    delta_vs_baseline: float | None = Field(default=None, description="best_improved - baseline")
    pct_change_vs_baseline: float | None = Field(default=None, description="Percent improvement over baseline")
    direction: str = Field(default="higher_is_better", description="higher_is_better or lower_is_better")
    # Statistical rigor — populated only when the runner emits the underlying data.
    n_eval_episodes: int | None = Field(default=None, description="Evaluation sample size (episodes)")
    relative_error_vs_paper: float | None = Field(default=None, description="|reproduced - paper_target| / |paper_target|")
    baseline_std: float | None = Field(default=None, description="Std-dev of the baseline metric, if emitted by the runner")
    improved_std: float | None = Field(default=None, description="Std-dev of the best improved metric, if emitted by the runner")
    effect_size: float | None = Field(default=None, description="Cohen's d (best improved vs baseline), if std available")
    ci95_half_width: float | None = Field(default=None, description="Half-width of the 95% CI around the reproduced value")


class RubricArea(BaseModel):
    """One PaperBench-style rubric dimension, scored deterministically from artifacts.

    Unlike a hardcoded score, every value here is derived from concrete pipeline
    state — claim-map completeness, environment spec fields, execution artifacts
    on disk, and verifier gate decisions — so a degenerate run scores low and an
    honest run scores exactly what it earned.
    """
    area: str
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    weight: float = Field(default=0.0, ge=0.0, le=1.0, description="Contribution to the weighted overall score")
    evidence: list[str] = Field(default_factory=list, description="Artifact files / state fields backing the score")
    rationale: str = ""


class PathSummary(BaseModel):
    """Summary of one parallel improvement path for the final report."""
    path_id: str
    hypothesis: str
    status: str = Field(default="", description="success, failed, or partial")
    diff_summary: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    delta_vs_baseline: dict[str, float] = Field(default_factory=dict, description="metric_name -> delta")
    recommendation: str = ""
    verdict: str = Field(default="", description="accept, reject, or inconclusive")


class FinalReport(BaseModel):
    """Final pipeline report: deltas, path summaries, and overall assessment."""
    model_config = {"extra": "ignore"}
    project_id: str = ""
    paper_title: str = ""
    core_contribution: str = ""

    # Reproduction fidelity
    reproduction_score: float = Field(default=0.0, ge=0.0, le=1.0, description="How close baseline matched paper targets (0-1)")
    reproduction_status: str = Field(default="", description="verified, partial, or failed")

    # Headline paper-vs-reproduction comparison (the primary claimed metric)
    primary_metric: str | None = Field(default=None, description="Primary claimed metric used for the headline comparison")
    paper_primary_target: float | None = Field(default=None, description="Paper-reported target for the primary metric")
    reproduction_primary_value: float | None = Field(default=None, description="Our reproduced value for the primary metric (best of baseline/improved)")
    reproduction_delta_vs_paper: float | None = Field(default=None, description="reproduced - paper target (numerical)")
    reproduction_pct_vs_paper: float | None = Field(default=None, description="Signed percentage gap of the reproduced value vs the paper target")

    # Metric deltas
    metric_deltas: list[MetricDelta] = Field(default_factory=list)

    # Computed PaperBench-style rubric (scored from artifacts, never hardcoded)
    rubric: list[RubricArea] = Field(default_factory=list)
    rubric_overall_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Weighted aggregate of rubric area scores")

    # Track 3 — LLM rubric-verifier assessment + self-improvement summary. All
    # optional: a run with the verifier disabled leaves these at None/0/"" and
    # the report is byte-identical to before Track 3.
    rubric_verification: RubricVerification | None = Field(
        default=None,
        description="Authoritative rubric-verifier assessment of the improved reproduction",
    )
    baseline_rubric_verification: RubricVerification | None = Field(
        default=None,
        description="Rubric-verifier assessment of the baseline, before improvement",
    )
    paperbench_baseline: dict[str, Any] | None = Field(
        default=None,
        description="Published PaperBench score for this paper if known: {score, source, model}",
    )
    verification_delta: float | None = Field(
        default=None,
        description="improved overall_score - baseline overall_score (None when either is missing)",
    )
    improvement_iterations: int = Field(
        default=0,
        description="Completed self-improvement re-iteration rounds",
    )
    comparison_summary: str = Field(
        default="",
        description="Honest 2-line verdict comparing our rubric score to the baseline / PaperBench",
    )

    # Statistical rigor
    statistical_notes: str = Field(default="", description="Honest summary of the statistical basis and its limitations")

    # Parallel improvement paths
    paths: list[PathSummary] = Field(default_factory=list)
    best_path_id: str | None = None
    best_overall_improvement_pct: float | None = None

    # Aggregate stats
    total_paths_run: int = 0
    paths_succeeded: int = 0
    paths_failed: int = 0
    paths_improved_over_baseline: int = 0

    # Final verdict
    overall_verdict: str = Field(default="", description="Concise one-line summary of the run outcome")
    next_steps: list[str] = Field(default_factory=list)


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
