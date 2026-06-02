"""Structured output schemas for all ReproLab agents.

Every agent returns an AgentOutput envelope containing typed structured_outputs.
These Pydantic models define the contract between agents.
"""

from __future__ import annotations

import enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field, field_validator, model_validator


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
    definition: str = ""
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

    # LLM-generated REPL code routinely hands the list fields below as bare
    # strings (`datasets=['Gaussian Linear', ...]`, `claims=['...']`) where the
    # schema expects dicts / submodels. One canonical coercion keeps claims,
    # datasets and metrics consistent — a bare string becomes a single-key dict
    # on the field's natural name key. Every non-string item (a dict, or an
    # already-built DatasetRequirement / MetricSpec instance from the offline
    # agent) passes through untouched for pydantic itself to validate.
    @staticmethod
    def _coerce_str_items(v: Any, key: str) -> Any:
        if not isinstance(v, list):
            return v
        return [{key: item} if isinstance(item, str) else item for item in v]

    @field_validator("claims", mode="before")
    @classmethod
    def _coerce_claims(cls, v: Any) -> Any:
        return cls._coerce_str_items(v, "claim")

    @field_validator("datasets", mode="before")
    @classmethod
    def _coerce_datasets(cls, v: Any) -> Any:
        return cls._coerce_str_items(v, "name")

    @field_validator("metrics", mode="before")
    @classmethod
    def _coerce_metrics(cls, v: Any) -> Any:
        # MetricSpec.definition defaults to "", so a {name} dict validates.
        return cls._coerce_str_items(v, "name")

    # 2026-05-25 Adam regression: agent built method_spec as
    #     "training_recipe": str(s2.get("training_recipe", ""))[:300]
    # — calling str() on the dict it got back from understand_section, in
    # the name of "defensive truncation". Pydantic refused to coerce that
    # string into TrainingRecipe and the run died at detect_environment.
    # Same kind of LLM-defensive-coding mismatch as the list-of-strings
    # case for claims/datasets/metrics. Coerce gracefully:
    #   * dict / TrainingRecipe → pass through (pydantic validates).
    #   * str → try ast.literal_eval; if it's a dict-repr, use it. Otherwise
    #     wrap in TrainingRecipe.other_hparams.raw so the agent's textual
    #     description is preserved (detect_environment doesn't strictly
    #     need a structured recipe — it's mostly metadata for downstream).
    #   * list → wrap items into other_hparams.
    #   * None → empty dict → uses TrainingRecipe default factory.
    @field_validator("training_recipe", mode="before")
    @classmethod
    def _coerce_training_recipe(cls, v: Any) -> Any:
        if v is None:
            return {}
        if isinstance(v, str):
            stripped = v.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    import ast as _ast
                    parsed = _ast.literal_eval(stripped)
                    if isinstance(parsed, dict):
                        return parsed
                except (ValueError, SyntaxError):
                    pass
            # Otherwise wrap the prose into other_hparams.raw so the agent's
            # information survives without crashing the schema.
            return {"other_hparams": {"raw": stripped[:1000]}}
        if isinstance(v, list):
            return {"other_hparams": {"items": [str(x)[:200] for x in v[:20]]}}
        return v

    # Lane V — generic string-field coercion for the three free-text fields
    # the agent routinely passes as dict / list when it builds method_spec.
    # 2026-05-25 Dropout regression: agent passed `model_architecture` as a
    # dict like `{"layers": [...]}` (extracted from understand_section) and
    # pydantic refused with `ValidationError: model_architecture: Input
    # should be a valid string (string_type)`. Same shape as the
    # training_recipe regression. One shared coercion helper keeps the
    # three validators DRY without inheritance gymnastics.

    @staticmethod
    def _coerce_to_string(v: Any) -> Any:
        """Best-effort str coercion. Dict → JSON, list → joined str, else pass through."""
        if v is None:
            return ""
        if isinstance(v, str):
            return v[:5000]
        if isinstance(v, dict):
            try:
                import json as _json
                return _json.dumps(v, default=str)[:2000]
            except Exception:  # noqa: BLE001
                return str(v)[:2000]
        if isinstance(v, list):
            return "; ".join(str(x)[:200] for x in v[:20])
        return str(v)[:2000]

    # Lane W — list[str] / list[Ambiguity] coercion. The LLM frequently passes
    # `hardware_clues="GPU"` or `hardware_clues="NVIDIA K40c, 24GB"` (a string)
    # where the schema expects list[str]. Same shape as the claims/datasets
    # regression — same surgical fix. Splits on commas / semicolons / slashes
    # / " and " so multi-item strings still parse into separate clues.
    @staticmethod
    def _coerce_to_str_list(v: Any) -> Any:
        if v is None:
            return []
        if isinstance(v, list):
            # Coerce each element to str so a stray int/None never breaks list[str].
            return [str(x) for x in v if x is not None]
        if isinstance(v, (tuple, set)):
            return [str(x) for x in v if x is not None]
        if isinstance(v, str):
            import re as _re
            parts = [p.strip() for p in _re.split(r",|;|/|\band\b", v) if p.strip()]
            return parts or ([v.strip()] if v.strip() else [])
        return [str(v)]

    @field_validator("hardware_clues", mode="before")
    @classmethod
    def _coerce_hardware_clues(cls, v: Any) -> Any:
        return cls._coerce_to_str_list(v)

    @field_validator("ambiguities", mode="before")
    @classmethod
    def _coerce_ambiguities(cls, v: Any) -> Any:
        # LLM-natural shapes:
        #   * None → []
        #   * str → single Ambiguity with bare detail and an auto-id.
        #   * list[str] → one Ambiguity per item with auto-ids A001, A002…
        #   * list[dict] / list[Ambiguity] → pass through (pydantic validates).
        if v is None:
            return []
        if isinstance(v, str):
            return [{"assumption_id": "A001", "detail": v}]
        if not isinstance(v, list):
            return v
        coerced = []
        for i, item in enumerate(v, start=1):
            if isinstance(item, str):
                coerced.append({"assumption_id": f"A{i:03d}", "detail": item})
            else:
                coerced.append(item)
        return coerced

    @field_validator("model_architecture", mode="before")
    @classmethod
    def _coerce_model_architecture(cls, v: Any) -> Any:
        return cls._coerce_to_string(v)

    @field_validator("evaluation_protocol", mode="before")
    @classmethod
    def _coerce_evaluation_protocol(cls, v: Any) -> Any:
        return cls._coerce_to_string(v)

    @field_validator("core_contribution", mode="before")
    @classmethod
    def _coerce_core_contribution(cls, v: Any) -> Any:
        return cls._coerce_to_string(v)


# ---------------------------------------------------------------------------
# Environment Detective (#24)
# ---------------------------------------------------------------------------
# Compute-adjusted rubric (spec 2026-05-25-compute-adjusted-rubric-design.md)
# ---------------------------------------------------------------------------


class MetricFloor(BaseModel):
    """Per-metric floor used by the compute-adjusted rubric grader.

    The grader scores `result_match` leaves against `floor` instead of the
    paper's `paper_target` when compute is clipped. `direction` tells the
    grader whether higher or lower values of the metric are better.
    """

    model_config = {"extra": "ignore"}

    metric: str = Field(description="Rubric leaf id / metric name (e.g. 'mnist_test_loss').")
    direction: Literal["higher", "lower"] = Field(
        description="'higher' for accuracy-style, 'lower' for loss/error-style.",
    )
    paper_target: float = Field(description="Paper's headline value for this metric.")
    floor: float = Field(description="Plausibly-reachable value given the actual compute budget.")
    rationale: str = Field(default="", description="One-sentence justification for the floor.")

    @model_validator(mode="after")
    def _floor_consistent_with_direction(self) -> "MetricFloor":
        if self.direction == "higher" and self.floor > self.paper_target:
            raise ValueError(
                f"MetricFloor: direction='higher' requires floor (= {self.floor}) "
                f"<= paper_target (= {self.paper_target}). A floor cannot exceed the paper's headline."
            )
        if self.direction == "lower" and self.floor < self.paper_target:
            raise ValueError(
                f"MetricFloor: direction='lower' requires floor (= {self.floor}) "
                f">= paper_target (= {self.paper_target}). A loss floor cannot beat the paper's headline."
            )
        return self


class ComputeScope(BaseModel):
    """Declared compute envelope for a reproduction.

    Emitted by the planning agent when `execution_profile.mode == "efficient"`
    or `minimize_compute=True`. Consumed by the grader to score result_match
    leaves against `metric_floors[*].floor` instead of paper targets.
    """

    model_config = {"extra": "ignore"}

    is_clipped: bool = Field(
        description="True when ANY axis (epochs, dataset size) is reduced vs paper.",
    )
    paper_epochs: int | None = None
    actual_epochs: int | None = None
    paper_dataset_size: int | None = None
    actual_dataset_size: int | None = None
    rationale: str = Field(default="", description="One-sentence summary of the budget reduction.")
    metric_floors: list[MetricFloor] = Field(
        default_factory=list,
        description="Per-result_match-leaf floor; empty when is_clipped=False.",
    )


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

class MetricPath(BaseModel):
    """Single metric the agent commits to emitting in metrics.json.

    Declared at plan_reproduction time. implement_baseline is bound to emit
    EXACTLY this dotted path. rubric_guard validates against the declared
    path, eliminating nested-vs-flat ambiguity at the source.

    When metrics_shape is non-empty on ReproductionContract, rubric_guard
    uses the declared json_path for authoritative path lookup; when empty,
    it falls back to the fingerprint matcher (commit befb51c).
    """
    model_config = {"extra": "ignore"}

    metric_id: str = Field(
        description="Stable id used by rubric leaves (e.g. 'mnist_logistic_adam_final_nll')"
    )
    json_path: str = Field(
        description="Dotted path inside metrics.json (e.g. 'per_model.mnist_logistic.per_dataset.mnist.adam_final_nll')"
    )
    rubric_leaf_ids: list[str] = Field(
        default_factory=list,
        description="Optional: explicit rubric leaf ids this metric satisfies. When empty, fingerprint matching falls back.",
    )


class ReproductionContract(BaseModel):
    """Defines what counts as reproduction for this paper."""
    model_config = {"extra": "ignore"}
    # str | list[str]: an LLM naturally returns plan/definition fields as a list
    # of steps. Accepting either avoids `plan_reproduction` fail-softing to an
    # empty contract on every such paper (observed run-1 `evaluation_plan`
    # Pydantic error). Mirrors `compatibility_notes` above.
    reproduction_definition: str | list[str] = ""
    smoke_test_plan: str | list[str] = ""
    full_run_plan: str | list[str] = ""
    expected_outputs: list[str] = Field(default_factory=list)
    dataset_plan: str | list[str] = ""
    evaluation_plan: str | list[str] = ""
    verification_checklist: list[str] = Field(default_factory=list)
    # Compute-adjusted rubric: opt-in; None on max mode or when not yet declared.
    # The planning agent fills this when execution_profile.mode == "efficient" or
    # minimize_compute=True (spec 2026-05-25-compute-adjusted-rubric-design.md).
    compute_scope: ComputeScope | None = None
    # θ: agent-declared metric paths (Solution IV, spec 2026-05-26).
    # When non-empty, rubric_guard validates against these declared json_paths
    # (authoritative); when empty, falls back to the fingerprint matcher.
    # Default empty list → backward compat: existing runs without metrics_shape
    # use the tier-2 fingerprint matcher unchanged.
    metrics_shape: list[MetricPath] = Field(
        default_factory=list,
        description=(
            "Agent's declared metrics.json shape. When non-empty, rubric_guard "
            "validates against MetricPath.json_path directly. When empty, falls "
            "back to the existing fingerprint matcher."
        ),
    )
    # λ: canonical loader recipes for paper-mentioned datasets (PR-λ, 2026-05-26).
    # Populated by plan_reproduction via dataset_recipes.find_recipes_in_text.
    # Each entry is a DatasetRecipe.__dict__ snapshot (plain dict, JSON-safe).
    # implement_baseline binds Sonnet to use these loaders verbatim — prevents
    # regressions like load_dataset('imdb') vs load_dataset('stanfordnlp/imdb').
    # Default empty list → backward compat: runs without data_recipes are
    # unaffected; the existing _RUNTIME_DETECTION_BLOCK static guidance applies.
    data_recipes: list[dict] = Field(
        default_factory=list,
        description=(
            "Canonical loader recipes for paper-mentioned datasets. Populated by "
            "plan_reproduction via dataset_recipes.find_recipes_in_text(paper_text). "
            "Each entry is a DatasetRecipe.__dict__ snapshot. implement_baseline binds "
            "Sonnet to use these loaders verbatim — prevents regressions like "
            "load_dataset('imdb') vs load_dataset('stanfordnlp/imdb')."
        ),
    )


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
    title: str = Field(
        default="",
        description="Short human-readable name for the candidate (model-generated). Emitted in candidate_proposed SSE events.",
    )
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
# Dynamic GPU selection (#dynamic-gpu spec 2026-05-23)
# ---------------------------------------------------------------------------

class GpuRequirements(BaseModel):
    """LLM-derived hardware requirements extracted from paper text.

    The RLM root constructs this from accumulated PaperClaimMap.hardware_clues
    plus reasoning over the full workload (training + inference + evaluation).
    """
    model_config = {"extra": "ignore"}
    estimated_vram_gb: int | None = Field(
        default=None, ge=0, le=1024,
        description="Whole-workload VRAM estimate; None when LLM cannot estimate",
    )
    paper_gpu_string: str | None = None
    paper_gpu_count: int | None = Field(default=None, ge=0, le=64)
    reasoning: str = Field(default="", description="One-line rationale, surfaced in SSE event")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class GpuPlan(BaseModel):
    """Resolved provisioning plan, consumed by RunpodBackend."""
    model_config = {"extra": "ignore"}
    runpod_id: str = Field(description="Verbatim RunPod gpu_type identifier")
    short_name: str = Field(description="Internal short name; matches gpu_catalog.GpuSku.short_name")
    vram_gb: int = Field(ge=1)
    gpu_count: int = Field(ge=1, le=8)
    cloud_type: Literal["COMMUNITY", "SECURE"]
    sku_usd_per_hr: float = Field(ge=0.0, description="Per-GPU rate from catalog")
    total_usd_per_hr: float = Field(ge=0.0, description="sku_usd_per_hr * gpu_count")
    container_disk_gb: int = Field(ge=1)
    volume_gb: int = Field(ge=1)
    source: Literal["paper", "fallback", "manual", "informational"]
    requirements: GpuRequirements
    ladder_remaining: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Short names of next-larger SKUs for OOM escalation",
    )
    resolved_at: str = Field(description="ISO-8601 timestamp")


# ---------------------------------------------------------------------------
# Scope, paper hints, and invariants
#
# Three layers compose into the effective run-scope:
#   1. Paper default (from PAPER_HINTS[paper_id].default_scope) — rubric expects
#   2. Operator override (from --scope-spec CLI flag) — narrows or expands
#   3. Empty defaults — when neither is set
# Operator absences fall back to paper defaults via ScopeSpec.merge_with_paper_default.
#
# Invariants are deterministic regex checks over the agent's code, run alongside
# the LLM leaf scorer. must_not_match violations are hard gates (leaf score 0);
# must_match presence is a soft signal fed to the LLM grader as evidence.
# ---------------------------------------------------------------------------


class DatasetSlice(BaseModel):
    """One dataset/environment the scope targets, with optional eval slice.

    For RL papers, ``name`` is the environment id (e.g. "ALFWorld"). For
    supervised papers, ``name`` is the dataset id. ``episodes`` / ``split``
    are advisory — the agent's prompt is told about them but the harness
    does not enforce them.
    """

    name: str
    episodes: int | None = None
    split: str | None = None

    def normalized_id(self) -> str:
        """Stable id used in scope.ran, scope.gaps, and experiment_runs.jsonl."""
        return self.name


class ScopeSpec(BaseModel):
    """Operator-stated reproduction scope.

    Each field defaults to "no constraint" (empty list / dict / ""). The
    effective scope for a run is built by ``merge_with_paper_default(paper_default)``
    — operator-set fields win, operator absences fall back to paper defaults.

    ``models`` and ``skip_models`` are post-merge reconciled: ``skip_models``
    items are removed from ``models`` so the agent never sees a contradicting
    pair (paper default lists Qwen-7B; operator skips it → effective models
    list is the smaller two without 7B).

    ``datasets`` and ``skip_datasets`` mirror that pair for the
    dataset/environment axis (2026-06-01). ``skip_datasets`` is the *verified*
    operator-scope source for environments: its entries are removed from
    ``datasets`` and become ``operator_scope`` rubric exclusions so out-of-scope
    environments (e.g. ALFWorld/WebShop on a Search-QA-only run) are excluded
    from the rubric — numerator AND denominator — instead of scored 0. When the
    operator narrows ``datasets`` to a subset of the paper default, the dropped
    paper datasets are folded into ``skip_datasets`` automatically by
    :meth:`merge_with_paper_default` (the narrowing IS the operator decision).
    """

    models: list[str] = Field(default_factory=list)
    skip_models: list[str] = Field(default_factory=list)
    datasets: list[DatasetSlice] = Field(default_factory=list)
    skip_datasets: list[str] = Field(default_factory=list)
    seeds: list[int] = Field(default_factory=list)
    eval_slice: dict[str, int] = Field(default_factory=dict)
    budget_per_model: dict[str, float] = Field(default_factory=dict)
    force_clean_cache: bool = False
    free_text: str = ""

    @field_validator("datasets", mode="before")
    @classmethod
    def _coerce_datasets(cls, v: object) -> object:
        # Accept ["ALFWorld", "WebShop"] OR [{"name": "ALFWorld", "episodes": 32}].
        # A bare string becomes DatasetSlice(name=<str>); a dict is constructed normally.
        if not isinstance(v, list):
            return v
        out: list[object] = []
        for item in v:
            if isinstance(item, str):
                out.append({"name": item})
            else:
                out.append(item)
        return out

    @property
    def is_multi_model(self) -> bool:
        return len(self.models) > 1

    @property
    def is_multi_dataset(self) -> bool:
        return len(self.datasets) > 1

    def dataset_ids(self) -> list[str]:
        return [d.normalized_id() for d in self.datasets]

    def requested_evidence_ids(self) -> set[str]:
        """Set of identifiers expected to appear in scope.ran when the run completes.

        Rules:
          - No models and no datasets → empty set (nothing scoped).
          - Models only → set of model ids.
          - Datasets only → set of dataset ids.
          - Both → set of "model/dataset" cross-product ids.
        """
        if not self.models and not self.datasets:
            return set()
        if not self.models:
            return set(self.dataset_ids())
        if not self.datasets:
            return set(self.models)
        return {f"{m}/{d}" for m in self.models for d in self.dataset_ids()}

    def merge_with_paper_default(
        self, paper_default: "ScopeSpec | None"
    ) -> "ScopeSpec":
        """Operator-supplied (``self``) wins; falls back to paper_default per field.

        Post-merge step: ``skip_models`` entries are removed from ``models`` so
        the effective scope never carries a model the operator explicitly excluded.
        ``free_text`` is concatenated (operator first, then paper default)
        because both may carry useful prose.
        """
        if paper_default is None:
            base = self.model_copy()
        else:
            base = ScopeSpec(
                models=self.models or paper_default.models,
                skip_models=list({*self.skip_models, *paper_default.skip_models}),
                datasets=self.datasets or paper_default.datasets,
                skip_datasets=list({*self.skip_datasets, *paper_default.skip_datasets}),
                seeds=self.seeds or paper_default.seeds,
                eval_slice=self.eval_slice or paper_default.eval_slice,
                budget_per_model=(
                    self.budget_per_model or paper_default.budget_per_model
                ),
                force_clean_cache=(
                    self.force_clean_cache or paper_default.force_clean_cache
                ),
                free_text=(
                    "\n".join(p for p in (self.free_text, paper_default.free_text) if p)
                ),
            )

        # Models (2026-06-01): an operator-narrowed ``models`` list implicitly
        # de-scopes the paper-default models it dropped. Fold those into
        # ``skip_models`` (symmetry with the datasets/skip_datasets rule below) so
        # the rubric excludes their leaves instead of scoring them 0. Only triggers
        # when the operator actually set ``models`` (a deliberate narrowing), never
        # on a pure paper-default run.
        if paper_default is not None and self.models:
            _active_m = set(base.models)
            _dropped_m = [m for m in paper_default.models if m not in _active_m]
            if _dropped_m:
                base = base.model_copy(
                    update={"skip_models": list({*base.skip_models, *_dropped_m})}
                )
        if base.skip_models and base.models:
            skipped = set(base.skip_models)
            base = base.model_copy(
                update={"models": [m for m in base.models if m not in skipped]}
            )

        # Datasets/environments (2026-06-01): an operator-narrowed ``datasets``
        # list implicitly de-scopes the paper-default datasets it dropped. Fold
        # those into ``skip_datasets`` — the verified operator-scope source for
        # the environment axis (mirrors ``skip_models``) — so the rubric excludes
        # their leaves instead of scoring them 0. Explicit ``skip_datasets``
        # entries union in. Only triggers when the operator actually set
        # ``datasets`` (a deliberate narrowing), never on a pure paper-default run.
        if paper_default is not None and self.datasets:
            _active_ds = {d.normalized_id() for d in base.datasets}
            _dropped = [
                d.normalized_id()
                for d in paper_default.datasets
                if d.normalized_id() not in _active_ds
            ]
            if _dropped:
                base = base.model_copy(
                    update={"skip_datasets": list({*base.skip_datasets, *_dropped})}
                )
        # Reconcile: a ``skip_datasets`` entry must not also remain in the
        # effective ``datasets`` the agent is told to run (case-insensitive id).
        if base.skip_datasets and base.datasets:
            _skip_ds = {s.strip().lower() for s in base.skip_datasets if s}
            base = base.model_copy(
                update={
                    "datasets": [
                        d
                        for d in base.datasets
                        if d.normalized_id().strip().lower() not in _skip_ds
                    ]
                }
            )
        return base


import re as _re  # local import to avoid polluting the module top namespace


class InvariantSpec(BaseModel):
    """Deterministic regex check for one algorithmic invariant in the agent's code.

    Two signal types:
      - ``must_match``: at least one pattern must appear in at least one file
        matching ``file_glob``. Soft signal — passed to the LLM grader as
        evidence; not a hard gate.
      - ``must_not_match``: NO pattern may appear in ANY matching file.
        Hard gate — the rubric leaf score is forced to 0 when any pattern
        appears.

    Patterns are validated at construction: malformed regex raises ValueError
    so a broken InvariantSpec cannot ship in PAPER_HINTS undetected.
    """

    name: str
    rationale: str
    file_glob: str = "**/*.py"
    must_match: list[str] = Field(default_factory=list)
    must_not_match: list[str] = Field(default_factory=list)

    @field_validator("file_glob", mode="before")
    @classmethod
    def _default_glob(cls, v: object) -> object:
        return v or "**/*.py"

    @field_validator("must_match", "must_not_match")
    @classmethod
    def _validate_regex(cls, v: list[str]) -> list[str]:
        for pat in v:
            try:
                _re.compile(pat)
            except _re.error as exc:
                raise ValueError(
                    f"InvariantSpec: malformed regex pattern {pat!r}: {exc}"
                ) from exc
        return v


class InvariantResult(BaseModel):
    """Per-invariant check result returned by the assert_invariant primitive.

    Layout chosen so the rubric scorer can both render evidence to the
    operator and reason about why a leaf failed:
      - ``passed``: True iff every ``must_match`` pattern matched AND no
        ``must_not_match`` pattern matched.
      - ``must_match_evidence``: per-pattern list of "<file>:<line>: <excerpt>"
        strings — the matches that satisfied a must_match pattern. Empty
        when no matches found for a pattern.
      - ``must_not_match_violations``: per-pattern list of "<file>:<line>: <excerpt>"
        strings — matches that VIOLATED a must_not_match pattern. Empty
        when no violations.
      - ``files_scanned``: count of files actually inspected.
    """

    name: str
    passed: bool
    must_match_evidence: dict[str, list[str]] = Field(default_factory=dict)
    must_not_match_violations: dict[str, list[str]] = Field(default_factory=dict)
    files_scanned: int = 0


class PaperHint(BaseModel):
    """Built-in paper-specific extras applied via ``--paper-hint <id>``.

    Three independent layers combine when a paper hint is applied to a run:
      - ``guidance``: free-text appended to REPROLAB_BASELINE_EXTRA_GUIDANCE
        (with a "[paper-hint <id>] " prefix) so it composes with operator-set
        guidance via the existing env-var hook in baseline_implementation.py.
      - ``default_scope``: a ScopeSpec providing rubric-default models / datasets
        / seeds. Operator's --scope-spec overrides per-field via
        ``ScopeSpec.merge_with_paper_default``.
      - ``invariants``: a list of InvariantSpec the rubric scorer applies to
        the agent's code; also callable by the agent as an advisory self-check
        via the assert_invariant primitive (PR D).

    ``primitive_share`` is an optional per-paper override for the wallclock
    partitioning introduced in PR E; None means use the run-level default.
    """

    guidance: str = ""
    default_scope: ScopeSpec | None = None
    invariants: list[InvariantSpec] = Field(default_factory=list)
    # #7 benchmark integrity: curated resources (the paper's OWN repo, etc.) that
    # NO agent may fetch — seeds every agent's RuntimeGuard. arXiv runs load
    # neither a PaperBench bundle nor --blacklist, so this is the curated source
    # that protects them. List ONLY the paper's own artifacts; never framework
    # deps (trl, etc.) the reproduction legitimately needs.
    blocked_resources: list[str] = Field(default_factory=list)
    primitive_share: dict[str, float] | None = None


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
