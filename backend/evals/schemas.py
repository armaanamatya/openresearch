"""Evaluation schemas for ReproLab agent evaluation system.

Covers both reproduction evaluation (ground-truth-based) and
innovation evaluation (judgment-based).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EvalMode(str, Enum):
    reproduction = "reproduction"
    innovation = "innovation"
    ab_test = "ab_test"
    elo_tournament = "elo_tournament"


# --- Reproduction Evaluation ---


@dataclass(frozen=True)
class CompositeWeights:
    """Weights for ``ReproductionScore.composite_score`` — a report/rank-only
    blend that must stay deterministic-dominated (spec §6.2). ``fidelity`` is the
    only LLM-derived term; the default zeroes it so a hallucinated grade can never
    lift the offline composite. Overridable per call for A/B weight tuning (spec
    §10 Q1) — the composite never reaches the verdict surface regardless of weights.
    """

    build: float = 0.15
    run: float = 0.25
    metric_match: float = 0.60
    fidelity: float = 0.0


DEFAULT_COMPOSITE_WEIGHTS = CompositeWeights()


class ReproductionScore(BaseModel):
    """Scores a single reproduction attempt against paper ground truth."""

    version: str = ""
    paper_id: str = ""
    build_success: bool = False
    run_success: bool = False
    metric_match: float = 0.0  # 0-1: how close reproduced metrics are to paper's
    fidelity_score: float = 0.0  # 0-1: LLM-judged reproduction fidelity (display term)
    assumption_accuracy: float = 0.0  # 0-1: fraction of assumptions that were valid
    step_count: int = 0  # number of agent steps/tool calls
    cost_usd: float = 0.0  # total LLM API cost
    wall_time_s: float = 0.0  # wall-clock seconds
    timestamp: float = Field(default_factory=time.time)
    details: dict[str, Any] = Field(default_factory=dict)

    def composite_score(self, weights: CompositeWeights | None = None) -> float:
        """Deterministic-dominated weighted composite (report/rank-only).

        Default weights (``DEFAULT_COMPOSITE_WEIGHTS``) put all mass on the
        measured components — build/run/metric_match — and ZERO on the LLM
        ``fidelity_score``, so the offline composite keys on evidence, not the
        grade (spec §6.2, north-star invariant). This value lives only in the
        ``backend/evals/`` leaderboard store and never reaches
        ``final_report.verdict``. Pass ``weights`` to re-blend for A/B tuning.
        """
        w = weights or DEFAULT_COMPOSITE_WEIGHTS
        return (
            w.build * float(self.build_success)
            + w.run * float(self.run_success)
            + w.metric_match * self.metric_match
            + w.fidelity * self.fidelity_score
        )


# --- Innovation Evaluation ---


class HypothesisScore(BaseModel):
    """5-dimension rubric for hypothesis quality (1-5 each)."""

    hypothesis_id: str = ""
    hypothesis_text: str = ""
    novelty: int = 1  # 1=parameter sweep, 5=structural insight
    feasibility: int = 1  # 1=impossible, 5=immediately testable
    significance: int = 1  # 1=trivial, 5=paradigm-shifting
    clarity: int = 1  # 1=vague, 5=precisely stated
    actionability: int = 1  # 1=no clear next step, 5=obvious implementation
    rationale: str = ""  # judge's reasoning

    def mean_score(self) -> float:
        return (self.novelty + self.feasibility + self.significance
                + self.clarity + self.actionability) / 5.0

    def is_above_baseline(self, threshold: float = 2.5) -> bool:
        """Above baseline = better than a pure parameter sweep."""
        return self.mean_score() > threshold


class IntegrityFlag(str, Enum):
    multi_variable_change = "multi_variable_change"
    metric_inconsistency = "metric_inconsistency"
    selective_reporting = "selective_reporting"
    data_leakage = "data_leakage"
    missing_control = "missing_control"


class IntegrityReport(BaseModel):
    """Results of integrity checks on an experiment path."""

    path_id: str = ""
    flags: list[IntegrityFlag] = Field(default_factory=list)
    variables_changed: list[str] = Field(default_factory=list)
    reported_metrics: dict[str, float] = Field(default_factory=dict)
    rerun_metrics: dict[str, float] = Field(default_factory=dict)
    metric_deviation: float = 0.0  # max |reported - rerun| / reported
    passed: bool = True
    details: str = ""

    @property
    def flag_count(self) -> int:
        return len(self.flags)


class ResearchMapScore(BaseModel):
    """Rubric for research map quality (adapted from ResearchRubrics)."""

    classification_accuracy: float = 0.0  # 0-1: dead ends correctly identified?
    direction_validity: float = 0.0  # 0-1: promising directions actually improved?
    next_experiment_novelty: float = 0.0  # 0-1: are next experiments novel?
    negative_result_honesty: float = 0.0  # 0-1: failures documented honestly?
    synthesis_quality: float = 0.0  # 0-1: integrates across sources?
    actionability: float = 0.0  # 0-1: can someone pick this up?
    rationale: str = ""

    def composite_score(self) -> float:
        """Weighted composite."""
        return (
            0.20 * self.classification_accuracy
            + 0.20 * self.direction_validity
            + 0.15 * self.next_experiment_novelty
            + 0.15 * self.negative_result_honesty
            + 0.15 * self.synthesis_quality
            + 0.15 * self.actionability
        )


class InnovationScore(BaseModel):
    """Aggregate innovation evaluation for a pipeline run."""

    version: str = ""
    paper_id: str = ""
    hypothesis_scores: list[HypothesisScore] = Field(default_factory=list)
    integrity_reports: list[IntegrityReport] = Field(default_factory=list)
    research_map_score: ResearchMapScore | None = None
    timestamp: float = Field(default_factory=time.time)

    def mean_hypothesis_quality(self) -> float:
        if not self.hypothesis_scores:
            return 0.0
        return sum(h.mean_score() for h in self.hypothesis_scores) / len(self.hypothesis_scores)

    def integrity_pass_rate(self) -> float:
        if not self.integrity_reports:
            return 1.0
        return sum(1 for r in self.integrity_reports if r.passed) / len(self.integrity_reports)


# --- A/B Testing ---


class ABTestResult(BaseModel):
    """Result of a Bayesian A/B test between two agent versions."""

    version_a: str
    version_b: str
    metric: str
    n_a: int = 0
    n_b: int = 0
    mean_a: float = 0.0
    mean_b: float = 0.0
    p_a_better: float = 0.5  # posterior probability A > B
    p_b_better: float = 0.5
    credible_interval_a: tuple[float, float] = (0.0, 1.0)
    credible_interval_b: tuple[float, float] = (0.0, 1.0)
    is_significant: bool = False  # p_better > 0.95
    winner: str | None = None  # version_a, version_b, or None (inconclusive)
    details: str = ""


# --- Elo Tournament ---


class EloRating(BaseModel):
    """Elo rating for an agent version."""

    version: str
    rating: float = 1500.0
    matches_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def win_rate(self) -> float:
        if self.matches_played == 0:
            return 0.0
        return self.wins / self.matches_played


class EloMatchResult(BaseModel):
    """Single match result in an Elo tournament."""

    version_a: str
    version_b: str
    paper_id: str
    winner: str | None = None  # None = draw
    score_a: float = 0.0
    score_b: float = 0.0
    judge_rationale: str = ""


# --- Eval Run (wraps everything) ---


class EvalRun(BaseModel):
    """A complete evaluation run — tracks one pipeline execution."""

    run_id: str = ""
    mode: EvalMode = EvalMode.reproduction
    version: str = ""
    paper_id: str = ""
    reproduction: ReproductionScore | None = None
    innovation: InnovationScore | None = None
    timestamp: float = Field(default_factory=time.time)
