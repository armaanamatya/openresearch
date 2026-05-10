"""Reproducibility scoring models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


AssessmentVerdict = Literal["verified", "caveated", "failed"]


class DynamicThresholdAssessment(BaseModel):
    model_config = ConfigDict(frozen=True)

    verification_item: str
    assessed_complexity: Literal["low", "medium", "high"]
    assessed_risk: Literal["low", "medium", "high", "critical"]
    evidence_quality: Literal["weak", "medium", "strong"]
    dynamic_threshold: float
    actual_confidence: float
    verdict: AssessmentVerdict
    threshold_reasoning: str


class ReproducibilityScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    environment_recovered: float = Field(ge=0, le=100)
    method_fidelity: float = Field(ge=0, le=100)
    data_pipeline_confidence: float = Field(ge=0, le=100)
    metric_validity: float = Field(ge=0, le=100)
    artifact_completeness: float = Field(ge=0, le=100)
    composite: float = Field(ge=0, le=100)
    assumption_risk: Literal["low", "medium", "high", "critical"] = "low"
    overall_status: str
    dynamic_thresholds: tuple[DynamicThresholdAssessment, ...] = ()
    caveats: tuple[str, ...] = ()
    blocking_issues: tuple[str, ...] = ()
    generated_at: datetime


__all__ = ["DynamicThresholdAssessment", "ReproducibilityScore"]
