"""Pydantic schemas for the budget-estimation feature.

Spec: docs/superpowers/specs/2026-05-25-budget-estimation-design.md
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ModelPriceEntry(BaseModel):
    """Pricing for one LLM model."""

    model_config = ConfigDict(extra="forbid")

    usd_per_1m_input: float = Field(ge=0.0)
    usd_per_1m_output: float = Field(ge=0.0)
    last_audited_utc: str
    subscription_note: str = ""
    subscription_usd_per_month: float | None = None
    assumed_runs_per_month: int | None = None

    @model_validator(mode="after")
    def _oauth_invariant(self) -> "ModelPriceEntry":
        if self.usd_per_1m_input == 0.0 and self.usd_per_1m_output == 0.0:
            if not self.subscription_note and self.subscription_usd_per_month is None:
                raise ValueError(
                    "Zero-price entries must have a non-empty subscription_note "
                    "or subscription_usd_per_month"
                )
        return self

    @model_validator(mode="after")
    def _subscription_amortization(self) -> "ModelPriceEntry":
        if self.subscription_usd_per_month is not None:
            if self.assumed_runs_per_month is None:
                raise ValueError(
                    "subscription_usd_per_month requires assumed_runs_per_month "
                    "for per-run amortization"
                )
        return self


class GpuPriceEntry(BaseModel):
    """Pricing for one RunPod GPU SKU."""

    model_config = ConfigDict(extra="forbid")

    usd_per_hour: float = Field(ge=0.0)
    last_audited_utc: str
    cloud_type: Literal["COMMUNITY", "SECURE", "ONDEMAND"]


class ApiCostBreakdown(BaseModel):
    """Per-provider cost row in the estimate response."""

    provider: str
    model_id: str
    input_tokens: int
    output_tokens: int
    usd: float
    is_subscription: bool
    subscription_note: str | None = None


class RecipeEstimate(BaseModel):
    """One recipe variant (strict or compressed)."""

    label: str
    description: str
    gpu_usd: float
    api_usd_best: float
    api_usd_worst: float
    wall_clock_hours_p50: float
    fidelity_label: Literal["high", "claim-match"]
    declared_reductions: list[str] = Field(default_factory=list)


class GpuEstimate(BaseModel):
    """GPU cost section of the estimate response."""

    sku_id: str
    label: str
    usd_per_hour: float
    estimated_hours: dict[str, float]
    usd_total: dict[str, float]
    # PR-ε.6: ensemble sigma and confidence badge (optional for backward compat)
    estimated_hours_sigma: float | None = None
    low_confidence: bool = False


class CalibrationMetadata(BaseModel):
    """Calibration provenance appended to every estimate."""

    based_on_n_preserved_runs: int
    precision_window_pct: int
    catalog_schema_version: int
    calibration_schema_version: int
    estimated_at_utc: str


class SourceBreakdown(BaseModel):
    """One source's contribution to the ensemble estimate."""

    source: str        # "heuristic" | "knn" | "llm"
    mean: float        # mean wall-clock hours from this source
    sigma: float       # std dev (inf = unavailable)
    weight: float      # normalised inverse-variance weight
    n_samples: int = 0 # 0 for model-based sources


class PaperBudgetEstimate(BaseModel):
    """Full response body for POST /paper/estimate."""

    model_config = ConfigDict(extra="ignore")  # forward-compat: ignore unknown keys

    paper: dict
    gpu: GpuEstimate
    api: list[ApiCostBreakdown]
    recipes: dict[str, RecipeEstimate]
    calibration_metadata: CalibrationMetadata

    # PR-ε.6: source breakdown for the "How this estimate was made" UI section.
    # Optional: old cached estimates return [] here.
    estimate_breakdown: list[SourceBreakdown] = Field(default_factory=list)

    # Cache / coupling key passed back to the client so Begin can forward it.
    estimate_id: str
