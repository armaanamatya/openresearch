"""Budget estimation service for paper reproduction runs.

Public surface:
  estimate_paper_budget — async driver, returns PaperBudgetEstimate-shaped dict
  PaperBudgetEstimate, RecipeEstimate, ApiCostBreakdown — Pydantic models
  CATALOG_SCHEMA_VERSION, MODEL_PRICING, GPU_PRICING — pricing tables
"""

from backend.services.pricing.catalog import (
    CATALOG_SCHEMA_VERSION,
    GPU_PRICING,
    MODEL_PRICING,
    check_audit_freshness,
)
from backend.services.pricing.estimator import estimate_paper_budget
from backend.services.pricing.schemas import (
    ApiCostBreakdown,
    CalibrationMetadata,
    GpuEstimate,
    GpuPriceEntry,
    ModelPriceEntry,
    PaperBudgetEstimate,
    RecipeEstimate,
)

__all__ = [
    "estimate_paper_budget",
    "PaperBudgetEstimate",
    "RecipeEstimate",
    "ApiCostBreakdown",
    "CalibrationMetadata",
    "GpuEstimate",
    "ModelPriceEntry",
    "GpuPriceEntry",
    "CATALOG_SCHEMA_VERSION",
    "MODEL_PRICING",
    "GPU_PRICING",
    "check_audit_freshness",
]
