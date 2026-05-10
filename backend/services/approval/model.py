"""Approval policy and checkpoint models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ApprovalAction = Literal[
    "dataset_download",
    "long_run",
    "gpu_spend",
    "unofficial_repo",
    "substitute_dataset",
    "high_risk_assumption",
    "unknown_license",
    "sandbox_network",
    "external_upload",
    "untrusted_code",
]

ApprovalState = Literal["pending", "approved", "rejected"]
ApprovalRisk = Literal["low", "medium", "high", "critical"]


class ApprovalPolicy(BaseModel):
    """Configurable human-in-the-loop thresholds.

    Defaults match the Phase 2/MVP posture: small local CPU runs may proceed,
    anything with non-trivial cost, network risk, or trust ambiguity pauses.
    """

    model_config = ConfigDict(frozen=True)

    max_dataset_download_gb_without_approval: float = 2.0
    max_runtime_minutes_without_approval: int = 30
    max_gpu_cost_without_approval_usd: float = 0.0
    allow_unofficial_repos: bool = False
    allow_network_during_build: bool = True
    allow_network_during_run: bool = False
    allow_external_data_for_improvements: bool = False
    require_approval_for_unknown_license: bool = True


class ApprovalEvaluation(BaseModel):
    model_config = ConfigDict(frozen=True)

    action: ApprovalAction
    requires_approval: bool
    reason: str
    risk: ApprovalRisk = "medium"
    policy_snapshot: ApprovalPolicy
    metadata: dict[str, Any] = Field(default_factory=dict)


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    approval_id: str
    project_id: str
    action: ApprovalAction
    label: str
    details: str
    state: ApprovalState = "pending"
    risk: ApprovalRisk = "medium"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    resolved_at: datetime | None = None
    resolved_by: str = ""
    resolution_note: str = ""


__all__ = [
    "ApprovalAction",
    "ApprovalEvaluation",
    "ApprovalPolicy",
    "ApprovalRequest",
    "ApprovalRisk",
    "ApprovalState",
]
