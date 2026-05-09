"""Verification record schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class VerificationStatus(str, Enum):
    verified = "verified"
    verified_with_caveats = "verified_with_caveats"
    partial_reproduction = "partial_reproduction"
    failed_reproduction = "failed_reproduction"
    blocked_requires_human = "blocked_requires_human"
    invalid_claim = "invalid_claim"


class VerificationRecord(BaseModel):
    verification_id: str
    run_id: str
    verifier_type: str
    status: VerificationStatus
    method_fidelity_score: Optional[float] = None
    environment_recovery_score: Optional[float] = None
    data_pipeline_confidence: Optional[float] = None
    artifact_completeness_score: Optional[float] = None
    caveats: list[str] = Field(default_factory=list)
    blocking_issues: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
