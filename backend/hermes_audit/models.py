"""Pydantic models for the Nous Hermes oversight layer."""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class HermesAuditScope(str, enum.Enum):
    step = "step"
    checkpoint = "checkpoint"


class HermesAuditStatus(str, enum.Enum):
    grounded = "grounded"
    caveat = "caveat"
    unsupported = "unsupported"
    unavailable = "unavailable"
    system_error = "system_error"


class HermesAuditConfidence(str, enum.Enum):
    low = "low"
    medium = "medium"
    high = "high"


class HermesInterventionType(str, enum.Enum):
    annotate = "annotate"
    retry_step = "retry_step"
    request_evidence = "request_evidence"
    downgrade_claim = "downgrade_claim"
    suppress_publication = "suppress_publication"
    escalate_human = "escalate_human"


class HermesEvidenceRef(BaseModel):
    kind: str
    path: str = ""
    snippet: str = ""
    description: str = ""


class HermesAuditReport(BaseModel):
    model_config = {"extra": "ignore"}

    target: str
    scope: HermesAuditScope
    status: HermesAuditStatus
    summary: str = ""
    findings: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)

    @field_validator("findings", "unsupported_claims", mode="before")
    @classmethod
    def _coerce_str_list(cls, v: Any) -> list[str]:
        """LLMs sometimes return dicts like {'claim': '...'} instead of plain strings."""
        if not isinstance(v, list):
            return v
        out: list[str] = []
        for item in v:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                out.append(item.get("claim") or item.get("description") or str(item))
            else:
                out.append(str(item))
        return out

    evidence_refs: list[HermesEvidenceRef] = Field(default_factory=list)
    recommended_intervention: HermesInterventionType = HermesInterventionType.annotate
    corrective_note: str = ""
    confidence: HermesAuditConfidence = HermesAuditConfidence.medium
    provider: str = "nous-hermes"
    raw_response: dict[str, Any] = Field(default_factory=dict)
    error_message: str = ""
