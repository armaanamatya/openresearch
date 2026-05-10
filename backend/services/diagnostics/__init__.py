"""Failure diagnosis and recovery taxonomy."""

from backend.services.diagnostics.model import FailureEvent, FailureKind
from backend.services.diagnostics.service import FailureDiagnosisService, failure_id_for

__all__ = [
    "FailureDiagnosisService",
    "FailureEvent",
    "FailureKind",
    "failure_id_for",
]
