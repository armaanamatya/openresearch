"""Nous Hermes oversight layer for ReproLab."""

from backend.hermes_audit.client import NousHermesClient
from backend.hermes_audit.memory import (
    AdapterMemory,
    ProviderStats,
    load_memory,
    save_memory,
)
from backend.hermes_audit.models import (
    HermesAuditConfidence,
    HermesAuditReport,
    HermesAuditScope,
    HermesAuditStatus,
    HermesEvidenceRef,
    HermesInterventionType,
)
from backend.hermes_audit.payloads import build_checkpoint_audit_payload, build_step_audit_payload
from backend.hermes_audit.providers import (
    AuditProvider,
    ClaudeAuditProvider,
    NousHermesProvider,
    OpenAIAuditProvider,
    extract_audit_json,
)
from backend.hermes_audit.service import HermesAuditService
from backend.hermes_audit.storage import HermesAuditStorage

__all__ = [
    "AdapterMemory",
    "AuditProvider",
    "ClaudeAuditProvider",
    "NousHermesClient",
    "NousHermesProvider",
    "OpenAIAuditProvider",
    "ProviderStats",
    "HermesAuditConfidence",
    "HermesAuditReport",
    "HermesAuditScope",
    "HermesAuditStatus",
    "HermesEvidenceRef",
    "HermesInterventionType",
    "HermesAuditService",
    "HermesAuditStorage",
    "build_step_audit_payload",
    "build_checkpoint_audit_payload",
    "extract_audit_json",
    "load_memory",
    "save_memory",
]
