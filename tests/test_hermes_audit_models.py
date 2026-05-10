from __future__ import annotations

from pathlib import Path

from backend.hermes_audit.models import (
    HermesAuditConfidence,
    HermesAuditReport,
    HermesAuditScope,
    HermesAuditStatus,
    HermesEvidenceRef,
    HermesInterventionType,
)
from backend.hermes_audit.storage import HermesAuditStorage


def test_storage_persists_step_and_checkpoint_reports(tmp_path: Path):
    storage = HermesAuditStorage(tmp_path, "prj_hermes")
    step_report = HermesAuditReport(
        target="paper-understanding",
        scope=HermesAuditScope.step,
        status=HermesAuditStatus.grounded,
        summary="Grounded extraction",
        evidence_refs=[HermesEvidenceRef(kind="trace", snippet="Claim matches source")],
        recommended_intervention=HermesInterventionType.annotate,
        confidence=HermesAuditConfidence.high,
    )
    checkpoint_report = HermesAuditReport(
        target="gate_2",
        scope=HermesAuditScope.checkpoint,
        status=HermesAuditStatus.caveat,
        summary="One unsupported metric summary",
        recommended_intervention=HermesInterventionType.downgrade_claim,
    )

    step_path = storage.save_report(step_report)
    checkpoint_path = storage.save_report(checkpoint_report)
    index = storage.load_index()

    assert step_path.exists()
    assert checkpoint_path.exists()
    assert "step:paper-understanding" in index
    assert "checkpoint:gate_2" in index
    assert index["checkpoint:gate_2"]["status"] == "caveat"


def test_storage_uses_portable_report_filenames(tmp_path: Path):
    storage = HermesAuditStorage(tmp_path, "prj_hermes")
    report = HermesAuditReport(
        target="improvement-path:path_1",
        scope=HermesAuditScope.step,
        status=HermesAuditStatus.grounded,
        summary="Grounded improvement",
        recommended_intervention=HermesInterventionType.annotate,
    )

    report_path = storage.save_report(report)
    index = storage.load_index()

    assert ":" not in report_path.name
    assert report_path.exists()
    assert index["step:improvement-path:path_1"]["path"].endswith(report_path.name)
