from __future__ import annotations

from pathlib import Path

from backend.hermes_audit.models import (
    HermesAuditConfidence,
    HermesAuditReport,
    HermesAuditScope,
    HermesAuditStatus,
    HermesInterventionType,
)
from backend.hermes_audit.payloads import build_checkpoint_audit_payload, build_step_audit_payload
from backend.hermes_audit.service import HermesAuditService
from backend.hermes_audit.storage import HermesAuditStorage


class FakeClient:
    def __init__(self, report: HermesAuditReport):
        self.report = report
        self.calls: list[tuple[str, dict]] = []

    def audit(self, *, scope: HermesAuditScope, target: str, payload: dict) -> HermesAuditReport:
        self.calls.append((target, payload))
        return self.report.model_copy(update={"target": target, "scope": scope})


def test_step_payload_captures_trace_and_artifacts():
    payload = build_step_audit_payload(
        project_id="prj_hermes",
        target="baseline-implementation",
        state_snapshot={"stage": "baseline_implemented"},
        structured_output={"mode": "adapt"},
        trace_text="Agent said it changed train.py",
        artifact_paths=["runs/prj_hermes/code/train.py"],
    )

    assert payload["trace_text"] == "Agent said it changed train.py"
    assert payload["artifact_paths"] == ["runs/prj_hermes/code/train.py"]
    assert payload["structured_output"]["mode"] == "adapt"


def test_service_persists_audit_reports(tmp_path: Path):
    storage = HermesAuditStorage(tmp_path, "prj_hermes")
    client = FakeClient(
        HermesAuditReport(
            target="placeholder",
            scope=HermesAuditScope.step,
            status=HermesAuditStatus.grounded,
            summary="Looks good",
            recommended_intervention=HermesInterventionType.annotate,
            confidence=HermesAuditConfidence.high,
        )
    )
    service = HermesAuditService(client=client, storage=storage)
    payload = build_checkpoint_audit_payload(
        project_id="prj_hermes",
        target="gate_2",
        state_snapshot={"stage": "gate_2_passed"},
        evidence_bundle={"metrics": {"reward": 500}},
        trace_text="Verifier said reward is supported",
        artifact_paths=["runs/prj_hermes/baseline/metrics.json"],
    )

    report = service.audit(scope=HermesAuditScope.checkpoint, target="gate_2", payload=payload)
    index = storage.load_index()

    assert report.target == "gate_2"
    assert client.calls
    assert "checkpoint:gate_2" in index
