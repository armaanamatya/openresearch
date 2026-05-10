from __future__ import annotations

from pathlib import Path

from backend.hermes_audit.client import NousHermesClient
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


# ---------------------------------------------------------------------------
# NousHermesClient enabled-flag contract.
#
# The pluggable provider chain (primary path / fallback / chain-exhaustion)
# is covered exhaustively by tests/test_hermes_audit_adapter.py; the
# Settings-driven key resolution is covered by
# tests/test_hermes_provider_settings.py. The only orthogonal contract
# left for this file is: when ``enabled=False``, no provider is reached
# and the report's ``provider`` field is the literal "disabled" sentinel.
# ---------------------------------------------------------------------------


class _CountingProvider:
    """Provider that counts every is_available()/call() invocation."""

    def __init__(self) -> None:
        self.name = "counting"
        self.is_available_calls = 0
        self.call_calls = 0

    def is_available(self) -> bool:
        self.is_available_calls += 1
        return True

    def call(self, prompt: str) -> str:
        self.call_calls += 1
        return '{"status":"grounded","summary":"x","recommended_intervention":"annotate"}'


def test_client_disabled_returns_unavailable_without_touching_providers(tmp_path: Path):
    counting = _CountingProvider()
    client = NousHermesClient(enabled=False, providers=[counting], runs_root=tmp_path)

    report = client.audit(
        scope=HermesAuditScope.step, target="baseline-implementation", payload={}
    )

    assert counting.is_available_calls == 0
    assert counting.call_calls == 0
    assert report.status == HermesAuditStatus.unavailable
    assert report.provider == "disabled"
