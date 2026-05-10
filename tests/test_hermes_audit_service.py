from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

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
# NousHermesClient: Hermes Agent primary + Claude Code SDK fallback
# ---------------------------------------------------------------------------

_VALID_JSON_RESPONSE = json.dumps(
    {
        "target": "baseline-implementation",
        "scope": "step",
        "status": "grounded",
        "summary": "Claims are supported by evidence",
        "findings": ["Loss decreased"],
        "unsupported_claims": [],
        "evidence_refs": [],
        "recommended_intervention": "annotate",
        "corrective_note": "",
        "confidence": "high",
    }
)


def test_client_uses_hermes_agent_when_available():
    """Primary path: Hermes Agent (run_agent.AIAgent) produces a valid report."""
    client = NousHermesClient(model="openai/gpt-4o", api_key="sk-test")
    fake_agent = MagicMock()
    fake_agent.chat.return_value = _VALID_JSON_RESPONSE

    with patch("backend.hermes_audit.client.AIAgent", create=True) as mock_cls:
        # Patch the import inside _run_hermes_agent
        import backend.hermes_audit.client as client_mod
        original_run = client_mod.NousHermesClient._run_hermes_agent

        def patched_run(self, prompt):
            mock_cls.return_value = fake_agent
            agent = mock_cls(
                model="openai/gpt-4o",
                api_key="sk-test",
                provider="openai",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
            return str(agent.chat(prompt))

        with patch.object(NousHermesClient, "_run_hermes_agent", patched_run):
            report = client.audit(
                scope=HermesAuditScope.step,
                target="baseline-implementation",
                payload={"mode": "adapt"},
            )

    assert report.status == HermesAuditStatus.grounded
    assert report.provider == "nous-hermes"


def test_client_falls_back_to_claude_sdk_when_hermes_unavailable():
    """Fallback: when Hermes Agent fails, Claude Code SDK is used."""
    client = NousHermesClient(model="openai/gpt-4o", api_key="sk-test")

    with patch.object(
        NousHermesClient, "_run_hermes_agent", side_effect=RuntimeError("hermes not available")
    ), patch.object(
        NousHermesClient, "_run_claude_sdk", return_value=_VALID_JSON_RESPONSE
    ) as mock_sdk:
        report = client.audit(
            scope=HermesAuditScope.step,
            target="baseline-implementation",
            payload={"mode": "adapt"},
        )

    assert mock_sdk.called
    assert report.status == HermesAuditStatus.grounded
    assert report.provider == "claude-code-sdk"


def test_client_returns_unavailable_when_both_backends_fail():
    """Both Hermes and Claude SDK fail: returns unavailable report, not crash."""
    client = NousHermesClient(model="openai/gpt-4o", api_key="sk-test")

    with patch.object(
        NousHermesClient, "_run_hermes_agent", side_effect=RuntimeError("hermes down")
    ), patch.object(
        NousHermesClient, "_run_claude_sdk", side_effect=RuntimeError("sdk down")
    ):
        report = client.audit(
            scope=HermesAuditScope.step,
            target="baseline-implementation",
            payload={},
        )

    assert report.status == HermesAuditStatus.unavailable
    assert "hermes" in report.error_message
    assert "sdk" in report.error_message


def test_client_disabled_returns_unavailable_without_calling_backends():
    """When enabled=False, no backend is touched."""
    client = NousHermesClient(enabled=False)

    with patch.object(NousHermesClient, "_run_hermes_agent") as mock_hermes, \
         patch.object(NousHermesClient, "_run_claude_sdk") as mock_sdk:
        report = client.audit(
            scope=HermesAuditScope.step,
            target="test",
            payload={},
        )

    assert not mock_hermes.called
    assert not mock_sdk.called
    assert report.status == HermesAuditStatus.unavailable


def test_client_resolve_config_prefers_anthropic_key(monkeypatch):
    """_resolve_hermes_config prefers ANTHROPIC_API_KEY over OPENAI_API_KEY."""
    from backend.hermes_audit.client import _resolve_hermes_config

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai-test")
    model, key, provider = _resolve_hermes_config()
    assert provider == "anthropic"
    assert key == "sk-ant-test"
    assert "claude" in model


def test_client_resolve_config_falls_back_to_openai_key(monkeypatch):
    """_resolve_hermes_config uses OPENAI_API_KEY when ANTHROPIC is empty."""
    from backend.hermes_audit.client import _resolve_hermes_config

    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai-test")
    model, key, provider = _resolve_hermes_config()
    assert provider == "openai"
    assert key == "sk-oai-test"
    assert "gpt" in model
