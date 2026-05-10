from __future__ import annotations

from pathlib import Path

from backend.agents.orchestrator import ReproLabOrchestrator
from backend.hermes_audit.client import NousHermesClient
from backend.hermes_audit.memory import load_memory
from backend.hermes_audit.models import HermesAuditScope, HermesAuditStatus
from backend.hermes_audit.providers import extract_audit_json


class FakeProvider:
    def __init__(
        self,
        name: str,
        *,
        response: str = "",
        available: bool = True,
        raises: Exception | None = None,
        calls: list[str] | None = None,
    ) -> None:
        self.name = name
        self.response = response
        self.available = available
        self.raises = raises
        self.calls = calls if calls is not None else []

    def is_available(self) -> bool:
        return self.available

    def call(self, prompt: str) -> str:
        self.calls.append(self.name)
        if self.raises is not None:
            raise self.raises
        return self.response


def _audit_json(*, provider: str = "fake") -> str:
    return (
        "{"
        '"target":"paper",'
        '"scope":"step",'
        '"status":"grounded",'
        '"summary":"grounded",'
        '"recommended_intervention":"annotate",'
        '"confidence":"high",'
        f'"provider":"{provider}"'
        "}"
    )


def test_extract_audit_json_handles_fenced_balanced_and_prefix_shapes():
    fenced = '```json\n{"status":"grounded"}\n```'
    balanced = 'Here is the result: {"status":"caveat", "findings":["x"]} trailing'
    prefixed = 'JSON: {"status":"unsupported"}'

    assert extract_audit_json(fenced)["status"] == "grounded"
    assert extract_audit_json(balanced)["findings"] == ["x"]
    assert extract_audit_json(prefixed)["status"] == "unsupported"


def test_hermes_client_falls_back_persists_memory_and_prefers_last_good(tmp_path: Path):
    calls: list[str] = []
    bad = FakeProvider("bad", raises=RuntimeError("boom"), calls=calls)
    good = FakeProvider("good", response=_audit_json(provider="good"), calls=calls)

    client = NousHermesClient(providers=[bad, good], runs_root=tmp_path)
    report = client.audit(scope=HermesAuditScope.step, target="paper", payload={"x": 1})

    assert report.status == HermesAuditStatus.grounded
    assert report.provider == "good"
    assert calls == ["bad", "good"]

    memory = load_memory(tmp_path)
    assert memory.last_successful_provider == "good"
    assert memory.stats_for("bad").failures == 1
    assert memory.stats_for("good").successes == 1

    second_calls: list[str] = []
    second_client = NousHermesClient(
        providers=[
            FakeProvider("bad", raises=RuntimeError("boom"), calls=second_calls),
            FakeProvider("good", response=_audit_json(provider="good"), calls=second_calls),
        ],
        runs_root=tmp_path,
    )
    second = second_client.audit(scope=HermesAuditScope.step, target="paper", payload={})

    assert second.provider == "good"
    assert second_calls == ["good"]


def test_hermes_client_returns_unavailable_after_chain_exhaustion(tmp_path: Path):
    client = NousHermesClient(
        providers=[
            FakeProvider("missing", available=False),
            FakeProvider("bad_json", response="not json"),
        ],
        runs_root=tmp_path,
    )

    report = client.audit(scope=HermesAuditScope.checkpoint, target="gate_2", payload={})

    assert report.status == HermesAuditStatus.unavailable
    assert report.provider == "bad_json"
    memory = load_memory(tmp_path)
    assert memory.stats_for("missing").failures == 1
    assert memory.stats_for("bad_json").failures == 1


def test_orchestrator_default_hermes_client_uses_configured_runs_root(tmp_path: Path):
    orchestrator = ReproLabOrchestrator(project_id="prj_hermes", runs_root=tmp_path)

    client = orchestrator._hermes_audit_service.client

    assert client._runs_root == tmp_path
