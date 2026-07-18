"""Shared fixtures for RLM primitive tests (Phase 2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.agents.dashboard_emitter import DashboardEmitter
from backend.agents.resilience.cost import RunCostLedger
from backend.agents.rlm.context import RunContext
from backend.agents.rlm.sse_bridge import make_emit


@pytest.fixture(autouse=True)
def _allow_lossy_stub_papers(monkeypatch):
    """RLM pipeline/mechanics tests run on STUB paper text (they never ingest a real
    paper). Allow the degraded-paper fallback here so the 2026-06-15 abort-by-default
    (``config.allow_lossy_paper_text=False`` — which stops a real run on near-empty
    context, the SDAR 469-char incident) doesn't block them. The abort behavior itself
    is covered by tests/services/ingestion/parser/test_precondition.py.
    """
    monkeypatch.setenv("OPENRESEARCH_ALLOW_LOSSY_PAPER_TEXT", "true")


@pytest.fixture(autouse=True)
def _disk_gate_host_independent(monkeypatch):
    """Pin the disk-headroom preflight OFF for RLM mechanics tests so the socket-
    hermetic suite is also DISK-hermetic. The full-run path reads
    ``OPENRESEARCH_MIN_DISK_GB`` (default 10) at run.py and aborts when the runs root
    is near-full — which on a low-disk dev host (the 2026-06-22 5.1 GB-free machine)
    fails ~10 run-pipeline tests that have nothing to do with disk safety. The gate's
    own behavior is covered directly in tests/rlm/test_disk_headroom.py, which calls
    ``_assert_disk_headroom(min_gb=...)`` explicitly and never reads this env var, so
    pinning it here cannot mask the gate's real coverage. A test that wants the gate
    active simply re-sets the env after this autouse fixture runs.
    """
    monkeypatch.setenv("OPENRESEARCH_MIN_DISK_GB", "0")


class FakeLlmClient:
    """Counting fake LlmClient. Returns scripted responses in order (last repeats)."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self.calls: list[dict] = []
        self._responses = responses or ["{}"]

    def complete(self, *, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        return self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]


@pytest.fixture
def make_context():
    """Factory fixture: build a RunContext rooted at a tmp dir."""

    def _make(tmp_path: Path, llm_responses: list[str] | None = None,
              project_id: str = "test_proj") -> RunContext:
        project_dir = tmp_path / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        dashboard = DashboardEmitter(project_id, tmp_path)
        return RunContext(
            project_id=project_id,
            project_dir=project_dir,
            runs_root=tmp_path,
            dashboard=dashboard,
            emit=make_emit(dashboard),
            cost_ledger=RunCostLedger.load_jsonl(
                project_dir / "cost_ledger.jsonl",
                project_id=project_id,
                attach_path=True,
            ),
            llm_client=FakeLlmClient(llm_responses),
            provider="anthropic",
            model="test-model",
        )

    return _make
