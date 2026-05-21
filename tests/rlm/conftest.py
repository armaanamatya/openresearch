"""Shared fixtures for RLM primitive tests (Phase 2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.agents.dashboard_emitter import DashboardEmitter
from backend.agents.resilience.cost import RunCostLedger
from backend.agents.rlm.context import RunContext


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
        return RunContext(
            project_id=project_id,
            project_dir=project_dir,
            runs_root=tmp_path,
            dashboard=DashboardEmitter(project_id, tmp_path),
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
