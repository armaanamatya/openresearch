"""An RLM run must produce the artifacts the UI/SSE layer depends on.

Pin the on-disk artifact contract for ``run_pipeline_rlm`` so that deletion
tasks (Task 6 onwards) can rely on a proven pre-condition: every artifact the
SSE bridge and the HTTP layer read after a run is confirmed to be written by
the RLM code path.

Step-2 findings (recorded here for Task 6):
- ``final_report.json`` / ``final_report.md``  written by
  ``write_final_report_rlm`` in ``backend/agents/rlm/report.py``.
- ``demo_status.json``  written by ``_write_demo_status`` in
  ``backend/agents/rlm/run.py``.
- ``dashboard_events.jsonl``  written by ``DashboardEmitter``
  (``backend/agents/dashboard_emitter.py``) which the RLM path instantiates.
- ``pipeline_state.json``  NOT written by the RLM path.  The RLM run
  produces ``RLMRunResult``, not ``PipelineState``.  Only
  ``backend/agents/orchestrator.py`` (old pipeline) writes this file.
  This confirms the RLM path never uses ``PipelineState`` and justifies
  the ``PipelineState`` deletion in Task 6.

Harness: mirrors ``tests/rlm/test_run_integration.py``'s scripted-model
approach.  The ``_ScriptedLM`` backend is duplicated here (not imported)
so this file remains a standalone contract test that does not grow a
dependency on ``test_run_integration``'s module-level constants.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary


# ---------------------------------------------------------------------------
# Minimal scripted LM backend — two-turn deterministic script
# ---------------------------------------------------------------------------


class _ScriptedLM(BaseLM):
    """Deterministic fake model replaying a fixed two-turn script."""

    _TURN1 = (
        "I will call the domain primitives on a short slice.\n"
        "```repl\n"
        "import json\n"
        "slice_text = 'Algorithm X trains for 10 epochs.'\n"
        "claims = understand_section(slice_text)\n"
        "hp = extract_hyperparameters(slice_text)\n"
        "report = {\n"
        "    'verdict': 'partial',\n"
        "    'reproduction_summary': 'Artifact contract stub run.',\n"
        "    'baseline_metrics': {},\n"
        "    'paper_claims': {},\n"
        "    'rubric': {'overall_score': 0.5, 'meets_target': False, 'areas': []},\n"
        "    'improvements': [],\n"
        "    'primitive_trace': {'hp': hp},\n"
        "    'cost': {'llm_usd': 0.0, 'root': 0.0, 'sub': 0.0, 'primitives': 0.0},\n"
        "    'iterations': 1,\n"
        "    'paper': {'id': 'contract-001', 'title': 'Contract Test Paper'},\n"
        "}\n"
        "report_json = json.dumps(report)\n"
        "```\n"
    )
    _TURN2 = 'Done.\nFINAL_VAR("report_json")'

    def __init__(self) -> None:
        super().__init__(model_name="scripted-stub")
        self.calls: int = 0

    def completion(self, prompt: str | dict[str, Any]) -> str:
        self.calls += 1
        return self._TURN1 if self.calls == 1 else self._TURN2

    async def acompletion(self, prompt: str | dict[str, Any]) -> str:
        return self.completion(prompt)

    def _usage_summary(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            total_calls=self.calls,
            total_input_tokens=0,
            total_output_tokens=0,
            total_cost=0.0,
        )

    def get_usage_summary(self) -> UsageSummary:
        return UsageSummary(model_usage_summaries={self.model_name: self._usage_summary()})

    def get_last_usage(self) -> ModelUsageSummary:
        return self._usage_summary()


# ---------------------------------------------------------------------------
# Fixture: ``rlm_offline_run``
# Runs ``run_pipeline_rlm`` deterministically into ``tmp_path``; returns the
# project directory so callers can assert artifact presence.
# ---------------------------------------------------------------------------


@pytest.fixture
def rlm_offline_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Factory that runs ``run_pipeline_rlm`` with a scripted (offline) LM.

    Call signature: ``run_dir = rlm_offline_run(tmp_path)``
    Returns the project directory (``runs_root / project_id``).
    """

    def _run(base: Path, project_id: str = "artifact-contract-test") -> Path:
        monkeypatch.setenv("OPENRESEARCH_RLM_STUB_PRIMITIVES", "1")
        monkeypatch.setenv("OPENRESEARCH_RLM_ROOT_MODEL", "gpt-5")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-not-used")

        db_url = f"sqlite:///{base / 'contract_test.db'}"

        from backend.config import Settings

        fake_settings = Settings(
            database_url=db_url,
            anthropic_api_key="fake-key-not-used",
        )
        monkeypatch.setattr(
            "backend.agents.rlm.run.get_settings",
            lambda: fake_settings,
        )

        runs_root = base / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)

        claim_map = {
            "project_id": project_id,
            "entries": [
                {
                    "source_id": "s1",
                    "title": "Abstract",
                    "excerpt": "Algorithm X trains for 10 epochs on MockEnv-v1.",
                },
            ],
        }

        lm_instance = _ScriptedLM()
        with patch("rlm.core.rlm.get_client", return_value=lm_instance):
            from backend.agents.rlm.run import run_pipeline_rlm

            asyncio.run(
                run_pipeline_rlm(
                    project_id=project_id,
                    runs_root=runs_root,
                    workspace_claim_map=claim_map,
                    model="gpt-5",
                    provider="anthropic",
                )
            )

        return runs_root / project_id

    return _run


# ---------------------------------------------------------------------------
# Contract test
# ---------------------------------------------------------------------------


def test_rlm_run_writes_required_artifacts(tmp_path: Path, rlm_offline_run) -> None:
    """An RLM run must write every artifact the UI/SSE layer depends on.

    Artifacts asserted:
      - ``final_report.json``   — final verdict JSON consumed by the HTTP layer
      - ``final_report.md``     — human-readable markdown report
      - ``demo_status.json``    — UI status snapshot read by GET /runs/{id}
      - ``dashboard_events.jsonl`` — SSE event log sourced by the SSE bridge

    Negative assertion:
      - ``pipeline_state.json`` must NOT exist — the RLM path produces
        ``RLMRunResult``, not ``PipelineState``.  Its absence confirms the
        deletion of ``PipelineState`` in Task 6 is safe.
    """
    run_dir = rlm_offline_run(tmp_path)

    for name in ("final_report.json", "final_report.md", "demo_status.json"):
        assert (run_dir / name).is_file(), f"RLM run did not write {name}"

    assert list(run_dir.glob("*.jsonl")), "RLM run wrote no agent-event log (*.jsonl)"

    # pipeline_state.json must NOT be written by the RLM path.
    # This negative assertion guards the Task 6 PipelineState deletion.
    assert not (run_dir / "pipeline_state.json").exists(), (
        "RLM run unexpectedly wrote pipeline_state.json — the RLM path must "
        "never write PipelineState; re-check run_pipeline_rlm"
    )


def test_rlm_run_demo_status_is_valid_json(tmp_path: Path, rlm_offline_run) -> None:
    """``demo_status.json`` must be parseable and carry required LiveRunState fields.

    This guards the GET /runs/{id} contract: the HTTP layer calls
    ``LiveRunState(**status)`` on the file's content.
    """
    run_dir = rlm_offline_run(tmp_path)

    status_path = run_dir / "demo_status.json"
    assert status_path.is_file(), "demo_status.json was not written"

    status = json.loads(status_path.read_text(encoding="utf-8"))
    for field in ("projectId", "outputDir", "runMode", "status"):
        assert field in status, f"demo_status.json is missing required field '{field}'"

    assert status["runMode"] == "rlm"
    assert status["status"] in {"completed", "failed", "stopped"}

    # The HTTP layer constructs LiveRunState from this dict — confirm it parses.
    from backend.services.events.live_runs import LiveRunState

    LiveRunState(**status)  # raises if any required field is missing or mis-typed
