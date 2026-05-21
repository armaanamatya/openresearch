"""Real-rlm integration harness for run_pipeline_rlm (Phase 3, Issue #60).

This test exercises the FULL orchestrator path end-to-end:

    run_pipeline_rlm
      → RLM(backend=<stub>, custom_tools=build_stub_custom_tools(...))
      → ReproLabRLMLogger.log() (sanitize + checkpoint + emit)
      → IterationCheckpointer.record() (SQLite event store + snapshot JSONL)
      → build_final_report() + write_final_report_rlm()
      → RLMRunResult

The rlm model backend is made DETERMINISTIC by monkeypatching
``rlm.core.rlm.get_client``:

    rlm.core.rlm imports ``get_client`` at module level (line 6):
        from rlm.clients import BaseLM, get_client

    RLM.__init__ calls ``get_client(self.backend, self.backend_kwargs)``
    at lines 197 and 202 to build the root and sub-call LM clients.

    We replace the module-level name ``rlm.core.rlm.get_client`` with a
    factory that returns a ``ScriptedLM`` — a concrete ``BaseLM`` subclass
    whose ``completion()`` replays a fixed two-turn script.  This is the
    only viable injection point: the rlm library offers no dependency-
    injection seam for the client, and ``LMHandler`` wraps the client
    opaquely after it is built.  Monkeypatching ``rlm.core.rlm.get_client``
    is therefore the narrowest correct patch.

Turn 1: the scripted model writes a REPL code block that calls two stub
primitives (``understand_section`` and ``extract_hyperparameters``) and
builds a ``report`` dict, then JSON-encodes it into ``report_json``.
Turn 2+: it emits ``FINAL_VAR("report_json")``, satisfying rlm's
``find_final_answer`` termination rule.

End-to-end corpus-leak (C2) assertion: the mock corpus embeds a unique
sentinel string (``PAPER_CORPUS_SENTINEL_e2e_xyzzy``) whose absence from
all persisted/streamed outputs is asserted after the run — proving the
``sanitize_iteration`` chokepoint works across the full pipeline.

Design spec: §14 (test plan) of
  docs/superpowers/specs/2026-05-21-rlm-phase3-orchestrator-design.md.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary

# ---------------------------------------------------------------------------
# Sentinel — embedded in the corpus; must NEVER appear in any output
# ---------------------------------------------------------------------------

_SENTINEL = "PAPER_CORPUS_SENTINEL_e2e_xyzzy"

# ---------------------------------------------------------------------------
# Scripted deterministic LM backend
# ---------------------------------------------------------------------------


class _ScriptedLM(BaseLM):
    """Deterministic fake root / sub-call model.

    Ignores its prompt; replays a fixed two-turn script:

    Turn 1 — write REPL code that:
      * calls ``understand_section`` and ``extract_hyperparameters`` on a
        small literal string (NOT on ``context`` — that is the Algorithm-2
        rule the test also implicitly checks);
      * builds a minimal ``report`` dict;
      * JSON-encodes it into ``report_json``;

    Turn 2+ — emit ``FINAL_VAR("report_json")`` to terminate the run.

    This is the exact pattern demonstrated in the Phase-2 spike
    (``tools/rlms_spike.py``, ``_make_scripted_lm``).
    """

    # Turn-1 REPL code: calls stub primitives on a literal slice, builds a
    # report dict, JSON-encodes it.  Deliberately uses a literal string
    # rather than ``context[...]`` so the corpus never flows through
    # primitive outputs — mirroring correct Algorithm-2 usage.
    _TURN1 = (
        "I will call the domain primitives on a short representative slice "
        "and assemble the reproduction report.\n"
        "```repl\n"
        "import json\n"
        "slice_text = 'Algorithm X trains for 10 epochs on MockEnv-v1.'\n"
        "claims = understand_section(slice_text)\n"
        "hp = extract_hyperparameters(slice_text)\n"
        "report = {\n"
        "    'verdict': 'partial',\n"
        "    'reproduction_summary': 'Stub run completed via scripted model.',\n"
        "    'baseline_metrics': {},\n"
        "    'paper_claims': {'accuracy': claims.get('metrics', [])},\n"
        "    'rubric': {'overall_score': 0.5, 'meets_target': False, 'areas': []},\n"
        "    'improvements': [],\n"
        "    'primitive_trace': {'hp': hp},\n"
        "    'cost': {'llm_usd': 0.0, 'root': 0.0, 'sub': 0.0, 'primitives': 0.0},\n"
        "    'iterations': 1,\n"
        "    'paper': {'id': 'stub-001', 'title': 'Mock Paper'},\n"
        "}\n"
        "report_json = json.dumps(report)\n"
        "```\n"
    )
    _TURN2 = 'All done.\nFINAL_VAR("report_json")'

    def __init__(self) -> None:
        super().__init__(model_name="scripted-stub")
        self.calls: int = 0

    def completion(self, prompt: str | dict[str, Any]) -> str:  # noqa: ANN001
        self.calls += 1
        return self._TURN1 if self.calls == 1 else self._TURN2

    async def acompletion(self, prompt: str | dict[str, Any]) -> str:  # noqa: ANN001
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
# Fixture: monkeypatch rlm.core.rlm.get_client
# ---------------------------------------------------------------------------


@pytest.fixture
def scripted_rlm_backend():
    """Monkeypatch ``rlm.core.rlm.get_client`` to return a ScriptedLM.

    ``rlm.core.rlm`` imports ``get_client`` at module level and calls it
    inside ``RLM.__init__`` (lines 197, 202) to build the root and sub-call
    clients.  Replacing the module-level name before ``RLM(...)`` is
    constructed is the minimal, least-invasive injection point.  The patch
    is scoped to this fixture and restored automatically at teardown.
    """
    lm_instance = _ScriptedLM()

    with patch("rlm.core.rlm.get_client", return_value=lm_instance):
        yield lm_instance


# ---------------------------------------------------------------------------
# Fixture: test database pointing at a temp file
# ---------------------------------------------------------------------------


@pytest.fixture
def integration_db_url(tmp_path: Path) -> str:
    """Point the SQLite event store at an isolated temp file."""
    return f"sqlite:///{tmp_path / 'integration_test.db'}"


# ---------------------------------------------------------------------------
# Fixture: mock workspace_claim_map with the corpus sentinel embedded
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace_claim_map() -> dict:
    """A tiny two-entry claim map.

    The sentinel is embedded in ``excerpt`` to prove the corpus reaches
    ``_build_context`` — but must never escape sanitization.
    """
    return {
        "project_id": "integration-test-001",
        "entries": [
            {
                "source_id": "s1",
                "title": "Abstract",
                "excerpt": (
                    f"Algorithm X achieves mean_reward 200 on MockEnv-v1. "
                    f"Sentinel: {_SENTINEL}."
                ),
            },
            {
                "source_id": "s2",
                "title": "Method",
                "excerpt": "We train for 10 epochs with lr=3e-4.",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


class TestRunPipelineRlmIntegration:
    """End-to-end integration harness for run_pipeline_rlm.

    Uses a scripted (deterministic) rlm model backend, stub primitives
    (REPROLAB_RLM_STUB_PRIMITIVES=1), and an isolated temp directory.
    No network, no Docker, no real LLM calls.
    """

    def test_full_run_green(
        self,
        tmp_path: Path,
        workspace_claim_map: dict,
        scripted_rlm_backend: _ScriptedLM,
        integration_db_url: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Assert every acceptance criterion from spec §14 is satisfied."""

        # --- Environment setup -----------------------------------------------
        # Force stub primitives so _resolve_custom_tools never tries binding.py
        monkeypatch.setenv("REPROLAB_RLM_STUB_PRIMITIVES", "1")

        # Force a known root model so resolve_root_model is deterministic
        # regardless of which API keys the test runner happens to have set.
        monkeypatch.setenv("REPROLAB_RLM_ROOT_MODEL", "gpt-5")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-not-used")  # satisfy fail-fast check

        # The production build_system_prompt is used UNCHANGED. It returns a
        # valid rlm .format() template — one {custom_tools_section} slot, every
        # other brace escaped — so rlm's build_rlm_system_prompt can .format()
        # it. This harness therefore also regression-tests that the real system
        # prompt survives rlm's .format() round-trip.

        # Point the event store at the isolated temp DB — patch get_settings()
        # directly in the run module (that is where SqliteEventStore is built).
        from backend.config import Settings
        fake_settings = Settings(
            database_url=integration_db_url,
            anthropic_api_key="fake-key-not-used",
        )
        monkeypatch.setattr(
            "backend.agents.rlm.run.get_settings",
            lambda: fake_settings,
        )

        project_id = "integration-test-001"
        runs_root = tmp_path / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)

        # --- Run the orchestrator --------------------------------------------
        from backend.agents.rlm.run import run_pipeline_rlm

        result = asyncio.run(
            run_pipeline_rlm(
                project_id=project_id,
                runs_root=runs_root,
                workspace_claim_map=workspace_claim_map,
                model="gpt-5",
                provider="anthropic",   # non-openai so _build_llm_client takes the Claude path
            )
        )

        project_dir = runs_root / project_id

        # --- 1. RLMRunResult is valid -----------------------------------------
        assert result.project_id == project_id
        assert result.status in {"completed", "partial", "failed"}
        # The scripted model produces "partial" in the report dict.
        assert result.status in {"partial", "failed"}

        # --- 2. Algorithm-1 loop ran (≥1 iteration) --------------------------
        # The scripted model fires two completion() calls before FINAL_VAR;
        # rlm counts iterations, not turns, so we may see 1 or 2.
        assert scripted_rlm_backend.calls >= 1, (
            "ScriptedLM.completion() was never called — rlm did not start the loop"
        )

        # --- 3. dashboard_events.jsonl exists and has repl_iteration + run_complete
        events_path = project_dir / "dashboard_events.jsonl"
        assert events_path.exists(), "dashboard_events.jsonl was not written"

        events: list[dict] = []
        for line in events_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                events.append(json.loads(line))

        event_types = [e.get("event") for e in events]
        assert "repl_iteration" in event_types, (
            f"'repl_iteration' event missing from dashboard_events.jsonl; "
            f"found: {event_types}"
        )
        assert "run_complete" in event_types, (
            f"'run_complete' event missing from dashboard_events.jsonl; "
            f"found: {event_types}"
        )

        # --- 4. rlm_state/iterations.jsonl has ≥1 sanitized iteration --------
        iterations_jsonl = project_dir / "rlm_state" / "iterations.jsonl"
        assert iterations_jsonl.exists(), "rlm_state/iterations.jsonl was not written"

        iter_records: list[dict] = []
        for line in iterations_jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                iter_records.append(json.loads(line))

        assert len(iter_records) >= 1, (
            "iterations.jsonl is empty — no iterations were checkpointed"
        )

        # --- 5. SQLite event store has ≥1 RLMRunIteration --------------------
        from backend.eventstore.sqlite_store import SqliteEventStore
        from backend.agents.rlm.checkpoint import RLMRunIteration

        store = SqliteEventStore(integration_db_url)
        stored_events = list(
            store.load(
                aggregate_id=f"rlm-run:{project_id}",
                from_version=0,
            )
        )
        rlm_iter_stored = [
            ev for ev in stored_events
            if ev.event_type == RLMRunIteration.event_type
        ]
        assert len(rlm_iter_stored) >= 1, (
            f"No RLMRunIteration events in the event store; "
            f"stored {len(stored_events)} events total"
        )
        store.close()

        # --- 6. final_report.json exists and has a valid verdict -------------
        final_json_path = project_dir / "final_report.json"
        assert final_json_path.exists(), "final_report.json was not written"

        final_report = json.loads(final_json_path.read_text(encoding="utf-8"))
        assert "verdict" in final_report, "final_report.json has no 'verdict' field"
        assert final_report["verdict"] in {"reproduced", "partial", "failed"}, (
            f"Unexpected verdict: {final_report['verdict']!r}"
        )

        # --- 7. final_report.md exists ---------------------------------------
        final_md_path = project_dir / "final_report.md"
        assert final_md_path.exists(), "final_report.md was not written"
        assert final_md_path.stat().st_size > 0, "final_report.md is empty"

        # --- 8. RLMRunResult.final_report_path points to the JSON file -------
        assert result.final_report_path is not None
        assert Path(result.final_report_path).exists()

        # --- 9. End-to-end C2 assertion: sentinel NEVER appears anywhere -----
        # The sentinel string is in the corpus excerpt.  sanitize_iteration
        # must strip it completely from every persisted output.
        _assert_no_sentinel(events_path.read_text(encoding="utf-8"), "dashboard_events.jsonl")
        _assert_no_sentinel(
            iterations_jsonl.read_text(encoding="utf-8"),
            "rlm_state/iterations.jsonl",
        )
        # Also check the event store payload text (StoredEvent.payload is a dict)
        for stored_ev in rlm_iter_stored:
            payload_text = json.dumps(stored_ev.payload)
            _assert_no_sentinel(payload_text, "SQLite event store payload")

    def test_relative_runs_root_is_resolved_absolute(
        self,
        tmp_path: Path,
        workspace_claim_map: dict,
        scripted_rlm_backend: _ScriptedLM,
        integration_db_url: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A relative ``runs_root`` must be resolved to an absolute path.

        Primitives execute inside the RLM REPL, whose working directory is not
        the repo root.  If ``runs_root`` stays relative, every primitive
        artifact write (``dashboard_events.jsonl``, ``code/`` ...) fails with
        ``FileNotFoundError``.  ``run_pipeline_rlm`` must resolve ``runs_root``
        at entry so every artifact path is CWD-independent.
        """
        monkeypatch.setenv("REPROLAB_RLM_STUB_PRIMITIVES", "1")
        monkeypatch.setenv("REPROLAB_RLM_ROOT_MODEL", "gpt-5")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-not-used")

        from backend.config import Settings

        fake_settings = Settings(
            database_url=integration_db_url,
            anthropic_api_key="fake-key-not-used",
        )
        monkeypatch.setattr(
            "backend.agents.rlm.run.get_settings", lambda: fake_settings
        )

        # CWD is tmp_path; runs_root is passed RELATIVE. Pre-fix it would stay
        # relative and every artifact path would be CWD-dependent.
        monkeypatch.chdir(tmp_path)

        from backend.agents.rlm.run import run_pipeline_rlm

        result = asyncio.run(
            run_pipeline_rlm(
                project_id="rel-runs-root-test",
                runs_root=Path("relative_runs"),
                workspace_claim_map=workspace_claim_map,
                model="gpt-5",
                provider="anthropic",
            )
        )

        # The fix: run_pipeline_rlm resolves runs_root → all artifact paths
        # are absolute and CWD-independent.
        assert result.final_report_path is not None
        assert Path(result.final_report_path).is_absolute(), (
            f"final_report_path is relative ({result.final_report_path!r}); "
            f"run artifacts are CWD-dependent and primitives running in the "
            f"RLM REPL's working directory will fail with FileNotFoundError"
        )
        # Artifacts landed at the resolved absolute location.
        resolved = tmp_path / "relative_runs" / "rel-runs-root-test"
        assert (resolved / "final_report.json").exists()
        assert (resolved / "dashboard_events.jsonl").exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_no_sentinel(text: str, source_label: str) -> None:
    """Assert the corpus sentinel does not appear anywhere in ``text``."""
    assert _SENTINEL not in text, (
        f"CORPUS LEAK (C2 violation): sentinel {_SENTINEL!r} "
        f"found in {source_label}.\n"
        f"  This means sanitize_iteration did not strip all locals/context values."
    )
