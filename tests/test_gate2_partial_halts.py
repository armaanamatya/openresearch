"""Contract test for Gate 2 supervisor verdict handling — Option D (Q1).

Option D inverts the Option-B semantics: ``GateStatus.partial_reproduction``
at Gate 2 is no longer a "salvageable, try improvements" verdict but a
terminal supervisor halt — equivalent to ``blocked_requires_human``,
``failed_reproduction``, and ``invalid_claim``. The supervisor gate becomes
purely binary: pass (verified / verified_with_caveats) or halt.

Both halting verdicts run through the same code path: the outer Gate 2
result block in ``orchestrator.run()`` calls ``_finalize_partial`` to write
``final_report.{json,md}`` and advances the stage to COMPLETE so
``finalize_benchmark`` in ``live_runs.py`` sees terminal state.

This file pins both halves of the contract:

  1. partial_reproduction → halt, ``_finalize_partial`` called once,
     ``run_improvements`` NOT called, final stage is COMPLETE.
  2. blocked_requires_human → identical halt behavior (defense-in-depth so
     a future refactor can't accidentally split the two cases apart).

The Track 4 fail-soft for un-buildable environments stays as it was —
it's an environment-build signal, not a supervisor verdict.

See ``docs/design/option-d-q1q2-refactor.md`` for the design.
``docs/design/option-b-investigation.md`` has the (now superseded)
historical context.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.agents.orchestrator import (
    PipelineStage,
    PipelineState,
    ReproLabOrchestrator,
)
from backend.agents.schemas import GateDecision, GateStatus


def _orch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, project_id: str
) -> ReproLabOrchestrator:
    monkeypatch.setattr(
        "backend.agents.orchestrator.get_settings",
        lambda: SimpleNamespace(
            rubric_verifier_enabled=False,  # rubric off — focus on gate-2 decision
            rubric_max_improvement_iterations=0,
            rubric_target_score=0.7,
            rubric_verifier_model="",
            environment_build_validation_enabled=False,
            environment_build_max_attempts=1,
        ),
    )
    return ReproLabOrchestrator(project_id, tmp_path, runtime=object())


def _seed_state_at_baseline_run(project_id: str) -> PipelineState:
    """Seed state to the point right before run_gate_2 fires.

    Track 4 must NOT fire: keep environment_build_ok True so the branch
    under test (the supervisor halt) is the one that matches.
    """
    state = PipelineState(project_id=project_id)
    state.stage = PipelineStage.BASELINE_RUN
    state.environment_build_ok = True
    state.environment_build_attempts = 0
    return state


def _install_stage_mocks(
    orch: ReproLabOrchestrator,
    monkeypatch: pytest.MonkeyPatch,
    gate_2_status: GateStatus,
    calls: dict[str, int],
) -> None:
    async def fake_run_gate_2(s: PipelineState) -> PipelineState:
        s.gate_2 = GateDecision(
            gate="gate_2",
            passed=False,
            status=gate_2_status,
        )
        s.advance_stage(PipelineStage.GATE_2_PASSED, orch.runs_root)
        return s

    async def fake_run_improvements(s: PipelineState, **kwargs) -> PipelineState:
        calls["improvements"] += 1
        s.advance_stage(PipelineStage.IMPROVEMENTS_RUN, orch.runs_root)
        return s

    async def fake_run_gate_3(s: PipelineState) -> PipelineState:
        calls["gate_3"] += 1
        s.gate_3 = GateDecision(
            gate="gate_3", passed=True, status=GateStatus.verified
        )
        s.advance_stage(PipelineStage.GATE_3_PASSED, orch.runs_root)
        return s

    async def fake_reiteration_loop(s: PipelineState, **kwargs) -> PipelineState:
        # rubric off in fixture; this is belt + suspenders.
        return s

    async def fake_generate_research_map(s: PipelineState) -> PipelineState:
        # If the test ever reaches this, the halt under test did not fire.
        raise AssertionError(
            "research_map must NOT run when Gate 2 supervisor verdict halts the run"
        )

    def fake_finalize_partial(s: PipelineState) -> None:
        calls["finalize_partial"] += 1
        s.advance_stage(PipelineStage.COMPLETE, orch.runs_root)

    monkeypatch.setattr(orch, "run_gate_2", fake_run_gate_2)
    monkeypatch.setattr(orch, "run_improvements", fake_run_improvements)
    monkeypatch.setattr(orch, "run_gate_3", fake_run_gate_3)
    monkeypatch.setattr(
        orch, "_run_improvement_reiteration_loop", fake_reiteration_loop
    )
    monkeypatch.setattr(orch, "generate_research_map", fake_generate_research_map)
    monkeypatch.setattr(orch, "_finalize_partial", fake_finalize_partial)


@pytest.mark.asyncio
async def test_gate2_partial_reproduction_now_halts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Option D (Q1): partial_reproduction at Gate 2 halts the run.

    Inverts the prior Option-B behavior. Improvements MUST NOT run;
    _finalize_partial writes the terminal artifact; final stage is COMPLETE.
    """
    orch = _orch(tmp_path, monkeypatch, project_id="prj_gate2_partial")
    state = _seed_state_at_baseline_run("prj_gate2_partial")
    state.save_checkpoint(tmp_path)

    calls = {"improvements": 0, "gate_3": 0, "finalize_partial": 0}
    _install_stage_mocks(
        orch,
        monkeypatch,
        gate_2_status=GateStatus.partial_reproduction,
        calls=calls,
    )

    final_state = await orch.run(resume=True)

    assert calls["improvements"] == 0, (
        "Option D (Q1): GateStatus.partial_reproduction must halt the pipeline; "
        "run_improvements was called when it should not have been"
    )
    assert calls["gate_3"] == 0, (
        "Gate 3 must not run after a Gate 2 halt"
    )
    assert calls["finalize_partial"] == 1, (
        "Gate 2 halt must call _finalize_partial exactly once so the terminal "
        "artifact lands regardless of rubric verifier state"
    )
    assert final_state.stage is PipelineStage.COMPLETE


@pytest.mark.asyncio
async def test_gate2_blocked_requires_human_still_halts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defense-in-depth: blocked_requires_human halts identically to partial.

    Under Option D, both verdicts share the same code path. This test pins
    that the halt behavior for the canonical "halt for human review" verdict
    is unchanged from before.
    """
    orch = _orch(tmp_path, monkeypatch, project_id="prj_gate2_blocked")
    state = _seed_state_at_baseline_run("prj_gate2_blocked")
    state.save_checkpoint(tmp_path)

    calls = {"improvements": 0, "gate_3": 0, "finalize_partial": 0}
    _install_stage_mocks(
        orch,
        monkeypatch,
        gate_2_status=GateStatus.blocked_requires_human,
        calls=calls,
    )

    final_state = await orch.run(resume=True)

    assert calls["improvements"] == 0, (
        "GateStatus.blocked_requires_human must halt the pipeline; "
        "run_improvements was called when it should not have been"
    )
    assert calls["finalize_partial"] == 1, (
        "Gate 2 halt must call _finalize_partial so the terminal artifact lands"
    )
    assert final_state.stage is PipelineStage.COMPLETE
