"""Contract tests for Gate 3 supervisor verdict handling — Option D (Q1 + Q2).

Under Option D:

  - The supervisor gate at Gate 3 is binary: any non-pass — partial,
    blocked, failed, invalid_claim — halts the run. The
    ``!= partial_reproduction`` exemption added in commit 263ebbb is
    reverted.
  - The unconditional pre-loop ``run_improvements + run_gate_3`` pair is
    deleted. The rubric-enabled path runs entirely through
    ``_run_improvement_reiteration_loop``. The loop seeds from
    ``baseline_verification`` on iteration 1.
  - One outer Gate 3 halt check in ``orchestrator.run()`` covers both the
    rubric-driven loop break path and the rubric-disabled single-round
    path (the rubric-disabled tests live in
    ``test_orchestrator_rubric_disabled_single_round.py``).

This file pins both gate-3 halting verdicts in the rubric-driven path:

  1. blocked_requires_human inside the loop → loop breaks, outer halt
     calls ``_finalize_partial`` exactly once, research_map does NOT run.
  2. partial_reproduction inside the loop → identical halt (same code
     path under Option D).

See ``docs/design/option-d-q1q2-refactor.md``. The historical
``option-c-gate3-result-handling.md`` documents the old partial-falls-
through behavior, now superseded.
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
from backend.agents.schemas import GateDecision, GateStatus, RubricVerification


def _orch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    project_id: str,
    *,
    rubric_max_improvement_iterations: int = 2,
) -> ReproLabOrchestrator:
    monkeypatch.setattr(
        "backend.agents.orchestrator.get_settings",
        lambda: SimpleNamespace(
            rubric_verifier_enabled=True,
            rubric_max_improvement_iterations=rubric_max_improvement_iterations,
            rubric_target_score=0.7,
            rubric_verifier_model="",
            environment_build_validation_enabled=False,
            environment_build_max_attempts=1,
        ),
    )
    return ReproLabOrchestrator(project_id, tmp_path, runtime=object())


def _verification(score: float, target: float, *, meets: bool | None = None) -> RubricVerification:
    return RubricVerification(
        overall_score=score,
        target_score=target,
        meets_target=meets if meets is not None else score >= target,
        rubric_source="generated",
    )


def _seed_state_with_baseline_below_target(project_id: str) -> PipelineState:
    """Seed state past run_baseline_run with a below-target baseline_verification.

    The widened ``_should_reiterate`` consults baseline_verification on the
    first loop iteration (improved is None at that point). Seeding it below
    target is what makes the loop fire under Option D's semantics.
    Track 4 is kept ON-RAILS (env_build_ok=True) so this test never hits
    the un-buildable fail-soft branch.
    """
    state = PipelineState(project_id=project_id)
    state.stage = PipelineStage.BASELINE_RUN
    state.environment_build_ok = True
    state.environment_build_attempts = 0
    state.baseline_verification = _verification(0.3, 0.7)
    state.verification_history = [state.baseline_verification]
    return state


def _install_common_mocks(
    orch: ReproLabOrchestrator,
    monkeypatch: pytest.MonkeyPatch,
    *,
    gate_3_status: GateStatus,
    calls: dict[str, int],
) -> None:
    async def fake_run_gate_2(s: PipelineState) -> PipelineState:
        # Gate 2 passes — we're testing Gate 3.
        s.gate_2 = GateDecision(
            gate="gate_2", passed=True, status=GateStatus.verified_with_caveats
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
            gate="gate_3", passed=False, status=gate_3_status
        )
        # Populate improved_verification so the verification_history-length
        # check inside the loop doesn't trip and mistake the gate non-pass
        # for a dead verifier.
        s.improved_verification = _verification(0.35, 0.7)
        s.verification_history.append(s.improved_verification)
        s.advance_stage(PipelineStage.GATE_3_PASSED, orch.runs_root)
        return s

    async def fake_research_map(s: PipelineState) -> PipelineState:
        raise AssertionError(
            "research_map must NOT run when Gate 3 supervisor verdict halts the run"
        )

    def fake_finalize_partial(s: PipelineState) -> None:
        calls["finalize_partial"] += 1
        s.advance_stage(PipelineStage.COMPLETE, orch.runs_root)

    monkeypatch.setattr(orch, "run_gate_2", fake_run_gate_2)
    monkeypatch.setattr(orch, "run_improvements", fake_run_improvements)
    monkeypatch.setattr(orch, "run_gate_3", fake_run_gate_3)
    monkeypatch.setattr(orch, "generate_research_map", fake_research_map)
    monkeypatch.setattr(orch, "_finalize_partial", fake_finalize_partial)


@pytest.mark.asyncio
async def test_gate3_blocked_in_loop_halts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gate 3 returns blocked_requires_human inside the rubric-driven loop.

    Expected: exactly one improvement round runs (loop iter 1), the inner
    Gate 3 break fires, the outer halt check calls _finalize_partial once,
    research_map never runs, final stage is COMPLETE.

    Critically: ``rubric_max_improvement_iterations=2`` gives the loop
    room for a second round, but the blocked verdict must stop it after
    the first. This is the wasted-LLM-cost protection that PR-equivalent
    commit 263ebbb introduced; Option D preserves it.
    """
    orch = _orch(
        tmp_path,
        monkeypatch,
        project_id="prj_gate3_blocked",
        rubric_max_improvement_iterations=2,
    )
    state = _seed_state_with_baseline_below_target("prj_gate3_blocked")
    state.save_checkpoint(tmp_path)

    calls = {"improvements": 0, "gate_3": 0, "finalize_partial": 0}
    _install_common_mocks(
        orch,
        monkeypatch,
        gate_3_status=GateStatus.blocked_requires_human,
        calls=calls,
    )

    final_state = await orch.run(resume=True)

    assert calls["improvements"] == 1, (
        f"expected exactly 1 improvement round before halt; got {calls['improvements']}"
    )
    assert calls["gate_3"] == 1, (
        f"expected exactly 1 Gate 3 call before halt; got {calls['gate_3']}"
    )
    assert calls["finalize_partial"] == 1, (
        "outer halt must call _finalize_partial exactly once"
    )
    assert final_state.stage is PipelineStage.COMPLETE


@pytest.mark.asyncio
async def test_gate3_partial_in_loop_halts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Option D (Q1): Gate 3 partial_reproduction inside the loop halts identically.

    Under Option D the supervisor gate is binary — partial_reproduction at
    Gate 3 is no longer "let the loop try another round." It halts on the
    first occurrence, same code path as blocked_requires_human.

    Consolidates the two old Option-C tests
    (``partial_on_first_call_falls_through_to_reiteration`` +
    ``partial_then_blocked_breaks_loop_and_halts``) into one — under
    Option D there's no chain to blocked, partial alone breaks the loop.
    """
    orch = _orch(
        tmp_path,
        monkeypatch,
        project_id="prj_gate3_partial",
        rubric_max_improvement_iterations=3,  # loop has room; expect early halt
    )
    state = _seed_state_with_baseline_below_target("prj_gate3_partial")
    state.save_checkpoint(tmp_path)

    calls = {"improvements": 0, "gate_3": 0, "finalize_partial": 0}
    _install_common_mocks(
        orch,
        monkeypatch,
        gate_3_status=GateStatus.partial_reproduction,
        calls=calls,
    )

    final_state = await orch.run(resume=True)

    assert calls["improvements"] == 1, (
        f"Option D (Q1): partial must halt after the first round, not loop further; "
        f"got {calls['improvements']} improvement rounds"
    )
    assert calls["gate_3"] == 1, (
        f"expected exactly 1 Gate 3 call before halt; got {calls['gate_3']}"
    )
    assert calls["finalize_partial"] == 1
    assert final_state.stage is PipelineStage.COMPLETE
