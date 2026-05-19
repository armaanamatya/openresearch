"""Contract tests for Gate 3 result handling (Option C).

After PRs #43 and #44, Gate 2's ``partial_reproduction`` falls through to the
improvement orchestration. Gate 3 (the supervisor's verdict on the improved
artifact) needs symmetric handling:

  - ``partial_reproduction`` → let the reiteration loop decide whether to
    keep trying (existing behavior).
  - ``blocked_requires_human`` / ``failed_reproduction`` / ``invalid_claim``
    → halt with ``_finalize_partial`` so the supervisor's halt-for-human
    semantic survives.

This test file pins both halves of the contract:

  1. blocked on the first ``run_gate_3`` → halt before the reiteration loop.
  2. partial on the first ``run_gate_3`` → continue (reiteration loop has
     a chance to fix the score).
  3. partial then blocked across iterations → the loop's inner break
     fires; the outer post-loop check halts the run.

See ``docs/design/option-c-gate3-result-handling.md``.
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
    rubric_verifier_enabled: bool = False,
    rubric_max_improvement_iterations: int = 0,
) -> ReproLabOrchestrator:
    monkeypatch.setattr(
        "backend.agents.orchestrator.get_settings",
        lambda: SimpleNamespace(
            rubric_verifier_enabled=rubric_verifier_enabled,
            rubric_max_improvement_iterations=rubric_max_improvement_iterations,
            rubric_target_score=0.7,
            rubric_verifier_model="",
            environment_build_validation_enabled=False,
            environment_build_max_attempts=1,
        ),
    )
    return ReproLabOrchestrator(project_id, tmp_path, runtime=object())


def _seed_state_past_baseline_run(project_id: str) -> PipelineState:
    """Seed state just past run_baseline_run so the pipeline's for-loop
    skips everything and lands on run_gate_2 first, then the improvement
    phase. Track 4 is kept ON-RAILS (env_build_ok=True) so this test
    never hits the un-buildable fail-soft branch."""
    state = PipelineState(project_id=project_id)
    state.stage = PipelineStage.BASELINE_RUN
    state.environment_build_ok = True
    state.environment_build_attempts = 0
    return state


def _verification(score: float, target: float, *, meets: bool | None = None) -> RubricVerification:
    return RubricVerification(
        overall_score=score,
        target_score=target,
        meets_target=meets if meets is not None else score >= target,
        rubric_source="generated",
    )


@pytest.mark.asyncio
async def test_gate3_blocked_on_first_call_halts_before_reiteration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """First run_gate_3 returns blocked → halt; reiteration loop never runs."""
    orch = _orch(
        tmp_path,
        monkeypatch,
        project_id="prj_gate3_blocked_first",
        rubric_verifier_enabled=True,
        rubric_max_improvement_iterations=2,  # loop COULD run; we expect it not to
    )
    state = _seed_state_past_baseline_run("prj_gate3_blocked_first")
    state.save_checkpoint(tmp_path)

    calls = {"improvements": 0, "gate_3": 0, "research_map": 0, "finalize_partial": 0}

    async def fake_run_gate_2(s):
        # Gate 2 passes — we're testing Gate 3, not Gate 2.
        s.gate_2 = GateDecision(
            gate="gate_2", passed=True, status=GateStatus.verified_with_caveats
        )
        s.advance_stage(PipelineStage.GATE_2_PASSED, orch.runs_root)
        return s

    async def fake_run_improvements(s, **kwargs):
        calls["improvements"] += 1
        s.advance_stage(PipelineStage.IMPROVEMENTS_RUN, orch.runs_root)
        return s

    async def fake_run_gate_3(s):
        calls["gate_3"] += 1
        # First (and only expected) call returns blocked.
        s.gate_3 = GateDecision(
            gate="gate_3", passed=False, status=GateStatus.blocked_requires_human
        )
        s.advance_stage(PipelineStage.GATE_3_PASSED, orch.runs_root)
        return s

    async def fake_research_map(s):
        calls["research_map"] += 1
        s.advance_stage(PipelineStage.RESEARCH_MAP_GENERATED, orch.runs_root)
        s.advance_stage(PipelineStage.COMPLETE, orch.runs_root)
        return s

    def fake_finalize_partial(s):
        calls["finalize_partial"] += 1
        s.advance_stage(PipelineStage.COMPLETE, orch.runs_root)

    monkeypatch.setattr(orch, "run_gate_2", fake_run_gate_2)
    monkeypatch.setattr(orch, "run_improvements", fake_run_improvements)
    monkeypatch.setattr(orch, "run_gate_3", fake_run_gate_3)
    monkeypatch.setattr(orch, "generate_research_map", fake_research_map)
    monkeypatch.setattr(orch, "_finalize_partial", fake_finalize_partial)

    final_state = await orch.run(resume=True)

    assert calls["improvements"] == 1, "exactly one improvement round before the halt"
    assert calls["gate_3"] == 1, "exactly one Gate 3 call; no reiteration"
    assert calls["finalize_partial"] == 1, "halt path must call _finalize_partial"
    assert calls["research_map"] == 0, "research map must NOT run on a blocked halt"
    assert final_state.stage is PipelineStage.COMPLETE


@pytest.mark.asyncio
async def test_gate3_partial_on_first_call_falls_through_to_reiteration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """First run_gate_3 returns partial → reiteration loop reached.

    With the rubric verifier disabled in this test fixture, the loop body
    won't actually execute (settings.rubric_verifier_enabled=False
    short-circuits ``_run_improvement_reiteration_loop``). What we're
    pinning is that the outer flow did NOT halt — the partial verdict
    is a continue, not a halt.
    """
    orch = _orch(
        tmp_path,
        monkeypatch,
        project_id="prj_gate3_partial_first",
        rubric_verifier_enabled=False,  # loop short-circuits, we focus on outer flow
    )
    state = _seed_state_past_baseline_run("prj_gate3_partial_first")
    state.save_checkpoint(tmp_path)

    calls = {"gate_3": 0, "research_map": 0, "finalize_partial": 0}

    async def fake_run_gate_2(s):
        s.gate_2 = GateDecision(
            gate="gate_2", passed=True, status=GateStatus.verified_with_caveats
        )
        s.advance_stage(PipelineStage.GATE_2_PASSED, orch.runs_root)
        return s

    async def fake_run_improvements(s, **kwargs):
        s.advance_stage(PipelineStage.IMPROVEMENTS_RUN, orch.runs_root)
        return s

    async def fake_run_gate_3(s):
        calls["gate_3"] += 1
        s.gate_3 = GateDecision(
            gate="gate_3", passed=False, status=GateStatus.partial_reproduction
        )
        s.advance_stage(PipelineStage.GATE_3_PASSED, orch.runs_root)
        return s

    async def fake_research_map(s):
        calls["research_map"] += 1
        s.advance_stage(PipelineStage.RESEARCH_MAP_GENERATED, orch.runs_root)
        s.advance_stage(PipelineStage.COMPLETE, orch.runs_root)
        return s

    def fake_finalize_partial(s):
        calls["finalize_partial"] += 1
        s.advance_stage(PipelineStage.COMPLETE, orch.runs_root)

    monkeypatch.setattr(orch, "run_gate_2", fake_run_gate_2)
    monkeypatch.setattr(orch, "run_improvements", fake_run_improvements)
    monkeypatch.setattr(orch, "run_gate_3", fake_run_gate_3)
    monkeypatch.setattr(orch, "generate_research_map", fake_research_map)
    monkeypatch.setattr(orch, "_finalize_partial", fake_finalize_partial)

    final_state = await orch.run(resume=True)

    # The decisive assertion: partial did NOT halt the pipeline. Research map
    # ran; _finalize_partial was NOT called. The reiteration loop's
    # bypass-when-disabled is irrelevant to this test — what matters is the
    # outer flow's halt decision.
    assert calls["finalize_partial"] == 0, (
        "partial_reproduction must NOT trigger _finalize_partial; the loop "
        "(or its absence) decides what to do"
    )
    assert calls["research_map"] == 1, "success path must complete the pipeline"
    assert final_state.stage is PipelineStage.COMPLETE


@pytest.mark.asyncio
async def test_gate3_partial_then_blocked_breaks_loop_and_halts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """First Gate 3 partial → enter reiteration → iteration 1 blocked → loop
    breaks; the outer post-loop check halts via ``_finalize_partial``.

    This exercises both the new inner ``break`` in
    ``_run_improvement_reiteration_loop`` and the new outer post-loop check.
    """
    orch = _orch(
        tmp_path,
        monkeypatch,
        project_id="prj_gate3_partial_then_blocked",
        rubric_verifier_enabled=True,
        rubric_max_improvement_iterations=3,  # plenty of room; expect loop to break early
    )
    state = _seed_state_past_baseline_run("prj_gate3_partial_then_blocked")
    state.save_checkpoint(tmp_path)

    # Seed improved_verification so the loop predicate _should_reiterate
    # evaluates True at entry (below target, iteration < max).
    state.improved_verification = _verification(0.3, 0.7, meets=False)
    state.verification_history = [state.improved_verification]
    state.save_checkpoint(tmp_path)

    gate3_calls = 0

    async def fake_run_gate_2(s):
        s.gate_2 = GateDecision(
            gate="gate_2", passed=True, status=GateStatus.verified_with_caveats
        )
        s.advance_stage(PipelineStage.GATE_2_PASSED, orch.runs_root)
        return s

    async def fake_run_improvements(s, **kwargs):
        s.advance_stage(PipelineStage.IMPROVEMENTS_RUN, orch.runs_root)
        return s

    async def fake_run_gate_3(s):
        # call 1 (unconditional): partial; loop enters
        # call 2 (iteration 1): blocked; loop must break
        nonlocal gate3_calls
        gate3_calls += 1
        if gate3_calls == 1:
            status = GateStatus.partial_reproduction
            s.improved_verification = _verification(0.3, 0.7, meets=False)
        else:
            status = GateStatus.blocked_requires_human
            s.improved_verification = _verification(0.4, 0.7, meets=False)
        s.gate_3 = GateDecision(gate="gate_3", passed=False, status=status)
        s.verification_history.append(s.improved_verification)
        s.advance_stage(PipelineStage.GATE_3_PASSED, orch.runs_root)
        return s

    finalize_calls = 0

    def fake_finalize_partial(s):
        nonlocal finalize_calls
        finalize_calls += 1
        s.advance_stage(PipelineStage.COMPLETE, orch.runs_root)

    monkeypatch.setattr(orch, "run_gate_2", fake_run_gate_2)
    monkeypatch.setattr(orch, "run_improvements", fake_run_improvements)
    monkeypatch.setattr(orch, "run_gate_3", fake_run_gate_3)
    monkeypatch.setattr(orch, "_finalize_partial", fake_finalize_partial)

    async def fake_research_map(s):
        raise AssertionError(
            "research_map must NOT run when the post-loop check halts the run"
        )

    monkeypatch.setattr(orch, "generate_research_map", fake_research_map)

    final_state = await orch.run(resume=True)

    # Exactly two gate_3 calls: unconditional first (partial) + one
    # reiteration round (blocked). The inner break + outer post-loop check
    # must prevent any third call.
    assert gate3_calls == 2, (
        f"expected exactly 2 Gate 3 calls (1 unconditional + 1 reiteration "
        f"before blocked break), got {gate3_calls}"
    )
    assert finalize_calls == 1, "outer post-loop halt must call _finalize_partial"
    assert final_state.stage is PipelineStage.COMPLETE
