"""Contract test for Track 3 give-it-a-shot — Option B fix.

When Gate 2 returns ``GateStatus.partial_reproduction`` AND the environment
build succeeded, the orchestrator must NOT early-return. It must fall through
to ``run_improvements`` so the rubric-driven improvement loop has a chance to
lift the score above target before we commit to partial as the verdict.

The companion negative test pins the other half of the contract:
``GateStatus.blocked_requires_human`` (or any non-partial failing verdict)
still halts the run as before. Without that, the supervisor gate's
"halt for human review" semantics would be silently softened — Option B's
explicit goal is to only relax `partial_reproduction`, not all gate-2
failures wholesale.

See ``docs/design/option-b-investigation.md`` for the diagnosis and
``backend/agents/orchestrator.py`` for the implementation (the elif branch
on ``state.gate_2.status == GateStatus.partial_reproduction``).
"""

from __future__ import annotations

import asyncio
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
    # Match the lightweight construction pattern from test_rubric_verifier.py:
    # a SimpleNamespace settings object short-circuits the rubric reiteration
    # loop so the test focuses purely on the gate-2 decision branch.
    monkeypatch.setattr(
        "backend.agents.orchestrator.get_settings",
        lambda: SimpleNamespace(
            rubric_verifier_enabled=False,  # skip the reiteration loop entirely
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

    All earlier stages are marked done (their step_fn calls are skipped by the
    orchestrator's `current_idx >= target_idx` guard), so the test only
    exercises run_gate_2 + the gate-2 result check + the improvement phase.
    """
    state = PipelineState(project_id=project_id)
    state.stage = PipelineStage.BASELINE_RUN
    # Track 4 must NOT fire: keep environment_build_ok True so the branch
    # under test (Track 3 partial_reproduction) is the one that matches.
    state.environment_build_ok = True
    state.environment_build_attempts = 0
    return state


def _install_stage_mocks(
    orch: ReproLabOrchestrator,
    monkeypatch: pytest.MonkeyPatch,
    gate_2_status: GateStatus,
    calls: dict[str, int],
) -> None:
    """Stub every downstream stage so orchestrator.run() returns quickly.

    The stub for run_gate_2 sets state.gate_2 to a fail with the requested
    status; the stubs for run_improvements / run_gate_3 record that they were
    reached so the test can assert on the decision flow.
    """

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
        # rubric_verifier_enabled=False in fixture settings already short-
        # circuits this, but keep an async no-op as belt + suspenders so we
        # don't accidentally exercise the real loop in this contract test.
        return s

    async def fake_generate_research_map(s: PipelineState) -> PipelineState:
        s.advance_stage(PipelineStage.RESEARCH_MAP_GENERATED, orch.runs_root)
        s.advance_stage(PipelineStage.COMPLETE, orch.runs_root)
        return s

    monkeypatch.setattr(orch, "run_gate_2", fake_run_gate_2)
    monkeypatch.setattr(orch, "run_improvements", fake_run_improvements)
    monkeypatch.setattr(orch, "run_gate_3", fake_run_gate_3)
    monkeypatch.setattr(
        orch, "_run_improvement_reiteration_loop", fake_reiteration_loop
    )
    monkeypatch.setattr(orch, "generate_research_map", fake_generate_research_map)


@pytest.mark.asyncio
async def test_gate2_partial_reproduction_falls_through_to_improvements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Track 3 give-it-a-shot branch: partial_reproduction must NOT halt."""
    orch = _orch(tmp_path, monkeypatch, project_id="prj_gate2_partial")
    state = _seed_state_at_baseline_run("prj_gate2_partial")
    # Seed the existing checkpoint on disk so resume picks it up.
    state.save_checkpoint(tmp_path)

    calls = {"improvements": 0, "gate_3": 0}
    _install_stage_mocks(
        orch,
        monkeypatch,
        gate_2_status=GateStatus.partial_reproduction,
        calls=calls,
    )

    final_state = await orch.run(resume=True)

    # The decisive assertion: the orchestrator did NOT early-return at the
    # gate-2 check — proven by run_improvements having been called.
    assert calls["improvements"] >= 1, (
        "GateStatus.partial_reproduction must fall through to run_improvements; "
        "the orchestrator early-returned instead"
    )
    assert calls["gate_3"] >= 1, (
        "After improvements, run_gate_3 must also be reached"
    )
    # And the pipeline finished cleanly all the way to COMPLETE.
    assert final_state.stage is PipelineStage.COMPLETE


@pytest.mark.asyncio
async def test_gate2_blocked_requires_human_still_halts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defense-in-depth: non-partial failing verdicts must still halt.

    Option B's deliberate scope is narrow — soften ONLY partial_reproduction.
    Without this negative test, a future refactor could accidentally let
    blocked_requires_human fall through too, silently removing the human-
    review escape that the supervisor gate is built around.
    """
    orch = _orch(tmp_path, monkeypatch, project_id="prj_gate2_blocked")
    state = _seed_state_at_baseline_run("prj_gate2_blocked")
    state.save_checkpoint(tmp_path)

    calls = {"improvements": 0, "gate_3": 0}
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
    # The orchestrator returned early at GATE_2_PASSED, before COMPLETE.
    assert final_state.stage is PipelineStage.GATE_2_PASSED
