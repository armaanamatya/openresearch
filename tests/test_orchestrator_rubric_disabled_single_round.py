"""Contract tests for the rubric-disabled improvement path — Option D.

When ``settings.rubric_verifier_enabled`` is False, the rubric-driven
improvement loop has no signal to gate on, so it would never fire under
the widened ``_should_reiterate`` predicate. To preserve pre-Option-D
behavior (and to give ``generate_research_map`` improvement data to
narrate), the outer flow in ``orchestrator.run()`` keeps an explicit
legacy branch: a single unconditional ``run_improvements`` + ``run_gate_3``
pair, then the same outer Gate 3 halt check that covers the rubric-driven
loop break path.

This file pins both cases:

  1. Gate 3 passes → exactly one improvement round, exactly one Gate 3
     call, research_map runs, final stage COMPLETE.
  2. Gate 3 returns blocked → outer halt fires, ``_finalize_partial``
     called once, research_map does NOT run, final stage COMPLETE.

These complement the rubric-enabled tests in
``tests/test_gate3_blocked_halts.py``. The contract is symmetric:
regardless of rubric mode, any Gate 3 non-pass halts and writes a
terminal artifact.

See ``docs/design/option-d-q1q2-refactor.md`` Task 5.
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


def _orch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, project_id: str) -> ReproLabOrchestrator:
    monkeypatch.setattr(
        "backend.agents.orchestrator.get_settings",
        lambda: SimpleNamespace(
            rubric_verifier_enabled=False,  # the focus of this file
            rubric_max_improvement_iterations=0,
            rubric_target_score=0.7,
            rubric_verifier_model="",
            environment_build_validation_enabled=False,
            environment_build_max_attempts=1,
        ),
    )
    return ReproLabOrchestrator(project_id, tmp_path, runtime=object())


def _seed_state_past_baseline_run(project_id: str) -> PipelineState:
    """Seed state past baseline_run so the for-loop lands on gate_2 first."""
    state = PipelineState(project_id=project_id)
    state.stage = PipelineStage.BASELINE_RUN
    state.environment_build_ok = True
    state.environment_build_attempts = 0
    return state


def _install_mocks(
    orch: ReproLabOrchestrator,
    monkeypatch: pytest.MonkeyPatch,
    *,
    gate_3_status: GateStatus,
    gate_3_passed: bool,
    calls: dict[str, int],
) -> None:
    async def fake_run_gate_2(s: PipelineState) -> PipelineState:
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
            gate="gate_3", passed=gate_3_passed, status=gate_3_status
        )
        s.advance_stage(PipelineStage.GATE_3_PASSED, orch.runs_root)
        return s

    async def fake_research_map(s: PipelineState) -> PipelineState:
        calls["research_map"] += 1
        s.advance_stage(PipelineStage.RESEARCH_MAP_GENERATED, orch.runs_root)
        s.advance_stage(PipelineStage.COMPLETE, orch.runs_root)
        return s

    def fake_finalize_partial(s: PipelineState) -> None:
        calls["finalize_partial"] += 1
        s.advance_stage(PipelineStage.COMPLETE, orch.runs_root)

    monkeypatch.setattr(orch, "run_gate_2", fake_run_gate_2)
    monkeypatch.setattr(orch, "run_improvements", fake_run_improvements)
    monkeypatch.setattr(orch, "run_gate_3", fake_run_gate_3)
    monkeypatch.setattr(orch, "generate_research_map", fake_research_map)
    monkeypatch.setattr(orch, "_finalize_partial", fake_finalize_partial)


@pytest.mark.asyncio
async def test_rubric_disabled_runs_single_round_then_research_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gate 3 passes → exactly one round, research_map runs, COMPLETE."""
    orch = _orch(tmp_path, monkeypatch, project_id="prj_rubric_off_pass")
    state = _seed_state_past_baseline_run("prj_rubric_off_pass")
    state.save_checkpoint(tmp_path)

    calls = {"improvements": 0, "gate_3": 0, "research_map": 0, "finalize_partial": 0}
    _install_mocks(
        orch,
        monkeypatch,
        gate_3_status=GateStatus.verified,
        gate_3_passed=True,
        calls=calls,
    )

    final_state = await orch.run(resume=True)

    assert calls["improvements"] == 1, (
        f"rubric off must run exactly one improvement round; got {calls['improvements']}"
    )
    assert calls["gate_3"] == 1, (
        f"rubric off must call gate_3 exactly once; got {calls['gate_3']}"
    )
    assert calls["research_map"] == 1, "Gate 3 pass must reach research_map"
    assert calls["finalize_partial"] == 0, "no halt on Gate 3 pass"
    assert final_state.stage is PipelineStage.COMPLETE


@pytest.mark.asyncio
async def test_rubric_disabled_gate3_blocked_halts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gate 3 blocked → outer halt fires, _finalize_partial once, no research_map."""
    orch = _orch(tmp_path, monkeypatch, project_id="prj_rubric_off_blocked")
    state = _seed_state_past_baseline_run("prj_rubric_off_blocked")
    state.save_checkpoint(tmp_path)

    calls = {"improvements": 0, "gate_3": 0, "research_map": 0, "finalize_partial": 0}
    _install_mocks(
        orch,
        monkeypatch,
        gate_3_status=GateStatus.blocked_requires_human,
        gate_3_passed=False,
        calls=calls,
    )

    final_state = await orch.run(resume=True)

    assert calls["improvements"] == 1
    assert calls["gate_3"] == 1
    assert calls["research_map"] == 0, (
        "outer Gate 3 halt must prevent research_map from running"
    )
    assert calls["finalize_partial"] == 1, (
        "outer halt must call _finalize_partial exactly once"
    )
    assert final_state.stage is PipelineStage.COMPLETE
