"""Guard tests for the Track 3 rubric verifier (Phases A-F)."""

from __future__ import annotations

import asyncio
import json

import pytest
from pydantic import ValidationError

from backend.agents.registry import AGENT_REGISTRY
from backend.agents.rubric_source import (
    BundleRubricSource,
    GeneratedRubricSource,
    RubricSource,
    resolve_rubric_source,
)
from backend.agents.schemas import (
    ExperimentArtifacts,
    RubricAreaScore,
    RubricVerification,
)


def _area(area: str, weight: float, score: float) -> RubricAreaScore:
    return RubricAreaScore(area=area, weight=weight, score=score)


# --- RubricVerification.from_areas: overall_score is computed, not trusted ---

def test_from_areas_weight_normalized_overall():
    areas = [_area("a", 0.5, 0.8), _area("b", 0.5, 0.4)]
    v = RubricVerification.from_areas(areas, rubric_source="generated", target_score=0.7)
    assert v.overall_score == pytest.approx(0.6)
    assert v.meets_target is False


def test_from_areas_normalizes_when_weights_do_not_sum_to_one():
    areas = [_area("a", 0.2, 1.0), _area("b", 0.2, 0.0)]
    v = RubricVerification.from_areas(areas, rubric_source="generated", target_score=0.4)
    assert v.overall_score == pytest.approx(0.5)
    assert v.meets_target is True


def test_from_areas_plain_mean_when_no_weights():
    areas = [_area("a", 0.0, 0.6), _area("b", 0.0, 0.2)]
    v = RubricVerification.from_areas(areas, rubric_source="generated", target_score=0.5)
    assert v.overall_score == pytest.approx(0.4)


def test_from_areas_empty_is_zero():
    v = RubricVerification.from_areas([], rubric_source="generated", target_score=0.7)
    assert v.overall_score == 0.0
    assert v.meets_target is False


def test_from_areas_passes_through_metadata():
    v = RubricVerification.from_areas(
        [_area("a", 1.0, 0.9)],
        rubric_source="paperbench_bundle",
        target_score=0.8,
        confidence=0.65,
        verified_at="2026-05-14T00:00:00Z",
    )
    assert v.rubric_source == "paperbench_bundle"
    assert v.confidence == 0.65
    assert v.verified_at == "2026-05-14T00:00:00Z"
    assert v.meets_target is True


def test_area_score_bounds_enforced():
    with pytest.raises(ValidationError):
        RubricAreaScore(area="a", weight=0.5, score=1.5)
    with pytest.raises(ValidationError):
        RubricAreaScore(area="a", weight=-0.1, score=0.5)


# --- RubricSource: bundle vs generated, degrades cleanly ---

def _write_bundle(root, paper_id):
    bundle_dir = root / paper_id
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "paper.md").write_text("# Paper", encoding="utf-8")
    (bundle_dir / "addendum.md").write_text("# Addendum", encoding="utf-8")
    rubric = {"sub_tasks": [{"requirements": "x", "weight": 1.0, "sub_tasks": []}]}
    (bundle_dir / "rubric.json").write_text(json.dumps(rubric), encoding="utf-8")
    return bundle_dir


def test_resolve_returns_bundle_source_when_bundle_exists(tmp_path):
    _write_bundle(tmp_path, "demo-paper")
    src = resolve_rubric_source(tmp_path, "demo-paper")
    assert isinstance(src, BundleRubricSource)
    assert src.kind == "paperbench_bundle"
    assert src.load_rubric() is not None
    assert isinstance(src, RubricSource)


def test_resolve_degrades_to_generated_when_no_bundle(tmp_path):
    src = resolve_rubric_source(tmp_path, "missing-paper")
    assert isinstance(src, GeneratedRubricSource)
    assert src.kind == "generated"
    assert src.load_rubric() is None


def test_resolve_degrades_when_no_paper_id(tmp_path):
    assert isinstance(resolve_rubric_source(tmp_path, None), GeneratedRubricSource)
    assert isinstance(resolve_rubric_source(None, "x"), GeneratedRubricSource)


def test_resolve_degrades_on_malformed_bundle(tmp_path):
    # bundle dir exists but rubric.json is missing -> PaperBenchBundleError -> generated
    bad = tmp_path / "bad-paper"
    bad.mkdir()
    (bad / "paper.md").write_text("x", encoding="utf-8")
    src = resolve_rubric_source(tmp_path, "bad-paper")
    assert isinstance(src, GeneratedRubricSource)


# --- rubric-verifier agent registration ---

def test_rubric_verifier_registered():
    spec = AGENT_REGISTRY["rubric-verifier"]
    assert spec.role == "verifier"
    assert set(spec.tools) == {"Read", "Bash"}
    assert "Write" not in spec.tools


def test_rubric_verifier_inherits_run_default_model():
    spec = AGENT_REGISTRY["rubric-verifier"]
    # No hardcoded anthropic model -> inherits the run's default model.
    assert spec.default_model_anthropic == ""
    runtime = spec.to_runtime_spec("anthropic")
    assert runtime.model  # resolved to a concrete model, not empty
    assert runtime.instructions  # prompt is wired


def test_rubric_verifier_registered_for_structured_output():
    from backend.agents.orchestrator import ReproLabOrchestrator

    assert ReproLabOrchestrator._OUTPUT_MODELS["rubric-verifier"] is RubricVerification


# --- Phase B: PipelineState carries verification fields through a checkpoint ---

def test_pipeline_state_round_trips_verification_fields(tmp_path):
    from backend.agents.orchestrator import PipelineState

    state = PipelineState(project_id="prj_verify")
    state.baseline_verification = RubricVerification.from_areas(
        [_area("env", 0.5, 0.4), _area("impl", 0.5, 0.6)],
        rubric_source="generated",
        target_score=0.7,
        confidence=0.8,
        verified_at="2026-05-14T00:00:00Z",
    )
    state.improved_verification = RubricVerification.from_areas(
        [_area("env", 0.5, 0.7), _area("impl", 0.5, 0.8)],
        rubric_source="generated",
        target_score=0.7,
    )
    state.verification_history = [
        state.baseline_verification,
        state.improved_verification,
    ]
    state.improvement_iteration = 2
    state.save_checkpoint(tmp_path)

    resumed = PipelineState.load_checkpoint(tmp_path, "prj_verify")
    assert resumed is not None
    assert resumed.baseline_verification is not None
    assert resumed.baseline_verification.overall_score == pytest.approx(0.5)
    assert resumed.baseline_verification.areas[0].area == "env"
    assert resumed.baseline_verification.confidence == pytest.approx(0.8)
    assert resumed.improved_verification is not None
    assert resumed.improved_verification.overall_score == pytest.approx(0.75)
    assert resumed.improved_verification.meets_target is True
    assert len(resumed.verification_history) == 2
    assert resumed.improvement_iteration == 2


def test_pipeline_state_omits_verification_when_unset(tmp_path):
    """A run with the verifier disabled checkpoints exactly as before."""
    from backend.agents.orchestrator import PipelineState

    state = PipelineState(project_id="prj_plain")
    state.save_checkpoint(tmp_path)
    raw = json.loads((tmp_path / "prj_plain" / "pipeline_state.json").read_text())
    assert "baseline_verification" not in raw
    assert "improved_verification" not in raw
    assert "verification_history" not in raw
    assert "improvement_iteration" not in raw

    resumed = PipelineState.load_checkpoint(tmp_path, "prj_plain")
    assert resumed is not None
    assert resumed.baseline_verification is None
    assert resumed.improved_verification is None
    assert resumed.verification_history == []
    assert resumed.improvement_iteration == 0


# --- Phase B: _clamp01 bounds LLM-supplied numbers at the agent boundary ---

def test_clamp01_bounds_and_coerces():
    from backend.agents.orchestrator import _clamp01

    assert _clamp01(0.5) == 0.5
    assert _clamp01(1.5) == 1.0
    assert _clamp01(-0.2) == 0.0
    assert _clamp01("0.3") == pytest.approx(0.3)
    assert _clamp01(None) == 0.0
    assert _clamp01("not-a-number") == 0.0


# --- Phase C: the capped self-improvement re-iteration loop ---

def _verification(score, target):
    return RubricVerification.from_areas(
        [_area("a", 1.0, score)], rubric_source="generated", target_score=target
    )


def test_should_reiterate_boundary():
    from backend.agents.orchestrator import _should_reiterate

    below = _verification(0.1, 0.7)
    meets = _verification(0.9, 0.7)
    assert _should_reiterate(None, 0, 2) is False     # no verification
    assert _should_reiterate(meets, 0, 2) is False    # target already met
    assert _should_reiterate(below, 2, 2) is False    # iteration cap reached
    assert _should_reiterate(below, 0, 2) is True     # room to improve
    assert _should_reiterate(below, 0, 0) is False    # cap of 0 disables the loop


def _loop_orchestrator(tmp_path, monkeypatch, *, max_iterations, enabled=True):
    from types import SimpleNamespace

    from backend.agents.orchestrator import ReproLabOrchestrator

    monkeypatch.setattr(
        "backend.agents.orchestrator.get_settings",
        lambda: SimpleNamespace(
            rubric_verifier_enabled=enabled,
            rubric_max_improvement_iterations=max_iterations,
            rubric_target_score=0.7,
            rubric_verifier_model="",
        ),
    )
    return ReproLabOrchestrator("prj_loop", tmp_path, runtime=object())


def test_reiteration_loop_is_hard_capped(tmp_path, monkeypatch):
    """A verifier permanently below target must not spin the loop forever."""
    from backend.agents.orchestrator import PipelineState

    orch = _loop_orchestrator(tmp_path, monkeypatch, max_iterations=2)
    state = PipelineState(project_id="prj_loop")
    state.improved_verification = _verification(0.1, 0.7)
    calls = {"improvements": 0, "gate_3": 0}

    async def fake_improvements(s, **kwargs):
        calls["improvements"] += 1
        return s

    async def fake_gate_3(s):
        calls["gate_3"] += 1
        v = _verification(0.1, 0.7)  # never meets target
        s.improved_verification = v
        s.verification_history.append(v)  # a successful verifier appends
        return s

    monkeypatch.setattr(orch, "run_improvements", fake_improvements)
    monkeypatch.setattr(orch, "run_gate_3", fake_gate_3)

    result = asyncio.run(
        orch._run_improvement_reiteration_loop(
            state, user_hints=None, n_improvement_paths=3
        )
    )
    assert calls["improvements"] == 2  # capped — terminates, never infinite
    assert calls["gate_3"] == 2
    assert result.improvement_iteration == 2


def test_reiteration_loop_stops_when_target_met(tmp_path, monkeypatch):
    from backend.agents.orchestrator import PipelineState

    orch = _loop_orchestrator(tmp_path, monkeypatch, max_iterations=5)
    state = PipelineState(project_id="prj_loop")
    state.improved_verification = _verification(0.1, 0.7)
    calls = {"improvements": 0}

    async def fake_improvements(s, **kwargs):
        calls["improvements"] += 1
        return s

    async def fake_gate_3(s):
        v = _verification(0.9, 0.7)  # meets target round 1
        s.improved_verification = v
        s.verification_history.append(v)  # a successful verifier appends
        return s

    monkeypatch.setattr(orch, "run_improvements", fake_improvements)
    monkeypatch.setattr(orch, "run_gate_3", fake_gate_3)

    result = asyncio.run(
        orch._run_improvement_reiteration_loop(
            state, user_hints=None, n_improvement_paths=3
        )
    )
    assert calls["improvements"] == 1  # stopped early — well under the cap of 5
    assert result.improvement_iteration == 1
    assert result.improved_verification.meets_target is True


def test_reiteration_loop_skips_when_already_met_or_disabled(tmp_path, monkeypatch):
    from backend.agents.orchestrator import PipelineState

    ran = {"called": False}

    async def fail_if_called(s, **kwargs):
        ran["called"] = True
        return s

    # Already meets target -> zero rounds.
    orch = _loop_orchestrator(tmp_path, monkeypatch, max_iterations=2)
    state = PipelineState(project_id="prj_loop")
    state.improved_verification = _verification(0.9, 0.7)
    monkeypatch.setattr(orch, "run_improvements", fail_if_called)
    monkeypatch.setattr(orch, "run_gate_3", fail_if_called)
    result = asyncio.run(
        orch._run_improvement_reiteration_loop(
            state, user_hints=None, n_improvement_paths=3
        )
    )
    assert ran["called"] is False
    assert result.improvement_iteration == 0

    # Verifier disabled -> zero rounds even though it is below target.
    orch2 = _loop_orchestrator(tmp_path, monkeypatch, max_iterations=2, enabled=False)
    state2 = PipelineState(project_id="prj_loop")
    state2.improved_verification = _verification(0.1, 0.7)
    monkeypatch.setattr(orch2, "run_improvements", fail_if_called)
    monkeypatch.setattr(orch2, "run_gate_3", fail_if_called)
    result2 = asyncio.run(
        orch2._run_improvement_reiteration_loop(
            state2, user_hints=None, n_improvement_paths=3
        )
    )
    assert ran["called"] is False
    assert result2.improvement_iteration == 0


# --- Phase D: FinalReport verification fields + honest comparison summary ---

def test_generate_final_report_without_verification_is_unchanged():
    """No verification args -> the new fields stay at their empty defaults."""
    from backend.agents.report_generator import generate_final_report

    report = generate_final_report("prj_plain", None, None, [], [], None)
    assert report.rubric_verification is None
    assert report.baseline_rubric_verification is None
    assert report.verification_delta is None
    assert report.improvement_iterations == 0
    assert report.comparison_summary == ""
    assert report.paperbench_baseline is None  # prj_* has no published baseline


def test_generate_final_report_populates_verification_fields():
    from backend.agents.report_generator import generate_final_report

    baseline = _verification(0.40, 0.70)
    improved = _verification(0.66, 0.70)
    report = generate_final_report(
        "prj_v",
        None,
        None,
        [],
        [],
        None,
        baseline_verification=baseline,
        improved_verification=improved,
        improvement_iterations=2,
    )
    assert report.rubric_verification.overall_score == pytest.approx(0.66)
    assert report.baseline_rubric_verification.overall_score == pytest.approx(0.40)
    assert report.verification_delta == pytest.approx(0.26)
    assert report.improvement_iterations == 2
    assert "0.66" in report.comparison_summary
    assert "Still below the 0.70 target" in report.comparison_summary


def test_comparison_summary_is_honest():
    from backend.agents.report_generator import _build_comparison_summary

    # Improvement helped and met the target.
    s = _build_comparison_summary(
        _verification(0.8, 0.7), _verification(0.5, 0.7), None, 1
    )
    assert "+0.30" in s and "Meets the 0.70 target" in s
    # Improvement made it worse — the summary must say so plainly.
    s = _build_comparison_summary(
        _verification(0.4, 0.7), _verification(0.6, 0.7), None, 2
    )
    assert "did not help" in s and "Still below" in s
    # No improved verification -> empty string.
    assert _build_comparison_summary(None, None, None, 0) == ""


def test_resolve_paperbench_baseline():
    from backend.agents.report_generator import _resolve_paperbench_baseline

    assert _resolve_paperbench_baseline("prj_uploaded_paper") is None
    assert _resolve_paperbench_baseline("paperbench_unknown_xyz") is None
    ftrl = _resolve_paperbench_baseline("paperbench_ftrl")
    assert ftrl is not None
    assert ftrl["source"] == "paperbench_published"
    # ftrl's strongest published agent is claude_3_5_sonnet_basicagent @ 0.093.
    assert ftrl["score"] == pytest.approx(0.093)
    assert ftrl["model"] == "claude_3_5_sonnet_basicagent"


# --- Phase C.5: canonical rubric persistence + weight-from-spec hardening ---

def _verifier_orchestrator(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from backend.agents.orchestrator import ReproLabOrchestrator

    monkeypatch.setattr(
        "backend.agents.orchestrator.get_settings",
        lambda: SimpleNamespace(
            rubric_verifier_enabled=True,
            rubric_target_score=0.7,
            rubric_verifier_model="",
            rubric_max_improvement_iterations=2,
        ),
    )
    return ReproLabOrchestrator("prj_rv", tmp_path, runtime=object())


def test_run_rubric_verifier_persists_and_reuses_canonical_rubric(tmp_path, monkeypatch):
    from backend.agents.orchestrator import PipelineState

    orch = _verifier_orchestrator(tmp_path, monkeypatch)
    state = PipelineState(project_id="prj_rv")
    state.experiment_artifacts = ExperimentArtifacts(success=True)
    captured = {}

    async def fake_invoke(agent_id, prompt, **kwargs):
        captured["prompt"] = prompt
        if state.rubric_spec is None:
            # First call: the model proposes the canonical areas + weights.
            return (
                '{"areas": ['
                '{"area": "code", "weight": 0.7, "score": 0.4},'
                '{"area": "results", "weight": 0.3, "score": 0.6}'
                '], "confidence": 0.8}'
            )
        # Later call: the model tries to cheat the weights — must be ignored.
        return (
            '{"areas": ['
            '{"area": "code", "weight": 0.99, "score": 0.9},'
            '{"area": "results", "weight": 0.01, "score": 0.9}'
            '], "confidence": 0.9}'
        )

    monkeypatch.setattr(orch, "_invoke_agent", fake_invoke)

    v1 = asyncio.run(orch._run_rubric_verifier(state, checkpoint="baseline"))
    assert v1 is not None
    # First call fixed the canonical rubric.
    assert state.rubric_spec is not None
    assert {a["area"]: a["weight"] for a in state.rubric_spec["areas"]} == {
        "code": 0.7,
        "results": 0.3,
    }
    # The verifier was handed baseline_result paths to inspect.
    assert '"baseline_result"' in captured["prompt"]
    assert v1.overall_score == pytest.approx(0.4 * 0.7 + 0.6 * 0.3)

    v2 = asyncio.run(orch._run_rubric_verifier(state, checkpoint="improved"))
    assert v2 is not None
    # The model's cheated weights (0.99 / 0.01) are ignored — persisted weights win.
    assert {a.area: a.weight for a in v2.areas} == {"code": 0.7, "results": 0.3}
    assert v2.overall_score == pytest.approx(0.9 * 0.7 + 0.9 * 0.3)


def test_run_rubric_verifier_fails_closed_on_bad_output(tmp_path, monkeypatch):
    from backend.agents.orchestrator import PipelineState

    orch = _verifier_orchestrator(tmp_path, monkeypatch)
    state = PipelineState(project_id="prj_rv")

    async def bad_invoke(agent_id, prompt, **kwargs):
        return "not json at all"

    monkeypatch.setattr(orch, "_invoke_agent", bad_invoke)
    result = asyncio.run(orch._run_rubric_verifier(state, checkpoint="baseline"))
    assert result is None              # fail-closed
    assert state.rubric_spec is None   # nothing persisted on failure


def test_resolve_run_rubric_source_degrades_to_generated():
    from backend.agents.orchestrator import _resolve_run_rubric_source
    from backend.agents.rubric_source import GeneratedRubricSource

    assert isinstance(
        _resolve_run_rubric_source("prj_uploaded"), GeneratedRubricSource
    )
    # paperbench_<id> with no on-disk bundle degrades cleanly to generated.
    assert isinstance(
        _resolve_run_rubric_source("paperbench_nonexistent_xyz"),
        GeneratedRubricSource,
    )


def test_pipeline_state_round_trips_rubric_spec(tmp_path):
    from backend.agents.orchestrator import PipelineState

    state = PipelineState(project_id="prj_spec")
    state.rubric_spec = {
        "source": "generated",
        "areas": [
            {"area": "code", "weight": 0.6},
            {"area": "results", "weight": 0.4},
        ],
    }
    state.save_checkpoint(tmp_path)
    resumed = PipelineState.load_checkpoint(tmp_path, "prj_spec")
    assert resumed is not None
    assert resumed.rubric_spec == state.rubric_spec


# --- Phase F: BenchmarkSummary carries Track 3 fields through LiveRunState ---

def test_benchmark_summary_preserves_track3_fields():
    """The backend run-state model must not drop the rubric-verifier fields —
    they have to survive LiveRunState(**status) to reach the UI."""
    from backend.services.events.live_runs import LiveRunState

    status = {
        "projectId": "prj_bench",
        "outputDir": "/tmp/prj_bench",
        "runMode": "sdk",
        "status": "completed",
        "benchmark": {
            "benchmarkName": "PaperBench-style final benchmark",
            "paperbenchTaskId": "pb-1",
            "overallScore": 66.0,
            "targetMetric": "accuracy",
            "targetValue": 0.9,
            "reproducedValue": 0.8,
            "deltaValue": -0.1,
            "verdict": "partial",
            "reportPath": "/tmp/r.md",
            "comparisonPath": "/tmp/r.json",
            "logPath": "/tmp/r.log",
            # Track 3 fields — must survive the parse, not be dropped.
            "paperbenchBaseline": {
                "score": 0.093,
                "source": "paperbench_published",
                "model": "x",
            },
            "ourRubricScore": 0.66,
            "verificationDelta": 0.26,
            "improvementIterations": 2,
            "meetsTarget": False,
            "comparisonSummary": "Our rubric score is 0.66. Still below the 0.70 target.",
            "rubricAreas": [
                {
                    "area": "code",
                    "weight": 0.6,
                    "score": 0.6,
                    "justification": "",
                    "weak_points": [],
                }
            ],
            "baselineRubricAreas": [
                {
                    "area": "code",
                    "weight": 0.6,
                    "score": 0.4,
                    "justification": "",
                    "weak_points": [],
                }
            ],
        },
    }
    run = LiveRunState(**status)
    assert run.benchmark is not None
    assert run.benchmark.ourRubricScore == 0.66
    assert run.benchmark.improvementIterations == 2
    assert run.benchmark.meetsTarget is False
    assert run.benchmark.comparisonSummary.startswith("Our rubric score")
    assert run.benchmark.paperbenchBaseline == {
        "score": 0.093,
        "source": "paperbench_published",
        "model": "x",
    }
    assert run.benchmark.rubricAreas[0]["score"] == 0.6
    assert run.benchmark.baselineRubricAreas[0]["score"] == 0.4
    # And they survive model_dump — the SSE / API serialization path to the UI.
    dumped = run.model_dump(mode="json")
    assert dumped["benchmark"]["comparisonSummary"].startswith("Our rubric score")
    assert dumped["benchmark"]["improvementIterations"] == 2


def test_benchmark_summary_back_compat_without_track3_fields():
    """An old demo_status.json (no Track 3 fields) still parses, with defaults."""
    from backend.services.events.live_runs import LiveRunState

    run = LiveRunState(
        projectId="prj_old",
        outputDir="/tmp/prj_old",
        runMode="sdk",
        status="completed",
        benchmark={
            "benchmarkName": "b",
            "paperbenchTaskId": "t",
            "overallScore": 50.0,
            "targetMetric": "m",
            "targetValue": 1.0,
            "reproducedValue": 0.5,
            "deltaValue": -0.5,
            "verdict": "partial",
            "reportPath": "/tmp/r.md",
            "comparisonPath": "/tmp/r.json",
            "logPath": "/tmp/r.log",
        },
    )
    assert run.benchmark is not None
    assert run.benchmark.ourRubricScore is None
    assert run.benchmark.improvementIterations == 0
    assert run.benchmark.comparisonSummary == ""
    assert run.benchmark.rubricAreas == []


# --- Phase F.5: Codex review #2 hardening — loop early-break + honesty cap ---

def test_reiteration_loop_breaks_when_verifier_produces_no_result(tmp_path, monkeypatch):
    """A round whose verifier fails (no new verification) stops the loop — it
    does not burn the remaining capped rounds against a dead verifier."""
    from backend.agents.orchestrator import PipelineState

    orch = _loop_orchestrator(tmp_path, monkeypatch, max_iterations=3)
    state = PipelineState(project_id="prj_loop")
    state.improved_verification = _verification(0.1, 0.7)
    calls = {"improvements": 0}

    async def fake_improvements(s, **kwargs):
        calls["improvements"] += 1
        return s

    async def fake_gate_3(s):
        # The verifier "fails": improved_verification unchanged, history NOT grown.
        return s

    monkeypatch.setattr(orch, "run_improvements", fake_improvements)
    monkeypatch.setattr(orch, "run_gate_3", fake_gate_3)
    result = asyncio.run(
        orch._run_improvement_reiteration_loop(
            state, user_hints=None, n_improvement_paths=3
        )
    )
    assert calls["improvements"] == 1  # broke after the first dead round, not 3
    assert result.improvement_iteration == 0  # the dead round was not counted


def test_run_rubric_verifier_caps_score_when_run_did_not_succeed(tmp_path, monkeypatch):
    """Honesty backstop: if the reproduction never executed successfully, the
    verifier's area scores are mechanically capped — not left to the prompt."""
    from backend.agents.orchestrator import PipelineState

    orch = _verifier_orchestrator(tmp_path, monkeypatch)
    state = PipelineState(project_id="prj_rv")
    state.experiment_artifacts = ExperimentArtifacts(success=False)

    async def fake_invoke(agent_id, prompt, **kwargs):
        # The model returns a high score the run has not earned.
        return (
            '{"areas": [{"area": "code", "weight": 1.0, "score": 0.95}], '
            '"confidence": 0.9}'
        )

    monkeypatch.setattr(orch, "_invoke_agent", fake_invoke)
    v = asyncio.run(orch._run_rubric_verifier(state, checkpoint="baseline"))
    assert v is not None
    assert v.areas[0].score <= 0.35  # capped — the reproduction did not execute
    assert v.overall_score <= 0.35
