"""Guard tests for the Track 3 rubric verifier (Phases A-F)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.agents.registry import AGENT_REGISTRY
from backend.agents.schemas import (
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


# --- Phase F: BenchmarkSummary carries Track 3 fields through LiveRunState ---

def test_benchmark_summary_preserves_track3_fields():
    """The backend run-state model must not drop the rubric-verifier fields —
    they have to survive LiveRunState(**status) to reach the UI."""
    from backend.services.events.live_runs import LiveRunState

    status = {
        "projectId": "prj_bench",
        "outputDir": "/tmp/prj_bench",
        "runMode": "rlm",
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
        runMode="rlm",
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
