"""ASHA campaign adapter — the AttemptAssessment→BranchObservation mapping +
cohort decision. Uses lightweight duck-typed fakes (the adapter reads only
attempt_n / final_report.score / failure_class), matching the real
AttemptAssessment/ReportDigest shape without the heavy fixture."""
from types import SimpleNamespace

from backend.agents.rlm.asha_campaign_adapter import (
    asha_decide_for_assessments,
    observation_from_assessment,
)
from backend.agents.rlm.asha_scheduler import RungConfig


def _assess(attempt_n, score, failure_class=None):
    report = SimpleNamespace(score=score) if score is not None else None
    return SimpleNamespace(
        attempt_n=attempt_n, final_report=report, failure_class=failure_class
    )


def test_training_diverged_maps_to_broken():
    assert observation_from_assessment(_assess(1, 0.0, "training_diverged")).broken is True


def test_repairable_class_is_not_broken():
    # cell_execution_error is repairable (freeze/salvage) — NOT a true-kill.
    assert observation_from_assessment(_assess(1, 0.5, "cell_execution_error")).broken is False


def test_missing_report_maps_to_none_score():
    assert observation_from_assessment(_assess(1, None)).score is None


def test_attempt_n_becomes_branch_id():
    assert observation_from_assessment(_assess(7, 0.5)).branch_id == "7"


def test_decide_over_cohort_halves_by_score():
    decisions = {d.branch_id: d for d in asha_decide_for_assessments(
        [_assess(1, 0.9), _assess(2, 0.1)],
        RungConfig(gpu_usd_budget=1.0, higher_is_better=True, noise_floor=0.0),
        gpu_usd_by_attempt={1: 1.0, 2: 1.0},
    )}
    assert decisions["1"].action == "promote"
    assert decisions["2"].action == "freeze"


def test_diverged_attempt_killed_in_cohort():
    decisions = {d.branch_id: d for d in asha_decide_for_assessments(
        [_assess(1, 0.9), _assess(2, 0.0, "training_diverged")],
        RungConfig(gpu_usd_budget=1.0, higher_is_better=True, noise_floor=0.0),
        gpu_usd_by_attempt={1: 1.0, 2: 1.0},
    )}
    assert decisions["1"].action == "promote"
    assert decisions["2"].action == "kill"


def test_no_gpu_costs_falls_back_to_eta():
    # 6 attempts, no per-attempt gpu-$ → geometric eta fallback (ceil(6/3)=2 promoted).
    decisions = asha_decide_for_assessments(
        [_assess(i, float(i)) for i in range(6)],
        RungConfig(eta=3.0, higher_is_better=True, noise_floor=0.0),
    )
    assert sum(1 for d in decisions if d.action == "promote") == 2
