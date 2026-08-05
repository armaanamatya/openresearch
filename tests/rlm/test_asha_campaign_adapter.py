"""ASHA campaign adapter — the AttemptAssessment→BranchObservation mapping +
cohort decision. Uses lightweight duck-typed fakes (the adapter reads only
attempt_n / final_report.score / failure_class), matching the real
AttemptAssessment/ReportDigest shape without the heavy fixture."""
from types import SimpleNamespace

import pytest

from backend.agents.rlm.asha_campaign_adapter import (
    asha_decide_for_assessments,
    observation_from_assessment,
)
from backend.agents.rlm.asha_scheduler import RungConfig


def _assess(attempt_n, score, failure_class=None, *, branch_type=None, is_safety_bracket=None):
    report = SimpleNamespace(score=score) if score is not None else None
    assessment = SimpleNamespace(
        attempt_n=attempt_n, final_report=report, failure_class=failure_class
    )
    # Omission is intentional: it exercises backward-compatible duck-typed
    # legacy objects, while explicit values exercise strict adapter validation.
    if branch_type is not None:
        assessment.branch_type = branch_type
    if is_safety_bracket is not None:
        assessment.is_safety_bracket = is_safety_bracket
    return assessment


def test_training_diverged_maps_to_broken():
    assert observation_from_assessment(_assess(1, 0.0, "training_diverged")).broken is True


def test_repairable_class_is_not_broken():
    # cell_execution_error is repairable (freeze/salvage) — NOT a true-kill.
    assert observation_from_assessment(_assess(1, 0.5, "cell_execution_error")).broken is False


def test_missing_report_maps_to_none_score():
    assert observation_from_assessment(_assess(1, None)).score is None


def test_attempt_n_becomes_branch_id():
    assert observation_from_assessment(_assess(7, 0.5)).branch_id == "7"


@pytest.mark.parametrize("branch_type", ("faithful", "ambiguity", "discovery"))
def test_explicit_valid_branch_type_is_preserved(branch_type):
    assert observation_from_assessment(_assess(1, 0.5, branch_type=branch_type)).branch_type == branch_type


def test_safety_marker_is_preserved_and_exempt_from_halving():
    decisions = {d.branch_id: d for d in asha_decide_for_assessments(
        [
            _assess(1, 0.1, is_safety_bracket=True),
            _assess(2, 0.9),
            _assess(3, 0.5),
        ],
        RungConfig(gpu_usd_budget=1.0, higher_is_better=True, noise_floor=0.0),
        gpu_usd_by_attempt={1: 1.0, 2: 1.0, 3: 1.0},
    )}
    assert decisions["1"].reason == "safety_bracket_exempt"
    assert decisions["1"].action == "promote"
    assert decisions["3"].action == "freeze"


def test_training_divergence_kills_before_safety_exemption():
    decisions = {d.branch_id: d for d in asha_decide_for_assessments(
        [_assess(1, 0.1, "training_diverged", is_safety_bracket=True)],
        RungConfig(),
    )}
    assert decisions["1"].action == "kill"
    assert decisions["1"].reason == "breakage_true_kill"


@pytest.mark.parametrize(
    "kwargs",
    (
        {"branch_type": "untrusted-free-text"},
        {"is_safety_bracket": "true"},
        {"branch_type": "ambiguity", "is_safety_bracket": True},
    ),
)
def test_explicit_invalid_scheduler_metadata_is_rejected(kwargs):
    with pytest.raises(ValueError):
        observation_from_assessment(_assess(1, 0.5, **kwargs))


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
