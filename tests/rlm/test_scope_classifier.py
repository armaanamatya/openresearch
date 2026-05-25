"""Tests for backend.agents.rlm.scope_classifier — Lane P.

Pinned invariants:

  * Uncontrollable failures (HF deprecation, RunPod 500, network flake)
    are EXCLUDED from the rubric denominator — no penalty.
  * cuda_oom WITHOUT a scope-reduction attempt = controllable (penalty).
  * cuda_oom WITH a scope-reduction attempt that still didn't fit =
    uncontrollable (no penalty).
  * preflight_blocked + a later successful iteration = full credit.
  * preflight_blocked + no resolution = half credit.
  * <50% experiment coverage caps the scope-adjusted score at 0.30.
  * If status=="ok", history is irrelevant — full credit. Recovery wins.
"""

from __future__ import annotations

import pytest

from backend.agents.rlm.scope_classifier import (
    CONTROLLABLE_CLASSES,
    UNCONTROLLABLE_CLASSES,
    classify_experiment,
    compute_scope_adjusted_rubric,
)


# ---------------------------------------------------------------------------
# classify_experiment — per-experiment judgement
# ---------------------------------------------------------------------------


def test_ok_status_full_credit():
    j = classify_experiment(final_status="ok")
    assert j.effective_status == "ran"
    assert j.credit == 1.0
    assert j.in_denominator is True


def test_ok_after_recovery_still_full_credit():
    """If iter 1 failed but iter 2 ran, no penalty — recovery wins."""
    j = classify_experiment(
        final_status="ok",
        history=[
            {"status": "preflight_blocked", "reason_class": "tensor_device_mismatch"},
            {"status": "ok", "accuracy": 0.97},
        ],
    )
    assert j.effective_status == "ran"
    assert j.credit == 1.0
    assert "recovery" in j.notes


def test_data_unavailable_uncontrollable():
    j = classify_experiment(
        final_status="data_unavailable",
        final_reason_class="missing_dataset",
    )
    assert j.effective_status == "uncontrollable_skip"
    assert j.credit == 0.0
    assert j.in_denominator is False
    assert "outside agent control" in j.notes


def test_network_flake_uncontrollable():
    j = classify_experiment(final_status="failed", final_reason_class="network_flake")
    assert j.in_denominator is False


def test_runpod_capacity_uncontrollable():
    j = classify_experiment(final_status="failed", final_reason_class="runpod_capacity")
    assert j.in_denominator is False


def test_runpod_balance_too_low_uncontrollable():
    j = classify_experiment(final_status="failed", final_reason_class="runpod_balance_too_low")
    assert j.in_denominator is False


def test_compute_too_large_uncontrollable():
    j = classify_experiment(final_status="scope_reduced", final_reason_class="compute_too_large")
    assert j.in_denominator is False
    assert "outside agent control" in j.notes


def test_cuda_oom_without_reduction_attempt_controllable():
    """Agent didn't try to scale down — controllable."""
    j = classify_experiment(
        final_status="cuda_oom",
        final_reason_class="cuda_oom",
        history=[],
    )
    assert j.effective_status == "controllable_fail"
    assert j.credit == 0.0
    assert j.in_denominator is True
    assert "without scope-reduction" in j.notes


def test_cuda_oom_with_reduction_but_still_oom_uncontrollable():
    """Agent tried smaller batch, still OOM — hardware genuinely can't fit."""
    j = classify_experiment(
        final_status="cuda_oom",
        final_reason_class="cuda_oom",
        history=[
            {"status": "cuda_oom", "batch_reduced": True, "scope_reduction": "batch 256 → 128"},
            {"status": "cuda_oom", "batch_reduced": True, "scope_reduction": "batch 128 → 64"},
        ],
    )
    assert j.effective_status == "uncontrollable_skip"
    assert j.credit == 0.0
    assert j.in_denominator is False


def test_cuda_oom_then_recovered_full_credit():
    """Agent OOM'd, scaled down, ran — final status=ok → full credit."""
    j = classify_experiment(
        final_status="ok",
        history=[
            {"status": "cuda_oom", "batch_reduced": True},
            {"status": "ok"},
        ],
    )
    assert j.effective_status == "ran"
    assert j.credit == 1.0


def test_preflight_unresolved_half_credit():
    """Pre-flight blocked, agent never resolved — half credit."""
    j = classify_experiment(
        final_status="preflight_blocked",
        final_reason_class="preflight_blocked",
        history=[
            {"status": "preflight_blocked", "reason_class": "torch_redundancy"},
            {"status": "preflight_blocked", "reason_class": "torch_redundancy"},
        ],
    )
    assert j.effective_status == "preflight_unresolved"
    assert j.credit == 0.5
    assert j.in_denominator is True


def test_preflight_then_resolved_full_credit():
    """Pre-flight blocked, agent fixed it on next iter — full credit
    (final_status would be 'ok' so this is covered by happy path)."""
    j = classify_experiment(
        final_status="preflight_blocked",
        history=[
            {"status": "preflight_blocked", "reason_class": "torch_redundancy"},
            {"status": "ok"},  # final_status arg doesn't reflect this, but history does
        ],
    )
    # Even though final_status was passed as preflight_blocked, history shows
    # a successful run — full credit (matches the "we caught it, agent fixed it" rule).
    assert j.effective_status == "preflight_caught_fixed"
    assert j.credit == 1.0


def test_syntax_error_controllable():
    j = classify_experiment(final_status="failed", final_reason_class="syntax_error")
    assert j.effective_status == "controllable_fail"
    assert j.credit == 0.0
    assert j.in_denominator is True


def test_torch_redundancy_controllable():
    j = classify_experiment(final_status="failed", final_reason_class="torch_redundancy")
    assert j.in_denominator is True


def test_unknown_class_defaults_controllable():
    """Default: be strict. Unknown failure_class → controllable."""
    j = classify_experiment(final_status="something_weird", final_reason_class="???")
    assert j.in_denominator is True


def test_taxonomy_no_overlap():
    """A class can't be both controllable and uncontrollable."""
    overlap = CONTROLLABLE_CLASSES & UNCONTROLLABLE_CLASSES
    assert overlap == set()


# ---------------------------------------------------------------------------
# compute_scope_adjusted_rubric — end-to-end
# ---------------------------------------------------------------------------


def test_all_experiments_ok_full_score():
    """Every experiment ran cleanly → scope-adjusted == raw score."""
    experiments = {
        "mnist_mlp": {"status": "ok"},
        "imdb":      {"status": "ok"},
        "cifar":     {"status": "ok"},
    }
    leaves = {
        "L1": {"score": 0.8, "weight": 1.0, "experiment": "mnist_mlp"},
        "L2": {"score": 0.7, "weight": 1.0, "experiment": "imdb"},
        "L3": {"score": 0.6, "weight": 1.0, "experiment": "cifar"},
    }
    r = compute_scope_adjusted_rubric(
        experiments=experiments, leaf_scores=leaves, target_score=0.6,
    )
    # Mean of 0.8, 0.7, 0.6 = 0.7
    assert abs(r.overall_score - 0.7) < 1e-6
    assert r.coverage == 1.0
    assert r.insufficient_coverage is False
    assert r.meets_target is True


def test_uncontrollable_skip_shrinks_denominator():
    """IMDB unavailable → its leaf is excluded; remaining 2 leaves average."""
    experiments = {
        "mnist_mlp": {"status": "ok"},
        "imdb":      {"status": "data_unavailable", "reason_class": "missing_dataset"},
        "cifar":     {"status": "ok"},
    }
    leaves = {
        "L_mnist": {"score": 0.9, "weight": 1.0, "experiment": "mnist_mlp"},
        "L_imdb":  {"score": 0.0, "weight": 1.0, "experiment": "imdb"},  # would-be 0
        "L_cifar": {"score": 0.7, "weight": 1.0, "experiment": "cifar"},
    }
    r = compute_scope_adjusted_rubric(
        experiments=experiments, leaf_scores=leaves, target_score=0.6,
    )
    # Mean of 0.9, 0.7 = 0.8 (IMDB excluded)
    assert abs(r.overall_score - 0.8) < 1e-6
    # 2 of 3 = 0.66 coverage, above floor
    assert r.coverage == pytest.approx(2/3, rel=1e-3)
    assert r.insufficient_coverage is False


def test_controllable_failure_keeps_denominator_zeros_leaf():
    """Tensor-device-mismatch in vae → its leaf counts but scores 0."""
    experiments = {
        "mnist": {"status": "ok"},
        "imdb":  {"status": "ok"},
        "vae":   {"status": "code_error", "reason_class": "tensor_device_mismatch"},
    }
    leaves = {
        "L_mnist": {"score": 1.0, "weight": 1.0, "experiment": "mnist"},
        "L_imdb":  {"score": 1.0, "weight": 1.0, "experiment": "imdb"},
        "L_vae":   {"score": 0.0, "weight": 1.0, "experiment": "vae"},
    }
    r = compute_scope_adjusted_rubric(
        experiments=experiments, leaf_scores=leaves, target_score=0.6,
    )
    # Mean of 1.0, 1.0, 0.0 = 0.666
    assert abs(r.overall_score - 2/3) < 1e-3
    assert r.coverage == pytest.approx(2/3, rel=1e-3)
    # Above 50% floor.
    assert r.insufficient_coverage is False


def test_preflight_unresolved_half_credit_applied():
    """A preflight-unresolved experiment leaf gets 0.5 × its score."""
    experiments = {
        "mnist": {"status": "ok"},
        "x":     {"status": "preflight_blocked", "reason_class": "preflight_blocked"},
    }
    leaves = {
        "L_mnist": {"score": 1.0, "weight": 1.0, "experiment": "mnist"},
        # If preflight_blocked unresolved → credit=0.5; score would be 0.6 × 0.5 = 0.3
        "L_x":     {"score": 0.6, "weight": 1.0, "experiment": "x"},
    }
    r = compute_scope_adjusted_rubric(
        experiments=experiments, leaf_scores=leaves, target_score=0.5,
    )
    # numerator = 1.0*1.0 + 0.6*0.5 = 1.3; denominator = 2.0; score = 0.65
    assert abs(r.overall_score - 0.65) < 1e-3


def test_insufficient_coverage_caps_score():
    """1 of 4 experiments ran → 25% coverage → cap at 0.30."""
    experiments = {
        "a": {"status": "ok"},
        "b": {"status": "data_unavailable", "reason_class": "missing_dataset"},
        "c": {"status": "data_unavailable", "reason_class": "missing_dataset"},
        "d": {"status": "data_unavailable", "reason_class": "missing_dataset"},
    }
    leaves = {
        "L_a": {"score": 1.0, "weight": 1.0, "experiment": "a"},
        "L_b": {"score": 0.0, "weight": 1.0, "experiment": "b"},
        "L_c": {"score": 0.0, "weight": 1.0, "experiment": "c"},
        "L_d": {"score": 0.0, "weight": 1.0, "experiment": "d"},
    }
    r = compute_scope_adjusted_rubric(
        experiments=experiments, leaf_scores=leaves, target_score=0.6,
    )
    # Raw would be 1.0 (only 'a' counts after excluding 3 uncontrollables),
    # but 25% coverage triggers the cap.
    assert r.insufficient_coverage is True
    assert r.overall_score == 0.30
    assert "insufficient_coverage" in r.notes


def test_50_percent_coverage_is_above_floor():
    """Exactly 50% ran → coverage gate NOT triggered (>= 0.50)."""
    experiments = {
        "a": {"status": "ok"},
        "b": {"status": "ok"},
        "c": {"status": "data_unavailable", "reason_class": "missing_dataset"},
        "d": {"status": "data_unavailable", "reason_class": "missing_dataset"},
    }
    leaves = {
        "L_a": {"score": 1.0, "weight": 1.0, "experiment": "a"},
        "L_b": {"score": 1.0, "weight": 1.0, "experiment": "b"},
        "L_c": {"score": 0.0, "weight": 1.0, "experiment": "c"},
        "L_d": {"score": 0.0, "weight": 1.0, "experiment": "d"},
    }
    r = compute_scope_adjusted_rubric(
        experiments=experiments, leaf_scores=leaves, target_score=0.6,
    )
    assert r.coverage == 0.5
    assert r.insufficient_coverage is False  # floor is strict <0.5
    assert r.overall_score == 1.0  # both included leaves are 1.0


def test_leaf_without_experiment_always_counts():
    """Paper-wide leaves (no experiment mapping) always count in denominator.

    Uses 2 experiments (1 ok, 1 uncontrollable_skip) so coverage = 50% and
    the gate doesn't fire — isolates the 'paper-wide leaves count' behavior.
    """
    experiments = {
        "a": {"status": "ok"},
        "b": {"status": "data_unavailable", "reason_class": "missing_dataset"},
    }
    leaves = {
        "L_paperwide": {"score": 0.5, "weight": 1.0},  # no experiment field
        "L_a":         {"score": 1.0, "weight": 1.0, "experiment": "a"},
        "L_b":         {"score": 0.0, "weight": 1.0, "experiment": "b"},  # excluded
    }
    r = compute_scope_adjusted_rubric(
        experiments=experiments, leaf_scores=leaves, target_score=0.5,
    )
    # L_b excluded; mean of L_paperwide (0.5) + L_a (1.0) = 0.75
    assert r.overall_score == 0.75


def test_empty_experiments_no_classification():
    """No experiments declared → no judgements, coverage defaults to 1.0."""
    r = compute_scope_adjusted_rubric(
        experiments={},
        leaf_scores={"L1": {"score": 0.7, "weight": 1.0}},
        target_score=0.5,
    )
    assert r.judgements == []
    assert r.coverage == 1.0
    assert r.insufficient_coverage is False
    assert r.overall_score == 0.7


def test_zero_weight_leaves_dont_crash():
    """A leaf with weight=0 must not divide-by-zero."""
    experiments = {"a": {"status": "ok"}}
    leaves = {"L": {"score": 0.5, "weight": 0.0, "experiment": "a"}}
    r = compute_scope_adjusted_rubric(
        experiments=experiments, leaf_scores=leaves, target_score=0.5,
    )
    assert r.overall_score == 0.0  # zero denominator → 0


def test_real_adam_regression_pattern():
    """The 2026-05-25 Adam scenario: 5 experiments, 1 ran, 4 failed for various
    reasons. Without scope adjustment Adam scored 0.0/0.6. With it:
      * mnist_mlp:   ran (full credit)
      * imdb:        data_unavailable (uncontrollable — exclude)
      * cifar:       data_unavailable (uncontrollable — exclude)
      * vae:         tensor_device_mismatch (controllable — score 0)
      * mnist_lr:    preflight_unresolved (half credit)
    Coverage: 2/5 = 40% → INSUFFICIENT_COVERAGE, cap at 0.30.
    """
    experiments = {
        "mnist_mlp": {"status": "ok"},
        "imdb":      {"status": "data_unavailable", "reason_class": "missing_dataset"},
        "cifar":     {"status": "data_unavailable", "reason_class": "missing_dataset"},
        "vae":       {"status": "code_error", "reason_class": "tensor_device_mismatch"},
        "mnist_lr":  {"status": "preflight_blocked", "reason_class": "preflight_blocked"},
    }
    leaves = {
        "L_mnist":    {"score": 0.9, "weight": 1.0, "experiment": "mnist_mlp"},
        "L_imdb":     {"score": 0.0, "weight": 1.0, "experiment": "imdb"},
        "L_cifar":    {"score": 0.0, "weight": 1.0, "experiment": "cifar"},
        "L_vae":      {"score": 0.0, "weight": 1.0, "experiment": "vae"},
        "L_mnist_lr": {"score": 0.4, "weight": 1.0, "experiment": "mnist_lr"},
    }
    r = compute_scope_adjusted_rubric(
        experiments=experiments, leaf_scores=leaves, target_score=0.6,
    )
    # Only ran: mnist_mlp (1 of 5 = 20% coverage — but preflight_caught_fixed
    # doesn't increment ran here because mnist_lr wasn't fixed, so 1/5 = 20%).
    assert r.coverage == 0.2
    assert r.insufficient_coverage is True
    assert r.overall_score == 0.30  # capped
    # Five judgements, one each.
    assert len(r.judgements) == 5
