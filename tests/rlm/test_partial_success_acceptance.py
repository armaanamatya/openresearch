"""β1: Partial-success contract — verify that partial evidence (success=False but
metrics non-empty) does NOT trigger the degraded cap in verify_against_rubric.

Today's VAE crash on WakeSleepVAE.reparameterize is the motivator: 95% of the
science completes and metrics are captured, then the script crashes. The old
binary degraded gate capped every leaf at 0.35. The tri-state contract lets
the grader see the real metrics.
"""

from __future__ import annotations

from backend.agents.rlm.primitives import _classify_run_experiment_outcome, PrimitiveOutcome


# ---------------------------------------------------------------------------
# Outcome classifier tests (tri-state source)
# ---------------------------------------------------------------------------


def test_success_true_classifies_ok():
    result = {"success": True, "metrics": {"loss": 0.42}}
    assert _classify_run_experiment_outcome(result) == PrimitiveOutcome.ok


def test_success_false_with_metrics_classifies_partial_evidence():
    """success=False but metrics dict non-empty → partial_evidence, NOT repairable."""
    result = {"success": False, "metrics": {"loss": 0.42, "acc": 0.88}}
    assert _classify_run_experiment_outcome(result) == PrimitiveOutcome.partial_evidence


def test_success_false_no_metrics_classifies_repairable():
    """success=False with empty metrics → repairable (no evidence at all)."""
    result = {"success": False, "metrics": {}}
    assert _classify_run_experiment_outcome(result) == PrimitiveOutcome.repairable


def test_success_false_metrics_none_classifies_repairable():
    result = {"success": False, "metrics": None}
    assert _classify_run_experiment_outcome(result) == PrimitiveOutcome.repairable


def test_success_false_no_metrics_key_classifies_repairable():
    result = {"success": False}
    assert _classify_run_experiment_outcome(result) == PrimitiveOutcome.repairable


# ---------------------------------------------------------------------------
# degraded flag logic tests (as computed in verify_against_rubric)
# ---------------------------------------------------------------------------


def _compute_degraded(results: dict) -> bool:
    """Mirror the tri-state degraded predicate from verify_against_rubric."""
    has_experiment_result = "success" in results or "metrics" in results
    metrics_present = bool(results.get("metrics") or {})
    return has_experiment_result and (
        (results.get("success") is False) and (not metrics_present)
    )


def test_degraded_false_when_success_true():
    assert _compute_degraded({"success": True, "metrics": {"loss": 0.5}}) is False


def test_degraded_false_when_partial_evidence():
    """The key β1 assertion: partial evidence → degraded=False."""
    assert _compute_degraded({"success": False, "metrics": {"loss": 0.5, "acc": 0.8}}) is False


def test_degraded_true_when_failed_no_metrics():
    assert _compute_degraded({"success": False, "metrics": {}}) is True


def test_degraded_true_when_failed_metrics_none():
    assert _compute_degraded({"success": False, "metrics": None}) is True


def test_degraded_false_when_no_success_key_present():
    """No experiment result key at all (e.g. pre-run verify call) → not degraded."""
    assert _compute_degraded({}) is False


def test_degraded_false_when_only_metrics_present_no_success_key():
    """Metrics present but no success key → not degraded (treat as partial evidence)."""
    assert _compute_degraded({"metrics": {"acc": 0.9}}) is False
