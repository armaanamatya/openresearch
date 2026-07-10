"""Tests for result_fidelity.evaluate — the deterministic per-claim checker (Task 2, §4.2).

The first four tests are the brief's required verbatim cases (numeric pass/fail
+ the ambiguous/unbound asymmetry). The rest exercise the remaining kinds
(relative/trend/qualitative), the sign-fold, and the aggregate fields the
"Design (locked)" section also specifies but does not hand over literal test
code for.
"""
import json

from backend.agents.rlm.result_fidelity import evaluate


def _run(tmp_path, metrics):
    (tmp_path / "code").mkdir(parents=True, exist_ok=True)
    (tmp_path / "code" / "metrics.json").write_text(json.dumps(metrics))
    return tmp_path


def _claim(**kw):
    base = {"claim_id": "primary_0", "is_primary": True, "kind": "numeric",
            "metric_name": "accuracy", "estimate_kind": "point",
            "claimed_effect": 0.99, "equivalence_margin": 0.01,
            "direction": "higher_is_better", "scope": {}, "ambiguous": False}
    base.update(kw); return base


# --------------------------------------------------------------------------- #
# Required verbatim tests (task brief Step 1)
# --------------------------------------------------------------------------- #

def test_numeric_within_margin_passes(tmp_path):
    run = _run(tmp_path, {"accuracy": 0.991})
    rf = evaluate({"claims": [_claim()]}, run)
    assert rf["per_claim"][0]["status"] == "pass" and rf["any_contradicted"] is False

def test_numeric_outside_margin_fails_only_with_verified_bind(tmp_path):
    run = _run(tmp_path, {"accuracy": 0.80})
    rf = evaluate({"claims": [_claim()]}, run)
    assert rf["per_claim"][0]["status"] == "fail" and rf["any_contradicted"] is True

def test_ambiguous_claim_is_unmeasured_never_fail(tmp_path):
    run = _run(tmp_path, {"accuracy": 0.10})
    rf = evaluate({"claims": [_claim(ambiguous=True)]}, run)
    assert rf["per_claim"][0]["status"] == "unmeasured" and rf["any_contradicted"] is False

def test_unbound_metric_is_unmeasured(tmp_path):
    run = _run(tmp_path, {"other": 1.0})
    rf = evaluate({"claims": [_claim(metric_name="nope")]}, run)
    assert rf["per_claim"][0]["status"] == "unmeasured"


# --------------------------------------------------------------------------- #
# qualitative — never measured, even when the metric is bindable
# --------------------------------------------------------------------------- #

def test_qualitative_claim_is_always_unmeasured(tmp_path):
    run = _run(tmp_path, {"accuracy": 0.99})
    rf = evaluate({"claims": [_claim(kind="qualitative")]}, run)
    assert rf["per_claim"][0]["status"] == "unmeasured"
    assert rf["per_claim"][0]["reason"] == "qualitative_claim_not_measurable"


# --------------------------------------------------------------------------- #
# relative — ordering + magnitude on the sign-folded claimed_effect
# --------------------------------------------------------------------------- #

def test_relative_within_margin_passes(tmp_path):
    run = _run(tmp_path, {"accuracy": 0.844})
    claim = _claim(kind="relative", claimed_effect=0.10, equivalence_margin=0.02,
                    baseline_value=0.75, direction="higher_is_better")
    rf = evaluate({"claims": [claim]}, run)
    assert rf["per_claim"][0]["status"] == "pass"
    assert abs(rf["per_claim"][0]["measured"] - 0.094) < 1e-9

def test_relative_missing_baseline_value_is_unmeasured_not_fail(tmp_path):
    run = _run(tmp_path, {"accuracy": 0.844})
    claim = _claim(kind="relative", claimed_effect=0.10, equivalence_margin=0.02)
    rf = evaluate({"claims": [claim]}, run)
    assert rf["per_claim"][0]["status"] == "unmeasured"
    assert rf["per_claim"][0]["reason"] == "missing_baseline_value"
    assert rf["any_contradicted"] is False

def test_relative_lower_is_better_sign_fold_fails_outside_margin(tmp_path):
    # loss improves (1.5 -> 1.45, a real but small 0.05 advantage) but the claim
    # asserts a much bigger 0.20 advantage -> ordering holds, magnitude does not.
    run = _run(tmp_path, {"loss": 1.45})
    claim = _claim(kind="relative", metric_name="loss", direction="lower_is_better",
                    claimed_effect=0.20, equivalence_margin=0.02, baseline_value=1.5)
    rf = evaluate({"claims": [claim]}, run)
    assert rf["per_claim"][0]["status"] == "fail"
    assert abs(rf["per_claim"][0]["measured"] - 0.05) < 1e-9
    assert rf["any_contradicted"] is True


# --------------------------------------------------------------------------- #
# trend — sign/slope of a history.* curve vs. the claimed direction
# --------------------------------------------------------------------------- #

def test_trend_matching_slope_passes(tmp_path):
    run = _run(tmp_path, {"loss": 1.5, "history": {"loss": [2.0, 1.7, 1.5]}})
    claim = _claim(kind="trend", metric_name="loss", direction="lower_is_better",
                    claimed_effect=0.5, equivalence_margin=0.05)
    rf = evaluate({"claims": [claim]}, run)
    assert rf["per_claim"][0]["status"] == "pass"

def test_trend_insufficient_points_is_unmeasured(tmp_path):
    run = _run(tmp_path, {"loss": 1.5, "history": {"loss": [2.0]}})
    claim = _claim(kind="trend", metric_name="loss", direction="lower_is_better",
                    claimed_effect=0.5, equivalence_margin=0.05)
    rf = evaluate({"claims": [claim]}, run)
    assert rf["per_claim"][0]["status"] == "unmeasured"
    assert rf["per_claim"][0]["reason"] == "insufficient_history_points"


# --------------------------------------------------------------------------- #
# Unrecognized kind -> unmeasured, never a guess
# --------------------------------------------------------------------------- #

def test_unknown_kind_is_unmeasured(tmp_path):
    run = _run(tmp_path, {"accuracy": 0.991})
    rf = evaluate({"claims": [_claim(kind="mystery")]}, run)
    assert rf["per_claim"][0]["status"] == "unmeasured"
    assert rf["per_claim"][0]["reason"] == "unknown_kind:mystery"


# --------------------------------------------------------------------------- #
# Aggregate fields: primary/secondary weighting, primary_all_measured
# --------------------------------------------------------------------------- #

def test_aggregate_score_weights_secondary_lower_and_flags_contradiction(tmp_path):
    run = _run(tmp_path, {"accuracy": 0.991, "f1": 0.50, "recall": 0.501})
    claims = [
        _claim(claim_id="primary_0", is_primary=True, metric_name="accuracy",
               claimed_effect=0.99, equivalence_margin=0.01),
        _claim(claim_id="primary_1", is_primary=True, metric_name="f1",
               claimed_effect=0.80, equivalence_margin=0.01),
        _claim(claim_id="secondary_0", is_primary=False, metric_name="recall",
               claimed_effect=0.50, equivalence_margin=0.01),
    ]
    rf = evaluate({"claims": claims}, run)
    statuses = {c["claim_id"]: c["status"] for c in rf["per_claim"]}
    assert statuses == {"primary_0": "pass", "primary_1": "fail", "secondary_0": "pass"}
    # weights: primary=1.0 each, secondary=0.5 -> pass_weight=1.0+0.5=1.5 / total=2.5
    assert abs(rf["result_fidelity_score"] - 0.6) < 1e-9
    assert rf["primary_all_measured"] is True
    assert rf["any_contradicted"] is True

def test_no_primary_claims_is_not_all_measured(tmp_path):
    run = _run(tmp_path, {"recall": 0.501})
    claim = _claim(claim_id="secondary_0", is_primary=False, metric_name="recall",
                    claimed_effect=0.50, equivalence_margin=0.01)
    rf = evaluate({"claims": [claim]}, run)
    assert rf["primary_all_measured"] is False


# --------------------------------------------------------------------------- #
# Never raises on degenerate input
# --------------------------------------------------------------------------- #

def test_malformed_repro_spec_never_raises(tmp_path):
    run = _run(tmp_path, {"accuracy": 0.99})
    assert evaluate(None, run) == {
        "per_claim": [], "result_fidelity_score": 0.0,
        "primary_all_measured": False, "any_contradicted": False,
    }
    assert evaluate({"claims": "not-a-list"}, run) == {
        "per_claim": [], "result_fidelity_score": 0.0,
        "primary_all_measured": False, "any_contradicted": False,
    }

def test_missing_run_dir_never_raises(tmp_path):
    rf = evaluate({"claims": [_claim()]}, tmp_path / "does_not_exist")
    assert rf["per_claim"][0]["status"] == "unmeasured"
