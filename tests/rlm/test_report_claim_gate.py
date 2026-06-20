"""Tests for report_claim_gate.py — §4.3 of the pre-GPU design spec."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend.agents.rlm.report_claim_gate import (
    apply_report_claim_gate,
    report_claim_gate_enabled,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_report(
    reproduction_summary: str = "",
    reported_metrics: dict | str | None = None,
    verdict: str = "reproduced",
):
    return SimpleNamespace(
        reproduction_summary=reproduction_summary,
        reported_metrics=reported_metrics,
        verdict=verdict,
        rubric={"overall_score": 0.8, "target_score": 0.6, "meets_target": True},
        stop_reason=None,
        degraded=False,
    )


def _make_dict(verdict: str = "reproduced", overall: float = 0.8, target: float = 0.6) -> dict:
    return {
        "verdict": verdict,
        "reproduction_summary": "We achieved accuracy 0.84.",
        "reported_metrics": None,
        "rubric": {"overall_score": overall, "target_score": target, "meets_target": overall >= target},
        "overall_score": overall,
        "target_score": target,
        "meets_target": overall >= target,
    }


def _write_metrics(tmp_path: Path, metrics: dict) -> None:
    code_dir = tmp_path / "code"
    code_dir.mkdir(exist_ok=True)
    (code_dir / "metrics.json").write_text(json.dumps(metrics))


# ---------------------------------------------------------------------------
# Flag tests
# ---------------------------------------------------------------------------

def test_flag_off_by_default():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("OPENRESEARCH_REPORT_CLAIM_GATE", None)
        assert not report_claim_gate_enabled()


def test_flag_on():
    with patch.dict(os.environ, {"OPENRESEARCH_REPORT_CLAIM_GATE": "1"}):
        assert report_claim_gate_enabled()


# ---------------------------------------------------------------------------
# A — ungrounded claim + measured present → cap + cite + stamp
# ---------------------------------------------------------------------------

def test_ungrounded_claim_caps_verdict(tmp_path):
    # Report claims accuracy 0.84 but metrics.json has no accuracy
    _write_metrics(tmp_path, {"loss": 0.12})
    report = _make_report(reproduction_summary="We achieved accuracy 0.84.")
    d = _make_dict("reproduced")
    d["reproduction_summary"] = "We achieved accuracy 0.84."
    result = apply_report_claim_gate(report, d, tmp_path)

    assert result["verdict"] == "partial", "verdict should be capped to partial"
    assert "claim_grounding" in result
    assert result["claim_grounding"]["verdict_capped"] is True
    assert result["claim_grounding"]["ungrounded"] >= 1
    assert "harness" in result["reproduction_summary"]


def test_cap_syncs_to_report_object(tmp_path):
    # codex Area-4: the cap must propagate to the report OBJECT, not just the dict —
    # final_report.md and the best-attempt marker render from the object, so without
    # the sync they would show a stale 'reproduced' while final_report.json says 'partial'.
    _write_metrics(tmp_path, {"loss": 0.12})
    report = _make_report(reproduction_summary="We achieved accuracy 0.84.")
    report.reproducibility = {"verdict": "reproduced"}
    report.meets_target = True
    d = _make_dict("reproduced")
    d["reproduction_summary"] = "We achieved accuracy 0.84."
    d["reproducibility"] = {"verdict": "reproduced"}
    apply_report_claim_gate(report, d, tmp_path)

    assert report.verdict == "partial", "report OBJECT verdict must be synced to the cap"
    assert "harness" in report.reproduction_summary
    assert report.reproducibility["verdict"] == "partial"


def test_ungrounded_claim_stamp_key_present(tmp_path):
    _write_metrics(tmp_path, {"loss": 0.12})
    report = _make_report(reproduction_summary="accuracy 0.90 achieved")
    d = _make_dict("reproduced")
    d["reproduction_summary"] = "accuracy 0.90 achieved"
    result = apply_report_claim_gate(report, d, tmp_path)
    assert "claim_grounding" in result


def test_grounded_claim_leaves_verdict_unchanged(tmp_path):
    """When claimed accuracy matches measured accuracy within 5%, no cap."""
    # flatten_measured_values: {"accuracy": 0.84} → [("accuracy", 0.84)]
    # extract_result_claims("accuracy 0.84") → [Claim(0.84, "accuracy", ...)]
    # check_claims_grounded: _canonical("accuracy") == _canonical("accuracy") → grounded
    _write_metrics(tmp_path, {"accuracy": 0.84, "loss": 0.1})
    report = _make_report(reproduction_summary="We achieved accuracy 0.84.")
    d = _make_dict("reproduced")
    d["reproduction_summary"] = "We achieved accuracy 0.84."
    result = apply_report_claim_gate(report, d, tmp_path)
    assert result["verdict"] == "reproduced"
    assert result["claim_grounding"]["verdict_capped"] is False


def test_codex_7_loss_does_not_ground_accuracy(tmp_path):
    """loss=0.84 must NOT ground a claimed accuracy=0.84 (identity mismatch)."""
    # flatten: {"loss": 0.84} → [("loss", 0.84)]; _canonical("loss") = "loss" ≠ "accuracy"
    _write_metrics(tmp_path, {"loss": 0.84})
    report = _make_report(reproduction_summary="We achieved accuracy 0.84.")
    d = _make_dict("reproduced")
    d["reproduction_summary"] = "We achieved accuracy 0.84."
    result = apply_report_claim_gate(report, d, tmp_path)
    assert result["verdict"] == "partial"
    assert result["claim_grounding"]["verdict_capped"] is True


def test_no_measured_evidence_noop(tmp_path):
    """When code/metrics.json does not exist, gate is a no-op."""
    report = _make_report(reproduction_summary="We achieved accuracy 0.84.")
    d = _make_dict("reproduced")
    d["reproduction_summary"] = "We achieved accuracy 0.84."
    result = apply_report_claim_gate(report, d, tmp_path)
    assert result["verdict"] == "reproduced"  # no-op
    assert result["claim_grounding"]["verdict_capped"] is False


def test_hyperparameter_not_flagged(tmp_path):
    """A number adjacent to a hyperparameter word is not a result claim."""
    _write_metrics(tmp_path, {"loss": 0.01})
    # "learning_rate 0.001" should not be treated as an accuracy claim
    report = _make_report(reproduction_summary="Set learning_rate 0.001 and achieved loss 0.01.")
    d = _make_dict("reproduced")
    d["reproduction_summary"] = "Set learning_rate 0.001 and achieved loss 0.01."
    result = apply_report_claim_gate(report, d, tmp_path)
    # learning_rate → config term, loss → not in _RESULT_TERMS_RE, so no claims extracted
    # verdict should be unchanged (no ungrounded result claims)
    assert result["verdict"] == "reproduced"
    assert result["claim_grounding"]["verdict_capped"] is False


def test_gate_always_stamps_when_called(tmp_path):
    """The gate function always adds claim_grounding stamp when called."""
    _write_metrics(tmp_path, {"accuracy": 0.84})
    report = _make_report(reproduction_summary="accuracy 0.84")
    d = _make_dict("reproduced")
    d["reproduction_summary"] = "accuracy 0.84"
    result = apply_report_claim_gate(report, d, tmp_path)
    # grounded claim → no cap, but stamp is present
    assert "claim_grounding" in result
    assert result["claim_grounding"]["verdict_capped"] is False


def test_codex_6_gate_runs_after_floor(tmp_path):
    """Verifies the gate correctly caps even when called last.
    In report.py the floor runs BEFORE the gate (gate is last).
    This test verifies the gate result is authoritative."""
    _write_metrics(tmp_path, {"loss": 0.12})
    report = _make_report(reproduction_summary="We achieved accuracy 0.84.")
    d = _make_dict("reproduced")
    d["reproduction_summary"] = "We achieved accuracy 0.84."
    # Gate caps to partial
    result = apply_report_claim_gate(report, d, tmp_path)
    assert result["verdict"] == "partial"
    assert result["claim_grounding"]["verdict_capped"] is True


def test_partial_verdict_also_gets_note(tmp_path):
    """Even a 'partial' verdict gets the cited note when ungrounded claims exist."""
    _write_metrics(tmp_path, {"loss": 0.12})
    report = _make_report(
        reproduction_summary="We achieved accuracy 0.84.",
        verdict="partial",
    )
    d = _make_dict("partial")
    d["reproduction_summary"] = "We achieved accuracy 0.84."
    result = apply_report_claim_gate(report, d, tmp_path)
    assert result["verdict"] == "partial"
    assert "harness" in result["reproduction_summary"]


def test_failed_verdict_no_cap(tmp_path):
    """'failed' verdict is never changed, and the gate doesn't touch it."""
    _write_metrics(tmp_path, {"loss": 0.12})
    report = _make_report(reproduction_summary="We achieved accuracy 0.84.", verdict="failed")
    d = _make_dict("failed")
    d["reproduction_summary"] = "We achieved accuracy 0.84."
    result = apply_report_claim_gate(report, d, tmp_path)
    assert result["verdict"] == "failed"  # already failed, no cap needed
    assert result["claim_grounding"]["verdict_capped"] is False


def test_two_axis_repro_also_capped(tmp_path):
    """When a two-axis reproducibility dict is present, its verdict is also capped."""
    _write_metrics(tmp_path, {"loss": 0.12})
    report = _make_report(reproduction_summary="accuracy 0.90 achieved.")
    d = _make_dict("reproduced")
    d["reproduction_summary"] = "accuracy 0.90 achieved."
    d["reproducibility"] = {"verdict": "reproduced", "implementation_verdict": "reproduced"}
    result = apply_report_claim_gate(report, d, tmp_path)
    assert result["verdict"] == "partial"
    repro = result.get("reproducibility", {})
    assert repro.get("verdict") == "partial"
    assert repro.get("implementation_verdict") == "partial"


def test_fail_soft_on_exception(tmp_path):
    """Any exception in the gate is caught; dict is returned."""
    report = _make_report()
    d = _make_dict("reproduced")
    # Pass a non-existent path — flatten_measured_values fails soft
    result = apply_report_claim_gate(report, d, Path("/nonexistent/path/xyz"))
    # Fail-soft: dict is returned (verdict unchanged)
    assert "verdict" in result


def test_reward_claim_grounded(tmp_path):
    """A reward claim is grounded when metrics.json has a matching reward value."""
    # flatten: {"mean_reward": 0.72} → [("reward", 0.72)]
    # claim from "reward 0.72": term="reward", _canonical("reward")="reward" ✓
    _write_metrics(tmp_path, {"mean_reward": 0.72})
    report = _make_report(reproduction_summary="Our model achieves reward 0.72 on evaluation.")
    d = _make_dict("reproduced")
    d["reproduction_summary"] = "Our model achieves reward 0.72 on evaluation."
    result = apply_report_claim_gate(report, d, tmp_path)
    assert result["verdict"] == "reproduced"
    assert result["claim_grounding"]["verdict_capped"] is False
