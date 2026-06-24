"""Tests for the report_claims_grounded validator predicate — §4.5 of the design spec."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from backend.agents.rlm.claim_grounding import Claim
from backend.agents.rlm.external_validator import (
    _machine_check,
    _VALID_PREDICATES,
    check_report_claims_grounded,
    run_validation_panel,
)


# ---------------------------------------------------------------------------
# Predicate registration
# ---------------------------------------------------------------------------

def test_report_claims_grounded_in_valid_predicates():
    assert "report_claims_grounded" in _VALID_PREDICATES


# ---------------------------------------------------------------------------
# check_report_claims_grounded
# ---------------------------------------------------------------------------

def test_empty_claims_returns_clean(tmp_path):
    assert check_report_claims_grounded(None, tmp_path) is True
    assert check_report_claims_grounded([], tmp_path) is True


def test_no_measured_returns_clean(tmp_path):
    """No code/metrics.json → unverifiable → clean (evidence_gate owns this)."""
    claims = [Claim(value=0.84, term="accuracy", context="accuracy 0.84")]
    assert check_report_claims_grounded(claims, tmp_path) is True


def test_grounded_claim_returns_clean(tmp_path):
    """accuracy claim grounded by {"accuracy": 0.84} in metrics.json.

    flatten_measured_values: {"accuracy": 0.84} → [("accuracy", 0.84)]
    _canonical("accuracy") == _canonical("accuracy") == "accuracy" → grounded.
    """
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "metrics.json").write_text(json.dumps({"accuracy": 0.84}))
    claims = [Claim(value=0.84, term="accuracy", context="accuracy 0.84")]
    assert check_report_claims_grounded(claims, tmp_path) is True


def test_ungrounded_claim_returns_violated(tmp_path):
    """accuracy claim NOT grounded when metrics.json has only loss."""
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "metrics.json").write_text(json.dumps({"loss": 0.12}))
    claims = [Claim(value=0.84, term="accuracy", context="accuracy 0.84")]
    assert check_report_claims_grounded(claims, tmp_path) is False


def test_identity_mismatch_not_grounded(tmp_path):
    """loss=0.84 does NOT ground accuracy=0.84 (identity mismatch, codex-7)."""
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "metrics.json").write_text(json.dumps({"loss": 0.84}))
    claims = [Claim(value=0.84, term="accuracy", context="accuracy 0.84")]
    assert check_report_claims_grounded(claims, tmp_path) is False


def test_reward_claim_grounded_by_mean_reward(tmp_path):
    """reward claim grounded by {'mean_reward': 0.72}.

    flatten: mean_reward → last segment "reward"; _canonical("reward") == "reward" ✓
    """
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "metrics.json").write_text(json.dumps({"mean_reward": 0.72}))
    claims = [Claim(value=0.72, term="reward", context="reward 0.72")]
    assert check_report_claims_grounded(claims, tmp_path) is True


# ---------------------------------------------------------------------------
# _machine_check dispatch
# ---------------------------------------------------------------------------

def test_machine_check_report_claims_grounded_clean(tmp_path):
    """accuracy claim grounded → predicate not violated."""
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "metrics.json").write_text(json.dumps({"accuracy": 0.84}))
    claims = [Claim(value=0.84, term="accuracy", context="accuracy 0.84")]
    verdict = _machine_check(
        "report_claims_grounded", "accuracy", {}, tmp_path, {}, report_claims=claims
    )
    assert verdict.violated is False
    assert verdict.predicate == "report_claims_grounded"


def test_machine_check_report_claims_grounded_violated(tmp_path):
    """accuracy claim NOT grounded when only loss is in metrics → violated."""
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "metrics.json").write_text(json.dumps({"loss": 0.12}))
    claims = [Claim(value=0.84, term="accuracy", context="accuracy 0.84")]
    verdict = _machine_check(
        "report_claims_grounded", "accuracy", {}, tmp_path, {}, report_claims=claims
    )
    assert verdict.violated is True
    assert "absent" in verdict.detail or "claims" in verdict.detail


def test_machine_check_no_report_claims_returns_clean(tmp_path):
    """When report_claims=None, the predicate is not violated (nothing to check)."""
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "metrics.json").write_text(json.dumps({"loss": 0.12}))
    verdict = _machine_check(
        "report_claims_grounded", "accuracy", {}, tmp_path, {}, report_claims=None
    )
    assert verdict.violated is False


# ---------------------------------------------------------------------------
# run_validation_panel — report_claims threading
# ---------------------------------------------------------------------------

def test_run_validation_panel_report_claims_threading(tmp_path):
    """When the panelist suspects report_claims_grounded and report_claims has an
    ungrounded claim, the machine-check vetos."""
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "metrics.json").write_text(json.dumps({"loss": 0.12}))

    # accuracy claim NOT grounded by {"loss": 0.12}
    claims = [Claim(value=0.84, term="accuracy", context="accuracy 0.84")]

    mock_client = MagicMock()
    with patch("backend.agents.rlm.grader_transport.sample_completions") as mock_sample:
        mock_sample.return_value = [
            '[{"predicate": "report_claims_grounded", "metric_ref": "accuracy"}]'
        ]
        verdict = run_validation_panel(
            validator_client=mock_client,
            panel_models=["test-model"],
            metrics={"loss": 0.12},
            project_dir=tmp_path,
            leaf_records=[],
            separation="independent",
            report_claims=claims,
        )
    assert verdict.status == "vetoed"
    assert "accuracy" in verdict.veto_set


def test_run_validation_panel_grounded_claims_not_vetoed(tmp_path):
    """When claims are grounded, the predicate is not violated."""
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "metrics.json").write_text(json.dumps({"accuracy": 0.84}))

    # accuracy claim grounded by {"accuracy": 0.84}
    claims = [Claim(value=0.84, term="accuracy", context="accuracy 0.84")]

    mock_client = MagicMock()
    with patch("backend.agents.rlm.grader_transport.sample_completions") as mock_sample:
        mock_sample.return_value = [
            '[{"predicate": "report_claims_grounded", "metric_ref": "accuracy"}]'
        ]
        verdict = run_validation_panel(
            validator_client=mock_client,
            panel_models=["test-model"],
            metrics={"accuracy": 0.84},
            project_dir=tmp_path,
            leaf_records=[],
            separation="independent",
            report_claims=claims,
        )
    assert verdict.status == "clean"
    assert not verdict.veto_set


def test_run_validation_panel_no_report_claims_default(tmp_path):
    """Default report_claims=None → predicate not triggered, panel still works."""
    mock_client = MagicMock()
    with patch("backend.agents.rlm.grader_transport.sample_completions") as mock_sample:
        mock_sample.return_value = ["[]"]
        verdict = run_validation_panel(
            validator_client=mock_client,
            panel_models=["test-model"],
            metrics={},
            project_dir=tmp_path,
            leaf_records=[],
            separation="independent",
            # report_claims omitted — default None
        )
    assert verdict.status == "clean"


def test_run_validation_panel_report_claims_grounded_with_none_client(tmp_path):
    """None validator_client → unavailable, report_claims is accepted without error."""
    claims = [Claim(value=0.84, term="accuracy", context="accuracy 0.84")]
    verdict = run_validation_panel(
        validator_client=None,
        panel_models=["test-model"],
        metrics={},
        project_dir=tmp_path,
        leaf_records=[],
        separation="independent",
        report_claims=claims,
    )
    assert verdict.status == "unavailable"
