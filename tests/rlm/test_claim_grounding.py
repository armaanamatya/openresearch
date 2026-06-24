"""
Tests for backend/agents/rlm/claim_grounding.py (§4.0b).

All tests are hermetic — no network I/O.  Filesystem tests use tmp_path.

Contract invariants verified:
  - success_rate=0.84 NOT grounded by measured loss=0.84 (identity mismatch)
  - success_rate=0.84 IS grounded by measured success_rate≈0.84
  - A config/hyperparameter number (e.g. "beta=5.0") is NOT extracted as a result claim
  - "84%" parses to 0.84 and matches measured 0.84
  - Empty measured → no ungrounded (unverifiable, not ungrounded)
  - A paper-target sentence number is not falsely grounded against unrelated measured values
"""

from __future__ import annotations

import json


from backend.agents.rlm.claim_grounding import (
    Claim,
    check_claims_grounded,
    extract_result_claims,
    flatten_measured_values,
)


# ---------------------------------------------------------------------------
# extract_result_claims
# ---------------------------------------------------------------------------

class TestExtractResultClaims:
    def test_simple_success_rate_claim(self):
        claims = extract_result_claims("Our method achieves success_rate of 0.84 on ALFWorld.")
        assert len(claims) >= 1
        matching = [c for c in claims if abs(c.value - 0.84) < 1e-9]
        assert matching, f"Expected a claim with value≈0.84, got {claims}"
        assert matching[0].term in {"success_rate", "success"}

    def test_config_number_not_extracted(self):
        # "beta=5.0" is a hyperparameter — must NOT be extracted as a result claim.
        claims = extract_result_claims("We set beta=5.0 and lambda=0.1 for the loss.")
        result_terms = {"accuracy", "success", "success_rate", "reward", "f1",
                        "precision", "recall", "score", "em", "exact_match", "return", "win"}
        result_claims = [c for c in claims if c.term in result_terms]
        # None of the config numbers should appear as result claims.
        for claim in result_claims:
            assert claim.value not in {5.0, 0.1}, (
                f"Config number {claim.value} was wrongly extracted as result claim: {claim}"
            )

    def test_percentage_parsed_as_fraction(self):
        # "84%" should yield a claim with value=0.84.
        claims = extract_result_claims("The model achieves accuracy of 84% on the test set.")
        pct_claims = [c for c in claims if abs(c.value - 0.84) < 1e-9]
        assert pct_claims, f"Expected a claim with value=0.84 (from 84%), got {claims}"

    def test_result_term_required(self):
        # A bare number with no result term nearby must NOT be extracted.
        claims = extract_result_claims("The training ran for 1e-5 seconds with 32 batch size.")
        # No result terms in this sentence — expect no claims.
        result_terms = {"accuracy", "success", "success_rate", "reward", "f1",
                        "precision", "recall", "score", "em", "exact_match", "return", "win"}
        assert not any(c.term in result_terms for c in claims), (
            f"Unexpected result claims from non-result sentence: {claims}"
        )

    def test_learning_rate_not_extracted(self):
        # "accuracy of 0.84" near "learning_rate=1e-5" — only the accuracy claim survives.
        # The 1e-5 is scientific notation; its digits (1 and 5) must be skipped.
        text = "With learning_rate=1e-5, accuracy of 0.84 was achieved."
        claims = extract_result_claims(text)
        # accuracy=0.84 must be extracted.
        accuracy_claims = [c for c in claims if c.term == "accuracy"]
        assert any(abs(c.value - 0.84) < 1e-6 for c in accuracy_claims), (
            f"Expected accuracy=0.84 claim, got {claims}"
        )
        # The scientific-notation components (1 and 5 from 1e-5) must NOT appear
        # as standalone claim values for any result term.
        sci_component_values = {1.0, 5.0}
        bad_claims = [c for c in claims if c.value in sci_component_values]
        assert not bad_claims, (
            f"Sci-notation exponent components must not be extracted as result claims: {bad_claims}"
        )

    def test_fail_soft_on_bad_input(self):
        assert extract_result_claims(None) == []  # type: ignore[arg-type]
        assert extract_result_claims(42) == []  # type: ignore[arg-type]

    def test_empty_string(self):
        assert extract_result_claims("") == []

    def test_claim_has_context(self):
        claims = extract_result_claims("The model achieves accuracy 0.75 on WebShop.")
        if claims:
            assert isinstance(claims[0].context, str)
            assert len(claims[0].context) > 0

    def test_no_cross_assignment_between_terms(self):
        # Each number must bind to its NEAREST term, not every term in the window.
        # "success 0.90 and accuracy 0.50" must yield success=0.90 AND accuracy=0.50,
        # never a spurious accuracy=0.90 / success=0.50 (which would false-downgrade).
        claims = extract_result_claims("We report success 0.90 and accuracy 0.50.")
        pairs = {(c.term, round(c.value, 4)) for c in claims}
        assert ("success", 0.90) in pairs
        assert ("accuracy", 0.50) in pairs
        assert ("accuracy", 0.90) not in pairs
        assert ("success", 0.50) not in pairs

    def test_integer_count_not_extracted_as_result(self):
        # "success after 150 steps" — 150 is a step count, not a rate; must NOT bind.
        claims = extract_result_claims("Strong success after 150 steps of training.")
        assert not any(abs(c.value - 150.0) < 1e-9 for c in claims), (
            f"step count 150 wrongly extracted as a result claim: {claims}"
        )

    def test_compound_config_adjacent_excluded(self):
        # "weight_decay 0.01" must not become a reward claim; "reward 5.0" should.
        claims = extract_result_claims("Using weight_decay 0.01, reward 5.0 was observed.")
        vals = {round(c.value, 4) for c in claims}
        assert 0.01 not in vals, f"weight_decay value leaked as a result claim: {claims}"
        assert any(c.term == "reward" and abs(c.value - 5.0) < 1e-9 for c in claims)

    def test_rate_term_rejects_implausible_value(self):
        # A rate term with an implausible value (>100) is not a rate → rejected.
        claims = extract_result_claims("The success metric was 250 over the run.")
        assert not any(c.term in {"success", "success_rate"} and c.value == 250.0
                       for c in claims)


# ---------------------------------------------------------------------------
# flatten_measured_values
# ---------------------------------------------------------------------------

class TestFlattenMeasuredValues:
    def test_reads_main_metrics_json(self, tmp_path):
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        (code_dir / "metrics.json").write_text(
            json.dumps({"success_rate": 0.84, "loss": 0.23, "seed": 42}),
            encoding="utf-8",
        )
        measured = flatten_measured_values(tmp_path)
        values = {t: v for t, v in measured}
        # The FULL metric key is the term (codex Area-2): success_rate, not "rate".
        assert "success_rate" in values
        assert "rate" not in values
        assert "loss" in values
        # seed is a config key — must be excluded
        assert "seed" not in values

    def test_nested_per_model_keeps_full_metric_term(self, tmp_path):
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        (code_dir / "metrics.json").write_text(
            json.dumps({"per_model": {"qwen3_1.7b": {"alfworld": {"sdar": {"success_rate": 0.84}}}}}),
            encoding="utf-8",
        )
        measured = flatten_measured_values(tmp_path)
        assert ("success_rate", 0.84) in measured

    def test_flatten_then_check_integration(self, tmp_path):
        # End-to-end: a measured success_rate must GROUND a claimed success_rate
        # (the codex Area-2 bug made flatten produce "rate", failing identity match).
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        (code_dir / "metrics.json").write_text(
            json.dumps({"success_rate": 0.84, "accuracy_avg": 0.71, "f1_avg": 0.55}),
            encoding="utf-8",
        )
        measured = flatten_measured_values(tmp_path)
        claims = [Claim(0.84, "success_rate", "ctx"), Claim(0.71, "accuracy", "ctx")]
        result = check_claims_grounded(claims, measured)
        assert len(result["grounded"]) == 2, f"expected both grounded, got {result}"
        assert not result["ungrounded"]

    def test_reads_outputs_metrics_json(self, tmp_path):
        code_dir = tmp_path / "code"
        (code_dir / "outputs" / "run1").mkdir(parents=True)
        (code_dir / "outputs" / "run1" / "metrics.json").write_text(
            json.dumps({"accuracy": 0.72}), encoding="utf-8"
        )
        measured = flatten_measured_values(tmp_path)
        terms = {t for t, _ in measured}
        assert "accuracy" in terms

    def test_missing_project_dir_returns_empty(self, tmp_path):
        result = flatten_measured_values(tmp_path / "nonexistent")
        assert result == []

    def test_malformed_json_skipped(self, tmp_path):
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        (code_dir / "metrics.json").write_text("not json", encoding="utf-8")
        result = flatten_measured_values(tmp_path)
        assert result == []

    def test_excludes_config_keys_from_measured(self, tmp_path):
        code_dir = tmp_path / "code"
        code_dir.mkdir()
        (code_dir / "metrics.json").write_text(
            json.dumps({"learning_rate": 1e-5, "accuracy": 0.85}),
            encoding="utf-8",
        )
        measured = flatten_measured_values(tmp_path)
        # learning_rate must not appear; accuracy must appear
        values_by_term = {t: v for t, v in measured}
        assert "learning_rate" not in values_by_term
        assert "accuracy" in values_by_term or "accuracy" in {t for t, _ in measured}


# ---------------------------------------------------------------------------
# check_claims_grounded
# ---------------------------------------------------------------------------

class TestCheckClaimsGrounded:
    def _make_claim(self, value: float, term: str) -> Claim:
        return Claim(value=value, term=term, context=f"{term}={value}")

    # --- Identity mismatch ---

    def test_success_rate_not_grounded_by_loss(self):
        """A measured loss=0.84 must NOT ground a claimed success_rate=0.84."""
        claims = [self._make_claim(0.84, "success_rate")]
        measured = [("loss", 0.84)]
        result = check_claims_grounded(claims, measured)
        assert len(result["ungrounded"]) == 1, (
            "success_rate=0.84 must not be grounded by loss=0.84 (identity mismatch)"
        )
        assert len(result["grounded"]) == 0

    # --- Match by synonym ---

    def test_success_rate_grounded_by_measured_success_rate(self):
        """A measured success_rate≈0.84 DOES ground claimed success_rate=0.84."""
        claims = [self._make_claim(0.84, "success_rate")]
        measured = [("success_rate", 0.84)]
        result = check_claims_grounded(claims, measured)
        assert len(result["grounded"]) == 1
        assert len(result["ungrounded"]) == 0

    def test_success_rate_grounded_by_success_synonym(self):
        """success_rate and success map to the same canonical metric."""
        claims = [self._make_claim(0.84, "success_rate")]
        measured = [("success", 0.84)]
        result = check_claims_grounded(claims, measured)
        assert len(result["grounded"]) == 1

    def test_accuracy_avg_grounded_by_accuracy(self):
        claims = [self._make_claim(0.75, "accuracy")]
        measured = [("accuracy_avg", 0.75)]
        result = check_claims_grounded(claims, measured)
        assert len(result["grounded"]) == 1

    def test_em_grounded_by_exact_match(self):
        claims = [self._make_claim(0.62, "em")]
        measured = [("exact_match", 0.62)]
        result = check_claims_grounded(claims, measured)
        assert len(result["grounded"]) == 1

    def test_reward_grounded_by_mean_reward(self):
        claims = [self._make_claim(31.1, "reward")]
        measured = [("mean_reward", 31.1)]
        result = check_claims_grounded(claims, measured)
        assert len(result["grounded"]) == 1

    # --- Percentage normalization ---

    def test_percent_claim_grounded_by_fraction_measured(self):
        """A claim extracted as 84% (→ 0.84) should match measured 0.84."""
        claims = [self._make_claim(0.84, "accuracy")]
        measured = [("accuracy", 0.84)]
        result = check_claims_grounded(claims, measured)
        assert len(result["grounded"]) == 1

    def test_raw_percent_value_grounded_via_div100(self):
        """A claim with value=84.0 (not yet normalized) is tried as 84/100=0.84 against measured."""
        claims = [self._make_claim(84.0, "accuracy")]
        measured = [("accuracy", 0.84)]
        result = check_claims_grounded(claims, measured)
        # Should be grounded via the /100 branch.
        assert len(result["grounded"]) == 1

    # --- Empty measured ---

    def test_empty_measured_returns_no_ungrounded(self):
        """Empty measured evidence → unverifiable, NOT ungrounded."""
        claims = [self._make_claim(0.84, "success_rate")]
        result = check_claims_grounded(claims, [])
        assert result == {"grounded": [], "ungrounded": []}

    # --- Relative tolerance ---

    def test_within_tolerance(self):
        claims = [self._make_claim(0.840, "accuracy")]
        # measured value 0.837 — within 5% relative tolerance
        measured = [("accuracy", 0.837)]
        result = check_claims_grounded(claims, measured)
        assert len(result["grounded"]) == 1

    def test_outside_tolerance(self):
        claims = [self._make_claim(0.84, "accuracy")]
        # measured 0.70 — well outside 5%
        measured = [("accuracy", 0.70)]
        result = check_claims_grounded(claims, measured)
        assert len(result["ungrounded"]) == 1

    # --- Mixed grounded / ungrounded ---

    def test_mixed_claims(self):
        claims = [
            self._make_claim(0.84, "success_rate"),  # grounded
            self._make_claim(0.99, "f1"),             # ungrounded (no matching f1)
        ]
        measured = [("success_rate", 0.84), ("loss", 0.23)]
        result = check_claims_grounded(claims, measured)
        assert len(result["grounded"]) == 1
        assert len(result["ungrounded"]) == 1
        assert result["grounded"][0].term == "success_rate"
        assert result["ungrounded"][0].term == "f1"

    # --- Paper target numbers not falsely grounded ---

    def test_paper_target_number_not_falsely_grounded(self):
        """A paper target like "0.95 from GPT-4" must not be grounded against
        unrelated measured success_rate=0.95 when the claim term is different."""
        # If a claim is extracted with term="score" but measured only has "accuracy",
        # the identity check prevents a false match.
        claims = [self._make_claim(0.95, "score")]
        measured = [("accuracy", 0.95)]  # different canonical term
        result = check_claims_grounded(claims, measured)
        # "score" canonical = "score"; "accuracy" canonical = "accuracy" → no match
        assert len(result["ungrounded"]) == 1

    # --- Fail-soft ---

    def test_fail_soft_on_bad_claims(self):
        result = check_claims_grounded(None, [("loss", 0.5)])  # type: ignore[arg-type]
        assert "grounded" in result
        assert "ungrounded" in result

    def test_empty_claims_returns_empty(self):
        result = check_claims_grounded([], [("loss", 0.5)])
        assert result == {"grounded": [], "ungrounded": []}
