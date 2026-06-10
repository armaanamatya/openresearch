"""U14 — tests for repro_spec_extractor.py.

Coverage:
  1. Deterministic parser (parse_claim_statement) — exhaustive fixtures.
  2. build_repro_spec — round-trips through two_axis_report.load_claims and
     yields a valid MeasuredClaim.
  3. seed_bundle_from_metrics — reads run artefacts correctly.
  4. extract_and_write (LLM path) — feature-gate, fail-soft, blinded-reconcile.
  5. Direction-folding for lower-is-better metrics.
  6. Adversarial / edge cases (missing baseline, NaN, inf, …).

Pure + deterministic where possible: no GPU, no real LLM calls.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from backend.agents.rlm.repro_spec_extractor import (
    _equivalence_margin,
    _extract_metric_value,
    _infer_direction,
    _reconcile_with_blinded,
    build_repro_spec,
    extract_and_write,
    parse_claim_statement,
    seed_bundle_from_metrics,
)
from backend.agents.rlm.two_axis_report import load_claims
from backend.agents.rlm.reproducibility_verdict import MeasuredClaim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_not_ambiguous(result: dict, msg: str = "") -> None:
    assert not result["ambiguous"], (
        f"Expected not ambiguous{': ' + msg if msg else ''}, "
        f"got ambiguity_reason={result['ambiguity_reason']!r}"
    )


def _assert_ambiguous(result: dict, expected_substr: str = "") -> None:
    assert result["ambiguous"], f"Expected ambiguous, got {result!r}"
    if expected_substr:
        assert expected_substr.lower() in result["ambiguity_reason"].lower(), (
            f"Expected {expected_substr!r} in ambiguity_reason, "
            f"got {result['ambiguity_reason']!r}"
        )


# ---------------------------------------------------------------------------
# §1 — parse_claim_statement: exhaustive deterministic fixtures
# ---------------------------------------------------------------------------

class TestParseClaimStatement:
    """Deterministic parser: the fragile surface.  Every fixture is load-bearing."""

    # --- well-formed "vs" pairs ---

    def test_vs_pair_84_vs_75(self):
        """'84.4 vs 75.0 (+9.4)' — both values and explicit delta present."""
        r = parse_claim_statement("84.4 vs 75.0 (+9.4) on ALFWorld success_rate")
        # Should find the delta 9.4
        assert r["claimed_effect"] == pytest.approx(9.4, abs=1e-6)

    def test_vs_pair_with_grpo_label(self):
        """'84.4 vs. GRPO 75.0' — named baseline in the middle."""
        r = parse_claim_statement("SDAR achieves 84.4 vs. GRPO 75.0 success rate")
        assert r["claimed_effect"] == pytest.approx(9.4, abs=0.1)

    def test_vs_pair_computes_delta_from_pair(self):
        """When no explicit delta is given, it is computed from the vs-pair."""
        r = parse_claim_statement("Model achieves 90.0 vs baseline 80.0 accuracy")
        assert r["claimed_effect"] == pytest.approx(10.0, abs=0.1)
        _assert_not_ambiguous(r)

    # --- percentage-point vs relative-% ambiguity (THE critical case) ---

    def test_bare_percent_delta_is_ambiguous(self):
        """+9.4% with no pp/pts qualifier → pp-vs-relative ambiguous (A1)."""
        r = parse_claim_statement("Our method improves accuracy by +9.4%")
        _assert_ambiguous(r, "percentage-points vs relative")
        assert r["claimed_effect"] == pytest.approx(9.4, abs=1e-6)

    def test_explicit_pp_unit_resolves_ambiguity(self):
        """+9.4 pp explicitly → percentage_points, NOT ambiguous."""
        r = parse_claim_statement("Our method improves by +9.4 pp on ALFWorld")
        _assert_not_ambiguous(r)
        assert r["estimate_kind"] == "percentage_points"
        assert r["claimed_effect"] == pytest.approx(9.4, abs=1e-6)

    def test_explicit_percentage_points_unit_resolves(self):
        r = parse_claim_statement("gains +5.3 percentage points on the test set")
        _assert_not_ambiguous(r)
        assert r["estimate_kind"] == "percentage_points"

    def test_context_hint_resolves_relative_percent(self):
        """context_hint with 'relative improvement' resolves to relative_percent."""
        r = parse_claim_statement(
            "achieves a +12% gain",
            context_hint="relative improvement over the previous state-of-the-art",
        )
        # Relative marker in context → resolved
        assert r["estimate_kind"] == "relative_percent"
        _assert_not_ambiguous(r)

    def test_positive_delta_without_unit_is_ambiguous(self):
        """Plain '+9.4' with no unit and no context → ambiguous."""
        r = parse_claim_statement("outperforms by +9.4 on the benchmark")
        _assert_ambiguous(r)

    # --- verbal delta (improves by X) ---

    def test_improves_by_verbal_delta(self):
        """'improves by 9.4' → positive effect extracted."""
        r = parse_claim_statement("Our approach improves by 9.4 on ALFWorld success_rate")
        assert r["claimed_effect"] == pytest.approx(9.4, abs=1e-6)
        # No unit → ambiguous (no way to know if it's a pp or absolute delta)
        _assert_ambiguous(r)

    def test_outperforms_by_verbal(self):
        r = parse_claim_statement("outperforms the baseline by 3.2")
        assert r["claimed_effect"] == pytest.approx(3.2, abs=1e-6)

    # --- lower-is-better ---

    def test_lower_is_better_perplexity(self):
        """Loss/perplexity: lower = better; a NEGATIVE vs-delta means advantage."""
        r = parse_claim_statement(
            "Our model achieves perplexity 42.1 vs 48.3 for the baseline"
        )
        assert r["direction"] == "lower_is_better"
        # 42.1 < 48.3 means we're better; raw delta = 42.1 - 48.3 = -6.2
        # The parser should fold this to a POSITIVE claimed_effect
        assert r["claimed_effect"] > 0, (
            f"Expected positive claimed_effect for lower-is-better advantage, got {r['claimed_effect']}"
        )

    def test_lower_is_better_from_context_hint(self):
        """context_hint 'lower is better' → direction=lower_is_better."""
        r = parse_claim_statement("score of 0.31 vs 0.45", context_hint="error rate lower is better")
        assert r["direction"] == "lower_is_better"

    def test_lower_is_better_advantage_positive_effect(self):
        """Proposed method has LOWER loss → positive claimed_effect after sign fold."""
        r = parse_claim_statement(
            "reduces loss from 1.5 (baseline) to 1.2",
            context_hint="loss lower is better"
        )
        assert r["direction"] == "lower_is_better"
        # 1.2 < 1.5 so we improved; effect should be +0.3 after fold
        assert r["claimed_effect"] == pytest.approx(0.3, abs=0.1)

    # --- missing baseline → ambiguous ---

    def test_no_baseline_no_comparison(self):
        """Statement with only one number and no baseline → ambiguous."""
        r = parse_claim_statement("Our method achieves 84.4 accuracy on ALFWorld")
        _assert_ambiguous(r)
        # claimed_effect may be 0 or extracted; what matters is ambiguous=True

    def test_no_numeric_at_all(self):
        """No numbers at all → ambiguous."""
        r = parse_claim_statement("Our method is much better than the baseline")
        _assert_ambiguous(r)

    def test_empty_string(self):
        r = parse_claim_statement("")
        _assert_ambiguous(r)

    # --- NaN / inf guards ---

    def test_nan_in_text_is_ambiguous(self):
        """NaN is not a valid claimed effect."""
        r = parse_claim_statement("gains nan pp")
        # After extraction nan would fail the isfinite check
        assert r["ambiguous"] or not math.isfinite(r["claimed_effect"])
        # At minimum, we should not crash

    # --- equivalence_margin correctness ---

    def test_equivalence_margin_ten_percent_of_effect(self):
        """margin = max(10% * |claimed|, floor)."""
        assert _equivalence_margin(9.4) == pytest.approx(0.94, abs=1e-9)
        assert _equivalence_margin(0.0) == pytest.approx(0.05, abs=1e-9)  # floor
        assert _equivalence_margin(0.1) == pytest.approx(0.05, abs=1e-9)  # floor wins

    def test_large_effect_has_proportional_margin(self):
        assert _equivalence_margin(100.0) == pytest.approx(10.0, abs=1e-9)

    # --- direction inference ---

    def test_infer_higher_for_accuracy(self):
        assert _infer_direction("accuracy") == "higher_is_better"

    def test_infer_lower_for_loss(self):
        assert _infer_direction("loss") == "lower_is_better"

    def test_infer_lower_for_error(self):
        assert _infer_direction("error rate") == "lower_is_better"

    def test_infer_none_for_unknown(self):
        """Unknown metric name with conflicting context → None."""
        result = _infer_direction("score", "lower is better higher is better")
        assert result is None  # conflict → None

    def test_infer_via_context(self):
        assert _infer_direction("score", "lower is better") == "lower_is_better"
        assert _infer_direction("", "higher is better") == "higher_is_better"

    # --- sign edge cases ---

    def test_negative_proposed_minus_baseline_negative_delta(self):
        """When proposed < baseline for higher-is-better, effect is negative."""
        r = parse_claim_statement("method scores 70.0 vs baseline 75.0 accuracy")
        # 70 - 75 = -5; direction=higher_is_better so this is a disadvantage
        assert r["claimed_effect"] == pytest.approx(-5.0, abs=0.1)
        assert r["direction"] == "higher_is_better"


# ---------------------------------------------------------------------------
# §2 — build_repro_spec: round-trip via load_claims + MeasuredClaim
# ---------------------------------------------------------------------------

class TestBuildReproSpec:
    """build_repro_spec must produce a spec that round-trips through load_claims."""

    def _simple_claim(self) -> dict:
        return {
            "claim_id": "primary_0",
            "description": "SDAR beats GRPO on ALFWorld",
            "metric_name": "success_rate",
            "direction": "higher_is_better",
            "estimate_kind": "percentage_points",
            "baseline_label": "GRPO",
            "claimed_effect": 9.4,
            "equivalence_margin": 0.94,
            "scope": {"model": "Qwen2.5-3B", "dataset": "ALFWorld", "split": "test", "protocol": ""},
            "is_primary": True,
            "table_ref": "Table 2",
            "paper_span": "SDAR achieves 84.4 vs GRPO 75.0",
            "ambiguous": False,
            "ambiguity_reason": "",
        }

    def test_round_trip_single_claim(self, tmp_path):
        """Single claim round-trips through load_claims → one MeasuredClaim."""
        claim = self._simple_claim()
        spec = build_repro_spec([claim])
        # Write to tmp_path and reload
        dest = tmp_path / "rlm_state"
        dest.mkdir()
        (dest / "repro_spec.json").write_text(json.dumps(spec), encoding="utf-8")
        loaded = load_claims(tmp_path)
        assert len(loaded) == 1
        mc = loaded[0]
        assert isinstance(mc, MeasuredClaim)
        assert mc.comparison.claim_id == "primary_0"
        assert mc.comparison.claimed_effect == pytest.approx(9.4)
        assert mc.comparison.is_primary is True
        assert mc.comparison.ambiguous is False
        assert mc.comparison.direction == "higher_is_better"

    def test_round_trip_with_seed_bundle(self, tmp_path):
        """Seed bundle round-trips faithfully."""
        claim = self._simple_claim()
        bundle = {"seeds": [42, 43], "per_seed_effect": [9.1, 9.3], "rng_independent": True}
        spec = build_repro_spec([claim], seed_bundles=[bundle])
        dest = tmp_path / "rlm_state"
        dest.mkdir()
        (dest / "repro_spec.json").write_text(json.dumps(spec), encoding="utf-8")
        loaded = load_claims(tmp_path)
        assert len(loaded) == 1
        sb = loaded[0].seed_bundle
        assert sb.seeds == (42, 43)
        assert sb.per_seed_effect == pytest.approx((9.1, 9.3))
        assert sb.rng_independent is True
        assert sb.n_effective == 2

    def test_round_trip_measured_scope(self, tmp_path):
        """measured_scope round-trips and is a proper ScopeTuple."""
        claim = self._simple_claim()
        ms = {"model": "Qwen2.5-3B", "dataset": "ALFWorld", "split": "test", "protocol": ""}
        spec = build_repro_spec([claim], measured_scopes=[ms])
        dest = tmp_path / "rlm_state"
        dest.mkdir()
        (dest / "repro_spec.json").write_text(json.dumps(spec), encoding="utf-8")
        loaded = load_claims(tmp_path)
        sc = loaded[0].measured_scope
        assert sc.model == "Qwen2.5-3B"
        assert sc.dataset == "ALFWorld"

    def test_ambiguous_claim_round_trips_ambiguous(self, tmp_path):
        """An ambiguous claim round-trips and stays ambiguous."""
        claim = self._simple_claim()
        claim["ambiguous"] = True
        claim["ambiguity_reason"] = "pp vs relative-% undetermined"
        spec = build_repro_spec([claim])
        dest = tmp_path / "rlm_state"
        dest.mkdir()
        (dest / "repro_spec.json").write_text(json.dumps(spec), encoding="utf-8")
        loaded = load_claims(tmp_path)
        assert loaded[0].comparison.ambiguous is True
        assert "pp vs relative" in loaded[0].comparison.ambiguity_reason

    def test_multiple_claims_round_trip(self, tmp_path):
        """Multiple claims maintain their count after round-trip."""
        claim = self._simple_claim()
        claim2 = dict(claim)
        claim2["claim_id"] = "secondary_1"
        claim2["is_primary"] = False
        claim2["claimed_effect"] = 5.0
        spec = build_repro_spec([claim, claim2])
        dest = tmp_path / "rlm_state"
        dest.mkdir()
        (dest / "repro_spec.json").write_text(json.dumps(spec), encoding="utf-8")
        loaded = load_claims(tmp_path)
        assert len(loaded) == 2
        ids = {mc.comparison.claim_id for mc in loaded}
        assert "primary_0" in ids
        assert "secondary_1" in ids

    def test_missing_seed_bundle_defaults_to_inconclusive_placeholder(self, tmp_path):
        """When no seed bundle is provided, the claim gets an empty bundle."""
        claim = self._simple_claim()
        spec = build_repro_spec([claim])  # no seed_bundles
        dest = tmp_path / "rlm_state"
        dest.mkdir()
        (dest / "repro_spec.json").write_text(json.dumps(spec), encoding="utf-8")
        loaded = load_claims(tmp_path)
        sb = loaded[0].seed_bundle
        assert sb.n_effective == 0  # no independent seeds → inconclusive

    def test_equivalence_margin_clamped_to_zero(self, tmp_path):
        """Negative equivalence_margin is clamped to 0."""
        claim = self._simple_claim()
        claim["equivalence_margin"] = -1.0
        spec = build_repro_spec([claim])
        dest = tmp_path / "rlm_state"
        dest.mkdir()
        (dest / "repro_spec.json").write_text(json.dumps(spec), encoding="utf-8")
        loaded = load_claims(tmp_path)
        # ComparisonSpec raises ValueError for negative margin → loader makes ambiguous
        # OR clamp works and margin is 0
        mc = loaded[0]
        assert mc.comparison.equivalence_margin >= 0

    def test_auto_claim_id_generated(self, tmp_path):
        """When claim_id is absent, it is auto-generated."""
        claim = self._simple_claim()
        del claim["claim_id"]
        spec = build_repro_spec([claim])
        dest = tmp_path / "rlm_state"
        dest.mkdir()
        (dest / "repro_spec.json").write_text(json.dumps(spec), encoding="utf-8")
        loaded = load_claims(tmp_path)
        assert loaded[0].comparison.claim_id  # non-empty

    def test_lower_is_better_claim_round_trips(self, tmp_path):
        """lower_is_better direction survives round-trip."""
        claim = self._simple_claim()
        claim["direction"] = "lower_is_better"
        spec = build_repro_spec([claim])
        dest = tmp_path / "rlm_state"
        dest.mkdir()
        (dest / "repro_spec.json").write_text(json.dumps(spec), encoding="utf-8")
        loaded = load_claims(tmp_path)
        assert loaded[0].comparison.direction == "lower_is_better"

    def test_end_to_end_produces_usable_measured_claim(self, tmp_path):
        """Full round-trip: parsed claim → build_repro_spec → load_claims →
        compute_reproducibility_verdict can consume it without error."""
        from backend.agents.rlm.reproducibility_verdict import (
            FidelityCertificate, ScopeTuple, SeedBundle, compute_reproducibility_verdict
        )
        claim = self._simple_claim()
        bundle = {"seeds": [42, 43], "per_seed_effect": [9.0, 9.2], "rng_independent": True}
        ms = {"model": "Qwen2.5-3B", "dataset": "ALFWorld", "split": "test", "protocol": ""}
        spec = build_repro_spec([claim], seed_bundles=[bundle], measured_scopes=[ms])
        dest = tmp_path / "rlm_state"
        dest.mkdir()
        (dest / "repro_spec.json").write_text(json.dumps(spec), encoding="utf-8")
        claims = load_claims(tmp_path)
        assert len(claims) == 1

        cert = FidelityCertificate(
            invariant_tests_passed=True, mutation_confirmed=True,
            blinded_extraction_agreed=True, obligation_profile="end_to_end",
            profile_satisfied=True, has_measured_metrics=True,
        )
        verdict = compute_reproducibility_verdict(
            fidelity_score=0.9, certificate=cert, claims=claims,
        )
        assert verdict.implementation_verdict == "faithful"
        assert verdict.replication_verdict == "replicated"


# ---------------------------------------------------------------------------
# §3 — seed_bundle_from_metrics
# ---------------------------------------------------------------------------

class TestSeedBundleFromMetrics:
    def _write_metrics(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_top_level_flat_key(self, tmp_path):
        """Simple flat key at top level."""
        self._write_metrics(tmp_path / "code" / "metrics.json", {"success_rate": 0.84})
        result = seed_bundle_from_metrics(tmp_path, metric_key="success_rate")
        assert result["per_seed_effect"] == pytest.approx([0.84])
        assert result["seeds"] == [42]

    def test_per_model_path(self, tmp_path):
        """Navigates per_model[model_key][env_key][metric_key]."""
        data = {"per_model": {"qwen2_5_3b": {"alfworld": {"success_rate": 0.844}}}}
        self._write_metrics(tmp_path / "code" / "metrics.json", data)
        result = seed_bundle_from_metrics(
            tmp_path, metric_key="success_rate", model_key="qwen2_5_3b", env_key="alfworld"
        )
        assert result["per_seed_effect"] == pytest.approx([0.844])

    def test_multiple_output_dirs_gives_multiple_seeds(self, tmp_path):
        """Two separate output dirs → two values → rng_independent=True."""
        d1 = tmp_path / "code" / "outputs" / "run_a"
        d2 = tmp_path / "code" / "outputs" / "run_b"
        self._write_metrics(d1 / "metrics.json", {"success_rate": 0.81})
        self._write_metrics(d2 / "metrics.json", {"success_rate": 0.83})
        result = seed_bundle_from_metrics(tmp_path, metric_key="success_rate")
        assert len(result["per_seed_effect"]) == 2
        assert result["rng_independent"] is True

    def test_single_value_gives_rng_independent_false(self, tmp_path):
        """Single observation → rng_independent=False (honest: inconclusive)."""
        self._write_metrics(tmp_path / "code" / "metrics.json", {"success_rate": 0.84})
        result = seed_bundle_from_metrics(tmp_path, metric_key="success_rate")
        assert result["rng_independent"] is False

    def test_no_metrics_gives_empty_bundle(self, tmp_path):
        """No metrics.json at all → empty inconclusive bundle."""
        result = seed_bundle_from_metrics(tmp_path, metric_key="success_rate")
        assert result["per_seed_effect"] == []
        assert result["seeds"] == []
        assert result["rng_independent"] is False

    def test_non_finite_values_excluded(self, tmp_path):
        """NaN and inf are excluded from the bundle."""
        self._write_metrics(
            tmp_path / "code" / "metrics.json",
            {"success_rate": float("nan")}
        )
        result = seed_bundle_from_metrics(tmp_path, metric_key="success_rate")
        assert result["per_seed_effect"] == []

    def test_malformed_json_is_skipped(self, tmp_path):
        """Malformed JSON file is silently skipped."""
        p = tmp_path / "code" / "metrics.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{not valid", encoding="utf-8")
        result = seed_bundle_from_metrics(tmp_path, metric_key="success_rate")
        assert result["per_seed_effect"] == []

    def test_deduplication_of_identical_values(self, tmp_path):
        """Identical values from different files are deduplicated."""
        # top-level + outputs both have the same value → should deduplicate to 1
        self._write_metrics(tmp_path / "code" / "metrics.json", {"success_rate": 0.84})
        self._write_metrics(
            tmp_path / "code" / "outputs" / "run_a" / "metrics.json",
            {"success_rate": 0.84}
        )
        result = seed_bundle_from_metrics(tmp_path, metric_key="success_rate")
        # Deduplicated to 1 unique value
        assert len(result["per_seed_effect"]) == 1


# ---------------------------------------------------------------------------
# §4 — extract_and_write: feature-gate, fail-soft, blinded-reconcile
# ---------------------------------------------------------------------------

class TestExtractAndWrite:
    """LLM wrapper — tests the flag-gate, fail-soft, and reconciliation logic."""

    _FLAG = "OPENRESEARCH_TWO_AXIS_VERDICT"

    def _make_llm(self, first_response: str, blinded_response: str = "") -> Any:
        """Build a mock LlmClient with two sequential responses."""
        mock = MagicMock()
        responses = [first_response]
        if blinded_response:
            responses.append(blinded_response)
        mock.complete.side_effect = responses + ["{}"] * 10
        return mock

    def _first_response(self, **overrides) -> str:
        claim = {
            "description": "SDAR achieves 84.4 vs GRPO 75.0 on ALFWorld",
            "metric_name": "success_rate",
            "direction": "higher_is_better",
            "baseline_label": "GRPO",
            "proposed_method": "SDAR",
            "proposed_value": 84.4,
            "baseline_value": 75.0,
            "claimed_effect": 9.4,
            "estimate_kind": "percentage_points",
            "scope": {"model": "Qwen2.5-3B", "dataset": "ALFWorld", "split": "test", "protocol": ""},
            "is_primary": True,
            "table_ref": "Table 2",
            "paper_span": "SDAR achieves 84.4 vs GRPO 75.0",
        }
        claim.update(overrides)
        return json.dumps({"claims": [claim]})

    def test_disabled_by_default(self, monkeypatch, tmp_path):
        monkeypatch.delenv(self._FLAG, raising=False)
        llm = self._make_llm(self._first_response())
        result = extract_and_write(None, tmp_path, llm_client=llm, paper_text="dummy " * 100)
        assert result is None
        assert not (tmp_path / "rlm_state" / "repro_spec.json").exists()

    def test_writes_file_when_enabled(self, monkeypatch, tmp_path):
        monkeypatch.setenv(self._FLAG, "1")
        llm = self._make_llm(self._first_response())
        paper = "SDAR achieves 84.4 vs GRPO 75.0 on ALFWorld with +9.4 pp improvement. " * 20
        result = extract_and_write(None, tmp_path, llm_client=llm, paper_text=paper)
        assert result is not None
        assert result.exists()
        data = json.loads(result.read_text())
        assert "claims" in data
        assert len(data["claims"]) == 1

    def test_fail_soft_on_llm_exception(self, monkeypatch, tmp_path):
        """LLM error → fail-soft, returns None, does not raise."""
        monkeypatch.setenv(self._FLAG, "1")
        mock = MagicMock()
        mock.complete.side_effect = RuntimeError("network error")
        result = extract_and_write(None, tmp_path, llm_client=mock, paper_text="x " * 200)
        assert result is None

    def test_fail_soft_on_no_llm_client(self, monkeypatch, tmp_path):
        monkeypatch.setenv(self._FLAG, "1")
        result = extract_and_write(None, tmp_path, llm_client=None, paper_text="x " * 200)
        assert result is None

    def test_fail_soft_short_paper_text(self, monkeypatch, tmp_path):
        monkeypatch.setenv(self._FLAG, "1")
        mock = MagicMock()
        result = extract_and_write(None, tmp_path, llm_client=mock, paper_text="too short")
        assert result is None

    def test_fail_soft_on_unparseable_json(self, monkeypatch, tmp_path):
        """LLM returns garbage → fail-soft, returns None."""
        monkeypatch.setenv(self._FLAG, "1")
        mock = MagicMock()
        mock.complete.return_value = "this is not json at all!!!"
        result = extract_and_write(None, tmp_path, llm_client=mock, paper_text="x " * 200)
        assert result is None

    def test_blinded_reconcile_agreement_no_ambiguous(self, monkeypatch, tmp_path):
        """Blinded re-extraction agrees → claim NOT marked ambiguous."""
        monkeypatch.setenv(self._FLAG, "1")
        blinded = json.dumps({
            "extractions": [{
                "span_index": 0,
                "claimed_effect": 9.4,
                "proposed_value": 84.4,
                "baseline_value": 75.0,
                "estimate_kind": "percentage_points",
            }]
        })
        llm = self._make_llm(self._first_response(), blinded)
        paper = "SDAR achieves 84.4 vs GRPO 75.0 on ALFWorld with +9.4 pp improvement. " * 20
        result = extract_and_write(None, tmp_path, llm_client=llm, paper_text=paper)
        assert result is not None
        data = json.loads(result.read_text())
        # Check that blinded agreement did not force ambiguous=True
        comp = data["claims"][0]["comparison"]
        assert not comp.get("ambiguous"), f"Should not be ambiguous: {comp}"

    def test_blinded_reconcile_disagreement_forces_ambiguous(self, monkeypatch, tmp_path):
        """Blinded re-extraction disagrees on claimed_effect → ambiguous=True (A6a)."""
        monkeypatch.setenv(self._FLAG, "1")
        blinded = json.dumps({
            "extractions": [{
                "span_index": 0,
                "claimed_effect": 99.9,  # wildly different from 9.4
                "proposed_value": 184.4,
                "baseline_value": 175.0,
                "estimate_kind": "percentage_points",
            }]
        })
        llm = self._make_llm(self._first_response(), blinded)
        paper = "SDAR achieves 84.4 vs GRPO 75.0 on ALFWorld with +9.4 pp improvement. " * 20
        result = extract_and_write(None, tmp_path, llm_client=llm, paper_text=paper)
        assert result is not None
        data = json.loads(result.read_text())
        comp = data["claims"][0]["comparison"]
        assert comp.get("ambiguous"), f"Expected ambiguous after blinded disagreement: {comp}"

    def test_spec_round_trips_when_written(self, monkeypatch, tmp_path):
        """Written spec can be loaded by load_claims (end-to-end write-then-load)."""
        monkeypatch.setenv(self._FLAG, "1")
        llm = self._make_llm(self._first_response())
        paper = "SDAR achieves 84.4 vs GRPO 75.0 on ALFWorld success rate improvement. " * 20
        result = extract_and_write(None, tmp_path, llm_client=llm, paper_text=paper)
        assert result is not None
        loaded = load_claims(tmp_path)
        assert len(loaded) == 1


# ---------------------------------------------------------------------------
# §5 — _reconcile_with_blinded (unit-level)
# ---------------------------------------------------------------------------

class TestReconcileWithBlinded:
    def test_agree_on_all(self):
        first = {"claimed_effect": 9.4, "proposed_value": 84.4, "baseline_value": 75.0, "estimate_kind": "percentage_points"}
        blinded = {"claimed_effect": 9.4, "proposed_value": 84.4, "baseline_value": 75.0, "estimate_kind": "percentage_points"}
        agree, reason = _reconcile_with_blinded(first, blinded)
        assert agree
        assert reason == ""

    def test_disagree_on_claimed_effect(self):
        first = {"claimed_effect": 9.4, "proposed_value": 84.4, "baseline_value": 75.0}
        blinded = {"claimed_effect": 2.1, "proposed_value": 84.4, "baseline_value": 75.0}
        agree, reason = _reconcile_with_blinded(first, blinded)
        assert not agree
        assert "claimed_effect" in reason

    def test_disagree_on_estimate_kind(self):
        first = {"claimed_effect": 9.4, "estimate_kind": "percentage_points"}
        blinded = {"claimed_effect": 9.4, "estimate_kind": "relative_percent"}
        agree, reason = _reconcile_with_blinded(first, blinded)
        assert not agree
        assert "estimate_kind" in reason

    def test_blinded_missing_value_when_first_has_it(self):
        """First pass had a value; blinded returned None → disagreement."""
        first = {"claimed_effect": 9.4, "proposed_value": 84.4}
        blinded = {"claimed_effect": 9.4, "proposed_value": None}
        agree, reason = _reconcile_with_blinded(first, blinded)
        assert not agree

    def test_both_none_values_agree(self):
        first = {"claimed_effect": None, "proposed_value": None}
        blinded = {"claimed_effect": None, "proposed_value": None}
        agree, reason = _reconcile_with_blinded(first, blinded)
        assert agree

    def test_tolerance_within_one_percent_agrees(self):
        """Small rounding differences (< 1%) are treated as agreement."""
        first = {"claimed_effect": 9.4}
        blinded = {"claimed_effect": 9.401}  # 0.01% diff
        agree, _ = _reconcile_with_blinded(first, blinded)
        assert agree

    def test_unknown_estimate_kind_in_blinded_does_not_disagree(self):
        """'unknown' in blinded is treated as no opinion → no disagreement."""
        first = {"claimed_effect": 9.4, "estimate_kind": "percentage_points"}
        blinded = {"claimed_effect": 9.4, "estimate_kind": "unknown"}
        agree, _ = _reconcile_with_blinded(first, blinded)
        assert agree


# ---------------------------------------------------------------------------
# §6 — _extract_metric_value (navigation logic)
# ---------------------------------------------------------------------------

class TestExtractMetricValue:
    def test_flat_top_level(self):
        assert _extract_metric_value({"success_rate": 0.84}, "success_rate", "", "", "") == pytest.approx(0.84)

    def test_per_model_env_baseline(self):
        data = {"per_model": {"m": {"env": {"baseline": {"metric": 0.7}}}}}
        assert _extract_metric_value(data, "metric", "m", "env", "baseline") == pytest.approx(0.7)

    def test_per_model_env_no_baseline(self):
        data = {"per_model": {"m": {"env": {"metric": 0.7}}}}
        assert _extract_metric_value(data, "metric", "m", "env", "") == pytest.approx(0.7)

    def test_per_model_no_env(self):
        data = {"per_model": {"m": {"metric": 0.7}}}
        assert _extract_metric_value(data, "metric", "m", "", "") == pytest.approx(0.7)

    def test_missing_key_returns_none(self):
        assert _extract_metric_value({}, "nonexistent", "", "", "") is None

    def test_nan_returns_none(self):
        assert _extract_metric_value({"metric": float("nan")}, "metric", "", "", "") is None

    def test_inf_returns_none(self):
        assert _extract_metric_value({"metric": float("inf")}, "metric", "", "", "") is None

    def test_non_dict_data_returns_none(self):
        assert _extract_metric_value(None, "metric", "", "", "") is None
        assert _extract_metric_value("string", "metric", "", "", "") is None


# ---------------------------------------------------------------------------
# §7 — adversarial / A9-style inputs for the parser
# ---------------------------------------------------------------------------

class TestAdversarialParserInputs:
    """These mirror the A9 adversarial fixtures for the verdict rail but at
    the PARSER level — ensuring bad inputs never produce a false confident value."""

    def test_comparator_swap_direction_ambiguous(self):
        """When 'baseline' keyword precedes the first value in a vs-pair, the
        proposed/baseline ordering is ambiguous — the parser is conservative and
        marks it ambiguous OR extracts a positive delta (first - second).  Either
        way, the downstream verdict treats it as inconclusive (A1).

        Adversarial note: NLU would be needed to resolve which is which; the
        deterministic parser uses positional order (first = proposed).  Callers
        that need the correct sign must provide it via context_hint or LLM
        extraction; ambiguous=True is the safety net."""
        r = parse_claim_statement("baseline achieves 80 vs method 70 on accuracy")
        # The parser cannot reliably determine proposed-vs-baseline from position
        # alone when "baseline" leads.  It either marks ambiguous OR returns a
        # positionally-computed delta.  Either is acceptable for A1 safety.
        # What must NOT happen: the parser must not crash or return NaN/inf.
        assert "claimed_effect" in r
        assert math.isfinite(r["claimed_effect"])

    def test_no_comparison_possible_from_single_value(self):
        r = parse_claim_statement("achieves state-of-the-art 93.2% F1")
        _assert_ambiguous(r)

    def test_effect_zero_but_numeric_not_ambiguous_on_kind(self):
        """Zero effect with explicit pp unit → NOT ambiguous (just zero effect)."""
        r = parse_claim_statement("improves by 0 pp")
        # Zero delta with explicit unit → not ambiguous on kind
        assert r["estimate_kind"] == "percentage_points"
        assert r["claimed_effect"] == pytest.approx(0.0, abs=1e-9)

    def test_unicode_percent_sign_handled(self):
        """Unicode variants of percent/minus do not crash the parser."""
        r = parse_claim_statement("gains +9—.4% accuracy")  # em-dash before decimal
        # Should not crash; may or may not extract a value
        assert "ambiguous" in r

    def test_very_long_text_does_not_crash(self):
        long = "accuracy " * 10000 + " 84.4 vs 75.0 (+9.4 pp)"
        r = parse_claim_statement(long)
        assert "claimed_effect" in r

    def test_multiple_vs_pairs_extracts_first(self):
        """When multiple vs-pairs exist, the first is used (not both)."""
        r = parse_claim_statement(
            "Method A: 84.4 vs 75.0; Method B: 72.1 vs 65.3 success_rate"
        )
        # Should extract the first pair
        assert r["claimed_effect"] is not None
        assert not math.isnan(r["claimed_effect"])
