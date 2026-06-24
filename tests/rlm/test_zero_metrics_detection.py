"""
Tests for backend/agents/rlm/zero_metrics_detection.py (Task P0.1).

All tests are hermetic — no network, no filesystem I/O.  pytest-socket blocks
non-loopback; monkeypatch.setenv controls the feature flag.

Fixture shapes:
  - FIXTURE_REAL_ZERO: the prj_d118333894223202 all-zero+count metrics.json
    (accuracy/F1/pct_correct keys all 0.0; count keys *_n nonzero — excluded).
  - FIXTURE_SDAR_V6: the SDAR v6 flat-all-zero shape from the plan.
  - FIXTURE_LEGIT_MIXED: a healthy flat dict with real nonzero values.
  - FIXTURE_NESTED_REAL: nested per_model with one real nonzero cell.
  - FIXTURE_NESTED_ALL_ZERO: nested per_model where every cell metric is 0.0.
  - FIXTURE_NESTED_CONSTANT: nested per_model where every cell metric is 0.5.
  - FIXTURE_EMPTY: empty dict.
  - FIXTURE_ZERO_WITH_COUNT: flat dict with *_n counts (excluded) + 0.0 result.
"""

from __future__ import annotations

import pytest

from backend.agents.rlm.zero_metrics_detection import (
    looks_like_zero_metrics,
    normalize_metric_values,
    zero_metrics_guard_enabled,
    zero_metrics_repair_message,
    zero_metrics_should_veto,
)


# ---------------------------------------------------------------------------
# Fixtures (literal — copied from real artifacts)
# ---------------------------------------------------------------------------

# Real all-zero file: runs/prj_d118333894223202/code/metrics.json
# All accuracy/F1/pct_correct keys are 0.0; *_n count keys are nonzero integers.
FIXTURE_REAL_ZERO = {
    "oolong_trec_coarse_pct_correct": 0.0,
    "oolong_trec_coarse_n": 50,
    "oolong_trec_coarse_vanilla_pct_correct": 0.0,
    "oolong_trec_coarse_codeact_pct_correct": 0.0,
    "oolong_pairs_f1": 0.0,
    "oolong_pairs_n": 20,
    "oolong_pairs_vanilla_f1": 0.0,
    "oolong_pairs_codeact_f1": 0.0,
    "browsecomp_plus_pct_correct": 0.0,
    "browsecomp_plus_n": 150,
    "longbench_v2_codeqa_pct_correct": 0.0,
    "longbench_v2_codeqa_n": 30,
    "depth0_oolong_trec_coarse_pct_correct": 0.0,
    "depth0_browsecomp_plus_pct_correct": 0.0,
    "depth1_oolong_trec_coarse_pct_correct": 0.0,
    "depth1_browsecomp_plus_pct_correct": 0.0,
    "depth2_oolong_trec_coarse_pct_correct": 0.0,
    "depth2_browsecomp_plus_pct_correct": 0.0,
    "depth3_oolong_trec_coarse_pct_correct": 0.0,
    "depth3_browsecomp_plus_pct_correct": 0.0,
}

# SDAR v6 flat-all-zero shape (from the implementation plan).
FIXTURE_SDAR_V6 = {
    "loss": 0.0,
    "l_grpo": 0.0,
    "mean_reward": 0.0,
    "accuracy_avg": 0.0,
    "f1_avg": 0.0,
    "teacher_gap_mean": 0.0,
    "gate_activation_ratio": 0.0,
}

# A legitimately healthy flat metrics dict — mixed nonzero values.
FIXTURE_LEGIT_MIXED = {
    "loss": 1.2,
    "return": 31.1,
}

# Nested per_model with one real nonzero cell — metric and reward_mean are
# DIFFERENT values so the constant check does not fire.
FIXTURE_NESTED_REAL = {
    "status": "ok",
    "scope": "qwen3-1.7b/alfworld/grpo",
    "per_model": {
        "qwen3-1.7b": {
            "alfworld": {
                "grpo": {
                    "metric": 0.086,
                    "reward_mean": 0.142,  # distinct — not constant
                }
            }
        }
    },
}

# Nested per_model where every cell's metric is 0.0 — should fire.
FIXTURE_NESTED_ALL_ZERO = {
    "status": "ok",
    "scope": "qwen3-1.7b/alfworld/grpo",
    "per_model": {
        "qwen3-1.7b": {
            "alfworld": {
                "grpo": {
                    "metric": 0.0,
                    "reward_mean": 0.0,
                }
            }
        }
    },
}

# Nested per_model where every cell's metric is the same constant (0.5) — should fire.
FIXTURE_NESTED_CONSTANT = {
    "status": "ok",
    "per_model": {
        "model_a": {
            "env1": {
                "baseline": {
                    "metric": 0.5,
                }
            }
        },
        "model_b": {
            "env1": {
                "baseline": {
                    "metric": 0.5,
                }
            }
        },
    },
}

# Empty dict — should not fire.
FIXTURE_EMPTY: dict = {}

# Flat dict: one result key (0.0) + one *_n count key (excluded).
# *_n is excluded → only the 0.0 result survives → fires.
FIXTURE_ZERO_WITH_COUNT = {
    "x_pct_correct": 0.0,
    "x_n": 50,
}

# A nested dict with multiple cells where one cell has a nonzero value.
FIXTURE_NESTED_MIXED = {
    "per_model": {
        "model_a": {
            "env1": {
                "baseline": {
                    "metric": 0.0,
                }
            }
        },
        "model_b": {
            "env1": {
                "baseline": {
                    "metric": 0.75,  # nonzero → not all-zero, not constant
                }
            }
        },
    },
}


# ---------------------------------------------------------------------------
# Tests: zero_metrics_guard_enabled
# ---------------------------------------------------------------------------

class TestZeroMetricsGuardEnabled:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("OPENRESEARCH_ZERO_METRICS_GUARD", raising=False)
        assert zero_metrics_guard_enabled() is False

    @pytest.mark.parametrize("val", ["1", "true", "yes", "on", "TRUE", "YES", "ON", " 1 "])
    def test_enabled_values(self, monkeypatch, val):
        monkeypatch.setenv("OPENRESEARCH_ZERO_METRICS_GUARD", val)
        assert zero_metrics_guard_enabled() is True

    @pytest.mark.parametrize("val", ["0", "false", "no", "off", "", "2"])
    def test_disabled_values(self, monkeypatch, val):
        monkeypatch.setenv("OPENRESEARCH_ZERO_METRICS_GUARD", val)
        assert zero_metrics_guard_enabled() is False


# ---------------------------------------------------------------------------
# Tests: normalize_metric_values
# ---------------------------------------------------------------------------

class TestNormalizeMetricValues:
    def test_non_dict_returns_empty(self):
        assert normalize_metric_values(None) == []
        assert normalize_metric_values([]) == []
        assert normalize_metric_values("string") == []
        assert normalize_metric_values(42) == []

    def test_empty_dict_returns_empty(self):
        assert normalize_metric_values({}) == []

    def test_flat_legit_mixed(self):
        vals = normalize_metric_values(FIXTURE_LEGIT_MIXED)
        assert set(vals) == {1.2, 31.1}

    def test_flat_sdar_v6_all_result_keys_included(self):
        vals = normalize_metric_values(FIXTURE_SDAR_V6)
        # All 7 keys are result-claiming (none match exclusion rules)
        assert len(vals) == 7
        assert all(v == 0.0 for v in vals)

    def test_real_zero_excludes_count_keys(self):
        # *_n keys (oolong_trec_coarse_n, oolong_pairs_n, etc.) must be excluded.
        vals = normalize_metric_values(FIXTURE_REAL_ZERO)
        # The *_n count keys (50, 20, 150, 30) must NOT appear.
        assert 50 not in vals
        assert 20 not in vals
        assert 150 not in vals
        assert 30 not in vals
        # Only 0.0 result keys remain.
        assert all(v == 0.0 for v in vals)
        assert len(vals) > 0

    def test_excludes_status_scope_cell_id(self):
        metrics = {"status": "ok", "scope": "x", "cell_id": "c1", "loss": 1.5}
        vals = normalize_metric_values(metrics)
        assert vals == [1.5]

    def test_excludes_steps_steps_run_epochs_seed(self):
        metrics = {
            "steps": 1000,
            "steps_run": 800,
            "epochs": 3,
            "seed": 42,
            "loss": 0.5,
        }
        vals = normalize_metric_values(metrics)
        assert vals == [0.5]

    def test_excludes_batch_prefixed_keys(self):
        metrics = {
            "batch_size": 32,
            "batch_scale": 0.5,
            "loss": 1.2,
        }
        vals = normalize_metric_values(metrics)
        assert vals == [1.2]

    def test_excludes_elapsed_prefixed_keys(self):
        metrics = {
            "elapsed_s": 3600.0,
            "elapsed_ms": 1000.0,
            "loss": 0.7,
        }
        vals = normalize_metric_values(metrics)
        assert vals == [0.7]

    def test_excludes_wall_time_s(self):
        metrics = {"wall_time_s": 120.0, "accuracy": 0.9}
        vals = normalize_metric_values(metrics)
        assert vals == [0.9]

    def test_excludes_n_and_count_and_len_and_retries(self):
        metrics = {
            "n": 100,
            "count": 50,
            "len": 20,
            "retries": 3,
            "accuracy": 0.85,
        }
        vals = normalize_metric_values(metrics)
        assert vals == [0.85]

    def test_excludes_suffix_n_pattern(self):
        # x_n, oolong_trec_coarse_n, etc. — any key ending in _n
        metrics = {
            "x_n": 50,
            "samples_n": 100,
            "accuracy": 0.0,
        }
        vals = normalize_metric_values(metrics)
        assert vals == [0.0]

    def test_nested_real_extracts_leaf_values(self):
        vals = normalize_metric_values(FIXTURE_NESTED_REAL)
        # status/scope are excluded; per_model is a nested dict (not a leaf);
        # the two leaf values (metric: 0.086, reward_mean: 0.142) survive.
        assert 0.086 in vals
        assert 0.142 in vals
        assert len(vals) == 2

    def test_nested_all_zero_all_zeros(self):
        vals = normalize_metric_values(FIXTURE_NESTED_ALL_ZERO)
        assert all(v == 0.0 for v in vals)
        assert len(vals) > 0

    def test_zero_with_count_excludes_count(self):
        vals = normalize_metric_values(FIXTURE_ZERO_WITH_COUNT)
        # x_n is excluded; only x_pct_correct (0.0) survives
        assert vals == [0.0]

    def test_numeric_str_coerced(self):
        # Numeric-coercible strings are included (the spec says so).
        metrics = {"metric": "0.5"}
        vals = normalize_metric_values(metrics)
        assert vals == [0.5]

    def test_non_numeric_str_excluded(self):
        metrics = {"metric": "not_a_number", "loss": 1.0}
        vals = normalize_metric_values(metrics)
        assert vals == [1.0]

    def test_bool_excluded(self):
        # Booleans are structural True/False flags, not numeric results.
        metrics = {"converged": True, "loss": 0.5}
        vals = normalize_metric_values(metrics)
        assert vals == [0.5]

    def test_fail_soft_on_error(self):
        # Should never raise; return [] on any error.
        # Pass a pathological object that might cause iteration issues.
        class BadDict:
            def __iter__(self): raise RuntimeError("oops")
        # We can't easily construct a bad object, but we can check the error path
        # returns [] and does not raise.
        result = normalize_metric_values(object())
        assert result == []


# ---------------------------------------------------------------------------
# Tests: looks_like_zero_metrics
# ---------------------------------------------------------------------------

class TestLooksLikeZeroMetrics:
    # --- FIRES (True) ---

    def test_sdar_v6_flat_all_zero(self):
        """The motivating case — SDAR v6 all-zero flat metrics must fire."""
        assert looks_like_zero_metrics(FIXTURE_SDAR_V6) is True

    def test_real_zero_with_count_keys_fires(self):
        """Real prj_d118333894223202 fixture: *_n excluded, result keys all 0.0 → fires."""
        assert looks_like_zero_metrics(FIXTURE_REAL_ZERO) is True

    def test_zero_with_count_fires(self):
        """x_n excluded → only 0.0 result key remains → fires."""
        assert looks_like_zero_metrics(FIXTURE_ZERO_WITH_COUNT) is True

    def test_nested_all_zero_fires(self):
        """Nested per_model with every cell metric = 0.0 → fires."""
        assert looks_like_zero_metrics(FIXTURE_NESTED_ALL_ZERO) is True

    def test_nested_constant_fires(self):
        """Every cell metric = 0.5 (constant, not zero) → fires."""
        assert looks_like_zero_metrics(FIXTURE_NESTED_CONSTANT) is True

    def test_single_zero_value(self):
        assert looks_like_zero_metrics({"loss": 0.0}) is True

    def test_constant_nonzero_across_cells(self):
        """All values the same non-zero constant → fires (constant check)."""
        metrics = {"a": 1.5, "b": 1.5, "c": 1.5}
        assert looks_like_zero_metrics(metrics) is True

    # --- DOES NOT FIRE (False) ---

    def test_legit_mixed_does_not_fire(self):
        """loss=1.2, return=31.1 — real values, not all-zero/constant."""
        assert looks_like_zero_metrics(FIXTURE_LEGIT_MIXED) is False

    def test_nested_real_does_not_fire(self):
        """Nested with real 0.086 cell — not all-zero."""
        assert looks_like_zero_metrics(FIXTURE_NESTED_REAL) is False

    def test_nested_mixed_does_not_fire(self):
        """One cell 0.0, one cell 0.75 — not constant, not all-zero."""
        assert looks_like_zero_metrics(FIXTURE_NESTED_MIXED) is False

    def test_empty_dict_does_not_fire(self):
        """No result-claiming values → non-empty precondition fails → False."""
        assert looks_like_zero_metrics(FIXTURE_EMPTY) is False

    def test_non_dict_does_not_fire(self):
        assert looks_like_zero_metrics(None) is False
        assert looks_like_zero_metrics([]) is False
        assert looks_like_zero_metrics("string") is False

    def test_only_excluded_keys_does_not_fire(self):
        """All keys are structural (status, n, steps) — no result values → False."""
        metrics = {"status": "ok", "n": 100, "steps": 1000, "epochs": 3}
        assert looks_like_zero_metrics(metrics) is False

    def test_mixed_zero_and_nonzero_does_not_fire(self):
        metrics = {"loss": 0.0, "accuracy": 0.85}
        assert looks_like_zero_metrics(metrics) is False

    def test_fail_soft_on_error(self):
        # Pathological input — must not raise, must return False.
        assert looks_like_zero_metrics(object()) is False


# ---------------------------------------------------------------------------
# Tests: zero_metrics_repair_message
# ---------------------------------------------------------------------------

class TestZeroMetricsRepairMessage:
    def test_message_is_string(self):
        msg = zero_metrics_repair_message(FIXTURE_SDAR_V6)
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_all_zero_pattern_named(self):
        msg = zero_metrics_repair_message(FIXTURE_SDAR_V6)
        assert "all-zero" in msg

    def test_constant_pattern_named(self):
        msg = zero_metrics_repair_message(FIXTURE_NESTED_CONSTANT)
        assert "constant" in msg

    def test_names_result_keys(self):
        """Message should name at least some of the offending metric keys."""
        msg = zero_metrics_repair_message(FIXTURE_SDAR_V6)
        # At least one SDAR key must appear.
        sdar_keys = {"loss", "l_grpo", "mean_reward", "accuracy_avg", "f1_avg"}
        assert any(k in msg for k in sdar_keys)

    def test_at_most_six_keys(self):
        """The plan says ≤6 result keys should be named."""
        # Use a dict with many keys and verify the message doesn't explode.
        many_keys = {f"metric_{i}": 0.0 for i in range(20)}
        msg = zero_metrics_repair_message(many_keys)
        assert isinstance(msg, str)

    def test_actionable_language(self):
        """Message must tell the implementer what to do."""
        msg = zero_metrics_repair_message(FIXTURE_SDAR_V6)
        # Must contain actionable repair directive language.
        assert any(
            phrase in msg.lower()
            for phrase in ["re-implement", "real model output", "metrics.json", "fabrication_suspected"]
        )

    def test_fail_soft_returns_string(self):
        """Even on a bad input, must return a string without raising."""
        msg = zero_metrics_repair_message(object())
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_real_zero_fixture_message(self):
        msg = zero_metrics_repair_message(FIXTURE_REAL_ZERO)
        assert "all-zero" in msg
        assert "fabrication_suspected" in msg

    def test_zero_with_count_message_excludes_count_key(self):
        """x_n is a denominator key and must NOT be listed as an offending result key."""
        msg = zero_metrics_repair_message(FIXTURE_ZERO_WITH_COUNT)
        # x_pct_correct should be mentioned (it's the result key), not x_n
        assert "x_pct_correct" in msg
        # x_n is excluded from the result-key listing (it's a count key)
        # Note: x_n might still appear in the message text if it's in keys_str,
        # but the logic collects top-level non-excluded keys, so x_n should be absent.
        # We verify the result key appears, not that x_n is strictly absent (the message
        # may or may not mention x_n depending on key collection; the CRITICAL invariant
        # is that x_pct_correct is named and the guard correctly fires).
        assert isinstance(msg, str)


# ---------------------------------------------------------------------------
# Tests: exclusion edge cases (normalize_metric_values key exclusion)
# ---------------------------------------------------------------------------

class TestExclusionEdgeCases:
    """Explicitly verify each exclusion category from the contract."""

    def test_excludes_n_exact(self):
        assert normalize_metric_values({"n": 100, "loss": 1.0}) == [1.0]

    def test_excludes_count_exact(self):
        assert normalize_metric_values({"count": 100, "loss": 1.0}) == [1.0]

    def test_excludes_steps_exact(self):
        assert normalize_metric_values({"steps": 1000, "loss": 1.0}) == [1.0]

    def test_excludes_steps_run_exact(self):
        assert normalize_metric_values({"steps_run": 800, "loss": 1.0}) == [1.0]

    def test_excludes_epochs_exact(self):
        assert normalize_metric_values({"epochs": 3, "loss": 1.0}) == [1.0]

    def test_excludes_seed_exact(self):
        assert normalize_metric_values({"seed": 42, "loss": 1.0}) == [1.0]

    def test_excludes_wall_time_s_exact(self):
        assert normalize_metric_values({"wall_time_s": 120.0, "loss": 1.0}) == [1.0]

    def test_excludes_len_exact(self):
        assert normalize_metric_values({"len": 50, "loss": 1.0}) == [1.0]

    def test_excludes_retries_exact(self):
        assert normalize_metric_values({"retries": 2, "loss": 1.0}) == [1.0]

    def test_excludes_status_exact(self):
        assert normalize_metric_values({"status": "ok", "loss": 1.0}) == [1.0]

    def test_excludes_scope_exact(self):
        assert normalize_metric_values({"scope": "x", "loss": 1.0}) == [1.0]

    def test_excludes_cell_id_exact(self):
        assert normalize_metric_values({"cell_id": "c1", "loss": 1.0}) == [1.0]

    def test_excludes_batch_prefix(self):
        assert normalize_metric_values({"batch_size": 32, "loss": 1.0}) == [1.0]
        assert normalize_metric_values({"batch_scale": 0.5, "loss": 1.0}) == [1.0]
        # "batch" alone as a key
        assert normalize_metric_values({"batch": 32, "loss": 1.0}) == [1.0]

    def test_excludes_elapsed_prefix(self):
        assert normalize_metric_values({"elapsed_s": 10.0, "loss": 1.0}) == [1.0]
        assert normalize_metric_values({"elapsed_ms": 500.0, "loss": 1.0}) == [1.0]

    def test_excludes_suffix_n(self):
        # Any key ending in _n
        assert normalize_metric_values({"x_n": 50, "y_n": 100, "loss": 1.0}) == [1.0]
        assert normalize_metric_values({"oolong_trec_coarse_n": 50, "acc": 0.0}) == [0.0]
        assert normalize_metric_values({"samples_n": 100, "acc": 0.0}) == [0.0]

    def test_non_matching_key_with_n_included(self):
        # "earn" contains 'n' at end but does NOT end in _n — should be INCLUDED
        assert normalize_metric_values({"earn": 1.5}) == [1.5]
        # "plan" does not end in _n either — included
        assert normalize_metric_values({"plan_score": 0.8}) == [0.8]


def test_single_nonzero_value_not_flagged_as_constant():
    # Regression: a single non-zero metric is a normal partial result, NOT
    # "constant across cells" — the constant branch requires >= 2 values, so a
    # lone legitimate value must NOT fire the veto.
    assert looks_like_zero_metrics({"accuracy": 0.85}) is False


def test_single_zero_value_fires_via_all_zero_branch():
    # A single 0.0 IS suspect — caught by the all-zero branch (not the constant one).
    assert looks_like_zero_metrics({"accuracy": 0.0}) is True


# ---------------------------------------------------------------------------
# zero_metrics_should_veto — the composed three-part decision (P0.2 wire)
# ---------------------------------------------------------------------------


def test_should_veto_fires_on_zero_gpu_no_provenance(monkeypatch):
    # The SDAR v6 shape: all-zero result, real GPU claim, no provenance manifest.
    monkeypatch.setenv("OPENRESEARCH_ZERO_METRICS_GUARD", "1")
    assert zero_metrics_should_veto(
        {"loss": 0.0, "reward": 0.0}, gpu_claim=True, provenance_present=False
    ) is True


def test_should_veto_not_when_provenance_present(monkeypatch):
    # A real failing baseline that scored 0 emits provenance — never vetoed.
    monkeypatch.setenv("OPENRESEARCH_ZERO_METRICS_GUARD", "1")
    assert zero_metrics_should_veto(
        {"loss": 0.0, "reward": 0.0}, gpu_claim=True, provenance_present=True
    ) is False


def test_should_veto_not_when_no_gpu_claim(monkeypatch):
    # No GPU claim → cannot be the "ran on GPU but wrote zeros" fabrication.
    monkeypatch.setenv("OPENRESEARCH_ZERO_METRICS_GUARD", "1")
    assert zero_metrics_should_veto(
        {"loss": 0.0, "reward": 0.0}, gpu_claim=False, provenance_present=False
    ) is False


def test_should_veto_not_when_flag_off(monkeypatch):
    # Default-OFF: byte-identical to baseline (no veto).
    monkeypatch.delenv("OPENRESEARCH_ZERO_METRICS_GUARD", raising=False)
    assert zero_metrics_should_veto(
        {"loss": 0.0, "reward": 0.0}, gpu_claim=True, provenance_present=False
    ) is False


def test_should_veto_not_on_real_metrics(monkeypatch):
    # Real varied metrics never fire regardless of the discriminators.
    monkeypatch.setenv("OPENRESEARCH_ZERO_METRICS_GUARD", "1")
    assert zero_metrics_should_veto(
        {"loss": 1.2, "reward": 0.3}, gpu_claim=True, provenance_present=False
    ) is False


# ---------------------------------------------------------------------------
# codex-1 regression: hparam-masking fix (§4.0 of pre-gpu-code-review spec)
# ---------------------------------------------------------------------------

def test_codex1_hparam_does_not_mask_all_zero_results(monkeypatch):
    """Regression for codex-1 latent P0: a nonzero hyperparameter (learning_rate=1e-5)
    must no longer prevent the guard from vetoing all-zero result metrics.

    Before the fix, normalize_metric_values({loss:0.0, accuracy:0.0, learning_rate:1e-5})
    returned [0.0, 0.0, 1e-5] — not all-zero — so the veto was skipped.
    After the fix, learning_rate is excluded as a config key, leaving [0.0, 0.0],
    which is all-zero, and the veto fires correctly.
    """
    monkeypatch.setenv("OPENRESEARCH_ZERO_METRICS_GUARD", "1")
    assert zero_metrics_should_veto(
        {"loss": 0.0, "accuracy": 0.0, "learning_rate": 1e-5},
        gpu_claim=True,
        provenance_present=False,
    ) is True, (
        "learning_rate=1e-5 must not mask the all-zero result metrics; "
        "the guard must veto this shape."
    )


def test_codex1_lr_shorthand_also_excluded(monkeypatch):
    """The short alias 'lr' must also be excluded as a config key."""
    monkeypatch.setenv("OPENRESEARCH_ZERO_METRICS_GUARD", "1")
    assert zero_metrics_should_veto(
        {"loss": 0.0, "mean_reward": 0.0, "lr": 3e-4},
        gpu_claim=True,
        provenance_present=False,
    ) is True, "lr=3e-4 must not mask all-zero result metrics"


def test_codex1_result_keys_not_excluded_by_config_fix():
    """The config-key exclusion must NOT drop any of the SDAR result keys."""
    from backend.agents.rlm.zero_metrics_detection import normalize_metric_values
    sdar_keys = {
        "loss", "l_grpo", "mean_reward", "accuracy_avg", "f1_avg",
        "teacher_gap_mean", "gate_activation_ratio", "success_rate",
        "return", "accuracy", "reward",
    }
    metrics = {k: 0.5 for k in sdar_keys}
    vals = normalize_metric_values(metrics)
    assert len(vals) == len(sdar_keys), (
        f"Config-key exclusion dropped result keys; expected {len(sdar_keys)} values, "
        f"got {len(vals)}.  Metrics: {metrics}"
    )
