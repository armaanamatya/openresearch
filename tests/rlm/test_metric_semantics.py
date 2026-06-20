"""Tests for backend/agents/rlm/metric_semantics.py (C — §4.6)."""

from __future__ import annotations

import backend.agents.rlm.metric_semantics as ms


class TestEnabled:
    def test_off_by_default(self, monkeypatch):
        monkeypatch.delenv("OPENRESEARCH_METRIC_SEMANTICS_GUARD", raising=False)
        assert ms.metric_semantics_guard_enabled() is False

    def test_on(self, monkeypatch):
        monkeypatch.setenv("OPENRESEARCH_METRIC_SEMANTICS_GUARD", "1")
        assert ms.metric_semantics_guard_enabled() is True


class TestViolation:
    def test_accuracy_out_of_range_vetoed(self):
        assert ms.metric_semantics_violation({"accuracy": 1.7}) is not None

    def test_accuracy_in_range_passes(self):
        assert ms.metric_semantics_violation({"accuracy": 0.4}) is None

    def test_zero_reward_passes(self):
        # A legitimately-zero reward is in range — must NOT fire.
        assert ms.metric_semantics_violation({"mean_reward": 0.0}) is None

    def test_one_point_zero_passes(self):
        assert ms.metric_semantics_violation({"f1": 1.0}) is None

    def test_negative_rate_vetoed(self):
        assert ms.metric_semantics_violation({"success_rate": -0.3}) is not None

    def test_nonfinite_loss_vetoed(self):
        assert ms.metric_semantics_violation({"loss": float("nan")}) is not None
        assert ms.metric_semantics_violation({"loss": float("inf")}) is not None

    def test_hyperparameter_not_flagged(self):
        # learning_rate is a config, not a rate — never flagged even though >1 or tiny.
        assert ms.metric_semantics_violation({"learning_rate": 1e-5}) is None

    def test_nested_out_of_range_vetoed(self):
        metrics = {"per_model": {"qwen": {"alfworld": {"sdar": {"accuracy": 1.5}}}}}
        assert ms.metric_semantics_violation(metrics) is not None

    def test_non_dict_returns_none(self):
        assert ms.metric_semantics_violation(None) is None
        assert ms.metric_semantics_violation(42) is None

    def test_bool_not_treated_as_number(self):
        assert ms.metric_semantics_violation({"accuracy": True}) is None
