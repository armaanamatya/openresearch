"""Tests for _validate_scope_metrics (PR B scope-shape hard gate)."""

from __future__ import annotations

import pytest

from backend.agents.rlm.primitives import _validate_scope_metrics
from backend.agents.schemas import ScopeSpec


class TestValidateScopeMetrics:
    def test_no_scope_returns_none(self):
        assert _validate_scope_metrics(None, {"acc": 0.5}) is None

    def test_empty_metrics_returns_none(self):
        # When metrics is empty the experiment failed at a different layer;
        # not our problem.
        assert _validate_scope_metrics(ScopeSpec(models=["a", "b"]), {}) is None

    def test_single_model_no_constraint(self):
        scope = ScopeSpec(models=["only_one"])  # is_multi_model is False
        assert _validate_scope_metrics(scope, {"acc": 0.5}) is None

    def test_multi_model_missing_per_model_key(self):
        scope = ScopeSpec(models=["a", "b"])
        hint = _validate_scope_metrics(scope, {"acc": 0.5})
        assert hint is not None
        assert "per_model_required" in hint

    def test_multi_model_per_model_not_dict(self):
        scope = ScopeSpec(models=["a", "b"])
        hint = _validate_scope_metrics(scope, {"per_model": "wrong-type"})
        assert hint is not None
        assert "per_model_required" in hint

    def test_multi_model_empty_per_model(self):
        scope = ScopeSpec(models=["a", "b"])
        hint = _validate_scope_metrics(scope, {"per_model": {}})
        assert hint is not None
        assert "per_model_required" in hint

    def test_multi_model_incomplete(self):
        scope = ScopeSpec(models=["a", "b", "c"])
        hint = _validate_scope_metrics(scope, {"per_model": {"a": {"acc": 0.5}}})
        assert hint is not None
        assert "per_model_incomplete" in hint
        assert "'b'" in hint and "'c'" in hint

    def test_multi_model_complete(self):
        scope = ScopeSpec(models=["a", "b"])
        metrics = {"per_model": {"a": {"acc": 0.5}, "b": {"acc": 0.4}}}
        assert _validate_scope_metrics(scope, metrics) is None

    def test_multi_dataset_single_model_top_level_required(self):
        scope = ScopeSpec(datasets=["X", "Y"])  # single-model (implicit) + multi-dataset
        hint = _validate_scope_metrics(scope, {"acc": 0.5})
        assert hint is not None
        assert "per_dataset_required" in hint

    def test_multi_dataset_single_model_complete(self):
        scope = ScopeSpec(datasets=["X", "Y"])
        metrics = {"per_dataset": {"X": {"acc": 0.5}, "Y": {"acc": 0.4}}}
        assert _validate_scope_metrics(scope, metrics) is None

    def test_multi_dataset_single_model_incomplete(self):
        scope = ScopeSpec(datasets=["X", "Y", "Z"])
        metrics = {"per_dataset": {"X": {"acc": 0.5}}}
        hint = _validate_scope_metrics(scope, metrics)
        assert hint is not None
        assert "per_dataset_incomplete" in hint

    def test_multi_model_multi_dataset_full_nesting(self):
        scope = ScopeSpec(models=["a", "b"], datasets=["X", "Y"])
        metrics = {
            "per_model": {
                "a": {"per_dataset": {"X": {"acc": 0.5}, "Y": {"acc": 0.4}}},
                "b": {"per_dataset": {"X": {"acc": 0.6}, "Y": {"acc": 0.3}}},
            }
        }
        assert _validate_scope_metrics(scope, metrics) is None

    def test_multi_model_multi_dataset_missing_per_dataset(self):
        scope = ScopeSpec(models=["a", "b"], datasets=["X", "Y"])
        metrics = {
            "per_model": {
                "a": {"per_dataset": {"X": {}, "Y": {}}},
                "b": {"acc": 0.5},  # missing per_dataset entirely
            }
        }
        hint = _validate_scope_metrics(scope, metrics)
        assert hint is not None
        # Env-keyed fallback (accept per_model[m][env]): model "b"'s only key ("acc")
        # is not a required dataset, so it is flagged as missing the datasets
        # (per_dataset_incomplete) rather than the stricter per_dataset_required.
        # Either way model 'b' is correctly flagged as deficient.
        assert "per_dataset_incomplete" in hint or "per_dataset_required" in hint
        assert "'b'" in hint

    def test_multi_model_multi_dataset_incomplete_per_dataset(self):
        scope = ScopeSpec(models=["a"], datasets=["X", "Y"])  # NOT multi-model
        metrics = {"per_dataset": {"X": {}}}  # single-model branch takes over
        hint = _validate_scope_metrics(scope, metrics)
        assert hint is not None
        assert "per_dataset_incomplete" in hint

    def test_multi_model_multi_dataset_inner_missing(self):
        scope = ScopeSpec(models=["a", "b"], datasets=["X", "Y", "Z"])
        metrics = {
            "per_model": {
                "a": {"per_dataset": {"X": {}, "Y": {}, "Z": {}}},
                "b": {"per_dataset": {"X": {}, "Y": {}}},
            }
        }
        hint = _validate_scope_metrics(scope, metrics)
        assert hint is not None
        assert "'Z'" in hint
        assert "b" in hint
