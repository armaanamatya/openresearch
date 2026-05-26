"""Tests for rubric_guard metrics_shape support (PR-θ, Change θ.4).

Verifies that assert_metrics_schema:
  1. When metrics_shape is provided and every declared json_path resolves → no failure.
  2. When metrics_shape is provided but one declared path is missing → fails with
     that path in missing_keys.
  3. When metrics_shape is empty/None → falls back to the existing fingerprint
     matcher (regression: the 8 existing fingerprint tests must still pass).
  4. _path_resolves helper works correctly.
"""

from __future__ import annotations

import json

import pytest

from backend.agents.rlm.rubric_guard import (
    RubricGuardFailure,
    assert_metrics_schema,
    _path_resolves,
)


# ---------------------------------------------------------------------------
# _path_resolves unit tests
# ---------------------------------------------------------------------------

def test_path_resolves_simple_flat() -> None:
    assert _path_resolves({"a": 1}, "a") is True


def test_path_resolves_nested() -> None:
    assert _path_resolves({"a": {"b": {"c": 3}}}, "a.b.c") is True


def test_path_resolves_missing_leaf() -> None:
    assert _path_resolves({"a": {"b": 1}}, "a.b.c") is False


def test_path_resolves_intermediate_not_dict() -> None:
    """Intermediate value is a scalar, not a dict → False."""
    assert _path_resolves({"a": 1}, "a.b") is False


def test_path_resolves_empty_path() -> None:
    assert _path_resolves({"a": 1}, "") is False


def test_path_resolves_non_dict_metrics() -> None:
    assert _path_resolves("not_a_dict", "a") is False
    assert _path_resolves(None, "a") is False  # type: ignore[arg-type]


def test_path_resolves_single_level_key_exists() -> None:
    metrics = {"per_model": {"foo": 1}}
    assert _path_resolves(metrics, "per_model") is True


# ---------------------------------------------------------------------------
# assert_metrics_schema with metrics_shape (authoritative path)
# ---------------------------------------------------------------------------

def test_all_declared_paths_resolve_no_failure() -> None:
    """Every declared json_path resolves → no RubricGuardFailure."""
    metrics = {
        "per_model": {
            "mnist_logistic": {
                "per_dataset": {
                    "mnist": {"adam_final_nll": 0.4137}
                }
            }
        },
        "cifar10_cnn": {"cifar10": {"adam_final_loss": 1.23}},
    }
    metrics_shape = [
        {
            "metric_id": "mnist_logistic_adam_final_nll",
            "json_path": "per_model.mnist_logistic.per_dataset.mnist.adam_final_nll",
            "rubric_leaf_ids": [],
        },
        {
            "metric_id": "cifar10_cnn_adam_final_loss",
            "json_path": "cifar10_cnn.cifar10.adam_final_loss",
            "rubric_leaf_ids": [],
        },
    ]
    # Must not raise.
    assert_metrics_schema(
        metrics,
        required_keys=[],
        metrics_shape=metrics_shape,
    )


def test_one_declared_path_missing_raises() -> None:
    """One declared json_path is absent → RubricGuardFailure with that path."""
    metrics = {
        "per_model": {
            "mnist_logistic": {
                "per_dataset": {
                    "mnist": {"adam_final_nll": 0.4137}
                }
            }
        },
        # cifar10_cnn key is MISSING from metrics
    }
    metrics_shape = [
        {
            "metric_id": "mnist_logistic_adam_final_nll",
            "json_path": "per_model.mnist_logistic.per_dataset.mnist.adam_final_nll",
            "rubric_leaf_ids": [],
        },
        {
            "metric_id": "cifar10_cnn_adam_final_loss",
            "json_path": "cifar10_cnn.cifar10.adam_final_loss",  # missing
            "rubric_leaf_ids": [],
        },
    ]
    with pytest.raises(RubricGuardFailure) as excinfo:
        assert_metrics_schema(
            metrics,
            required_keys=[],
            metrics_shape=metrics_shape,
        )
    detail = json.loads(str(excinfo.value))
    assert detail["rubric_guard"] == "schema_violation"
    # The failing path should be named in missing_keys.
    assert any("cifar10_cnn.cifar10.adam_final_loss" in k for k in detail["missing_keys"])
    # The present path should NOT be in missing_keys.
    assert not any("mnist_logistic" in k and "nll" in k for k in detail["missing_keys"])


def test_metrics_shape_missing_path_entry_skipped() -> None:
    """A MetricPath entry with empty json_path is silently skipped."""
    metrics = {"val_loss": 0.5}
    metrics_shape = [
        {"metric_id": "good_metric", "json_path": "val_loss", "rubric_leaf_ids": []},
        {"metric_id": "broken_entry", "json_path": "", "rubric_leaf_ids": []},  # empty path
    ]
    # The empty-path entry is skipped; val_loss resolves → no failure.
    assert_metrics_schema(metrics, required_keys=[], metrics_shape=metrics_shape)


def test_metrics_shape_dict_or_model_instance_both_accepted() -> None:
    """metrics_shape accepts both plain dicts and MetricPath model instances."""
    from backend.agents.schemas import MetricPath
    metrics = {"flat_acc": 0.9}
    # Mix: one plain dict, one MetricPath instance.
    metrics_shape = [
        {"metric_id": "acc", "json_path": "flat_acc", "rubric_leaf_ids": []},
        MetricPath(metric_id="acc2", json_path="flat_acc"),
    ]
    # Both resolve (same key) → no failure.
    assert_metrics_schema(metrics, required_keys=[], metrics_shape=metrics_shape)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Fallback: empty/None metrics_shape → fingerprint matcher (backward compat)
# ---------------------------------------------------------------------------

def test_empty_metrics_shape_uses_fingerprint_fallback() -> None:
    """metrics_shape=[] → falls back to fingerprint matcher on required_keys."""
    # This is the 2026-05-25 Adam regression case — flat required key matches
    # nested json path via fingerprint.
    metrics = {
        "per_model": {
            "mnist_logistic": {
                "per_dataset": {
                    "mnist": {"adam_final_nll": 0.4137}
                }
            }
        }
    }
    # Flat required key resolves via fingerprint even without metrics_shape.
    assert_metrics_schema(
        metrics,
        required_keys=["mnist_logistic_adam_final_nll"],
        metrics_shape=[],
    )


def test_none_metrics_shape_uses_fingerprint_fallback() -> None:
    """metrics_shape=None → falls back to fingerprint matcher."""
    metrics = {"plain_acc": 0.81}
    assert_metrics_schema(
        metrics,
        required_keys=["plain_acc"],
        metrics_shape=None,
    )


def test_fingerprint_fallback_still_fails_on_missing(tmp_path) -> None:
    """When metrics_shape is None/empty and key truly missing → still fails."""
    metrics = {"other_key": 0.5}
    with pytest.raises(RubricGuardFailure) as excinfo:
        assert_metrics_schema(
            metrics,
            required_keys=["missing_key"],
            metrics_shape=None,
        )
    detail = json.loads(str(excinfo.value))
    assert "missing_key" in detail["missing_keys"]


# ---------------------------------------------------------------------------
# Interaction: metrics_shape + required_artifacts
# ---------------------------------------------------------------------------

def test_metrics_shape_and_artifact_check_both_run(tmp_path) -> None:
    """When metrics_shape is set, artifact check still runs independently."""
    metrics = {"val_loss": 0.5}
    metrics_shape = [
        {"metric_id": "val_loss", "json_path": "val_loss", "rubric_leaf_ids": []},
    ]
    # Artifact is MISSING → should fail on artifact, not on metrics.
    with pytest.raises(RubricGuardFailure) as excinfo:
        assert_metrics_schema(
            metrics,
            required_keys=[],
            metrics_shape=metrics_shape,
            required_artifacts=["README.md"],
            artifact_dir=tmp_path,  # empty dir
        )
    detail = json.loads(str(excinfo.value))
    assert "README.md" in detail["missing_artifacts"]
    assert detail["missing_keys"] == []


# ---------------------------------------------------------------------------
# Regression: all 8 original fingerprint tests still pass with metrics_shape=None
# (No metrics_shape → fingerprint path is unchanged)
# ---------------------------------------------------------------------------

def test_regression_exact_match_unchanged() -> None:
    """Tier-1 exact match unchanged."""
    metrics = {"mnist_baseline_final_acc": 0.81}
    assert_metrics_schema(metrics, required_keys=["mnist_baseline_final_acc"], metrics_shape=None)


def test_regression_fingerprint_nested_matches_flat_underscore() -> None:
    """Required `foo_bar` matches nested {"foo": {"bar": ...}}."""
    metrics = {"foo": {"bar": 1.0}}
    assert_metrics_schema(metrics, required_keys=["foo_bar"], metrics_shape=None)


def test_regression_fingerprint_deeply_nested_with_generic_keys() -> None:
    """The 2026-05-25 Adam case still works with metrics_shape=None."""
    metrics = {
        "per_model": {
            "mnist_logistic": {
                "per_dataset": {
                    "mnist": {"adam_final_nll": 0.4137}
                }
            }
        }
    }
    assert_metrics_schema(
        metrics, required_keys=["mnist_logistic_adam_final_nll"], metrics_shape=None
    )


def test_regression_fingerprint_truly_missing_raises() -> None:
    """Fingerprint fallback still fails when key is truly absent."""
    metrics = {"baz": {"qux": 1.0}}
    with pytest.raises(RubricGuardFailure) as excinfo:
        assert_metrics_schema(metrics, required_keys=["foo_bar"], metrics_shape=None)
    detail = json.loads(str(excinfo.value))
    assert "foo_bar" in detail["missing_keys"]
