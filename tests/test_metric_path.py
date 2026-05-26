"""Tests for MetricPath schema and ReproductionContract.metrics_shape (PR-θ).

Change θ.1 — MetricPath on ReproductionContract.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.agents.schemas import MetricPath, ReproductionContract


# ---------------------------------------------------------------------------
# MetricPath construction
# ---------------------------------------------------------------------------

def test_metric_path_valid_construction() -> None:
    """Basic valid MetricPath round-trips cleanly."""
    mp = MetricPath(
        metric_id="mnist_logistic_adam_final_nll",
        json_path="per_model.mnist_logistic.per_dataset.mnist.adam_final_nll",
    )
    assert mp.metric_id == "mnist_logistic_adam_final_nll"
    assert mp.json_path == "per_model.mnist_logistic.per_dataset.mnist.adam_final_nll"
    assert mp.rubric_leaf_ids == []


def test_metric_path_with_rubric_leaf_ids() -> None:
    """MetricPath accepts explicit rubric_leaf_ids."""
    mp = MetricPath(
        metric_id="cifar10_cnn_adam_final_loss",
        json_path="per_model.cifar10_cnn.per_dataset.cifar10.adam_final_loss",
        rubric_leaf_ids=["cifar10_loss_leaf", "accuracy_leaf"],
    )
    assert mp.rubric_leaf_ids == ["cifar10_loss_leaf", "accuracy_leaf"]


def test_metric_path_extra_fields_ignored() -> None:
    """extra='ignore' means unknown fields are silently dropped (not a crash)."""
    mp = MetricPath(
        metric_id="foo",
        json_path="foo.bar",
        unknown_future_field="should be ignored",  # type: ignore[call-arg]
    )
    assert mp.metric_id == "foo"
    assert not hasattr(mp, "unknown_future_field")


def test_metric_path_missing_required_fields() -> None:
    """Missing metric_id or json_path raises ValidationError."""
    with pytest.raises(ValidationError):
        MetricPath(json_path="a.b")  # type: ignore[call-arg]

    with pytest.raises(ValidationError):
        MetricPath(metric_id="x")  # type: ignore[call-arg]


def test_metric_path_model_dump_round_trips() -> None:
    """model_dump() produces a plain dict that re-constructs identically."""
    mp = MetricPath(
        metric_id="mnist_acc",
        json_path="metrics.mnist.accuracy",
        rubric_leaf_ids=["leaf_a"],
    )
    dumped = mp.model_dump()
    assert isinstance(dumped, dict)
    rebuilt = MetricPath(**dumped)
    assert rebuilt.metric_id == mp.metric_id
    assert rebuilt.json_path == mp.json_path
    assert rebuilt.rubric_leaf_ids == mp.rubric_leaf_ids


# ---------------------------------------------------------------------------
# ReproductionContract.metrics_shape
# ---------------------------------------------------------------------------

def test_reproduction_contract_metrics_shape_default_empty() -> None:
    """ReproductionContract.metrics_shape defaults to an empty list (backward compat)."""
    contract = ReproductionContract()
    assert contract.metrics_shape == []


def test_reproduction_contract_with_non_empty_metrics_shape() -> None:
    """ReproductionContract accepts and stores a non-empty metrics_shape."""
    contract = ReproductionContract(
        reproduction_definition="Reproduce the paper baseline.",
        metrics_shape=[
            {
                "metric_id": "mnist_logistic_adam_final_nll",
                "json_path": "per_model.mnist_logistic.per_dataset.mnist.adam_final_nll",
                "rubric_leaf_ids": [],
            },
            {
                "metric_id": "cifar10_cnn_adam_final_loss",
                "json_path": "cifar10_cnn.cifar10.adam_final_loss",
                "rubric_leaf_ids": ["some_leaf"],
            },
        ],
    )
    assert len(contract.metrics_shape) == 2
    assert contract.metrics_shape[0].metric_id == "mnist_logistic_adam_final_nll"
    assert contract.metrics_shape[1].rubric_leaf_ids == ["some_leaf"]


def test_reproduction_contract_metrics_shape_round_trips() -> None:
    """ReproductionContract with metrics_shape round-trips via model_dump."""
    contract = ReproductionContract(
        metrics_shape=[
            MetricPath(
                metric_id="val_loss",
                json_path="results.val_loss",
            )
        ],
    )
    dumped = contract.model_dump()
    assert "metrics_shape" in dumped
    assert len(dumped["metrics_shape"]) == 1
    rebuilt = ReproductionContract(**dumped)
    assert rebuilt.metrics_shape[0].metric_id == "val_loss"
    assert rebuilt.metrics_shape[0].json_path == "results.val_loss"


def test_reproduction_contract_extra_fields_still_ignored() -> None:
    """extra='ignore' on ReproductionContract still applies after adding metrics_shape."""
    contract = ReproductionContract(
        metrics_shape=[],
        totally_unknown_key="dropped",  # type: ignore[call-arg]
    )
    assert contract.metrics_shape == []
