"""Lock the ComputeScope / MetricFloor schema contract.

Tests:
- MetricFloor direction-aware floor/paper_target consistency
- ComputeScope round-trip and field constraints
- ReproductionContract.compute_scope optional (backward compat)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.agents.schemas import ComputeScope, MetricFloor, ReproductionContract


# ---------------------------------------------------------------------------
# MetricFloor
# ---------------------------------------------------------------------------


def test_metric_floor_higher_is_better_floor_must_not_exceed_target():
    """For higher-is-better metrics (accuracy), floor must be <= paper_target."""
    with pytest.raises(ValidationError, match="floor"):
        MetricFloor(
            metric="cifar_test_acc",
            direction="higher",
            paper_target=0.91,
            floor=0.95,  # higher than target — invalid
            rationale="impossible",
        )


def test_metric_floor_lower_is_better_floor_must_not_be_below_target():
    """For lower-is-better metrics (loss), floor must be >= paper_target."""
    with pytest.raises(ValidationError, match="floor"):
        MetricFloor(
            metric="mnist_test_loss",
            direction="lower",
            paper_target=0.5,
            floor=0.3,  # lower than target — invalid (means we expect to beat paper)
            rationale="impossible",
        )


def test_metric_floor_valid_higher_direction():
    m = MetricFloor(
        metric="cifar_test_acc",
        direction="higher",
        paper_target=0.91,
        floor=0.78,
        rationale="5/45 epochs",
    )
    assert m.floor < m.paper_target
    assert m.direction == "higher"


def test_metric_floor_valid_lower_direction():
    m = MetricFloor(
        metric="mnist_test_loss",
        direction="lower",
        paper_target=0.5,
        floor=0.85,
        rationale="1/10 epochs",
    )
    assert m.floor > m.paper_target


def test_metric_floor_invalid_direction_rejected():
    with pytest.raises(ValidationError):
        MetricFloor(
            metric="x",
            direction="medium",  # type: ignore[arg-type]
            paper_target=0.5,
            floor=0.3,
            rationale="x",
        )


def test_metric_floor_floor_equal_to_target_is_valid():
    """Floor exactly equal to paper_target is allowed (degenerate step function)."""
    m = MetricFloor(
        metric="acc",
        direction="higher",
        paper_target=0.9,
        floor=0.9,
        rationale="no budget reduction",
    )
    assert m.floor == m.paper_target


# ---------------------------------------------------------------------------
# ComputeScope
# ---------------------------------------------------------------------------


def test_compute_scope_is_clipped_true_when_actual_lt_paper_epochs():
    s = ComputeScope(
        is_clipped=True,
        paper_epochs=100,
        actual_epochs=20,
        rationale="efficient mode",
        metric_floors=[],
    )
    assert s.is_clipped is True
    assert s.paper_epochs == 100
    assert s.actual_epochs == 20


def test_compute_scope_metric_floors_list():
    s = ComputeScope(
        is_clipped=True,
        paper_epochs=50,
        actual_epochs=5,
        rationale="1/10 budget",
        metric_floors=[
            MetricFloor(metric="acc", direction="higher", paper_target=0.91, floor=0.7, rationale="x"),
            MetricFloor(metric="loss", direction="lower", paper_target=0.5, floor=0.85, rationale="x"),
        ],
    )
    assert len(s.metric_floors) == 2
    assert s.metric_floors[0].metric == "acc"


def test_compute_scope_unclipped_with_empty_floors():
    """is_clipped=False is valid (e.g., max mode); floors can be empty."""
    s = ComputeScope(
        is_clipped=False,
        paper_epochs=100,
        actual_epochs=100,
        rationale="full budget",
        metric_floors=[],
    )
    assert s.is_clipped is False


def test_compute_scope_optional_on_reproduction_contract():
    """Old code paths construct ReproductionContract without compute_scope."""
    c = ReproductionContract.model_construct(compute_scope=None)
    assert c.compute_scope is None


def test_reproduction_contract_compute_scope_defaults_to_none():
    """Default construction has compute_scope=None (backward compat)."""
    c = ReproductionContract()
    assert c.compute_scope is None


def test_reproduction_contract_accepts_compute_scope():
    """compute_scope can be set on a ReproductionContract."""
    scope = ComputeScope(
        is_clipped=True,
        paper_epochs=45,
        actual_epochs=5,
        rationale="efficient",
        metric_floors=[],
    )
    c = ReproductionContract(compute_scope=scope)
    assert c.compute_scope is not None
    assert c.compute_scope.is_clipped is True
