"""Floor-anchored grading for result_match leaves under ComputeScope (β3 Task 3).

Tests:
- score_with_floor: all 9 direction-aware cases from the plan spec + edge cases
- _apply_compute_adjusted_scoring: unclipped always-emit + clipped re-scoring
"""

from __future__ import annotations

import pytest

from backend.agents.rlm.primitives import score_with_floor, _apply_compute_adjusted_scoring


# ---------------------------------------------------------------------------
# score_with_floor — higher-is-better cases
# ---------------------------------------------------------------------------


def test_higher_at_floor_gives_full_credit():
    s = score_with_floor(actual=0.78, paper_target=0.91, floor=0.78, direction="higher")
    assert s == pytest.approx(1.0)


def test_higher_above_paper_gives_full_credit_no_upside_cap():
    s = score_with_floor(actual=0.95, paper_target=0.91, floor=0.78, direction="higher")
    assert s == pytest.approx(1.0)


def test_higher_between_floor_and_target_linear_interp():
    # half-way between 0.78 and 0.91 → 0.5
    s = score_with_floor(actual=0.845, paper_target=0.91, floor=0.78, direction="higher")
    assert 0.49 < s < 0.51


def test_higher_below_floor_gives_zero():
    s = score_with_floor(actual=0.70, paper_target=0.91, floor=0.78, direction="higher")
    assert s == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# score_with_floor — lower-is-better cases
# ---------------------------------------------------------------------------


def test_lower_at_floor_gives_full_credit():
    s = score_with_floor(actual=0.85, paper_target=0.5, floor=0.85, direction="lower")
    assert s == pytest.approx(1.0)


def test_lower_below_paper_gives_full_credit():
    """A loss BETTER than paper target is still full credit (no upside cap)."""
    s = score_with_floor(actual=0.4, paper_target=0.5, floor=0.85, direction="lower")
    assert s == pytest.approx(1.0)


def test_lower_between_floor_and_target_linear_interp():
    # half-way between 0.85 and 0.5 is 0.675
    s = score_with_floor(actual=0.675, paper_target=0.5, floor=0.85, direction="lower")
    assert 0.49 < s < 0.51


def test_lower_above_floor_gives_zero():
    """Loss worse than the floor means the algorithm isn't working."""
    s = score_with_floor(actual=0.9, paper_target=0.5, floor=0.85, direction="lower")
    assert s == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# score_with_floor — edge cases
# ---------------------------------------------------------------------------


def test_floor_equals_paper_target_higher_returns_step_function():
    """When floor == paper_target, scoring degenerates to pass/fail."""
    s_pass = score_with_floor(actual=0.91, paper_target=0.91, floor=0.91, direction="higher")
    s_fail = score_with_floor(actual=0.90, paper_target=0.91, floor=0.91, direction="higher")
    assert s_pass == pytest.approx(1.0)
    assert s_fail == pytest.approx(0.0)


def test_floor_equals_paper_target_lower_returns_step_function():
    """Same step-function for lower-is-better."""
    s_pass = score_with_floor(actual=0.5, paper_target=0.5, floor=0.5, direction="lower")
    s_fail = score_with_floor(actual=0.51, paper_target=0.5, floor=0.5, direction="lower")
    assert s_pass == pytest.approx(1.0)
    assert s_fail == pytest.approx(0.0)


def test_invalid_direction_raises():
    with pytest.raises(ValueError, match="direction"):
        score_with_floor(actual=0.5, paper_target=0.9, floor=0.7, direction="sideways")


# ---------------------------------------------------------------------------
# _apply_compute_adjusted_scoring — unclipped always-emit
# ---------------------------------------------------------------------------


def test_apply_compute_adjusted_unclipped_run_copies_raw():
    """When compute_scope is None, adjusted == raw on every area."""
    raw = {
        "overall_score": 0.85,
        "areas": [
            {"area": "Method fidelity", "score": 0.9, "weight": 0.4, "leaves": []},
            {"area": "Result match vs paper", "score": 0.8, "weight": 0.6, "leaves": []},
        ],
    }
    adjusted = _apply_compute_adjusted_scoring(raw, compute_scope=None, actual_metrics={})
    assert adjusted["compute_adjusted_score"] == pytest.approx(0.85)
    assert adjusted["compute_scope"] is None
    assert all("compute_adjusted_score" in a for a in adjusted["areas"])
    assert adjusted["areas"][0]["compute_adjusted_score"] == pytest.approx(0.9)
    assert adjusted["areas"][1]["compute_adjusted_score"] == pytest.approx(0.8)


def test_apply_compute_adjusted_unclipped_explicit_scope_copies_raw():
    """is_clipped=False with no metric_floors → adjusted == raw."""
    from backend.agents.schemas import ComputeScope
    scope = ComputeScope(is_clipped=False, paper_epochs=100, actual_epochs=100,
                         rationale="full budget", metric_floors=[])
    raw = {"overall_score": 0.70, "areas": [{"area": "x", "score": 0.70, "weight": 1.0, "leaves": []}]}
    out = _apply_compute_adjusted_scoring(raw, scope, {})
    assert out["compute_adjusted_score"] == pytest.approx(0.70)


# ---------------------------------------------------------------------------
# _apply_compute_adjusted_scoring — clipped re-scoring
# ---------------------------------------------------------------------------


def test_apply_compute_adjusted_clipped_run_rescores_result_match():
    """When clipping is active, result_match leaves get re-scored against floors."""
    from backend.agents.schemas import ComputeScope, MetricFloor

    scope = ComputeScope(
        is_clipped=True,
        paper_epochs=45,
        actual_epochs=5,
        rationale="x",
        metric_floors=[
            MetricFloor(metric="mnist_acc", direction="higher",
                        paper_target=0.99, floor=0.85, rationale="x"),
        ],
    )
    raw = {
        "overall_score": 0.4,
        "areas": [
            {
                "area": "Method fidelity",
                "score": 1.0, "weight": 0.4,
                "leaves": [{"id": "L1", "score": 1.0}],
            },
            {
                "area": "Result match vs paper",
                "score": 0.0, "weight": 0.6,
                "leaves": [{"id": "L2", "metric": "mnist_acc", "score": 0.0}],
            },
        ],
    }
    actual = {"mnist_acc": 0.88}  # above floor (0.85), below paper (0.99) → partial credit
    out = _apply_compute_adjusted_scoring(raw, scope, actual)

    # Method fidelity stays 1.0 × 0.4 = 0.4 contribution
    # Result match: 0.88 between 0.85 and 0.99 → (0.88-0.85)/(0.99-0.85) ≈ 0.214
    # 0.214 × 0.6 ≈ 0.129 contribution → total ≈ 0.529
    assert 0.50 < out["compute_adjusted_score"] < 0.56
    assert out["areas"][1]["leaves"][0]["compute_adjusted_score"] > 0.0
    assert out["areas"][0]["compute_adjusted_score"] == pytest.approx(1.0)
    assert out["compute_scope"] is not None
    assert out["compute_scope"]["is_clipped"] is True


def test_apply_compute_adjusted_leaf_without_floor_uses_raw():
    """A leaf that has no floor entry falls back to its raw score."""
    from backend.agents.schemas import ComputeScope, MetricFloor

    scope = ComputeScope(
        is_clipped=True, paper_epochs=10, actual_epochs=1, rationale="x",
        metric_floors=[
            MetricFloor(metric="acc", direction="higher", paper_target=0.9, floor=0.7, rationale="x"),
        ],
    )
    raw = {
        "overall_score": 0.5,
        "areas": [{
            "area": "Result match vs paper",
            "score": 0.5, "weight": 1.0,
            "leaves": [
                {"id": "L1", "metric": "acc", "score": 0.0},
                {"id": "L2", "metric": "unknown_metric", "score": 0.4},  # no floor
            ],
        }],
    }
    out = _apply_compute_adjusted_scoring(raw, scope, {"acc": 0.75})
    # L1: 0.75 between floor 0.7 and target 0.9 → (0.75-0.7)/(0.9-0.7) = 0.25
    # L2: no floor → falls back to raw 0.4
    # area avg = (0.25 + 0.4) / 2 = 0.325
    area = out["areas"][0]
    assert area["leaves"][0]["compute_adjusted_score"] == pytest.approx(0.25)
    assert area["leaves"][1]["compute_adjusted_score"] == pytest.approx(0.4)
