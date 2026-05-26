"""Tests for ensemble.py — inverse-variance combiner.

Spec: docs/superpowers/specs/2026-05-25-three-source-budget-estimator-design.md §ensemble
"""

from __future__ import annotations

import math

import pytest

from backend.services.pricing.ensemble import PointEstimate, combine


# ---------------------------------------------------------------------------
# Basic combine math
# ---------------------------------------------------------------------------

def test_single_estimate_passthrough():
    est = PointEstimate(mean=4.0, sigma=0.5, source="heuristic")
    mu, sigma, breakdown = combine([est])
    assert mu == pytest.approx(4.0, rel=0.01)
    assert math.isfinite(sigma)
    assert len(breakdown) == 1
    assert breakdown[0]["weight"] == pytest.approx(1.0)


def test_two_equal_weight_estimates():
    """Two estimates with equal sigma → mu = mean of means."""
    e1 = PointEstimate(mean=2.0, sigma=1.0, source="a")
    e2 = PointEstimate(mean=4.0, sigma=1.0, source="b")
    mu, sigma, breakdown = combine([e1, e2])
    assert mu == pytest.approx(3.0, rel=0.01)
    assert math.isfinite(sigma)


def test_high_confidence_source_dominates():
    """Low-sigma source should dominate the combination."""
    confident = PointEstimate(mean=5.0, sigma=0.1, source="confident")
    uncertain  = PointEstimate(mean=1.0, sigma=10.0, source="uncertain")
    mu, _, _ = combine([confident, uncertain])
    # mu should be close to confident source's mean
    assert abs(mu - 5.0) < 0.5, f"expected mu≈5.0, got {mu}"


def test_all_sigma_inf_returns_zero_and_inf():
    """All unavailable → (0.0, inf)."""
    e1 = PointEstimate(mean=3.0, sigma=float("inf"), source="a")
    e2 = PointEstimate(mean=5.0, sigma=float("inf"), source="b")
    mu, sigma, breakdown = combine([e1, e2])
    assert mu == 0.0
    assert sigma == float("inf")
    assert all(b["weight"] == 0.0 for b in breakdown)


def test_sigma_zero_capped():
    """σ=0 should not cause divide-by-zero; result is finite."""
    e1 = PointEstimate(mean=3.0, sigma=0.0, source="exact")
    e2 = PointEstimate(mean=5.0, sigma=1.0, source="noisy")
    mu, sigma, _ = combine([e1, e2])
    assert math.isfinite(mu)
    assert math.isfinite(sigma)
    # mu should be close to the σ=0 source but not exactly equal due to ε cap
    assert 2.5 <= mu <= 5.5


def test_all_agree_sigma_shrinks():
    """When three sources give the same mean, combined σ should be smaller."""
    e1 = PointEstimate(mean=4.0, sigma=1.0, source="a")
    e2 = PointEstimate(mean=4.0, sigma=1.0, source="b")
    e3 = PointEstimate(mean=4.0, sigma=1.0, source="c")
    _, sigma_combined, _ = combine([e1, e2, e3])
    # Individual sigma = 1.0; combined should be smaller (≈ 1/√3 ≈ 0.577)
    assert sigma_combined < 1.0, f"sigma_combined {sigma_combined} not < individual sigma 1.0"


def test_disagree_widens_mu_toward_higher_weight():
    """Disagreement: mu falls between the sources, weighted toward low sigma."""
    low_sigma  = PointEstimate(mean=10.0, sigma=0.5, source="confident")
    high_sigma = PointEstimate(mean=2.0,  sigma=5.0, source="uncertain")
    mu, _, _ = combine([low_sigma, high_sigma])
    # Should be much closer to 10.0 than to 2.0
    assert mu > 7.0, f"expected mu > 7.0, got {mu:.2f}"


def test_negative_mu_clamped():
    """Combined mu < 0 should be clamped to 0."""
    e1 = PointEstimate(mean=-1.0, sigma=0.1, source="negative")
    e2 = PointEstimate(mean=-2.0, sigma=0.2, source="also_negative")
    mu, _, _ = combine([e1, e2])
    assert mu == 0.0


def test_empty_estimates_returns_zero_inf():
    mu, sigma, breakdown = combine([])
    assert mu == 0.0
    assert sigma == float("inf")
    assert breakdown == []


def test_breakdown_weights_sum_to_one():
    """Normalised weights in the breakdown should sum to ~1.0."""
    e1 = PointEstimate(mean=3.0, sigma=0.5, source="a")
    e2 = PointEstimate(mean=5.0, sigma=1.0, source="b")
    e3 = PointEstimate(mean=4.0, sigma=2.0, source="c")
    _, _, breakdown = combine([e1, e2, e3])
    total_weight = sum(b["weight"] for b in breakdown)
    assert total_weight == pytest.approx(1.0, abs=0.01)


def test_inf_sigma_source_excluded_from_weights():
    """A source with σ=∞ should have weight=0 in breakdown."""
    e1 = PointEstimate(mean=5.0, sigma=1.0, source="real")
    e2 = PointEstimate(mean=3.0, sigma=float("inf"), source="unavailable")
    mu, _, breakdown = combine([e1, e2])
    avail_b = next(b for b in breakdown if b["source"] == "unavailable")
    real_b  = next(b for b in breakdown if b["source"] == "real")
    assert avail_b["weight"] == 0.0
    assert real_b["weight"] == pytest.approx(1.0, abs=0.01)
    assert mu == pytest.approx(5.0, rel=0.01)
