"""Tests for estimators/heuristic.py.

Spec: docs/superpowers/specs/2026-05-25-three-source-budget-estimator-design.md §heuristic
"""

from __future__ import annotations

import pytest

from backend.services.pricing.estimators.heuristic import (
    _COLD_START_SIGMA_RATIO,
    estimate_heuristic,
    _hours_per_experiment,
)


def _make_features(category: str, model_size_class: str, num_experiments: int = 1):
    """Construct PaperFeatures with the given category/size (text-derived)."""
    # Use a text snippet that will resolve to `other` so we can fully control
    # category/size by passing keyword arguments instead of relying on the
    # text classifier.  We patch the dataclass directly via extract_features
    # after overriding with known fields.
    from backend.services.pricing.paper_features import (
        PaperFeatures,
        _make_feature_vector,
    )
    fv = _make_feature_vector(24, num_experiments, category, model_size_class, (), ())
    return PaperFeatures(
        sha8="test0000",
        category=category,
        model_size_class=model_size_class,
        estimated_vram_gb=24,
        num_experiments=num_experiments,
        datasets=(),
        gpu_hints=(),
        feature_vector=fv,
    )


# ---------------------------------------------------------------------------
# Per-category estimates land in an expected range
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("category,size,n_exp,expected_min_h,expected_max_h", [
    # Tiny VAE: 0.5h per experiment × 1 = 0.5h
    ("vae", "tiny", 1, 0.4, 0.8),
    # Large CNN: 50h per experiment × 1 = 50h
    ("cnn", "large", 1, 40.0, 70.0),
    # RL small × 2 experiments: 2.0h × 2 = 4.0h
    ("rl", "small", 2, 3.0, 6.0),
    # NLP unknown × 1: 6.0h
    ("nlp", "unknown", 1, 4.0, 9.0),
    # Optimizer tiny: 0.2h × 1 = 0.2h
    ("optimizer", "tiny", 1, 0.1, 0.4),
    # Other medium: 10.0h × 1 = 10.0h
    ("other", "medium", 1, 7.0, 14.0),
])
def test_heuristic_estimate_in_expected_range(
    category, size, n_exp, expected_min_h, expected_max_h
):
    est = estimate_heuristic(_make_features(category, size, n_exp))
    assert est.mean >= expected_min_h, (
        f"mean {est.mean:.2f} < expected_min {expected_min_h} "
        f"for {category}/{size} × {n_exp}"
    )
    assert est.mean <= expected_max_h, (
        f"mean {est.mean:.2f} > expected_max {expected_max_h} "
        f"for {category}/{size} × {n_exp}"
    )


# ---------------------------------------------------------------------------
# Sigma >= 50% of mean (cold-start safety net)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("category,size", [
    ("vae", "tiny"),
    ("cnn", "large"),
    ("rl", "medium"),
    ("nlp", "small"),
    ("other", "unknown"),
])
def test_sigma_at_least_half_of_mean(category, size):
    est = estimate_heuristic(_make_features(category, size))
    assert est.sigma >= _COLD_START_SIGMA_RATIO * est.mean, (
        f"sigma {est.sigma:.2f} < {_COLD_START_SIGMA_RATIO} × mean {est.mean:.2f}"
    )


# ---------------------------------------------------------------------------
# Source name and metadata
# ---------------------------------------------------------------------------

def test_source_name_is_heuristic():
    est = estimate_heuristic(_make_features("vae", "small"))
    assert est.source == "heuristic"
    assert est.n_samples == 0
    assert est.detail is not None
    assert "category" in est.detail
    assert "num_experiments" in est.detail


# ---------------------------------------------------------------------------
# Zero / degenerate num_experiments guard
# ---------------------------------------------------------------------------

def test_zero_num_experiments_returns_positive_estimate():
    features = _make_features("cnn", "small", num_experiments=0)
    est = estimate_heuristic(features)
    assert est.mean > 0.0, "heuristic should return a positive estimate even for 0 experiments"
    assert est.sigma > 0.0


# ---------------------------------------------------------------------------
# Unknown category falls back to "other" row
# ---------------------------------------------------------------------------

def test_unknown_category_uses_other_fallback():
    features = _make_features("completely_unknown_category", "medium")
    est = estimate_heuristic(features)
    other_h = _hours_per_experiment("other", "medium")
    assert est.mean == pytest.approx(other_h)
