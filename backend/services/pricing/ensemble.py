"""Inverse-variance ensemble combiner for the three-source estimator.

Combines a list of `PointEstimate` instances into a single weighted mean and
combined sigma, using inverse-variance weighting:

    weight_i   = 1 / max(sigma_i ** 2, EPSILON)
    mu_final   = sum(mu_i * w_i) / sum(w_i)
    sigma_final = 1 / sqrt(sum(w_i))

`EPSILON` prevents a zero-sigma estimate from dominating by infinite weight.

Spec: docs/superpowers/specs/2026-05-25-three-source-budget-estimator-design.md §ensemble
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

_EPSILON: float = 1e-9  # floor for variance to avoid div-by-zero


@dataclass
class PointEstimate:
    """A single estimator's output for one scalar quantity (e.g. wall_clock_hours)."""

    mean: float
    sigma: float        # Standard deviation; inf → unknown / unavailable
    source: str         # Human-readable name for the breakdown UI
    n_samples: int = 0  # 0 for model-based estimators
    detail: dict | None = None  # Extra info for the breakdown panel


def combine(estimates: list[PointEstimate]) -> tuple[float, float, list[dict]]:
    """Inverse-variance weighted combination.

    Args:
        estimates: One or more PointEstimate instances.  Estimates with
            sigma=inf (or NaN) contribute zero weight and are excluded from
            the mean — equivalent to "unavailable" sources.

    Returns:
        (mu_final, sigma_final, breakdown)

        `breakdown` is a list of dicts suitable for the UI:
          [{"source": name, "mean": μ, "sigma": σ, "weight": w_normalised}, ...]
        sources with zero weight are included with weight=0 so the UI can
        show "k-NN: unavailable".

    Edge cases:
        - All sigmas are inf: returns (0.0, inf, breakdown) — caller must
          handle "no estimate" state.
        - Negative mu: clamped to 0.0 (negative time/cost is unphysical).
        - sigma=0: capped at EPSILON so a single degenerate source cannot
          dominate the entire combination by infinite weight.
    """
    if not estimates:
        return 0.0, float("inf"), []

    raw_weights: list[float] = []
    for est in estimates:
        sigma = est.sigma
        if math.isinf(sigma) or math.isnan(sigma) or sigma < 0:
            raw_weights.append(0.0)
        else:
            capped_sigma = max(sigma, _EPSILON**0.5)  # cap sigma, not variance
            raw_weights.append(1.0 / max(capped_sigma**2, _EPSILON))

    total_weight = sum(raw_weights)

    if total_weight == 0.0:
        # All sources unavailable.
        breakdown = [
            {"source": est.source, "mean": est.mean, "sigma": est.sigma, "weight": 0.0}
            for est in estimates
        ]
        return 0.0, float("inf"), breakdown

    mu_final = sum(est.mean * w for est, w in zip(estimates, raw_weights)) / total_weight
    mu_final = max(0.0, mu_final)  # clamp negative (unphysical)

    sigma_final = 1.0 / math.sqrt(total_weight)

    breakdown = [
        {
            "source": est.source,
            "mean": est.mean,
            "sigma": est.sigma,
            "weight": round(w / total_weight, 4),
            "n_samples": est.n_samples,
        }
        for est, w in zip(estimates, raw_weights)
    ]

    return mu_final, sigma_final, breakdown
