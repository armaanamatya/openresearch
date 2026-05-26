"""K-NN estimator over preserved run timings.

Loads `runs/<id>/timing.json` from every preserved run and returns a
`PointEstimate` based on the k nearest neighbors in feature-vector space.

When fewer than `MIN_NEIGHBORS` matching runs exist, returns `None` — the
caller (estimator.py) then skips this source (σ=∞ contribution = 0 weight).

Spec: docs/superpowers/specs/2026-05-25-three-source-budget-estimator-design.md §knn
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

from backend.services.pricing.ensemble import PointEstimate
from backend.services.pricing.paper_features import PaperFeatures

logger = logging.getLogger(__name__)

MIN_NEIGHBORS: int = 3  # Fewer → return None (unavailable)
K_DEFAULT: int = 5      # k for k-NN search


def _euclidean(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _weighted_median(values: list[float], weights: list[float]) -> float:
    """Weighted median of values. Closer neighbors get higher weight."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    # Sort by value
    pairs = sorted(zip(values, weights), key=lambda x: x[0])
    total = sum(w for _, w in pairs)
    cumulative = 0.0
    for v, w in pairs:
        cumulative += w
        if cumulative >= total / 2.0:
            return v
    return pairs[-1][0]


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return float("inf")
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance)


def estimate_from_knn(
    features: PaperFeatures,
    preserved_runs: list[dict],
    *,
    k: int = K_DEFAULT,
) -> PointEstimate | None:
    """Return a wall-clock PointEstimate from k-NN over preserved runs.

    Filtering strategy:
    1. Filter to same category.  If fewer than MIN_NEIGHBORS, expand to all
       categories (the superset fallback).
    2. Among candidates, compute Euclidean distance over feature_vector.
    3. Pick k nearest.  If still fewer than MIN_NEIGHBORS total, return None.
    4. Weighted median for mean (inverse-distance weights); sample std-dev for σ.

    Each `preserved_run` dict must have:
        - `wall_clock_s`: float (seconds)
        - optionally `paper_features.feature_vector`: list[float] (10-dim)
        - `category` (optional — for filtering)
        - `rubric_score` (not used in distance; kept for context)

    Runs with `wall_clock_s == 0` are skipped (uncaptured timing).

    Returns:
        PointEstimate | None.  None means "unavailable" — the combiner treats
        this as σ=∞ (zero weight).
    """
    if not preserved_runs:
        return None

    # Filter to runs with usable wall-clock data.
    usable: list[dict] = [
        r for r in preserved_runs
        if isinstance(r.get("wall_clock_s"), (int, float)) and r["wall_clock_s"] > 0
    ]
    if not usable:
        return None

    # Prefer same-category runs; fall back to all if fewer than MIN_NEIGHBORS.
    same_cat = [
        r for r in usable
        if _run_category(r) == features.category
    ]
    candidates = same_cat if len(same_cat) >= MIN_NEIGHBORS else usable

    if len(candidates) < MIN_NEIGHBORS:
        logger.debug(
            "knn: insufficient neighbors (category=%s, same_cat=%d, total=%d < %d)",
            features.category, len(same_cat), len(usable), MIN_NEIGHBORS,
        )
        return None

    # Compute distances using feature_vector when available.
    target_fv = features.feature_vector
    distances: list[tuple[float, dict]] = []
    for run in candidates:
        run_fv = _run_feature_vector(run)
        if run_fv is not None and len(run_fv) == len(target_fv):
            dist = _euclidean(target_fv, tuple(run_fv))
        else:
            # No feature vector: use a large but finite distance so this run
            # participates but with low weight.
            dist = 5.0
        distances.append((dist, run))

    # Sort by distance and take k nearest.
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]

    wall_clocks_h = [d / 3600.0 for _, r in neighbors for d in [float(r["wall_clock_s"])]]
    dists_only = [d for d, _ in neighbors]

    # Inverse-distance weights (guard against zero distance).
    inv_dist_weights = [1.0 / max(d, 1e-6) for d in dists_only]

    mean_h = _weighted_median(wall_clocks_h, inv_dist_weights)
    sigma_h = _sample_std(wall_clocks_h)

    # If all neighbors have identical values, sigma is 0.  Use a small floor
    # so this source has finite (but high) weight rather than dominating
    # the combination entirely.
    if sigma_h == 0.0:
        sigma_h = 0.05 * max(mean_h, 0.1)

    logger.debug(
        "knn: category=%s, n_candidates=%d, k=%d, mean=%.2fh sigma=%.2fh",
        features.category, len(candidates), len(neighbors), mean_h, sigma_h,
    )
    return PointEstimate(
        mean=mean_h,
        sigma=sigma_h,
        source="knn",
        n_samples=len(neighbors),
        detail={
            "category_filter": features.category,
            "candidates_total": len(candidates),
            "k": len(neighbors),
            "neighbor_wall_clocks_h": [round(h, 3) for h in wall_clocks_h],
        },
    )


def _run_category(run: dict) -> str:
    """Extract category from a timing dict.

    The timing.json written by timing.write_timing_json doesn't yet embed
    paper_features (we keep timing.json lean).  When paper_features is
    present (e.g. from a richer timing.json), use it.  Otherwise default
    to "other" — these runs contribute to the superset fallback.
    """
    pf = run.get("paper_features")
    if isinstance(pf, dict):
        return str(pf.get("category", "other"))
    return "other"


def _run_feature_vector(run: dict) -> list[float] | None:
    pf = run.get("paper_features")
    if isinstance(pf, dict):
        fv = pf.get("feature_vector")
        if isinstance(fv, list):
            return [float(x) for x in fv]
    return None
