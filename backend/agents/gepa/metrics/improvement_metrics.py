"""Scoring for propose_improvements outputs."""
from __future__ import annotations

import math
from collections import Counter

_KNOWN_CATEGORIES = frozenset({
    "hyperparameter", "architecture", "data", "regularization",
    "training", "evaluation", "other",
})


def weak_area_coverage(
    hypotheses: list[dict],
    weak_leaves: list[str],
) -> tuple[float, str]:
    """Fraction of weak rubric leaves covered by at least one hypothesis.

    Matching is by category field (case-insensitive substring match).
    """
    if not weak_leaves:
        return 1.0, "no weak_leaves constraints"
    if not hypotheses:
        return 0.0, f"no hypotheses; uncovered: {weak_leaves}"
    categories_used = {h.get("category", "").lower() for h in hypotheses}
    covered = [
        leaf for leaf in weak_leaves
        if any(leaf.lower() in cat or cat in leaf.lower() for cat in categories_used)
    ]
    uncovered = [leaf for leaf in weak_leaves if leaf not in covered]
    score = len(covered) / len(weak_leaves)
    return score, f"covered={covered} uncovered={uncovered}"


def category_diversity(hypotheses: list[dict]) -> float:
    """Entropy-based diversity of hypothesis categories, normalized to [0,1].

    Max entropy for N hypotheses = log2(min(N, |KNOWN_CATEGORIES|)).
    """
    if not hypotheses:
        return 0.0
    counts = Counter(h.get("category", "other").lower() for h in hypotheses)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)
    max_entropy = math.log2(min(len(hypotheses), len(_KNOWN_CATEGORIES)))
    if max_entropy == 0:
        return 1.0
    return min(1.0, entropy / max_entropy)


def combined_improvement_score(
    hypotheses: list[dict],
    weak_leaves: list[str],
    *,
    coverage_weight: float = 0.7,
    diversity_weight: float = 0.3,
) -> tuple[float, str]:
    coverage, fb = weak_area_coverage(hypotheses, weak_leaves)
    diversity = category_diversity(hypotheses)
    score = coverage_weight * coverage + diversity_weight * diversity
    return score, f"coverage={coverage:.2f} diversity={diversity:.2f} | {fb}"
