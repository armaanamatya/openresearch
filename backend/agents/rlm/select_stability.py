"""Pure SELECT-stability statistics for A1. At small N (often 2 candidates)
Kendall's tau is degenerate (+/-1); report top-1 flip rate, per-candidate win
probability, and the top-2 margin distribution instead. Stdlib only."""
from __future__ import annotations
from statistics import median
from typing import Any


def _top1(grades: dict[str, float]) -> str:
    return max(grades, key=lambda k: grades[k])


def stability_report(regrades: list[dict[str, float]]) -> dict[str, Any]:
    if not regrades:
        return {"k": 0, "top1_flip_rate": 0.0, "win_prob": {}, "margin_p50": 0.0, "margin_min": 0.0}
    k = len(regrades)
    tops = [_top1(g) for g in regrades]
    cands = sorted({c for g in regrades for c in g})
    win_prob = {c: sum(1 for t in tops if t == c) / k for c in cands}
    # Plurality (modal) winner — the candidate that is top-1 in the most re-grades.
    plurality = max(cands, key=lambda c: win_prob[c])
    # Flip rate: fraction of re-grades where top-1 disagrees with the plurality winner.
    flips = sum(1 for t in tops if t != plurality)
    margins = []
    for g in regrades:
        ordered = sorted(g.values(), reverse=True)
        margins.append(ordered[0] - ordered[1] if len(ordered) > 1 else 0.0)
    return {
        "k": k,
        "top1_flip_rate": flips / k,
        "win_prob": win_prob,
        "margin_p50": float(median(margins)),
        "margin_min": float(min(margins)),
    }
