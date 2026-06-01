"""run_gepa_mini: thin wrapper around gepa.optimize() for per-call optimization."""
from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


def run_gepa_mini(
    *,
    seed_prompt: str,
    component_name: str,
    trainset: list,
    evaluator: Callable,
    max_metric_calls: int,
    timeout_s: int,
    reflection_model: str,
    callbacks: list,
) -> tuple[str, float, int]:
    """Run a mini GEPA optimization loop.

    Returns (best_prompt, best_score, total_metric_calls).
    On any failure returns (seed_prompt, 0.0, 0) — never raises.
    """
    try:
        import gepa
        from gepa.utils import TimeoutStopCondition, NoImprovementStopper
    except ImportError as exc:
        logger.warning("gepa not installed, skipping optimization: %s", exc)
        return seed_prompt, 0.0, 0

    try:
        result = gepa.optimize(
            seed_candidate={component_name: seed_prompt},
            trainset=trainset,
            evaluator=evaluator,
            reflection_lm=reflection_model,
            max_metric_calls=max_metric_calls,
            stop_callbacks=[
                TimeoutStopCondition(timeout_seconds=timeout_s),
                NoImprovementStopper(max_iterations_without_improvement=3),
            ],
            callbacks=callbacks,
            display_progress_bar=False,
            seed=0,
        )
        best = result.best_candidate
        best_prompt = best.get(component_name, seed_prompt) if isinstance(best, dict) else str(best)
        scores = result.val_aggregate_scores or []
        best_score = max(scores) if scores else 0.0
        total_calls = result.total_metric_calls or 0
        return best_prompt, best_score, total_calls
    except Exception as exc:
        logger.warning("GEPA mini-optimization failed for %s: %s", component_name, exc)
        return seed_prompt, 0.0, 0
