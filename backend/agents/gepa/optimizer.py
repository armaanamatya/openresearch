"""run_gepa_mini: thin wrapper around gepa.optimize() for per-call optimization."""
from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


class _GepaAdapterWrapper:
    """Minimal GEPAAdapter wrapping our callable evaluators.

    Our evaluators: (candidate: dict, example: Any) -> (float, dict)
    GEPAAdapter protocol:
      evaluate(batch, candidate, capture_traces) -> EvaluationBatch
      make_reflective_dataset(candidate, eval_batch, components) -> dict
      propose_new_texts = None  (use GEPA's default reflective proposer)
    """

    # Explicitly None so GEPA uses its built-in reflective mutation proposer
    propose_new_texts = None

    def __init__(self, evaluator_fn: Callable) -> None:
        self._fn = evaluator_fn

    def evaluate(
        self,
        batch: list,
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> Any:
        from gepa.core.adapter import EvaluationBatch

        outputs: list = []
        scores: list[float] = []
        trajectories: list | None = [] if capture_traces else None

        for example in batch:
            try:
                result = self._fn(candidate=candidate, example=example)
                score, side_info = (result if isinstance(result, tuple) else (float(result), {}))
            except Exception as exc:
                logger.debug("GepaAdapterWrapper evaluate error: %s", exc)
                score, side_info = 0.0, {"error": str(exc)}

            outputs.append(side_info if isinstance(side_info, dict) else {})
            scores.append(float(score))
            if trajectories is not None:
                trajectories.append({"score": score, "side_info": side_info})

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: Any,
        components_to_update: list[str],
    ) -> dict:
        dataset: dict[str, list] = {}
        for comp in components_to_update:
            records = []
            outputs = getattr(eval_batch, "outputs", []) or []
            scores = getattr(eval_batch, "scores", []) or []
            for i, output in enumerate(outputs):
                score = scores[i] if i < len(scores) else 0.0
                feedback = output.get("feedback", "") if isinstance(output, dict) else ""
                records.append({
                    "Inputs": {"index": i},
                    "Generated Outputs": output if isinstance(output, dict) else {},
                    "Feedback": f"Score: {score:.3f}" + (f" | {feedback}" if feedback else ""),
                })
            dataset[comp] = records
        return dataset


def _make_reflection_lm(model: str) -> str | Callable:
    """Resolve a reflection LM for gepa.optimize(reflection_lm=...).

    Accepted values for REPROLAB_GEPA_REFLECTION_MODEL:
      "claude-oauth"            — Claude Code subscription (no API key needed);
                                  uses the same ClaudeLlmClient path as sub-agents.
      "anthropic/claude-*"      — Anthropic API key (ANTHROPIC_API_KEY must have credits).
      "openai/gpt-*"            — OpenAI API key (OPENAI_API_KEY must have credits).
      any litellm model string  — passed through to litellm.completion().

    Returns a callable (prompt: str) -> str for "claude-oauth", or the
    model string as-is for everything else (gepa.optimize wraps it in litellm).
    """
    if model != "claude-oauth":
        return model

    # Lazy import to avoid circular deps and startup overhead
    from backend.services.context.workspace.tools.rlm_query import ClaudeLlmClient

    _client = ClaudeLlmClient()

    def _claude_oauth_lm(prompt: str) -> str:
        # GEPA passes a single combined prompt; use an empty system so the full
        # reflection instruction lands in the user turn (matching how the RLM
        # root model uses ClaudeLlmClient for sub-calls).
        return _client.complete(system="", user=prompt)

    return _claude_oauth_lm


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

    reflection_model is resolved through _make_reflection_lm, so
    "claude-oauth" works out of the box without any API key.
    """
    try:
        import gepa
        from gepa.utils import TimeoutStopCondition, NoImprovementStopper
    except ImportError as exc:
        logger.warning("gepa not installed, skipping optimization: %s", exc)
        return seed_prompt, 0.0, 0

    try:
        resolved_lm = _make_reflection_lm(reflection_model)
        result = gepa.optimize(
            seed_candidate={component_name: seed_prompt},
            trainset=trainset,
            adapter=_GepaAdapterWrapper(evaluator),
            reflection_lm=resolved_lm,
            max_metric_calls=max_metric_calls,
            stop_callbacks=[
                TimeoutStopCondition(timeout_seconds=timeout_s),
                NoImprovementStopper(max_iterations_without_improvement=3),
            ],
            callbacks=callbacks,
            display_progress_bar=False,
            raise_on_exception=False,
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
