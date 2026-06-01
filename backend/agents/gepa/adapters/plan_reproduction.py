"""GEPA evaluator for plan_reproduction system prompt optimization.

Callable evaluator compatible with gepa.optimize(evaluator=...):
  (candidate, example, **kwargs) -> tuple[float, dict] | float
"""
from __future__ import annotations

import json
import logging
from typing import Any

from backend.agents.gepa.metrics.plan_metrics import combined_plan_score
from backend.agents.gepa.trainset.paper_examples import PaperExample
from backend.agents.rlm.primitives import _METRICS_SHAPE_INSTRUCTION

logger = logging.getLogger(__name__)

_USER_TEMPLATE = """Method spec:
{method_spec_json}

Environment spec:
{env_spec_json}
"""


class PlanReproductionEvaluator:
    """Callable evaluator for plan_reproduction system prompt optimization.

    Calls llm_client.complete() DIRECTLY (not via wrap_primitive) to avoid
    the re-entrancy issue — the re-entrancy guard in hooks.py prevents
    wrap_primitive from triggering another GEPA loop, but calling directly
    is cleaner and avoids any overhead.
    """

    def __init__(self, *, llm_client: Any) -> None:
        self._llm = llm_client

    def __call__(
        self,
        candidate: dict[str, str] | str,
        example: PaperExample | None = None,
        **kwargs: Any,
    ) -> tuple[float, dict]:
        if example is None:
            return 0.0, {"error": "no example provided"}

        system = (
            candidate["plan_reproduction_system"]
            if isinstance(candidate, dict)
            else str(candidate)
        )
        system_full = system + _METRICS_SHAPE_INSTRUCTION

        user = _USER_TEMPLATE.format(
            method_spec_json=json.dumps(
                example.method_spec if isinstance(example, PaperExample) else example.get("method_spec", {}),
                indent=2,
            )[:3000],
            env_spec_json=json.dumps(
                example.env_spec if isinstance(example, PaperExample) else example.get("env_spec", {}),
                indent=2,
            )[:1000],
        )

        try:
            raw = self._llm.complete(system=system_full, user=user)
        except Exception as exc:
            logger.debug("PlanReproductionEvaluator LLM error: %s", exc)
            return 0.0, {"error": str(exc)}

        try:
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            contract = json.loads(text)
        except json.JSONDecodeError as exc:
            return 0.0, {"error": f"json_parse_error: {exc}", "raw_preview": raw[:200]}

        expected_ids = (
            example.expected_metric_ids
            if isinstance(example, PaperExample)
            else example.get("expected_metric_ids", [])
        )
        score, feedback = combined_plan_score(contract, expected_ids)
        return score, {
            "score": score,
            "feedback": feedback,
            "metrics_shape": contract.get("metrics_shape", []),
            "Output": raw[:500],
        }
