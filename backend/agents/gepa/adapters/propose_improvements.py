"""GEPA evaluator for propose_improvements system prompt optimization."""
from __future__ import annotations

import json
import logging
from typing import Any

from backend.agents.gepa.metrics.improvement_metrics import combined_improvement_score

logger = logging.getLogger(__name__)

_USER_TEMPLATE = """Current results: {results_json}
Rubric scores: {rubric_json}
Generate {k} improvement hypotheses."""


class ProposeImprovementsEvaluator:
    def __init__(self, *, llm_client: Any) -> None:
        self._llm = llm_client

    def __call__(
        self,
        candidate: dict[str, str] | str,
        example: dict | None = None,
        **kwargs: Any,
    ) -> tuple[float, dict]:
        example = example or {}
        system = (
            candidate["propose_improvements_system"]
            if isinstance(candidate, dict)
            else str(candidate)
        )
        user = _USER_TEMPLATE.format(
            results_json=json.dumps(example.get("current_results", {}))[:500],
            rubric_json=json.dumps(example.get("rubric_scores", {}))[:500],
            k=3,
        )
        try:
            raw = self._llm.complete(system=system, user=user)
        except Exception as exc:
            return 0.0, {"error": str(exc)}

        try:
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return 0.0, {"error": f"json_error: {exc}", "raw": raw[:200]}

        hypotheses = parsed.get("hypotheses", [])
        if not isinstance(hypotheses, list):
            return 0.0, {"error": "hypotheses not a list"}

        weak_leaves = example.get("weak_leaves", [])
        score, feedback = combined_improvement_score(hypotheses, weak_leaves)
        return score, {"score": score, "feedback": feedback, "hypothesis_count": len(hypotheses)}
