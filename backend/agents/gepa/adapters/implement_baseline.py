"""GEPA proxy evaluator for implement_baseline — no real sub-agent call.

Uses structural validity of a synthetic output dict as the proxy metric.
The real sub-agent is too expensive (30-120s) to call during GEPA's mini-loop.
"""
from __future__ import annotations

from typing import Any

from backend.agents.gepa.metrics.baseline_metrics import code_plan_structural_validity


class ImplementBaselineProxyEvaluator:
    """Score implement_baseline prompt using structural validity of example dict.

    The example IS the expected output structure — we score how complete it is.
    """

    def __call__(
        self,
        candidate: dict[str, str] | str,
        example: dict | None = None,
        **kwargs: Any,
    ) -> tuple[float, dict]:
        score, feedback = code_plan_structural_validity(example or {})
        return score, {"score": score, "feedback": feedback}
