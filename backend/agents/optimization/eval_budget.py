"""Per-eval budget enforcement for GEPA optimization runs.

Wraps ``RunBudget`` with kill-on-overrun semantics that map a runaway eval
to ``score=0.0`` + a structured timeout trace, never a raised exception.
Only systemic setup failures (missing API keys, broken docker daemon)
propagate.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class EvalBudgetEnforcer:
    """Bounds wall-clock + USD per (candidate, paper) evaluation."""

    max_usd: float = 0.50
    max_wall_clock_seconds: float = 1800.0  # 30 min default
    max_parallel_pods: int = 2

    @classmethod
    def from_env(cls) -> "EvalBudgetEnforcer":
        return cls(
            max_usd=float(os.environ.get("REPROLAB_GEPA_MAX_USD_PER_EVAL", "0.50")),
            max_wall_clock_seconds=float(
                os.environ.get("REPROLAB_GEPA_MAX_WALL_S_PER_EVAL", "1800")
            ),
            max_parallel_pods=int(
                os.environ.get("REPROLAB_GEPA_MAX_PARALLEL_PODS", "2")
            ),
        )

    def to_run_budget(self) -> Any:
        """Return a ``RunBudget`` configured with these limits."""
        from backend.services.runs.budget import RunBudget

        return RunBudget(
            max_usd=self.max_usd,
            max_wall_clock_seconds=int(self.max_wall_clock_seconds),
        )

    def timeout_record(self, *, component_id: str, example_id: str, elapsed_s: float) -> dict:
        """Synthetic §4.4-shape record returned when an eval is killed."""
        return {
            "component_id": component_id,
            "example_id": example_id,
            "paper_archetype": "unknown",
            "score": 0.0,
            "hermes_clamped_score": 0.0,
            "input": {
                "rubric_overall_before": 0.0,
                "rubric_areas_before": {},
                "weak_leaves_before": [],
                "current_results_digest": "",
            },
            "candidate_output": {},
            "execution_trace": {
                "rubric_overall_after": 0.0,
                "rubric_delta_areas": {},
                "weak_leaves_after": [],
                "run_experiment_success": False,
                "repair_summaries": [],
                "forced_iteration_warnings": 0,
                "blanket_decline_count": 0,
                "hermes": {
                    "status": "system_error",
                    "unsupported_claims": [],
                    "evidence_refs_snippets": [],
                },
                "timeout": {
                    "elapsed_s": round(elapsed_s, 2),
                    "max_wall_clock_s": self.max_wall_clock_seconds,
                    "max_usd": self.max_usd,
                },
            },
        }


__all__ = ["EvalBudgetEnforcer"]
