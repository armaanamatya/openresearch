"""Adapter: campaign ``AttemptAssessment`` cohort → ASHA scheduler decisions.

Bridges the campaign's recorded per-attempt evidence onto the campaign-agnostic
``asha_scheduler`` core. Pure + duck-typed (reads only ``attempt_n`` /
``final_report.score`` / ``failure_class``), so it stays decoupled from the full
assessment machinery and is **unwired** until a flag-gated call site
(``OPENRESEARCH_SCHEDULER_TREE``) adopts it.

Mapping decisions (the substance of the integration):

- ``broken`` (→ true-KILL) is CONSERVATIVE — only a provably-dead-network
  ``failure_class`` (``"training_diverged"``, from ``dead_training_guard``) maps
  to breakage. Repairable classes (``cell_execution_error``,
  ``fabrication_suspected``, ``all_models_failed``, …) map to ``broken=False`` →
  FREEZE, never kill.
- ``score`` = ``final_report.score`` (``None`` when the report is missing).
- Per-attempt ``gpu_usd`` is supplied by the caller — the campaign tracks
  *cumulative* spend, not per-attempt GPU-$ yet (an ENRICHMENT gap); absent → 0.0,
  so the width meter degrades to the geometric ``eta`` fallback.
- ``branch_type`` is ``"faithful"`` (the campaign has no typed branches yet) and
  ``is_safety_bracket`` is ``False`` (no Hyperband ``s=0`` slot yet) — both are
  enrichment gaps the full integration fills.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from backend.agents.rlm.asha_scheduler import (
    BranchObservation,
    RungConfig,
    SchedulerDecision,
    asha_decide,
)

# Provably-broken-code failure classes → the ONLY true-KILL. Deliberately narrow;
# every other class is repairable-and-frozen (never deleted). Extend only with
# unambiguous breakage (a dead/diverged network), never a repairable class.
_BREAKAGE_CLASSES: frozenset[str] = frozenset({"training_diverged"})


def observation_from_assessment(
    assessment: Any,
    *,
    gpu_usd: float = 0.0,
) -> BranchObservation:
    """Map one ``AttemptAssessment`` (duck-typed) onto a ``BranchObservation``."""
    report = getattr(assessment, "final_report", None)
    score = getattr(report, "score", None) if report is not None else None
    failure_class = getattr(assessment, "failure_class", None)
    return BranchObservation(
        branch_id=str(getattr(assessment, "attempt_n", "?")),
        branch_type="faithful",
        score=score,
        gpu_usd=gpu_usd,
        broken=failure_class in _BREAKAGE_CLASSES,
        is_safety_bracket=False,
    )


def asha_decide_for_assessments(
    assessments: Sequence[Any],
    config: RungConfig,
    *,
    gpu_usd_by_attempt: Mapping[int, float] | None = None,
) -> list[SchedulerDecision]:
    """Run the ASHA halving decision over a same-paper cohort of attempts."""
    costs = gpu_usd_by_attempt or {}
    obs = [
        observation_from_assessment(
            a, gpu_usd=float(costs.get(getattr(a, "attempt_n", -1), 0.0))
        )
        for a in assessments
    ]
    return asha_decide(obs, config)
