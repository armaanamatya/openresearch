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
  deterministic per-attempt spend; absent → 0.0, so the width meter degrades to
  the geometric ``eta`` fallback.
- ``branch_type`` and ``is_safety_bracket`` are durable campaign metadata. Bad
  duck-typed/legacy values fail closed to faithful/non-safety; the adapter never
  infers either from a grade, score, or failure class.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from backend.agents.rlm.asha_scheduler import (
    BranchType,
    BranchObservation,
    RungConfig,
    SchedulerDecision,
    asha_decide,
)

# Provably-broken-code failure classes → the ONLY true-KILL. Deliberately narrow;
# every other class is repairable-and-frozen (never deleted). Extend only with
# unambiguous breakage (a dead/diverged network), never a repairable class.
_BREAKAGE_CLASSES: frozenset[str] = frozenset({"training_diverged"})
_BRANCH_TYPES: frozenset[str] = frozenset({"faithful", "ambiguity", "discovery"})
_MISSING = object()


def _scheduler_metadata_from_assessment(assessment: Any) -> tuple[BranchType, bool]:
    """Use legacy defaults only for *absent* attrs; reject bad explicit attrs.

    The shadow caller catches this and omits the whole advisory, so malformed
    evidence can never earn a scheduler exemption by being silently coerced.
    """
    raw_branch_type = getattr(assessment, "branch_type", _MISSING)
    if raw_branch_type is _MISSING:
        branch_type: BranchType = "faithful"
    elif isinstance(raw_branch_type, str) and raw_branch_type in _BRANCH_TYPES:
        branch_type = raw_branch_type  # type: ignore[assignment]
    else:
        raise ValueError(f"branch_type must be one of {sorted(_BRANCH_TYPES)}, got {raw_branch_type!r}")

    raw_safety_bracket = getattr(assessment, "is_safety_bracket", _MISSING)
    if raw_safety_bracket is _MISSING:
        is_safety_bracket = False
    elif type(raw_safety_bracket) is bool:
        is_safety_bracket = raw_safety_bracket
    else:
        raise ValueError(f"is_safety_bracket must be bool, got {raw_safety_bracket!r}")
    if is_safety_bracket and branch_type != "faithful":
        raise ValueError("is_safety_bracket is allowed only for a faithful branch")
    return branch_type, is_safety_bracket


def observation_from_assessment(
    assessment: Any,
    *,
    gpu_usd: float = 0.0,
) -> BranchObservation:
    """Map one ``AttemptAssessment`` (duck-typed) onto a ``BranchObservation``."""
    report = getattr(assessment, "final_report", None)
    score = getattr(report, "score", None) if report is not None else None
    failure_class = getattr(assessment, "failure_class", None)
    branch_type, is_safety_bracket = _scheduler_metadata_from_assessment(assessment)
    return BranchObservation(
        branch_id=str(getattr(assessment, "attempt_n", "?")),
        branch_type=branch_type,
        score=score,
        gpu_usd=gpu_usd,
        broken=failure_class in _BREAKAGE_CLASSES,
        is_safety_bracket=is_safety_bracket,
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
