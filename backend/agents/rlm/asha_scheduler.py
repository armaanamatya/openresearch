"""ASHA-style successive-halving decision core for the reproduction scheduler.

Pure logic — no I/O, no campaign coupling. Given a cohort of **same-paper**
branches observed at one rung, decide which to PROMOTE (climb to the next rung),
FREEZE (evict to the revivable frozen pool — underperforming-but-faithful), or
KILL (true-delete — provably-broken code ONLY). A thin adapter (future, wired into
``campaign_policy.decide`` behind ``OPENRESEARCH_SCHEDULER_TREE``) maps the
campaign's ``AttemptAssessment`` onto ``BranchObservation``; this module stays
inert until then.

Design (locked — see
``docs/superpowers/specs/2026-07-18-dynamic-reproduction-scheduler-design.md``):

- **Two meters, never merged.** The fidelity LADDER is metered in optimizer steps
  and is owned by the caller (this module sees one rung at a time). The promotion
  WIDTH is metered here: ``k = min(rung_gpu_usd_budget / cost_per_branch,
  a100_cap)`` — cost-and-capacity-derived, not a constant — with a geometric
  ``eta`` fallback when budget/cost are unavailable.
- **Deterministic kill only, and reversible by default.** Breakage (``broken``,
  set upstream by the deterministic kill predicate / ``dead_training_guard``) is
  the ONLY true-KILL (irreversible). Underperformance is a *finding* → FREEZE
  (revivable), never kill.
- **Rank-inversion is mainline.** The envelope-shrink gate refuses to cull until
  the score spread exceeds ``noise_floor``; safety-bracket branches (Hyperband
  ``s=0``) are exempt from halving so a wrongly-deranked-but-correct branch still
  runs to ``r_max``.
- **Typed branches.** faithful-replication is kept by FIDELITY (never killed for a
  low score); discovery is quarantined (scheduled on its own criterion, never
  halved against faithful).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

BranchType = Literal["faithful", "ambiguity", "discovery"]
BranchAction = Literal["promote", "freeze", "kill"]

_DEFAULT_ETA = 3.0


@dataclass(frozen=True)
class BranchObservation:
    """One branch's measured state at a rung. Every field is measured evidence."""

    branch_id: str
    branch_type: BranchType = "faithful"
    # Defining-metric value at this rung; ``None`` when no measurable score exists
    # (ran but produced nothing rankable) — such a branch is FROZEN, never promoted.
    score: float | None = None
    gpu_usd: float = 0.0  # measured spend so far (width-meter input)
    # Deterministic breakage (NaN/Inf, frozen params, diverging loss, output
    # collapse, shape mismatch) detected upstream → the only true-KILL.
    broken: bool = False
    # Hyperband s=0 full-budget slot — exempt from halving (rank-inversion insurance).
    is_safety_bracket: bool = False


@dataclass(frozen=True)
class RungConfig:
    rung: int = 0
    # Width meter: GPU-$ available for promoted branches at this rung.
    gpu_usd_budget: float | None = None
    # Hard concurrency cap (physical GPU count) — the capacity half of the width meter.
    a100_cap: int | None = None
    eta: float = _DEFAULT_ETA  # geometric fallback reduction factor
    higher_is_better: bool = True  # metric direction (accuracy ↑ / loss ↓)
    # Envelope-shrink gate: do not cull until the score spread exceeds this.
    noise_floor: float = 0.0


@dataclass(frozen=True)
class SchedulerDecision:
    branch_id: str
    action: BranchAction
    reason: str


def _promotion_width(
    n_ranked: int, cohort: Sequence[BranchObservation], config: RungConfig
) -> int:
    """``k = min(budget / cost_per_branch, a100_cap)`` clamped to ``[0, n_ranked]``;
    geometric ``eta`` fallback when no budget/cost signal is available.

    ``cost_per_branch`` is the *max* observed spend in the cohort (conservative —
    it never over-promotes past the $ budget)."""
    # An explicit zero is not an unavailable meter. It means that no ordinary
    # branch may consume another rung; retaining a minimum-one promotion here
    # would silently breach the caller's remaining-GPU-$ constraint.
    if config.gpu_usd_budget is not None and config.gpu_usd_budget <= 0:
        return 0

    k_cost: int | None = None
    if config.gpu_usd_budget is not None:
        costs = [b.gpu_usd for b in cohort if b.gpu_usd > 0]
        cost_per_branch = max(costs) if costs else 0.0
        if cost_per_branch > 0:
            k_cost = int(config.gpu_usd_budget // cost_per_branch)
    if k_cost is None:
        k = math.ceil(n_ranked / config.eta) if config.eta > 1 else n_ranked
    else:
        k = k_cost
    if config.a100_cap is not None:
        k = min(k, config.a100_cap)
    return max(0, min(k, n_ranked))


def asha_decide(
    branches: Sequence[BranchObservation],
    config: RungConfig,
) -> list[SchedulerDecision]:
    """Decide promote/freeze/kill for one same-paper cohort at a rung. Pure.

    Order of operations is load-bearing: breakage-kill → discovery-quarantine →
    safety-bracket exemption → N=1 → envelope-shrink gate → top-k halving.
    """
    decisions: list[SchedulerDecision] = []

    # 1. Breakage is the ONLY true-KILL (irreversible) — provably-broken code, not
    #    underperformance. Applies regardless of branch type.
    live: list[BranchObservation] = []
    for b in branches:
        if b.broken:
            decisions.append(SchedulerDecision(b.branch_id, "kill", "breakage_true_kill"))
        else:
            live.append(b)

    # 2. Discovery is quarantined — never halved against faithful; promoted on its
    #    own criterion (its budget is enforced by the caller).
    cohort: list[BranchObservation] = []
    for b in live:
        if b.branch_type == "discovery":
            decisions.append(
                SchedulerDecision(b.branch_id, "promote", "discovery_quarantined")
            )
        else:
            cohort.append(b)

    # 3. Safety-bracket branches (Hyperband s=0) are exempt from halving.
    halvable: list[BranchObservation] = []
    for b in cohort:
        if b.is_safety_bracket:
            decisions.append(
                SchedulerDecision(b.branch_id, "promote", "safety_bracket_exempt")
            )
        else:
            halvable.append(b)

    if not halvable:
        return decisions

    # 4. A branch with no measurable score cannot justify promotion → FREEZE
    #    (revivable), never kill.
    ranked = [b for b in halvable if b.score is not None]
    for b in halvable:
        if b.score is None:
            decisions.append(
                SchedulerDecision(b.branch_id, "freeze", "no_score_at_rung")
            )
    if not ranked:
        return decisions

    # 5. N=1 degradation — nothing to rank against, but it is still bound by
    # both width meters. A lone unscored branch was already frozen above.
    if len(ranked) == 1:
        if _promotion_width(1, halvable, config) >= 1:
            decisions.append(
                SchedulerDecision(ranked[0].branch_id, "promote", "n1_single_scored")
            )
        else:
            decisions.append(
                SchedulerDecision(ranked[0].branch_id, "freeze", "width_meter_exhausted")
            )
        return decisions

    scores = [b.score for b in ranked if b.score is not None]
    spread = max(scores) - min(scores)

    # 6. Envelope-shrink gate — refuse score-derived culling until curves
    # separate beyond noise. It never permits a physical/dollar-cap bypass.
    if spread <= config.noise_floor:
        k = _promotion_width(len(ranked), halvable, config)
        for index, branch in enumerate(ranked):
            if index < k:
                decisions.append(
                    SchedulerDecision(branch.branch_id, "promote", "envelope_not_shrunk")
                )
            else:
                decisions.append(
                    SchedulerDecision(branch.branch_id, "freeze", "width_meter_exhausted")
                )
        return decisions

    # 7. Halving: promote top-k by score; FREEZE the rest (never kill — under-
    #    performance is a finding, and faithful branches are kept by fidelity).
    ordered = sorted(
        ranked,
        key=lambda b: (b.score if b.score is not None else 0.0),
        reverse=config.higher_is_better,
    )
    k = _promotion_width(len(ordered), halvable, config)
    for i, b in enumerate(ordered):
        if i < k:
            decisions.append(SchedulerDecision(b.branch_id, "promote", "top_k_by_score"))
        else:
            decisions.append(
                SchedulerDecision(b.branch_id, "freeze", "halved_below_topk")
            )
    return decisions
