"""Campaign money policy (Unit 2) + DECIDE policy (Unit 5).

Part 1: spec §10.1 (money red lines) + §10.2 (headroom rule), Codex F2
(``--max-usd`` is LLM/provider spend ONLY, never GPU) / F3 (``stage_on_gpu``
bills provisioning to the GPU meters) — ``CampaignBudget``/``AttemptEnvelope``,
``derive_envelope``, ``check_enforceability``, ``attempt_estimate``.

Part 2: spec §8.1 (quarantine vs. seeding), §8.2 (the DECIDE rule table),
§8.3 (budget-gated width), §8.4 (the typed novelty gate), Codex F5
(campaign-owned, guard-filtered champion selection — the score-ranked rails
never choose the campaign seed) and F10 (the novelty fingerprint hashes only
typed fields; LLM prose is never an input) — ``decide``, ``select_champion``,
``seeding_pool``, ``campaign_floor``, ``lineage_arms``, ``next_scope_rung``,
``width_for_next``, ``directives_fingerprint``.

Pure stdlib policy math — no I/O, no env reads, no ``backend.*`` imports
beyond :mod:`attempt_assessment` (the ``AttemptAssessment`` type DECIDE
reads) and :mod:`campaign_types` (the shared ``CampaignSpend`` record,
re-exported here as ``campaign_policy.CampaignSpend``). Every input is an
explicit argument.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from backend.agents.rlm.attempt_assessment import AttemptAssessment
from backend.agents.rlm.asha_scheduler import BranchType
from backend.agents.rlm.campaign_types import CampaignSpend
from backend.agents.rlm.doomed_run_comparator import DOOMED_KILLED_CLASS

_BRANCH_TYPES: frozenset[str] = frozenset({"faithful", "ambiguity", "discovery"})


def _validate_scheduler_plan_metadata(plan: NextAttemptPlan) -> None:
    """Prevent malformed scheduler metadata from becoming a durable decision."""
    if plan.branch_type not in _BRANCH_TYPES:
        raise ValueError(f"branch_type must be one of {sorted(_BRANCH_TYPES)}, got {plan.branch_type!r}")
    if type(plan.is_safety_bracket) is not bool:
        raise ValueError(f"is_safety_bracket must be bool, got {plan.is_safety_bracket!r}")

# --- Exceptions --------------------------------------------------------


class EnvelopeExhausted(Exception):
    """A campaign meter's remaining budget is already below its per-attempt
    floor. DECIDE's budget-floor rule should have stopped the campaign
    before ``derive_envelope`` is ever called in this state — this
    exception is the second lock."""

    def __init__(self, meter: str, remaining: float, floor: float) -> None:
        self.meter = meter
        self.remaining = remaining
        self.floor = floor
        super().__init__(
            f"campaign meter {meter!r} exhausted: remaining={remaining:g} < floor={floor:g}"
        )


class EnforceabilityError(Exception):
    """One or more campaign money meters cannot be enforced on the chosen
    driver/backend. Fail-closed (spec §10.1): refuse to launch unattended
    rather than launch with a meter nothing will actually check."""

    def __init__(self, reasons: tuple[str, ...]) -> None:
        self.reasons = reasons
        super().__init__(f"campaign envelope unenforceable: {', '.join(reasons)}")


# --- Constants -----------------------------------------------------------

#: spec §10.2 — the VM control-plane ceiling always sits >=30min above the
#: attempt wall clock so a finalize/salvage pass has room to run.
FINALIZE_HEADROOM_S = 1800.0
#: below this, a wall-clock allotment is not worth launching an attempt for.
MIN_USEFUL_WALL_S = 900.0
#: per-meter minimum viable envelope share (conservative v1 constants).
ENVELOPE_FLOORS: dict[str, float] = {
    "llm_usd": 1.0,
    "gpu_usd": 1.0,
    "gpu_hours": 0.25,
    "wall_s": 1800.0,
}
#: per-attempt wall clock used only when the campaign has no aggregate
#: wall-clock meter (``CampaignBudget.max_wall_clock_s is None``).
DEFAULT_ATTEMPT_WALL_S = 21600.0
#: tiering strategies that already stage on cheap CPU ahead of the GPU
#: lease, satisfying the F3 require-cpu-tier hard guarantee (rule 6).
_TIERING_STRATEGIES_SATISFYING_CPU_TIER: frozenset[str] = frozenset(
    {"machine_type_flip", "cpu_warm_disk_then_gpu_attach"}
)
#: the three campaign-wide money meters CampaignBudget validates as
#: finite-positive (max_wall_clock_s is optional, validated separately).
_MONEY_METERS: tuple[str, ...] = ("max_llm_usd", "max_gpu_usd", "max_gpu_hours")


def _fmt_g(value: float) -> str:
    """Stable, arg-parse-friendly float formatting for CLI/env string values."""
    return f"{float(value):g}"


# --- CampaignBudget / CampaignSpend / AttemptEnvelope ---------------------


@dataclass(frozen=True)
class CampaignBudget:
    """Campaign-wide money ceiling. Three money meters (spec §10.1, F1/F2)
    kept structurally separate, never conflated: LLM/provider spend, GPU
    dollars, GPU-hours. ``max_wall_clock_s=None`` means no aggregate
    wall-clock ceiling — each attempt still gets :data:`DEFAULT_ATTEMPT_WALL_S`.
    """

    max_llm_usd: float
    max_gpu_usd: float
    max_gpu_hours: float
    max_attempts: int = 6
    max_wall_clock_s: float | None = None

    def __post_init__(self) -> None:
        for name in _MONEY_METERS:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"CampaignBudget.{name} must be a finite positive number, got {value!r}"
                )
        if self.max_wall_clock_s is not None and (
            not math.isfinite(self.max_wall_clock_s) or self.max_wall_clock_s <= 0
        ):
            raise ValueError(
                "CampaignBudget.max_wall_clock_s must be a finite positive number "
                f"or None, got {self.max_wall_clock_s!r}"
            )
        if self.max_attempts < 1:
            raise ValueError(
                f"CampaignBudget.max_attempts must be >= 1, got {self.max_attempts!r}"
            )


@dataclass(frozen=True)
class AttemptEnvelope:
    """The per-attempt slice of the campaign budget, produced by
    :func:`derive_envelope`."""

    llm_usd: float
    gpu_usd: float
    gpu_hours: float
    wall_s: float
    vm_ceiling_s: float

    def to_dict(self) -> dict[str, float]:
        return {
            "llm_usd": self.llm_usd,
            "gpu_usd": self.gpu_usd,
            "gpu_hours": self.gpu_hours,
            "wall_s": self.wall_s,
            "vm_ceiling_s": self.vm_ceiling_s,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> AttemptEnvelope:
        """Reconstruct from a persisted mapping. Strict — see
        ``CampaignSpend.from_mapping`` for the fail-closed rationale."""
        return cls(
            llm_usd=float(mapping["llm_usd"]),
            gpu_usd=float(mapping["gpu_usd"]),
            gpu_hours=float(mapping["gpu_hours"]),
            wall_s=float(mapping["wall_s"]),
            vm_ceiling_s=float(mapping["vm_ceiling_s"]),
        )


# --- Enforcement context + plan --------------------------------------------


@dataclass(frozen=True)
class EnforcementContext:
    """Everything :func:`check_enforceability` needs about the chosen
    driver/backend to map an :class:`AttemptEnvelope` onto real, checkable
    enforcement knobs."""

    driver_kind: str  # "live" | "unified"
    sandbox: str  # "local"|"docker"|"runpod"|"gcp"|"gke"|"azure"|...
    mode: str  # "unattended" | "checkpoint"
    tiering_strategy: str | None  # None when no VM provider involved
    max_gpu_count: int  # >=1; the structural bound used for hours/$ math
    gpu_usd_per_hr: float | None  # known rate, or None
    require_cpu_tier: bool  # OPENRESEARCH_CAMPAIGN_REQUIRE_CPU_TIER
    provision_overhead_s: float = 900.0  # stage_on_gpu provision estimate


@dataclass(frozen=True)
class EnforcementPlan:
    """The concrete result of :func:`check_enforceability`: exact CLI/env
    knobs plus the co-tightened wall clock and VM ceiling actually used."""

    cli_args: tuple[tuple[str, str], ...]
    env: Mapping[str, str]
    effective_wall_s: float
    vm_ceiling_s: float
    provision_charged_to_gpu: bool
    notes: tuple[str, ...]


# --- derive_envelope --------------------------------------------------------


def derive_envelope(
    budget: CampaignBudget, spent: CampaignSpend, *, attempts_completed: int
) -> AttemptEnvelope:
    """Split what remains of the campaign budget into one attempt's slice.

    Per meter: ``envelope_m = min(remaining_m, max(floor_m, remaining_m /
    expected_remaining))`` where ``expected_remaining = max(1,
    budget.max_attempts - attempts_completed)``. Raises
    :class:`EnvelopeExhausted` if any meter's remaining is already below its
    floor — the second lock (DECIDE's budget-floor rule should have stopped
    the campaign before this is ever called in that state).

    ``wall_s`` is only subject to the remaining/floor math when the campaign
    has an aggregate wall-clock meter (``budget.max_wall_clock_s`` set);
    otherwise it is per-attempt-only and fixed at
    :data:`DEFAULT_ATTEMPT_WALL_S`.
    """
    expected_remaining = max(1, budget.max_attempts - attempts_completed)

    def _share(remaining: float, meter: str) -> float:
        floor = ENVELOPE_FLOORS[meter]
        if remaining < floor:
            raise EnvelopeExhausted(meter, remaining, floor)
        return min(remaining, max(floor, remaining / expected_remaining))

    llm_usd = _share(budget.max_llm_usd - spent.llm_usd, "llm_usd")
    gpu_usd = _share(budget.max_gpu_usd - spent.gpu_usd, "gpu_usd")
    gpu_hours = _share(budget.max_gpu_hours - spent.gpu_hours, "gpu_hours")

    if budget.max_wall_clock_s is not None:
        wall_s = _share(budget.max_wall_clock_s - spent.wall_s, "wall_s")
    else:
        wall_s = DEFAULT_ATTEMPT_WALL_S

    return AttemptEnvelope(
        llm_usd=llm_usd,
        gpu_usd=gpu_usd,
        gpu_hours=gpu_hours,
        wall_s=wall_s,
        vm_ceiling_s=wall_s + FINALIZE_HEADROOM_S,
    )


# --- check_enforceability ---------------------------------------------------


def check_enforceability(envelope: AttemptEnvelope, ctx: EnforcementContext) -> EnforcementPlan:
    """Map an :class:`AttemptEnvelope` onto real CLI/env enforcement knobs
    for ``ctx``'s driver/backend (Codex F2/F3).

    Fail-closed: every rule below runs regardless of whether an earlier rule
    already failed, and ALL unenforceable reasons are collected into a
    single :class:`EnforceabilityError` — an operator sees the whole picture
    at once instead of fixing one problem only to hit the next.
    """
    reasons: list[str] = []
    notes: list[str] = []
    cli_args: list[tuple[str, str]] = []
    env: dict[str, str] = {}

    # Rule 1 — LLM meter (RunBudget.max_usd / cost_ledger) is always enforceable.
    cli_args.append(("--max-usd", _fmt_g(envelope.llm_usd)))

    # Rule 2 — wall meter co-tightened against the GPU-derived bounds so
    # neither gpu_hours nor gpu_usd can be structurally overrun (there is no
    # live gpu_hours knob; gpu_hours/gpu_usd are enforced THROUGH wall clock).
    stage_on_gpu = ctx.tiering_strategy == "stage_on_gpu"
    rate_known = ctx.gpu_usd_per_hr is not None and ctx.gpu_usd_per_hr > 0
    gpu_count_known = ctx.max_gpu_count >= 1

    bounds = [envelope.wall_s]

    if gpu_count_known:
        hours_bound_s = envelope.gpu_hours / ctx.max_gpu_count * 3600.0
        if stage_on_gpu:
            hours_bound_s -= ctx.provision_overhead_s
        bounds.append(hours_bound_s)

    if rate_known and ctx.sandbox != "local" and gpu_count_known:
        usd_bound_s = envelope.gpu_usd / (ctx.gpu_usd_per_hr * ctx.max_gpu_count) * 3600.0
        if stage_on_gpu:
            usd_bound_s -= ctx.provision_overhead_s
        bounds.append(usd_bound_s)

    effective_wall_s = min(bounds)
    if effective_wall_s < MIN_USEFUL_WALL_S:
        reasons.append("wall_clock_starved")
    cli_args.append(("--max-wall-clock", _fmt_g(effective_wall_s)))

    # Rule 3 — gpu_usd meter: vacuous on local (owned GPUs, $0); on a cloud
    # sandbox it needs a known $/hr rate or it cannot be trusted/enforced.
    if ctx.sandbox == "local":
        notes.append("gpu_usd_vacuous_local")
    elif not rate_known:
        reasons.append("gpu_usd_unenforceable:no_rate")
    else:
        cli_args.append(("--max-run-gpu-usd", _fmt_g(envelope.gpu_usd)))
        env["OPENRESEARCH_MAX_RUN_GPU_USD"] = _fmt_g(envelope.gpu_usd)

    # Rule 4 — gpu_hours meter is structural, via the wall co-tightening in
    # rule 2; an unknown GPU count means it could not be bounded there at all.
    if not gpu_count_known:
        reasons.append("gpu_hours_unenforceable:unknown_gpu_count")

    # Rule 5 — VM control-plane ceiling is ALWAYS explicit; never the 28h
    # ``max-run-duration`` default.
    vm_ceiling_s = effective_wall_s + FINALIZE_HEADROOM_S

    # Rule 5b — on runpod, ``--max-pod-seconds`` is the control-plane
    # ceiling analog (``RunBudget.max_pod_seconds`` enforces it); make it
    # explicit too, appended after --max-run-gpu-usd for a deterministic
    # cli_args order.
    if ctx.sandbox == "runpod":
        cli_args.append(("--max-pod-seconds", _fmt_g(vm_ceiling_s)))

    # Rule 6 — F3 hard guarantee: an unattended stage_on_gpu run needs a real
    # CPU tier ahead of the GPU lease, or an explicit alternative tiering.
    if (
        ctx.require_cpu_tier
        and ctx.mode == "unattended"
        and ctx.tiering_strategy not in _TIERING_STRATEGIES_SATISFYING_CPU_TIER
    ):
        reasons.append(f"cpu_tier_required:{ctx.tiering_strategy}")

    # Rule 7 — stage_on_gpu bills provisioning to the GPU meters (F3).
    if stage_on_gpu:
        notes.append("stage_on_gpu:provision_billed_to_gpu")

    if reasons:
        raise EnforceabilityError(tuple(reasons))

    return EnforcementPlan(
        cli_args=tuple(cli_args),
        env=dict(env),
        effective_wall_s=effective_wall_s,
        vm_ceiling_s=vm_ceiling_s,
        provision_charged_to_gpu=stage_on_gpu,
        notes=tuple(notes),
    )


# --- attempt_estimate --------------------------------------------------------


def attempt_estimate(*, est_gpu_hours: float, est_usd: float, ctx: EnforcementContext) -> CampaignSpend:
    """Conservative next-attempt spend estimate for DECIDE/PLAN's lookahead.

    Adds the ``stage_on_gpu`` provisioning overhead to BOTH ``gpu_hours`` and
    ``gpu_usd`` when a known rate exists (F3: provisioning is GPU-billed, not
    free). ``llm_usd`` uses the fixed envelope floor as a v1 conservative
    constant; ``wall_s`` is not estimated here (the envelope's wall bound
    already covers it).
    """
    gpu_hours = est_gpu_hours
    gpu_usd = est_usd

    rate_known = ctx.gpu_usd_per_hr is not None and ctx.gpu_usd_per_hr > 0
    if ctx.tiering_strategy == "stage_on_gpu" and rate_known:
        provision_hours = ctx.provision_overhead_s / 3600.0 * ctx.max_gpu_count
        gpu_hours += provision_hours
        gpu_usd += provision_hours * ctx.gpu_usd_per_hr

    return CampaignSpend(
        llm_usd=ENVELOPE_FLOORS["llm_usd"],
        gpu_usd=gpu_usd,
        gpu_hours=gpu_hours,
        wall_s=0.0,
    )


# ===========================================================================
# Part 2 (Unit 5) — DECIDE: terminal rules, guard-filtered champion
# selection (F5), lineage arms, scope ladder, plateau, and the typed
# prose-free novelty fingerprint (F10). Spec §8.1-§8.4.
# ===========================================================================


@dataclass(frozen=True)
class NextAttemptPlan:
    """One candidate shape for the next attempt. ``lineage_arms`` returns
    these in preference order; ``decide`` adopts arm 0's lineage/seed fields
    but recomputes ``scope_rung``/``width`` itself (orthogonal to lineage
    choice, spec §8.2)."""

    lineage: str  # "fresh" | "champion" | "runner_up"
    seed_attempt_n: int | None
    seed_pointer: str | None  # archived code/ path, campaign-selected
    scope_rung: int
    width: int
    # Scheduler metadata is deliberately trailing + defaulted: historical
    # decision rows and every existing positional constructor remain the
    # faithful, non-safety serial policy unless an explicit future policy
    # supplies otherwise.
    branch_type: BranchType = "faithful"
    is_safety_bracket: bool = False

    def __post_init__(self) -> None:
        _validate_scheduler_plan_metadata(self)
        if self.is_safety_bracket and self.branch_type != "faithful":
            raise ValueError("is_safety_bracket is allowed only for a faithful branch")


@dataclass(frozen=True)
class Decision:
    """One DECIDE verdict: terminal ``kind``, the machine id of the rule row
    that fired, an optional ``stop_reason``, and (CONTINUE only) the next
    attempt's plan.

    ``champion_attempt_n`` (spec §5 "the champion-so-far shipped" + §12
    deliverable) is set on EVERY terminal branch — REPRODUCED/CONTRADICTED/
    INFEASIBLE/EXHAUSTED all ship a champion pointer, not just the happy
    path. ``reproduction_campaign._finish_from_decision`` reads it straight
    into ``campaign.json``'s terminal payload
    (``decision.get("champion_attempt_n")``); ``campaign_report.py`` renders
    "none (no guard-clean attempt)" when it is ``None``. REPRODUCED uses the
    specific qualifying attempt's ``attempt_n`` (rule 1's checks are a
    strict superset of ``select_champion``'s guard filter, so the qualifying
    attempt IS the champion by construction); the other terminals use
    ``select_champion(assessments).attempt_n``, or ``None`` when nothing
    guard-clean has run yet. CONTINUE (non-terminal) always leaves it
    ``None``."""

    kind: str  # "REPRODUCED"|"CONTRADICTED"|"INFEASIBLE"|"EXHAUSTED"|"CONTINUE"
    rule: str
    stop_reason: str | None
    next_plan: NextAttemptPlan | None
    champion_attempt_n: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Full ``dataclasses.asdict`` payload — ``next_plan`` included as a
        nested dict, not a dataclass instance — matching the dict-shaped
        contract ``reproduction_campaign.py`` consumes decisions through
        (``CampaignStages.decide -> dict``, ``dict(decision)``,
        ``decision.get(...)``). Lets a future composition layer hand a
        ``Decision`` straight to the state machine without a lossy
        hand-rolled conversion."""
        payload = asdict(self)
        # These scheduler enrichments are additive. Keep default serial-policy
        # decisions byte-for-byte compatible with historical decision rows;
        # explicit non-default branch policy remains durable.
        next_plan = payload.get("next_plan")
        if isinstance(next_plan, dict):
            assert self.next_plan is not None
            _validate_scheduler_plan_metadata(self.next_plan)
            if next_plan.get("branch_type") == "faithful":
                next_plan.pop("branch_type", None)
            if next_plan.get("is_safety_bracket") is False:
                next_plan.pop("is_safety_bracket", None)
        return payload


@dataclass(frozen=True)
class PolicyConfig:
    """Tunable DECIDE knobs, resolved from ``OPENRESEARCH_CAMPAIGN_*`` env by
    the caller — this module never reads env itself."""

    max_attempts: int
    plateau_k: int = 2
    width: int = 1
    width_skip_score: float = 0.5
    ladder_len: int = 1  # rung count; full grid == ladder_len - 1


# --- Evidence counting + guard-filtered champion ranking (F5) --------------


def evidence_count(a: AttemptAssessment) -> int:
    """True evidence predicates, excluding the aggregate ``run_level_clean``
    key (derived from the others — counting it too would double-weight)."""
    return sum(1 for key, value in a.evidence_predicates.items() if key != "run_level_clean" and value)


def is_doomed_killed(a: AttemptAssessment) -> bool:
    """Spec §10.3: the campaign STOPPED PAYING for this attempt mid-flight
    (``OPENRESEARCH_DOOMED_KILL``); it is not an observation of the science.

    "We cut our losses" and "the science failed" are different facts, and
    conflating them corrupts every retry rule below: a killed attempt has no
    report (so it would trip ``report_missing_twice``), no evidence (so it
    would look like a plateau), and no verdict (so it would drag the champion
    ranking). It is therefore invisible to every SCIENCE counter — while
    still counting toward ``max_attempts`` and still accruing its real spend,
    because the money was genuinely spent and the kill count must stay bounded.
    """
    return a.failure_class == DOOMED_KILLED_CLASS


def _science_attempts(assessments: Sequence[AttemptAssessment]) -> list[AttemptAssessment]:
    """The attempts that are actually observations of the science — i.e. every
    attempt the campaign let run to its own conclusion."""
    return [a for a in assessments if not is_doomed_killed(a)]


def _meets_target_distance(a: AttemptAssessment) -> float:
    """``max(0, target - score)`` when both are numeric, else +inf (missing
    data can never out-rank a measured pair — fail-closed on trust)."""
    report = a.final_report
    if report is None or report.target is None or report.score is None:
        return math.inf
    try:
        return max(0.0, float(report.target) - float(report.score))
    except (TypeError, ValueError):
        return math.inf


def _champion_rank_key(a: AttemptAssessment) -> tuple[int, float, int, int]:
    """§8.2/F5 ranking: evidence count desc > target distance asc > leaf-pass
    count desc (``None`` worst) > attempt_n asc (deterministic tie-break)."""
    leaf = a.leaf_pass_count if a.leaf_pass_count is not None else -1
    return (-evidence_count(a), _meets_target_distance(a), -leaf, a.attempt_n)


def select_champion(assessments: Sequence[AttemptAssessment]) -> AttemptAssessment | None:
    """§8.2/F5: the campaign-owned, guard-filtered champion. Filters to
    ``grade_usable_for_terminal`` FIRST, ranks second — a higher-scoring
    guard-tripped attempt can never win. ``None`` if nothing qualifies.

    A doomed-killed attempt is excluded explicitly as well as structurally
    (the kill marks it hard-quarantined, which ``grade_usable_for_terminal``
    already rejects). Belt-and-braces on purpose: "an attempt we cut short can
    never be shipped as the champion" is a MONEY-and-honesty invariant, and it
    must not depend on one flag surviving a dict round-trip."""
    pool = [a for a in assessments if a.grade_usable_for_terminal and not is_doomed_killed(a)]
    return min(pool, key=_champion_rank_key) if pool else None


def _champion_attempt_n(assessments: Sequence[AttemptAssessment]) -> int | None:
    """``select_champion(assessments)``'s ``attempt_n``, or ``None`` when
    nothing guard-clean has run yet — the shared "champion-so-far" lookup
    every non-REPRODUCED terminal branch of :func:`decide` stamps onto its
    :class:`Decision` (spec §5, §12)."""
    champion = select_champion(assessments)
    return champion.attempt_n if champion is not None else None


def seeding_pool(assessments: Sequence[AttemptAssessment]) -> list[AttemptAssessment]:
    """§8.1: attempts usable to seed the next lineage — not hard-quarantined.
    Soft-quarantined (e.g. validator-stale) attempts stay seedable. A
    doomed-killed attempt never seeds: we interrupted it, so its ``code/`` is
    a half-finished state nobody ever evaluated."""
    return [a for a in assessments if a.usable_for_seeding and not is_doomed_killed(a)]


def campaign_floor(assessments: Sequence[AttemptAssessment]) -> float | None:
    """§8.2/F5: max score across ``grade_usable_for_terminal`` attempts only
    — never floors off a quarantined score. ``None`` if none qualify."""
    scores = [
        a.final_report.score
        for a in assessments
        if a.grade_usable_for_terminal
        and not is_doomed_killed(a)
        and a.final_report is not None
        and a.final_report.score is not None
    ]
    return max(scores) if scores else None


# --- Lineage arms (§8.2) ----------------------------------------------------


def _evidence_improved_over_prior(
    sorted_assessments: Sequence[AttemptAssessment], attempt: AttemptAssessment
) -> bool:
    """True iff ``attempt``'s evidence_count exceeds the best evidence_count
    among all strictly earlier attempts (campaign-wide, by attempt_n). The
    shared "improved over best-before-them" test used by plateau (rule 4c)
    and the champion-lineage stall check below."""
    earlier = [x for x in sorted_assessments if x.attempt_n < attempt.attempt_n]
    return evidence_count(attempt) > max((evidence_count(x) for x in earlier), default=-1)


def _fresh_plan() -> NextAttemptPlan:
    return NextAttemptPlan(
        lineage="fresh",
        seed_attempt_n=None,
        seed_pointer=None,
        scope_rung=0,
        width=1,
        branch_type="faithful",
        is_safety_bracket=False,
    )


def _seeded_plan(lineage: str, attempt: AttemptAssessment, seed_pointer: str) -> NextAttemptPlan:
    return NextAttemptPlan(
        lineage=lineage,
        seed_attempt_n=attempt.attempt_n,
        seed_pointer=seed_pointer,
        scope_rung=0,
        width=1,
        branch_type="faithful",
        is_safety_bracket=False,
    )


def _runner_up(
    assessments: Sequence[AttemptAssessment], *, exclude_attempt_n: int | None = None
) -> AttemptAssessment | None:
    """Best-ranked attempt by the same §8.2/F5 key, over the looser seeding
    pool (hard-quarantine filtered only — a soft-quarantined attempt stays
    eligible, spec §8.1/F4). With ``exclude_attempt_n`` set, this is the
    second-ranked attempt behind an already-chosen champion; left unset
    (``None``), this IS the top-ranked seedable attempt — used by
    :func:`lineage_arms` when :func:`select_champion` found no guard-clean
    champion at all (the F4 seeding-starvation fix: a soft-quarantined
    attempt "remains usable for lineage seeding" even when none is clean
    enough to be champion)."""
    pool = [a for a in seeding_pool(assessments) if exclude_attempt_n is None or a.attempt_n != exclude_attempt_n]
    return min(pool, key=_champion_rank_key) if pool else None


def _champion_lineage_stalled_twice(
    sorted_assessments: Sequence[AttemptAssessment], lineage_by_attempt: Mapping[int, str]
) -> bool:
    """True iff the champion lineage has >=2 recorded attempts and its last
    two each failed to raise evidence_count over the best-before-them."""
    science = _science_attempts(sorted_assessments)
    champion_attempts = [a for a in science if lineage_by_attempt.get(a.attempt_n) == "champion"]
    if len(champion_attempts) < 2:
        return False
    return all(not _evidence_improved_over_prior(science, a) for a in champion_attempts[-2:])


def lineage_arms(
    assessments: Sequence[AttemptAssessment],
    *,
    runs_dir_hint: Mapping[int, str],
    lineage_by_attempt: Mapping[int, str] | None = None,
) -> tuple[NextAttemptPlan, ...]:
    """Ordered candidate arms for the next attempt (§8.2 lineage rules):
    no assessments -> ``(fresh,)``; champion lineage improving ->
    ``(champion, runner_up?, fresh)``; champion lineage stalled twice
    running -> ``(runner_up?, fresh)`` (no runner-up -> ``(fresh,)``); NO
    guard-clean champion at all but a seedable (not hard-quarantined)
    attempt exists -> ``(runner_up(best-seedable), fresh)`` (F4
    seeding-starvation fix, spec §8.1: a soft-quarantined attempt — e.g.
    every attempt in a validator-less campaign — "remains usable for
    lineage seeding" even though none is clean enough to be champion;
    without this, a campaign with no guard-clean attempt would collapse to
    ``(fresh,)`` forever and seeded progress could never happen). ``decide``
    adopts arm 0 and recomputes its scope_rung/width; later arms exist for a
    caller retrying after a novelty refusal at directive synthesis (§8.4 —
    not checked here). A seed pointer is ``runs_dir_hint[attempt_n]``; an
    arm whose attempt_n is absent from the hint is skipped, not emitted
    with a null pointer.

    ``lineage_by_attempt`` (added beyond the brief's literal stub, per its
    own rule text — "the campaign passes lineage_by_attempt, see decide()")
    identifies which attempts were seeded from the current champion, needed
    for the stall check below; ``decide`` already tracks and threads it
    through. Absent/empty -> no attempt is attributable to the champion
    lineage, so the stall check never fires and champion leads by default —
    the same shape a healthy young campaign produces.
    """
    if not assessments:
        return (_fresh_plan(),)

    sorted_assessments = sorted(assessments, key=lambda a: a.attempt_n)
    champion = select_champion(assessments)
    if champion is None:
        best_seedable = _runner_up(assessments)
        if best_seedable is not None:
            seed_pointer = runs_dir_hint.get(best_seedable.attempt_n)
            if seed_pointer is not None:
                return (_seeded_plan("runner_up", best_seedable, seed_pointer), _fresh_plan())
        return (_fresh_plan(),)

    stalled = _champion_lineage_stalled_twice(sorted_assessments, lineage_by_attempt or {})

    arms: list[NextAttemptPlan] = []
    if not stalled:
        champion_pointer = runs_dir_hint.get(champion.attempt_n)
        if champion_pointer is not None:
            arms.append(_seeded_plan("champion", champion, champion_pointer))

    runner_up = _runner_up(assessments, exclude_attempt_n=champion.attempt_n)
    if runner_up is not None:
        runner_up_pointer = runs_dir_hint.get(runner_up.attempt_n)
        if runner_up_pointer is not None:
            arms.append(_seeded_plan("runner_up", runner_up, runner_up_pointer))

    arms.append(_fresh_plan())
    return tuple(arms)


# --- Scope ladder + budget-gated width (§8.2 scope ladder, §8.3) -----------


def next_scope_rung(current_rung: int, latest: AttemptAssessment | None, *, ladder_len: int) -> int:
    """Advance one rung toward the full grid when ``latest`` (the attempt
    the CALLER has already confirmed ran at ``current_rung``) is
    ``grade_usable_for_terminal`` AND measured ``meets_target`` AND carries
    the authoritative ``verdict == "reproduced"``; else hold. The verdict
    check is an AND on top of ``meets_target``, never a replacement (spec
    §4.3): a grade-high attempt whose deterministic verdict is
    ``inconclusive``/``contradicted``/``partial`` must not escalate scope
    either. Never retreats below ``current_rung`` (the target never
    shrinks, locked decision 2). Clamped to ``ladder_len - 1`` (the
    full-grid rung)."""
    max_rung = max(0, ladder_len - 1)
    green = (
        latest is not None
        and latest.grade_usable_for_terminal
        and latest.final_report is not None
        and latest.final_report.meets_target is True
        and latest.final_report.verdict == "reproduced"
    )
    return min(max((current_rung + 1) if green else current_rung, 0), max_rung)


def width_for_next(
    assessments: Sequence[AttemptAssessment],
    *,
    config: PolicyConfig,
    budget: CampaignBudget,
    spent: CampaignSpend,
    next_estimate: CampaignSpend,
) -> int:
    """§8.3: ``config.width`` candidates only when width is configured (>1),
    history is weak (no RAW recorded score >= ``config.width_skip_score`` —
    unfiltered by guards on purpose: crediting even an untrusted high score
    as "not weak" only SUPPRESSES width, the conservative/cheaper direction
    on ambiguous grounds), remaining budget covers ``width`` conservative
    estimates on EVERY money meter, and enough attempts remain. Any failing
    condition -> 1 (no width)."""
    if config.width <= 1:
        return 1

    history_weak = not any(
        a.final_report is not None
        and a.final_report.score is not None
        and a.final_report.score >= config.width_skip_score
        for a in assessments
    )
    if not history_weak:
        return 1

    width = config.width
    meters = (
        (budget.max_llm_usd - spent.llm_usd, next_estimate.llm_usd),
        (budget.max_gpu_usd - spent.gpu_usd, next_estimate.gpu_usd),
        (budget.max_gpu_hours - spent.gpu_hours, next_estimate.gpu_hours),
    )
    if any(remaining < width * estimate for remaining, estimate in meters):
        return 1

    return width if (config.max_attempts - len(assessments)) >= width else 1


# --- Typed, prose-free novelty fingerprint (§8.4, F10) ----------------------

#: quantization buckets so small remaining-budget drift does not make every
#: plan trivially "novel" (documented constants, spec §8.4).
_FINGERPRINT_ENVELOPE_QUANTA: dict[str, float] = {
    "llm_usd": 1.0,
    "gpu_usd": 1.0,
    "gpu_hours": 0.25,
    "wall_s": 900.0,
    "vm_ceiling_s": 900.0,
}


def _quantize(value: float, quantum: float) -> float:
    if quantum <= 0:
        return float(value)
    return round(float(value) / quantum) * quantum


def directives_fingerprint(
    *,
    seed_lineage: str,
    scope_rung: int,
    repair_action_kinds: Iterable[str],
    failure_classes: Iterable[str],
    envelope: Mapping[str, float],
    branch_type: BranchType = "faithful",
) -> str:
    """F10 typed novelty hash over ONLY the deterministic action schema:
    seed lineage, scope rung, the typed per-leaf repair-action KINDS (never
    justification text), the failure-class set, and the quantized attempt
    envelope, and an explicit typed branch. PROSE IS NEVER AN INPUT — the signature has no field for it,
    so two attempts differing only in LLM prose (``propose_improvements``
    output, grader justifications) hash identically and are NOT novel.
    Unknown envelope keys default to a 1.0 quantum (defensive; every known
    ``AttemptEnvelope`` field is covered above)."""
    quantized_envelope = {
        key: _quantize(value, _FINGERPRINT_ENVELOPE_QUANTA.get(key, 1.0)) for key, value in envelope.items()
    }
    payload = {
        "lineage": seed_lineage,
        "rung": scope_rung,
        "kinds": sorted(set(repair_action_kinds)),
        "classes": sorted(set(failure_classes)),
        "envelope": quantized_envelope,
    }
    # Keep the existing F10 value for the serial faithful default while
    # ensuring an explicit ambiguity/discovery fork cannot collide with it.
    # The safety marker is intentionally absent: it is a scheduler-only
    # advisory exemption, not a different hypothesis.
    if branch_type != "faithful":
        payload["branch_type"] = branch_type
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --- DECIDE (§8.2 rule table, evaluated in order) ---------------------------


def _budget_floor_stop_reason(
    budget: CampaignBudget, spent: CampaignSpend, next_estimate: CampaignSpend
) -> str | None:
    for meter in ("llm_usd", "gpu_usd", "gpu_hours"):
        if getattr(budget, f"max_{meter}") - getattr(spent, meter) < getattr(next_estimate, meter):
            return f"budget_floor:{meter}"
    if budget.max_wall_clock_s is not None and budget.max_wall_clock_s - spent.wall_s < next_estimate.wall_s:
        return "budget_floor:wall_s"
    return None


def _plateaued(sorted_assessments: Sequence[AttemptAssessment], *, plateau_k: int) -> bool:
    """§8.2 rule 4c: ``plateau_k`` CONSECUTIVE attempts with no EVIDENCE
    improvement over the best attempt before them.

    The failure-signature novelty escape was REMOVED (2026-07-13). It used to
    reset the rule whenever the latest attempt failed with a signature never
    seen before — which meant an agent that failed a DIFFERENT way every
    attempt could never plateau and never stop, and so ran to max_attempts or
    budget exhaustion while measurably learning nothing. (Campaign
    prj_09047604e591d969: $24.23 of LLM spend, 4.5h, terminal EXHAUSTED,
    score 0.000.) Novelty of a FAILURE LABEL is not progress; the deterministic
    evidence layer is the only fitness signal, here as everywhere else, so a
    run of six inventive new ways to produce zero evidence is exactly what
    "plateau" is supposed to mean.

    Doomed-killed attempts are invisible on BOTH sides of the comparison —
    neither window members nor part of the "best before them" baseline: the
    campaign stopped them, so their (absent) evidence says nothing about
    whether the science is still moving.
    """
    science = _science_attempts(sorted_assessments)
    if len(science) < plateau_k:
        return False
    return all(not _evidence_improved_over_prior(science, a) for a in science[-plateau_k:])


def _infra_signature_repeated(sorted_assessments: Sequence[AttemptAssessment]) -> bool:
    science = _science_attempts(sorted_assessments)
    if len(science) < 3:
        return False
    window = science[-3:]
    signatures = {a.failure_signature for a in window}
    return len(signatures) == 1 and None not in signatures and all(a.failure_scope == "infra" for a in window)


def _report_missing_twice(sorted_assessments: Sequence[AttemptAssessment]) -> bool:
    # Killed attempts are filtered out on BOTH counts: a killed child writes no
    # report, so without this a pair of kills would read as "the harness cannot
    # produce a report twice running" and terminate the campaign for the wrong
    # reason — and a kill landing between two genuine report_missing attempts
    # would equally wrongly MASK the real rule.
    science = _science_attempts(sorted_assessments)
    if len(science) < 2:
        return False
    latest, previous = science[-1], science[-2]
    return latest.failure_class == "report_missing" and previous.failure_class == "report_missing"


def decide(
    assessments: Sequence[AttemptAssessment],
    *,
    budget: CampaignBudget,
    spent: CampaignSpend,
    config: PolicyConfig,
    next_estimate: CampaignSpend,
    lineage_by_attempt: Mapping[int, str],
    scope_rung_by_attempt: Mapping[int, int],
    runs_dir_hint: Mapping[int, str],
    current_rung: int,
    blocking_gap: str | None = None,
) -> Decision:
    """§8.2: the deterministic retry policy, rules evaluated IN ORDER. The
    campaign adds NO LLM judgment — every branch is a pure function of
    recorded, machine-checkable :class:`AttemptAssessment` fields."""
    full_rung = config.ladder_len - 1

    # 1. REPRODUCED — a quarantined grade (hard OR soft) never satisfies
    # this, whatever its score (F4/F5); neither does an attempt the campaign
    # itself killed (§10.3). ``verdict == "reproduced"`` is an
    # AND on top of ``meets_target``, never a replacement (spec §4.3,
    # "consumers must read the authority, not the grade"): a
    # grade-high/meets_target=True attempt whose deterministic verdict is
    # inconclusive/contradicted/partial must not terminate the campaign as
    # reproduced.
    for a in assessments:
        if (
            a.grade_usable_for_terminal
            and not is_doomed_killed(a)
            and a.final_report is not None
            and a.final_report.meets_target is True
            and a.final_report.verdict == "reproduced"
            and scope_rung_by_attempt.get(a.attempt_n) == full_rung
            and a.evidence_predicates.get("run_level_clean") is True
            and a.rubric_sha256_ok is not False
        ):
            return Decision(
                kind="REPRODUCED", rule="reproduced", stop_reason=None, next_plan=None,
                champion_attempt_n=a.attempt_n,
            )

    # 2. CONTRADICTED — >=2 clean attempts from >=2 DIFFERENT labeled
    # lineages (an unlabeled/unknown lineage never counts as "differing").
    contradicted_candidates = [
        a
        for a in assessments
        if a.grade_usable_for_terminal
        and not is_doomed_killed(a)
        and a.final_report is not None
        and a.final_report.implementation_verdict == "faithful"
        and a.final_report.replication_verdict == "contradicted"
    ]
    labeled_lineages = {
        lineage_by_attempt[a.attempt_n] for a in contradicted_candidates if a.attempt_n in lineage_by_attempt
    }
    if len(contradicted_candidates) >= 2 and len(labeled_lineages) >= 2:
        return Decision(
            kind="CONTRADICTED", rule="contradicted_two_lineages", stop_reason=None, next_plan=None,
            champion_attempt_n=_champion_attempt_n(assessments),
        )

    # 3. INFEASIBLE — the caller passes probe-confirmed gaps only.
    if blocking_gap is not None:
        return Decision(
            kind="INFEASIBLE", rule="blocking_gap", stop_reason=f"infeasible:{blocking_gap}", next_plan=None,
            champion_attempt_n=_champion_attempt_n(assessments),
        )

    # 4. EXHAUSTED — sub-rules a-e, in order; first match wins.
    sorted_assessments = sorted(assessments, key=lambda a: a.attempt_n)

    budget_stop = _budget_floor_stop_reason(budget, spent, next_estimate)
    if budget_stop is not None:
        return Decision(
            kind="EXHAUSTED", rule="budget_floor", stop_reason=budget_stop, next_plan=None,
            champion_attempt_n=_champion_attempt_n(assessments),
        )

    if len(assessments) >= config.max_attempts:
        return Decision(
            kind="EXHAUSTED", rule="max_attempts", stop_reason="max_attempts", next_plan=None,
            champion_attempt_n=_champion_attempt_n(assessments),
        )

    if _plateaued(sorted_assessments, plateau_k=config.plateau_k):
        return Decision(
            kind="EXHAUSTED", rule="plateau", stop_reason="plateau", next_plan=None,
            champion_attempt_n=_champion_attempt_n(assessments),
        )

    if _infra_signature_repeated(sorted_assessments):
        return Decision(
            kind="EXHAUSTED", rule="infra_signature_repeated", stop_reason="infra_signature_repeated",
            next_plan=None, champion_attempt_n=_champion_attempt_n(assessments),
        )

    if _report_missing_twice(sorted_assessments):
        return Decision(
            kind="EXHAUSTED", rule="report_missing_twice", stop_reason="report_missing_twice", next_plan=None,
            champion_attempt_n=_champion_attempt_n(assessments),
        )

    # 5. CONTINUE — lineage + scope + width. The novelty gate (§8.4) is
    # checked at directive synthesis, not here.
    latest_overall = sorted_assessments[-1] if sorted_assessments else None
    latest_at_rung = (
        latest_overall
        if latest_overall is not None and scope_rung_by_attempt.get(latest_overall.attempt_n) == current_rung
        else None
    )
    first_arm = lineage_arms(assessments, runs_dir_hint=runs_dir_hint, lineage_by_attempt=lineage_by_attempt)[0]
    next_plan = NextAttemptPlan(
        lineage=first_arm.lineage,
        seed_attempt_n=first_arm.seed_attempt_n,
        seed_pointer=first_arm.seed_pointer,
        scope_rung=next_scope_rung(current_rung, latest_at_rung, ladder_len=config.ladder_len),
        width=width_for_next(assessments, config=config, budget=budget, spent=spent, next_estimate=next_estimate),
    )
    return Decision(kind="CONTINUE", rule="continue", stop_reason=None, next_plan=next_plan)
