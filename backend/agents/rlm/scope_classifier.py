"""Scope-adjusted rubric calculator — Lane P.

Principle (2026-05-25): the rubric should not be downgraded for failures
outside the agent's control. A reproduction that hit `HfUriError` on a
deprecated HF dataset short name OR ran out of VRAM AND couldn't scale
down further should not be scored as worse than one that genuinely
botched its own implementation. But a reproduction that hit a CUDA OOM
on iteration 1 and SUCCESSFULLY shrank the batch + completed on iter 2
should also not be penalised — the issue was resolvable and got resolved.

This module owns three things:

  * The **controllable taxonomy** — which failure_classifier classes are
    the agent's fault vs the environment's.
  * The **per-experiment classifier** — given a final status + the
    agent's iteration history, decides credit:
        ran             → full credit, included in denominator
        uncontrollable  → excluded from denominator (no penalty)
        preflight_caught_then_fixed → full credit (we caught it, agent fixed it)
        preflight_unresolved → half credit
        controllable    → 0 credit, included in denominator (penalty)
  * The **scope-adjusted rubric** — recomputes overall_score over the
    new denominator + applies a 50% coverage floor.

Both ``rubric.original`` (canonical, full denominator, skipped=0) and
``rubric.scope_adjusted`` (re-normalized) are emitted side-by-side so
the user can compare. The leaderboard sorts by scope_adjusted but the
original is one click away — honest about both.

Pure functions; no I/O; safe to call from verify_against_rubric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Literal


# ---------------------------------------------------------------------------
# Failure-class taxonomy
# ---------------------------------------------------------------------------


# Outside the agent's control. Reproduction skips the leaf; denominator
# shrinks so the score isn't dragged down by environment failures.
UNCONTROLLABLE_CLASSES: Final[frozenset[str]] = frozenset({
    "missing_dataset",          # HF URI deprecation, dataset 404
    "network_flake",            # transient download truncation
    "runpod_capacity",          # RunPod has no instances
    "runpod_transient_500",     # bare 500 from RunPod
    "runpod_ssh_timeout",       # pod created but never reachable
    "runpod_balance_too_low",   # billing
    "compute_too_large",        # paper used hardware we don't have
})


# Agent's responsibility. The leaf stays in the denominator and scores 0
# if the issue went unresolved.
CONTROLLABLE_CLASSES: Final[frozenset[str]] = frozenset({
    "torch_redundancy",
    "syntax_error",
    "scope_shape_violation",
    "contract_violation",
    "exec_timeout",             # agent's loop didn't fit budget
    "permission_denied",
    "watchdog_killed",          # agent's code wedged
    "unknown",                  # default: be strict
})


# Boundary cases — depend on whether the agent eventually resolved them.
# ``cuda_oom``: if the agent scaled batch down and completed, no penalty.
# ``preflight_blocked``: we caught the agent's bug. Half credit if
# unresolved, full credit if a subsequent iteration ran successfully.
BOUNDARY_CLASSES: Final[frozenset[str]] = frozenset({
    "cuda_oom",
    "preflight_blocked",
})


# Credit weights for the four effective-status outcomes.
_CREDIT: Final[dict[str, float]] = {
    "ran":                    1.0,
    "uncontrollable_skip":    0.0,  # excluded from denominator
    "preflight_caught_fixed": 1.0,  # we caught it, agent fixed it — full credit
    "preflight_unresolved":   0.5,  # half credit
    "controllable_fail":      0.0,  # in denominator, scores 0
}


# ---------------------------------------------------------------------------
# Per-experiment classifier
# ---------------------------------------------------------------------------


@dataclass
class ExperimentJudgement:
    """The result of classifying one experiment for rubric weighting."""
    effective_status: Literal[
        "ran",
        "uncontrollable_skip",
        "preflight_caught_fixed",
        "preflight_unresolved",
        "controllable_fail",
    ]
    credit: float
    in_denominator: bool
    reason_class: str
    notes: str = ""


def classify_experiment(
    *,
    final_status: str,
    history: list[dict[str, Any]] | None = None,
    final_reason_class: str | None = None,
) -> ExperimentJudgement:
    """Map an experiment's final state + iteration history to a credit weight.

    Inputs:
      * ``final_status`` — the agent's emitted status string in
        ``metrics.json.experiments.<name>.status`` (e.g. "ok",
        "data_unavailable", "cuda_oom", "preflight_blocked",
        "scope_reduced", "code_error", "unknown").
      * ``history`` — optional list of prior iteration entries for the
        same experiment, in order. Each entry is at minimum
        ``{"status": <str>, "reason_class": <str>}``; richer entries
        (batch size, scope_reduction notes) are tolerated and improve
        the boundary classification.
      * ``final_reason_class`` — explicit override if the agent didn't
        roll the failure_classifier output into the experiment dict.

    Returns ``ExperimentJudgement``. Pure function; safe to call from any
    thread / process.
    """
    history = history or []

    # Status string canonicalisation.
    fs = (final_status or "").strip().lower()
    rc = (final_reason_class or "").strip().lower()

    # 1. Happy path — experiment ran successfully, regardless of prior history.
    #    If iter 1 was cuda_oom and iter 2 ran with smaller batch, that's a
    #    win — agent scaled appropriately. No penalty.
    if fs == "ok":
        notes = ""
        if any((h.get("status") or "").lower() != "ok" for h in history):
            notes = "ran after recovery from earlier iteration failure"
        return ExperimentJudgement(
            effective_status="ran",
            credit=_CREDIT["ran"],
            in_denominator=True,
            reason_class="ok",
            notes=notes,
        )

    # 2. Uncontrollable terminal — never penalise.
    if rc in UNCONTROLLABLE_CLASSES or fs in {
        "data_unavailable", "scope_reduced", "compute_too_large",
    }:
        notes = f"excluded from denominator: {rc or fs} is outside agent control"
        return ExperimentJudgement(
            effective_status="uncontrollable_skip",
            credit=_CREDIT["uncontrollable_skip"],
            in_denominator=False,
            reason_class=rc or fs,
            notes=notes,
        )

    # 3. Pre-flight boundary cases.
    if fs == "preflight_blocked" or rc == "preflight_blocked":
        # If any iteration ran successfully, give full credit.
        if any((h.get("status") or "").lower() == "ok" for h in history):
            return ExperimentJudgement(
                effective_status="preflight_caught_fixed",
                credit=_CREDIT["preflight_caught_fixed"],
                in_denominator=True,
                reason_class="preflight_blocked",
                notes="pre-flight caught the bug; agent's later iteration ran clean",
            )
        # Otherwise: half credit.
        return ExperimentJudgement(
            effective_status="preflight_unresolved",
            credit=_CREDIT["preflight_unresolved"],
            in_denominator=True,
            reason_class="preflight_blocked",
            notes="pre-flight caught the bug; agent didn't resolve in remaining iterations",
        )

    # 4. CUDA OOM boundary case.
    if rc == "cuda_oom" or fs == "cuda_oom":
        # If the agent ever attempted a scope reduction (smaller batch, etc.)
        # AND the experiment STILL OOM'd, that's uncontrollable — we tried,
        # the hardware can't fit it.
        attempted_reduction = any(
            h.get("scope_reduction") or h.get("batch_reduced") for h in history
        )
        if attempted_reduction:
            return ExperimentJudgement(
                effective_status="uncontrollable_skip",
                credit=_CREDIT["uncontrollable_skip"],
                in_denominator=False,
                reason_class="cuda_oom",
                notes="agent attempted batch reduction but the workload still doesn't fit",
            )
        # Otherwise the agent didn't scale down — controllable.
        return ExperimentJudgement(
            effective_status="controllable_fail",
            credit=_CREDIT["controllable_fail"],
            in_denominator=True,
            reason_class="cuda_oom",
            notes="cuda_oom without scope-reduction attempt — agent should have shrunk batch",
        )

    # 5. Default — controllable failure (be strict; agent should fix).
    return ExperimentJudgement(
        effective_status="controllable_fail",
        credit=_CREDIT["controllable_fail"],
        in_denominator=True,
        reason_class=rc or fs or "unknown",
        notes="controllable failure (agent's code or contract violation)",
    )


# ---------------------------------------------------------------------------
# Scope-adjusted rubric calculator
# ---------------------------------------------------------------------------


# Coverage floor — below this fraction of experiments ran, the
# scope-adjusted rubric is flagged as "insufficient_coverage" and the
# overall_score is capped at this value to prevent 1-of-5 partial
# reproductions from claiming high scores.
_MIN_COVERAGE_FLOOR: Final[float] = 0.50
_INSUFFICIENT_COVERAGE_CAP: Final[float] = 0.30


@dataclass
class ScopeAdjustedRubric:
    """Output of ``compute_scope_adjusted_rubric``."""
    overall_score: float
    target_score: float
    meets_target: bool
    coverage: float
    insufficient_coverage: bool
    judgements: list[ExperimentJudgement] = field(default_factory=list)
    notes: str = ""


def compute_scope_adjusted_rubric(
    *,
    experiments: dict[str, dict[str, Any]],
    leaf_scores: dict[str, dict[str, Any]],
    target_score: float,
) -> ScopeAdjustedRubric:
    """Compute the scope-adjusted rubric.

    Inputs:
      * ``experiments`` — agent-emitted ``metrics.json["experiments"]``
        dict. Each entry has at least ``{"status": <str>}``; richer
        entries are tolerated.
      * ``leaf_scores`` — ``{leaf_name: {"score": float in [0,1],
        "weight": float, "experiment": <name>}}``. The ``experiment``
        field maps the leaf back to one of the keys in ``experiments``
        (so the leaf knows whose status governs its credit). Leaves
        without an experiment field are always counted (treated as
        global / paper-wide).
      * ``target_score`` — the run's target threshold.

    Output: ``ScopeAdjustedRubric`` carrying the renormalized score, the
    coverage fraction, and per-experiment judgements for the report.
    """
    judgements: list[ExperimentJudgement] = []
    name_to_judgement: dict[str, ExperimentJudgement] = {}
    for name, ex in (experiments or {}).items():
        if not isinstance(ex, dict):
            continue
        j = classify_experiment(
            final_status=str(ex.get("status") or ""),
            history=ex.get("history") or [],
            final_reason_class=str(ex.get("reason_class") or ""),
        )
        judgements.append(j)
        name_to_judgement[name] = j

    # Coverage = fraction of experiments that ran successfully (status=ok).
    total = len(judgements)
    ran = sum(1 for j in judgements if j.effective_status == "ran"
              or j.effective_status == "preflight_caught_fixed")
    coverage = (ran / total) if total > 0 else 1.0

    # Score numerator + denominator over leaves whose experiment is
    # in_denominator. Leaves without an experiment mapping always count.
    num = 0.0
    den = 0.0
    for leaf_name, leaf in (leaf_scores or {}).items():
        if not isinstance(leaf, dict):
            continue
        weight = float(leaf.get("weight", 1.0) or 0.0)
        score = float(leaf.get("score", 0.0) or 0.0)
        exp_name = leaf.get("experiment")
        # Decide credit + inclusion.
        credit = 1.0  # default: count fully
        included = True
        if isinstance(exp_name, str) and exp_name in name_to_judgement:
            j = name_to_judgement[exp_name]
            included = j.in_denominator
            credit = j.credit
        if not included:
            continue
        num += weight * score * credit
        den += weight

    raw_score = (num / den) if den > 0 else 0.0

    # Apply coverage floor.
    insufficient = total > 0 and coverage < _MIN_COVERAGE_FLOOR
    if insufficient:
        capped = min(raw_score, _INSUFFICIENT_COVERAGE_CAP)
        notes = (
            f"insufficient_coverage: only {ran}/{total} experiments ran "
            f"(<{int(_MIN_COVERAGE_FLOOR*100)}% floor); score capped at "
            f"{_INSUFFICIENT_COVERAGE_CAP:.2f}"
        )
    else:
        capped = raw_score
        notes = ""

    return ScopeAdjustedRubric(
        overall_score=round(capped, 4),
        target_score=target_score,
        meets_target=capped >= target_score,
        coverage=round(coverage, 4),
        insufficient_coverage=insufficient,
        judgements=judgements,
        notes=notes,
    )


__all__ = [
    "CONTROLLABLE_CLASSES",
    "ExperimentJudgement",
    "ScopeAdjustedRubric",
    "UNCONTROLLABLE_CLASSES",
    "classify_experiment",
    "compute_scope_adjusted_rubric",
]
