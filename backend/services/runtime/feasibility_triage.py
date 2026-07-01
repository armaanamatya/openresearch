"""Deterministic pre-GPU-lease feasibility gate: cost estimate + 3-axis triage (Phase 1b).

Pure and UNWIRED: no live code path calls this yet (a later Phase-1c
``ReproductionRun`` will). No network, no LLM, no real GPU --
``est_train_seconds``/``estimate_scope_cost`` are a conservative (over-estimate,
never under), no-LLM, per-scope-cell cost model; the live ``WATCH``-time budget
check remains the true backstop, so this need only be directionally correct.
``FeasibilityTriage.triage`` combines a (injected, fakeable) data-reachability
probe with the cost model into one of three decisions: ``PROCEED`` |
``DOWN_SCOPE`` | ``PLAN_ONLY``. Fail-soft throughout: bad input never raises
into a caller, it degrades to the most conservative decision (``PLAN_ONLY``).

DRY note: this deliberately does NOT reuse/duplicate the async LLM-driven
``backend.services.pricing.estimator.estimate_paper_budget`` -- that is a
one-shot, paper-level, LLM-backed guess. This module is a synchronous,
deterministic, per-scope-cell estimate purpose-built for the pre-lease gate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from backend.agents.resilience.budget import RunBudget
    from backend.agents.schemas import ScopeSpec
    from backend.services.runtime.gpu_catalog import GpuSku
    from backend.services.runtime.run_plan import RequiredAsset, RunPlan

# ---------------------------------------------------------------------------
# Cost model (Task 10)
# ---------------------------------------------------------------------------

# A "<n>b" parameter-count hint, e.g. "qwen3-1.7b" -> "1.7", "qwen2.5-7b" -> "7".
# The version-number prefix ("2.5", "3") is not immediately followed by a
# literal "b", so it never matches -- only the trailing size suffix does.
_SIZE_HINT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b\b", re.IGNORECASE)

# Conservative (over-estimate, never under) seconds/step per size bucket.
_SECONDS_PER_STEP = {"tiny": 0.5, "small": 1.5, "medium": 3.0, "large": 6.0}

# Ordinal for "cheapest bucket" comparisons (used by the scope trimmer below).
_BUCKET_ORDER = {"tiny": 0, "small": 1, "medium": 2, "large": 3}

# Default step count the triage cost model estimates against (a representative
# short-run probe, not the paper's actual full training length).
_DEFAULT_TRIAGE_STEPS = 400


def _model_size_bucket(model_key: str) -> str:
    """Parse a "<n>b" parameter-count hint out of a model key into a size bucket.

    Conservative: an unparseable/unknown key buckets as "medium" -- never
    silently assume the cheapest bucket for a model we can't identify.
    """
    match = _SIZE_HINT_RE.search(model_key or "")
    if not match:
        return "medium"
    try:
        billions = float(match.group(1))
    except ValueError:
        return "medium"
    if billions < 1.0:
        return "tiny"
    if billions < 4.0:
        return "small"
    if billions < 13.0:
        return "medium"
    return "large"


def est_train_seconds(model_key: str, steps: int) -> float:
    """Conservative, deterministic wall-clock estimate (seconds) for one training cell.

    No network, no LLM: a small explicit seconds/step table keyed by a parsed
    model-size bucket. Always > 0 -- even for an unparseable model key -- so a
    caller can never fold a false zero into a cost total.
    """
    bucket = _model_size_bucket(model_key)
    rate = _SECONDS_PER_STEP.get(bucket, _SECONDS_PER_STEP["medium"])
    return max(1, int(steps)) * rate


def estimate_scope_cost(
    scope: "ScopeSpec", sku: "GpuSku", *, steps: int, overhead: float = 2.0
) -> tuple[float, float]:
    """Sum ``est_train_seconds`` over every (model, dataset, seed) cell in ``scope``.

    Returns ``(gpu_hours, usd)``. ``overhead`` (default 2.0x) is a conservative
    multiplier covering setup/eval/retry time the raw training seconds don't
    capture. An empty scope (no models or no datasets) costs nothing:
    ``(0.0, 0.0)``.
    """
    models = scope.models or []
    datasets = scope.dataset_ids() if scope.datasets else []
    seeds = scope.seeds or [0]
    if not models or not datasets:
        return (0.0, 0.0)
    total_s = 0.0
    for model_key in models:
        total_s += est_train_seconds(model_key, steps) * len(datasets) * len(seeds)
    gpu_hours = round(total_s * overhead / 3600.0, 4)
    usd = round(gpu_hours * float(sku.approx_usd_per_hr), 4)
    return (gpu_hours, usd)


# ---------------------------------------------------------------------------
# 3-axis triage (Task 11)
# ---------------------------------------------------------------------------


def _within_budget(budget: "RunBudget | None", gpu_hours: float, usd: float) -> bool:
    """Mirror RunBudget's own disable-guard idiom: unset/<=0 caps never block."""
    if budget is None:
        return True
    gh_ok = not budget.max_gpu_hours or gpu_hours <= budget.max_gpu_hours
    usd_ok = not budget.max_run_gpu_usd or usd <= budget.max_run_gpu_usd
    return gh_ok and usd_ok


def _trim_scope(
    scope: "ScopeSpec", sku: "GpuSku", budget: "RunBudget | None", steps: int
) -> "ScopeSpec | None":
    """Return the first (cheapest-first) narrowed non-empty scope that fits ``budget``.

    Tries the single smallest model (by ``_model_size_bucket``) + one seed with
    datasets left unchanged first; if that still doesn't fit, further narrows
    to a single dataset. Returns None when nothing in this small candidate set
    fits -- the caller reports PLAN_ONLY in that case.
    """
    if scope is None or not scope.models:
        return None
    smallest = min(
        scope.models,
        key=lambda m: _BUCKET_ORDER.get(_model_size_bucket(m), _BUCKET_ORDER["medium"]),
    )
    seed = scope.seeds[0] if scope.seeds else 0
    candidates = [scope.model_copy(update={"models": [smallest], "seeds": [seed]})]
    if scope.datasets and len(scope.datasets) > 1:
        candidates.append(
            scope.model_copy(
                update={"models": [smallest], "seeds": [seed], "datasets": scope.datasets[:1]}
            )
        )
    for candidate in candidates:
        gh, usd = estimate_scope_cost(candidate, sku, steps=steps)
        if _within_budget(budget, gh, usd):
            return candidate
    return None


@dataclass(frozen=True)
class TriageDecision:
    """The outcome of :meth:`FeasibilityTriage.triage`."""

    decision: str  # "PROCEED" | "DOWN_SCOPE" | "PLAN_ONLY"
    scope: "ScopeSpec | None"
    reasons: tuple[str, ...]
    est_gpu_hours: float
    est_usd: float


class FeasibilityTriage:
    """Deterministic pre-GPU-lease gate combining data reachability + compute cost.

    ``reachability_probe`` (``RequiredAsset -> "reachable"|"gated"|"missing"``)
    is injected/fakeable -- the real HEAD/HF-metadata check + CredentialBroker
    land in a later phase (1d). The default here makes no network call and
    always reports "reachable", so an un-configured triage never invents a
    false blocker. ``adapters`` is accepted for forward-compatibility with the
    (later-phase) environment-standability axis; in 1b every dataset is treated
    as generically resolvable, so it does not currently affect the decision.
    """

    def __init__(
        self,
        *,
        reachability_probe: "Callable[[RequiredAsset], str] | None" = None,
        adapters: object = None,
    ) -> None:
        self._reachability_probe = reachability_probe or (lambda _asset: "reachable")
        self._adapters = adapters

    def _reachability_blockers(self, required_assets: object, reasons: list[str]) -> bool:
        """Probe each dataset/weights asset; a gated or missing asset is a hard blocker."""
        blocked = False
        for asset in required_assets or ():
            if asset.kind not in ("dataset", "weights"):
                continue
            status = self._reachability_probe(asset)
            if status in ("gated", "missing"):
                blocked = True
                reasons.append(f"required {asset.kind} '{asset.identifier}' is {status}")
        return blocked

    def triage(self, plan: "RunPlan", sku: "GpuSku") -> TriageDecision:
        """Combine reachability + compute-cost into PROCEED / DOWN_SCOPE / PLAN_ONLY.

        Fail-soft: any internal error degrades to ``PLAN_ONLY`` with reason
        ``"triage_error"`` rather than raising into the caller.
        """
        try:
            scope = plan.scope
            reasons: list[str] = []

            blocked = self._reachability_blockers(plan.required_assets, reasons)

            if scope is not None:
                gh, usd = estimate_scope_cost(scope, sku, steps=_DEFAULT_TRIAGE_STEPS)
            else:
                gh, usd = (0.0, 0.0)

            if blocked:
                return TriageDecision(
                    decision="PLAN_ONLY",
                    scope=scope,
                    reasons=tuple(reasons),
                    est_gpu_hours=gh,
                    est_usd=usd,
                )

            if _within_budget(plan.budget, gh, usd):
                reasons.append(f"scope fits budget ({gh:.4f} gpu-h, ${usd:.4f})")
                return TriageDecision(
                    decision="PROCEED",
                    scope=scope,
                    reasons=tuple(reasons),
                    est_gpu_hours=gh,
                    est_usd=usd,
                )

            trimmed = _trim_scope(scope, sku, plan.budget, _DEFAULT_TRIAGE_STEPS)
            if trimmed is not None:
                t_gh, t_usd = estimate_scope_cost(trimmed, sku, steps=_DEFAULT_TRIAGE_STEPS)
                reasons.append(f"scope trimmed to fit budget ({gh:.4f} -> {t_gh:.4f} gpu-h)")
                return TriageDecision(
                    decision="DOWN_SCOPE",
                    scope=trimmed,
                    reasons=tuple(reasons),
                    est_gpu_hours=t_gh,
                    est_usd=t_usd,
                )

            reasons.append(f"even the minimal scope exceeds budget ({gh:.4f} gpu-h, ${usd:.4f})")
            return TriageDecision(
                decision="PLAN_ONLY",
                scope=scope,
                reasons=tuple(reasons),
                est_gpu_hours=gh,
                est_usd=usd,
            )
        except Exception:
            return TriageDecision(
                decision="PLAN_ONLY",
                scope=getattr(plan, "scope", None),
                reasons=("triage_error",),
                est_gpu_hours=0.0,
                est_usd=0.0,
            )


__all__ = ["est_train_seconds", "estimate_scope_cost", "TriageDecision", "FeasibilityTriage"]
