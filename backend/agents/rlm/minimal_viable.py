"""
Minimal Viable Reproduction (MVR) -- OPENRESEARCH_MINIMAL_VIABLE, default OFF.

A standalone opt-in mode that composes with adapt/reference/execute: run the
paper's central claim at the SMALLEST scope (1 model x 1 env/dataset x 1 seed)
on a SHORT budget, and judge DIRECTIONAL VIABILITY -- does the mechanism show a
real learning signal, and does the headline metric move the paper's claimed
way -- WITHOUT requiring the exact reported number. Answers "is this paper
reproducible at all?" cheaply, before committing to a full reproduction.

Design mirrors no_learning_signal.py / zero_metrics_detection.py:
  - Flag default-OFF: ``OPENRESEARCH_MINIMAL_VIABLE``.
  - Evidence-not-grade (the project red line): the viability verdict is
    computed ONLY from the deterministic evidence layer (training curves,
    metrics.json, the honesty-guard flags already stamped into the report) --
    NEVER from the LLM rubric grade. Zero new LLM calls.
  - Fail-soft everywhere: any error degrades to "inconclusive", never raises.

Because the selected scope is 1x1x1, a run narrowed by :func:`select_viability_scope`
naturally stops after its single viability cell -- there is no full grid to gate.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from backend.agents.rlm.no_learning_signal import (
    _is_leaf,
    _leaf_no_learning,
    _loss_curve,
    _reward_curve,
    _walk_per_model,
)
from backend.agents.rlm.zero_metrics_detection import normalize_metric_values

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Additive threshold for "did the curve move at all in the claimed direction".
# Deliberately much smaller than no_learning_signal's proportional _TREND_EPS
# (0.05) -- MVR only needs to observe SOME directional movement over a short
# budget, not a strong learning signal (that stronger question is `learned`).
_DIRECTION_EPS: float = 1e-9

_BUCKET_ORDER: dict[str, int] = {"tiny": 0, "small": 1, "medium": 2, "large": 3}

# Scalar leaf keys checked (in priority order) for the "measured_metric" report
# field and to feed the has-evidence / directional fallbacks. Mirrors the
# higher-is-better family named in the module design.
_SCALAR_METRIC_PRIORITY: tuple[str, ...] = (
    "success_rate", "accuracy", "reward", "f1", "em", "score",
)

# Validator statuses that count as an explicit veto (external_validator.py /
# report.py's validation stamp: status in {"clean", "vetoed", "unavailable",
# "missing"} -- only "vetoed" is a confirmed honesty-guard hit).
_VALIDATOR_VETO_STATUSES: frozenset[str] = frozenset({"vetoed"})

# A narrow, high-confidence marker set for the "any run_warning-style
# fabrication flag reachable in the dict" scan -- deliberately NOT a loose
# substring search (that would false-positive on prose mentioning the term).
_FABRICATION_MARKERS: frozenset[str] = frozenset({"fabrication_suspected"})

_MAX_SCAN_DEPTH: int = 6


# ---------------------------------------------------------------------------
# 1. Feature flag
# ---------------------------------------------------------------------------

def minimal_viable_enabled() -> bool:
    """True iff ``OPENRESEARCH_MINIMAL_VIABLE`` is in {'1','true','yes','on'}."""
    return os.environ.get("OPENRESEARCH_MINIMAL_VIABLE", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


# ---------------------------------------------------------------------------
# 2. Scope selection
# ---------------------------------------------------------------------------

def _model_bucket_rank(model_key: str) -> int:
    from backend.services.runtime.feasibility_triage import _model_size_bucket
    return _BUCKET_ORDER.get(_model_size_bucket(model_key), _BUCKET_ORDER["medium"])


def _axis(scope: Any, attr: str) -> list[Any]:
    """A scope axis (``models``/``datasets``/``seeds``) as a plain list, or ``[]``
    when the scope is None / lacks the attr. Lets an attribute that RAISES on
    access (a hostile duck-typed object) propagate to the caller's fail-soft
    handler rather than being silently swallowed as empty here."""
    if scope is None:
        return []
    return list(getattr(scope, attr, None) or [])


def select_viability_scope(arxiv_id: str | None, operator_scope: Any) -> Any:
    """Reduce the EFFECTIVE reproduction scope to the smallest central-claim cell
    (smallest-model x first-dataset x first-seed) for MVR, or ``None`` when there
    is nothing to narrow. Fail-soft: any error returns ``None``.

    ``--minimal-viable`` is the operator's explicit opt-in to "reduce to the
    smallest cell", so this does NOT treat a pre-populated ``operator_scope`` as
    "hands off" (the CLI merges the paper hint's ``default_scope`` into the scope
    env even without ``--scope-spec``, so the operator scope arrives populated for
    exactly the hinted papers MVR most wants to shrink -- SDAR). Instead it builds
    the effective per-axis source (operator's value where present, else the paper
    hint's default) and collapses it:

      - ``not models_src`` (no operator models AND no hint models) -> ``None``: the
        run proceeds un-narrowed and the viability grade still works on whatever
        single-ish evidence lands. This is the paper-agnostic no-hint path.
      - already 1x1x1 (each axis <= 1) -> ``None``: a no-op that preserves the
        operator's exact single-cell spec (incl. per-dataset episode counts),
        rather than rebuilding it.
      - otherwise -> ``ScopeSpec(models=[smallest], datasets=[first]|[], seeds=[first])``.

    ``operator_scope`` is duck-typed (a ``ScopeSpec`` or ``None``) so a caller can
    pass whatever it already has on hand without importing the schema.
    """
    try:
        op_models = _axis(operator_scope, "models")
        op_datasets = _axis(operator_scope, "datasets")
        op_seeds = _axis(operator_scope, "seeds")

        from backend.agents.prompts.paper_hints import lookup_paper_hint
        hint = lookup_paper_hint(arxiv_id)
        hint_scope = getattr(hint, "default_scope", None) if hint is not None else None
        hint_models = _axis(hint_scope, "models")
        hint_datasets = _axis(hint_scope, "datasets")
        hint_seeds = _axis(hint_scope, "seeds")

        # Effective per-axis source: operator's value wins, else the hint default.
        models_src = op_models or hint_models
        datasets_src = op_datasets or hint_datasets
        seeds_src = op_seeds or hint_seeds

        if not models_src:
            return None  # nothing to rank "smallest" over -- grade works un-narrowed

        if len(models_src) <= 1 and len(datasets_src) <= 1 and len(seeds_src) <= 1:
            return None  # already minimal -- preserve the operator's exact 1x1x1 spec

        smallest_model = min(models_src, key=_model_bucket_rank)
        first_dataset = datasets_src[0] if datasets_src else None
        first_seed = seeds_src[0] if seeds_src else 0

        from backend.agents.schemas import ScopeSpec
        return ScopeSpec(
            models=[smallest_model],
            datasets=([first_dataset] if first_dataset is not None else []),
            seeds=[first_seed],
        )
    except Exception:  # noqa: BLE001 — fail-soft
        return None


# ---------------------------------------------------------------------------
# 3. Implementer guidance
# ---------------------------------------------------------------------------

def viability_guidance() -> str:
    """Guidance appended to OPENRESEARCH_BASELINE_EXTRA_GUIDANCE under MVR."""
    return (
        "MINIMAL VIABLE REPRODUCTION MODE: implement and train ONLY the single "
        "smallest configuration in scope -- one model x one environment/dataset "
        "x one seed. Run a SHORT budget: just enough steps or epochs to show "
        "the headline metric moving in the paper's claimed direction (a real "
        "learning signal), NOT to fully converge or match the paper's exact "
        "reported number. The goal is to cheaply demonstrate the paper's "
        "central mechanism is viable, not to reproduce it precisely -- stop "
        "once the metric shows directional movement."
    )


# ---------------------------------------------------------------------------
# Leaf collection (shared by directional / measured_metric)
# ---------------------------------------------------------------------------

def _leaves_from_metrics_blob(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return []
    per_model = data.get("per_model")
    if isinstance(per_model, dict):
        return _walk_per_model(per_model)
    if _is_leaf(data):
        return [data]
    return []


def _collect_leaves(code_dir: Path, report_dict: dict) -> list[dict[str, Any]]:
    """Leaves from ``code_dir/metrics.json`` first, falling back to
    ``report_dict['baseline_metrics']`` when the on-disk file is absent/empty."""
    leaves: list[dict[str, Any]] = []
    try:
        metrics_path = code_dir / "metrics.json"
        if metrics_path.exists():
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            leaves = _leaves_from_metrics_blob(data)
    except Exception:  # noqa: BLE001
        leaves = []
    if leaves:
        return leaves
    try:
        bm = report_dict.get("baseline_metrics") if isinstance(report_dict, dict) else None
        leaves = _leaves_from_metrics_blob(bm)
    except Exception:  # noqa: BLE001
        leaves = []
    return leaves


# ---------------------------------------------------------------------------
# Learning signal (flag-independent tri-state)
# ---------------------------------------------------------------------------

def _compute_learned(leaves: list[dict[str, Any]]) -> bool | None:
    """Flag-INDEPENDENT learning-signal tri-state over the run's leaves.

    ``True``  = at least one judgeable leaf shows learning;
    ``False`` = every judgeable leaf is flat;
    ``None``  = no judgeable curve exists to decide.

    Mirrors ``_leaf_no_learning`` semantics but is NOT gated on
    ``OPENRESEARCH_NO_LEARNING_SIGNAL_GATE`` -- MVR is standalone, so its
    learning signal must be faithful regardless of that sibling flag (whose
    ``detect_no_learning_signal`` short-circuits to "no signal" when off, which
    would spuriously read as "learned").
    """
    try:
        judged = [_leaf_no_learning(leaf) for leaf in leaves if isinstance(leaf, dict)]
        judged = [x for x in judged if x is not None]
        if not judged:
            return None
        return any(x is False for x in judged)  # any leaf shows learning
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Directional movement
# ---------------------------------------------------------------------------

def _compute_directional(leaves: list[dict[str, Any]], learned: bool) -> bool:
    """First judgeable leaf's reward/loss curve, first vs best value.

    higher-is-better (reward): directional iff best > first + eps.
    lower-is-better (loss):    directional iff best < first - eps.
    No usable curve on any leaf -> fall back to `learned`.
    """
    try:
        for leaf in leaves:
            if not isinstance(leaf, dict):
                continue
            reward = _reward_curve(leaf)
            if len(reward) >= 2:
                return max(reward) > reward[0] + _DIRECTION_EPS
            loss = _loss_curve(leaf)
            if len(loss) >= 2:
                return min(loss) < loss[0] - _DIRECTION_EPS
        return bool(learned)
    except Exception:  # noqa: BLE001
        return bool(learned)


# ---------------------------------------------------------------------------
# Evidence presence (reuses the zero-metrics floor's shape-normalizer, which is
# a pure predicate not gated on OPENRESEARCH_ZERO_METRICS_GUARD -- but NOT its
# "constant across cells" branch: an MVR run's narrow 1x1x1 scope means a
# single genuinely-flat leaf naturally has its curve points and final scalar
# all equal by construction, which is a different question ("did the metric
# move") from has_evidence's literal one ("is there a non-zero measurement").
# not_viable (flat, never-moving evidence) must stay reachable, not collapse
# into inconclusive because "flat" incidentally also reads as "constant".
# ---------------------------------------------------------------------------

def _compute_has_evidence(code_dir: Path) -> bool:
    try:
        metrics_path = code_dir / "metrics.json"
        if not metrics_path.exists():
            return False
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    values = normalize_metric_values(data)
    if not values:
        return False
    return any(v != 0.0 for v in values)


# ---------------------------------------------------------------------------
# Honesty-guard veto scan
# ---------------------------------------------------------------------------

def _scan_for_fabrication_marker(obj: Any, depth: int = 0) -> bool:
    if depth > _MAX_SCAN_DEPTH:
        return False
    try:
        if isinstance(obj, dict):
            for value in obj.values():
                if isinstance(value, str) and value.strip().lower() in _FABRICATION_MARKERS:
                    return True
                if isinstance(value, (dict, list)) and _scan_for_fabrication_marker(value, depth + 1):
                    return True
            return False
        if isinstance(obj, list):
            return any(_scan_for_fabrication_marker(item, depth + 1) for item in obj)
        return False
    except Exception:  # noqa: BLE001
        return False


def _guards_clean(report_dict: Any, has_evidence: bool) -> bool:
    """Conservative: unknown/error -> clean only if a measured value exists."""
    try:
        if not isinstance(report_dict, dict):
            return has_evidence

        replication_verdict = report_dict.get("replication_verdict")
        if isinstance(replication_verdict, str) and replication_verdict.strip().lower() == "inconclusive":
            return False

        if report_dict.get("evidence_gate_passed") is False:
            return False

        validation = report_dict.get("validation")
        if isinstance(validation, dict):
            status = str(validation.get("status") or "").strip().lower()
            if status in _VALIDATOR_VETO_STATUSES:
                return False

        if _scan_for_fabrication_marker(report_dict):
            return False

        return True
    except Exception:  # noqa: BLE001
        return has_evidence


# ---------------------------------------------------------------------------
# Auxiliary report fields
# ---------------------------------------------------------------------------

def _central_claim_from_scope(scope: Any) -> dict[str, Any] | None:
    try:
        if scope is None:
            return None
        models = getattr(scope, "models", None) or []
        datasets = getattr(scope, "datasets", None) or []
        seeds = getattr(scope, "seeds", None) or []
        if not models and not datasets:
            return None
        model = models[0] if models else None
        env = None
        if datasets:
            first = datasets[0]
            env = getattr(first, "name", None) if not isinstance(first, str) else first
        seed = seeds[0] if seeds else 0
        return {"model": model, "env": env, "seed": seed}
    except Exception:  # noqa: BLE001
        return None


def _first_named_metric(leaves: list[dict[str, Any]]) -> dict[str, Any] | None:
    try:
        for leaf in leaves:
            if not isinstance(leaf, dict):
                continue
            for key in _SCALAR_METRIC_PRIORITY:
                if key in leaf:
                    try:
                        return {"name": key, "value": float(leaf[key])}
                    except (TypeError, ValueError):
                        continue
            reward = _reward_curve(leaf)
            if reward:
                return {"name": "reward", "value": reward[-1]}
            loss = _loss_curve(leaf)
            if loss:
                return {"name": "loss", "value": loss[-1]}
        return None
    except Exception:  # noqa: BLE001
        return None


def _headline_reference(arxiv_id: str | None) -> dict[str, Any] | None:
    """Paper-declared headline value for the central metric, if the hint
    records one. PaperHint carries no structured headline-value field today
    (``guidance`` is free text) -- always None until such a field exists. Kept
    as a named seam so a future PAPER_HINTS extension needs no verdict-schema
    change.
    """
    return None


def _compute_gap(
    measured_metric: dict[str, Any] | None, headline_reference: dict[str, Any] | None,
) -> float | None:
    """measured - headline, only when both are known numeric values.

    ``headline_reference`` is always None today (no structured store exists
    yet -- see :func:`_headline_reference`), so this is currently always None
    too; kept as a real computation (not a hardcoded None) so a future
    PAPER_HINTS headline-value extension needs no change here.
    """
    try:
        if measured_metric is None or headline_reference is None:
            return None
        measured = measured_metric.get("value")
        headline = headline_reference.get("value")
        if isinstance(measured, (int, float)) and isinstance(headline, (int, float)):
            return float(measured) - float(headline)
        return None
    except Exception:  # noqa: BLE001
        return None


def _rationale_for(
    verdict: str, *, has_evidence: bool, learned_tri: bool | None,
    directional: bool, guards_clean: bool,
) -> str:
    if verdict == "viable":
        return "measured evidence shows the central mechanism learning and moving in the claimed direction"
    if verdict == "not_viable":
        if learned_tri is False:
            return "measured evidence exists but training shows no learning signal (reward/loss flat)"
        return "measured evidence exists but the central metric did not move in the paper's claimed direction"
    if not has_evidence:
        return "no measured, non-zero central metric found on disk"
    if not guards_clean:
        return "an honesty guard (fabrication/validator/no-learning-signal) vetoed the evidence"
    if learned_tri is None:
        return "measured evidence exists but no training curve was found to judge directional viability"
    return "viability could not be determined"


# ---------------------------------------------------------------------------
# 4. Core verdict
# ---------------------------------------------------------------------------

def compute_viability_verdict(
    code_dir: Any, *, arxiv_id: str | None, scope: Any, report_dict: dict,
) -> dict[str, Any]:
    """Deterministic directional-viability verdict for MVR. Never raises.

    Reads ONLY the on-disk evidence layer (``code_dir/metrics.json``) plus
    whatever honesty-guard flags are already present in ``report_dict`` --
    never an LLM grade. On any internal error, degrades to "inconclusive".
    """
    try:
        code_path = Path(code_dir)
        safe_report_dict: dict = report_dict if isinstance(report_dict, dict) else {}

        leaves = _collect_leaves(code_path, safe_report_dict)
        # Flag-independent tri-state (True learned / False flat / None no-curve).
        learned_tri = _compute_learned(leaves)
        learned = learned_tri is True  # the report field + the viable gate
        directional = _compute_directional(leaves, learned)

        has_evidence = _compute_has_evidence(code_path)
        guards_clean = _guards_clean(safe_report_dict, has_evidence)

        if not (has_evidence and guards_clean):
            verdict = "inconclusive"
        elif learned_tri is None:
            verdict = "inconclusive"  # no learning curve to judge viability
        elif learned_tri is True and directional:
            verdict = "viable"
        else:  # judged flat, or learned-but-not-directional
            verdict = "not_viable"

        evidence_refs: list[str] = []
        metrics_path = code_path / "metrics.json"
        if metrics_path.exists():
            evidence_refs.append(str(metrics_path))

        measured_metric = _first_named_metric(leaves)
        headline_reference = _headline_reference(arxiv_id)
        gap = _compute_gap(measured_metric, headline_reference)

        return {
            "verdict": verdict,
            "central_claim": _central_claim_from_scope(scope),
            "measured_metric": measured_metric,
            "headline_reference": headline_reference,
            "gap": gap,
            "learning_signal": bool(learned),
            "directional": bool(directional),
            "guards_clean": bool(guards_clean),
            "rationale": _rationale_for(
                verdict, has_evidence=has_evidence, learned_tri=learned_tri,
                directional=directional, guards_clean=guards_clean,
            ),
            "evidence_refs": evidence_refs,
        }
    except Exception:  # noqa: BLE001 — fail-soft, evidence-not-grade
        return {"verdict": "inconclusive", "rationale": "viability verdict computation error"}


__all__ = [
    "minimal_viable_enabled",
    "select_viability_scope",
    "viability_guidance",
    "compute_viability_verdict",
]
