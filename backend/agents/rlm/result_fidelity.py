"""result_fidelity.py — the deterministic per-claim result checker (Track A, closes §4.2).

``metric_binding.bind_claims`` (Task 1) resolves each claim's prose ``metric_name``
to a scope-verified ``metric_key``/``model_key``/``env_key``/``baseline_key`` path
into ``code/metrics.json``, but a resolved path is not yet a verdict — nothing
compares the measured number to what the paper actually claimed. This module is
that comparison: ``evaluate(repro_spec, run_dir) -> ResultFidelity`` reuses the
bind, reads the measured value via ``repro_spec_extractor.seed_bundle_from_metrics``,
and applies one of four small, deterministic, per-``kind`` tests.

Design (locked — see
``docs/superpowers/specs/2026-07-09-eval-integrity-track-a-design.md`` §4.2):

* **numeric** — ``abs(measured - target) <= equivalence_margin`` where
  ``target`` is the claim's ``claimed_effect`` taken as a direct absolute value
  (e.g. "accuracy should be ~0.99").
* **relative** — the claim's ``claimed_effect`` is an *advantage* over a
  ``baseline_value`` (paper convention: positive == the proposed method wins,
  regardless of metric direction — folded into the sign exactly like
  ``repro_spec_extractor.parse_claim_statement``'s step-4 sign fold, reused
  here verbatim via ``_fold_sign``, never re-derived). The measured advantage
  is ``fold(measured - baseline_value, direction)``; a claim missing a numeric
  ``baseline_value`` cannot be measured this way and stays ``unmeasured``
  (conservative — never invents a comparator). Passes when the folded effect
  has the same sign as the claim (**ordering**) *and* sits within
  ``equivalence_margin`` of ``claimed_effect`` (**ratio/effect**) — both
  conditions named explicitly in the design, not just one.
* **trend** — reads the per-claim training curve from ``metrics["history"]``
  (the shape already established by ``convergence_evidence.py``:
  ``{"<model_key>": {"<metric_key>": [v0, v1, ...]}}``, flat
  ``{"<metric_key>": [...]}`` fallback for single-model runs), folds
  ``curve[-1] - curve[0]`` the same way as ``relative``, and applies the same
  ordering+magnitude test against ``claimed_effect``. Needs >=2 points; fewer
  is an honest ``unmeasured``, never a guess.
* **qualitative** / **ambiguous** / **unbound** (``metric_binding.bound`` is
  False) -> always ``unmeasured``, checked *before* any per-kind test runs —
  never auto-pass, never a false ``fail``. This is the asymmetry the brief
  calls out: a ``fail`` can only be reached through a scope-verified bind
  (``metric_binding.bound is True``); an ambiguous claim or an unresolved bind
  can reach ``fail`` under no code path in this module.

Each ``per_claim`` result additionally carries ``is_primary``/``ambiguous``
(passed through from the input claim) alongside the brief's literal
``{claim_id, kind, status, measured, target, margin, reason}`` shape — the
downstream ``VerdictAuthority.decide`` (Task 3) taxonomy is defined entirely in
terms of *primary* claims, so the per-claim payload needs to say which claims
those are; the two extra keys are additive and do not change any of the
brief's four required fields.

Deferred (YAGNI): this module deliberately does NOT reimplement the richer
seed-bundle confidence-interval grading already in ``reproducibility_verdict.py``
(``_grade_claim`` — multi-seed CI vs. an equivalence-region-around-zero test).
Per the design doc, this module *supersedes* that engine's D2-blind seed-only
path by feeding it a single, real, per-claim measured result; the two live
side by side, not merged. No LLM-proposed fallback measurement is built here
either — an unresolved bind stays honestly ``unmeasured``.

Pure, stdlib-only, no LLM/network/flag check (mirrors ``metric_binding.py`` —
flag-gating, if any, belongs to the caller). Never raises: any malformed
``repro_spec``/``run_dir``/claim degrades to ``unmeasured``/an empty result,
never an exception.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from backend.agents.rlm.metric_binding import bind_claims
from backend.agents.rlm.repro_spec_extractor import seed_bundle_from_metrics

# {"per_claim": [...], "result_fidelity_score": float,
#  "primary_all_measured": bool, "any_contradicted": bool}
ResultFidelity = dict[str, Any]

# Secondary (non-primary) claims count toward ``result_fidelity_score`` at this
# fraction of a primary claim's weight — the score is "fraction of PRIMARY
# claims that pass", with secondaries folded in at reduced influence rather
# than ignored outright. Tunable; documented rather than derived from data.
_SECONDARY_WEIGHT = 0.5

# A trend needs at least this many logged points to have a defensible slope.
_MIN_HISTORY_POINTS = 2

_EMPTY_RESULT_FIDELITY: ResultFidelity = {
    "per_claim": [],
    "result_fidelity_score": 0.0,
    "primary_all_measured": False,
    "any_contradicted": False,
}


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------


def _is_finite_number(value: Any) -> bool:
    """True iff ``value`` is a finite, non-bool int/float."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Best-effort coercion to a finite float; degrades to ``default``."""
    try:
        f = float(value)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _finite_floats(seq: Any) -> list[float]:
    """Coerce a raw value to a list of finite floats. Fail-soft -> ``[]``."""
    if not isinstance(seq, (list, tuple)):
        return []
    return [float(v) for v in seq if _is_finite_number(v)]


def _mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _same_sign(measured_effect: float, target_effect: float) -> bool:
    """True iff both sit on the same side of zero (a claimed null-effect needs
    an exact-zero match — the "ordering" half of the relative/trend test)."""
    if target_effect > 0:
        return measured_effect > 0
    if target_effect < 0:
        return measured_effect < 0
    return measured_effect == 0


def _fold_sign(raw_delta: float, direction: str) -> float:
    """Fold a raw ``(proposed - baseline)`` delta into the sign convention a
    positive ``claimed_effect`` always uses: positive == the proposed method's
    advantage, regardless of metric direction.

    Mirrors ``repro_spec_extractor.parse_claim_statement``'s step-4 sign fold
    (a lower-is-better metric's *decrease* is the advantage) — same two-branch
    transform, reused rather than re-derived per the brief.
    """
    return -raw_delta if direction == "lower_is_better" else raw_delta


def _read_history_curve(metrics: dict[str, Any], metric_key: str, model_key: str) -> list[float]:
    """Best-effort read of a per-claim training curve from ``metrics["history"]``.

    Shape mirrors ``convergence_evidence.derive_convergence_metrics``'s
    documented convention: ``{"<model_key>": {"<metric_key>": [v0, v1, ...],
    ...}, ...}``, with a flat ``{"<metric_key>": [...]}`` fallback for
    single-model runs that never nest under a model key. Never raises.
    """
    hist = metrics.get("history")
    if not isinstance(hist, dict):
        return []
    if model_key:
        node = hist.get(model_key)
        if isinstance(node, dict):
            curve = _finite_floats(node.get(metric_key))
            if curve:
                return curve
    return _finite_floats(hist.get(metric_key))


def _load_metrics(run_dir: Path) -> dict[str, Any]:
    """Best-effort parse of ``<run_dir>/code/metrics.json``. Never raises."""
    try:
        data = json.loads((run_dir / "code" / "metrics.json").read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------------------
# Per-claim result builders
# ---------------------------------------------------------------------------


def _unmeasured(
    claim_id: str, kind: str, target: float, margin: float, reason: str
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "kind": kind,
        "status": "unmeasured",
        "measured": None,
        "target": target,
        "margin": margin,
        "reason": reason,
    }


def _decided(
    claim_id: str,
    kind: str,
    status: str,
    measured: float,
    target: float,
    margin: float,
    reason: str,
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "kind": kind,
        "status": status,
        "measured": measured,
        "target": target,
        "margin": margin,
        "reason": reason,
    }


def _ordering_and_magnitude(effect: float, target: float, margin: float) -> bool:
    return _same_sign(effect, target) and abs(effect - target) <= margin


def _evaluate_claim(claim: dict[str, Any], metrics: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Apply the per-kind deterministic test to one already-bound claim."""
    claim_id = str(claim.get("claim_id") or "")
    kind = str(claim.get("kind") or "")
    target = _safe_float(claim.get("claimed_effect"), 0.0)
    margin = max(0.0, _safe_float(claim.get("equivalence_margin"), 0.0))
    direction = str(claim.get("direction") or "higher_is_better")

    # Ambiguous claims are NEVER auto-passed/failed — the extractor's own
    # honest escape hatch always wins, checked before any bind/measurement is
    # even attempted.
    if bool(claim.get("ambiguous", False)):
        return _unmeasured(claim_id, kind, target, margin, "ambiguous_claim")

    if kind == "qualitative":
        return _unmeasured(claim_id, kind, target, margin, "qualitative_claim_not_measurable")

    binding = claim.get("metric_binding")
    if not isinstance(binding, dict) or not binding.get("bound"):
        reason = str(binding.get("reason")) if isinstance(binding, dict) and binding.get("reason") else "unbound"
        return _unmeasured(claim_id, kind, target, margin, reason)

    metric_key = str(binding.get("metric_key") or "")
    model_key = str(binding.get("model_key") or "")
    env_key = str(binding.get("env_key") or "")
    baseline_key = str(binding.get("baseline_key") or "")

    if kind == "trend":
        curve = _read_history_curve(metrics, metric_key, model_key)
        if len(curve) < _MIN_HISTORY_POINTS:
            return _unmeasured(claim_id, kind, target, margin, "insufficient_history_points")
        effect = _fold_sign(curve[-1] - curve[0], direction)
        status = "pass" if _ordering_and_magnitude(effect, target, margin) else "fail"
        reason = "" if status == "pass" else "trend_direction_or_magnitude_mismatch"
        return _decided(claim_id, kind, status, effect, target, margin, reason)

    # numeric / relative both read a scalar measurement at the bound path.
    bundle = seed_bundle_from_metrics(
        run_dir,
        metric_key=metric_key,
        model_key=model_key,
        env_key=env_key,
        baseline_key=baseline_key,
    )
    measured = _mean(_finite_floats(bundle.get("per_seed_effect")))
    if measured is None:
        return _unmeasured(claim_id, kind, target, margin, "no_measured_value")

    if kind == "numeric":
        status = "pass" if abs(measured - target) <= margin else "fail"
        reason = "" if status == "pass" else "outside_equivalence_margin"
        return _decided(claim_id, kind, status, measured, target, margin, reason)

    if kind == "relative":
        baseline_value = claim.get("baseline_value")
        if not _is_finite_number(baseline_value):
            return _unmeasured(claim_id, kind, target, margin, "missing_baseline_value")
        effect = _fold_sign(measured - float(baseline_value), direction)
        status = "pass" if _ordering_and_magnitude(effect, target, margin) else "fail"
        reason = "" if status == "pass" else "relative_ordering_or_magnitude_mismatch"
        return _decided(claim_id, kind, status, effect, target, margin, reason)

    return _unmeasured(claim_id, kind, target, margin, f"unknown_kind:{kind}" if kind else "missing_kind")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def evaluate(repro_spec: dict[str, Any], run_dir: Path | str) -> ResultFidelity:
    """Deterministically grade every claim in ``repro_spec`` against ``run_dir``.

    Pure, stdlib-only, no LLM/network call. Never raises: any malformed input
    (bad ``repro_spec`` shape, missing ``run_dir``/``code/metrics.json``,
    malformed individual claims) degrades to ``unmeasured``/an empty result.

    Steps: parse ``<run_dir>/code/metrics.json``; call ``bind_claims`` (Task 1)
    to attach a scope-verified ``metric_binding`` to each claim; apply the
    per-kind deterministic test (see module docstring); aggregate.
    """
    try:
        return _evaluate_inner(repro_spec, Path(run_dir))
    except Exception:  # noqa: BLE001 — the contract is "never raise".
        return dict(_EMPTY_RESULT_FIDELITY)


def _evaluate_inner(repro_spec: dict[str, Any], run_dir: Path) -> ResultFidelity:
    if not isinstance(repro_spec, dict):
        return dict(_EMPTY_RESULT_FIDELITY)
    claims = repro_spec.get("claims")
    if not isinstance(claims, list):
        return dict(_EMPTY_RESULT_FIDELITY)

    metrics = _load_metrics(run_dir)

    try:
        bound_spec = bind_claims(repro_spec, metrics)
        bound_claims = bound_spec.get("claims") if isinstance(bound_spec, dict) else None
        if not isinstance(bound_claims, list):
            bound_claims = claims
    except Exception:  # noqa: BLE001 — a bind failure never blocks grading.
        bound_claims = claims

    per_claim: list[dict[str, Any]] = []
    for claim in bound_claims:
        if not isinstance(claim, dict):
            continue
        try:
            result = _evaluate_claim(claim, metrics, run_dir)
        except Exception as exc:  # noqa: BLE001 — one bad claim never aborts the batch.
            result = _unmeasured(
                str(claim.get("claim_id") or ""),
                str(claim.get("kind") or ""),
                0.0,
                0.0,
                f"eval_error:{type(exc).__name__}",
            )
        result["is_primary"] = bool(claim.get("is_primary", False))
        result["ambiguous"] = bool(claim.get("ambiguous", False))
        per_claim.append(result)

    primary = [r for r in per_claim if r["is_primary"]]
    secondary = [r for r in per_claim if not r["is_primary"]]
    total_weight = len(primary) * 1.0 + len(secondary) * _SECONDARY_WEIGHT
    pass_weight = sum(1.0 for r in primary if r["status"] == "pass") + sum(
        _SECONDARY_WEIGHT for r in secondary if r["status"] == "pass"
    )
    score = (pass_weight / total_weight) if total_weight else 0.0

    # Vacuously False (not True) when there are no primary claims at all — "no
    # measurable primary" should never read as "all measured".
    primary_all_measured = bool(primary) and all(r["status"] != "unmeasured" for r in primary)
    any_contradicted = any(r["status"] == "fail" for r in per_claim)

    return {
        "per_claim": per_claim,
        "result_fidelity_score": score,
        "primary_all_measured": primary_all_measured,
        "any_contradicted": any_contradicted,
    }
