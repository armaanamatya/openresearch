"""Attach the two-axis reproducibility verdict to a final report (U11).

This is the integration seam between the pure decision engine
(``reproducibility_verdict.py``) and the on-disk ``final_report.json``.  It is:

  * **flag-gated** — off unless ``OPENRESEARCH_TWO_AXIS_VERDICT`` is truthy, so the
    existing single-score behaviour is byte-for-byte unchanged by default;
  * **additive** — it adds ``reproducibility`` + top-level mirror fields and sets
    ``schema_version=2``, leaving ``overall_score`` / ``rubric`` intact for the
    leaderboard + ``best_runs`` consumers;
  * **A4-correct** — when it applies, it sets ``report["verdict"]`` from the
    FIDELITY axis and signals the caller to SKIP ``reconcile_verdict_with_score``,
    so a faithful-contradicted run never collapses to legacy ``failed``;
  * **fail-soft** — any error leaves the report untouched and returns ``False``
    (the legacy reconcile path then runs as before).

Inputs are loaded from artifacts the upstream units write, with conservative
fallbacks so this is safe to enable before those units land:
  * fidelity score → the rubric's non-"Result match" areas (already computed);
  * certificate    → ``rlm_state/fidelity_certificate.json`` (U16) else not-green;
  * claims         → ``rlm_state/repro_spec.json`` (U13/U14/U15) else ``[]``.

With no certificate + no claims the verdict is honestly ``(partial|broken,
inconclusive)`` — never a false positive — until the real inputs exist.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from backend.agents.rlm.reproducibility_verdict import (
    ComparisonSpec,
    FidelityCertificate,
    MeasuredClaim,
    ScopeTuple,
    SeedBundle,
    compute_reproducibility_verdict,
)

logger = logging.getLogger(__name__)

_RESULT_MATCH_PREFIX = "result match"  # the replication area; excluded from fidelity
_DEFAULT_MIN_SEEDS = 2


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def is_enabled() -> bool:
    """Two-axis verdict is opt-in (default OFF). Escape hatch: unset the flag."""
    return _truthy(os.environ.get("OPENRESEARCH_TWO_AXIS_VERDICT"))


def _min_seeds_for_contradiction() -> int:
    """User-configurable seed floor for a contradiction (Q2); minimum 2."""
    raw = os.environ.get("OPENRESEARCH_MIN_SEEDS_FOR_CONTRADICTION")
    try:
        return max(2, int(raw)) if raw else _DEFAULT_MIN_SEEDS
    except (TypeError, ValueError):
        return _DEFAULT_MIN_SEEDS


# --------------------------------------------------------------------------- #
# Input assembly (fail-soft loaders)
# --------------------------------------------------------------------------- #

def fidelity_score_from_rubric(rubric: dict[str, Any]) -> float:
    """Fidelity-axis score = the rubric areas EXCEPT "Result match …".

    The replication axis owns the result-match area; everything else (method &
    code fidelity, data fidelity, eval-protocol correctness, execution, artifact
    completeness) measures whether WE built + ran the method faithfully.  Weighted
    by area weight when present, else a simple mean.  Falls back to the rubric's
    overall_score when no usable areas exist (legacy reports).
    """
    areas = (rubric or {}).get("areas") or []
    num = 0.0
    den = 0.0
    for area in areas:
        if not isinstance(area, dict):
            continue
        name = str(area.get("area", "")).strip().lower()
        if name.startswith(_RESULT_MATCH_PREFIX):
            continue
        score = area.get("score")
        if score is None:
            continue
        weight = area.get("weight")
        w = float(weight) if isinstance(weight, (int, float)) and weight > 0 else 1.0
        num += w * float(score)
        den += w
    if den > 0:
        return num / den
    return float((rubric or {}).get("overall_score", 0.0) or 0.0)


def _scope_from_dict(d: Any) -> ScopeTuple:
    if not isinstance(d, dict):
        return ScopeTuple()
    return ScopeTuple(
        model=str(d.get("model", "")),
        dataset=str(d.get("dataset", "")),
        split=str(d.get("split", "")),
        protocol=str(d.get("protocol", "")),
    )


def _comparison_from_dict(d: dict[str, Any]) -> ComparisonSpec:
    """Build a ComparisonSpec from a repro_spec claim dict.

    Missing/unparseable required numerics force ``ambiguous=True`` so the claim
    can never decide a verdict — the conservative default (A1).
    """
    ambiguous = bool(d.get("ambiguous", False))
    reason = str(d.get("ambiguity_reason", ""))
    try:
        claimed_effect = float(d["claimed_effect"])
        equivalence_margin = float(d.get("equivalence_margin", 0.0))
    except (KeyError, TypeError, ValueError):
        claimed_effect = 0.0
        equivalence_margin = 0.0
        ambiguous = True
        reason = reason or "claimed_effect / equivalence_margin missing or non-numeric"
    direction = d.get("direction", "higher_is_better")
    if direction not in ("higher_is_better", "lower_is_better"):
        direction, ambiguous, reason = "higher_is_better", True, reason or "unknown metric direction"
    estimate_kind = d.get("estimate_kind", "absolute")
    if estimate_kind not in ("percentage_points", "relative_percent", "absolute"):
        estimate_kind, ambiguous, reason = "absolute", True, reason or "unknown estimate kind"
    return ComparisonSpec(
        claim_id=str(d.get("claim_id", "claim")),
        description=str(d.get("description", "")),
        metric_name=str(d.get("metric_name", "")),
        direction=direction,  # type: ignore[arg-type]
        estimate_kind=estimate_kind,  # type: ignore[arg-type]
        baseline_label=str(d.get("baseline_label", "")),
        claimed_effect=claimed_effect,
        equivalence_margin=max(0.0, equivalence_margin),
        scope=_scope_from_dict(d.get("scope")),
        is_primary=bool(d.get("is_primary", False)),
        table_ref=str(d.get("table_ref", "")),
        paper_span=str(d.get("paper_span", "")),
        ambiguous=ambiguous,
        ambiguity_reason=reason,
    )


def _seed_bundle_from_dict(d: Any) -> SeedBundle:
    if not isinstance(d, dict):
        return SeedBundle()
    try:
        seeds = tuple(int(s) for s in (d.get("seeds") or []))
        effects = tuple(float(e) for e in (d.get("per_seed_effect") or []))
    except (TypeError, ValueError):
        return SeedBundle()
    return SeedBundle(seeds=seeds, per_seed_effect=effects, rng_independent=bool(d.get("rng_independent", False)))


def load_claims(run_dir: Path) -> list[MeasuredClaim]:
    """Load the frozen ReproSpec claims (+ measured data). [] when absent/bad."""
    path = run_dir / "rlm_state" / "repro_spec.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — a bad artifact must not break the report
        logger.warning("two_axis: could not read repro_spec.json (%s)", exc)
        return []
    claims: list[MeasuredClaim] = []
    for entry in (data.get("claims") if isinstance(data, dict) else None) or []:
        if not isinstance(entry, dict) or not isinstance(entry.get("comparison"), dict):
            continue
        try:
            claims.append(
                MeasuredClaim(
                    comparison=_comparison_from_dict(entry["comparison"]),
                    seed_bundle=_seed_bundle_from_dict(entry.get("seed_bundle")),
                    measured_scope=_scope_from_dict(entry.get("measured_scope")),
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("two_axis: skipping malformed claim (%s)", exc)
    return claims


def load_certificate(run_dir: Path, *, has_measured_metrics: bool) -> FidelityCertificate:
    """Load the executable fidelity certificate (U16); conservative when absent.

    Absent artifact → NOT green (we have not executably certified fidelity), so
    the implementation verdict caps at ``partial`` and no contradiction can be
    minted.  Honest until U16 writes real certificates.
    """
    path = run_dir / "rlm_state" / "fidelity_certificate.json"
    if not path.exists():
        return FidelityCertificate(
            invariant_tests_passed=False,
            mutation_confirmed=False,
            blinded_extraction_agreed=False,
            obligation_profile="end_to_end",
            profile_satisfied=False,
            has_measured_metrics=has_measured_metrics,
            invariant_tests_ran=False,  # no executable certificate yet → partial, not broken
        )
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        profile = d.get("obligation_profile", "end_to_end")
        if profile not in ("static", "forward_pass", "multi_step", "trace", "end_to_end"):
            profile = "end_to_end"
        return FidelityCertificate(
            invariant_tests_passed=bool(d.get("invariant_tests_passed", False)),
            mutation_confirmed=bool(d.get("mutation_confirmed", False)),
            blinded_extraction_agreed=bool(d.get("blinded_extraction_agreed", False)),
            obligation_profile=profile,  # type: ignore[arg-type]
            profile_satisfied=bool(d.get("profile_satisfied", False)),
            has_measured_metrics=bool(d.get("has_measured_metrics", has_measured_metrics)),
            invariant_tests_ran=bool(d.get("invariant_tests_ran", d.get("invariant_tests_passed", False))),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("two_axis: could not read fidelity_certificate.json (%s)", exc)
        return FidelityCertificate(
            invariant_tests_passed=False, mutation_confirmed=False,
            blinded_extraction_agreed=False, obligation_profile="end_to_end",
            profile_satisfied=False, has_measured_metrics=has_measured_metrics,
            invariant_tests_ran=False,
        )


def _has_measured_metrics(run_dir: Path) -> bool:
    code = run_dir / "code"
    if (code / "metrics.json").exists():
        return True
    outputs = code / "outputs"
    if outputs.exists():
        for p in outputs.rglob("metrics.json"):
            if p.exists():
                return True
    return False


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def _verdict_to_payload(v, fidelity_score: float) -> dict[str, Any]:
    return {
        "schema_version": v.schema_version,
        "implementation_verdict": v.implementation_verdict,
        "replication_verdict": v.replication_verdict,
        "replication_credit": v.replication_credit,
        "fidelity_score": round(fidelity_score, 4),
        "legacy_verdict": v.legacy_verdict,
        "per_claim": [
            {
                "claim_id": cv.claim_id,
                "status": cv.status,
                "credit": cv.credit,
                "reason": cv.reason,
                "eligible": cv.eligible,
                "measured_mean": cv.measured_mean,
                "ci_low": cv.ci_low,
                "ci_high": cv.ci_high,
            }
            for cv in v.per_claim
        ],
        "rationale": list(v.rationale),
    }


def compute_and_attach(report: dict[str, Any], run_dir: Path) -> bool:
    """Compute the two-axis verdict and attach it to ``report`` in place.

    Returns ``True`` iff the two-axis verdict was applied (the caller must then
    SKIP the legacy ``reconcile_verdict_with_score``).  Returns ``False`` when
    disabled or on any error (legacy behaviour preserved).
    """
    if not is_enabled():
        return False
    try:
        rubric = report.get("rubric") if isinstance(report.get("rubric"), dict) else {}
        fidelity_score = fidelity_score_from_rubric(rubric)
        has_metrics = _has_measured_metrics(run_dir)
        certificate = load_certificate(run_dir, has_measured_metrics=has_metrics)
        claims = load_claims(run_dir)
        verdict = compute_reproducibility_verdict(
            fidelity_score=fidelity_score,
            certificate=certificate,
            claims=claims,
            min_seeds_for_contradiction=_min_seeds_for_contradiction(),
        )
        report["reproducibility"] = _verdict_to_payload(verdict, fidelity_score)
        # Top-level mirrors for the leaderboard / UI (cheap to read).
        report["implementation_verdict"] = verdict.implementation_verdict
        report["replication_verdict"] = verdict.replication_verdict
        report["schema_version"] = verdict.schema_version
        # A4 — project the legacy verdict from FIDELITY, NOT the blended score.
        report["verdict"] = verdict.legacy_verdict
        return True
    except Exception as exc:  # noqa: BLE001 — never break report finalisation
        logger.warning("two_axis: compute_and_attach failed (%s) — falling back to legacy", exc)
        return False
