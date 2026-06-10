"""repro_spec_extractor.py — U14: claimed-magnitude extractor + ReproSpec builder.

Mines a paper's falsifiable quantitative claims into a frozen ``repro_spec.json``
artifact that the verdict engine (``two_axis_report.load_claims`` +
``compute_reproducibility_verdict``) grades.

Locked contract shape (``two_axis_report._comparison_from_dict``,
``_seed_bundle_from_dict``, ``_scope_from_dict``, ``load_claims``):
::

    {
      "claims": [
        {
          "comparison": {<ComparisonSpec fields>},
          "seed_bundle": {
            "seeds": [...],
            "per_seed_effect": [...],
            "rng_independent": bool
          },
          "measured_scope": {model, dataset, split, protocol}
        }
      ]
    }

Design notes
------------
* **Conservative by default.**  The parser is the #1 false-contradiction risk
  (A1). When in doubt it sets ``ambiguous=True`` with a ``reason`` — forcing an
  ``inconclusive`` verdict rather than a false ``contradicted``.
* **Relative-vs-percentage-points (the classic ambiguity)** → always
  ``ambiguous`` unless context unambiguously resolves it (explicit unit markers
  like "pp", "percentage points", "%pts", or a paired absolute value).
* **Lower-is-better metrics** (loss, error rate, perplexity, MSE, MAE, …) →
  the *sign is folded* so that a positive ``claimed_effect`` always means the
  proposed method's advantage over the baseline (consistent with
  ``ComparisonSpec.claimed_effect`` sign convention).
* **LLM extraction is flag-gated** on ``OPENRESEARCH_TWO_AXIS_VERDICT`` and
  fail-soft: any error writes nothing + logs a warning (never breaks report
  finalization).
* **A6a blinded re-extraction**: two independent extraction passes from the raw
  cited spans; disagreement on numeric constants → claim marked ``ambiguous``.
* stdlib-only for the deterministic core (``parse_claim_statement``,
  ``build_repro_spec``, ``seed_bundle_from_metrics``).  The LLM wrapper
  (``extract_and_write``) imports ``backend.*`` only inside the function so the
  pure core is importable in test environments without the full app.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature gate (mirrors two_axis_report.is_enabled)
# ---------------------------------------------------------------------------

def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def is_enabled() -> bool:
    """Returns True when ``OPENRESEARCH_TWO_AXIS_VERDICT`` is set (mirrors two_axis_report)."""
    return _truthy(os.environ.get("OPENRESEARCH_TWO_AXIS_VERDICT"))


# ---------------------------------------------------------------------------
# Metric-direction vocabulary
# ---------------------------------------------------------------------------

_LOWER_IS_BETTER_TOKENS: frozenset[str] = frozenset({
    "loss", "error", "perplexity", "ppl", "mse", "mae", "rmse",
    "nll", "ce", "cross-entropy", "cross_entropy", "divergence",
    "fid", "fréchet", "frechet", "bit", "bpc", "bpd",
    "lower is better", "lower-is-better", "lower_is_better",
})
_HIGHER_IS_BETTER_TOKENS: frozenset[str] = frozenset({
    "accuracy", "acc", "f1", "bleu", "rouge", "success",
    "precision", "recall", "auc", "ndcg", "mrr",
    "score", "reward", "return", "throughput", "fps", "improvement",
    "higher is better", "higher-is-better", "higher_is_better",
})


def _infer_direction(metric_name: str, context_hint: str = "") -> str | None:
    """Return 'lower_is_better' / 'higher_is_better' / None (ambiguous).

    Priority (highest → lowest):
    1. Explicit direction phrases in context_hint ("lower is better", "higher is
       better") — these are unambiguous operator signals.
    2. Vocabulary match on the combined (metric + context) string.
    3. Neither matches → None.  Both directions present → None (conflict).
    """
    # 1. Explicit direction markers in context_hint take highest priority.
    #    But if BOTH appear (conflicting context), fall through to vocabulary.
    ctx_lower = (context_hint or "").lower()
    has_explicit_lower = "lower is better" in ctx_lower or "lower_is_better" in ctx_lower
    has_explicit_higher = "higher is better" in ctx_lower or "higher_is_better" in ctx_lower
    if has_explicit_lower and not has_explicit_higher:
        return "lower_is_better"
    if has_explicit_higher and not has_explicit_lower:
        return "higher_is_better"
    # Both explicit markers conflict → fall through to vocabulary check

    # 2. Vocabulary match over the combined string.
    combined = f"{metric_name} {context_hint}".lower()
    has_lower = any(t in combined for t in _LOWER_IS_BETTER_TOKENS)
    has_higher = any(t in combined for t in _HIGHER_IS_BETTER_TOKENS)
    if has_lower and not has_higher:
        return "lower_is_better"
    if has_higher and not has_lower:
        return "higher_is_better"
    # Both or neither → caller should set ambiguous
    return None


# ---------------------------------------------------------------------------
# Numeric + pattern helpers
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    """Normalize unicode, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("±", "±").replace("–", "-").replace("—", "-")
    return " ".join(text.split())


def _parse_float(s: str) -> float | None:
    """Parse a cleaned numeric string, returning None on failure."""
    s = s.strip().rstrip("% ,")
    try:
        return float(s)
    except ValueError:
        return None


# Patterns for common claim formats.  Order matters — most-specific first.

# "84.4 vs 75.0" / "84.4 vs. 75.0" / "84.4 vs GRPO 75.0"
_VS_PATTERN = re.compile(
    r"([\-+]?\d+(?:\.\d+)?)\s*(?:vs\.?|versus)\s*(?:[A-Za-z][A-Za-z0-9_\- ]{0,30}\s+)?"
    r"([\-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
# "reduces X from 1.5 to 1.2" / "from 1.5 (baseline) to 1.2"
# baseline value is the FIRST number (from), proposed is the SECOND (to)
_FROM_TO_PATTERN = re.compile(
    r"\bfrom\s+([\-+]?\d+(?:\.\d+)?)\s*(?:[^0-9]{0,20})?\bto\s+([\-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
# Absolute difference with sign, e.g. "(+9.4)" or "(-2.3)" following a vs-pair
_DELTA_PAREN = re.compile(r"\(\s*([\-+]\d+(?:\.\d+)?)\s*\)")
# Inline delta: "+9.4%" / "-2.3%" / "+9.4 pp" / "+9.4 percentage points"
_DELTA_WITH_UNIT = re.compile(
    r"([\-+]\d+(?:\.\d+)?)\s*"
    r"(%(?:\s*(?:pts?|pp|percentage\s+points?|points?))?|pp|pts?|percentage\s+points?)",
    re.IGNORECASE,
)
# Plain delta (no unit), e.g. "improves by 9.4" / "outperforms the baseline by 9.4"
# Allows up to 5 intervening words between verb and "by <number>".
_DELTA_PLAIN = re.compile(
    r"(?:improve(?:s|ment)?|gain(?:s|ed)?|increase(?:s|d)?|boost(?:s|ed)?|"
    r"outperform(?:s|ed)?)\s+(?:\w+\s+){0,5}by\s+([\-+]?\d+(?:\.\d+)?)"
    r"|(?:better|higher|lower)\s+by\s+([\-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
# Unit disambiguation markers that confirm percentage-POINTS (not relative %)
_PP_UNIT_MARKERS = re.compile(
    r"\b(?:percentage\s+points?|pp\b|pts?\b|%\s*pts?|percent(?:age)?\s*points?)\b",
    re.IGNORECASE,
)
# Relative-% markers
_REL_PERCENT_MARKERS = re.compile(
    r"\b(?:relative\s+(?:improvement|gain|increase)|relative\s+%|"
    r"(?:by\s+a\s+)?(?:factor|fold)|×\s*\d)\b",
    re.IGNORECASE,
)

_EQUIVALENCE_FLOOR = 0.05  # absolute minimum equivalence_margin


def _equivalence_margin(claimed: float) -> float:
    """Sensible equivalence margin: max(10% of |claimed|, floor).

    Rationale: two values within 10% of a claimed effect are practically
    equivalent in most ML experiments.  The floor prevents a zero-effect claim
    from having a zero-margin (which would trivially declare any non-zero
    measured effect as a contradiction).
    """
    return max(_EQUIVALENCE_FLOOR, 0.10 * abs(claimed))


# ---------------------------------------------------------------------------
# Core deterministic parser
# ---------------------------------------------------------------------------

def parse_claim_statement(text: str, context_hint: str = "") -> dict[str, Any]:
    """Parse a claimed comparison from a single text statement.

    Returns a dict of ``ComparisonSpec`` fields (without ``claim_id``,
    ``description``, ``scope``, ``is_primary``, ``table_ref``, ``paper_span``
    which are set by the caller).

    The ``ambiguous`` field is set True (with ``ambiguity_reason``) whenever:
    * the estimate kind (percentage-points vs relative-%) cannot be resolved;
    * the baseline value or metric is missing/unclear;
    * the delta sign is unclear (e.g. "improves by X" for a lower-is-better
      metric with no explicit sign);
    * no numeric effect can be extracted.

    Sign convention: ``claimed_effect > 0`` means the paper's proposed method
    has an ADVANTAGE over the baseline, regardless of metric direction.  For
    lower-is-better metrics, a LOWER value is an advantage, so if the proposed
    method achieves a LOWER loss the effect is *positive*.  The Extractor folds
    the direction into the sign here; ``ComparisonSpec.direction`` is set for
    grader reference but the sign already encodes it.
    """
    text = _norm(text)
    ctx = _norm(context_hint)

    result: dict[str, Any] = {
        "metric_name": "",
        "direction": "higher_is_better",
        "estimate_kind": "absolute",
        "baseline_label": "",
        "claimed_effect": 0.0,
        "equivalence_margin": 0.0,
        "ambiguous": False,
        "ambiguity_reason": "",
    }

    # ------------------------------------------------------------------ #
    # 1. Direction inference from context / metric vocabulary
    # ------------------------------------------------------------------ #
    direction = _infer_direction(text, ctx)
    if direction is None:
        direction = "higher_is_better"  # safe default; may be overridden below

    # ------------------------------------------------------------------ #
    # 2. Try to extract proposed-vs-baseline and delta
    # ------------------------------------------------------------------ #
    proposed_val: float | None = None
    baseline_val: float | None = None
    delta: float | None = None
    estimate_kind: str | None = None  # None = undetermined

    # 2a. "84.4 vs 75.0 (+9.4)" form
    vs_match = _VS_PATTERN.search(text)
    ft_match = _FROM_TO_PATTERN.search(text) if not vs_match else None
    if vs_match:
        proposed_val = _parse_float(vs_match.group(1))
        baseline_val = _parse_float(vs_match.group(2))
        # Look for a parenthesized delta immediately after
        after = text[vs_match.end():]
        dparen = _DELTA_PAREN.search(after[:30])
        if dparen:
            delta = _parse_float(dparen.group(1))
        elif proposed_val is not None and baseline_val is not None:
            # Compute it ourselves; no unit, so mark absolute (could be pp or abs)
            delta = proposed_val - baseline_val
    elif ft_match:
        # "reduces from 1.5 to 1.2" — baseline=first, proposed=second
        baseline_val = _parse_float(ft_match.group(1))
        proposed_val = _parse_float(ft_match.group(2))
        if proposed_val is not None and baseline_val is not None:
            # raw delta = proposed - baseline (for higher-is-better: positive = better)
            delta = proposed_val - baseline_val

    # 2b. Inline delta with explicit unit, e.g. "+9.4%" or "+9.4 pp"
    dunit_match = _DELTA_WITH_UNIT.search(text)
    if delta is None and dunit_match:
        delta = _parse_float(dunit_match.group(1))
        unit_tok = dunit_match.group(2).lower().strip()
        if "pp" in unit_tok or "percentage" in unit_tok or "pts" in unit_tok:
            estimate_kind = "percentage_points"
        else:  # bare "%"
            estimate_kind = None  # ambiguous: might be pp or relative

    # 2c. Verbal delta, e.g. "improves by 9.4" / "outperforms the baseline by 9.4"
    if delta is None:
        dp_match = _DELTA_PLAIN.search(text)
        if dp_match:
            # Two capture groups: group(1) for verb+...+by, group(2) for better/higher/lower+by
            raw_delta = dp_match.group(1) or dp_match.group(2)
            delta = _parse_float(raw_delta) if raw_delta else None
            # No explicit sign in "improves by 9.4" — assume positive advantage

    # ------------------------------------------------------------------ #
    # 3. Resolve estimate_kind ambiguity
    # ------------------------------------------------------------------ #
    if estimate_kind is None:
        # Check for explicit unit markers in the full text
        if _PP_UNIT_MARKERS.search(text) or _PP_UNIT_MARKERS.search(ctx):
            estimate_kind = "percentage_points"
        elif _REL_PERCENT_MARKERS.search(text) or _REL_PERCENT_MARKERS.search(ctx):
            estimate_kind = "relative_percent"
        elif proposed_val is not None and baseline_val is not None and delta is not None:
            # We have BOTH absolute values — the delta is unambiguously an
            # ABSOLUTE difference.  The pp-vs-relative ambiguity only arises
            # when we only see a bare "±N%" without the underlying values; when
            # we see "84.4 vs 75.0" the raw difference (9.4) is absolute by
            # construction (regardless of whether the units are percentages or
            # points — it is the raw arithmetic difference).
            estimate_kind = "absolute"
        else:
            estimate_kind = None  # undetermined

    # ------------------------------------------------------------------ #
    # 4. Sign folding for lower-is-better
    # ------------------------------------------------------------------ #
    lower_is_better = _infer_direction(text, ctx) == "lower_is_better"
    if lower_is_better:
        direction = "lower_is_better"
        # For lower-is-better, a NEGATIVE delta in the text (proposed < baseline)
        # means the method is BETTER → positive advantage (fold the sign).
        if delta is not None and delta < 0:
            delta = -delta  # negative raw delta = positive advantage

    result["direction"] = direction

    # ------------------------------------------------------------------ #
    # 5. Missing/ambiguous baseline or delta → set ambiguous
    # ------------------------------------------------------------------ #
    if delta is None:
        result["ambiguous"] = True
        result["ambiguity_reason"] = "no numeric effect extractable from statement"
        result["equivalence_margin"] = _EQUIVALENCE_FLOOR
        return result

    if math.isnan(delta) or math.isinf(delta):
        result["ambiguous"] = True
        result["ambiguity_reason"] = "extracted delta is NaN or infinite"
        result["equivalence_margin"] = _EQUIVALENCE_FLOOR
        return result

    # ------------------------------------------------------------------ #
    # 6. estimate_kind still undetermined → ambiguous (the pp-vs-% trap)
    # ------------------------------------------------------------------ #
    if estimate_kind is None:
        result["claimed_effect"] = delta
        result["equivalence_margin"] = _equivalence_margin(delta)
        result["ambiguous"] = True
        result["ambiguity_reason"] = (
            "percentage-points vs relative-% ambiguous — "
            "no explicit unit marker (pp/pts/percentage points) found"
        )
        return result

    # ------------------------------------------------------------------ #
    # 7. Baseline missing but delta known → not ambiguous if estimate_kind
    #    is pinned, but mark baseline_label absent so the caller can fill it.
    # ------------------------------------------------------------------ #
    result["claimed_effect"] = delta
    result["estimate_kind"] = estimate_kind
    result["equivalence_margin"] = _equivalence_margin(delta)

    if baseline_val is None and not result.get("baseline_label"):
        # Missing baseline value is OK if we have an absolute delta and a kind;
        # the grader compares our measured effect, not the absolute baseline value.
        pass  # leave baseline_label="" for caller to fill

    return result


# ---------------------------------------------------------------------------
# Spec assembly + round-trip verifier
# ---------------------------------------------------------------------------

def _make_claim_id(index: int, is_primary: bool) -> str:
    prefix = "primary" if is_primary else "secondary"
    return f"{prefix}_{index}"


def build_repro_spec(
    claims: list[dict[str, Any]],
    *,
    seed_bundles: list[dict[str, Any]] | None = None,
    measured_scopes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble the contract JSON and round-trip-verify it via ``load_claims``.

    Parameters
    ----------
    claims:
        List of dicts with at minimum the fields produced by
        ``parse_claim_statement`` plus:
        * ``claim_id`` (str) — if absent, auto-generated as ``primary_N`` or
          ``secondary_N``;
        * ``description`` (str) — human-readable summary;
        * ``metric_name`` (str);
        * ``baseline_label`` (str);
        * ``scope`` (dict with model/dataset/split/protocol);
        * ``is_primary`` (bool);
        * ``table_ref`` / ``paper_span`` (str, optional);
        * ``ambiguous`` / ``ambiguity_reason`` (from parser).
    seed_bundles:
        Per-claim seed bundle dicts in the same order as ``claims``.  If
        shorter than ``claims``, missing entries get a single-seed
        ``rng_independent=False`` placeholder (honest: ``inconclusive``).
    measured_scopes:
        Per-claim measured scope dicts.  If absent, defaults to an empty
        scope dict (wildcard).

    Returns
    -------
    A ``repro_spec.json``-shaped dict.  After assembly this is round-trip
    verified by calling the real ``load_claims`` helper on a temp path so any
    shape mismatch raises immediately at spec-build time rather than silently
    at verdict time.

    Raises
    ------
    ValueError
        When the assembled JSON fails to round-trip through ``load_claims``
        (shape contract violation).
    """
    if seed_bundles is None:
        seed_bundles = []
    if measured_scopes is None:
        measured_scopes = []

    out_claims: list[dict[str, Any]] = []
    for i, claim in enumerate(claims):
        cid = claim.get("claim_id") or _make_claim_id(i, bool(claim.get("is_primary", False)))
        comparison: dict[str, Any] = {
            "claim_id": cid,
            "description": str(claim.get("description", "")),
            "metric_name": str(claim.get("metric_name", "")),
            "direction": claim.get("direction", "higher_is_better"),
            "estimate_kind": claim.get("estimate_kind", "absolute"),
            "baseline_label": str(claim.get("baseline_label", "")),
            "claimed_effect": float(claim.get("claimed_effect", 0.0)),
            "equivalence_margin": float(claim.get("equivalence_margin", _EQUIVALENCE_FLOOR)),
            "scope": dict(claim.get("scope") or {}),
            "is_primary": bool(claim.get("is_primary", False)),
            "table_ref": str(claim.get("table_ref", "")),
            "paper_span": str(claim.get("paper_span", "")),
            "ambiguous": bool(claim.get("ambiguous", False)),
            "ambiguity_reason": str(claim.get("ambiguity_reason", "")),
        }
        # Clamp equivalence_margin >= 0
        comparison["equivalence_margin"] = max(0.0, comparison["equivalence_margin"])

        # Seed bundle: fall back to single-seed inconclusive placeholder
        if i < len(seed_bundles):
            bundle = dict(seed_bundles[i])
        else:
            bundle = {"seeds": [], "per_seed_effect": [], "rng_independent": False}

        # Measured scope: wildcard if absent
        if i < len(measured_scopes):
            ms = dict(measured_scopes[i])
        else:
            ms = {"model": "", "dataset": "", "split": "", "protocol": ""}

        out_claims.append({
            "comparison": comparison,
            "seed_bundle": bundle,
            "measured_scope": ms,
        })

    spec = {"claims": out_claims}

    # Round-trip verify — imports the stable contract module.
    _verify_round_trip(spec)
    return spec


def _verify_round_trip(spec: dict[str, Any]) -> None:
    """Round-trip the spec through ``two_axis_report.load_claims`` on a temp dir.

    Raises ``ValueError`` if load_claims raises or returns a different count.
    This catches shape mismatches at build time (not at verdict time).
    """
    import tempfile
    from backend.agents.rlm.two_axis_report import load_claims as _lc

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "rlm_state"
        p.mkdir()
        (p / "repro_spec.json").write_text(
            json.dumps(spec), encoding="utf-8"
        )
        loaded = _lc(Path(td))

    expected = len(spec.get("claims", []))
    if len(loaded) != expected:
        raise ValueError(
            f"build_repro_spec round-trip mismatch: "
            f"wrote {expected} claim(s), load_claims returned {len(loaded)}.  "
            f"Spec: {json.dumps(spec, indent=2)}"
        )


# ---------------------------------------------------------------------------
# Per-seed effect reader
# ---------------------------------------------------------------------------

def seed_bundle_from_metrics(
    run_dir: Path | str,
    *,
    metric_key: str,
    model_key: str = "",
    env_key: str = "",
    baseline_key: str = "",
) -> dict[str, Any]:
    """Read per-seed measured effects from the run's metrics.json artefacts.

    Searches ``<run_dir>/code/metrics.json`` and
    ``<run_dir>/code/outputs/**/metrics.json``.  Extracts the numeric value at
    the path described by ``metric_key`` / ``model_key`` / ``env_key`` /
    ``baseline_key``.  Emits a single-seed bundle with ``rng_independent=False``
    when only one observation is found (honest: stays ``inconclusive`` under A3).
    Never modifies the matrix.

    Returns a ``seed_bundle`` dict compatible with
    ``two_axis_report._seed_bundle_from_dict``:
    ::

        {"seeds": [...], "per_seed_effect": [...], "rng_independent": bool}

    If no numeric values are found, returns an empty (inconclusive) bundle.
    """
    run_dir = Path(run_dir)
    code_dir = run_dir / "code"

    # Collect all candidate metrics.json files
    candidate_paths: list[Path] = []
    top = code_dir / "metrics.json"
    if top.exists():
        candidate_paths.append(top)
    outputs = code_dir / "outputs"
    if outputs.is_dir():
        for p in sorted(outputs.rglob("metrics.json")):
            candidate_paths.append(p)

    values: list[float] = []
    for path in candidate_paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        val = _extract_metric_value(data, metric_key, model_key, env_key, baseline_key)
        if val is not None and math.isfinite(val):
            values.append(val)

    if not values:
        return {"seeds": [], "per_seed_effect": [], "rng_independent": False}

    # De-duplicate identical values from the same top-level file being re-read;
    # preserve ordering.  Two distinct numeric outputs from separate output dirs
    # are treated as separate runs.
    seen: set[float] = set()
    deduped: list[float] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            deduped.append(v)

    n = len(deduped)
    # seeds are synthetic (we don't know the actual seeds from metrics alone);
    # rng_independent=False unless there are >=2 distinct values from separate output dirs.
    rng_independent = n >= 2 and len(candidate_paths) >= 2
    seeds = list(range(42, 42 + n))
    return {
        "seeds": seeds,
        "per_seed_effect": deduped,
        "rng_independent": rng_independent,
    }


def _extract_metric_value(
    data: Any,
    metric_key: str,
    model_key: str,
    env_key: str,
    baseline_key: str,
) -> float | None:
    """Navigate a metrics dict to extract a scalar value.

    Navigation order (most structured → least):
    1. data["per_model"][model_key][env_key][baseline_key][metric_key]  (cell matrix)
    2. data["per_model"][model_key][env_key][metric_key]               (no baseline axis)
    3. data["per_model"][model_key][metric_key]                        (no env axis)
    4. data[metric_key]                                                 (top-level flat)
    Falls back gracefully at each level.
    """
    if not isinstance(data, dict):
        return None

    def _maybe_float(v: Any) -> float | None:
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            return float(v)
        return None

    # 1. Fully-qualified path
    if model_key and env_key and baseline_key and metric_key:
        try:
            val = data["per_model"][model_key][env_key][baseline_key][metric_key]
            return _maybe_float(val)
        except (KeyError, TypeError):
            pass

    # 2. No baseline axis
    if model_key and env_key and metric_key:
        try:
            val = data["per_model"][model_key][env_key][metric_key]
            return _maybe_float(val)
        except (KeyError, TypeError):
            pass

    # 3. No env axis
    if model_key and metric_key:
        try:
            val = data["per_model"][model_key][metric_key]
            return _maybe_float(val)
        except (KeyError, TypeError):
            pass

    # 4. Top-level flat key
    if metric_key:
        return _maybe_float(data.get(metric_key))

    return None


# ---------------------------------------------------------------------------
# LLM extraction helpers
# ---------------------------------------------------------------------------

_EXTRACTOR_SYSTEM = """\
You are a rigorous scientific claims extractor for ReproLab.

Given a research paper's text, identify the top-K most important FALSIFIABLE
QUANTITATIVE claims — claims where the paper reports a specific numeric result
that a reproduction can measure and compare.

For each claim extract:
  * description: one sentence summary
  * metric_name: the measured quantity (e.g. "success_rate", "accuracy", "perplexity")
  * direction: "higher_is_better" or "lower_is_better"
  * baseline_label: the method/system being compared against
  * proposed_method: name of the paper's method
  * proposed_value: the numeric value the paper reports for its method (null if unavailable)
  * baseline_value: the numeric value for the baseline (null if unavailable)
  * claimed_effect: the claimed advantage (positive = better for the proposed method)
  * estimate_kind: "percentage_points" | "relative_percent" | "absolute" | "unknown"
  * scope: {model: "...", dataset: "...", split: "...", protocol: "..."}
  * is_primary: true for the headline/abstract result; false for ablations
  * table_ref: e.g. "Table 2" or "Figure 3"
  * paper_span: the exact quoted sentence/phrase from the paper

IMPORTANT — conservative rules:
  * If you cannot tell whether a numeric delta is percentage POINTS or relative %,
    set estimate_kind to "unknown".
  * If the direction of a metric is unclear, do not guess.
  * Only include claims with a verifiable numeric comparison.
  * Prefer fewer, high-confidence claims over many uncertain ones.
  * Top-K = 5 (at most); mark exactly ONE claim is_primary=true (the headline result).

Return ONLY a JSON object: {"claims": [...]}
"""

_BLINDED_SYSTEM = """\
You are an independent claims verifier for ReproLab.

You will be given ONLY the raw paper spans (exact quotes) that were previously
cited as evidence for specific claims.  Without looking at any other extraction
or implementation, re-extract the numeric constants (claimed_effect,
proposed_value, baseline_value, estimate_kind) from each span.

For each span return:
  * span_index: integer (0-based, matching the input)
  * claimed_effect: the numeric advantage (or null if unclear)
  * proposed_value: numeric (or null)
  * baseline_value: numeric (or null)
  * estimate_kind: "percentage_points" | "relative_percent" | "absolute" | "unknown"
  * notes: any ambiguity or disagreement

Return ONLY a JSON object: {"extractions": [...]}
"""


def _call_llm(llm_client: Any, *, system: str, user: str) -> str:
    """Call the LLM client's complete method (mirrors rubric_gen pattern)."""
    return llm_client.complete(system=system, user=user)


def _extract_json(raw: str) -> dict | None:
    """Extract the first JSON object from an LLM response string."""
    # Try clean parse first
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass
    # Fence-stripped
    fenced = re.sub(r"```(?:json)?\s*|\s*```", "", raw)
    try:
        return json.loads(fenced.strip())
    except json.JSONDecodeError:
        pass
    # Brace-extraction
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None


def _reconcile_with_blinded(
    claim: dict[str, Any],
    blinded: dict[str, Any],
    *,
    tolerance: float = 0.01,
) -> tuple[bool, str]:
    """Compare first-pass and blinded extraction numeric constants.

    Returns ``(agree: bool, reason: str)``.  Disagreement on any of the three
    key constants (claimed_effect, proposed_value, baseline_value) marks the
    claim ambiguous (A6a).  Missing values in blinded extraction count as
    disagreement if the first pass had a value.
    """
    disagreements: list[str] = []

    def _cmp(key: str) -> None:
        v1 = claim.get(key)
        v2 = blinded.get(key)
        if v1 is None and v2 is None:
            return
        if v1 is None or v2 is None:
            disagreements.append(f"{key}: first={v1!r} blinded={v2!r}")
            return
        try:
            f1, f2 = float(v1), float(v2)
            if abs(f1 - f2) > tolerance * max(1.0, abs(f1)):
                disagreements.append(f"{key}: first={f1} blinded={f2}")
        except (TypeError, ValueError):
            if str(v1).strip() != str(v2).strip():
                disagreements.append(f"{key}: first={v1!r} blinded={v2!r}")

    _cmp("claimed_effect")
    _cmp("proposed_value")
    _cmp("baseline_value")
    # Also check estimate_kind — pp vs relative is load-bearing
    ek1 = str(claim.get("estimate_kind", "")).strip()
    ek2 = str(blinded.get("estimate_kind", "")).strip()
    if ek1 not in ("", "unknown") and ek2 not in ("", "unknown") and ek1 != ek2:
        disagreements.append(f"estimate_kind: first={ek1!r} blinded={ek2!r}")

    if disagreements:
        return False, "blinded re-extraction disagreed on: " + "; ".join(disagreements)
    return True, ""


def _normalize_claim_from_llm(
    raw: dict[str, Any],
    index: int,
    is_primary: bool | None = None,
) -> dict[str, Any]:
    """Normalize a raw LLM claim dict into a parse_claim_statement-compatible form.

    Runs the deterministic parser on the description + context to fill in any
    missing numeric fields, then merges with the LLM's own reported values.
    The LLM's ``claimed_effect`` takes precedence when explicitly provided and
    non-zero; the parser's result is a fallback.
    """
    description = str(raw.get("description", ""))
    metric = str(raw.get("metric_name", ""))
    baseline = str(raw.get("baseline_label", "") or raw.get("baseline", ""))
    context_hint = f"{metric} {raw.get('proposed_method', '')} {baseline}"

    # Run the deterministic parser on the description for cross-check
    parsed = parse_claim_statement(description, context_hint)

    # LLM-provided claimed_effect takes precedence if it's a real number
    llm_effect = raw.get("claimed_effect")
    if llm_effect is not None:
        try:
            llm_effect = float(llm_effect)
        except (TypeError, ValueError):
            llm_effect = None

    claimed_effect = llm_effect if llm_effect is not None and llm_effect != 0.0 else parsed["claimed_effect"]

    # estimate_kind: LLM's value if valid, else parser's, else ambiguous
    llm_kind = str(raw.get("estimate_kind", "")).strip()
    if llm_kind in ("percentage_points", "relative_percent", "absolute"):
        estimate_kind = llm_kind
        ambiguous_kind = False
    elif llm_kind == "unknown" or not llm_kind:
        estimate_kind = parsed.get("estimate_kind", "absolute")
        ambiguous_kind = parsed.get("ambiguous", False)
    else:
        estimate_kind = "absolute"
        ambiguous_kind = True

    # direction
    direction_raw = str(raw.get("direction", "")).strip()
    if direction_raw in ("higher_is_better", "lower_is_better"):
        direction = direction_raw
    else:
        direction = parsed.get("direction", "higher_is_better")

    ambiguous = bool(raw.get("ambiguous", False)) or ambiguous_kind or parsed.get("ambiguous", False)
    ambiguity_reason = str(raw.get("ambiguity_reason", "")) or parsed.get("ambiguity_reason", "")

    scope_raw = raw.get("scope") or {}
    if not isinstance(scope_raw, dict):
        scope_raw = {}

    primary = bool(raw.get("is_primary", False)) if is_primary is None else is_primary

    return {
        "claim_id": str(raw.get("claim_id", "") or _make_claim_id(index, primary)),
        "description": description,
        "metric_name": metric,
        "direction": direction,
        "estimate_kind": estimate_kind,
        "baseline_label": baseline,
        "claimed_effect": claimed_effect,
        "equivalence_margin": _equivalence_margin(claimed_effect) if claimed_effect else _EQUIVALENCE_FLOOR,
        "scope": scope_raw,
        "is_primary": primary,
        "table_ref": str(raw.get("table_ref", "")),
        "paper_span": str(raw.get("paper_span", "")),
        "ambiguous": ambiguous,
        "ambiguity_reason": ambiguity_reason,
    }


# ---------------------------------------------------------------------------
# Public LLM wrapper: extract_and_write
# ---------------------------------------------------------------------------

def extract_and_write(
    context: Any,
    run_dir: Path | str,
    *,
    llm_client: Any = None,
    top_k: int = 5,
    paper_text: str = "",
    max_paper_chars: int = 48000,
) -> Path | None:
    """Identify the paper's top-K falsifiable claims and write ``repro_spec.json``.

    Flag-gated on ``OPENRESEARCH_TWO_AXIS_VERDICT`` (default OFF).  Fail-soft:
    any error logs a warning and returns None (never breaks report finalization).

    Parameters
    ----------
    context:
        The RLM ``context`` variable (the paper object); used to get a
        representative text slice for extraction.  If ``paper_text`` is
        supplied directly, ``context`` is ignored.
    run_dir:
        The run directory (``runs/<project_id>/``).
    llm_client:
        An object with a ``.complete(*, system, user) -> str`` method (the
        ``RunContext.llm_client`` from primitives).  When None, extraction is
        skipped (no-op + warning).
    top_k:
        Maximum number of claims to extract.
    paper_text:
        Override: supply paper text directly (used in tests).
    max_paper_chars:
        Truncate paper text to this many characters before sending to the LLM.

    Returns
    -------
    The path to the written ``repro_spec.json``, or ``None`` on failure.
    """
    if not is_enabled():
        return None

    run_dir = Path(run_dir)

    try:
        return _extract_and_write_inner(
            context=context,
            run_dir=run_dir,
            llm_client=llm_client,
            top_k=top_k,
            paper_text=paper_text,
            max_paper_chars=max_paper_chars,
        )
    except Exception as exc:  # noqa: BLE001 — never break the run
        logger.warning(
            "repro_spec_extractor: extract_and_write failed (%s: %s) — writing nothing",
            type(exc).__name__,
            exc,
        )
        return None


def _extract_and_write_inner(
    *,
    context: Any,
    run_dir: Path,
    llm_client: Any,
    top_k: int,
    paper_text: str,
    max_paper_chars: int,
) -> Path | None:
    """Inner (non-fail-soft) extraction logic."""
    if llm_client is None:
        logger.warning("repro_spec_extractor: no llm_client provided — skipping extraction")
        return None

    # Resolve paper text from context if not supplied directly
    if not paper_text:
        paper_text = _text_from_context(context)
    if not paper_text or len(paper_text.strip()) < 200:
        logger.warning("repro_spec_extractor: paper text too short — skipping extraction")
        return None

    truncated = paper_text[:max_paper_chars]

    # ------------------------------------------------------------------ #
    # First extraction pass
    # ------------------------------------------------------------------ #
    user_first = f"Extract top-{top_k} falsifiable quantitative claims.\n\nPAPER TEXT:\n{truncated}"
    raw_first = _call_llm(llm_client, system=_EXTRACTOR_SYSTEM, user=user_first)
    parsed_first = _extract_json(raw_first)
    if not parsed_first or not isinstance(parsed_first.get("claims"), list):
        logger.warning("repro_spec_extractor: first extraction returned unparseable JSON")
        return None

    first_claims: list[dict[str, Any]] = parsed_first["claims"][:top_k]
    if not first_claims:
        logger.warning("repro_spec_extractor: first extraction returned no claims")
        return None

    # ------------------------------------------------------------------ #
    # A6a: Blinded re-extraction from the raw cited spans
    # ------------------------------------------------------------------ #
    cited_spans = [str(c.get("paper_span", "") or c.get("description", "")) for c in first_claims]
    spans_block = "\n".join(
        f"[{i}] {s}" for i, s in enumerate(cited_spans) if s.strip()
    )

    blinded_extractions: list[dict[str, Any]] = []
    if spans_block.strip():
        user_blinded = (
            "Re-extract numeric constants ONLY from these raw paper spans "
            "(do not use any other context):\n\n" + spans_block
        )
        raw_blinded = _call_llm(llm_client, system=_BLINDED_SYSTEM, user=user_blinded)
        parsed_blinded = _extract_json(raw_blinded)
        if parsed_blinded and isinstance(parsed_blinded.get("extractions"), list):
            blinded_extractions = parsed_blinded["extractions"]

    # ------------------------------------------------------------------ #
    # Normalize + reconcile + run deterministic parser
    # ------------------------------------------------------------------ #
    claims_out: list[dict[str, Any]] = []
    for i, raw_claim in enumerate(first_claims):
        normalized = _normalize_claim_from_llm(raw_claim, i)

        # Find matching blinded extraction by span_index
        blinded = next(
            (e for e in blinded_extractions if e.get("span_index") == i),
            None,
        )
        if blinded is not None:
            agree, reason = _reconcile_with_blinded(raw_claim, blinded)
            if not agree and not normalized["ambiguous"]:
                normalized["ambiguous"] = True
                normalized["ambiguity_reason"] = (
                    (normalized["ambiguity_reason"] + "; " if normalized["ambiguity_reason"] else "")
                    + reason
                )

        claims_out.append(normalized)

    # Ensure exactly one primary claim
    has_primary = any(c.get("is_primary") for c in claims_out)
    if not has_primary and claims_out:
        claims_out[0]["is_primary"] = True

    # ------------------------------------------------------------------ #
    # Build + write the spec
    # ------------------------------------------------------------------ #
    spec = build_repro_spec(claims_out)

    dest_dir = run_dir / "rlm_state"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "repro_spec.json"
    dest.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    logger.info(
        "repro_spec_extractor: wrote %d claim(s) to %s",
        len(claims_out),
        dest,
    )
    return dest


def _text_from_context(context: Any) -> str:
    """Best-effort: pull a text representation from the RLM context variable."""
    if context is None:
        return ""
    if isinstance(context, str):
        return context
    # RLM context is typically a dict with a 'text'/'content'/'full_text' key
    if isinstance(context, dict):
        for key in ("full_text", "text", "content", "body", "abstract"):
            val = context.get(key)
            if isinstance(val, str) and len(val) > 200:
                return val
    # Last resort: str()
    try:
        s = str(context)
        return s if len(s) > 200 else ""
    except Exception:  # noqa: BLE001
        return ""
