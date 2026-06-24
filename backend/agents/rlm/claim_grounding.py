"""
Claim-grounding engine — §4.0b of 2026-06-20-pre-gpu-code-review-and-report-validation-design.md.

Pure stdlib module.  Extracts result claims from text and checks them against
on-disk measured values.  Used by the report-claim gate (A), the in-loop claim
refusal (B), and the validator predicate (4.5).

Key design:
  - Result claims are numbers found near a RESULT_TERM word in text.
  - Hyperparameter/config numbers (adjacent to a _CONFIG_TERMS token) are EXCLUDED
    — they are not achieved-result claims (fixes codex-7 identity mismatch).
  - Matching is identity-based (term synonym map), not bare-float: a measured
    loss=0.84 does NOT ground a claimed success_rate=0.84.
  - Empty measured evidence → "unverifiable" (no ungrounded claims emitted) so that
    the evidence_gate — not this module — owns the no-evidence case.
  - Fail-soft throughout: any exception returns [] or empty dicts.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from backend.agents.rlm.zero_metrics_detection import _CONFIG_TERMS


# ---------------------------------------------------------------------------
# Claim dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Claim:
    """A numeric result claim extracted from text."""
    value: float
    term: str    # normalized result word, e.g. "success_rate"
    context: str # ~40-char snippet around the match


# ---------------------------------------------------------------------------
# Result term vocabulary
# ---------------------------------------------------------------------------

# Words that signal an achieved-result claim.  The matched word becomes Claim.term.
# Conservative set: high-precision metric words only.  The ambiguous English words
# "score"/"return"/"win" are deliberately EXCLUDED — they false-positive in prose, and
# A has teeth (a spurious ungrounded claim would wrongly cap a good run's verdict).
_RESULT_TERMS_RE = re.compile(
    r"\b(accuracy|success_rate|success|reward|f1|precision|recall|em|exact_match)\b",
    re.IGNORECASE,
)

# Rate-type terms whose value must be a plausible rate: a [0,1.2] fraction or a
# (1.2,100] percentage.  An integer like "success after 150 steps" is implausible as a
# rate and is rejected — this stops step/epoch/batch counts from becoming result claims.
_RATE_TERMS: frozenset[str] = frozenset({
    "accuracy", "success", "success_rate", "f1", "precision", "recall", "em", "exact_match",
})

# Count/duration words near a number signal a non-result quantity (step/epoch/batch
# counts, durations, GPU counts); a number adjacent to one is excluded.
_COUNT_TERMS: frozenset[str] = frozenset({
    "step", "steps", "epoch", "epochs", "iter", "iters", "iteration", "iterations",
    "batch", "sample", "samples", "task", "tasks", "episode", "episodes",
    "rollout", "rollouts", "token", "tokens", "second", "seconds", "minute", "minutes",
    "hour", "hours", "day", "days", "card", "cards", "example", "examples",
    "trajectory", "trajectories",
})

# Number pattern: matches floats/ints with optional trailing %.
# Two-step approach: first a "skip" pattern eats scientific-notation tokens so
# they are never captured as bare integers; then a capture group picks up the
# remaining plain numbers.  In practice we do this by checking in the loop
# whether the raw match position falls inside a scientific-notation span —
# see _SCINOTATION_RE below.
# Group 1: numeric part; Group 2: optional '%'.
_NUMBER_RE = re.compile(r"\b(\d+(?:\.\d+)?)(%?)")

# Pattern that matches full scientific-notation literals, used to filter out
# their digit components from _NUMBER_RE matches (e.g. "1e-5" → skip "1" and "5").
_SCINOTATION_RE = re.compile(r"\b\d+(?:\.\d+)?[eE][+\-]?\d+")


# Window (characters) within which a number is considered "near" a result term.
_WINDOW = 40


# Combined config+count label regex; compound config terms ("learning_rate") match
# with an underscore OR a space ("learning rate").  A number binds to its NEAREST label
# (result term vs. one of these); if a config/count word is at least as near, the number
# belongs to it and is not a result claim.
_NONRESULT_LABEL_RE = re.compile(
    r"\b(?:"
    + "|".join(
        sorted(
            (re.escape(t).replace("_", "[ _]") for t in (_CONFIG_TERMS | _COUNT_TERMS)),
            key=len,
            reverse=True,
        )
    )
    + r")\b",
    re.IGNORECASE,
)


def _nearest_span_dist(pos: int, spans: list[tuple[int, int]]) -> int | None:
    """Smallest gap from `pos` to any (start, end) span; None when there are no spans."""
    best: int | None = None
    for ts, te in spans:
        dist = ts - pos if pos < ts else (pos - te if pos > te else 0)
        if best is None or dist < best:
            best = dist
    return best


def _scinotation_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) spans of scientific-notation literals in `text`."""
    return [(m.start(), m.end()) for m in _SCINOTATION_RE.finditer(text)]


def _in_scinotation_span(pos: int, spans: list[tuple[int, int]]) -> bool:
    """Return True iff `pos` falls inside any scientific-notation span."""
    return any(s <= pos < e for s, e in spans)


def extract_result_claims(text: str) -> list[Claim]:
    """Find result claims in `text`.

    Nearest-label-wins: each number binds to its NEAREST label (a result term, or a
    config/count word).  It becomes a claim only when the nearest label is a result
    term — so "success 0.84 and accuracy 0.80" yields success=0.84 AND accuracy=0.80
    (no cross-assignment), and "learning_rate=1e-5, accuracy 0.84" yields only
    accuracy=0.84 (the 1e-5 belongs to learning_rate).  A number is also dropped when it
    is part of a scientific-notation literal or — for a rate-type term — implausible as
    a rate (outside [0,1.2] and (1.2,100]).  Percentages normalize (84% → 0.84).
    Fail-soft → [].
    """
    try:
        if not isinstance(text, str):
            return []
        result_spans = [
            (m.start(), m.end(), m.group(0).lower())
            for m in _RESULT_TERMS_RE.finditer(text)
        ]
        if not result_spans:
            return []
        nonresult_spans = [(m.start(), m.end()) for m in _NONRESULT_LABEL_RE.finditer(text)]
        sci_spans = _scinotation_spans(text)
        claims: list[Claim] = []
        for num_match in _NUMBER_RE.finditer(text):
            abs_pos = num_match.start()
            if _in_scinotation_span(abs_pos, sci_spans):
                continue
            # Nearest result term to this number.
            best_term: str | None = None
            best_dist = _WINDOW + 1
            for ts, te, term in result_spans:
                dist = ts - abs_pos if abs_pos < ts else (abs_pos - te if abs_pos > te else 0)
                if dist < best_dist:
                    best_dist = dist
                    best_term = term
            if best_term is None or best_dist > _WINDOW:
                continue
            # If a config/count label is at least as near, the number belongs to it.
            nr_dist = _nearest_span_dist(abs_pos, nonresult_spans)
            if nr_dist is not None and nr_dist <= best_dist:
                continue
            value = float(num_match.group(1))
            is_percent = num_match.group(2) == "%"
            if is_percent:
                value = value / 100.0
            # Rate-term plausibility: a fraction (<=1.2) or, when not already a percent,
            # a (1.2,100] value to be tried as a percentage downstream.
            if best_term in _RATE_TERMS and not (
                0.0 <= value <= 1.2 or (not is_percent and 1.2 < value <= 100.0)
            ):
                continue
            ctx_start = max(0, abs_pos - 20)
            ctx_end = min(len(text), abs_pos + 20)
            context = text[ctx_start:ctx_end].replace("\n", " ")
            claims.append(Claim(value=value, term=best_term, context=context))
        return claims
    except Exception:  # noqa: BLE001
        return []


# ---------------------------------------------------------------------------
# Measured value flattening
# ---------------------------------------------------------------------------

def flatten_measured_values(project_dir: Path) -> list[tuple[str, float]]:
    """Read code/metrics.json and every outputs/**/metrics.json under project_dir.

    Returns (term, value) pairs where `term` is the lowercased last segment of
    the leaf key path (e.g. per_model.model.env.success_rate → "success_rate").
    Applies the same result-only exclusion as normalize_metric_values (via the
    zero_metrics_detection normalizer on the top-level dict, plus leaf-key
    extraction here).

    Fail-soft → [].
    """
    try:
        results: list[tuple[str, float]] = []
        candidates: list[Path] = []
        main_metrics = project_dir / "code" / "metrics.json"
        if main_metrics.exists():
            candidates.append(main_metrics)
        outputs_dir = project_dir / "code" / "outputs"
        if outputs_dir.is_dir():
            candidates.extend(outputs_dir.rglob("metrics.json"))

        for metrics_path in candidates:
            try:
                with open(metrics_path, encoding="utf-8") as f:
                    data = json.load(f)
                _collect_leaves(data, results)
            except Exception:  # noqa: BLE001
                continue
        return results
    except Exception:  # noqa: BLE001
        return []


def _collect_leaves(
    obj: object,
    out: list[tuple[str, float]],
    last_key: str | None = None,
) -> None:
    """Recursively collect (last_segment, float_value) for result-claiming leaves."""
    from backend.agents.rlm.zero_metrics_detection import _is_excluded_key  # local to avoid circular
    if isinstance(obj, dict):
        for k, v in obj.items():
            if _is_excluded_key(k):
                continue
            _collect_leaves(v, out, k)
    elif isinstance(obj, list):
        for item in obj:
            _collect_leaves(item, out, last_key)
    else:
        if last_key is not None and _is_excluded_key(last_key):
            return
        # Coerce to float.
        if isinstance(obj, bool):
            return
        if isinstance(obj, (int, float)):
            fv = float(obj)
        elif isinstance(obj, str):
            try:
                fv = float(obj)
            except (ValueError, TypeError):
                return
        else:
            return
        if last_key is not None:
            # The full leaf key IS the metric identity (e.g. "success_rate",
            # "accuracy_avg") — canonicalised downstream by _SYNONYM_MAP.  Do NOT
            # split on "_" (that turned "success_rate" -> "rate", breaking identity
            # matching so a measured success_rate could not ground a success_rate
            # claim — codex Area-2 false positive).
            term = last_key.lower()
            out.append((term, fv))


# ---------------------------------------------------------------------------
# Synonym map for claim↔measured term matching
# ---------------------------------------------------------------------------

# Maps multiple surface forms to a single canonical metric name.
# Keys are lowercased; values are canonical names.
_SYNONYM_MAP: dict[str, str] = {
    "success": "success",
    "success_rate": "success",
    "success_avg": "success",
    "em": "exact_match",
    "exact_match": "exact_match",
    "acc": "accuracy",
    "accuracy": "accuracy",
    "accuracy_avg": "accuracy",
    "f1": "f1",
    "f1_avg": "f1",
    "reward": "reward",
    "mean_reward": "reward",
    "return": "return",
    "mean_return": "return",
    "score": "score",
    "win": "win",
    "precision": "precision",
    "recall": "recall",
}


def _canonical(term: str) -> str:
    """Map a term to its canonical metric name; falls back to the term itself."""
    return _SYNONYM_MAP.get(term.lower(), term.lower())


# ---------------------------------------------------------------------------
# Grounding check
# ---------------------------------------------------------------------------

def check_claims_grounded(
    claims: list[Claim],
    measured: list[tuple[str, float]],
    *,
    rel_tol: float = 0.05,
) -> dict:
    """Check which claims are grounded by measured values.

    Returns {"grounded": [Claim, ...], "ungrounded": [Claim, ...]}.

    A claim is grounded iff some measured (term', value') satisfies:
      - _canonical(claim.term) == _canonical(term')  (identity, not bare float)
      - abs(claim.value - value') <= rel_tol * max(abs(value'), 1e-9)
        OR (treating claim as a percentage) abs(claim.value/100 - value') within tol

    If `measured` is empty → return {"grounded": [], "ungrounded": []} (unverifiable,
    NOT ungrounded — evidence_gate owns the no-evidence case).

    Fail-soft → {"grounded": [], "ungrounded": []}.
    """
    try:
        if not measured:
            return {"grounded": [], "ungrounded": []}

        grounded: list[Claim] = []
        ungrounded: list[Claim] = []

        for claim in claims:
            canon_claim = _canonical(claim.term)
            matched = False
            for mterm, mval in measured:
                if _canonical(mterm) != canon_claim:
                    continue
                # Try direct match and percent→fraction match.
                if _within_tol(claim.value, mval, rel_tol):
                    matched = True
                    break
                if _within_tol(claim.value / 100.0, mval, rel_tol):
                    matched = True
                    break
            if matched:
                grounded.append(claim)
            else:
                ungrounded.append(claim)

        return {"grounded": grounded, "ungrounded": ungrounded}
    except Exception:  # noqa: BLE001
        return {"grounded": [], "ungrounded": []}


def _within_tol(a: float, b: float, rel_tol: float) -> bool:
    """True iff a and b are within rel_tol relative tolerance."""
    denom = max(abs(b), 1e-9)
    return abs(a - b) <= rel_tol * denom
