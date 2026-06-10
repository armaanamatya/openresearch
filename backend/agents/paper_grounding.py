"""Paper grounding postflight — reject hallucinated dataset/method names."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "are", "was", "were", "be", "been", "by", "from",
    "we", "our", "this", "that", "it", "its", "as", "not", "also", "both",
    "all", "each", "which", "have", "has", "can", "will", "their", "than",
    "more", "such", "into", "used", "using", "show", "shows", "paper",
})


@dataclass(frozen=True)
class GroundingViolation:
    field: str         # "datasets" | "methods" | "metrics"
    value: str         # the unfounded name
    suggestion: str    # human-readable next step


def _normalize(s: str) -> str:
    """Lowercase, replace underscores/hyphens with spaces, collapse whitespace."""
    return re.sub(r"\s+", " ", s.lower().replace("_", " ").replace("-", " ")).strip()


def _token_overlap(name: str, paper_text: str) -> float:
    """Jaccard overlap of normalized tokens between name and paper_text.

    Only non-stopword tokens from name are checked for presence in paper_text.
    Returns ratio of matching tokens / total key-term tokens in name.
    For single-word names, returns 1.0 if literal normalized name is in
    normalized paper_text, 0.0 otherwise.
    """
    norm_name = _normalize(name)
    norm_paper = _normalize(paper_text)
    tokens = [t for t in norm_name.split() if t not in _STOPWORDS]
    if not tokens:
        # All stopwords — treat as grounded (don't flag)
        return 1.0
    if len(tokens) == 1:
        return 1.0 if tokens[0] in norm_paper else 0.0
    matched = sum(1 for t in tokens if t in norm_paper)
    return matched / len(tokens)


# A groundable NAME is short and single-line. Anything longer / multi-line is
# LLM prose or a serialized structure — ungrepable against the paper text and
# guaranteed to fire a false "not found in paper text" warning (2026-06-08:
# values like "Based on the provided excerpt:\n\n**ML ..." and
# "[{'name': 'CIFAR-10', 'source': 'torchvision', ...}]" flooded the warnings
# on every run while real contamination would have drowned in the noise).
_MAX_NAME_CHARS = 80


def _extract_name(value: object) -> str | None:
    """Coerce a claim-map entry to a groundable name string, or None to skip.

    Handles the shapes agents actually emit: plain names, dicts carrying a
    name-like key, serialized dict/list literals, and free prose (skipped).
    """
    if isinstance(value, dict):
        for key in ("name", "id", "dataset", "title"):
            inner = value.get(key)
            if isinstance(inner, str) and inner.strip():
                value = inner
                break
        else:
            return None
    if not isinstance(value, str) or not value.strip():
        return None
    s = value.strip()
    # Serialized structure ("[{...}]" / "{...}") → recover the name field.
    if s.startswith(("[", "{")):
        import ast as _ast
        import json as _json
        for parser in (_json.loads, _ast.literal_eval):
            try:
                parsed = parser(s)
            except Exception:  # noqa: BLE001 — try the next parser
                continue
            if isinstance(parsed, list) and parsed:
                parsed = parsed[0]
            if isinstance(parsed, dict):
                return _extract_name(parsed)
            break
        return None
    if "\n" in s or len(s) > _MAX_NAME_CHARS:
        return None  # prose, not a name — unverifiable, never warn on it
    return s


def _is_grounded(value: str, paper_text: str, threshold: float) -> bool:
    """Return True if value appears to be grounded in paper_text."""
    norm_value = _normalize(value)
    norm_paper = _normalize(paper_text)

    # Fast path: literal substring match (case/whitespace insensitive)
    if norm_value in norm_paper:
        return True

    # Token-overlap for multi-word names
    tokens = norm_value.split()
    if len(tokens) > 1:
        overlap = _token_overlap(value, paper_text)
        return overlap >= threshold

    # Single-word: require literal presence (already checked above)
    return False


def assert_paper_grounded(
    paper_claim_map: dict,
    paper_text: str,
    *,
    min_overlap_threshold: float = 0.5,
) -> list[GroundingViolation]:
    """Validate that every dataset/method/metric in paper_claim_map appears in paper_text.

    Checks:
      - paper_claim_map["datasets"]: list[str]
      - paper_claim_map["claims"][i]["dataset"], ["method"], ["metric"]
      - paper_claim_map["core_contribution"] — token overlap on key terms only

    Token-overlap rule: normalize both, split on whitespace, compute ratio of
    key terms (non-stopwords) from the claim value that appear in paper_text.
    For single-word names, require literal substring presence.

    Default threshold 0.5 is intentionally loose — false-positive rejections
    are worse than missed contamination.
    """
    violations: list[GroundingViolation] = []

    # Check top-level datasets list (dict entries / serialized literals are
    # reduced to their name field; prose entries are unverifiable → skipped)
    for ds in (paper_claim_map.get("datasets") or []):
        name = _extract_name(ds)
        if name is None:
            continue
        if not _is_grounded(name, paper_text, min_overlap_threshold):
            violations.append(GroundingViolation(
                field="datasets",
                value=name,
                suggestion=(
                    f"'{name}' not found in paper text — verify the dataset name "
                    f"matches the paper's wording exactly"
                ),
            ))

    # Check per-claim fields: dataset, method, metric
    for claim in (paper_claim_map.get("claims") or []):
        if not isinstance(claim, dict):
            continue
        for field_name in ("dataset", "method", "metric"):
            name = _extract_name(claim.get(field_name))
            if name is None:
                continue
            if not _is_grounded(name, paper_text, min_overlap_threshold):
                violations.append(GroundingViolation(
                    field=field_name,
                    value=name,
                    suggestion=(
                        f"'{name}' (claim.{field_name}) not found in paper text — "
                        f"check if this name is from a different paper"
                    ),
                ))

    # Check core_contribution — key-term overlap only (skip stopwords)
    core = paper_claim_map.get("core_contribution") or ""
    if isinstance(core, str) and core.strip():
        norm_paper = _normalize(paper_text)
        key_terms = [
            t for t in _normalize(core).split()
            if t not in _STOPWORDS and len(t) > 2
        ]
        if key_terms:
            matched = sum(1 for t in key_terms if t in norm_paper)
            ratio = matched / len(key_terms)
            if ratio == 0.0:
                violations.append(GroundingViolation(
                    field="core_contribution",
                    value=core[:200],
                    suggestion=(
                        "None of the key terms from core_contribution appear in paper text — "
                        "the claim map may be from a different paper"
                    ),
                ))

    return violations
