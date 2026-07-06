"""Deterministic paper -> skill-library matcher (Release-1, part 5.B).

Scores the vendored skill catalog (:mod:`backend.agents.rlm.skill_catalog`)
against a paper's claim-map and environment-spec fields and returns a small,
deterministic set of candidate skill names plus a coarse domain label.

``claim_map``/``environment_spec`` are taken as plain ``Mapping[str, Any]``
(i.e. ``PaperClaimMap.model_dump()`` / ``EnvironmentSpec.model_dump()``
output, per ``backend/agents/schemas.py``) rather than pydantic instances, so
callers — and tests — can build fixtures without pydantic in the loop.

Pure library: zero LLM calls, zero network access, no side effects at import
time. Fail-soft — a malformed or degenerate input never raises; it degrades
to the empty match.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from backend.agents.rlm.skill_catalog import SkillMeta

logger = logging.getLogger(__name__)

# ``PaperClaimMap`` fields consulted for matching (backend/agents/schemas.py:88).
# ``environment_spec["framework"]`` (schemas.py:362) is folded in separately
# since it lives on a different schema (``EnvironmentSpec``).
_CLAIM_MAP_FIELDS = (
    "core_contribution",
    "claims",
    "model_architecture",
    "training_recipe",
    "datasets",
    "metrics",
    "evaluation_protocol",
)

_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")

# Standard function-word stopword list (articles/prepositions/conjunctions/
# common auxiliary verbs) — deliberately NOT domain words like "model" or
# "method": those carry real matching signal (e.g. the vendored
# "model-merging" / "model-pruning" skill names), so filtering them out would
# throw away signal the matcher needs.
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "with",
    "using", "use", "used", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "we", "our", "it", "its", "as", "by",
    "at", "from", "via", "into", "over", "under", "not", "no", "than",
    "such", "which", "their", "can", "will", "also", "each", "per",
})

# Coarse domain keyword table (presence-only, not frequency-weighted —
# deterministic). Each domain's score is the count of its distinct keywords
# present in the paper token set; the highest-scoring ML domain wins, ties
# broken by the fixed priority order below. A non-ML domain (physics /
# chemistry / biology) is reported ONLY when it strictly DOMINATES every ML
# domain's score — this harness reproduces ML papers almost exclusively, so
# ML is the tie-favored default; with no signal anywhere the result is
# "ml-other".
_DOMAIN_KEYWORDS: dict[str, frozenset[str]] = {
    "ml-rl": frozenset({"rl", "grpo", "ppo", "reward", "policy"}),
    "ml-inference": frozenset({"serving", "inference", "quantization", "vllm"}),
    "ml-vision": frozenset({"image", "vision", "cnn", "segmentation", "detection"}),
    "ml-nlp": frozenset({"language", "translation", "token"}),
    "ml-interp": frozenset({"interpretability", "probing", "circuit"}),
}
_ML_DOMAIN_PRIORITY = ("ml-rl", "ml-vision", "ml-nlp", "ml-inference", "ml-interp")

_NON_ML_DOMAIN_KEYWORDS: dict[str, frozenset[str]] = {
    "physics": frozenset({"physics", "quantum", "particle", "relativity"}),
    "chemistry": frozenset({"chemistry", "molecule", "molecular", "reaction"}),
    "biology": frozenset({"biology", "genome", "protein", "organism"}),
}
_NON_ML_DOMAIN_PRIORITY = ("physics", "chemistry", "biology")

_MAX_REASON_TOKENS = 5


@dataclass(frozen=True)
class SkillMatch:
    """Result of matching a paper against the skill catalog."""

    domain: str
    skill_names: tuple[str, ...]
    reasons: tuple[str, ...]


_EMPTY_MATCH = SkillMatch(domain="ml-other", skill_names=(), reasons=())


def match_skills(
    claim_map: Mapping[str, Any],
    environment_spec: Mapping[str, Any],
    catalog: Mapping[str, SkillMeta],
    *,
    top_k: int = 8,
) -> SkillMatch:
    """Match a paper's claim map + environment spec against the skill catalog.

    Returns up to ``top_k`` skill names (ties broken by name, for
    determinism) whose ``{name tokens} ∪ {tag tokens}`` overlap the paper's
    token set, paired index-wise with a ``reasons`` entry naming the top
    overlapping tokens for that skill. Never raises — any failure (a
    degenerate/malformed ``claim_map``, an empty catalog, ...) degrades to
    ``SkillMatch(domain="ml-other", skill_names=(), reasons=())``.
    """
    try:
        return _match_skills(claim_map, environment_spec, catalog, top_k=top_k)
    except Exception:
        logger.warning(
            "skill_matcher: match_skills failed; returning the empty match",
            exc_info=True,
        )
        return _EMPTY_MATCH


def _match_skills(
    claim_map: Mapping[str, Any],
    environment_spec: Mapping[str, Any],
    catalog: Mapping[str, SkillMeta],
    *,
    top_k: int,
) -> SkillMatch:
    paper_tokens = _paper_token_set(claim_map, environment_spec)
    domain = _classify_domain(paper_tokens)

    if not paper_tokens or not catalog:
        return SkillMatch(domain=domain, skill_names=(), reasons=())

    scored: list[tuple[str, int, tuple[str, ...]]] = []
    for name, meta in catalog.items():
        overlap = paper_tokens & _skill_token_set(meta)
        if not overlap:
            continue
        reason_tokens = tuple(sorted(overlap)[:_MAX_REASON_TOKENS])
        scored.append((name, len(overlap), reason_tokens))

    scored.sort(key=lambda row: (-row[1], row[0]))
    top = scored[:top_k]
    skill_names = tuple(name for name, _score, _reason in top)
    reasons = tuple(", ".join(reason) for _name, _score, reason in top)
    return SkillMatch(domain=domain, skill_names=skill_names, reasons=reasons)


def _flatten_to_strings(value: Any) -> list[str]:
    """Recursively flatten a claim-map-ish value (str / list / dict / nested)
    into a flat list of strings.

    Dict KEYS are schema field names (generic and low-signal, e.g.
    ``"optimizer"`` on ``TrainingRecipe``) — only VALUES are flattened, since
    those carry the paper-specific content. Never raises: an unrecognized
    scalar (number/bool/etc.) is stringified rather than dropped.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        out: list[str] = []
        for v in value.values():
            out.extend(_flatten_to_strings(v))
        return out
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            out.extend(_flatten_to_strings(item))
        return out
    return [str(value)]


def _tokenize(text: str) -> set[str]:
    if not text:
        return set()
    raw = _TOKEN_SPLIT_RE.split(text.lower())
    return {t for t in raw if len(t) > 1 and t not in _STOPWORDS}


def _paper_token_set(
    claim_map: Mapping[str, Any], environment_spec: Mapping[str, Any]
) -> set[str]:
    tokens: set[str] = set()
    for field in _CLAIM_MAP_FIELDS:
        if field not in claim_map:
            continue
        for s in _flatten_to_strings(claim_map[field]):
            tokens |= _tokenize(s)

    framework = (
        environment_spec.get("framework") if isinstance(environment_spec, Mapping) else None
    )
    for s in _flatten_to_strings(framework):
        tokens |= _tokenize(s)
    return tokens


def _skill_token_set(meta: SkillMeta) -> set[str]:
    tokens = _tokenize(meta.name)
    for tag in meta.tags:
        tokens |= _tokenize(tag)
    return tokens


def _best_by_priority(scores: Mapping[str, int], priority: tuple[str, ...]) -> tuple[str, int]:
    """Return ``(domain, score)`` for the highest-scoring domain in ``scores``,
    ties broken by earliest position in ``priority``."""
    best_domain = priority[0]
    best_score = scores.get(best_domain, 0)
    for domain in priority[1:]:
        score = scores.get(domain, 0)
        if score > best_score:
            best_domain, best_score = domain, score
    return best_domain, best_score


def _classify_domain(tokens: set[str]) -> str:
    if not tokens:
        return "ml-other"

    ml_scores = {d: len(tokens & kws) for d, kws in _DOMAIN_KEYWORDS.items()}
    best_ml, best_ml_score = _best_by_priority(ml_scores, _ML_DOMAIN_PRIORITY)

    non_ml_scores = {d: len(tokens & kws) for d, kws in _NON_ML_DOMAIN_KEYWORDS.items()}
    best_non_ml, best_non_ml_score = _best_by_priority(non_ml_scores, _NON_ML_DOMAIN_PRIORITY)

    if best_non_ml_score > 0 and best_non_ml_score > best_ml_score:
        return best_non_ml
    if best_ml_score > 0:
        return best_ml
    return "ml-other"
