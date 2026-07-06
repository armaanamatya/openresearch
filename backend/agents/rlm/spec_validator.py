"""Spec validator — rubric-vs-paper pre-loop validation (a sibling of
``external_validator.py``'s Tier-2 metrics-vs-disk adversarial panel).

Fires ONCE, between rubric generation and the RLM loop — a "before any GPU
spend" gate. An LLM panel is asked to *point at suspicions* in the
auto-generated rubric by naming typed predicates against the paper's own
full text; every veto rests on a HARNESS-side deterministic machine-check of
the named predicate, never the LLM's opinion. This closes the "the
auto-rubric hallucinated a leaf" gap that no runtime guard catches, since a
bad rubric leaf poisons every downstream ``verify_against_rubric`` call for
the rest of the run.

Min-aggregation: a leaf is flagged iff ANY panelist's predicate
machine-verifies as violated. No majority vote (consensus collapse) — mirrors
``external_validator.run_validation_panel``'s aggregation rule exactly.

Verdict store: persisted to ``rlm_state/spec_validation_verdict.json``
(atomic), keyed by a rubric fingerprint. A stale verdict (fingerprint
mismatch — the rubric changed since the verdict was minted) is ignored.

Corpus isolation: this module reads the FULL paper text server-side to run
its machine-checks and to build the panel prompt, but the
returned/persisted :class:`SpecValidatorVerdict` never carries paper text —
only leaf ids, enums, counts, and model names.

Default-OFF: ``OPENRESEARCH_SPEC_VALIDATOR`` must be set to enable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# MODULE-LEVEL import (not lazy, unlike external_validator.py's in-function
# import): callers monkeypatch `sv.sample_completions` directly (replacing
# this module's own attribute), which only takes effect if the panel runner
# below resolves the bare name `sample_completions` through THIS module's
# global namespace at call time — a lazy `from ... import sample_completions`
# inside the function would re-bind the real function on every call and
# silently ignore the monkeypatch.
from backend.agents.rlm.claim_grounding import _canonical, _within_tol, extract_result_claims
from backend.agents.rlm.grader_transport import sample_completions
from backend.agents.rlm.rubric_gen import _is_placeholder_requirement, _normalize_weights

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpecPredicateVerdict:
    """Result of a single HARNESS-side machine-check on one panelist suspicion."""

    predicate: str       # hallucinated_leaf | wrong_target | placeholder_leaf | missing_key_claim
    leaf_id: str          # the rubric leaf's id (or a panel-invented gap label for missing_key_claim)
    violated: bool        # True = machine-verified violated (NOT the LLM's opinion)
    detail: str           # human-readable explanation


@dataclass(frozen=True)
class SpecValidatorVerdict:
    """Aggregated verdict from the full panel."""

    status: str                            # clean | flagged | unavailable
    flagged_leaves: list[str]              # leaf_ids confirmed (min-aggregation)
    predicates: list[SpecPredicateVerdict]
    panel_models: list[str]
    separation: str                        # independent | weak | degraded | unavailable (caller-supplied)
    rubric_fingerprint: str


# ---------------------------------------------------------------------------
# Feature flags + config
# ---------------------------------------------------------------------------

_ENABLED_VALUES = frozenset({"1", "true", "yes", "on"})


def spec_validator_enabled() -> bool:
    """True iff ``OPENRESEARCH_SPEC_VALIDATOR`` opts the panel ON.

    Default-OFF: with the var unset or falsey the module is inert and the
    caller gets ``status="unavailable"``.
    """
    return os.environ.get("OPENRESEARCH_SPEC_VALIDATOR", "").strip().lower() in _ENABLED_VALUES


def spec_validator_panel_n() -> int:
    """Number of independent panel completions to request.

    Reads ``OPENRESEARCH_SPEC_VALIDATOR_PANEL_N``; defaults to 2.
    """
    raw = os.environ.get("OPENRESEARCH_SPEC_VALIDATOR_PANEL_N", "").strip()
    try:
        n = int(raw)
        return max(1, n)
    except (ValueError, TypeError):
        return 2


def spec_validator_block_enabled() -> bool:
    """True iff a CONFIRMED veto should actually mutate the rubric (``apply_block``).

    Kept as a separate gate from ``spec_validator_enabled`` so an operator can
    run the panel in report-only mode (flag suspicions, never edit the
    rubric) before trusting it to block.
    """
    return os.environ.get("OPENRESEARCH_SPEC_VALIDATOR_BLOCK", "").strip().lower() in _ENABLED_VALUES


# ---------------------------------------------------------------------------
# Rubric fingerprint
# ---------------------------------------------------------------------------


def rubric_fingerprint(rubric: dict) -> str:
    """Stable sha256 fingerprint of the rubric dict for the verdict store.

    Canonical JSON serialisation (sorted keys, default=str) so key-order
    differences never change the fingerprint. Fails soft to a hash of "{}"
    if rubric is not a dict.
    """
    try:
        canonical = json.dumps(
            rubric if isinstance(rubric, dict) else {},
            sort_keys=True,
            default=str,
        )
    except Exception:  # noqa: BLE001
        canonical = "{}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Token-overlap helper (pure, local — no existing helper fits this shape)
# ---------------------------------------------------------------------------
#
# NOTE: backend.agents.paper_grounding also defines a private `_normalize` /
# `_token_overlap` pair, but that helper computes overlap for a SHORT
# candidate NAME (e.g. a single dataset string from a claim map) against
# non-stopword tokens — applied to a full leaf REQUIREMENT SENTENCE it
# over-penalizes ordinary descriptive words ("Report", "success", "rate",
# "near"), which are not stopwords but also carry no paper-specific citation
# signal. A grounded leaf like "Report ALFWorld success rate near 84.4" would
# score ~0.33 overlap under that helper and get wrongly flagged. This module
# therefore defines its OWN small helper, scoped to "distinctive cited
# terms/numbers" (numbers, and CamelCase/acronym-style names) rather than
# every non-stopword content word. Do not conflate the two.

_NUMBER_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?")
_WORD_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")

_HALLUCINATION_OVERLAP_FLOOR = 0.5


def _normalize(text: str) -> str:
    """Lowercase, replace underscores/hyphens with spaces, collapse whitespace.

    Deliberately preserves digits (unlike ``failure_attribution._normalize``,
    which scrubs numbers/paths/hex for traceback-signature dedup — using that
    normalizer here would destroy the exact numeric literals this module
    needs to match, e.g. "84.4").
    """
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.lower().replace("_", " ").replace("-", " ")).strip()


def _has_internal_capital(word: str) -> bool:
    """True iff `word` has an uppercase letter at position >= 1.

    Distinguishes a domain-specific CamelCase/acronym term (ALFWorld,
    WebShop, ImageNet, GRPO, SDAR) from an ordinary Title-case English word
    that is capitalized only because it opens a sentence (Report, Verify,
    The). Both look identical by "starts with a capital" alone; the internal
    capital is what actually signals a compound/acronym name.
    """
    return any(ch.isupper() for ch in word[1:])


def _distinctive_tokens(text: str) -> list[str]:
    """Extract the 'distinctive cited terms/numbers' from a leaf requirement.

    A token is distinctive iff it is numeric (a candidate result/target
    value) or an alphabetic token with an internal capital letter (a
    candidate dataset/method name written in CamelCase or as an acronym).
    Ordinary English words are excluded, since they carry no paper-specific
    citation signal and would otherwise swamp the overlap ratio with generic
    rubric phrasing ("Report", "verify", "correctly").

    Returns [] when the leaf cites nothing distinctive (an abstract/
    procedural requirement with no dataset/method/number) — the caller
    treats this as unverifiable, not hallucinated.
    """
    if not isinstance(text, str) or not text:
        return []
    tokens: list[str] = [m.group(0) for m in _NUMBER_TOKEN_RE.finditer(text)]
    tokens.extend(m.group(0) for m in _WORD_TOKEN_RE.finditer(text) if _has_internal_capital(m.group(0)))
    return tokens


def _contains_token(token: str, normalized_haystack: str) -> bool:
    """Word-boundary-anchored containment check of `token` in a haystack that
    has already been run through :func:`_normalize`. Using ``\\b`` (rather
    than plain substring ``in``) stops a short numeric token like "1" from
    matching inside an unrelated longer number ("212"); adjacent digits
    never form a word-boundary so the anchors correctly reject that case.
    """
    try:
        return re.search(r"\b" + re.escape(_normalize(token)) + r"\b", normalized_haystack) is not None
    except re.error:
        return _normalize(token) in normalized_haystack


def _token_overlap(leaf_text: str, paper_text: str) -> float:
    """Fraction of leaf_text's distinctive tokens present in paper_text.

    Returns 1.0 (grounded, fail-soft) when the leaf cites nothing distinctive
    to check — an abstract requirement is not a hallucination candidate.
    """
    tokens = _distinctive_tokens(leaf_text)
    if not tokens:
        return 1.0
    norm_paper = _normalize(paper_text)
    matched = sum(1 for t in tokens if _contains_token(t, norm_paper))
    return matched / len(tokens)


# ---------------------------------------------------------------------------
# Deterministic machine-checks (the four typed predicates)
# ---------------------------------------------------------------------------


def check_hallucinated_leaf(leaf_text: str, paper_text: str) -> bool:
    """True iff the leaf's distinctive cited terms/numbers are adequately
    grounded in paper_text (token-overlap >= 0.5). Fail-soft: any error
    returns True (healthy, no veto on uncertainty).

    Note: returns True == HEALTHY (grounded). The violated field is set to
    ``not check_hallucinated_leaf(...)``.
    """
    try:
        return _token_overlap(leaf_text, paper_text) >= _HALLUCINATION_OVERLAP_FLOOR
    except Exception:  # noqa: BLE001
        return True


_WRONG_TARGET_REL_TOL = 0.05


def _wrong_target_violated(leaf_text: str, paper_text: str) -> bool:
    """True iff a leaf claim cleanly contradicts a paper-reported number for
    the SAME canonical metric term. Only judges when BOTH the leaf and the
    paper text yield a clean ``extract_result_claims`` hit for a shared
    canonical term (reuses ``claim_grounding``) — otherwise returns False
    (fail-soft: cannot contradict what was never cleanly stated on both
    sides).
    """
    leaf_claims = extract_result_claims(leaf_text)
    if not leaf_claims:
        return False
    paper_claims = extract_result_claims(paper_text)
    if not paper_claims:
        return False
    for lc in leaf_claims:
        canon = _canonical(lc.term)
        same_term = [pc for pc in paper_claims if _canonical(pc.term) == canon]
        if not same_term:
            continue
        if not any(_within_tol(lc.value, pc.value, _WRONG_TARGET_REL_TOL) for pc in same_term):
            return True  # this leaf claim contradicts every paper-reported value for the metric
    return False  # every comparable leaf claim agreed (or none were comparable)


def check_wrong_target(leaf_text: str, paper_text: str) -> bool:
    """True iff the leaf's numeric target does not (cleanly-verifiably)
    contradict the paper. Fail-soft: any error returns True (healthy).
    """
    try:
        return not _wrong_target_violated(leaf_text, paper_text)
    except Exception:  # noqa: BLE001
        return True


def check_placeholder_leaf(leaf_text: str) -> bool:
    """True iff the leaf requirement is a concrete (non-placeholder) string —
    a belt-and-suspenders re-check of ``rubric_gen._is_placeholder_requirement``
    for a VENDORED rubric that never passed through ``_clean_categories``.
    Fail-soft: any error returns True (concrete/healthy).
    """
    try:
        return not _is_placeholder_requirement(leaf_text)
    except Exception:  # noqa: BLE001
        return True


def check_missing_key_claim(*args: Any, **kwargs: Any) -> bool:
    """Advisory-only stub: a panel-nominated coverage gap ("the paper claims
    X but no leaf checks it") is inherently an open-world absence — there is
    no deterministic way to machine-disprove it. Mirrors
    ``external_validator.check_rerun_agrees``'s deliberate-stub pattern:
    always returns False (never "healthy"), so the suspicion is always
    recorded for operator review, but :func:`apply_block` structurally
    refuses to act on it (never auto-remediated).
    """
    return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_VALID_PREDICATES = frozenset({
    "hallucinated_leaf",
    "wrong_target",
    "placeholder_leaf",
    "missing_key_claim",
})

# Only these predicate types are eligible for apply_block's drop set —
# placeholder_leaf is recorded but not auto-dropped (belt-and-suspenders
# diagnostic only); missing_key_claim is explicitly advisory-only.
_BLOCKABLE_PREDICATES = frozenset({"hallucinated_leaf", "wrong_target"})

_MAX_PAPER_CHARS_FOR_PROMPT = 48000
_MAX_LEAVES_FOR_PROMPT = 300
_MAX_LEAF_TEXT_FOR_PROMPT = 300

_ADVERSARIAL_SYSTEM = """You are an adversarial rubric auditor for a research-reproduction harness.
Your job is to FIND rubric leaves that misrepresent the paper: leaves that cite a
dataset/method/number absent from the paper (hallucinated), leaves whose numeric
target contradicts the paper's own reported number, leaves that are unfilled
templates, or key claims the paper makes that NO leaf checks for.

For each suspicion you have, output a JSON object with:
  {"predicate": "<name>", "leaf_id": "<the leaf's id>"}

Valid predicates:
- hallucinated_leaf: the leaf cites a dataset/method/number you cannot find in the paper text
- wrong_target: the leaf's numeric target contradicts the paper's own reported number for that metric
- placeholder_leaf: the leaf requirement is an unfilled template (empty parens, vague boilerplate)
- missing_key_claim: the paper makes a key claim that NO leaf in the rubric checks for
  (leaf_id may name the closest related leaf, or a short label for the missing claim)

Output a JSON array of suspicions. If you find no issues output [].
Do not explain — only output the JSON array."""

_ADVERSARIAL_USER_TEMPLATE = """Examine this rubric against the paper text and flag suspicious predicates.

RUBRIC LEAVES:
{rubric_json}

PAPER TEXT (excerpt):
{paper_excerpt}

Output ONLY a JSON array of {{predicate, leaf_id}} suspicions. Example:
[{{"predicate": "hallucinated_leaf", "leaf_id": "L2"}}, {{"predicate": "wrong_target", "leaf_id": "L5"}}]"""


def _leaf_text(leaf: dict[str, Any]) -> str:
    """Return the leaf's requirement text, tolerating either key spelling —
    ``requirement`` (this module's flat test/interface shape) or
    ``requirements`` (the nested PaperBench tree-node shape from
    ``rubric_gen.py``) — so a caller can pass either shape unmodified.
    """
    for key in ("requirement", "requirements"):
        val = leaf.get(key)
        if isinstance(val, str):
            return val
    return ""


def _parse_suspicions(text: str) -> list[dict[str, str]]:
    """Parse a panelist's response into a list of {predicate, leaf_id} dicts.

    Tolerates markdown fences, extra prose before/after the JSON array.
    Returns [] on any parse failure (fail-soft).
    """
    text = text.strip()
    for fence in ("```json", "```"):
        if fence in text:
            start = text.find(fence) + len(fence)
            end = text.rfind("```")
            if end > start:
                text = text[start:end].strip()
                break
    idx = text.find("[")
    if idx != -1:
        depth = 0
        for i, ch in enumerate(text[idx:]):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    candidate = text[idx: idx + i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, list):
                            result = []
                            for item in parsed:
                                if isinstance(item, dict):
                                    pred = str(item.get("predicate", "")).strip()
                                    lid = str(item.get("leaf_id", "")).strip()
                                    if pred and lid and pred in _VALID_PREDICATES:
                                        result.append({"predicate": pred, "leaf_id": lid})
                            return result
                    except (json.JSONDecodeError, ValueError):
                        pass
                    break
    return []


def _machine_check(
    predicate: str,
    leaf_id: str,
    leaf_lookup: dict[str, dict[str, Any]],
    paper_text: str,
) -> SpecPredicateVerdict:
    """Run the deterministic machine-check for `predicate` and return a
    SpecPredicateVerdict. ``violated=True`` means the machine-check confirmed
    the predicate is violated (the suspicion is substantiated).
    """
    if predicate == "missing_key_claim":
        check_missing_key_claim()
        return SpecPredicateVerdict(
            predicate=predicate,
            leaf_id=leaf_id,
            violated=True,
            detail=(
                "advisory: panel-nominated coverage gap — not machine-verifiable "
                "(open-world absence); recorded for operator review, never auto-remediated"
            ),
        )

    leaf = leaf_lookup.get(leaf_id)
    if not isinstance(leaf, dict):
        return SpecPredicateVerdict(
            predicate=predicate,
            leaf_id=leaf_id,
            violated=False,
            detail="leaf_id not present in the rubric — cannot verify, fail-soft",
        )
    leaf_text = _leaf_text(leaf)

    if predicate == "hallucinated_leaf":
        grounded = check_hallucinated_leaf(leaf_text, paper_text)
        detail = (
            "leaf's distinctive cited terms/numbers meet the 0.5 token-overlap "
            "floor against the paper text"
            if grounded
            else
            "leaf's distinctive cited terms/numbers fall below the 0.5 token-overlap "
            "floor against the paper text — likely hallucinated"
        )
        return SpecPredicateVerdict(predicate=predicate, leaf_id=leaf_id, violated=not grounded, detail=detail)

    elif predicate == "wrong_target":
        consistent = check_wrong_target(leaf_text, paper_text)
        detail = (
            "leaf's numeric target is consistent with (or unverifiable against) "
            "the paper's reported value"
            if consistent
            else
            "leaf's numeric target contradicts the paper's own reported value for the same metric"
        )
        return SpecPredicateVerdict(predicate=predicate, leaf_id=leaf_id, violated=not consistent, detail=detail)

    elif predicate == "placeholder_leaf":
        concrete = check_placeholder_leaf(leaf_text)
        detail = (
            "leaf requirement is concrete, not a placeholder"
            if concrete
            else
            "leaf requirement matches the empty/comma-only placeholder pattern"
        )
        return SpecPredicateVerdict(predicate=predicate, leaf_id=leaf_id, violated=not concrete, detail=detail)

    else:
        return SpecPredicateVerdict(
            predicate=predicate, leaf_id=leaf_id, violated=False,
            detail=f"unknown predicate {predicate!r} — ignored",
        )


# ---------------------------------------------------------------------------
# Core panel runner
# ---------------------------------------------------------------------------


def run_spec_validation_panel(
    *,
    spec_validator_client: Any,
    panel_models: list[str],
    rubric: dict,
    paper_text: str,
    separation: str,
) -> SpecValidatorVerdict:
    """Run the adversarial rubric-vs-paper panel and return a min-aggregated verdict.

    Algorithm:
      1. If ``spec_validator_client`` is None → return ``status="unavailable"`` immediately.
      2. Call ``sample_completions`` with the adversarial prompt, n=spec_validator_panel_n().
      3. Parse each completion into a list of ``{predicate, leaf_id}`` suspicions.
      4. For each suspicion, run the HARNESS deterministic machine-check.
      5. Min-aggregation: a leaf_id is in flagged_leaves iff ANY panelist's
         predicate machine-verifies as violated.
      6. status = "flagged" if flagged_leaves else "clean".
    """
    fp = rubric_fingerprint(rubric)

    if spec_validator_client is None:
        return SpecValidatorVerdict(
            status="unavailable",
            flagged_leaves=[],
            predicates=[],
            panel_models=list(panel_models),
            separation=separation,
            rubric_fingerprint=fp,
        )

    leaves = rubric.get("leaves") if isinstance(rubric, dict) else None
    leaf_lookup: dict[str, dict[str, Any]] = {
        lf["id"]: lf for lf in (leaves or []) if isinstance(lf, dict) and isinstance(lf.get("id"), str)
    }

    try:
        leaves_for_prompt = [
            {"id": lf.get("id"), "requirement": _leaf_text(lf)[:_MAX_LEAF_TEXT_FOR_PROMPT]}
            for lf in (leaves or [])
            if isinstance(lf, dict)
        ][:_MAX_LEAVES_FOR_PROMPT]
        rubric_json = json.dumps(leaves_for_prompt, indent=2, default=str)
    except Exception:  # noqa: BLE001
        rubric_json = "[]"

    paper_excerpt = paper_text[:_MAX_PAPER_CHARS_FOR_PROMPT] if isinstance(paper_text, str) else ""

    user_prompt = _ADVERSARIAL_USER_TEMPLATE.format(
        rubric_json=rubric_json,
        paper_excerpt=paper_excerpt,
    )

    n = spec_validator_panel_n()

    try:
        completions = sample_completions(
            spec_validator_client,
            system=_ADVERSARIAL_SYSTEM,
            user=user_prompt,
            n=n,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("spec_validator: panel call failed (%s) — unavailable", exc)
        return SpecValidatorVerdict(
            status="unavailable",
            flagged_leaves=[],
            predicates=[],
            panel_models=list(panel_models),
            separation=separation,
            rubric_fingerprint=fp,
        )

    all_predicate_verdicts: list[SpecPredicateVerdict] = []
    flagged_ids: set[str] = set()

    for completion in completions:
        suspicions = _parse_suspicions(completion)
        for sus in suspicions:
            pred = sus["predicate"]
            lid = sus["leaf_id"]
            # The FULL, uncapped paper_text is used for the machine-check
            # (only the panel PROMPT above is length-capped for context economy).
            verdict = _machine_check(pred, lid, leaf_lookup, paper_text)
            all_predicate_verdicts.append(verdict)
            if verdict.violated:
                flagged_ids.add(lid)

    flagged_leaves = sorted(flagged_ids)
    status = "flagged" if flagged_leaves else "clean"

    return SpecValidatorVerdict(
        status=status,
        flagged_leaves=flagged_leaves,
        predicates=all_predicate_verdicts,
        panel_models=list(panel_models),
        separation=separation,
        rubric_fingerprint=fp,
    )


# ---------------------------------------------------------------------------
# apply_block — mutate the rubric on CONFIRMED violations only
# ---------------------------------------------------------------------------


def apply_block(rubric: dict, verdict: SpecValidatorVerdict) -> dict:
    """Return a NEW rubric with machine-CONFIRMED ``hallucinated_leaf`` /
    ``wrong_target`` leaves dropped from ``rubric["leaves"]``, sibling
    weights renormalized via ``rubric_gen._normalize_weights``.

    NEVER drops a leaf whose only confirmed predicate is
    ``missing_key_claim`` (open-world absence, advisory-only) or
    ``placeholder_leaf`` (recorded but not auto-dropped — belt-and-suspenders
    diagnostic only, per the task interface). Never hard-aborts: any
    malformed input or internal error returns ``rubric`` unchanged.
    """
    try:
        if not isinstance(rubric, dict):
            return rubric
        leaves = rubric.get("leaves")
        if not isinstance(leaves, list) or not leaves:
            return rubric

        drop_ids = {
            pv.leaf_id
            for pv in verdict.predicates
            if pv.violated and pv.predicate in _BLOCKABLE_PREDICATES
        }
        if not drop_ids:
            return rubric

        kept = [lf for lf in leaves if not (isinstance(lf, dict) and lf.get("id") in drop_ids)]
        if len(kept) == len(leaves):
            return rubric  # nothing in drop_ids actually matched a real leaf

        normalized_weights = _normalize_weights(
            [lf.get("weight") if isinstance(lf, dict) else None for lf in kept]
        )
        new_leaves: list[Any] = []
        for lf, w in zip(kept, normalized_weights):
            if isinstance(lf, dict):
                nl = dict(lf)
                nl["weight"] = w
                new_leaves.append(nl)
            else:
                new_leaves.append(lf)

        new_rubric = dict(rubric)
        new_rubric["leaves"] = new_leaves
        return new_rubric
    except Exception as exc:  # noqa: BLE001
        logger.warning("spec_validator: apply_block failed (%s) — rubric returned unchanged", exc)
        return rubric


# ---------------------------------------------------------------------------
# Verdict store — atomic persist + load
# ---------------------------------------------------------------------------

_VERDICT_FILENAME = "spec_validation_verdict.json"


def _verdict_path(project_dir: Path) -> Path:
    """Return the canonical path for the spec validation verdict file."""
    return project_dir / "rlm_state" / _VERDICT_FILENAME


def _verdict_to_dict(verdict: SpecValidatorVerdict) -> dict[str, Any]:
    """Serialise a SpecValidatorVerdict to a JSON-compatible dict."""
    return asdict(verdict)


def _dict_to_verdict(d: dict[str, Any]) -> SpecValidatorVerdict:
    """Reconstruct a SpecValidatorVerdict from its dict form."""
    predicates = [
        SpecPredicateVerdict(**p) if isinstance(p, dict) else p
        for p in d.get("predicates", [])
    ]
    return SpecValidatorVerdict(
        status=d.get("status", "unavailable"),
        flagged_leaves=list(d.get("flagged_leaves", [])),
        predicates=predicates,
        panel_models=list(d.get("panel_models", [])),
        separation=d.get("separation", "unavailable"),
        rubric_fingerprint=d.get("rubric_fingerprint", ""),
    )


def persist_spec_verdict(project_dir: Path, verdict: SpecValidatorVerdict) -> None:
    """Atomically persist the verdict to ``rlm_state/spec_validation_verdict.json``.

    Uses a temp file + os.replace so a concurrent reader never sees a partial
    write. Fail-soft: logs and returns on any error.
    """
    target = _verdict_path(project_dir)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = _verdict_to_dict(verdict)
        blob = json.dumps(payload, indent=2, sort_keys=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=target.parent,
            prefix=".tmp_spec_validation_verdict_",
            suffix=".json",
        )
        try:
            with os.fdopen(fd, "w") as f:
                f.write(blob)
            os.replace(tmp_path, target)
        except Exception:  # noqa: BLE001
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("spec_validator: failed to persist verdict (%s)", exc)


def load_spec_verdict(
    project_dir: Path,
    *,
    expect_fingerprint: str | None = None,
) -> SpecValidatorVerdict | None:
    """Load the persisted verdict, or return None if absent or stale.

    If ``expect_fingerprint`` is given and the stored verdict's
    ``rubric_fingerprint`` does not match, returns None (stale verdict
    ignored — a verdict for a different rubric must never influence the
    current run).
    """
    target = _verdict_path(project_dir)
    try:
        if not target.exists():
            return None
        with open(target) as f:
            d = json.load(f)
        verdict = _dict_to_verdict(d)
        if expect_fingerprint is not None and verdict.rubric_fingerprint != expect_fingerprint:
            logger.debug(
                "spec_validator: stale verdict (stored fp=%r, expected fp=%r) — ignored",
                verdict.rubric_fingerprint,
                expect_fingerprint,
            )
            return None
        return verdict
    except Exception as exc:  # noqa: BLE001
        logger.warning("spec_validator: failed to load verdict (%s)", exc)
        return None
