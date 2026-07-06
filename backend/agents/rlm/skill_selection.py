"""Relevance-gated, agent-selected skill activation (understand-phase).

Adds a *selection* layer on top of the already-merged skill library:

  1. Deterministic thorough recall — :func:`skill_matcher.match_skills` scores
     the vendored catalog against the paper's claim-map + environment-spec and
     returns a recall-biased candidate shortlist (names + token reasons + a
     coarse domain). This module does NOT reimplement that matcher; it reuses
     it verbatim so the deterministic candidate provenance stays single-source.
  2. Bounded agent/LLM precision pick — :func:`llm_prune_candidates` asks the
     paper-understanding transport to prune the shortlist to the subset genuinely
     needed to reproduce THIS paper (it can only prune, never invent names).
     Fail-soft: any error / empty / unparseable output degrades to the
     deterministic shortlist.

The result is a per-run **active skill set** persisted to
``rlm_state/active_skills.json`` and surfaced to both the root (via
``consult_skill``'s index) and the verifier (via a bounded grader-prompt
context) — closing the two gaps ("relevance not tied to the paper" and "the
verifier can't consult skills").

Flags (all default-OFF; byte-identical when off):
  * ``OPENRESEARCH_SKILLS``                    – master gate (existing).
  * ``OPENRESEARCH_SKILL_SELECT``              – enables this selection layer.
  * ``OPENRESEARCH_SKILL_SELECT_DETERMINISTIC``– skip the LLM pick (top-K).
  * ``OPENRESEARCH_SKILL_CANDIDATES_MAX``      – deterministic candidate cap (15).
  * ``OPENRESEARCH_SKILL_VERIFIER_BODIES``     – # playbook bodies inlined to the
                                                 grader (size guard, default 2).

Invariant: skills are **advisory** — they sharpen the LLM implementer/verifier
but never become a fitness signal. The deterministic evidence layer stays the
sole authority. Pure-ish + fail-soft throughout: no exception ever propagates
out of the public entry points.
"""
from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from backend.agents.rlm.skill_catalog import SkillMeta, get_skill_body
from backend.agents.rlm.skill_matcher import match_skills

logger = logging.getLogger(__name__)

_ACTIVE_SKILLS_RELPATH = ("rlm_state", "active_skills.json")

_DEFAULT_CANDIDATES_MAX = 15
_DEFAULT_VERIFIER_BODIES = 2
# Per-body char cap when inlining playbook bodies into the grader prompt — a
# hard size guard so a long playbook can't blow the grader context.
_VERIFIER_BODY_CHAR_CAP = 2000
# Bound the LLM-prune completion; unparseable/oversized output falls back.
_LLM_PRUNE_MAX_CHARS = 8000


# ---------------------------------------------------------------------------
# Flag helpers (canonical convention; default-OFF)
# ---------------------------------------------------------------------------

def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def select_enabled() -> bool:
    """True only when the master gate AND the selection gate are both on."""
    return _flag("OPENRESEARCH_SKILLS") and _flag("OPENRESEARCH_SKILL_SELECT")


def _deterministic_only() -> bool:
    return _flag("OPENRESEARCH_SKILL_SELECT_DETERMINISTIC")


def _int_flag(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        val = int(raw)
    except ValueError:
        return default
    return val if val > 0 else default


def _candidates_max() -> int:
    return _int_flag("OPENRESEARCH_SKILL_CANDIDATES_MAX", _DEFAULT_CANDIDATES_MAX)


def _verifier_bodies() -> int:
    # 0 is a valid choice here (descriptions only, no inlined bodies).
    raw = os.environ.get("OPENRESEARCH_SKILL_VERIFIER_BODIES", "").strip()
    if not raw:
        return _DEFAULT_VERIFIER_BODIES
    try:
        val = int(raw)
    except ValueError:
        return _DEFAULT_VERIFIER_BODIES
    return max(0, val)


# ---------------------------------------------------------------------------
# Subject-matter summary (provenance + LLM-prune grounding)
# ---------------------------------------------------------------------------

def _names_of(items: Any, key: str = "name") -> list[str]:
    """Pull short display names from a claim-map list field.

    Items are typically ``{name: ...}`` dicts (per ``DatasetRequirement`` /
    ``MetricSpec``) but the LLM-generated REPL sometimes hands bare strings —
    handle both. Never raises.
    """
    out: list[str] = []
    if not isinstance(items, (list, tuple)):
        return out
    for item in items:
        if isinstance(item, str):
            token = item.strip()
        elif isinstance(item, Mapping):
            token = str(item.get(key) or "").strip()
        else:
            token = ""
        if token:
            out.append(token[:80])
    return out


def _subject_matter(claim_map: Mapping[str, Any], environment_spec: Mapping[str, Any]) -> dict[str, list[str]]:
    """Build a compact, JSON-serialisable subject-matter summary.

    Used both as the LLM-prune grounding and as ``subject_matter_keys`` in the
    persisted provenance. Deterministic, deduped-order-preserving, capped.
    """
    cm: Mapping[str, Any] = claim_map if isinstance(claim_map, Mapping) else {}
    env: Mapping[str, Any] = environment_spec if isinstance(environment_spec, Mapping) else {}

    datasets = _names_of(cm.get("datasets"))
    metrics = _names_of(cm.get("metrics"))

    methods: list[str] = []
    core = cm.get("core_contribution")
    if isinstance(core, str) and core.strip():
        methods.append(core.strip()[:200])
    arch = cm.get("model_architecture")
    if isinstance(arch, str) and arch.strip():
        methods.append(arch.strip()[:120])
    recipe = cm.get("training_recipe")
    if isinstance(recipe, Mapping):
        opt = str(recipe.get("optimizer") or "").strip()
        if opt:
            methods.append(opt[:60])

    frameworks: list[str] = []
    fw = env.get("framework")
    if isinstance(fw, str) and fw.strip():
        frameworks.append(fw.strip()[:60])
    pip = env.get("pip_packages")
    if isinstance(pip, Mapping):
        for pkg in list(pip.keys())[:12]:
            if isinstance(pkg, str) and pkg.strip():
                frameworks.append(pkg.strip()[:40])

    def _dedup(seq: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for s in seq:
            k = s.lower()
            if k not in seen:
                seen.add(k)
                out.append(s)
        return out

    return {
        "datasets": _dedup(datasets)[:12],
        "metrics": _dedup(metrics)[:12],
        "methods": _dedup(methods)[:6],
        "frameworks": _dedup(frameworks)[:12],
    }


# ---------------------------------------------------------------------------
# Bounded LLM precision pick
# ---------------------------------------------------------------------------

_LLM_PRUNE_SYSTEM = (
    "You are selecting which reference skill playbooks are genuinely needed to "
    "reproduce a specific research paper. You are given the paper's subject "
    "matter and a shortlist of candidate skills (name, category, description). "
    "Return ONLY the candidate skills whose subject matter is actually required "
    "to implement or evaluate THIS paper — omit the rest. You may not invent "
    "names: every returned name MUST come verbatim from the candidate list. "
    'Output ONLY a JSON object: {"selected": [{"name": "<candidate-name>", '
    '"reason": "<one short clause>"}]}. No other text.'
)


def _candidate_prompt(subject_matter: Mapping[str, Any], candidates: list[dict[str, Any]]) -> str:
    lines = ["## Paper subject matter", json.dumps(subject_matter, indent=2, default=str), ""]
    lines.append("## Candidate skills")
    for c in candidates:
        lines.append(
            f"- name: {c['name']} | category: {c.get('category', '')} | "
            f"description: {str(c.get('description', ''))[:240]}"
        )
    lines.append("")
    lines.append(
        "Return the JSON object of the subset genuinely needed to reproduce this paper."
    )
    return "\n".join(lines)


def _parse_selected(raw: str, candidate_names: set[str]) -> dict[str, str]:
    """Parse the LLM-prune response into ``{name: reason}``, restricted to
    candidate names. Tolerant: pulls the first JSON object/array out of the raw
    text. Returns ``{}`` on any failure (caller treats that as "fall back")."""
    if not raw:
        return {}
    text = raw.strip()
    if len(text) > _LLM_PRUNE_MAX_CHARS:
        text = text[:_LLM_PRUNE_MAX_CHARS]
    # Isolate the first {...} or [...] block so prose around the JSON is ignored.
    obj_match = re.search(r"\{.*\}", text, re.DOTALL)
    arr_match = re.search(r"\[.*\]", text, re.DOTALL)
    blob = None
    if obj_match:
        blob = obj_match.group(0)
    elif arr_match:
        blob = arr_match.group(0)
    if blob is None:
        return {}
    try:
        data = json.loads(blob)
    except (json.JSONDecodeError, ValueError):
        return {}

    rows: Any
    if isinstance(data, Mapping):
        rows = data.get("selected", [])
    elif isinstance(data, list):
        rows = data
    else:
        return {}
    if not isinstance(rows, list):
        return {}

    out: dict[str, str] = {}
    for row in rows:
        if isinstance(row, str):
            name, reason = row.strip(), ""
        elif isinstance(row, Mapping):
            name = str(row.get("name") or "").strip()
            reason = str(row.get("reason") or "").strip()
        else:
            continue
        if name in candidate_names and name not in out:
            out[name] = reason[:200]
    return out


def llm_prune_candidates(
    subject_matter: Mapping[str, Any],
    candidates: list[dict[str, Any]],
    llm_client: Any,
) -> dict[str, str]:
    """Bounded single-call LLM prune of the candidate shortlist.

    Returns ``{name: reason}`` for the selected subset (a subset of the
    candidate names). Fail-soft: returns ``{}`` on any error / empty /
    unparseable output — the caller then keeps the deterministic shortlist.
    """
    if llm_client is None or not candidates:
        return {}
    candidate_names = {c["name"] for c in candidates}
    try:
        raw = llm_client.complete(
            system=_LLM_PRUNE_SYSTEM,
            user=_candidate_prompt(subject_matter, candidates),
        )
    except Exception:  # noqa: BLE001 — the pick is advisory; never break the run
        logger.warning("skill_selection: LLM prune call failed", exc_info=True)
        return {}
    return _parse_selected(raw if isinstance(raw, str) else str(raw), candidate_names)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _augment_env_for_recall(environment_spec: Mapping[str, Any]) -> dict[str, Any]:
    """Fold every dependency signal the shared matcher ignores into its one
    consulted field.

    :func:`skill_matcher.match_skills` only tokenizes ``environment_spec
    ["framework"]`` (plus the claim-map). But a paper's real toolchain lives in
    ``pip_packages`` (e.g. ``vllm``/``verl``/``trl``) and ``system_packages`` —
    without these, a serving/RL library the paper depends on but never names in
    the framework field would be missed. This is selection-layer-local recall
    widening: it builds a synthetic ``framework`` list; the shared matcher (and
    its implementer-shortlist consumer) is left untouched.
    """
    env: Mapping[str, Any] = environment_spec if isinstance(environment_spec, Mapping) else {}
    terms: list[Any] = []
    fw = env.get("framework")
    if fw:
        terms.append(fw)
    fwv = env.get("framework_version")
    if isinstance(fwv, Mapping):
        terms.extend(str(k) for k in fwv.keys())
    pip = env.get("pip_packages")
    if isinstance(pip, Mapping):
        terms.extend(str(k) for k in pip.keys())
    syspkgs = env.get("system_packages")
    if isinstance(syspkgs, (list, tuple)):
        terms.extend(str(s) for s in syspkgs)
    return {"framework": [t for t in terms if t]}


def match_candidate_skills(
    claim_map: Mapping[str, Any],
    environment_spec: Mapping[str, Any],
    catalog: Mapping[str, SkillMeta],
    *,
    candidates_max: int | None = None,
) -> list[dict[str, Any]]:
    """Deterministic thorough recall → candidate shortlist.

    Thin, testable wrapper over :func:`skill_matcher.match_skills` that shapes
    the result into ``[{name, category, description, reason}]`` (ranked, capped).
    Widens the recall input via :func:`_augment_env_for_recall` so dependency
    signals (pip/system packages) the paper never names in its framework field
    still surface. Never raises (``match_skills`` is already fail-soft).
    """
    cap = candidates_max if candidates_max is not None else _candidates_max()
    sm = match_skills(claim_map, _augment_env_for_recall(environment_spec), catalog, top_k=cap)
    out: list[dict[str, Any]] = []
    for name, reason in zip(sm.skill_names, sm.reasons):
        meta = catalog.get(name)
        out.append(
            {
                "name": name,
                "category": meta.category if meta else "",
                "description": meta.description if meta else "",
                "reason": reason,
            }
        )
    return out


def select_active_skills(
    claim_map: Mapping[str, Any],
    environment_spec: Mapping[str, Any],
    catalog: Mapping[str, SkillMeta],
    *,
    llm_client: Any = None,
) -> dict[str, Any]:
    """Produce the per-run active skill set (does NOT write to disk).

    Deterministic recall → (bounded LLM prune unless ``_DETERMINISTIC``) →
    an artifact dict::

        {selected, candidates, domain, subject_matter_keys, selector, reasons}

    ``selected`` is a subset of ``candidates`` by name. Fail-soft: any failure
    degrades to the deterministic shortlist (never raises, never empty when the
    matcher found candidates).
    """
    try:
        candidates = match_candidate_skills(claim_map, environment_spec, catalog)
    except Exception:  # noqa: BLE001 — matcher is already fail-soft, but belt-and-braces
        logger.warning("skill_selection: candidate matching failed", exc_info=True)
        candidates = []

    subject_matter = _subject_matter(claim_map, environment_spec)
    # Deterministic per-candidate token reasons (always available as a floor).
    det_reasons: dict[str, str] = {c["name"]: c["reason"] for c in candidates}
    candidate_names = [c["name"] for c in candidates]

    selected: list[str]
    reasons: dict[str, str]
    selector: str

    if candidates and not _deterministic_only() and llm_client is not None:
        pruned = llm_prune_candidates(subject_matter, candidates, llm_client)
        if pruned:
            # Preserve the deterministic candidate ORDER for the selected subset.
            selected = [n for n in candidate_names if n in pruned]
            reasons = {n: (pruned[n] or det_reasons.get(n, "")) for n in selected}
            selector = "deterministic+llm"
        else:
            selected = list(candidate_names)
            reasons = dict(det_reasons)
            selector = "deterministic"
    else:
        selected = list(candidate_names)
        reasons = dict(det_reasons)
        selector = "deterministic"

    return {
        "selected": selected,
        "candidates": candidates,
        "domain": match_skills(claim_map, _augment_env_for_recall(environment_spec), catalog).domain,
        "subject_matter_keys": subject_matter,
        "selector": selector,
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _active_path(project_dir: Path | str) -> Path:
    return Path(project_dir).joinpath(*_ACTIVE_SKILLS_RELPATH)


def write_active_skills(project_dir: Path | str, artifact: Mapping[str, Any]) -> None:
    """Persist the active skill set to ``rlm_state/active_skills.json``. Fail-soft."""
    try:
        path = _active_path(project_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
    except OSError:
        logger.debug("skill_selection: active_skills.json write failed", exc_info=True)


def load_active_skills(project_dir: Path | str) -> dict[str, Any] | None:
    """Read the persisted active skill set, or ``None`` if absent/unreadable."""
    try:
        path = _active_path(project_dir)
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# ---------------------------------------------------------------------------
# Consumption — root (consult_skill index) + verifier (grader prompt)
# ---------------------------------------------------------------------------

def active_skill_entries(
    artifact: Mapping[str, Any], catalog: Mapping[str, SkillMeta]
) -> list[dict[str, str]]:
    """Shape the selected set into ``[{name, category, description, reason}]``
    for the root's ``consult_skill`` index. Fail-soft → ``[]``."""
    if not isinstance(artifact, Mapping):
        return []
    selected = artifact.get("selected") or []
    reasons = artifact.get("reasons") or {}
    if not isinstance(selected, list):
        return []
    out: list[dict[str, str]] = []
    for name in selected:
        if not isinstance(name, str):
            continue
        meta = catalog.get(name)
        out.append(
            {
                "name": name,
                "category": meta.category if meta else "",
                "description": meta.description if meta else "",
                "reason": str(reasons.get(name, "")) if isinstance(reasons, Mapping) else "",
            }
        )
    return out


def build_verifier_skill_context(
    artifact: Mapping[str, Any],
    catalog: Mapping[str, SkillMeta],
    *,
    max_bodies: int | None = None,
) -> str | None:
    """Build the bounded grader-prompt context for the active skill set.

    Descriptions for every selected skill + the sanitized bodies of the top
    ``max_bodies`` (each char-capped) — so the grader can judge fidelity against
    the domain playbook. Advisory only. Returns ``None`` when there is nothing
    to inject (keeps the grader prompt byte-identical when off)."""
    entries = active_skill_entries(artifact, catalog)
    if not entries:
        return None
    n_bodies = _verifier_bodies() if max_bodies is None else max(0, max_bodies)

    lines = [
        "## Skill playbooks relevant to this paper",
        "",
        "The reproduction targets the domain skills below. Use them as reference "
        "for what a faithful implementation of THIS paper's methods/frameworks "
        "should contain. Judge fidelity against them, but score ONLY from the "
        "actual code and the MEASURED metrics — a playbook is context, never proof.",
        "",
    ]
    for e in entries:
        cat = f" ({e['category']})" if e["category"] else ""
        lines.append(f"- {e['name']}{cat}: {e['description']}")
    lines.append("")

    for e in entries[:n_bodies]:
        body = get_skill_body(e["name"])
        if not body:
            continue
        lines.append(f"### Playbook: {e['name']}")
        lines.append(body[:_VERIFIER_BODY_CHAR_CAP])
        lines.append("")

    return "\n".join(lines).strip()
