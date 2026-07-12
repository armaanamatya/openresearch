"""reference_from_skills.py — compose a display-only eval reference structure
from the run's selected skill library (Track E Task 7, spec §6.4).

The skill-select machinery (``backend.agents.rlm.skill_selection``) already
recalls + prunes the vendored playbook catalog down to a per-run **active
set**, persisted to ``rlm_state/active_skills.json`` — a
``{selected, candidates, domain, subject_matter_keys, selector, reasons}``
artifact written to steer the LLM implementer/verifier. This module reshapes
that SAME artifact into an eval-facing **reference**: a plain description of
what a faithful reproduction of this paper's domain is expected to
measure / compare against / report on::

    {expected_metric_families, standard_baselines, eval_protocol,
     dataset_expectations}

Every entry in every list is provenance-tagged ``"evaluator_computed"`` — this
is the harness's OWN derivation from the skill catalog, never a
paper-reported or agent-measured value.

**Leniency guard (STRUCTURE ONLY — the §6.4 invariant):** a skill can supply
structure, never a pass. This module reads only
``rlm_state/active_skills.json`` and returns an inert dict of name-only
entries; it does not import, call, or in any way reference
``result_fidelity`` (or ``verdict_authority`` / any other module that writes a
claim's ``status`` or a verdict). There is consequently NO code path — no
parameter, no return value, no import — connecting anything this module
produces to a per-claim ``pass``/``fail`` decision.
``tests/test_reference_from_skills.py`` proves this both statically (a
source-scan guard mirroring Track E Task 1's verdict-surface guard) and
empirically (a load-bearing test that injects this module's own output, plus
forged ``status``/``expected_status`` fields, straight onto a claim and
asserts ``result_fidelity.evaluate`` never reads them — the per-claim status
stays ``"unmeasured"``).

Pure, stdlib-only, no LLM/network call. Never raises: a missing, unreadable,
or malformed ``active_skills.json`` degrades to a well-formed EMPTY reference
(every key present, every value ``[]``) — never an exception, never a
guessed/synthesized entry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_ACTIVE_SKILLS_RELPATH = ("rlm_state", "active_skills.json")

# Every reference value is tagged with this provenance — this module's own
# derivation, distinct from ``paper_reported``/``agent_measured`` (the other
# two legs of the scorecard's three-way provenance taxonomy, spec §6.3).
_PROVENANCE = "evaluator_computed"

# The four reference fields this module populates (spec §6.4). Every call
# returns exactly these keys — present-but-empty when there is nothing to
# derive, never omitted.
_REFERENCE_KEYS = (
    "expected_metric_families",
    "standard_baselines",
    "eval_protocol",
    "dataset_expectations",
)


def _empty_reference() -> dict[str, list[dict[str, Any]]]:
    return {key: [] for key in _REFERENCE_KEYS}


def _tag(value: str, *, source_skill: str | None = None) -> dict[str, Any]:
    """Wrap one reference value with its provenance tag.

    A plain ``{value, provenance, source_skill}`` dict — never a status/score
    field of any kind. ``source_skill`` names the specific selected skill the
    value traces back to (``None`` when the value comes from the paper's own
    subject-matter summary rather than a specific skill).
    """
    return {"value": value, "provenance": _PROVENANCE, "source_skill": source_skill}


def _dedup_strings(items: Any) -> list[str]:
    """Best-effort coercion of a raw JSON field to a deduped, order-preserving
    list of non-empty strings. Never raises."""
    out: list[str] = []
    seen: set[str] = set()
    if not isinstance(items, (list, tuple)):
        return out
    for item in items:
        if not isinstance(item, str):
            continue
        token = item.strip()
        key = token.lower()
        if token and key not in seen:
            seen.add(key)
            out.append(token)
    return out


def _load_active_skills(project_dir: Path) -> dict[str, Any] | None:
    """Best-effort parse of ``<project_dir>/rlm_state/active_skills.json``.

    Returns ``None`` on any absence/read/parse failure — never raises.
    """
    try:
        path = Path(project_dir).joinpath(*_ACTIVE_SKILLS_RELPATH)
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def compose_reference(project_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Compose a display-only eval reference STRUCTURE from the run's
    selected-skill artifact (``rlm_state/active_skills.json``).

    Returns ``{expected_metric_families, standard_baselines, eval_protocol,
    dataset_expectations}`` — every entry a ``{value, provenance,
    source_skill}`` dict tagged ``provenance="evaluator_computed"``.
    STRUCTURE ONLY: this names WHAT a faithful reproduction of this paper's
    domain is expected to measure / compare against / report on — it never
    says whether any claim passed or failed (see module docstring's leniency
    guard).

    Deterministic derivation, no LLM call:
      * ``expected_metric_families`` <- ``subject_matter_keys.metrics``
      * ``dataset_expectations``     <- ``subject_matter_keys.datasets``
      * ``eval_protocol``            <- ``subject_matter_keys.methods`` (the
        paper's own named training/optimization recipe elements — the
        artifact carries no dedicated protocol field of its own)
      * ``standard_baselines``       <- the names of every SELECTED skill
        (the skills the run's own selection layer determined were genuinely
        relevant to this paper's domain), each tagged with itself as
        ``source_skill``

    Fail-soft to a well-formed EMPTY reference (every key present, each an
    empty list) when ``active_skills.json`` is absent, unreadable, or
    malformed. Never raises.
    """
    artifact = _load_active_skills(Path(project_dir))
    if artifact is None:
        return _empty_reference()

    subject_matter = artifact.get("subject_matter_keys")
    subject_matter = subject_matter if isinstance(subject_matter, dict) else {}

    metrics = _dedup_strings(subject_matter.get("metrics"))
    datasets = _dedup_strings(subject_matter.get("datasets"))
    methods = _dedup_strings(subject_matter.get("methods"))
    selected = _dedup_strings(artifact.get("selected"))

    return {
        "expected_metric_families": [_tag(m) for m in metrics],
        "standard_baselines": [_tag(name, source_skill=name) for name in selected],
        "eval_protocol": [_tag(m) for m in methods],
        "dataset_expectations": [_tag(d) for d in datasets],
    }
