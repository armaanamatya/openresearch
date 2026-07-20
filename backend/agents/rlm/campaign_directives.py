"""PLAN_ATTEMPT directive synthesis -- the clean-context contract (spec §9,
§8.4 novelty-fingerprint inputs).

Deterministic assembly of attempt N+1's structured inputs from attempt N's
recorded artifacts. No new LLM surface: every field is copied, reduced, or
capped from an on-disk artifact or an already-computed policy value. CCRM
(2605.08563): contaminated retries cascade 7.1x, clean restarts resolve
~21% more at equal budget -- so any input that could smuggle a raw
transcript, REPL history, or prior prompt into attempt N+1's context FAILS
THE BUILD (:class:`DirectiveContractError`); it is never filtered silently.

Pure function module: no env reads, no filesystem access beyond the
caller-supplied artifact paths, no imports beyond stdlib +
:mod:`campaign_policy` (the frozen ``AttemptEnvelope`` / ``NextAttemptPlan``
/ ``directives_fingerprint`` this module composes, never reimplements).
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.agents.rlm.asha_scheduler import BranchType
from backend.agents.rlm.campaign_policy import AttemptEnvelope, NextAttemptPlan, directives_fingerprint

_BRANCH_TYPES: frozenset[str] = frozenset({"faithful", "ambiguity", "discovery"})


def _validated_branch_type(value: Any) -> BranchType:
    """Reject malformed plan data rather than persisting free-text branches."""
    if isinstance(value, str) and value in _BRANCH_TYPES:
        return value  # type: ignore[return-value]
    raise DirectiveContractError(f"plan.branch_type must be one of {sorted(_BRANCH_TYPES)}, got {value!r}")


def _validated_safety_bracket(value: Any) -> bool:
    """Only an actual harness boolean may receive the ASHA exemption."""
    if type(value) is bool:
        return value
    raise DirectiveContractError(f"plan.is_safety_bracket must be bool, got {value!r}")

# --- Clean-context contract vocabulary (§9) ---------------------------------

#: prior_run_artifacts keys the campaign may carry forward, each pinned to
#: the ONE basename that artifact must have. Anything else -- an unknown
#: key or a mismatched basename -- is a build failure, not a silent skip.
_ALLOWED_ARTIFACT_BASENAMES: dict[str, str] = {
    "leaf_triage": "leaf_triage.json",
    "failure_capsules": "failure_capsules.jsonl",
    "metrics": "metrics.json",
    "prior_report": "final_report.json",
    "understanding": "understanding.json",
}

#: substrings marking a path as carrying raw transcript/REPL/prompt content
#: rather than a structured artifact (CCRM). ``.log`` and ``attempt_`` are
#: BOTH listed independently (deliberately redundant, reject-harder) so the
#: campaign's own per-attempt driver log (``attempt_<n>.log``, see the
#: campaign file layout) is caught from either direction.
_FORBIDDEN_PATH_MARKERS: tuple[str, ...] = (
    "dashboard_events.jsonl",
    "user_messages.jsonl",
    "iterations.jsonl",
    "repl_state",
    ".log",
    "prompt",
    "transcript",
    "attempt_",
)

#: reduced leaf-plan-entry keys (spec §9 item 2) -- justification (grader
#: prose) is dropped: never context the next attempt needs, never a
#: fingerprint input.
_LEAF_PLAN_ENTRY_KEYS: tuple[str, ...] = ("leaf_id", "repair_class", "cost", "directive")

_MAX_UNRESOLVED_BULLETS = 5
_MAX_LEAF_BULLETS = 8
_MAX_MEMORY_HINTS = 5
_MAX_MEMORY_HINT_CHARS = 200
_MAX_IMPROVEMENT_NOTES = 8
_MAX_IMPROVEMENT_NOTE_CHARS = 400
_MAX_EXTRA_GUIDANCE_CHARS = 4000
_TRUNCATION_MARKER = "\n[truncated]"

#: MLE-STAR-style localized-refinement directive (spec, Tier-1 repair-mode
#: wiring). Prepended right after the [campaign] header whenever a seeded
#: (non-fresh) lineage has concrete repair guidance to act on -- steers the
#: implementer to edit the seeded ``code/_best_attempt/`` tree in place
#: rather than re-implementing from scratch.
_REPAIR_MODE_TEXT = (
    "[repair-mode] A seeded working tree from the prior attempt is staged at "
    "code/_best_attempt/. START from it: copy it into code/ and EDIT IN PLACE. "
    "Do NOT re-implement from scratch. Change ONLY what the [leaf-repairs]/"
    "[memory] items below name; keep every passing cell's code byte-identical."
)


class DirectiveContractError(Exception):
    """PLAN_ATTEMPT's clean-context contract was violated. Fail the build --
    a prohibited or corrupt input is never filtered silently (spec §9)."""


@dataclass(frozen=True)
class AttemptDirectives:
    """Attempt N+1's complete, structured-only input set -- persisted to
    ``directives/<n>.json`` (spec §9). Every field traces to a recorded
    artifact or a policy value; none can carry raw transcript content."""

    attempt_n: int
    project_id: str
    paper_ref: str
    enforcement: Mapping[str, Any]
    target_floor: float | None
    understanding_ref: str | None
    unresolved_warnings: tuple[str, ...]
    leaf_repair_plan: Mapping[str, Any] | None
    improvement_notes: tuple[str, ...]
    failure_capsules_ref: str | None
    prior_evidence_ref: str | None
    memory_hints: tuple[str, ...]
    injected_lesson_signatures: tuple[str, ...]
    scope_spec: str | None
    scope_rung: int
    seed_lineage: str
    seed_pointer: str | None
    envelope: AttemptEnvelope
    extra_guidance: str
    run_spec_path: str | None
    fingerprint: str
    # Trailing defaults keep legacy in-memory/directive construction on the
    # faithful serial path. A future branch policy may opt in explicitly.
    branch_type: BranchType = "faithful"
    is_safety_bracket: bool = False

    def __post_init__(self) -> None:
        branch_type = _validated_branch_type(self.branch_type)
        is_safety_bracket = _validated_safety_bracket(self.is_safety_bracket)
        if is_safety_bracket and branch_type != "faithful":
            raise DirectiveContractError("is_safety_bracket is allowed only for a faithful branch")

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe representation -- the persisted ``directives/<n>.json``
        payload. Tuples become lists so a reload via ``json.loads`` compares
        equal to this dict."""
        branch_type = _validated_branch_type(self.branch_type)
        is_safety_bracket = _validated_safety_bracket(self.is_safety_bracket)
        payload = {
            "attempt_n": self.attempt_n,
            "project_id": self.project_id,
            "paper_ref": self.paper_ref,
            "enforcement": dict(self.enforcement),
            "target_floor": self.target_floor,
            "understanding_ref": self.understanding_ref,
            "unresolved_warnings": list(self.unresolved_warnings),
            "leaf_repair_plan": (
                {"plan": [dict(entry) for entry in self.leaf_repair_plan.get("plan", [])]}
                if self.leaf_repair_plan is not None
                else None
            ),
            "improvement_notes": list(self.improvement_notes),
            "failure_capsules_ref": self.failure_capsules_ref,
            "prior_evidence_ref": self.prior_evidence_ref,
            "memory_hints": list(self.memory_hints),
            "injected_lesson_signatures": list(self.injected_lesson_signatures),
            "scope_spec": self.scope_spec,
            "scope_rung": self.scope_rung,
            "seed_lineage": self.seed_lineage,
            "seed_pointer": self.seed_pointer,
            "envelope": self.envelope.to_dict(),
            "extra_guidance": self.extra_guidance,
            "run_spec_path": self.run_spec_path,
            "fingerprint": self.fingerprint,
        }
        # Preserve historical default-OFF directive bytes. These are emitted
        # only for an explicit typed branch or the tree-gated advisory marker.
        if branch_type != "faithful":
            payload["branch_type"] = branch_type
        if is_safety_bracket:
            payload["is_safety_bracket"] = True
        return payload

    def persist(self, path: Path) -> None:
        """Atomic same-directory write -- never a half-written directives
        file on disk (mirrors ``attempt_driver._atomic_write_json``)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        os.replace(tmp, path)


# --- Clean-context validation ------------------------------------------------


def _reject_transcript_shaped(path: str, *, field: str) -> None:
    for marker in _FORBIDDEN_PATH_MARKERS:
        if marker in path:
            raise DirectiveContractError(
                f"{field}: {path!r} contains forbidden marker {marker!r} -- "
                "transcript-shaped paths never reach attempt N+1 (spec §9, CCRM)"
            )


def _validate_prior_run_artifacts(prior_run_artifacts: Mapping[str, str]) -> None:
    for key, path in prior_run_artifacts.items():
        expected_basename = _ALLOWED_ARTIFACT_BASENAMES.get(key)
        if expected_basename is None:
            raise DirectiveContractError(
                f"prior_run_artifacts: unknown artifact key {key!r} "
                f"(allowed: {sorted(_ALLOWED_ARTIFACT_BASENAMES)})"
            )
        actual_basename = Path(path).name
        if actual_basename != expected_basename:
            raise DirectiveContractError(
                f"prior_run_artifacts[{key!r}]: expected basename {expected_basename!r}, "
                f"got {actual_basename!r} (path={path!r})"
            )
        _reject_transcript_shaped(path, field=f"prior_run_artifacts[{key!r}]")


def _parse_leaf_triage(path: str) -> dict[str, Any]:
    """Parse+reduce ``leaf_triage.json`` (spec §9 item 2). Any shape defect
    -- unreadable file, invalid JSON, missing ``plan`` list, a non-object
    entry, an entry missing a required key, or a required key holding a
    non-``str`` value -- is a build failure: a corrupt input is never
    silently skipped, and never escapes this module's typed contract to
    crash downstream as a raw ``TypeError`` (e.g. ``campaign_policy``'s
    ``directives_fingerprint`` does ``sorted(set(repair_action_kinds))``,
    which raises on an unhashable ``repair_class`` such as a list)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DirectiveContractError(
            f"prior_run_artifacts['leaf_triage']: unparseable file {path!r}: {exc}"
        ) from exc
    if not isinstance(data, dict) or not isinstance(data.get("plan"), list):
        raise DirectiveContractError(
            f"prior_run_artifacts['leaf_triage']: malformed content at {path!r} "
            "(expected an object with a 'plan' list)"
        )
    reduced: list[dict[str, Any]] = []
    for entry in data["plan"]:
        if not isinstance(entry, dict):
            raise DirectiveContractError(
                f"prior_run_artifacts['leaf_triage']: plan entry is not an object: {entry!r}"
            )
        reduced_entry: dict[str, Any] = {}
        for key in _LEAF_PLAN_ENTRY_KEYS:
            try:
                value = entry[key]
            except KeyError as exc:
                raise DirectiveContractError(
                    f"prior_run_artifacts['leaf_triage']: plan entry {entry!r} missing key {exc}"
                ) from exc
            if not isinstance(value, str):
                raise DirectiveContractError(
                    f"prior_run_artifacts['leaf_triage']: plan entry {entry!r} key {key!r} "
                    f"must be a str, got {type(value).__name__} ({value!r})"
                )
            reduced_entry[key] = value
        reduced.append(reduced_entry)
    return {"plan": reduced}


def _cap_strings(items: Sequence[str], *, max_items: int, max_chars: int) -> list[str]:
    return [str(item)[:max_chars] for item in list(items)[:max_items]]


# --- extra_guidance assembly (spec §9 item 4) --------------------------------


def _assemble_extra_guidance(
    *,
    attempt_n: int,
    plan: NextAttemptPlan,
    unresolved_warnings: Sequence[str],
    leaf_plan_entries: Sequence[Mapping[str, Any]],
    improvement_notes: Sequence[str],
    memory_hints: Sequence[str],
    scope_spec: str | None,
    target_floor: float | None,
) -> str:
    lines: list[str] = [f"[campaign] Attempt {attempt_n} ({plan.lineage}, scope rung {plan.scope_rung})."]

    if plan.lineage != "fresh" and (leaf_plan_entries or memory_hints):
        lines.append(_REPAIR_MODE_TEXT)

    if unresolved_warnings:
        lines.append("[unresolved-understanding]")
        lines.extend(f"- {w}" for w in list(unresolved_warnings)[:_MAX_UNRESOLVED_BULLETS])

    if leaf_plan_entries:
        lines.append("[leaf-repairs]")
        lines.extend(
            f"- {e['leaf_id']} [{e['repair_class']}] {e['directive']}"
            for e in list(leaf_plan_entries)[:_MAX_LEAF_BULLETS]
        )

    if improvement_notes:
        lines.append("[prior-improvements]")
        lines.extend(f"- {note}" for note in improvement_notes)

    if memory_hints:
        lines.append("[memory]")
        lines.extend(f"- {hint}" for hint in memory_hints)

    if scope_spec:
        lines.append(f"[scope] {scope_spec}")

    if target_floor is not None:
        lines.append(f"[target-floor] Prior guard-clean best: {target_floor}")

    text = "\n".join(lines)
    if len(text) > _MAX_EXTRA_GUIDANCE_CHARS:
        text = text[: _MAX_EXTRA_GUIDANCE_CHARS - len(_TRUNCATION_MARKER)].rstrip() + _TRUNCATION_MARKER
    return text


# --- Public entry point -------------------------------------------------------


def synthesize_directives(
    *,
    attempt_n: int,
    project_id: str,
    paper_ref: str,
    plan: NextAttemptPlan,
    envelope: AttemptEnvelope,
    enforcement: Mapping[str, Any],
    run_spec_path: str | None,
    understanding_ref: str | None,
    unresolved_warnings: Sequence[str],
    prior_run_artifacts: Mapping[str, str],
    improvement_notes: Sequence[str],
    memory_hints: Sequence[str],
    injected_lesson_signatures: Sequence[str],
    failure_classes: Sequence[str],
    scope_spec: str | None,
    target_floor: float | None,
    out_dir: Path,
) -> AttemptDirectives:
    """Deterministically assemble attempt ``attempt_n``'s directives and
    persist them to ``out_dir/directives/<attempt_n>.json`` (spec §9).

    Fails the build (:class:`DirectiveContractError`) on any clean-context
    violation instead of silently dropping the offending input -- see the
    module docstring. ``failure_classes`` (the latest assessment's
    failure-class set) feeds only the §8.4 novelty fingerprint; it is not
    itself a persisted field.
    """
    _validate_prior_run_artifacts(prior_run_artifacts)
    if understanding_ref is not None:
        _reject_transcript_shaped(understanding_ref, field="understanding_ref")
    if run_spec_path is not None:
        _reject_transcript_shaped(run_spec_path, field="run_spec_path")

    leaf_repair_plan: dict[str, Any] | None = None
    leaf_plan_entries: list[dict[str, Any]] = []
    leaf_triage_path = prior_run_artifacts.get("leaf_triage")
    if leaf_triage_path is not None:
        leaf_repair_plan = _parse_leaf_triage(leaf_triage_path)
        leaf_plan_entries = leaf_repair_plan["plan"]

    capped_improvement_notes = _cap_strings(
        improvement_notes, max_items=_MAX_IMPROVEMENT_NOTES, max_chars=_MAX_IMPROVEMENT_NOTE_CHARS
    )
    capped_memory_hints = _cap_strings(
        memory_hints, max_items=_MAX_MEMORY_HINTS, max_chars=_MAX_MEMORY_HINT_CHARS
    )

    extra_guidance = _assemble_extra_guidance(
        attempt_n=attempt_n,
        plan=plan,
        unresolved_warnings=unresolved_warnings,
        leaf_plan_entries=leaf_plan_entries,
        improvement_notes=capped_improvement_notes,
        memory_hints=capped_memory_hints,
        scope_spec=scope_spec,
        target_floor=target_floor,
    )

    branch_type = _validated_branch_type(plan.branch_type)
    is_safety_bracket = _validated_safety_bracket(plan.is_safety_bracket)
    fingerprint = directives_fingerprint(
        seed_lineage=f"{plan.lineage}:{plan.seed_attempt_n or 0}",
        scope_rung=plan.scope_rung,
        repair_action_kinds=[entry["repair_class"] for entry in leaf_plan_entries],
        failure_classes=failure_classes,
        envelope=envelope.to_dict(),
        branch_type=branch_type,
    )

    directives = AttemptDirectives(
        attempt_n=attempt_n,
        project_id=project_id,
        paper_ref=paper_ref,
        enforcement=dict(enforcement),
        target_floor=target_floor,
        understanding_ref=understanding_ref,
        unresolved_warnings=tuple(unresolved_warnings),
        leaf_repair_plan=leaf_repair_plan,
        improvement_notes=tuple(capped_improvement_notes),
        failure_capsules_ref=prior_run_artifacts.get("failure_capsules"),
        prior_evidence_ref=prior_run_artifacts.get("metrics") or prior_run_artifacts.get("prior_report"),
        memory_hints=tuple(capped_memory_hints),
        injected_lesson_signatures=tuple(injected_lesson_signatures),
        scope_spec=scope_spec,
        scope_rung=plan.scope_rung,
        seed_lineage=plan.lineage,
        seed_pointer=plan.seed_pointer,
        envelope=envelope,
        extra_guidance=extra_guidance,
        run_spec_path=run_spec_path,
        fingerprint=fingerprint,
        branch_type=branch_type,
        is_safety_bracket=is_safety_bracket,
    )
    directives.persist(Path(out_dir) / "directives" / f"{attempt_n}.json")
    return directives
