"""EvidenceVector held-out non-regression admission gate.

Decides whether a candidate self-improvement lesson (Phase 1e Unit A's
``FailureAttribution`` plus an advisory patch payload) may be promoted from
``"candidate"`` to ``"active"`` -- i.e. safe to later surface as a guidance
hint. The trust signal is the SAME deterministic multi-predicate evidence
layer the harness already computes for a live run
(``evidence_audit.EvidenceAudit``): ``backed_by_ledger``,
``provenance_present``, ``metrics_non_degenerate``, ``metric_keys_real``,
``rerun_agrees``, ``run_level_clean``. This module NEVER reads a scalar LLM
grade -- the red line from the Phase 1e spec (Global Constraints, "Evidence,
not grade").

Admission mirrors a held-out replay set: for each ``ReplayCase`` (a known
baseline evidence vector), a caller-supplied ``apply_fn`` applies the
candidate lesson and re-audits; the result is projected through
:func:`evidence_vector` and compared against ``case.expected_predicates``.
The candidate is promoted to ``"active"`` only if EVERY case's post-apply
vector clears the validator veto (``run_level_clean``) AND no held-out
predicate regresses True -> False vs that case's baseline AND at least one
held-out predicate improves False -> True in at least one case. A veto
failure or any regression rejects the candidate outright -- the veto is an
absolute gate, never outvoted by unrelated improvements.

Advisory only: admission never mutates run mechanics, it only decides
whether ``candidate.patch`` (a guidance-hint payload) may later be surfaced.

See ``docs/history/plans/2026-07-01-phase-1e-experience-memory.md``,
Unit C.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from backend.agents.rlm.failure_attribution import FailureAttribution

# ---------------------------------------------------------------------------
# Predicate vocabulary
# ---------------------------------------------------------------------------
# The four held-out predicates compared before/after applying a candidate
# lesson (regression- and improvement-eligible).
_HELD_OUT_PREDICATES: tuple[str, ...] = (
    "backed_by_ledger",
    "provenance_present",
    "metrics_non_degenerate",
    "metric_keys_real",
)

# The validator-veto predicate -- an ABSOLUTE gate. A False veto rejects the
# candidate even if every held-out predicate improved. Never outvoted.
_VETO_PREDICATE = "run_level_clean"

# Full predicate vocabulary read into evidence_vector()'s output dict --
# mirrors every boolean-shaped field/property on evidence_audit.EvidenceAudit
# (excludes the non-boolean `fingerprint`/`reasons` fields).
_VECTOR_PREDICATES: tuple[str, ...] = (*_HELD_OUT_PREDICATES, "rerun_agrees", _VETO_PREDICATE)


@dataclass(frozen=True)
class CandidateLesson:
    """A candidate self-improvement lesson pending held-out admission.

    ``patch`` is the advisory hint payload only (guidance text + provenance
    refs) -- never run mechanics; admission never mutates a live run, only
    decides whether the hint may later be surfaced (Global Constraints:
    advisory only, never auto-mutates run mechanics).
    """

    attribution: FailureAttribution
    patch: dict
    admission_state: str = "candidate"  # "candidate" | "active" | "rejected"


@dataclass(frozen=True)
class ReplayCase:
    """A single held-out replay case: an id plus the baseline (pre-apply)
    evidence predicates a candidate's post-apply vector is compared against.

    The callable that applies the candidate and re-audits (``apply_fn``) is
    supplied separately to :func:`admit`, not stored on the case, so the
    SAME replay set can be reused to gate many candidates.
    """

    id: str
    expected_predicates: dict


def _coerce_bool(value: Any) -> bool:
    try:
        return bool(value)
    except Exception:  # noqa: BLE001 -- fail-soft, a predicate read never raises
        return False


def evidence_vector(audit: Any) -> dict:
    """Project an evidence audit into a ``{predicate_name: bool}`` vector.

    Accepts EITHER a real ``evidence_audit.EvidenceAudit`` instance (read via
    attribute access, so the computed ``run_level_clean`` property resolves
    correctly) OR a plain dict (the held-out replay harness' ``apply_fn``
    returns dicts in Phase 1e -- see this module's tests). A missing
    attribute/key on either shape resolves to ``False`` for that predicate;
    this function never raises.
    """
    vector: dict[str, bool] = {}
    is_mapping = isinstance(audit, dict)
    for name in _VECTOR_PREDICATES:
        try:
            raw = audit.get(name, False) if is_mapping else getattr(audit, name, False)
        except Exception:  # noqa: BLE001 -- fail-soft per predicate
            raw = False
        vector[name] = _coerce_bool(raw)
    return vector


def admit(
    candidate: CandidateLesson,
    replay_set: list[ReplayCase],
    apply_fn: Callable[[CandidateLesson, ReplayCase], Any],
) -> CandidateLesson:
    """Held-out non-regression admission gate.

    For each ``ReplayCase`` in ``replay_set``: apply the candidate via
    ``apply_fn(candidate, case)`` (Phase 1e: a CPU/cheap-tier re-audit),
    project the result through :func:`evidence_vector`, and compare it to
    ``case.expected_predicates`` (the held-out baseline). Promotes
    ``candidate`` to ``admission_state="active"`` IFF, across ALL cases:

    1. The veto predicate (``run_level_clean``) is True in every case's
       post-apply vector -- an absolute gate; a single False veto rejects
       the candidate even if everything else improved.
    2. NO held-out predicate regresses True -> False vs its case's
       baseline, in any case.
    3. At least one held-out predicate improves False -> True, in at
       least one case.

    Otherwise returns ``candidate`` with ``admission_state="rejected"``
    (logged by the caller, never applied). Reads ONLY boolean evidence
    predicates -- never a scalar grade. Fail-soft: an empty/invalid
    ``replay_set`` or a raising ``apply_fn`` degrades to "rejected" rather
    than raising -- a candidate must earn promotion, errors never promote.
    """
    try:
        cases = list(replay_set) if replay_set else []
    except TypeError:
        cases = []

    improved = False
    for case in cases:
        try:
            post_apply = apply_fn(candidate, case)
        except Exception:  # noqa: BLE001 -- a failing replay never promotes
            return dataclasses.replace(candidate, admission_state="rejected")

        vector = evidence_vector(post_apply)
        baseline = case.expected_predicates if isinstance(case.expected_predicates, dict) else {}

        if not vector.get(_VETO_PREDICATE, False):
            return dataclasses.replace(candidate, admission_state="rejected")

        for name in _HELD_OUT_PREDICATES:
            before = _coerce_bool(baseline.get(name, False))
            after = vector.get(name, False)
            if before and not after:
                return dataclasses.replace(candidate, admission_state="rejected")
            if not before and after:
                improved = True

    if improved:
        return dataclasses.replace(candidate, admission_state="active")
    return dataclasses.replace(candidate, admission_state="rejected")


__all__ = ["CandidateLesson", "ReplayCase", "admit", "evidence_vector"]
