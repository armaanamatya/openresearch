"""A7 EVIDENCE_GATE — the honest backstop (2026-06-16).

The grader is an LLM; left unchecked it can credit a result that was never
computed (MLR-Bench arXiv:2505.19955 reports ~80% fabricated results in agentic
science harnesses; RewardHacking arXiv:2605.02964 measures −87.7% exploits after
env-hardening). The companion appendix to the grader-fidelity spec elevates this
gate to *the single highest-value correctness lever* in the corpus.

The gate is a **veto layer over the LLM score**, not a re-grade: for a leaf that
*claims a measured result* and that the grader *credited* (score > 0) but whose
cited ``per_model`` cell has **no successful on-disk evidence**, the score is
vetoed to 0.0. It composes with — does not duplicate — the leaf_scorer's existing
on-disk subject matching (``_successful_metric_subjects`` / ``_leaf_has_disk_evidence``):
that caller computes ``has_disk_evidence`` (with the same alias loosening the
data-unavailable detector uses) and this module makes the pure veto decision.

This module is **pure + stdlib-only** so it is trivially unit-testable and carries
no import dependency on leaf_scorer (no circular import).

``OPENRESEARCH_LEAF_EVIDENCE_GATE`` is **default-OFF**: with the flag unset
:func:`evidence_gate_enabled` returns False and the leaf_scorer never calls
:func:`gate_decision`, so scoring is byte-for-byte identical to today. The gate is
flipped ON only after the calibration σ-gate clears (see the spec's rollout
sequence). Conservative by construction — see :func:`leaf_claims_measured_result`:
a leaf is gated only when we are *confident* it asserts a measured numeric result,
so a judgment/method/theory leaf (which legitimately has no ``per_model`` cell) is
never vetoed.

Note on the name split: this module's :func:`evidence_gate_enabled` reads
``OPENRESEARCH_LEAF_EVIDENCE_GATE`` (the per-leaf veto, default-OFF) — distinct from
``OPENRESEARCH_EVIDENCE_GATE`` which is the verdict-level gate in
``report.py::_apply_evidence_gate`` (default-ON). The two were previously conflated
on one var with opposite defaults; the split makes them independently controllable.
"""

from __future__ import annotations

import os
from typing import Any

# Substrings that, in a rubric leaf's task category, mark it as a *measured
# result* claim (vs an implementation/method/theory/process leaf). PaperBench
# taxonomy uses "Result Analysis"; auto-generated rubrics commonly use "result"
# or "metric". Matched case-insensitively as a substring.
_RESULT_CATEGORY_HINTS: tuple[str, ...] = ("result", "metric")


def evidence_gate_enabled() -> bool:
    """True iff the per-leaf evidence veto gate is opted ON.

    Reads ``OPENRESEARCH_LEAF_EVIDENCE_GATE`` (default-OFF).  This is a
    DISTINCT variable from ``OPENRESEARCH_EVIDENCE_GATE``, which controls the
    verdict-level gate in ``report.py::_apply_evidence_gate`` (default-ON).

    Split-default rationale: both gates previously shared
    ``OPENRESEARCH_EVIDENCE_GATE`` but with opposite defaults — the verdict gate
    defaulted ON while the leaf-veto gate defaulted OFF.  Sharing one var with
    opposite defaults made it impossible to independently control them.  The
    leaf-veto gate now reads ``OPENRESEARCH_LEAF_EVIDENCE_GATE`` so the two
    can be toggled independently.

    Backward-compat note: operators who previously set
    ``OPENRESEARCH_EVIDENCE_GATE=1`` to activate the leaf veto should now set
    ``OPENRESEARCH_LEAF_EVIDENCE_GATE=1`` instead.  Setting
    ``OPENRESEARCH_EVIDENCE_GATE=1`` now ONLY controls the verdict gate.
    """
    return os.environ.get("OPENRESEARCH_LEAF_EVIDENCE_GATE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def leaf_claims_measured_result(leaf: dict[str, Any]) -> bool:
    """Conservatively decide whether ``leaf`` asserts a measured numeric result.

    Only a leaf we are *confident* is a measured-result claim is eligible for the
    veto — this is what keeps the gate from tanking legitimate judgment/method
    leaves (which have no ``per_model`` cell and would otherwise look like
    "no on-disk evidence"). Three confident signals, in order:

    1. The task category contains "result" or "metric" (PaperBench "Result
       Analysis"; most auto-rubrics). This is the primary signal.
    2. The leaf carries an explicit ``check_kind`` of ``deterministic:numeric``
       (the same rubric-gen annotation A2's deterministic checker consumes) — an
       unambiguous "this leaf is a numeric result" marker.
    3. The leaf carries an explicit ``evidence_required: true`` annotation.

    A leaf with **no** category and no annotation returns False (NOT gated) — we
    cannot safely tell it is a result claim, so we leave its LLM score alone.
    """
    category = str(
        leaf.get("task_category") or leaf.get("finegrained_task_category") or ""
    ).lower()
    if any(hint in category for hint in _RESULT_CATEGORY_HINTS):
        return True
    check_kind = str(leaf.get("check_kind") or "").strip().lower()
    if check_kind.startswith("deterministic:numeric"):
        return True
    return bool(leaf.get("evidence_required") is True)


def gate_decision(
    *,
    score: float | None,
    claims_result: bool,
    has_disk_evidence: bool,
) -> tuple[float | None, bool]:
    """Pure veto decision for one already-graded leaf.

    Returns ``(possibly_vetoed_score, vetoed)``. A leaf is vetoed to 0.0 **only**
    when all three hold:

    * the grader credited it (``score`` is a real positive number), AND
    * it claims a measured result (caller's :func:`leaf_claims_measured_result`), AND
    * **no** successful on-disk metric subject substantiates it
      (``has_disk_evidence`` is False).

    Every other shape is a pass-through (returns the score unchanged, ``vetoed``
    False): unscored/zero leaves, non-result leaves, and result leaves that DO
    have matching on-disk evidence.
    """
    if score is None or score <= 0.0:
        return score, False
    if not claims_result:
        return score, False
    if has_disk_evidence:
        return score, False
    return 0.0, True
