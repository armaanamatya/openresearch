"""Typed ``EvaluationReport`` adapter + ``ScorecardRow`` model (Track E Task 5,
spec ``docs/history/specs/2026-07-10-reproduction-eval-framework-design.md``
§6.1/§6.3, plan ``docs/history/plans/2026-07-10-track-e-eval-scorecard.md``
Task 5).

WHY this exists
---------------
Track E wants a per-run diagnostic scorecard covering 11 evaluator dimensions
plus the new deterministic instrumentation (human-intervention, per-experiment
GPU efficiency), without ever letting a new signal reach ``final_report.verdict``.
``EvaluationReport`` is a thin, read-only VIEW that COMPOSES the existing run
artifacts — it never forks ``RLMFinalReport``'s schema and never recomputes a
verdict:

  * ``verdict`` is copied VERBATIM from ``final_report.json`` (whatever value
    is on disk — this module does not validate, project, or reconcile it).
  * ``scorecard`` is a list of :class:`ScorecardRow` — populated by the CALLER
    (Task 6's ``scorecard.build_scorecard``, out of scope here); ``from_run``
    always starts it empty, since this module owns composition of the report
    shell, not the 11-dimension mapping.
  * ``composite`` is the Task 1 deterministic-dominated
    ``ReproductionScore.composite_score`` value when a caller chooses to
    attach one — display/rank-only, never a verdict driver. ``from_run`` never
    computes it (no experiment-scoring context available at this adapter
    layer); it stays ``None`` unless a caller sets it explicitly.
  * ``provenance_bundle_sha256`` / ``autonomy`` / ``gpu_efficiency`` are
    best-effort reads off their respective sidecars
    (``rlm_state/evidence_bundle.json``, ``human_interventions.jsonl`` via
    :func:`human_intervention.autonomy_metric`, ``gpu_ledger.jsonl`` via
    :func:`gpu_ledger.aggregate_gpu_cost``) — every one of those readers is
    already fail-soft (a missing sidecar degrades to ``None``/an all-zero
    dict), and this module additionally wraps each call so an unexpected
    error can NEVER abort ``from_run``.

North-star invariant (root CLAUDE.md, "Evidence, not grade"): this module
imports NOTHING that writes a verdict. It never calls
``verdict_authority.decide`` and never writes ``verdict``/
``implementation_verdict``/``replication_verdict``/``rubric``/
``meets_target``/``target_score`` anywhere. The only cross-import from the
verdict machinery is ``verdict_authority._VERDICT_RANK`` — a plain severity
LOOKUP TABLE (``{"inconclusive": 0, "contradicted": 1, "partial": 2,
"reproduced": 3}``), never a callable that decides anything.

``gate_caps()`` mirrors this: it is a PRE-``decide()`` input exactly like
``report_claim_gate``'s ``claim_gate_cap`` — a caller may fold its return value
into ``verdict_authority.decide(claim_gate_cap=...)`` BEFORE the verdict is
struck, but ``EvaluationReport`` itself never writes back to ``self.verdict``
(or anywhere else) as a result of calling it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.agents.rlm.evidence_bundle import bundle_path
from backend.agents.rlm.gpu_ledger import aggregate_gpu_cost
from backend.agents.rlm.human_intervention import autonomy_metric
from backend.agents.rlm.verdict_authority import _VERDICT_RANK

__all__ = ["ScorecardRow", "EvaluationReport"]

_FINAL_REPORT_FILENAME = "final_report.json"

ScorecardStatus = Literal["pass", "fail", "unmeasured", "excluded", "display"]
ScorecardProvenance = Literal["paper_reported", "agent_measured", "evaluator_computed"]

# The implied downward verdict cap for a GATING row in each of the two
# capping statuses — mirrors verdict_authority.decide()'s own taxonomy (Rule 2
# "any primary fail" -> contradicted; Rule 3 "any primary unmeasured, none
# failed" -> partial), reusing the SAME severity vocabulary rather than
# inventing a second one. "pass"/"excluded"/"display" never appear here: they
# are excluded structurally by the membership check in gate_caps().
_ROW_STATUS_TO_CAP: dict[str, str] = {
    "fail": "contradicted",
    "unmeasured": "partial",
}


class ScorecardRow(BaseModel):
    """One evaluator-scorecard dimension row (spec §6.1).

    ``gates`` marks a DETERMINISTIC dimension eligible to cap the verdict
    (via :meth:`EvaluationReport.gate_caps`); LLM-judged dimensions are always
    ``gates=False`` (``status="display"``) and can never move the verdict.
    """

    dimension: str
    status: ScorecardStatus
    provenance: ScorecardProvenance
    gates: bool
    evidence_refs: list[str] = Field(default_factory=list)
    detail: str = ""


def _read_verdict(project_dir: Path) -> str:
    """Read ``final_report.json``'s ``verdict`` key, verbatim, read-only.

    Never recomputes, never calls ``verdict_authority.decide``. Fail-soft to
    ``""`` when the file is absent/unreadable/malformed or the key is missing
    or not a string — the field stays a plain ``str`` (never ``None``) so a
    caller can always safely compare/print it.
    """
    try:
        data = json.loads((project_dir / _FINAL_REPORT_FILENAME).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — a missing/corrupt report degrades to "", never raises
        return ""
    if not isinstance(data, dict):
        return ""
    verdict = data.get("verdict")
    return verdict if isinstance(verdict, str) else ""


def _read_provenance_bundle_sha256(project_dir: Path) -> str | None:
    """Best-effort ``metrics_sha256`` off ``rlm_state/evidence_bundle.json``.

    Read directly off disk — independent of
    ``OPENRESEARCH_CANONICAL_EVIDENCE_BUNDLE`` and without re-verifying
    on-disk coherence (that is ``evidence_bundle.resolve_bundle``'s scoring
    concern; this is a read-only display composition). ``None`` when the
    sidecar is absent, unreadable, malformed, or carries no sha.
    """
    try:
        data = json.loads(bundle_path(project_dir).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — an absent/corrupt bundle is just "no receipt yet"
        return None
    if not isinstance(data, dict):
        return None
    sha = data.get("metrics_sha256")
    return sha if isinstance(sha, str) else None


def _safe_autonomy(project_dir: Path) -> dict[str, Any] | None:
    """``human_intervention.autonomy_metric`` is already fail-soft (an absent
    ``human_interventions.jsonl`` degrades to an all-zero stat dict, never
    ``None``); this wrapper additionally guards against an unexpected error
    in that reader itself so it can never abort composition here.
    """
    try:
        return autonomy_metric(project_dir)
    except Exception:  # noqa: BLE001 — telemetry composition must never raise
        return None


def _safe_gpu_efficiency(project_dir: Path) -> dict[str, Any] | None:
    """``gpu_ledger.aggregate_gpu_cost`` is already fail-soft (an absent
    ``gpu_ledger.jsonl`` degrades to an all-zero summary, never ``None``);
    this wrapper additionally guards against an unexpected error in that
    reader itself so it can never abort composition here.
    """
    try:
        return aggregate_gpu_cost(project_dir)
    except Exception:  # noqa: BLE001 — telemetry composition must never raise
        return None


class EvaluationReport(BaseModel):
    """Typed evaluator scorecard — COMPOSES the run report, never forks it.

    ``verdict`` is copied read-only from ``final_report.json`` and is NEVER
    recomputed by this class; nothing in this module calls
    ``verdict_authority.decide`` or writes any verdict/rubric key.
    """

    verdict: str = ""
    scorecard: list[ScorecardRow] = Field(default_factory=list)
    composite: float | None = None
    provenance_bundle_sha256: str | None = None
    autonomy: dict[str, Any] | None = None
    gpu_efficiency: dict[str, Any] | None = None

    @classmethod
    def from_run(cls, project_dir: Path) -> EvaluationReport:
        """Compose an :class:`EvaluationReport` from on-disk artifacts.

        Fail-soft per field: a missing/unreadable artifact degrades ONLY its
        own field (``None`` for the optional sidecar fields, ``""`` for
        ``verdict``, ``[]`` for ``scorecard``) — never raises. ``verdict`` is
        exactly whatever ``final_report.json`` holds today; ``scorecard``
        starts empty (populating the 11 dimensions is Task 6's
        ``build_scorecard``, layered on top of this adapter).
        """
        project_dir = Path(project_dir)
        return cls(
            verdict=_read_verdict(project_dir),
            scorecard=[],
            composite=None,
            provenance_bundle_sha256=_read_provenance_bundle_sha256(project_dir),
            autonomy=_safe_autonomy(project_dir),
            gpu_efficiency=_safe_gpu_efficiency(project_dir),
        )

    def gate_caps(self) -> str | None:
        """The most-severe downward-only verdict cap implied by the scorecard.

        Considers only rows with ``gates=True`` AND ``status`` in
        ``("fail", "unmeasured")``; display rows (``gates=False``) and
        passing/excluded rows contribute NOTHING. Returns ``None`` on an
        empty or all-clear scorecard. This is a PRE-``decide()`` INPUT (like
        ``claim_gate_cap``) — it never writes ``self.verdict`` or anything
        else; a caller who wants the cap enforced must thread it into
        ``verdict_authority.decide(claim_gate_cap=...)`` itself.
        """
        caps = [
            _ROW_STATUS_TO_CAP[row.status]
            for row in self.scorecard
            if row.gates and row.status in _ROW_STATUS_TO_CAP
        ]
        if not caps:
            return None
        return min(caps, key=lambda cap: _VERDICT_RANK.get(cap, 0))

    def to_markdown(self) -> str:
        """Render a human-readable scorecard summary.

        Purely a display rendering of the fields already on this object; it
        does not recompute or re-derive anything (``gate_caps()`` is called
        only to SHOW the implied cap, never to mutate ``verdict``).
        """
        lines: list[str] = [
            "# Evaluation Report",
            "",
            f"**Verdict:** {self.verdict or '(none)'}",
        ]
        if self.composite is not None:
            lines.append(f"**Composite (display-only, never a verdict driver):** {self.composite:.4f}")
        cap = self.gate_caps()
        if cap is not None:
            lines.append(f"**Implied gate cap (pre-decide input, informational only):** {cap}")
        if self.provenance_bundle_sha256:
            lines.append(f"**Evidence-bundle sha256:** {self.provenance_bundle_sha256}")
        if self.autonomy is not None:
            lines.append(f"**Autonomy score (display-only):** {self.autonomy.get('autonomy_score')}")
        if self.gpu_efficiency is not None:
            lines.append(
                f"**GPU cost (display-only):** "
                f"{self.gpu_efficiency.get('total_est_cost_usd')} USD / "
                f"{self.gpu_efficiency.get('total_gpu_hours')} h"
            )
        lines.append("")
        if self.scorecard:
            lines.append("| Dimension | Status | Provenance | Gates | Detail |")
            lines.append("|---|---|---|---|---|")
            for row in self.scorecard:
                gates_str = "yes" if row.gates else "no"
                lines.append(
                    f"| {row.dimension} | {row.status} | {row.provenance} | "
                    f"{gates_str} | {row.detail} |"
                )
        else:
            lines.append("_No scorecard rows._")
        return "\n".join(lines) + "\n"
