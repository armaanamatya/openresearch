"""The 11-dimension scorecard mapper (Track E Task 6, spec
``docs/superpowers/specs/2026-07-10-reproduction-eval-framework-design.md``
§6.1, plan ``docs/superpowers/plans/2026-07-10-track-e-eval-scorecard.md``
Task 6).

WHY this exists
---------------
``EvaluationReport.from_run`` (Task 5) composes the run report SHELL but
deliberately leaves ``scorecard`` empty -- populating the 11 evaluator
dimensions is a SEPARATE concern, layered on top. This module is that layer:
:func:`build_scorecard` maps each dimension onto a
:class:`~backend.evals.evaluation_report.ScorecardRow` using ONLY signals
ALREADY produced elsewhere in the harness (result_fidelity, the evidence
gate's own success predicate, the ok-receipt, env-liveness, the exclusion
detector, the figure sidecars, the human-intervention/GPU-ledger telemetry,
the rubric fidelity axis, the raw experiment/failure/candidate logs). It never
invents a new detector and never fabricates a pass.

Two disjoint row families (spec §6.1):

* **GATE rows** (``gates=True`` -- :data:`GATE_DIMENSIONS`): deterministic,
  artifact-anchored. When the underlying artifact is flatly ABSENT the row is
  ``status="unmeasured"`` -- NEVER auto-``"pass"``. A dimension whose signal
  represents a fair, harness-verified exclusion (a dead RL env, a
  confirmed-unavailable dataset) reports ``status="excluded"`` rather than
  ``"fail"`` -- an excluded leaf is not the agent's fault and must not read as
  a contradiction. ``tables_figures`` is explicitly GATE-*lite*: its detector
  never reaches ``"fail"`` (only ``"pass"``/``"unmeasured"``), since the
  absence of a figure sidecar does not by itself prove a paper needed one.
* **DISPLAY rows** (``gates=False`` -- :data:`DISPLAY_DIMENSIONS`): ALWAYS
  ``status="display"`` -- the literal string never varies, whether or not
  the signal is present. Only ``detail``/``evidence_refs`` change: populated
  when the artifact exists, empty when it does not. A display row can NEVER
  contribute a verdict gate cap (:meth:`EvaluationReport.gate_caps` only ever
  looks at ``gates=True`` rows) -- this is a structural invariant, not a
  per-row check this module has to get right on every call site.

North-star invariant (root CLAUDE.md, "Evidence, not grade"): this module
reads ``final_report.json``'s ``rubric``/``verdict`` fields VERBATIM (via
:func:`fidelity_score_from_rubric`, an existing display-only diagnostic axis)
and never writes back to them, never calls ``verdict_authority.decide``, and
never computes/writes ``meets_target``/``target_score``. Every builder is
wrapped fail-soft in :func:`build_scorecard` -- one bad dimension degrades
only its own row (to the safe unmeasured/display default), never the other
10, and never raises out to a caller composing a finalize report.

Gating: :func:`scorecard_enabled` reads ``OPENRESEARCH_EVAL_SCORECARD``
(default OFF). :func:`write_evaluation_report` is the flag-gated entry point:
off -> returns ``None``, writes nothing (byte-identical); on -> composes
:class:`~backend.evals.evaluation_report.EvaluationReport`, attaches
:func:`build_scorecard`'s rows, and writes
``evaluation_report.{json,md}`` next to the run's other sidecars, returning
the json path.

Out of scope for this module (a separate, later hot-file phase per the plan):
wiring ``write_evaluation_report`` into ``report.py``'s finalize chokepoint or
folding any gate cap into ``verdict_authority.decide``'s pre-decide inputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from backend.agents.rlm import env_liveness
from backend.agents.rlm import result_fidelity
from backend.agents.rlm.feature_flags import env_truthy
from backend.agents.rlm.gpu_ledger import aggregate_gpu_cost
from backend.agents.rlm.human_intervention import autonomy_metric
from backend.agents.rlm.ok_receipt import count_ok_receipts
from backend.agents.rlm.report import _has_experiment_evidence
from backend.agents.rlm.two_axis_report import fidelity_score_from_rubric
from backend.evals.evaluation_report import EvaluationReport, ScorecardRow
from backend.evals.paperbench.leaf_scorer import (
    _detect_data_unavailable_leaves,
    flatten_leaves,
)

__all__ = [
    "GATE_DIMENSIONS",
    "DISPLAY_DIMENSIONS",
    "DIMENSIONS",
    "scorecard_enabled",
    "build_scorecard",
    "write_evaluation_report",
]

_FLAG = "OPENRESEARCH_EVAL_SCORECARD"

# The 5 deterministic, artifact-gated dimensions (spec §6.1 table).
GATE_DIMENSIONS: tuple[str, ...] = (
    "numerical_reproduction",
    "execution_completeness",
    "environment_fidelity",
    "dataset_availability",
    "tables_figures",
)
# The 6 LLM-judged / telemetry dimensions -- display-only, never a cap.
DISPLAY_DIMENSIONS: tuple[str, ...] = (
    "autonomy",
    "efficiency",
    "paper_understanding",
    "dag_planning",
    "debugging",
    "scientific_analysis",
)
DIMENSIONS: tuple[str, ...] = GATE_DIMENSIONS + DISPLAY_DIMENSIONS

_GATE_PROVENANCE = "agent_measured"
_DISPLAY_PROVENANCE = "evaluator_computed"
_RUBRIC_TREE_FILENAMES = ("rubric_tree.json", "generated_rubric.json")


def scorecard_enabled() -> bool:
    """``OPENRESEARCH_EVAL_SCORECARD`` -- default OFF, read at call time."""
    return env_truthy(_FLAG)


# --------------------------------------------------------------------------- #
# Small fail-soft readers shared by the row builders below.
# --------------------------------------------------------------------------- #


def _read_json(path: Path) -> Any:
    """Best-effort parse of ``path`` as JSON. ``None`` on any I/O/parse error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — an absent/corrupt artifact is just "no signal"
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Best-effort parse of ``path`` as JSONL. ``[]`` when absent/corrupt/empty.

    A file that exists but yields zero parseable dict rows is treated exactly
    like an absent file (no information either way) -- callers below key their
    "artifact absent" branch on an empty return, not on ``path.exists()``.
    """
    rows: list[dict[str, Any]] = []
    try:
        if not path.exists():
            return rows
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:  # noqa: BLE001 — tolerate a torn/corrupt line
                continue
            if isinstance(row, dict):
                rows.append(row)
    except Exception:  # noqa: BLE001 — never break the scorecard over one bad log
        return []
    return rows


def _load_rubric_tree(project_dir: Path) -> tuple[dict[str, Any] | None, str]:
    """Read the first usable rubric tree off disk, mirroring
    ``leaf_scorer.finalize_rescore``'s own lookup order (``rubric_tree.json``
    then ``generated_rubric.json``). Returns ``(None, "")`` when neither
    exists or parses to a non-empty dict.
    """
    for name in _RUBRIC_TREE_FILENAMES:
        data = _read_json(project_dir / name)
        if isinstance(data, dict) and data:
            return data, name
    return None, ""


def _gate_row(
    dimension: str, status: str, *, evidence_refs: list[str] | None = None, detail: str = ""
) -> ScorecardRow:
    return ScorecardRow(
        dimension=dimension,
        status=status,
        provenance=_GATE_PROVENANCE,
        gates=True,
        evidence_refs=list(evidence_refs or []),
        detail=detail,
    )


def _display_row(
    dimension: str, *, evidence_refs: list[str] | None = None, detail: str = ""
) -> ScorecardRow:
    return ScorecardRow(
        dimension=dimension,
        status="display",
        provenance=_DISPLAY_PROVENANCE,
        gates=False,
        evidence_refs=list(evidence_refs or []),
        detail=detail,
    )


# --------------------------------------------------------------------------- #
# GATE rows
# --------------------------------------------------------------------------- #


def _row_numerical_reproduction(project_dir: Path) -> ScorecardRow:
    """result_fidelity / repro_spec.json — per-claim numeric checker (§4.2).

    ``unmeasured`` when ``rlm_state/repro_spec.json`` is absent or carries no
    claims at all (never auto-``pass``). Once claims exist, mirrors
    ``VerdictAuthority.decide``'s own vocabulary WITHOUT calling it: any
    contradicted primary -> ``fail``; every primary measured (and none
    contradicted) -> ``pass``; anything else (no genuine primary measured) ->
    ``unmeasured``. This reads the SAME evidence the authority reads —
    ``verdict`` itself is never recomputed or written here.
    """
    dim = "numerical_reproduction"
    repro_spec = _read_json(project_dir / "rlm_state" / "repro_spec.json")
    if not isinstance(repro_spec, dict) or not repro_spec.get("claims"):
        return _gate_row(dim, "unmeasured")

    rf = result_fidelity.evaluate(repro_spec, project_dir)
    per_claim = rf.get("per_claim") or []
    refs = ["rlm_state/repro_spec.json"]
    if (project_dir / "code" / "metrics.json").exists():
        refs.append("code/metrics.json")
    if not per_claim:
        return _gate_row(dim, "unmeasured", evidence_refs=refs)

    n_primary = sum(1 for c in per_claim if c.get("is_primary"))
    n_primary_pass = sum(
        1 for c in per_claim if c.get("is_primary") and c.get("status") == "pass"
    )
    detail = (
        f"{n_primary_pass}/{n_primary} primary claim(s) pass; "
        f"result_fidelity_score={rf.get('result_fidelity_score', 0.0):.3f}"
    )
    if rf.get("any_contradicted"):
        return _gate_row(dim, "fail", evidence_refs=refs, detail=detail)
    if rf.get("primary_all_measured"):
        return _gate_row(dim, "pass", evidence_refs=refs, detail=detail)
    return _gate_row(dim, "unmeasured", evidence_refs=refs, detail=detail)


def _row_execution_completeness(project_dir: Path) -> ScorecardRow:
    """``_has_experiment_evidence`` + the forge-resistant ok-receipt count.

    ``unmeasured`` when ``experiment_runs.jsonl`` is absent/empty (no attempt
    recorded at all -- we genuinely don't know). Once rows exist: an
    in-process ``success=True``+non-empty-metrics row -> ``pass``; absent
    that but a forge-resistant ok-receipt exists (the Task 4 out-of-process
    re-grade fallback) -> ``pass``; otherwise the run DID attempt experiments
    but none cleanly completed -> ``fail``.
    """
    dim = "execution_completeness"
    rows = _read_jsonl(project_dir / "experiment_runs.jsonl")
    if not rows:
        return _gate_row(dim, "unmeasured")

    has_evidence = _has_experiment_evidence(project_dir)
    receipt_count = count_ok_receipts(project_dir)
    refs = ["experiment_runs.jsonl"]
    if receipt_count:
        refs.append("rlm_state/experiment_ok_receipts.jsonl")

    if has_evidence:
        detail = (
            f"{len(rows)} experiment_runs.jsonl row(s); in-process "
            "success+metrics evidence present"
        )
        return _gate_row(dim, "pass", evidence_refs=refs, detail=detail)
    if receipt_count > 0:
        detail = f"{receipt_count} forge-resistant ok-receipt(s) (out-of-process fallback)"
        return _gate_row(dim, "pass", evidence_refs=refs, detail=detail)
    detail = f"{len(rows)} experiment_runs.jsonl row(s); none succeeded with measured metrics"
    return _gate_row(dim, "fail", evidence_refs=refs, detail=detail)


def _row_environment_fidelity(project_dir: Path) -> ScorecardRow:
    """env_health.jsonl exclusions (F2 env-liveness).

    ``unmeasured`` when no ``env_health.jsonl`` is found anywhere under
    ``code/`` (not every paper uses an interactive RL env — absence proves
    nothing). Once health data exists: any env with positive evidence of
    zero served episodes is a verified, fair ``env_setup_failed`` exclusion
    -> ``excluded`` (never ``fail`` — a dead server is not the agent's
    fault); every env served >=1 episode -> ``pass``. Reads
    ``env_liveness.read_env_health`` directly (the unflagged aggregator) and
    reapplies its own documented dead-env predicate, rather than
    ``dead_envs()`` — the latter is itself gated on
    ``OPENRESEARCH_ENV_LIVENESS_GATE``, an unrelated enforcement flag this
    read-only diagnostic must not depend on.
    """
    dim = "environment_fidelity"
    code_dir = project_dir / "code"
    health = env_liveness.read_env_health(code_dir)
    if not health:
        return _gate_row(dim, "unmeasured")

    dead = sorted(
        env
        for env, stats in health.items()
        if stats.get("episodes_total", 0) > 0 and stats.get("episodes_served", 0) == 0
    )
    try:
        refs = [str(p.relative_to(project_dir)) for p in sorted(code_dir.rglob("env_health.jsonl"))][:5]
    except Exception:  # noqa: BLE001 — evidence_refs is cosmetic; never fail the row over it
        refs = []

    if dead:
        detail = (
            f"{len(dead)}/{len(health)} env(s) served 0 episodes "
            f"(verified env_setup_failed exclusion): {', '.join(dead)}"
        )
        return _gate_row(dim, "excluded", evidence_refs=refs, detail=detail)
    detail = f"{len(health)} env(s) all served >=1 episode"
    return _gate_row(dim, "pass", evidence_refs=refs, detail=detail)


def _row_dataset_availability(project_dir: Path) -> ScorecardRow:
    """Data-unavailable rubric leaves (``leaf_scorer._detect_data_unavailable_leaves``).

    ``unmeasured`` when no rubric tree is on disk (nothing to check dataset
    dependencies against). Once a tree with >=1 leaf exists: any leaf tied to
    a harness-verified-unavailable dataset -> ``excluded`` (fair, not a
    failure — the SAME anti-gaming exclusion the leaf scorer itself applies
    at grading/finalize time); none found -> ``pass``.
    """
    dim = "dataset_availability"
    tree, tree_name = _load_rubric_tree(project_dir)
    if tree is None:
        return _gate_row(dim, "unmeasured")

    leaves = flatten_leaves(tree)
    if not leaves:
        return _gate_row(dim, "unmeasured", evidence_refs=[tree_name])

    refs = [tree_name]
    if (project_dir / "code" / "metrics.json").exists():
        refs.append("code/metrics.json")

    unavailable = _detect_data_unavailable_leaves(leaves, project_dir)
    if unavailable:
        detail = (
            f"{len(unavailable)}/{len(leaves)} leaf/leaves depend on a "
            "verified-unavailable dataset"
        )
        return _gate_row(dim, "excluded", evidence_refs=refs, detail=detail)
    detail = f"{len(leaves)} rubric leaf/leaves; no dataset-unavailable signal found"
    return _gate_row(dim, "pass", evidence_refs=refs, detail=detail)


def _row_tables_figures(project_dir: Path) -> ScorecardRow:
    """``fig_*.json`` sidecars under ``code/`` (GATE-lite: never ``fail``).

    ``unmeasured`` when there is no ``code/`` dir, OR ``code/`` exists but no
    ``fig_*.json`` sidecar is found — a text-only paper legitimately needs
    none, so absence alone can never be graded a hard failure (hence
    "GATE-lite": it can only cap through ``unmeasured``, never ``fail``).
    >=1 sidecar found (``leaf_scorer._gather_figure_sidecars``'s own glob) ->
    ``pass``.
    """
    dim = "tables_figures"
    code_dir = project_dir / "code"
    if not code_dir.exists():
        return _gate_row(dim, "unmeasured")

    sidecars = sorted(code_dir.rglob("fig_*.json"))
    if not sidecars:
        return _gate_row(dim, "unmeasured")

    refs = [str(p.relative_to(project_dir)) for p in sidecars[:5]]
    detail = f"{len(sidecars)} figure sidecar(s) found"
    return _gate_row(dim, "pass", evidence_refs=refs, detail=detail)


# --------------------------------------------------------------------------- #
# DISPLAY rows — status is ALWAYS "display"; only detail/evidence_refs vary.
# --------------------------------------------------------------------------- #


def _row_autonomy(project_dir: Path) -> ScorecardRow:
    """``human_intervention.autonomy_metric`` — display-only autonomy stat."""
    dim = "autonomy"
    path = project_dir / "human_interventions.jsonl"
    if not path.exists():
        return _display_row(dim)
    m = autonomy_metric(project_dir)
    detail = (
        f"{m.get('n_interventions', 0)} intervention(s) recorded "
        f"({m.get('n_blocking', 0)} blocking); "
        f"autonomy_score={m.get('autonomy_score', 0.0):.3f}"
    )
    return _display_row(dim, evidence_refs=["human_interventions.jsonl"], detail=detail)


def _row_efficiency(project_dir: Path) -> ScorecardRow:
    """``gpu_ledger.aggregate_gpu_cost`` — display-only per-run GPU efficiency."""
    dim = "efficiency"
    path = project_dir / "gpu_ledger.jsonl"
    if not path.exists():
        return _display_row(dim)
    agg = aggregate_gpu_cost(project_dir)
    detail = (
        f"{agg.get('total_gpu_hours', 0.0):.3f} GPU-hour(s), "
        f"~${agg.get('total_est_cost_usd', 0.0):.2f} estimated"
    )
    return _display_row(dim, evidence_refs=["gpu_ledger.jsonl"], detail=detail)


def _row_paper_understanding(project_dir: Path) -> ScorecardRow:
    """``fidelity_score_from_rubric`` over ``final_report.json``'s ``rubric``.

    ``rubric`` is READ VERBATIM (never recomputed/written back — it is one of
    the read-only verdict-adjacent surfaces); the resulting fidelity score is
    already the diagnostic (non-headline-verdict) axis
    ``two_axis_report.compute_and_attach`` itself uses.
    """
    dim = "paper_understanding"
    report = _read_json(project_dir / "final_report.json")
    rubric = report.get("rubric") if isinstance(report, dict) else None
    if not isinstance(rubric, dict) or not rubric:
        return _display_row(dim)
    score = fidelity_score_from_rubric(rubric)
    detail = f"fidelity_score_from_rubric={score:.3f} (diagnostic; never the headline verdict)"
    return _display_row(dim, evidence_refs=["final_report.json#rubric"], detail=detail)


def _row_dag_planning(project_dir: Path) -> ScorecardRow:
    """Post-hoc S0 sequence over ``experiment_runs.jsonl`` (Track G upgrades this to S1)."""
    dim = "dag_planning"
    rows = _read_jsonl(project_dir / "experiment_runs.jsonl")
    if not rows:
        return _display_row(dim)
    detail = (
        f"{len(rows)} experiment attempt(s) observed in post-hoc linear order "
        "(S0; no recorded dependency graph yet)"
    )
    return _display_row(dim, evidence_refs=["experiment_runs.jsonl"], detail=detail)


def _row_debugging(project_dir: Path) -> ScorecardRow:
    """Failure capsules (``OPENRESEARCH_FAILURE_CAPSULES``' own artifact)."""
    dim = "debugging"
    rows = _read_jsonl(project_dir / "rlm_state" / "failure_capsules.jsonl")
    if not rows:
        return _display_row(dim)
    classes = sorted({str(r.get("failure_class") or "unknown") for r in rows})
    detail = (
        f"{len(rows)} failure capsule(s) across {len(classes)} class(es): "
        f"{', '.join(classes[:5])}"
    )
    return _display_row(dim, evidence_refs=["rlm_state/failure_capsules.jsonl"], detail=detail)


def _row_scientific_analysis(project_dir: Path) -> ScorecardRow:
    """``candidate_proposed``/``candidate_outcome`` counts off ``dashboard_events.jsonl``.

    ``HypothesisScore``/``IntegrityReport`` (``backend/evals/schemas.py``) are
    typed LLM-judge rubrics with no existing per-run on-disk producer — this
    module does not invent one. The best AVAILABLE grounded signal is a plain
    count of the already-recorded candidate-proposal/outcome events; absent
    those, the row stays empty rather than fabricating a score.
    """
    dim = "scientific_analysis"
    rows = _read_jsonl(project_dir / "dashboard_events.jsonl")
    n_proposed = sum(1 for r in rows if r.get("event") == "candidate_proposed")
    n_outcome = sum(1 for r in rows if r.get("event") == "candidate_outcome")
    if not n_proposed and not n_outcome:
        return _display_row(dim)
    detail = (
        f"{n_proposed} candidate hypothesis proposal(s), {n_outcome} outcome(s) recorded "
        "(HypothesisScore/IntegrityReport not computed by this build)"
    )
    return _display_row(dim, evidence_refs=["dashboard_events.jsonl"], detail=detail)


_GATE_BUILDERS: dict[str, Callable[[Path], ScorecardRow]] = {
    "numerical_reproduction": _row_numerical_reproduction,
    "execution_completeness": _row_execution_completeness,
    "environment_fidelity": _row_environment_fidelity,
    "dataset_availability": _row_dataset_availability,
    "tables_figures": _row_tables_figures,
}
_DISPLAY_BUILDERS: dict[str, Callable[[Path], ScorecardRow]] = {
    "autonomy": _row_autonomy,
    "efficiency": _row_efficiency,
    "paper_understanding": _row_paper_understanding,
    "dag_planning": _row_dag_planning,
    "debugging": _row_debugging,
    "scientific_analysis": _row_scientific_analysis,
}


def build_scorecard(project_dir: Path) -> list[ScorecardRow]:
    """Map all 11 evaluator dimensions onto :class:`ScorecardRow`\\s.

    Fail-soft PER DIMENSION: an unexpected error in one builder degrades only
    that row (to the safe ``unmeasured``/``display`` default), never blanks
    the rest of the scorecard and never raises into the caller. Always
    returns exactly ``len(DIMENSIONS)`` rows, in the stable
    ``GATE_DIMENSIONS + DISPLAY_DIMENSIONS`` order.
    """
    project_dir = Path(project_dir)
    rows: list[ScorecardRow] = []
    for dim in GATE_DIMENSIONS:
        try:
            rows.append(_GATE_BUILDERS[dim](project_dir))
        except Exception:  # noqa: BLE001 — one bad dimension must never blank the scorecard
            rows.append(_gate_row(dim, "unmeasured"))
    for dim in DISPLAY_DIMENSIONS:
        try:
            rows.append(_DISPLAY_BUILDERS[dim](project_dir))
        except Exception:  # noqa: BLE001
            rows.append(_display_row(dim))
    return rows


def write_evaluation_report(project_dir: Path) -> Path | None:
    """Flag-gated: compose + write ``evaluation_report.{json,md}``.

    Off (``OPENRESEARCH_EVAL_SCORECARD`` unset/falsy) -> returns ``None``,
    writes nothing (byte-identical to a build with no scorecard module at
    all). On -> ``EvaluationReport.from_run(project_dir)`` (verdict copied
    read-only), ``scorecard`` set to :func:`build_scorecard`'s rows, both
    sidecars written next to the run's other artifacts. Fail-soft: any
    internal error (bad ``project_dir``, a write failure) degrades to
    ``None`` rather than raising — this is report-adjacent telemetry, never
    allowed to break a finalize path.
    """
    if not scorecard_enabled():
        return None
    try:
        project_dir = Path(project_dir)
        report = EvaluationReport.from_run(project_dir)
        report.scorecard = build_scorecard(project_dir)
        json_path = project_dir / "evaluation_report.json"
        md_path = project_dir / "evaluation_report.md"
        json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        md_path.write_text(report.to_markdown(), encoding="utf-8")
        return json_path
    except Exception:  # noqa: BLE001 — never break a finalize path over a sidecar write
        return None
