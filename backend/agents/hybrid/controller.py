"""Hybrid RDR+RLM controller — the default ``--mode rlm`` run path.

Phase 1: Deterministic RDR initial pass (``max_repair_iterations=0``).  Every
cluster gets one agent call; the rubric is scored; weak clusters are identified.

Phase 2 (conditional): If any leaf scores below ``repair_target``, the RLM
adaptive engine is invoked on the same project dir, seeded with Phase 1's code
and weak-cluster justifications.  Bounded by ``max_iterations`` and ``max_usd``.

Both phases share the same ``project_id`` and ``runs_root`` so all artifacts
accumulate in a single run directory.  The final ``final_report.json`` is
the Phase 2 report when Phase 2 runs, or the Phase 1 report otherwise.

Signature mirrors ``run_pipeline_rlm`` for drop-in CLI / live_runs dispatch.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from backend.agents.rlm.run import RLMRunResult

logger = logging.getLogger(__name__)

# Target cluster score below which Phase 2 repair is triggered.
_DEFAULT_REPAIR_TARGET: float = 0.6


def _build_workspace_claim_map_from_bundle(bundle: Any, project_id: str) -> dict[str, Any]:
    """Convert a PaperBenchBundle into the workspace_claim_map shape expected by
    ``run_pipeline_rlm``.  Used when the hybrid is called with a bundle object.

    If the caller already provides a ``workspace_claim_map``, this is not called.
    """
    from backend.services.ingestion.paperbench import bundle_to_workspace_claim_map

    claim_map = bundle_to_workspace_claim_map(bundle)
    claim_map["project_id"] = project_id
    claim_map["rubric_spec"] = bundle.rubric()
    return claim_map


def _extract_weak_clusters(
    phase1_report_path: str,
    repair_target: float,
) -> list[dict[str, Any]]:
    """Read the Phase 1 ``final_report.json`` and return weak leaf entries.

    Returns a list of ``{"id": ..., "score": ..., "justification": ...}``
    for leaves whose score is below *repair_target*.  Returns ``[]`` on any
    read/parse failure (fail-soft: the caller treats no weak clusters as
    Phase 1 sufficient).
    """
    try:
        data = json.loads(Path(phase1_report_path).read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(
            "hybrid/controller: could not read phase1 report at %s (%s: %s) — "
            "treating all leaves as passing (skipping Phase 2)",
            phase1_report_path, type(exc).__name__, exc,
        )
        return []

    rubric = data.get("rubric") or {}
    leaf_scores: list[dict[str, Any]] = rubric.get("leaf_scores") or []
    weak: list[dict[str, Any]] = []
    for entry in leaf_scores:
        if not isinstance(entry, dict):
            continue
        score = float(entry.get("score", 0.0))
        if score < repair_target:
            weak.append({
                "id": entry.get("id", ""),
                "score": score,
                "justification": entry.get("justification", ""),
            })
    return weak


async def run_pipeline_hybrid(
    project_id: str,
    runs_root: Path,
    workspace_claim_map: dict[str, Any],
    *,
    model: str | None = None,
    provider: str | None = None,
    runtime: Any = None,
    run_budget: Any = None,
    sandbox_mode: Any = None,
    seed: int | None = None,
    execution_profile: Any = None,
    attempt_id: str | None = None,
    run_group_id: str | None = None,
    workspace_service: Any = None,
    workspace_id: str | None = None,
    repair_target: float = _DEFAULT_REPAIR_TARGET,
    # Internal: injected by tests to avoid real I/O.
    _rdr_runner: Any = None,
    _rlm_runner: Any = None,
) -> RLMRunResult:
    """Run the hybrid Phase 1 (RDR) + Phase 2 (RLM repair) pipeline.

    Args:
        project_id: Unique run identifier; determines the output directory.
        runs_root: Root directory under which ``<project_id>/`` is created.
        workspace_claim_map: Standard RLM claim map.  Must carry a
            ``rubric_spec`` key when coming from a PaperBench bundle run.
        model: Root model override forwarded to both phases.
        provider: LLM provider hint, forwarded to both phases.
        runtime: Sub-agent runtime override (tests / callers); forwarded.
        run_budget: ``RunBudget`` controlling cost + wall-clock for Phase 2.
        sandbox_mode: Sandbox for experiment execution, forwarded to both phases.
        seed / execution_profile / attempt_id / run_group_id: parity args.
        workspace_service / workspace_id: parity args.
        repair_target: Leaf score threshold below which Phase 2 is triggered.
        _rdr_runner / _rlm_runner: Injection points for unit tests.

    Returns:
        An :class:`RLMRunResult` — Phase 2's result when Phase 2 ran, Phase 1's
        result (translated) when Phase 1 was sufficient, or Phase 1's partial
        result when Phase 1 failed.
    """
    from backend.agents.rdr.run import run_pipeline_rdr
    from backend.agents.rlm.run import run_pipeline_rlm
    from backend.agents.execution import DEFAULT_SANDBOX_MODE

    _rdr = _rdr_runner if _rdr_runner is not None else run_pipeline_rdr
    _rlm = _rlm_runner if _rlm_runner is not None else run_pipeline_rlm

    runs_root = Path(runs_root).resolve()

    # --- Phase 1: RDR initial pass (no repair) ---
    # Extract bundle args from workspace_claim_map; required for run_pipeline_rdr.
    pb = workspace_claim_map.get("paperbench") or {}
    paper_id: str = (
        pb.get("paper_id")
        or (pb.get("metadata") or {}).get("id")
        or workspace_claim_map.get("project_id", "")
    )

    bundles_root: str | Path | None = workspace_claim_map.get("_bundles_root")

    # --- Bundle-presence guard ---
    # Phase 1 RDR requires a vendored PaperBench bundle at
    # third_party/paperbench/<paper_id>. For PDF/arXiv uploads (project_id=prj_*),
    # there is no bundle — RDR would raise FileNotFoundError and kill the run.
    # When no bundle, dispatch directly to pure RLM (which handles arXiv/PDF
    # via rubric_gen.py inside run_pipeline_rlm).
    from pathlib import Path as _Path
    if bundles_root is not None:
        _bundle_dir = _Path(bundles_root) / paper_id
    else:
        # Default bundles location: repo_root/third_party/paperbench
        _bundle_dir = _Path(__file__).resolve().parents[3] / "third_party" / "paperbench" / paper_id
    if not _bundle_dir.is_dir():
        logger.info(
            "hybrid/controller[%s]: no PaperBench bundle at %s for paper_id=%r "
            "— dispatching pure RLM (no Phase 1 RDR)",
            project_id, _bundle_dir, paper_id,
        )
        return await _rlm(
            project_id,
            runs_root,
            workspace_claim_map,
            model=model,
            provider=provider,
            runtime=runtime,
            run_budget=run_budget,
            sandbox_mode=sandbox_mode if sandbox_mode is not None else DEFAULT_SANDBOX_MODE,
            seed=seed,
            execution_profile=execution_profile,
            attempt_id=attempt_id,
            run_group_id=run_group_id,
            workspace_service=workspace_service,
            workspace_id=workspace_id,
        )

    logger.info(
        "hybrid/controller[%s]: Phase 1 (RDR, no repair) — paper_id=%r",
        project_id, paper_id,
    )

    phase1_result = None
    phase1_failed = False
    try:
        phase1_result = await _rdr(
            project_id,
            runs_root,
            paper_id=paper_id,
            provider=provider,
            model=model,
            sandbox_mode=sandbox_mode if sandbox_mode is not None else DEFAULT_SANDBOX_MODE,
            max_repair_iterations=0,   # Phase 1 only — no RDR repair
            repair_target=repair_target,
            bundles_root=bundles_root,
        )
    except Exception as exc:
        phase1_failed = True
        logger.error(
            "hybrid/controller[%s]: Phase 1 (RDR) raised %s: %s — returning partial result",
            project_id, type(exc).__name__, exc,
        )

    # If Phase 1 errored, surface the failure without spawning Phase 2.
    if phase1_failed or phase1_result is None:
        return RLMRunResult(
            project_id=project_id,
            status="failed",
            iterations=0,
            rubric_score=None,
            cost_usd=None,
            final_report_path=None,
        )

    rubric_score: float = phase1_result.rubric_score or 0.0
    final_report_path: str | None = phase1_result.final_report_path

    # Emit a Phase 1 summary to the log.
    logger.info(
        "hybrid/controller[%s]: Phase 1 done — score=%.3f clusters=%d/%d failed repair_iterations=%d",
        project_id,
        rubric_score,
        phase1_result.clusters_failed,
        phase1_result.clusters_total,
        phase1_result.repair_iterations,
    )

    # Skip Phase 2 when Phase 1 produced no usable code — there is nothing to
    # repair when every cluster failed. Saves budget on a hopeless adaptive pass.
    if (
        phase1_result.clusters_total > 0
        and phase1_result.clusters_failed == phase1_result.clusters_total
    ):
        logger.warning(
            "hybrid/controller[%s]: Phase 1 produced no working clusters "
            "(%d/%d failed) — skipping Phase 2 (no code to repair)",
            project_id, phase1_result.clusters_failed, phase1_result.clusters_total,
        )
        return RLMRunResult(
            project_id=phase1_result.project_id,
            status="failed",
            iterations=phase1_result.clusters_total,
            rubric_score=rubric_score,
            cost_usd=phase1_result.cost_usd,
            final_report_path=final_report_path,
        )

    # --- Decide whether Phase 2 is needed ---
    weak_clusters: list[dict[str, Any]] = []
    if final_report_path:
        weak_clusters = _extract_weak_clusters(final_report_path, repair_target)

    if not weak_clusters:
        logger.info(
            "hybrid/controller[%s]: all leaves meet target (%.2f) — skipping Phase 2",
            project_id, repair_target,
        )
        # Translate RdrResult → RLMRunResult for a uniform return type.
        return RLMRunResult(
            project_id=phase1_result.project_id,
            status=phase1_result.status,
            iterations=phase1_result.clusters_total,
            rubric_score=rubric_score,
            cost_usd=phase1_result.cost_usd,
            final_report_path=final_report_path,
        )

    # --- Phase 2: RLM adaptive repair ---
    logger.info(
        "hybrid/controller[%s]: %d weak cluster(s) — launching Phase 2 (RLM repair)",
        project_id, len(weak_clusters),
    )

    # Decrement Phase 2's budget by what Phase 1 already spent so total
    # cost honors the user's max_usd ceiling. If Phase 1 already consumed
    # the budget, skip Phase 2 entirely.
    remaining_budget = run_budget
    if run_budget is not None and getattr(run_budget, "max_usd", None) is not None:
        phase1_cost = float(phase1_result.cost_usd or 0.0)
        remaining_usd = max(0.0, float(run_budget.max_usd) - phase1_cost)
        if remaining_usd <= 0.0:
            logger.warning(
                "hybrid/controller[%s]: Phase 1 consumed full budget "
                "($%.4f >= $%.4f) — skipping Phase 2",
                project_id, phase1_cost, float(run_budget.max_usd),
            )
            return RLMRunResult(
                project_id=phase1_result.project_id,
                status=phase1_result.status,
                iterations=phase1_result.clusters_total,
                rubric_score=rubric_score,
                cost_usd=phase1_cost,
                final_report_path=final_report_path,
            )
        from backend.agents.resilience.budget import RunBudget
        remaining_budget = RunBudget(
            max_usd=remaining_usd,
            max_wall_clock_seconds=getattr(run_budget, "max_wall_clock_seconds", None),
        )

    # Seed the claim map with Phase 1 artifacts.
    phase2_result = await _rlm(
        project_id,           # same project dir — Phase 2 continues where Phase 1 left off
        runs_root,
        workspace_claim_map,
        model=model,
        provider=provider,
        runtime=runtime,
        run_budget=remaining_budget,
        sandbox_mode=sandbox_mode if sandbox_mode is not None else DEFAULT_SANDBOX_MODE,
        seed=seed,
        execution_profile=execution_profile,
        attempt_id=attempt_id,
        run_group_id=run_group_id,
        workspace_service=workspace_service,
        workspace_id=workspace_id,
        hybrid_repair_only=True,
        phase1_weak_clusters=weak_clusters,
    )

    logger.info(
        "hybrid/controller[%s]: Phase 2 done — status=%s score=%s iterations=%d",
        project_id,
        phase2_result.status,
        phase2_result.rubric_score,
        phase2_result.iterations,
    )
    return phase2_result


__all__ = ["run_pipeline_hybrid"]
