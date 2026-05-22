"""Phase-4 RDR Controller — deterministic control flow for rubric-cluster reproduction.

``run_rdr`` owns all control flow: decompose → cluster loop → assemble → env →
experiment → score → repair loop → report.  No LLM in the control path; the
controller is pure Python except for the async agent calls it dispatches.

See ``docs/superpowers/specs/2026-05-22-rubric-driven-harness-design.md`` §7.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from backend.agents.rdr.decomposer import decompose
from backend.agents.rdr.context_engineer import build_context
from backend.agents.rdr.models import Artifacts, RdrResult, WorkCluster
from backend.agents.rlm.primitives import (
    detect_environment,
    build_environment,
    run_experiment,
)
from backend.evals.paperbench.leaf_scorer import score_reproduction
from backend.agents.rlm.report import (
    RLMFinalReport,
    reconcile_verdict_with_score,
    write_final_report_rlm,
)

if TYPE_CHECKING:
    from backend.agents.rlm.context import RunContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cluster_score(cluster: WorkCluster, scores: dict[str, Any]) -> float:
    """Weighted average score for *cluster*'s leaves from the leaf_scores list.

    Leaves absent from leaf_scores are treated as 0.0 (conservative).
    """
    leaf_scores_list: list[dict[str, Any]] = scores.get("leaf_scores", [])
    by_id: dict[str, float] = {
        entry["id"]: float(entry.get("score", 0.0))
        for entry in leaf_scores_list
        if isinstance(entry, dict) and "id" in entry
    }
    total_weight = sum(leaf.weight for leaf in cluster.leaves)
    if total_weight == 0.0:
        return 0.0
    weighted_sum = sum(
        by_id.get(leaf.id, 0.0) * leaf.weight for leaf in cluster.leaves
    )
    return weighted_sum / total_weight


def _dedup_commands(done: dict[str, Artifacts]) -> list[str]:
    """Return deduplication-preserving union of all cluster commands."""
    seen: dict[str, None] = {}  # ordered set
    for art in done.values():
        for cmd in art.commands:
            seen[cmd] = None
    return list(seen)


def _write_cluster_checkpoint(
    iterations_dir: Path,
    index: int,
    cluster: WorkCluster,
    art: Artifacts,
) -> None:
    """Write a per-cluster JSON checkpoint under ``iterations/``."""
    iterations_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "cluster_id": cluster.id,
        "cluster_title": cluster.title,
        "leaf_ids": [leaf.id for leaf in cluster.leaves],
        "failed": art.failed,
        "notes": art.notes,
        "file_count": len(art.files),
    }
    path = iterations_dir / f"cluster_{index}_{cluster.id}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _write_repl_state(
    project_dir: Path,
    clusters: list[WorkCluster],
    done: dict[str, Artifacts],
    scores: dict[str, Any],
    repair_iterations: int,
) -> None:
    """Pickle a redacted run-state dict to ``repl_state.pickle``.

    MUST NOT contain raw paper corpus text or full file contents.
    """
    state = {
        "clusters_summary": [
            {
                "id": c.id,
                "title": c.title,
                "leaf_count": len(c.leaves),
                "weight": c.weight,
                "dominant_category": c.dominant_category,
            }
            for c in clusters
        ],
        "artifacts_summary": {
            cid: {
                "file_count": len(art.files),
                "failed": art.failed,
                "command_count": len(art.commands),
            }
            for cid, art in done.items()
        },
        "scores": {
            "overall_score": scores.get("overall_score"),
            "leaf_count": scores.get("leaf_count"),
            "graded": scores.get("graded"),
        },
        "repair_iterations": repair_iterations,
    }
    path = project_dir / "repl_state.pickle"
    tmp = path.with_suffix(".pickle.tmp")
    tmp.write_bytes(pickle.dumps(state, protocol=4))
    os.replace(tmp, path)


def _write_repair_checkpoint(
    iterations_dir: Path,
    rep_n: int,
    cluster: WorkCluster,
    art: Artifacts,
) -> None:
    """Write a per-cluster JSON checkpoint for a repair pass under ``iterations/``.

    File name: ``repair_<rep_n>_cluster_<cluster_id>.json``.
    Shape mirrors the initial checkpoint plus a ``repair_pass`` field.
    """
    iterations_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "cluster_id": cluster.id,
        "cluster_title": cluster.title,
        "leaf_ids": [leaf.id for leaf in cluster.leaves],
        "failed": art.failed,
        "file_count": len(art.files),
        "repair_pass": rep_n,
    }
    path = iterations_dir / f"repair_{rep_n}_cluster_{cluster.id}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _resolve_reproduce_fn(reproduce_fn: Callable | None) -> Callable:
    """Return the agent callable: injected or lazily-imported real one."""
    if reproduce_fn is not None:
        return reproduce_fn

    def _lazy_reproduce(
        agent_context: Any, *, ctx: Any
    ) -> Coroutine[Any, Any, Artifacts]:
        from backend.agents.rdr.agent import reproduce  # lazy — agent.py built in parallel

        return reproduce(agent_context, ctx=ctx)

    return _lazy_reproduce


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_rdr(
    bundle: Any,
    *,
    ctx: "RunContext",
    max_repair_iterations: int = 2,
    repair_target: float = 0.6,
    max_leaves_per_cluster: int = 12,
    reproduce_fn: Callable | None = None,
) -> RdrResult:
    """Deterministic controller for a rubric-driven paper reproduction run.

    Args:
        bundle: A ``PaperBenchBundle`` (see ``backend.evals.paperbench.bundle``).
        ctx: Run-scoped context (paths, LLM client, cost ledger, …).
        max_repair_iterations: Maximum repair loops after initial scoring.
        repair_target: Cluster-level score threshold below which a cluster
            is flagged as weak and queued for a repair pass.
        max_leaves_per_cluster: Passed directly to ``decompose``.
        reproduce_fn: Injected async agent callable — signature
            ``reproduce_fn(agent_context, *, ctx) -> Artifacts``.
            Defaults to the real ``backend.agents.rdr.agent.reproduce``
            (lazy-imported so this module stays importable even if agent.py
            does not exist yet).

    Returns:
        An :class:`RdrResult` — always; per-cluster and per-phase failures
        are fail-soft and produce an honest partial or completed result.
    """
    _reproduce = _resolve_reproduce_fn(reproduce_fn)

    # ------------------------------------------------------------------
    # Step 1: Decompose
    # ------------------------------------------------------------------
    rubric = bundle.rubric()
    clusters: list[WorkCluster] = decompose(
        rubric, max_leaves_per_cluster=max_leaves_per_cluster
    )
    logger.info(
        "run_rdr[%s]: %d clusters from rubric", ctx.project_id, len(clusters)
    )

    # ------------------------------------------------------------------
    # Step 2: Write paper file; ensure code dir exists
    # ------------------------------------------------------------------
    paper: str = bundle.read_paper_markdown()
    paper_full_path = ctx.project_dir / "paper_full.md"
    paper_full_path.write_text(paper, encoding="utf-8")

    code_dir = ctx.project_dir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)

    iterations_dir = ctx.project_dir / "iterations"
    iterations_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 3: Cluster loop
    # ------------------------------------------------------------------
    done: dict[str, Artifacts] = {}
    clusters_failed = 0

    for idx, cluster in enumerate(clusters):
        agctx = build_context(
            cluster,
            paper=paper,
            artifacts=done,
            prior_scores=None,
        )
        try:
            art = await _reproduce(agctx, ctx=ctx)
        except Exception as exc:  # noqa: BLE001 — per-cluster fail-soft
            logger.warning(
                "run_rdr[%s]: cluster %s raised %s: %s — marking failed",
                ctx.project_id, cluster.id, type(exc).__name__, exc,
            )
            art = Artifacts(
                cluster_id=cluster.id,
                failed=True,
                error=f"{type(exc).__name__}: {exc}",
            )

        done[cluster.id] = art

        # Merge files into the shared code/ dir (agent writes its own copies;
        # files dict carries the canonical content from the agent's
        # perspective).  Defensive: skip files we can't write (e.g. cache
        # locks with restricted perms that slipped past the agent-side
        # exclusions) so a single bad path doesn't kill the run.
        for rel_path, content in art.files.items():
            dest = code_dir / rel_path
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content, encoding="utf-8")
            except (PermissionError, OSError) as exc:
                logger.warning(
                    "rdr/controller: skipping %r (%s: %s)",
                    str(dest), type(exc).__name__, exc,
                )

        _write_cluster_checkpoint(iterations_dir, idx, cluster, art)

        if art.failed:
            clusters_failed += 1
            logger.warning(
                "run_rdr[%s]: cluster %s failed: %s",
                ctx.project_id, cluster.id, art.error,
            )

    # ------------------------------------------------------------------
    # Step 4: Assemble — write commands.json
    # ------------------------------------------------------------------
    commands = _dedup_commands(done)
    if not commands:
        commands = ["python train.py"]
    (code_dir / "commands.json").write_text(
        json.dumps(commands), encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # Step 5: Environment detect + build (fail-soft)
    #
    # If the agent wrote a Dockerfile into code/, promote it to project_dir/
    # so that run_experiment picks it up.  When a code/Dockerfile is present
    # we skip detect_environment (agent already specified the env) and build
    # directly from the promoted file.
    # ------------------------------------------------------------------
    meta = bundle.metadata()
    env_id: str = ""
    try:
        agent_dockerfile = code_dir / "Dockerfile"
        root_dockerfile = ctx.project_dir / "Dockerfile"
        if agent_dockerfile.exists():
            # Promote to project_dir/Dockerfile (overwrite).
            shutil.copy2(agent_dockerfile, root_dockerfile)
            logger.info(
                "run_rdr[%s]: agent-supplied Dockerfile promoted to %s",
                ctx.project_id, root_dockerfile,
            )
            dockerfile_content = root_dockerfile.read_text(encoding="utf-8")
            env_spec: dict[str, Any] = {"dockerfile": dockerfile_content}
        else:
            method_spec = {"core_contribution": meta.get("title", "")}
            env_spec = detect_environment(method_spec, ctx=ctx)
            if env_spec.get("success") is False:
                logger.warning(
                    "run_rdr[%s]: detect_environment failed: %s",
                    ctx.project_id, env_spec.get("error"),
                )
                env_spec = {}

        if env_spec:
            build = build_environment(env_spec, ctx=ctx)
            if build.get("ok"):
                env_id = build.get("image_tag", "")
            else:
                logger.warning(
                    "run_rdr[%s]: build_environment failed (%d attempts): %s",
                    ctx.project_id,
                    build.get("attempts", 0),
                    build.get("error"),
                )
    except Exception as exc:  # noqa: BLE001 — env chain is fail-soft
        logger.warning(
            "run_rdr[%s]: env detect/build raised %s: %s — skipping experiment",
            ctx.project_id, type(exc).__name__, exc,
        )

    # ------------------------------------------------------------------
    # Step 6: Experiment (fail-soft; only if env_id available)
    # ------------------------------------------------------------------
    exp: dict[str, Any] = {"success": False, "metrics": {}}
    if env_id:
        try:
            exp = run_experiment(str(code_dir), env_id, ctx=ctx)
        except Exception as exc:  # noqa: BLE001 — fail-soft
            logger.warning(
                "run_rdr[%s]: run_experiment raised %s: %s",
                ctx.project_id, type(exc).__name__, exc,
            )

    # ------------------------------------------------------------------
    # Step 7: Initial scoring
    # ------------------------------------------------------------------
    scores = score_reproduction(rubric, ctx.project_dir, ctx.llm_client)

    # ------------------------------------------------------------------
    # Step 8: Repair loop
    #
    # Track the actual number of agent dispatches across initial + repair
    # passes so the report's ``iterations`` field is accurate.
    # ------------------------------------------------------------------
    repair_iterations = 0
    # Initial pass already dispatched one call per cluster.
    total_agent_dispatches = len(clusters)

    for _rep in range(max_repair_iterations):
        weak = [c for c in clusters if _cluster_score(c, scores) < repair_target]
        if not weak:
            break

        rep_n = repair_iterations + 1
        logger.info(
            "run_rdr[%s]: repair iteration %d — %d weak clusters",
            ctx.project_id, rep_n, len(weak),
        )

        for cluster in weak:
            agctx = build_context(
                cluster,
                paper=paper,
                artifacts=done,
                prior_scores=scores,
            )
            try:
                art = await _reproduce(agctx, ctx=ctx)
            except Exception as exc:  # noqa: BLE001 — per-cluster fail-soft
                logger.warning(
                    "run_rdr[%s]: repair cluster %s raised %s: %s",
                    ctx.project_id, cluster.id, type(exc).__name__, exc,
                )
                art = Artifacts(
                    cluster_id=cluster.id,
                    failed=True,
                    error=f"{type(exc).__name__}: {exc}",
                )
            done[cluster.id] = art
            total_agent_dispatches += 1

            # Merge repaired files back into code/ (defensive — see the
            # initial-pass note above).
            for rel_path, content in art.files.items():
                dest = code_dir / rel_path
                try:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(content, encoding="utf-8")
                except (PermissionError, OSError) as exc:
                    logger.warning(
                        "rdr/controller: skipping %r (%s: %s)",
                        str(dest), type(exc).__name__, exc,
                    )

            # Write a repair-pass checkpoint alongside the initial cluster checkpoints.
            _write_repair_checkpoint(iterations_dir, rep_n, cluster, art)

        # Re-assemble commands
        commands = _dedup_commands(done)
        if not commands:
            commands = ["python train.py"]
        (code_dir / "commands.json").write_text(
            json.dumps(commands), encoding="utf-8"
        )

        # Re-experiment (only if env is available)
        if env_id:
            try:
                exp = run_experiment(str(code_dir), env_id, ctx=ctx)
            except Exception as exc:  # noqa: BLE001 — fail-soft
                logger.warning(
                    "run_rdr[%s]: repair run_experiment raised %s: %s",
                    ctx.project_id, type(exc).__name__, exc,
                )

        # Re-score
        scores = score_reproduction(rubric, ctx.project_dir, ctx.llm_client)
        repair_iterations += 1

    # ------------------------------------------------------------------
    # Step 9: Write final report
    # ------------------------------------------------------------------
    overall_score: float = float(scores.get("overall_score", 0.0))
    verdict = reconcile_verdict_with_score("partial", overall_score)

    # Deterministic summary — no LLM
    clusters_failed_count = sum(1 for art in done.values() if art.failed)
    reproduction_summary = (
        f"RDR run: {len(clusters)} cluster(s), "
        f"{clusters_failed_count} failed, "
        f"{repair_iterations} repair iteration(s). "
        f"Overall rubric score: {overall_score:.3f}."
    )

    cost_dict: dict[str, Any] = {}
    if ctx.cost_ledger is not None:
        try:
            cost_dict = {"llm_usd": ctx.cost_ledger.total_usd(), "primitives": ctx.cost_ledger.total_usd()}
        except Exception:  # noqa: BLE001
            pass

    # total_agent_dispatches counts the actual number of agent calls:
    # one per cluster on the initial pass + one per weak cluster per repair pass.
    report = RLMFinalReport(
        paper=meta,
        verdict=verdict,
        reproduction_summary=reproduction_summary,
        baseline_metrics=exp.get("metrics") or {},
        paper_claims={},
        rubric=scores,
        improvements=[],
        primitive_trace={},
        cost=cost_dict or {"llm_usd": 0.0, "primitives": 0.0},
        iterations=total_agent_dispatches,
    )

    json_path, _md_path = write_final_report_rlm(report, ctx.project_dir)

    # ------------------------------------------------------------------
    # Step 10: DC#4 artifacts — repl_state.pickle
    # ------------------------------------------------------------------
    _write_repl_state(
        ctx.project_dir, clusters, done, scores, repair_iterations
    )

    # ------------------------------------------------------------------
    # Step 11: Return RdrResult
    # ------------------------------------------------------------------
    cost_usd: float | None = None
    if ctx.cost_ledger is not None:
        try:
            cost_usd = ctx.cost_ledger.total_usd()
        except Exception:  # noqa: BLE001
            pass

    # completed when scoring produced a real score; partial when scoring
    # returned 0.0 default due to failure.
    status = "completed" if scores.get("graded", 0) > 0 else "partial"

    return RdrResult(
        project_id=ctx.project_id,
        status=status,
        rubric_score=overall_score,
        clusters_total=len(clusters),
        clusters_failed=clusters_failed_count,
        repair_iterations=repair_iterations,
        final_report_path=str(json_path),
        cost_usd=cost_usd,
    )


__all__ = ["run_rdr"]
