"""Phase-4 RDR Controller — deterministic control flow for rubric-cluster reproduction.

``run_rdr`` owns all control flow: decompose → cluster loop → assemble → env →
experiment → score → repair loop → report.  No LLM in the control path; the
controller is pure Python except for the async agent calls it dispatches.

See ``docs/superpowers/specs/2026-05-22-rubric-driven-harness-design.md`` §7.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pickle
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from backend.agents.rdr.decomposer import decompose
from backend.agents.rdr.context_engineer import build_context
from backend.agents.rdr.models import Artifacts, RdrResult, WorkCluster
from backend.agents.resilience.cost import CostLedgerEntry
from backend.agents.rlm.primitives import (
    detect_environment,
    build_environment,
    run_experiment,
)
from backend.evals.paperbench.leaf_scorer import score_reproduction
from backend.config import get_settings
from backend.agents.rlm.report import (
    RLMFinalReport,
    reconcile_verdict_with_score,
    write_final_report_rlm,
)
from backend.agents.rlm.sse_bridge import (
    build_cluster_artifact_emitted,
    build_cluster_scored,
    build_cluster_started,
    build_repair_dispatched,
)

if TYPE_CHECKING:
    from backend.agents.rlm.context import RunContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-cluster watchdog
# ---------------------------------------------------------------------------

_RDR_WATCHDOG_DEFAULT_S = float(os.environ.get("RDR_CLUSTER_WATCHDOG_S", "900"))

# Default concurrency for the Code Dev cluster batch. Code Execution and
# Result Analysis remain sequential because they depend on Code Dev artifacts.
# Set via --cluster-concurrency or RDR_CLUSTER_CONCURRENCY env var.
_RDR_CLUSTER_CONCURRENCY_DEFAULT = int(
    os.environ.get("RDR_CLUSTER_CONCURRENCY", "8")
)


class _ClusterWatchdog:
    """Per-cluster wall-clock guard.

    Mitigates a known Claude-SDK aclose() deadlock: when the SDK's async
    generator can't be closed cleanly after its subprocess exits, the
    controller wedges in futex_wait_queue indefinitely. This watchdog runs
    in a daemon thread (independent of the asyncio loop) so it fires even
    when the loop deadlocks. When fired, it calls ``os._exit(124)`` — abrupt
    but reliable. A best-effort ``final_report.json`` is written to
    ``project_dir`` before exit so the operator has something to inspect.
    """

    def __init__(
        self,
        timeout_s: float = _RDR_WATCHDOG_DEFAULT_S,
        *,
        label: str = "",
        project_dir: Path | None = None,
    ):
        self.timeout_s = float(timeout_s)
        self.label = label
        self.project_dir = project_dir
        self._timer: threading.Timer | None = None

    def _fire(self) -> None:
        logger.critical(
            "rdr/watchdog: no progress for %.0fs (%s) — terminating python (exit 124). "
            "This usually means the Claude SDK deadlocked on aclose() after a subprocess exit.",
            self.timeout_s, self.label,
        )
        # Best-effort emergency report so the operator finds *something*.
        if self.project_dir is not None:
            try:
                report = {
                    "status": "watchdog_killed",
                    "label": self.label,
                    "timeout_s": self.timeout_s,
                    "verdict": "failed",
                    "rubric": {"overall_score": None, "rubric_source": "watchdog"},
                    "reproduction_summary": (
                        f"rdr watchdog fired at {self.label}; the run was terminated."
                    ),
                }
                (self.project_dir / "final_report.json").write_text(
                    json.dumps(report, indent=2), encoding="utf-8"
                )
            except Exception:  # noqa: BLE001 — never raise from the watchdog thread
                pass
        os._exit(124)

    def arm(self) -> None:
        self.disarm()
        t = threading.Timer(self.timeout_s, self._fire)
        t.daemon = True
        t.start()
        self._timer = t
        logger.debug("rdr/watchdog: armed for %.0fs (%s)", self.timeout_s, self.label)

    def disarm(self) -> None:
        t = self._timer
        if t is not None:
            t.cancel()
            self._timer = None


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


_LANG_BY_SUFFIX: dict[str, str] = {
    ".py": "python",
    ".ipynb": "jupyter",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".json": "json",
    ".md": "markdown",
    ".sh": "shell",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
}


def _artifact_language(path: str) -> str | None:
    """Return a lightweight language hint for UI artifact chips."""
    return _LANG_BY_SUFFIX.get(Path(path).suffix.lower())


def _leaf_scores_for_cluster(cluster: WorkCluster, scores: dict[str, Any]) -> dict[str, float]:
    """Return leaf-score mapping scoped to one cluster."""
    leaf_scores_list: list[dict[str, Any]] = scores.get("leaf_scores", [])
    by_id: dict[str, float] = {
        entry["id"]: float(entry.get("score", 0.0))
        for entry in leaf_scores_list
        if isinstance(entry, dict) and "id" in entry
    }
    return {leaf.id: by_id.get(leaf.id, 0.0) for leaf in cluster.leaves}


def _failed_leaves_for_cluster(
    cluster: WorkCluster,
    scores: dict[str, Any],
    threshold: float,
) -> list[str]:
    """Return cluster leaf ids below the repair threshold."""
    leaf_scores = _leaf_scores_for_cluster(cluster, scores)
    return [leaf_id for leaf_id, score in leaf_scores.items() if score < threshold]


def _is_degraded_experiment(exp: dict[str, Any]) -> bool:
    """Metricless experiments must be scored through the degraded honesty cap."""
    return not bool(exp.get("metrics") or {})


def _zero_scores(*, degraded: bool) -> dict[str, Any]:
    """Safe scorer fallback; degraded runs stay capped and visible in reports."""
    return {
        "overall_score": 0.0,
        "leaf_count": 0,
        "graded": 0,
        "rubric_source": "paperbench_bundle",
        "leaf_scores": [],
        "degraded": degraded,
        "target_score": None,
    }


def _record_primitive_cost(ctx: "RunContext", primitive: str) -> None:
    """Record direct RDR calls into RLM primitives in the run cost ledger."""
    ledger = getattr(ctx, "cost_ledger", None)
    if ledger is None:
        return
    try:
        ledger.append(
            CostLedgerEntry(
                timestamp=datetime.now(timezone.utc),
                agent_id=primitive,
                attempt_index=0,
                provider=getattr(ctx, "provider", "anthropic"),
                model=getattr(ctx, "model", ""),
            )
        )
    except Exception:  # noqa: BLE001 — ledger writes must never break a run
        logger.warning("rdr/controller: failed to append cost ledger row for %s", primitive)


async def _call_primitive(
    ctx: "RunContext",
    primitive: str,
    fn: Callable[..., Any],
    *args: Any,
) -> Any:
    """Call an RLM primitive from RDR and always ledger the attempt."""
    try:
        return await asyncio.to_thread(fn, *args, ctx=ctx)
    finally:
        _record_primitive_cost(ctx, primitive)


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
        "error": art.error,    # surface the agent error string for diagnostics
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
        "error": art.error,
        "file_count": len(art.files),
        "repair_pass": rep_n,
    }
    path = iterations_dir / f"repair_{rep_n}_cluster_{cluster.id}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _merge_cluster_files(
    art: Artifacts,
    cluster: WorkCluster,
    code_dir: Path,
    emit: Callable[[str, dict[str, Any]], None],
    file_merge_lock: threading.Lock,
) -> None:
    """Merge an Artifacts payload into code_dir with concurrency-safe writes.

    Holds *file_merge_lock* across the per-cluster write set so a single
    cluster's files always land atomically with respect to other clusters
    writing the same paths. Defensive against path-escape and PermissionError
    (cache files, .lock files, etc.) — bad paths are skipped, not fatal.
    """
    for rel_path, content in art.files.items():
        dest = code_dir / rel_path
        try:
            if not dest.resolve().is_relative_to(code_dir.resolve()):
                logger.warning(
                    "rdr/controller: refusing to write %r — outside code_dir (cluster %s)",
                    rel_path, cluster.id,
                )
                continue
        except (OSError, ValueError):
            logger.warning("rdr/controller: refusing to resolve %r — skipping", rel_path)
            continue
        try:
            with file_merge_lock:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content, encoding="utf-8")
            artifact_path = str(rel_path).replace("\\", "/")
            emit("cluster_artifact_emitted", build_cluster_artifact_emitted(
                cluster_id=cluster.id,
                artifact_path=artifact_path,
                byte_size=len(content.encode("utf-8")),
                language=_artifact_language(artifact_path),
            ))
        except (PermissionError, OSError) as exc:
            logger.warning(
                "rdr/controller: skipping %r (%s: %s)",
                str(dest), type(exc).__name__, exc,
            )


async def _dispatch_competing_candidates(
    cluster: WorkCluster,
    agctx: Any,
    *,
    n: int,
    reproduce: Callable,
    ctx: "RunContext",
    code_dir: Path,
    cluster_timeout_s: float,
    bes_score_fn: Callable,
    select_metric: str,
    emit: Callable[[str, dict[str, Any]], None],
) -> Artifacts:
    """BES v1 — build N isolated candidates for one cluster; return the winner's Artifacts.

    Each candidate writes into its OWN scratch dir (``candidates/<cluster.id>#i/code``)
    via ``AgentContext.candidate_code_dir`` so they never stomp the shared ``code/``.
    Each is scored STATICALLY by the injected ``bes_score_fn`` (the leaf scorer over the
    scratch dir — no GPU); the best by ``select_metric`` wins and its files merge into
    ``code/`` downstream exactly as a single attempt would. N× token cost, 1× GPU cost.
    Fail-soft per candidate; an empty/all-failed pool returns a failed Artifacts.
    """
    from backend.agents.rdr.candidates import Candidate, select_best
    import dataclasses as _dc
    import re as _re
    import shutil as _shutil

    candidates_root = code_dir.parent / "candidates"
    # Sanitize the cluster id used in the scratch path so a traversal-bearing id
    # ('..', '/') can never escape candidates_root (Codex should-fix).
    _safe_cluster = _re.sub(r"[^A-Za-z0-9_-]", "_", str(cluster.id))[:64] or "cluster"
    pool: list[Candidate] = []
    for i in range(n):
        cid = f"{_safe_cluster}#{i}"
        cand_run_dir = candidates_root / cid
        # Clear stale state from a prior attempt/resume so the snapshot + winner-merge
        # only ever carry THIS candidate's files (Codex should-fix).
        if cand_run_dir.exists():
            _shutil.rmtree(cand_run_dir, ignore_errors=True)
        cand_code_dir = cand_run_dir / "code"
        cand_code_dir.mkdir(parents=True, exist_ok=True)
        cand_ctx = _dc.replace(agctx, candidate_code_dir=cand_code_dir)
        try:
            art_i = await asyncio.wait_for(
                reproduce(cand_ctx, ctx=ctx),
                timeout=cluster_timeout_s,
            )
        except asyncio.TimeoutError:
            art_i = Artifacts(
                cluster_id=cluster.id, failed=True,
                error=f"TimeoutError: candidate {cid} exceeded {cluster_timeout_s:.0f}s",
            )
        except Exception as exc:  # noqa: BLE001 — per-candidate fail-soft
            art_i = Artifacts(
                cluster_id=cluster.id, failed=True,
                error=f"{type(exc).__name__}: {exc}",
            )

        score: float | None = None
        failed_leaves: list[str] = []
        if not art_i.failed:
            try:
                scores = await asyncio.to_thread(bes_score_fn, cand_run_dir, cluster)
                score = _cluster_score(cluster, scores or {})
                failed_leaves = _failed_leaves_for_cluster(cluster, scores or {}, 0.6)
            except Exception as exc:  # noqa: BLE001 — scoring fail-soft → unscored candidate
                logger.warning(
                    "run_rdr[%s]: candidate %s scoring failed: %s",
                    ctx.project_id, cid, exc,
                )
        pool.append(Candidate(
            candidate_id=cid, cluster_id=cluster.id, scratch_dir=cand_run_dir,
            artifacts=art_i, score=score, failed_leaves=failed_leaves,
        ))
        emit("candidate_proposed", {
            "candidate_id": cid, "cluster_id": cluster.id,
            "score": score, "failed": art_i.failed,
        })

    winner = select_best(pool, select_metric=select_metric)
    if winner is None:
        return Artifacts(cluster_id=cluster.id, failed=True, error="BES: no candidates produced")
    emit("candidate_outcome", {
        "candidate_id": winner.candidate_id, "cluster_id": cluster.id,
        "outcome": "selected", "score": winner.score, "n_candidates": len(pool),
    })
    logger.info(
        "run_rdr[%s]: cluster %s — BES selected %s (score=%s) from %d candidates",
        ctx.project_id, cluster.id, winner.candidate_id, winner.score, len(pool),
    )
    return winner.artifacts


async def _dispatch_one_cluster(
    cluster: WorkCluster,
    idx: int,
    *,
    reproduce: Callable,
    ctx: "RunContext",
    paper: str,
    done: dict[str, Artifacts],
    done_lock: threading.Lock,
    file_merge_lock: threading.Lock,
    code_dir: Path,
    iterations_dir: Path,
    emit: Callable[[str, dict[str, Any]], None],
    semaphore: asyncio.Semaphore,
    cluster_timeout_s: float,
    is_repair: bool = False,
    repair_pass: int = 0,
    prior_scores: dict[str, Any] | None = None,
    bes_score_fn: Callable | None = None,
    bes_settings: Any | None = None,
) -> None:
    """Run one cluster end-to-end. Safe to call concurrently for distinct clusters.

    Emits SSE lifecycle events, builds the agent context (reading *done* under
    *done_lock*), invokes *reproduce* with an asyncio-level timeout, merges
    artifacts into *code_dir* under *file_merge_lock*, writes a per-cluster
    checkpoint, and stores the cluster's Artifacts in *done* under *done_lock*.

    Per-cluster failures are caught and recorded as Artifacts(failed=True);
    they NEVER propagate to the gather. This preserves fail-soft semantics
    under parallel dispatch.
    """
    async with semaphore:
        # SSE: cluster lifecycle started
        if is_repair:
            emit("repair_dispatched", build_repair_dispatched(
                cluster_id=cluster.id,
                attempt=repair_pass,
                prior_score=_cluster_score(cluster, prior_scores or {}),
                failed_leaves=_failed_leaves_for_cluster(
                    cluster, prior_scores or {}, threshold=0.6,
                ),
            ))
        else:
            emit("rdr_cluster_started", {
                "cluster_id": cluster.id,
                "cluster_index": idx,
                "cluster_title": cluster.title,
                "leaf_count": len(cluster.leaves),
                "weight": cluster.weight,
                "dominant_category": cluster.dominant_category,
            })
            emit("cluster_started", build_cluster_started(
                cluster_id=cluster.id,
                cluster_title=cluster.title,
                leaves=[
                    {
                        "id": leaf.id,
                        "weight": leaf.weight,
                        "requirements": leaf.requirements,
                    }
                    for leaf in cluster.leaves
                ],
                iteration=idx + 1,
            ))

        # Snapshot done under lock before passing into build_context: parallel
        # peers in the same batch are intentionally invisible to each other
        # (they're independent code-dev clusters by category).
        with done_lock:
            done_snapshot = dict(done)
        agctx = build_context(
            cluster,
            paper=paper,
            artifacts=done_snapshot,
            prior_scores=prior_scores,
        )

        # Per-cluster timeout via asyncio.wait_for. Replaces the legacy
        # os._exit-based _ClusterWatchdog, which would kill the entire process
        # — unsafe when multiple clusters are in flight. Each cluster gets its
        # own asyncio task; a timeout cancels only that task.
        # BES competing candidates (MASTER-gated, default OFF). When bes_enabled
        # with N>1 + an injected score_fn (and not a repair pass), build N isolated
        # candidates, score each statically, and keep the winner's artifacts.
        # Disabled / N==1 / no score_fn / repair => the single reproduce call in the
        # else branch below, byte-identical to today's path.
        _bes_n = 1
        if bes_settings is not None and getattr(bes_settings, "bes_enabled", False):
            _bes_n = max(1, int(getattr(bes_settings, "bes_candidates_per_cluster", 1) or 1))
        if _bes_n > 1 and bes_score_fn is not None and not is_repair:
            art = await _dispatch_competing_candidates(
                cluster, agctx, n=_bes_n,
                reproduce=reproduce, ctx=ctx, code_dir=code_dir,
                cluster_timeout_s=cluster_timeout_s,
                bes_score_fn=bes_score_fn,
                select_metric=str(getattr(bes_settings, "bes_select_metric", "cluster_score")),
                emit=emit,
            )
        else:
            try:
                art = await asyncio.wait_for(
                    reproduce(agctx, ctx=ctx),
                    timeout=cluster_timeout_s,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "run_rdr[%s]: cluster %s timed out after %.0fs — marking failed",
                    ctx.project_id, cluster.id, cluster_timeout_s,
                )
                art = Artifacts(
                    cluster_id=cluster.id,
                    failed=True,
                    error=f"TimeoutError: cluster exceeded {cluster_timeout_s:.0f}s",
                )
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

        # Publish to done before merge so any peers built right after see it.
        with done_lock:
            done[cluster.id] = art

        # SSE: completion event
        if is_repair:
            emit("rdr_repair_cluster_completed", {
                "pass": repair_pass,
                "cluster_id": cluster.id,
                "failed": art.failed,
                "error": art.error,
            })
        else:
            emit("rdr_cluster_completed", {
                "cluster_id": cluster.id,
                "cluster_index": idx,
                "failed": art.failed,
                "error": art.error,
                "file_count": len(art.files),
            })

        # Merge files (concurrency-safe).
        _merge_cluster_files(art, cluster, code_dir, emit, file_merge_lock)

        # Per-cluster checkpoint.
        if is_repair:
            _write_repair_checkpoint(iterations_dir, repair_pass, cluster, art)
        else:
            _write_cluster_checkpoint(iterations_dir, idx, cluster, art)

        if art.failed:
            logger.warning(
                "run_rdr[%s]: %scluster %s failed: %s",
                ctx.project_id,
                f"repair pass {repair_pass} " if is_repair else "",
                cluster.id, art.error,
            )


async def _run_cluster_batch(
    clusters_with_idx: list[tuple[int, WorkCluster]],
    *,
    reproduce: Callable,
    ctx: "RunContext",
    paper: str,
    done: dict[str, Artifacts],
    done_lock: threading.Lock,
    file_merge_lock: threading.Lock,
    code_dir: Path,
    iterations_dir: Path,
    emit: Callable[[str, dict[str, Any]], None],
    cluster_concurrency: int,
    cluster_timeout_s: float,
    is_repair: bool = False,
    repair_pass: int = 0,
    prior_scores: dict[str, Any] | None = None,
    bes_score_fn: Callable | None = None,
    bes_settings: Any | None = None,
) -> None:
    """Run a list of clusters with bounded concurrency.

    Concurrency level applies uniformly across the input list — callers that
    need a mix (e.g., Code Dev parallel, Code Execution sequential) should
    split their cluster list and call this helper twice.

    Returns after every cluster has either completed or been marked failed
    via _dispatch_one_cluster's fail-soft path. Never raises per-cluster.
    """
    if not clusters_with_idx:
        return
    semaphore = asyncio.Semaphore(max(1, int(cluster_concurrency)))
    tasks = [
        _dispatch_one_cluster(
            cluster, idx,
            reproduce=reproduce,
            ctx=ctx,
            paper=paper,
            done=done,
            done_lock=done_lock,
            file_merge_lock=file_merge_lock,
            code_dir=code_dir,
            iterations_dir=iterations_dir,
            emit=emit,
            semaphore=semaphore,
            cluster_timeout_s=cluster_timeout_s,
            is_repair=is_repair,
            repair_pass=repair_pass,
            prior_scores=prior_scores,
            bes_score_fn=bes_score_fn,
            bes_settings=bes_settings,
        )
        for idx, cluster in clusters_with_idx
    ]
    # return_exceptions=True is defensive — _dispatch_one_cluster already
    # catches everything, but we don't want a stray cancellation to bring
    # down the entire batch.
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException):
            logger.warning(
                "run_rdr[%s]: _dispatch_one_cluster surfaced %s: %s",
                ctx.project_id, type(result).__name__, result,
            )


async def _run_rdr_preflight_gate(
    *,
    code_dir: Path,
    parallel_batch: list[tuple[int, WorkCluster]],
    reproduce: Callable,
    ctx: "RunContext",
    paper: str,
    done: dict[str, Artifacts],
    done_lock: threading.Lock,
    file_merge_lock: threading.Lock,
    iterations_dir: Path,
    emit: Callable[[str, dict[str, Any]], None],
    cluster_concurrency: int,
    cluster_timeout_s: float,
    max_regens: int,
) -> None:
    """Phase 2 mode-agnostic pre-run gate: scan ``code/`` for HARD static violations
    BEFORE the GPU experiment; on a violation, regenerate the Code Dev clusters
    (Codex: a violation can't be attributed to one cluster, so re-run all) up to
    ``max_regens`` times, then proceed regardless. Fail-soft — never blocks the run,
    never raises."""
    try:
        from backend.agents.rlm.preflight_ast import scan_code_dir
    except Exception:  # noqa: BLE001 — gate import fail-soft
        return
    for attempt in range(max(0, int(max_regens)) + 1):
        try:
            violations = scan_code_dir(code_dir)
        except Exception:  # noqa: BLE001 — scan fail-soft (never block the experiment)
            return
        hard = [v for v in violations if getattr(v, "severity", "soft") == "hard"]
        if not hard:
            return
        emit("rdr_preflight_blocked", {
            "attempt": attempt,
            "hard_count": len(hard),
            "violations": [str(getattr(v, "detail", v))[:200] for v in hard][:8],
        })
        if attempt >= max_regens:
            logger.warning(
                "run_rdr[%s]: pre-run gate found %d hard violation(s) after %d regen(s) "
                "— proceeding to experiment (fail-soft)", ctx.project_id, len(hard), attempt,
            )
            return
        logger.info(
            "run_rdr[%s]: pre-run gate found %d hard violation(s); regenerating Code Dev "
            "clusters (regen %d/%d)", ctx.project_id, len(hard), attempt + 1, max_regens,
        )
        await _run_cluster_batch(
            parallel_batch, reproduce=reproduce, ctx=ctx, paper=paper, done=done,
            done_lock=done_lock, file_merge_lock=file_merge_lock, code_dir=code_dir,
            iterations_dir=iterations_dir, emit=emit, cluster_concurrency=cluster_concurrency,
            cluster_timeout_s=cluster_timeout_s, is_repair=True, repair_pass=90 + attempt,
        )
        try:
            _cmds = _dedup_commands(done)
            (code_dir / "commands.json").write_text(
                json.dumps(_cmds or ["python train.py"]), encoding="utf-8")
        except Exception:  # noqa: BLE001 — command re-assembly is best-effort
            pass


def _split_clusters_by_parallelism(
    clusters: list[WorkCluster],
) -> tuple[list[WorkCluster], list[WorkCluster]]:
    """Partition clusters into (parallelizable, sequential).

    Parallelizable = Code Development (no inter-cluster deps within category).
    Sequential = Code Execution + Result Analysis (depend on Code Dev outputs
    and on each other). The split is by ``dominant_category`` — the cluster's
    own depends_on list is implicit in this category-based topology.
    """
    parallel: list[WorkCluster] = []
    sequential: list[WorkCluster] = []
    for c in clusters:
        if c.dominant_category == "Code Development":
            parallel.append(c)
        else:
            sequential.append(c)
    return parallel, sequential


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


def _load_resume_done(iterations_dir: Path, clusters: list[Any]) -> dict[str, Any]:
    """Hydrate ``done`` from existing cluster checkpoints for resume runs.

    Returns a mapping from cluster_id → placeholder :class:`Artifacts` for
    every checkpoint that exists under *iterations_dir*. The placeholder
    carries the checkpointed ``failed`` flag; files/commands are empty because
    file content is already on disk in ``code/``.
    """
    done: dict[str, Any] = {}
    if not iterations_dir.is_dir():
        return done

    # Build a lookup so we can match checkpoint cluster_id → WorkCluster
    cluster_by_id: dict[str, Any] = {c.id: c for c in clusters}

    for checkpoint in iterations_dir.glob("cluster_*.json"):
        try:
            data = json.loads(checkpoint.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — corrupt checkpoint: skip
            logger.warning("rdr/resume: could not parse checkpoint %s — skipping", checkpoint)
            continue
        cid = data.get("cluster_id", "")
        if not cid or cid not in cluster_by_id:
            continue
        if cid in done:
            continue  # already hydrated (take the first / only one)
        done[cid] = Artifacts(
            cluster_id=cid,
            failed=bool(data.get("failed", False)),
            files={},
            commands=[],
            notes="resumed",
            error="",
        )
        logger.info("rdr/resume: skipping cluster %s (checkpoint exists)", cid)
    return done


async def run_rdr(
    bundle: Any,
    *,
    ctx: "RunContext",
    max_repair_iterations: int = 2,
    repair_target: float = 0.6,
    max_leaves_per_cluster: int = 12,
    reproduce_fn: Callable | None = None,
    resume: bool = False,
    cluster_concurrency: int | None = None,
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
        resume: When True, load completed cluster checkpoints from
            ``ctx.project_dir/iterations/`` and skip those clusters.
        cluster_concurrency: Maximum number of Code Development clusters to
            dispatch concurrently. Code Execution and Result Analysis
            clusters remain sequential because they depend on Code Dev
            artifacts. ``None`` → resolves to RDR_CLUSTER_CONCURRENCY env
            var (default 8). Pass 1 to force fully sequential execution.

    Returns:
        An :class:`RdrResult` — always; per-cluster and per-phase failures
        are fail-soft and produce an honest partial or completed result.
    """
    run_started_at = datetime.now(timezone.utc).isoformat()
    _reproduce = _resolve_reproduce_fn(reproduce_fn)

    # Concurrency level — explicit arg > env > default.
    if cluster_concurrency is None:
        cluster_concurrency = _RDR_CLUSTER_CONCURRENCY_DEFAULT
    cluster_concurrency = max(1, int(cluster_concurrency))
    cluster_timeout_s = _RDR_WATCHDOG_DEFAULT_S

    # Locks shared across the parallel cluster batch.
    done_lock = threading.Lock()
    file_merge_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Step 1: Decompose
    # ------------------------------------------------------------------
    rubric = bundle.rubric()
    # BES on RDR (spec 2026-06-07): read flags once. Build the STATIC candidate
    # score_fn only when the master gate is on AND N>1; otherwise _bes_score_fn stays
    # None so the dispatcher takes the byte-identical legacy path. score_reproduction
    # is the static leaf scorer (no GPU) over a candidate's scratch dir.
    _bes_settings = get_settings()
    _bes_score_fn: Callable | None = None
    if _bes_settings.bes_enabled and _bes_settings.bes_candidates_per_cluster > 1:
        def _bes_score_fn(cand_run_dir: Path, _cluster: WorkCluster, _rubric=rubric, _ctx=ctx) -> dict:
            # degraded=False so the leaf scorer GRADES the candidate's code (Codex
            # blocker): degraded short-circuits every leaf to 0.0 without reading the
            # code, which ties all candidates and always picks #0. The candidate has
            # no metrics.json yet, so this is a code-only static grade — the SELECT
            # signal that discriminates competing candidates pre-GPU.
            return score_reproduction(
                _rubric, cand_run_dir, _ctx.llm_client,
                rubric_source=str((_rubric or {}).get("source") or "paperbench_bundle"),
                degraded=False,
            )
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
    # SSE helper — defensive: skip silently when dashboard is absent
    # ------------------------------------------------------------------
    def _emit(event_type: str, payload: dict[str, Any]) -> None:
        """Emit a dashboard_event via ctx.dashboard.emit (if available)."""
        try:
            emitter = ctx.dashboard
            if emitter is None:
                return
            emit_fn = getattr(emitter, "emit", None)
            if callable(emit_fn):
                emit_fn(event_type, payload)
        except Exception:  # noqa: BLE001 — never raise from emitter
            pass

    # ------------------------------------------------------------------
    # SSE: rdr_run_started
    # ------------------------------------------------------------------
    _emit("rdr_run_started", {
        "project_id": ctx.project_id,
        "paper_id": getattr(bundle, "paper_id", ""),
        "cluster_count": len(clusters),
    })

    # ------------------------------------------------------------------
    # Step 3: Cluster loop
    # ------------------------------------------------------------------
    # On resume, hydrate done from existing checkpoints so completed
    # clusters are skipped.
    done: dict[str, Artifacts] = {}
    if resume:
        done = _load_resume_done(iterations_dir, clusters)

    # Partition clusters into the parallelizable Code Dev batch and the
    # sequential tail (Code Execution + Result Analysis). Within each
    # partition we preserve original index for checkpoint filenames.
    indexed_clusters = [
        (idx, cluster) for idx, cluster in enumerate(clusters)
        if cluster.id not in done
    ]
    parallel_batch = [
        (idx, c) for idx, c in indexed_clusters
        if c.dominant_category == "Code Development"
    ]
    sequential_tail = [
        (idx, c) for idx, c in indexed_clusters
        if c.dominant_category != "Code Development"
    ]

    logger.info(
        "run_rdr[%s]: dispatching %d parallel Code Dev clusters "
        "(concurrency=%d, timeout=%.0fs/cluster), then %d sequential "
        "Code Exec/Result Analysis clusters",
        ctx.project_id, len(parallel_batch), cluster_concurrency,
        cluster_timeout_s, len(sequential_tail),
    )

    # Phase 1a: Code Dev clusters in parallel.
    await _run_cluster_batch(
        parallel_batch,
        reproduce=_reproduce,
        ctx=ctx,
        paper=paper,
        done=done,
        done_lock=done_lock,
        file_merge_lock=file_merge_lock,
        code_dir=code_dir,
        iterations_dir=iterations_dir,
        emit=_emit,
        cluster_concurrency=cluster_concurrency,
        cluster_timeout_s=cluster_timeout_s,
        bes_score_fn=_bes_score_fn,
        bes_settings=_bes_settings,
    )

    # Phase 1b: Code Execution + Result Analysis sequentially.
    await _run_cluster_batch(
        sequential_tail,
        reproduce=_reproduce,
        ctx=ctx,
        paper=paper,
        done=done,
        done_lock=done_lock,
        file_merge_lock=file_merge_lock,
        code_dir=code_dir,
        iterations_dir=iterations_dir,
        emit=_emit,
        cluster_concurrency=1,
        cluster_timeout_s=cluster_timeout_s,
    )

    clusters_failed = sum(1 for art in done.values() if art.failed)

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
    _emit("rdr_environment_started", {"project_id": ctx.project_id})
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
            env_spec = await _call_primitive(
                ctx,
                "detect_environment",
                detect_environment,
                method_spec,
            )
            if env_spec.get("success") is False:
                logger.warning(
                    "run_rdr[%s]: detect_environment failed: %s",
                    ctx.project_id, env_spec.get("error"),
                )
                env_spec = {}

        if env_spec:
            build = await _call_primitive(
                ctx,
                "build_environment",
                build_environment,
                env_spec,
            )
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
    _emit("rdr_environment_completed", {"project_id": ctx.project_id, "env_id": env_id})

    # Mode-agnostic RDR pre-run gate (Phase 2, default OFF). Scan the assembled
    # code/ for HARD static violations BEFORE spending GPU on run_experiment; on a
    # violation, regenerate the Code Dev clusters (bounded) and re-scan. Fail-soft.
    if getattr(_bes_settings, "rdr_preflight_gate", False):
        await _run_rdr_preflight_gate(
            code_dir=code_dir, parallel_batch=parallel_batch, reproduce=_reproduce,
            ctx=ctx, paper=paper, done=done, done_lock=done_lock,
            file_merge_lock=file_merge_lock, iterations_dir=iterations_dir, emit=_emit,
            cluster_concurrency=cluster_concurrency, cluster_timeout_s=cluster_timeout_s,
            max_regens=int(getattr(_bes_settings, "rdr_preflight_max_regens", 1)),
        )

    # ------------------------------------------------------------------
    # Step 6: Experiment (fail-soft; only if env_id available)
    # ------------------------------------------------------------------
    exp: dict[str, Any] = {"success": False, "metrics": {}}
    _emit("rdr_experiment_started", {"project_id": ctx.project_id})
    if env_id:
        try:
            exp = await _call_primitive(
                ctx,
                "run_experiment",
                run_experiment,
                str(code_dir),
                env_id,
            )
        except Exception as exc:  # noqa: BLE001 — fail-soft
            logger.warning(
                "run_rdr[%s]: run_experiment raised %s: %s",
                ctx.project_id, type(exc).__name__, exc,
            )
    _emit("rdr_experiment_completed", {
        "success": exp.get("success", False),
        "metrics_keys": list(exp.get("metrics", {}).keys()),
    })

    # ------------------------------------------------------------------
    # Step 7: Initial scoring
    # ------------------------------------------------------------------
    # score_reproduction is sync but its ClaudeLlmClient.complete() internally
    # calls asyncio.run() — which deadlocks if we invoke it from the running
    # event-loop thread.  Run in a worker thread so the scorer's own loop is
    # independent.  Surfaced live: "asyncio.run() cannot be called from a
    # running event loop" in seqnn's score batches.
    #
    # Fail-soft: flatten_leaves(), _gather_evidence(), or any OOM/OSError on the
    # rubric tree may raise through asyncio.to_thread — catch and substitute safe
    # zero-scores so the run always produces a final_report.
    degraded_run = _is_degraded_experiment(exp)
    _emit("rdr_scoring_started", {"project_id": ctx.project_id})
    try:
        scores = await asyncio.to_thread(
            score_reproduction,
            rubric,
            ctx.project_dir,
            ctx.llm_client,
            degraded=degraded_run,
        )
    except Exception as exc:  # noqa: BLE001 — fail-soft
        logger.warning(
            "rdr/controller: score_reproduction raised %s: %s — defaulting to zero scores",
            type(exc).__name__, exc,
        )
        scores = _zero_scores(degraded=degraded_run)
    if degraded_run:
        scores["degraded"] = True
    _emit("rdr_scoring_completed", {
        "overall_score": scores.get("overall_score"),
        "leaf_count": scores.get("leaf_count"),
        "graded": scores.get("graded"),
    })
    for cluster in clusters:
        _emit("cluster_scored", build_cluster_scored(
            cluster_id=cluster.id,
            score=_cluster_score(cluster, scores),
            leaf_scores=_leaf_scores_for_cluster(cluster, scores),
            degraded=bool(scores.get("degraded", False)),
        ))

    # ------------------------------------------------------------------
    # Step 8: Repair loop
    #
    # Track the actual number of agent dispatches across initial + repair
    # passes so the report's ``iterations`` field is accurate.
    # ------------------------------------------------------------------
    repair_iterations = 0
    # Initial pass already dispatched one call per cluster.
    total_agent_dispatches = len(clusters)

    # Build a set of already-completed repair checkpoints for resume.
    _completed_repair_keys: set[str] = set()
    if resume and iterations_dir.is_dir():
        for _rcp in iterations_dir.glob("repair_*.json"):
            try:
                _rcdata = json.loads(_rcp.read_text(encoding="utf-8"))
                _rn = _rcdata.get("repair_pass")
                _rcid = _rcdata.get("cluster_id", "")
                if _rn is not None and _rcid:
                    _completed_repair_keys.add(f"{_rn}:{_rcid}")
            except Exception:  # noqa: BLE001
                pass

    for _rep in range(max_repair_iterations):
        weak = [c for c in clusters if _cluster_score(c, scores) < repair_target]
        if not weak:
            break

        rep_n = repair_iterations + 1
        logger.info(
            "run_rdr[%s]: repair iteration %d — %d weak clusters",
            ctx.project_id, rep_n, len(weak),
        )
        _emit("rdr_repair_pass_started", {"pass": rep_n, "weak_count": len(weak)})

        # Skip weak clusters whose repair checkpoint already exists (resume).
        weak_pending = [
            c for c in weak
            if f"{rep_n}:{c.id}" not in _completed_repair_keys
        ]
        skipped_for_resume = len(weak) - len(weak_pending)
        if skipped_for_resume:
            logger.info(
                "rdr/resume: pass %d skipping %d clusters with existing checkpoints",
                rep_n, skipped_for_resume,
            )

        # Partition by parallelizability (same rule as initial pass).
        # Use the cluster's natural index from the original list so repair
        # checkpoints don't collide with each other across passes.
        index_by_id = {c.id: i for i, c in enumerate(clusters)}
        weak_indexed = [(index_by_id[c.id], c) for c in weak_pending]
        weak_parallel = [
            (i, c) for i, c in weak_indexed
            if c.dominant_category == "Code Development"
        ]
        weak_sequential = [
            (i, c) for i, c in weak_indexed
            if c.dominant_category != "Code Development"
        ]

        # Repair pass 2a: parallel Code Dev clusters.
        await _run_cluster_batch(
            weak_parallel,
            reproduce=_reproduce,
            ctx=ctx,
            paper=paper,
            done=done,
            done_lock=done_lock,
            file_merge_lock=file_merge_lock,
            code_dir=code_dir,
            iterations_dir=iterations_dir,
            emit=_emit,
            cluster_concurrency=cluster_concurrency,
            cluster_timeout_s=cluster_timeout_s,
            is_repair=True,
            repair_pass=rep_n,
            prior_scores=scores,
        )

        # Repair pass 2b: sequential Code Exec + Result Analysis.
        await _run_cluster_batch(
            weak_sequential,
            reproduce=_reproduce,
            ctx=ctx,
            paper=paper,
            done=done,
            done_lock=done_lock,
            file_merge_lock=file_merge_lock,
            code_dir=code_dir,
            iterations_dir=iterations_dir,
            emit=_emit,
            cluster_concurrency=1,
            cluster_timeout_s=cluster_timeout_s,
            is_repair=True,
            repair_pass=rep_n,
            prior_scores=scores,
        )

        # One dispatch counted per cluster we actually attempted this pass.
        total_agent_dispatches += len(weak_pending)

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
                exp = await _call_primitive(
                    ctx,
                    "run_experiment",
                    run_experiment,
                    str(code_dir),
                    env_id,
                )
            except Exception as exc:  # noqa: BLE001 — fail-soft
                logger.warning(
                    "run_rdr[%s]: repair run_experiment raised %s: %s",
                    ctx.project_id, type(exc).__name__, exc,
                )

        # Re-score (off-loop, see Step 7 note).
        degraded_run = _is_degraded_experiment(exp)
        try:
            scores = await asyncio.to_thread(
                score_reproduction,
                rubric,
                ctx.project_dir,
                ctx.llm_client,
                degraded=degraded_run,
            )
        except Exception as exc:  # noqa: BLE001 — fail-soft
            logger.warning(
                "rdr/controller: score_reproduction raised %s: %s — defaulting to zero scores",
                type(exc).__name__, exc,
            )
            scores = _zero_scores(degraded=degraded_run)
        if degraded_run:
            scores["degraded"] = True
        for cluster in clusters:
            _emit("cluster_scored", build_cluster_scored(
                cluster_id=cluster.id,
                score=_cluster_score(cluster, scores),
                leaf_scores=_leaf_scores_for_cluster(cluster, scores),
                degraded=bool(scores.get("degraded", False)),
            ))
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
        # Lane G: append() buffers writes; flush at run end so the on-disk
        # cost_ledger.jsonl reflects every entry recorded during the run.
        # RDR does not go through the RLM binding wrapper that auto-flushes
        # at primitive boundaries, so an explicit flush is required here.
        try:
            ctx.cost_ledger.flush()
        except Exception:  # noqa: BLE001 — flush failure must not abort the report
            logger.exception("cost ledger flush failed at RDR run completion")
        try:
            cost_dict = {"llm_usd": ctx.cost_ledger.total_usd(), "primitives": ctx.cost_ledger.total_usd()}
        except Exception:  # noqa: BLE001
            pass

    # total_agent_dispatches counts the actual number of agent calls:
    # one per cluster on the initial pass + one per weak cluster per repair pass.
    started_at: str | None = run_started_at
    try:
        status_path = ctx.project_dir / "demo_status.json"
        if status_path.exists():
            status_data = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(status_data, dict):
                raw_started_at = status_data.get("startedAt") or status_data.get("started_at")
                if isinstance(raw_started_at, str):
                    started_at = raw_started_at
    except Exception:  # noqa: BLE001 — metadata is best-effort
        started_at = run_started_at

    model_name = getattr(ctx, "model", None)
    agent_model = getattr(ctx, "agent_model", None) or model_name
    degraded = bool(scores.get("degraded", False))
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
        degraded=degraded,
        mode="rdr",
        models={
            "planner": model_name,
            "executor": agent_model,
            "verifier": None,
            "grader": model_name,
        },
        started_at=started_at,
        completed_at=datetime.now(timezone.utc).isoformat(),
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

    # SSE: rdr_run_completed
    _emit("rdr_run_completed", {
        "status": status,
        "rubric_score": overall_score,
        "clusters_total": len(clusters),
        "clusters_failed": clusters_failed_count,
        "repair_iterations": repair_iterations,
    })

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
