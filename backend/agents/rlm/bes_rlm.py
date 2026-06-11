"""BES competing candidates on the RLM path (2026-06-11, default OFF).

BES-on-RDR (``rdr/controller._dispatch_competing_candidates``) never engages
for arXiv/PDF uploads: the hybrid bundle guard dispatches those runs straight
to pure RLM, so the only implementation the run ever gets is one
``implement_baseline`` shot. This module brings BES v1's single delta —
N competing implementations + static rubric SELECT, experiment runs ONCE on
the winner — to that path, reusing the mode-agnostic pieces verbatim:
``rdr/candidates.py`` (Candidate + select_best) and the same master flags
(``REPROLAB_BES_ENABLED`` + ``REPROLAB_BES_CANDIDATES_PER_CLUSTER`` +
``REPROLAB_BES_SELECT_METRIC``), so one switch drives both paths.

Mechanics (mirror of the RDR dispatch, adapted to the file-on-disk contract):
``implement_baseline`` writes into the fixed ``code/``; for each candidate we
run one inner implementation (cache-busted via a ``_bes_candidate_idx`` plan
key; prompt-diversified via a per-candidate angle appended to
``REPROLAB_BASELINE_EXTRA_GUIDANCE``), snapshot ``code/`` into
``candidates/rlm_impl_<i>/code/``, clear, and statically grade the snapshot
with the leaf scorer (``degraded=False`` — the code-only grade is the SELECT
signal; no GPU spend). The winner's snapshot is restored into ``code/`` and
re-harvested so the returned envelope honours implement_baseline's contract.

Repairs (``repair_context``) and re-entrant candidate calls stay single-shot —
exact BES v1 semantics. Everything is fail-soft: any error falls back to one
normal implementation. A winner marker in ``rlm_state/`` makes repeat calls
idempotent (a second identical implement_baseline call returns the winner
instead of re-competing or overwriting it).

A/B observability: the pool is persisted to ``rlm_state/bes_candidates.json``,
``candidate_proposed``/``candidate_outcome`` SSE events are emitted (the UI
already renders both), and ``experiment_arm_stamp`` is merged into
``final_report.json`` by ``report.write_final_report_rlm`` so paired
with/without-BES runs are explicitly labelled for ``scripts/ab_compare.py``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)

ENV_AB_ARM = "REPROLAB_AB_ARM"
ENV_AB_PAIR_ID = "REPROLAB_AB_PAIR_ID"
ENV_MIN_REMAINING_S = "REPROLAB_BES_MIN_REMAINING_S"
ENV_CONTINUE_MIN_S = "REPROLAB_BES_CONTINUE_MIN_S"

# Don't start competing with less than this much wall-clock left (each inner
# implementation is a multi-minute Sonnet sub-agent + a static grade).
_DEFAULT_MIN_REMAINING_S = 9000.0
# Don't start candidate i>0 with less than this much wall-clock left; grade
# whatever the pool holds instead.
_DEFAULT_CONTINUE_MIN_S = 7200.0

# Leaf below this static score counts toward failed_leaves (tie-break only) —
# same threshold the RDR dispatch uses.
_FAILED_LEAF_THRESHOLD = 0.6

# Heavy artifacts never belong in a candidate snapshot (mirror of
# best_attempt.seed_reference_code's ignore list).
_SNAPSHOT_IGNORE = shutil.ignore_patterns(
    "__pycache__", "datasets", "outputs", ".venv", "wandb",
    "*.pt", "*.pth", "*.ckpt", "*.safetensors",
)

# Per-candidate prompt angles. Index 0 is the parity candidate (no extra
# guidance — identical prompt to a non-BES run); later candidates explore a
# distinct emphasis so the pool isn't N samples of the same prompt.
_CANDIDATE_ANGLES: tuple[str, ...] = (
    "",
    (
        "CANDIDATE ANGLE (fidelity-first): where the paper's specification and "
        "implementation convenience disagree, implement the paper's EXACT "
        "specification — architectures, hyperparameters, schedules, "
        "initialization, preprocessing — even when a simpler variant would "
        "train faster. Note the paper section for each key constant in a "
        "comment."
    ),
    (
        "CANDIDATE ANGLE (coverage-first): prioritize emitting every artifact "
        "the rubric names — a complete cells.json covering the full "
        "model×dataset grid, metrics.json with every declared key, and the "
        "figures/tables the paper reports. Prefer broad correct coverage over "
        "deep optimization of any single cell."
    ),
)


def _env_float(name: str, default: float) -> float:
    try:
        raw = os.environ.get(name, "").strip()
        return float(raw) if raw else default
    except ValueError:
        return default


def is_enabled() -> bool:
    """Master gate — same flags as BES-on-RDR; default OFF."""
    try:
        from backend.config import get_settings

        s = get_settings()
        return bool(s.bes_enabled) and int(s.bes_candidates_per_cluster) > 1
    except Exception:  # noqa: BLE001 — settings failure must never block implement
        return False


def _rubric_path(ctx: Any) -> Path:
    return Path(ctx.project_dir) / "generated_rubric.json"


def _adaptive_decision(ctx: Any, settings: Any) -> dict:
    """Engage the pool only where selection has variance to remove.

    The allcnn-ab-20260611 pool discriminated weakly (0.549 vs 0.557) because
    the seeded best-attempt + champion rails already anchor implementation
    quality on papers with measured history. Deterministic rule: engage on a
    project with NO prior attempt or a weak best (< bes_adaptive_skip_score);
    skip when champion-grade history dominates. The decision (either way) is
    persisted for the A/B report and the experiment_arm stamp.
    """
    threshold = float(getattr(settings, "bes_adaptive_skip_score", 0.5) or 0.5)
    best_score: float | None = None
    try:
        from backend.agents.rlm.best_attempt import find_best_attempt

        best = find_best_attempt(Path(ctx.project_dir))
        if best is not None:
            best_score = float(best.get("score"))
    except Exception:  # noqa: BLE001 — unreadable history reads as "no history"
        logger.debug("bes_rlm: adaptive history probe failed", exc_info=True)

    if best_score is None:
        decision = {"engage": True, "reason": "no_prior_history"}
    elif best_score < threshold:
        decision = {"engage": True, "reason": f"weak_history({best_score:.3f}<{threshold:g})"}
    else:
        decision = {"engage": False, "reason": f"strong_history({best_score:.3f}>={threshold:g})"}
    decision["best_score"] = best_score
    decision["threshold"] = threshold

    try:
        state_dir = Path(ctx.project_dir) / "rlm_state"
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "bes_adaptive.json").write_text(
            json.dumps(decision, indent=2), encoding="utf-8",
        )
    except OSError:
        pass
    return decision


def should_compete(ctx: Any, plan: dict) -> bool:
    """True when this implement_baseline call should run the candidate pool.

    Requires: master flag on, an initial (non-repair) implementation, a rubric
    on disk to grade against, and enough wall-clock left for N implementations.
    """
    if not is_enabled():
        return False
    if plan.get("repair_context") is not None:
        return False
    if not _rubric_path(ctx).is_file():
        logger.info(
            "bes_rlm[%s]: no generated_rubric.json — single-shot implementation",
            getattr(ctx, "project_id", "?"),
        )
        return False
    try:
        remaining = ctx.remaining_s()
    except Exception:  # noqa: BLE001
        remaining = None
    min_remaining = _env_float(ENV_MIN_REMAINING_S, _DEFAULT_MIN_REMAINING_S)
    if remaining is not None and remaining < min_remaining:
        logger.info(
            "bes_rlm[%s]: %.0fs remaining < %.0fs floor — single-shot implementation",
            getattr(ctx, "project_id", "?"), remaining, min_remaining,
        )
        return False
    try:
        from backend.config import get_settings

        settings = get_settings()
        if getattr(settings, "bes_adaptive", False):
            decision = _adaptive_decision(ctx, settings)
            logger.info(
                "bes_rlm[%s]: adaptive gate — engage=%s (%s)",
                getattr(ctx, "project_id", "?"), decision["engage"], decision["reason"],
            )
            if not decision["engage"]:
                return False
    except Exception:  # noqa: BLE001 — adaptive gate failure never blocks the pool
        logger.debug("bes_rlm: adaptive gate errored — pool proceeds", exc_info=True)
    return True


@contextmanager
def _angle_guidance(angle: str) -> Iterator[None]:
    """Append a candidate angle to REPROLAB_BASELINE_EXTRA_GUIDANCE, restoring after."""
    key = "REPROLAB_BASELINE_EXTRA_GUIDANCE"
    old = os.environ.get(key)
    try:
        if angle:
            base = (old or "").strip()
            os.environ[key] = f"{base}\n\n{angle}".strip()
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


def _clear_dir(path: Path) -> None:
    for child in path.iterdir():
        try:
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        except OSError:
            logger.warning("bes_rlm: could not clear %s", child)


def _snapshot_code(code_dir: Path, cand_dir: Path) -> Path:
    """Copy code/ into the candidate scratch dir; return the snapshot code path."""
    if cand_dir.exists():
        shutil.rmtree(cand_dir, ignore_errors=True)
    snap = cand_dir / "code"
    shutil.copytree(code_dir, snap, ignore=_SNAPSHOT_IGNORE, dirs_exist_ok=True)
    return snap


def _stage_paper_text(project_dir: Path, cand_dir: Path) -> None:
    """Give the leaf grader the paper text the run dir would normally hold."""
    for name in ("parsed_full_text.txt", "paper_full.md"):
        src = project_dir / name
        if src.is_file():
            try:
                shutil.copy2(src, cand_dir / name)
            except OSError:
                pass


def _static_grade(rubric: dict, cand_dir: Path, ctx: Any) -> tuple[float | None, list[str]]:
    """Code-only rubric grade of one candidate snapshot (the SELECT signal).

    ``degraded=False`` deliberately (the RDR Codex blocker): degraded
    short-circuits every leaf to 0.0 without reading the code, which ties all
    candidates and always picks #0. There is no metrics.json yet, so
    result-match leaves score ~0 for every candidate equally; code-fidelity
    leaves discriminate.
    """
    from backend.evals.paperbench.leaf_scorer import score_reproduction

    scored = score_reproduction(
        rubric_tree=rubric,
        run_dir=cand_dir,
        llm_client=ctx.llm_client,
        rubric_source=str(rubric.get("source") or "generated"),
        degraded=False,
    )
    overall = scored.get("overall_score")
    failed: list[str] = []
    for leaf in scored.get("leaf_scores") or []:
        if not isinstance(leaf, dict):
            continue
        try:
            if float(leaf.get("score") or 0.0) < _FAILED_LEAF_THRESHOLD:
                failed.append(str(leaf.get("id") or leaf.get("leaf_id") or ""))
        except (TypeError, ValueError):
            continue
    try:
        overall_f = float(overall) if overall is not None else None
    except (TypeError, ValueError):
        overall_f = None
    return overall_f, failed


def _emit(ctx: Any, event_type: str, payload: dict) -> None:
    try:
        from backend.agents.rlm.primitives import _emit_dashboard_event

        _emit_dashboard_event(ctx, event_type=event_type, payload=payload)
    except Exception:  # noqa: BLE001 — events are best-effort
        logger.debug("bes_rlm: %s emit failed", event_type, exc_info=True)


def _marker_path(ctx: Any) -> Path:
    return Path(ctx.project_dir) / "rlm_state" / "bes_winner.json"


def compete(plan: dict, *, ctx: Any, implement_fn: Callable[..., dict]) -> dict:
    """Run the candidate pool; never raises — falls back to single-shot."""
    try:
        return _compete_inner(plan, ctx=ctx, implement_fn=implement_fn)
    except Exception:  # noqa: BLE001 — BES must never cost the run its baseline
        logger.warning(
            "bes_rlm[%s]: compete failed — single-shot fallback",
            getattr(ctx, "project_id", "?"), exc_info=True,
        )
        try:
            return implement_fn(dict(plan), ctx=ctx, _bes_inner=True)
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "error_code": "bes_compete_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "repairable": True,
            }


def _compete_inner(plan: dict, *, ctx: Any, implement_fn: Callable[..., dict]) -> dict:
    from backend.agents.rdr.candidates import Candidate, select_best
    from backend.agents.rdr.models import Artifacts
    from backend.agents.rlm.primitives import _harvest_baseline_artifacts
    from backend.config import get_settings

    settings = get_settings()
    n = max(2, int(settings.bes_candidates_per_cluster))
    select_metric = str(settings.bes_select_metric or "cluster_score")

    project_dir = Path(ctx.project_dir)
    code_dir = Path(ctx.runs_root) / ctx.project_id / "code"
    code_dir.mkdir(parents=True, exist_ok=True)

    # Idempotency: a prior compete already selected a winner for this run.
    marker = _marker_path(ctx)
    if marker.is_file():
        try:
            saved = json.loads(marker.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            saved = {}
        harvested = _harvest_baseline_artifacts(code_dir)
        if harvested.get("ok") is True:
            harvested["bes"] = saved.get("bes") or {"selected": saved.get("winner")}
            logger.info(
                "bes_rlm[%s]: winner marker present — returning selected baseline",
                ctx.project_id,
            )
            return harvested
        # Winner code vanished (archived) — recompete from scratch.
        marker.unlink(missing_ok=True)

    rubric = json.loads(_rubric_path(ctx).read_text(encoding="utf-8"))
    candidates_root = project_dir / "candidates"
    candidates_root.mkdir(parents=True, exist_ok=True)
    continue_min = _env_float(ENV_CONTINUE_MIN_S, _DEFAULT_CONTINUE_MIN_S)

    pool: list[Candidate] = []
    envelopes: dict[str, dict] = {}
    records: list[dict] = []
    last_failure: dict | None = None

    logger.info(
        "bes_rlm[%s]: competing %d implementations (select=%s)",
        ctx.project_id, n, select_metric,
    )
    for i in range(n):
        if i > 0:
            try:
                remaining = ctx.remaining_s()
            except Exception:  # noqa: BLE001
                remaining = None
            if remaining is not None and remaining < continue_min:
                logger.info(
                    "bes_rlm[%s]: %.0fs remaining < %.0fs — truncating pool at %d",
                    ctx.project_id, remaining, continue_min, len(pool),
                )
                break

        cid = f"rlm_impl#{i}"
        cand_dir = candidates_root / f"rlm_impl_{i}"
        angle = _CANDIDATE_ANGLES[i % len(_CANDIDATE_ANGLES)]

        plan_i = dict(plan)
        plan_i["_bes_candidate_idx"] = i  # Lane-A cache bust per candidate
        with _angle_guidance(angle):
            result = implement_fn(plan_i, ctx=ctx, _bes_inner=True)

        ok = bool(isinstance(result, dict) and result.get("ok") is True)
        score: float | None = None
        failed_leaves: list[str] = []
        if ok:
            try:
                _snapshot_code(code_dir, cand_dir)
                _stage_paper_text(project_dir, cand_dir)
                envelopes[cid] = result
                score, failed_leaves = _static_grade(rubric, cand_dir, ctx)
            except Exception:  # noqa: BLE001 — grading fail-soft → unscored candidate
                logger.warning(
                    "bes_rlm[%s]: candidate %s snapshot/grade failed",
                    ctx.project_id, cid, exc_info=True,
                )
        else:
            last_failure = result if isinstance(result, dict) else None

        # Clear for the next candidate; the winner is restored from its snapshot.
        _clear_dir(code_dir)

        pool.append(Candidate(
            candidate_id=cid,
            cluster_id="rlm_baseline",
            scratch_dir=cand_dir,
            artifacts=Artifacts(
                cluster_id="rlm_baseline",
                failed=not ok,
                error=None if ok else str((result or {}).get("error") or "implementation failed"),
            ),
            score=score,
            failed_leaves=failed_leaves,
        ))
        records.append({
            "candidate_id": cid,
            "ok": ok,
            "score": score,
            "failed_leaf_count": len(failed_leaves),
            "angle": angle[:80],
            "dir": str(cand_dir),
        })
        _emit(ctx, "candidate_proposed", {
            "candidate_id": cid,
            "cluster_id": "rlm_baseline",
            "score": score,
            "failed": not ok,
        })

    winner = select_best(pool, select_metric=select_metric)
    if winner is None or winner.candidate_id not in envelopes:
        # Every candidate failed — honour implement_baseline's failure contract.
        logger.warning(
            "bes_rlm[%s]: no usable candidate from pool of %d", ctx.project_id, len(pool),
        )
        return last_failure or {
            "ok": False,
            "error_code": "bes_all_candidates_failed",
            "error": f"all {len(pool)} BES candidates failed to implement",
            "repairable": True,
        }

    # Restore the winner into code/ and re-harvest so the envelope contract
    # (commands.json + runnable source) is verified on the REAL path.
    snap_code = Path(winner.scratch_dir) / "code"
    _clear_dir(code_dir)
    shutil.copytree(snap_code, code_dir, dirs_exist_ok=True)
    final = _harvest_baseline_artifacts(code_dir)
    if final.get("ok") is not True:
        # Restore produced an incomplete tree — extremely unlikely; fall back to
        # the winner's original envelope so the root still gets a usable result.
        final = envelopes[winner.candidate_id]

    bes_meta = {
        "selected": winner.candidate_id,
        "n_candidates": len(pool),
        "select_metric": select_metric,
        "scores": {r["candidate_id"]: r["score"] for r in records},
    }
    final["bes"] = bes_meta

    _emit(ctx, "candidate_outcome", {
        "candidate_id": winner.candidate_id,
        "cluster_id": "rlm_baseline",
        "outcome": "selected",
        "score": winner.score,
        "n_candidates": len(pool),
    })
    logger.info(
        "bes_rlm[%s]: selected %s (score=%s) from %d candidates",
        ctx.project_id, winner.candidate_id, winner.score, len(pool),
    )

    state_dir = project_dir / "rlm_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "enabled": True,
        "n_requested": n,
        "select_metric": select_metric,
        "winner": winner.candidate_id,
        "candidates": records,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        (state_dir / "bes_candidates.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8",
        )
        marker.write_text(
            json.dumps({"winner": winner.candidate_id, "bes": bes_meta}, indent=2),
            encoding="utf-8",
        )
    except OSError:
        logger.warning("bes_rlm[%s]: could not persist pool state", ctx.project_id)

    return final


def experiment_arm_stamp(project_dir: Path | str) -> dict:
    """A/B arm label + BES flag snapshot for final_report.json.

    Arm precedence: explicit ``REPROLAB_AB_ARM`` > derived from the master
    flag. Always returns a stamp (control runs are labelled too) so paired
    runs are explicit for ``scripts/ab_compare.py`` and the leaderboard.
    """
    enabled = False
    n = 1
    metric = "cluster_score"
    try:
        from backend.config import get_settings

        s = get_settings()
        enabled = bool(s.bes_enabled)
        n = int(s.bes_candidates_per_cluster)
        metric = str(s.bes_select_metric)
    except Exception:  # noqa: BLE001
        pass

    arm = os.environ.get(ENV_AB_ARM, "").strip() or (
        "bes" if (enabled and n > 1) else "control"
    )
    stamp: dict[str, Any] = {
        "arm": arm,
        "ab_pair_id": os.environ.get(ENV_AB_PAIR_ID, "").strip() or None,
        "bes": {
            "enabled": enabled,
            "candidates_per_cluster": n,
            "select_metric": metric,
        },
    }
    try:
        cand_file = Path(project_dir) / "rlm_state" / "bes_candidates.json"
        if cand_file.is_file():
            data = json.loads(cand_file.read_text(encoding="utf-8"))
            stamp["bes"]["winner"] = data.get("winner")
            stamp["bes"]["pool"] = [
                {
                    "candidate_id": c.get("candidate_id"),
                    "ok": c.get("ok"),
                    "score": c.get("score"),
                }
                for c in (data.get("candidates") or [])
                if isinstance(c, dict)
            ]
    except Exception:  # noqa: BLE001 — stamp must never block the report write
        logger.debug("bes_rlm: pool summary unavailable for stamp", exc_info=True)
    try:
        adaptive_file = Path(project_dir) / "rlm_state" / "bes_adaptive.json"
        if adaptive_file.is_file():
            stamp["bes"]["adaptive"] = json.loads(adaptive_file.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        logger.debug("bes_rlm: adaptive decision unavailable for stamp", exc_info=True)
    return stamp


__all__ = [
    "ENV_AB_ARM",
    "ENV_AB_PAIR_ID",
    "compete",
    "experiment_arm_stamp",
    "is_enabled",
    "should_compete",
]
