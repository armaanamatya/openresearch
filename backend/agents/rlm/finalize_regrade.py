"""Finalize-time freshness re-grade — never ship a stale grade of grown evidence.

2026-06-13 All-CNN v5: the root graded the rubric ONCE at 01:35 when the cells
grid was ~16 cells in, scored 0.5413, then the grid ran 9 more hours to a
13-of-14-converged completion (≈ the v4 grid that scored 0.744) — and the root
never re-graded. The final shipped the stale partial grade. The best-of-run
floor and finalize_rescore both RE-ROLL already-graded leaves; neither
RE-GRADES evidence that landed after the last grade. So a complete, earned
grid sat on disk ungraded and the run shipped ~0.18 below what it earned.

This module closes that gap generically: at finalize, when the on-disk
``code/metrics.json`` is materially newer than the last grade
(``rubric_evaluation.json``) AND the recorded grade is below target (room to
recover), re-run the leaf scorer against the COMPLETE evidence and adopt the
result only if it scores HIGHER (best-of-run MAX semantics — a re-grade never
lowers what the run already earned). The fresh grade is persisted so
``write_final_report_rlm``'s merge ships it.

One extra LLM grading call, gated to fire only on the stale-and-below-target
shape (not every finalize). Default ON; ``REPROLAB_FINALIZE_REGRADE=0``
disables (env_pin precedent — a correctness rail). Fail-soft everywhere: any
error keeps the recorded grade untouched. Paper-agnostic — keys off mtimes and
the rubric on disk, nothing paper-specific.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ENV_FLAG = "REPROLAB_FINALIZE_REGRADE"
# Evidence must be at least this much newer than the last grade to count as
# "grown since graded" (filters same-second re-writes / metadata touches).
_STALENESS_MARGIN_S = 120.0


def is_enabled() -> bool:
    return os.environ.get(ENV_FLAG, "").strip().lower() not in ("0", "false", "off")


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


# Recognized scalar-metric keys across every paper shape we grade — All-CNN
# (test_error_pct/test_accuracy), Adam (test_accuracy/final_nll/elbo per
# family×optimizer), SDAR (metric), VAE (elbo_final/final_nll). The regrade
# gate only needs "is there ≥1 real measured number here worth grading" — it
# does NOT judge quality (the LLM grader does), so it must be SHAPE-ROBUST
# rather than assume a fixed per_model[model][env][baseline] depth (the
# 2026-06-13 Adam v10 miss: a populated per_model[family][optimizer] shape
# read as 0 evidence under the old rigid 3-level walk, self-skipping the
# regrade on the very run that earned a 78-cell grid and never graded it).
_METRIC_KEYS: frozenset[str] = frozenset({
    "test_error_pct", "test_accuracy", "accuracy", "train_accuracy", "metric",
    "final_nll", "nll", "elbo", "elbo_final", "final_train_loss", "val_accuracy",
    "test_nll", "reward", "score",
})


def _metric_bearing_leaves(node: Any, _depth: int = 0) -> int:
    """Recursively count measured leaves under per_model (shape-agnostic).

    A dict is one measured leaf when EITHER:
      (a) it carries a recognized metric KEY with a finite value
          (All-CNN: ``{test_error_pct, test_accuracy, ...}``), or
      (b) it is a flat label→number map — numeric values, no nested dicts
          (Adam: ``per_model[family] = {adam: 0.33, sgd_nesterov: 0.31, ...}``
          where the optimizer name maps straight to a scalar, no metric key).
    Otherwise recurse into nested dicts. Depth-bounded. Counts measured
    EVIDENCE, not quality (the grader judges quality); the gate only needs > 0
    to justify one LLM grading call. The (b) rule is the 2026-06-13 Adam v10
    fix: its populated 5-family per_model read as 0 under a metric-key-only
    walk, self-skipping the regrade on a 78-cell ungraded grid.
    """
    if not isinstance(node, dict) or _depth > 8:
        return 0
    # (a) explicit metric key.
    for k, v in node.items():
        if k in _METRIC_KEYS:
            try:
                float(v)
                return 1
            except (TypeError, ValueError):
                continue
    vals = list(node.values())
    if vals and not any(isinstance(v, dict) for v in vals):
        # (b) flat {label: number} leaf — count iff ≥1 value is numeric.
        for v in vals:
            try:
                float(v)
                return 1
            except (TypeError, ValueError):
                continue
        return 0
    return sum(_metric_bearing_leaves(v, _depth + 1) for v in vals)


def _converged_cell_count(metrics: dict) -> int:
    """How many measured leaves the on-disk grid carries (shape-robust)."""
    pm = metrics.get("per_model") if isinstance(metrics, dict) else None
    if not isinstance(pm, dict):
        return 0
    return sum(_metric_bearing_leaves(model) for model in pm.values())


def should_regrade(project_dir: Path, *, recorded_score: float | None,
                   target: float | None) -> tuple[bool, str]:
    """Deterministic gate. Returns (fire, reason)."""
    if not is_enabled():
        return False, "disabled"
    code_metrics = project_dir / "code" / "metrics.json"
    eval_path = project_dir / "rubric_evaluation.json"
    if not code_metrics.is_file():
        return False, "no_metrics_on_disk"
    if recorded_score is None:
        # No grade recorded at all but a grid exists → grade it.
        return True, "no_recorded_grade"
    eval_mtime = _mtime(eval_path)
    metrics_mtime = _mtime(code_metrics)
    if eval_mtime is None:
        return True, "no_prior_eval_file"
    # The target gate is now SUBORDINATE to evidence freshness: skip-at-target
    # only when there is no MATERIAL new evidence since the grade. When the grid
    # grew after the grade we re-grade EVEN at/above target — a complete grid can
    # out-score the partial that first reached the floored target (record-chase /
    # maximization); best-of-run MAX in maybe_regrade only adopts a strictly-higher
    # result, so a no-improvement regrade is discarded. 2026-06-14 Codex review:
    # the old early `already_meets_target` return capped a maximization run at the
    # floor and never re-graded a grown grid past it.
    if metrics_mtime is None:
        if target is not None and recorded_score >= target:
            return False, "already_meets_target"
        return False, "metrics_unstat"
    if metrics_mtime - eval_mtime < _STALENESS_MARGIN_S:
        # No material new evidence since the grade — the common no-op.
        if target is not None and recorded_score >= target:
            return False, "already_meets_target"
        return False, "grade_is_fresh"
    return True, f"evidence_grew_{int(metrics_mtime - eval_mtime)}s_after_grade"


def _load_rubric(project_dir: Path) -> tuple[dict | None, str]:
    """The rubric the run was graded against (arXiv: generated; bundle: spec)."""
    gen = project_dir / "generated_rubric.json"
    if gen.is_file():
        try:
            r = json.loads(gen.read_text(encoding="utf-8"))
            if isinstance(r, dict) and r:
                return r, str(r.get("source") or "generated")
        except (OSError, json.JSONDecodeError):
            pass
    return None, ""


def maybe_regrade(ctx: Any, report: Any) -> dict | None:
    """Re-grade the complete on-disk evidence if the recorded grade is stale.

    Mutates ``report.rubric`` in place and returns the fresh grade dict when it
    adopts a higher score; returns None (report untouched) otherwise. Never
    raises.
    """
    try:
        project_dir = Path(ctx.project_dir)
        rubric_block = dict(getattr(report, "rubric", None) or {})
        recorded = rubric_block.get("overall_score")
        try:
            recorded_f = float(recorded) if recorded is not None else None
        except (TypeError, ValueError):
            recorded_f = None
        try:
            target_f = float(rubric_block.get("target_score")) if rubric_block.get("target_score") is not None else None
        except (TypeError, ValueError):
            target_f = None

        fire, reason = should_regrade(project_dir, recorded_score=recorded_f, target=target_f)
        if not fire:
            logger.debug("finalize_regrade: skip (%s)", reason)
            return None

        rubric, source = _load_rubric(project_dir)
        if rubric is None:
            logger.info("finalize_regrade: no rubric on disk — cannot re-grade")
            return None

        llm_client = getattr(ctx, "llm_client", None)
        if llm_client is None:
            return None

        # Quick evidence-growth sanity: only spend an LLM call when the complete
        # grid actually carries real converged cells (not an empty/placeholder).
        try:
            metrics = json.loads((project_dir / "code" / "metrics.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if _converged_cell_count(metrics) <= 0:
            logger.info("finalize_regrade: on-disk metrics carry no converged cells — skip")
            return None

        logger.info(
            "finalize_regrade: re-grading complete evidence (%s; recorded=%.4f target=%s)",
            reason, recorded_f if recorded_f is not None else -1.0, target_f,
        )
        from backend.evals.paperbench.leaf_scorer import score_reproduction

        fresh = score_reproduction(
            rubric_tree=rubric,
            run_dir=project_dir,
            llm_client=llm_client,
            rubric_source=source,
            # NOT degraded by construction: the _converged_cell_count gate above proved
            # real converged cells. Without this explicit False, score_reproduction's
            # degraded=None auto-detect reads a stale on-disk final_report.json and, on
            # an empty-baseline_metrics / verdict="failed" report, caps EVERY leaf at 0.35
            # — nuking the very complete-grid grade this regrade exists to recover.
            degraded=False,
            invariants=list(getattr(ctx, "paper_hint_invariants", None) or []),
            # Layer 4 (2026-06-16): exclude out-of-inclusion-scope leaves on the
            # freshness re-grade too, so a re-grade can't re-introduce the
            # un-excluded in-loop score. No-op unless REPROLAB_SCOPE_INCLUSION
            # _EXCLUDE is on. Mirrors the verify_against_rubric plumbing.
            operator_dataset_inclusion=[
                (getattr(d, "name", None) or str(d))
                for d in (getattr(getattr(ctx, "scope_spec", None), "datasets", None) or [])
                if d
            ],
        )
        fresh_score = fresh.get("overall_score")
        try:
            fresh_f = float(fresh_score) if fresh_score is not None else None
        except (TypeError, ValueError):
            fresh_f = None
        if fresh_f is None:
            return None

        # MAX semantics: adopt only a strict improvement (a re-grade never
        # lowers the high-water mark the run already earned).
        if recorded_f is not None and fresh_f <= recorded_f + 1e-9:
            logger.info(
                "finalize_regrade: fresh grade %.4f did not beat recorded %.4f — keeping recorded",
                fresh_f, recorded_f,
            )
            return None

        # Adopt: write the authoritative fresh grade so write_final_report_rlm
        # merges it, and update the report's rubric block now.
        try:
            target_for_meets = target_f if target_f is not None else fresh.get("target_score")
            if target_for_meets is not None:
                fresh["meets_target"] = bool(fresh_f >= float(target_for_meets))
                fresh["target_score"] = float(target_for_meets)
        except (TypeError, ValueError):
            pass
        try:
            (project_dir / "rubric_evaluation.json").write_text(
                json.dumps(fresh, indent=2), encoding="utf-8"
            )
        except OSError:
            logger.warning("finalize_regrade: could not persist fresh eval", exc_info=True)

        merged = dict(rubric_block)
        for k in ("overall_score", "target_score", "meets_target", "leaf_scores",
                  "weak_leaves", "leaf_count", "graded", "coverage_pct", "areas"):
            if fresh.get(k) is not None:
                merged[k] = fresh[k]
        report.rubric = merged
        logger.info(
            "finalize_regrade: ADOPTED fresh grade %.4f (was %.4f) — recovered stale-partial grade",
            fresh_f, recorded_f if recorded_f is not None else -1.0,
        )
        return fresh
    except Exception:  # noqa: BLE001 — finalize re-grade is advisory, never fatal
        logger.warning("finalize_regrade: failed (non-fatal); keeping recorded grade", exc_info=True)
        return None


def regrade_and_emit(ctx: Any, report: Any, emit: Any) -> dict | None:
    """maybe_regrade + ALWAYS emit the decision (fire/skip + reason).

    The single entry point every ctx-bearing finalize path calls. Observability
    is unconditional: a skip emits ``finalize_regrade_skipped`` with the gate
    reason so a stale/zero final is never silent about WHY it wasn't recovered
    (the 2026-06-13 All-CNN v6 / Adam v10 debugging gap — both shipped 0 with a
    complete grid and no trace of whether the regrade ran). Flips the report
    verdict failed→reproduced when an adopted score meets target. Never raises.
    """
    try:
        if not is_enabled():
            return None
        # Compute the gate reason up front for observability.
        project_dir = Path(ctx.project_dir)
        rubric_block = dict(getattr(report, "rubric", None) or {})
        rec = rubric_block.get("overall_score")
        try:
            rec_f = float(rec) if rec is not None else None
        except (TypeError, ValueError):
            rec_f = None
        try:
            tgt_f = float(rubric_block.get("target_score")) if rubric_block.get("target_score") is not None else None
        except (TypeError, ValueError):
            tgt_f = None
        fire, reason = should_regrade(project_dir, recorded_score=rec_f, target=tgt_f)

        fresh = maybe_regrade(ctx, report) if fire else None
        if fresh is not None:
            _safe_emit(emit, "run_warning", {
                "code": "finalize_regrade_adopted",
                "message": (
                    "finalize re-graded the complete on-disk grid (it grew after "
                    f"the last verify) and adopted the higher score {fresh.get('overall_score')}."
                ),
            })
            try:
                if report.verdict == "failed" and bool(fresh.get("meets_target")):
                    report.verdict = "reproduced"
            except Exception:  # noqa: BLE001
                pass
            return fresh
        _safe_emit(emit, "run_warning", {
            "code": "finalize_regrade_skipped",
            "message": (
                f"finalize re-grade did not change the score (gate: {reason}; "
                f"recorded={rec_f})."
            ),
        })
        return None
    except Exception:  # noqa: BLE001 — observability must never break finalize
        logger.warning("finalize_regrade: regrade_and_emit failed (non-fatal)", exc_info=True)
        return None


def _safe_emit(emit: Any, event: str, payload: dict) -> None:
    try:
        if callable(emit):
            emit(event, payload)
    except Exception:  # noqa: BLE001
        logger.debug("finalize_regrade: emit failed", exc_info=True)


def regrade_for_hard_stop(project_dir: Path | str, llm_client: Any) -> dict | None:
    """Re-grade the completed grid on the hard-stop path (no ctx available).

    The watchdog/SIGTERM salvage finalizer (``_hard_stop_with_report``) has no
    RunContext — only project_dir and a captured llm_client. It salvages the
    best RECORDED score, which is ZERO for a run that completed its grid but
    never verified (Adam's long runs that hit the wall-clock). This grades the
    on-disk grid directly so salvage can floor to it. Returns the fresh grade
    dict (with overall_score) when the grid carries real evidence, else None.
    Never raises.
    """
    try:
        if not is_enabled() or llm_client is None:
            return None
        project_dir = Path(project_dir)
        if not (project_dir / "code" / "metrics.json").is_file():
            return None
        try:
            metrics = json.loads((project_dir / "code" / "metrics.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if _converged_cell_count(metrics) <= 0:
            return None
        rubric, source = _load_rubric(project_dir)
        if rubric is None:
            return None
        from backend.evals.paperbench.leaf_scorer import score_reproduction

        fresh = score_reproduction(
            rubric_tree=rubric, run_dir=project_dir, llm_client=llm_client,
            rubric_source=source,
            degraded=False,  # converged cells proven above; see maybe_regrade for the 0.35-cap rationale
        )
        if fresh.get("overall_score") is None:
            return None
        try:
            (project_dir / "rubric_evaluation.json").write_text(
                json.dumps(fresh, indent=2), encoding="utf-8"
            )
        except OSError:
            pass
        logger.info(
            "finalize_regrade: hard-stop re-graded completed grid → %.4f",
            float(fresh["overall_score"]),
        )
        return fresh
    except Exception:  # noqa: BLE001 — salvage re-grade is best-effort
        logger.warning("finalize_regrade: hard-stop re-grade failed (non-fatal)", exc_info=True)
        return None


__all__ = [
    "ENV_FLAG",
    "is_enabled",
    "maybe_regrade",
    "regrade_and_emit",
    "regrade_for_hard_stop",
    "should_regrade",
]
