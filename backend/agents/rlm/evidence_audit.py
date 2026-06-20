"""Unified deterministic evidence critic.

The single aggregation point over the existing fabrication-predicate modules.
Composes them — never reimplements. The fitness signal is this deterministic
evidence layer, NEVER the LLM grade (the red line).

Master flag ``OPENRESEARCH_EVIDENCE_AUDIT`` (default OFF) -> dormant -> the
per-result veto returns None -> byte-identical to today. See
docs/superpowers/specs/2026-06-20-actor-critic-evidence-critic-redesign-design.md.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_TRUE = ("1", "true", "yes", "on")


def evidence_audit_enabled() -> bool:
    """True iff OPENRESEARCH_EVIDENCE_AUDIT opts the unified critic ON. Default OFF."""
    return os.environ.get("OPENRESEARCH_EVIDENCE_AUDIT", "").strip().lower() in _TRUE


def _provenance_on_disk(ctx: Any) -> bool:
    """True iff code/provenance.json exists under the run dir. Fail-soft -> False.

    Mirrors the inline check at primitives.py:6596 (existence only; well-formedness
    is the validator's concern, not this discriminator's)."""
    try:
        code = ctx.project_dir / "code"
        return code.is_dir() and any(code.rglob("provenance.json"))
    except Exception:
        return False


@dataclass(frozen=True)
class EvidenceAudit:
    """Run-level deterministic evidence snapshot. Built from on-disk state by
    ``audit_evidence``; consumed by the verdict gate, recipe admission, and the
    validator. ``run_level_clean`` is the ONE run-level evidence predicate."""

    backed_by_ledger: bool
    provenance_present: bool
    metrics_non_degenerate: bool
    metric_keys_real: bool
    fingerprint: str
    reasons: tuple[str, ...] = ()
    rerun_agrees: bool | None = None  # populated in Pillar 2; None == no disagreement

    @property
    def run_level_clean(self) -> bool:
        return (
            self.backed_by_ledger
            and self.metrics_non_degenerate
            and self.metric_keys_real
            and (self.rerun_agrees is None or self.rerun_agrees)
        )


def result_is_fabricated(result: Any, ctx: Any, *, peak_vram_gb: float | None = None) -> str | None:
    """Unified per-result fabrication veto for run_experiment.

    Composes the zero-metrics, stub, and VRAM-antifab predicates into one seam,
    returning a repair-reason string when fabrication is suspected, else None.
    Dormant (returns None) unless the master flag is on -> byte-identical to today.
    Fail-soft: any internal error returns None (the critic never blocks a run on its
    own bug). Wiring into run_experiment is Plan 2 (this ships dormant + tested).
    """
    if not evidence_audit_enabled():
        return None
    if not isinstance(result, dict) or not result.get("success"):
        return None
    metrics = result.get("metrics")
    try:
        from backend.agents.rlm.zero_metrics_detection import (  # noqa: PLC0415
            looks_like_zero_metrics,
            zero_metrics_repair_message,
        )
        from backend.agents.rlm.stub_detection import (  # noqa: PLC0415
            looks_like_stub_metrics,
            stub_repair_message,
        )
        from backend.agents.rlm.gpu_cell_runner import (  # noqa: PLC0415
            metrics_claim_gpu_training,
            vram_evidence_verdict,
        )
    except Exception:
        return None

    try:
        if looks_like_stub_metrics(metrics):
            return stub_repair_message(metrics)
        gpu_claim = metrics_claim_gpu_training(metrics)
        provenance = _provenance_on_disk(ctx)
        if gpu_claim and not provenance and looks_like_zero_metrics(metrics):
            return zero_metrics_repair_message(metrics)
        if vram_evidence_verdict(peak_vram_gb, claims_gpu_training=gpu_claim):
            return (
                f"gpu training claimed but peak VRAM {peak_vram_gb:.2f} GiB "
                f"is below the fabrication floor"
            )
    except Exception:
        return None
    return None


def apply_result_veto(
    result: Any,
    ctx: Any,
    *,
    peak_vram_gb: float | None = None,
    emit: Callable[[str], None] | None = None,
) -> Any:
    """Degrade a result the unified critic flags to a repairable fabrication_suspected
    failure (mirrors the existing per-guard pattern in run_experiment). Returns the input
    ``result`` unchanged when not flagged — byte-identical when the master flag is off
    (``result_is_fabricated`` returns None). ``emit`` (if given) is called with the reason
    for a run_warning; emit failures are swallowed (diagnostics must never break a run)."""
    reason = result_is_fabricated(result, ctx, peak_vram_gb=peak_vram_gb)
    if reason is None:
        return result
    degraded = {
        **result,
        "success": False,
        "failure_class": "fabrication_suspected",
        "error": reason,
    }
    if emit is not None:
        try:
            emit(reason)
        except Exception:
            pass
    return degraded


def _load_latest_metrics(ctx: Any) -> dict:
    """Latest on-disk metrics for the run. Tries leaf_scorer._latest_metrics_path,
    falls back to code/metrics.json. Fail-soft -> {}."""
    import json  # noqa: PLC0415

    path = None
    try:
        from backend.evals.paperbench.leaf_scorer import _latest_metrics_path  # noqa: PLC0415

        path = _latest_metrics_path(ctx.project_dir)
    except Exception:
        path = None
    try:
        if path is None:
            cand = ctx.project_dir / "code" / "metrics.json"
            path = cand if cand.exists() else None
        if path is not None and path.exists():
            data = json.loads(path.read_text())
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    return {}


def _load_latest_metrics_from_dir(project_dir: Any) -> dict:
    """Latest on-disk metrics from ``project_dir`` alone. Same loader chain as
    ``_load_latest_metrics`` but takes a Path instead of a ctx. Fail-soft -> {}."""
    import json  # noqa: PLC0415

    from pathlib import Path  # noqa: PLC0415

    project_dir = Path(project_dir)
    path = None
    try:
        from backend.evals.paperbench.leaf_scorer import _latest_metrics_path  # noqa: PLC0415

        path = _latest_metrics_path(project_dir)
    except Exception:
        path = None
    try:
        if path is None:
            cand = project_dir / "code" / "metrics.json"
            path = cand if cand.exists() else None
        if path is not None and path.exists():
            data = json.loads(path.read_text())
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    return {}


def _provenance_on_disk_from_dir(project_dir: Any) -> bool:
    """True iff code/provenance.json exists under project_dir. Fail-soft -> False."""
    from pathlib import Path  # noqa: PLC0415

    try:
        code = Path(project_dir) / "code"
        return code.is_dir() and any(code.rglob("provenance.json"))
    except Exception:
        return False


def audit_evidence_from_dir(
    project_dir: Any,
    *,
    ok_count: int | None = None,
) -> EvidenceAudit:
    """Ctx-free variant of :func:`audit_evidence`.

    Computes the same run-level deterministic evidence snapshot from
    ``project_dir`` alone, without requiring a live ``RunContext``.

    Args:
        project_dir: Path to the run project directory (coerced via ``Path``).
        ok_count: In-process success-compatible ``run_experiment`` call count
            from the cost ledger. ``None`` means the ledger is unavailable;
            ``backed_by_ledger`` then defaults to ``True`` (replay / postmortem
            rule — trust content rather than fail-closed). Pass ``0`` to
            explicitly mark no successful call.

    Returns:
        An :class:`EvidenceAudit` snapshot. Fail-soft: any error yields the
        safe all-true snapshot (never blocks a run on its own bug).
    """
    from pathlib import Path  # noqa: PLC0415

    project_dir = Path(project_dir)
    metrics = _load_latest_metrics_from_dir(project_dir)

    # backed_by_ledger: None ok_count -> trust content (replay rule)
    backed = True
    if ok_count is not None:
        backed = ok_count >= 1

    non_degen = True
    keys_real = True
    try:
        from backend.agents.rlm.zero_metrics_detection import looks_like_zero_metrics  # noqa: PLC0415
        from backend.agents.rlm.stub_detection import looks_like_stub_metrics  # noqa: PLC0415

        non_degen = not looks_like_zero_metrics(metrics)
        keys_real = not looks_like_stub_metrics(metrics)
    except Exception:
        non_degen = True
        keys_real = True

    fingerprint = ""
    try:
        from backend.agents.rlm.evidence_key import evidence_key  # noqa: PLC0415

        scope = metrics.get("scope") if isinstance(metrics, dict) else None
        fingerprint = evidence_key(metrics if isinstance(metrics, dict) else {}, scope)
    except Exception:
        fingerprint = ""

    reasons: list[str] = []
    if not backed:
        reasons.append("no in-process run_experiment success")
    if not non_degen:
        reasons.append("metrics all-zero/constant")
    if not keys_real:
        reasons.append("metrics keys are placeholders")

    return EvidenceAudit(
        backed_by_ledger=backed,
        provenance_present=_provenance_on_disk_from_dir(project_dir),
        metrics_non_degenerate=non_degen,
        metric_keys_real=keys_real,
        fingerprint=fingerprint,
        reasons=tuple(reasons),
    )


def audit_evidence(ctx: Any) -> EvidenceAudit:
    """Run-level deterministic evidence snapshot from on-disk state + the ledger.

    Pure read-only function (modulo ledger session counters) -> deterministic
    ``fingerprint``. Consumed by the verdict gate, recipe admission, and the
    validator. Fail-soft: any error yields the safe all-true snapshot.

    Delegates to :func:`audit_evidence_from_dir` after extracting
    ``project_dir`` and ``ok_count`` from ``ctx``."""
    ok_count: int | None = None
    try:
        from backend.agents.rlm.report import run_experiment_success_count  # noqa: PLC0415

        ok_count = run_experiment_success_count(ctx)
    except Exception:
        ok_count = None

    try:
        project_dir = ctx.project_dir
    except Exception:
        # Fail-soft: ctx missing project_dir -> return the safe all-true snapshot
        return EvidenceAudit(
            backed_by_ledger=True,
            provenance_present=False,
            metrics_non_degenerate=True,
            metric_keys_real=True,
            fingerprint="",
            reasons=(),
        )

    return audit_evidence_from_dir(project_dir, ok_count=ok_count)
