"""Final-report builder for RLM-mode paper-reproduction runs.

Converts an `RLMChatCompletion` result into an `RLMFinalReport` Pydantic
model and writes `final_report.{json,md}` atomically under the project dir.

Design contract: spec §11 (2026-05-21-rlm-phase3-orchestrator-design.md).
"""

from __future__ import annotations

import ast
import json
import logging
import os
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from rlm.core.types import RLMChatCompletion

from backend.agents.rlm.context import RunContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class RLMFinalReport(BaseModel):
    """Structured output of one RLM-mode paper-reproduction run.

    All fields use honest defaults: an under-reporting root model produces a
    `partial` or `failed` verdict, never an exception.
    """

    paper: dict = Field(
        default_factory=dict,
        description="Paper identity: {id, title, ...}",
    )
    verdict: Literal["reproduced", "partial", "failed"] = "failed"
    reproduction_summary: str = ""
    baseline_metrics: dict = Field(
        default_factory=dict,
        description="Measured metrics; may be {} until Phase 5.",
    )
    paper_claims: dict = Field(default_factory=dict)
    # ── paper_claims coercer ──
    # The root model occasionally returns paper_claims as a LIST of claim
    # dicts (e.g. [{"method": "RLM(GPT-5,…)", "expected_result": "62.0"}, …])
    # instead of the keyed dict the schema expects. Rather than rejecting the
    # entire run at the final-report step (which discards 30+ minutes of work),
    # coerce list → dict by keying on the first available identity field
    # (method/claim/id/name) and falling back to integer index. The downstream
    # report renderer at line 578-584 only iterates over the dict's keys for
    # display, so any stable string key works.
    @field_validator("paper_claims", mode="before")
    @classmethod
    def _coerce_paper_claims(cls, v: object) -> dict:
        if isinstance(v, dict):
            return v
        if not isinstance(v, list):
            return {}
        out: dict = {}
        for i, item in enumerate(v):
            if not isinstance(item, dict):
                continue
            key = (
                item.get("method")
                or item.get("claim")
                or item.get("claim_id")
                or item.get("id")
                or item.get("name")
                or f"claim_{i}"
            )
            out[str(key)] = item
        return out
    rubric: dict = Field(
        # C2c (second pass, 2026-05-22): unscored runs honestly read as null on
        # both overall_score and meets_target — never a fabricated 0.0 / False.
        # The post-run leaf scorer overwrites these with real values via
        # amend_final_report when scoring actually happens. A run that dies
        # before reaching the scorer (e.g. credential failure at iter 0) keeps
        # the nulls — which is the honest "not scored" signal.
        default_factory=lambda: {
            "overall_score": None,
            "meets_target": None,
            "target_score": None,
            "degraded": None,
            "areas": [],
        },
    )
    improvements: list[dict] = Field(default_factory=list)
    primitive_trace: dict = Field(default_factory=dict)
    cost: dict = Field(
        default_factory=lambda: {"llm_usd": 0.0, "primitives": 0.0},
        description=(
            "Honest cost dict. 'llm_usd' = total (root + sub + primitives). "
            "'primitives' = primitive-internal LLM cost from the cost ledger. "
            "The false 'root'/'sub' split is dropped (T7/M-BUDGET) because rlm "
            "does not tag usage by tier — all rlm cost is reported as a single "
            "honest 'llm_usd' figure."
        ),
    )
    iterations: int = 0
    primitive_provider: str = "real"  # "real" | "stub" (T21 / review I8)
    degraded: bool = False  # True for stub runs and other degraded states (T21)

    # --- Phase-4-forward-compat fields (spec 2026-05-23-rubric-climb-leaderboard §4.5)
    # Forward-compatible with the cleanup-spec Phase-4 per-role model picker.
    # Unknown roles stay None until the picker lands.
    mode: Literal["rlm", "rdr"] = Field(
        default="rlm",
        description="Reproduction mode (rlm root-loop or rdr controller).",
    )
    models: dict[str, str | None] = Field(
        default_factory=lambda: {
            "planner": None,
            "executor": None,
            "verifier": None,
            "grader": None,
        },
        description=(
            "Per-role model identifiers — forward-compatible with the "
            "future per-role picker."
        ),
    )
    started_at: str | None = Field(
        default=None,
        description="ISO-8601 UTC timestamp when the run started.",
    )
    completed_at: str | None = Field(
        default=None,
        description="ISO-8601 UTC timestamp when the report was written.",
    )
    title: str | None = Field(
        default_factory=lambda: (os.environ.get("REPROLAB_RUN_TITLE") or "").strip() or None,
        description=(
            "Human-readable run title (e.g. 'SDAR full · 2026-05-31 19:05'), "
            "surfaced as the leaderboard row label so repeated runs of the same "
            "paper are distinguishable. Sourced from REPROLAB_RUN_TITLE at "
            "report-construction time."
        ),
    )

    # --- Scope section (spec 2026-05-23-sdar-baseline-handoff §Lane 4)
    # Distinguishes "what the user / operator scoped the run to" from "what the
    # rubric evaluates". A partial scope (e.g. only the smallest 2 of 3 model
    # sizes) is not the same as a partial rubric pass; conflating them
    # misrepresents the run.
    scope: dict = Field(
        default_factory=lambda: {
            "requested": "",
            "ran": [],
            "gaps": [],
        },
        description=(
            "User/operator-stated scope vs. what actually ran. "
            "`requested` = the scope statement (e.g. operator guidance, "
            "CLI hint, default 'full paper'). `ran` = list of items actually "
            "executed (model names, dataset slices, seeds). `gaps` = list of "
            "items requested but not executed, with a short reason each."
        ),
    )
    stop_reason: dict | None = Field(
        default=None,
        description=(
            "Set when the run STOPPED on an un-recoverable capacity/OOM condition "
            "rather than completing (2026-05-31): "
            "{kind: 'oom_shrink_exhausted'|'capacity_exhausted', detail, "
            "per_gpu_vram_gb, models_skipped, environments_skipped, gaps}. "
            "None on a normally-completed run."
        ),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_HONEST_DEFAULTS: dict[str, Any] = {
    "paper": {},
    "verdict": "failed",
    "reproduction_summary": "",
    "baseline_metrics": {},
    "paper_claims": {},
    # C2c (second pass): unscored defaults are null, not 0.0 / False. See
    # RLMFinalReport.rubric default_factory above for the rationale.
    "rubric": {
        "overall_score": None,
        "meets_target": None,
        "target_score": None,
        "degraded": None,
        "areas": [],
    },
    "improvements": [],
    "scope": {"requested": "", "ran": [], "gaps": []},
    "primitive_trace": {},
    "cost": {"llm_usd": 0.0, "primitives": 0.0},
    "iterations": 0,
}

_VALID_VERDICTS = frozenset({"reproduced", "partial", "failed"})

# ---------------------------------------------------------------------------
# Rubric-evidence score thresholds for verdict reconciliation
#
# PaperBench scores run low in practice: partial credit on a handful of leaves
# can clear 0.15 even without a real reproduction, but a "reproduced" claim
# requires the majority of rubric criteria to be satisfied.  The ceilings below
# are deliberately conservative so that a zero-score run (e.g. the `ftrl` run:
# verdict "reproduced" at leaf score 0.000) is downgraded to "failed" rather
# than silently mislabelled as a success.
# ---------------------------------------------------------------------------
_VERDICT_REPRODUCED_MIN_SCORE: float = 0.60  # need most rubric leaves satisfied
_VERDICT_PARTIAL_MIN_SCORE: float = 0.15     # at least some rubric leaves graded

# Numeric rank so we can compare verdicts (higher = stronger claim)
_VERDICT_RANK: dict[str, int] = {"failed": 0, "partial": 1, "reproduced": 2}


def _parse_response(raw: str) -> dict | None:
    """Try to parse `raw` as JSON, then as Python repr (ast.literal_eval).

    Returns a dict on success, None on both failing.  Never raises.
    """
    # Fast path: clean JSON
    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
        logger.warning("report: json.loads succeeded but result is %s, not dict", type(result))
    except json.JSONDecodeError:
        pass

    # Fallback: Python repr (FINAL_VAR str()-ifies a dict)
    try:
        result = ast.literal_eval(raw)
        if isinstance(result, dict):
            return result
        logger.warning(
            "report: ast.literal_eval succeeded but result is %s, not dict", type(result)
        )
    except (ValueError, SyntaxError):
        pass

    return None


def _reconcile_verdict_against_evidence(
    verdict: str,
    *,
    baseline_metrics: dict,
    rubric: dict,
    primitive_trace: dict,
) -> tuple[str, str | None]:
    """Downgrade an over-claimed verdict; return (verdict, reason_or_None).

    A run can only claim "reproduced" if all three honesty checks pass:
      - run_experiment was actually called (primitive_trace records it)
      - baseline_metrics is non-empty (real measured numbers exist)
      - rubric.overall_score >= 0.5 (the score actually shows reproduction)

    Any failure downgrades "reproduced" -> "partial" with a reason string.
    "partial" and "failed" verdicts are passed through unchanged.
    """
    if verdict != "reproduced":
        return verdict, None
    score = float((rubric or {}).get("overall_score", 0.0) or 0.0)
    ran_experiment = bool(primitive_trace.get("by_primitive", {}).get("run_experiment"))
    reasons: list[str] = []
    if not ran_experiment:
        reasons.append("run_experiment never ran")
    if not baseline_metrics:
        reasons.append("no measured baseline metrics")
    if score < 0.5:
        reasons.append(f"rubric score {score:.3f} < 0.5")
    if reasons:
        return "partial", "; ".join(reasons)
    return verdict, None


def _normalise_model_name(name: str) -> str:
    """Lowercase + strip for case-insensitive model-name comparison."""
    return name.strip().lower()


def _operator_skip_set(operator_skip_models: list[str] | None) -> frozenset[str]:
    """Return a normalised frozenset of operator-intended skip model names."""
    if not operator_skip_models:
        return frozenset()
    return frozenset(_normalise_model_name(m) for m in operator_skip_models if m)


def _collect_data_unavailable_gaps(
    project_dir: Path,
    operator_skip_models: list[str] | None = None,
) -> list[str]:
    """Scan the run's emitted metrics for datasets AND models the agent recorded
    as unavailable, returning clear, deduped gap strings for ``scope.gaps``.

    Runtime signals (the same ones the unavailable-component-aware grader honours):
      * ``data_load_failures[]`` / ``experiments[*].status=='data_unavailable'`` — datasets.
      * ``per_model[*].status in {model_load_failed, failed, skipped, ...}``,
        ``scope.models_skipped[]``, ``model_load_failures[]`` — models.

    These are surfaced so the report states plainly which datasets/models could
    not be obtained — and, per the grader, were EXCLUDED from the rubric score
    (numerator + denominator) rather than scored zero. So the run always reaches
    a final summary + rubric and only the failed pieces drop out (2026-05-30
    graceful-degradation mandate). Best-effort: [] when no metrics file is present.

    ``operator_skip_models``: the operator-intended skip list from
    ``ScopeSpec.skip_models``.  A model that appears in ``scope.models_skipped``
    BUT is NOT in ``operator_skip_models`` was REQUESTED yet failed to load —
    the agent's code caught a load exception (``TypeError``, architecture error,
    ``unexpected keyword argument``, etc.) and silently laundered it into a
    scope-reduction entry.  These are reported as "load failure (repairable code
    bug)" and are NOT silently excluded from the rubric — they surface as
    repair-context for the next iteration.  Only genuinely operator-intended
    skips (present in ``operator_skip_models``) are tagged "scope reduction" and
    excluded from scoring.
    """
    datasets: dict[str, str] = {}   # name(lower) -> reason
    # models dict: name(lower) -> (reason, is_repairable_bug)
    # is_repairable_bug=True  → requested model whose load failed (code bug)
    # is_repairable_bug=False → operator-intended scope reduction (legitimate)
    models: dict[str, tuple[str, bool]] = {}
    _MODEL_FAIL = {"model_load_failed", "failed", "skipped", "data_unavailable", "unavailable"}
    op_skip = _operator_skip_set(operator_skip_models)

    outputs = project_dir / "code" / "outputs"
    for mpath in (sorted(outputs.rglob("metrics.json")) if outputs.exists() else []):
        try:
            data = json.loads(mpath.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — a malformed metrics file must not break the report
            continue
        if not isinstance(data, dict):
            continue
        # --- datasets ---
        for entry in data.get("data_load_failures") or []:
            if isinstance(entry, dict):
                name = str(entry.get("dataset") or entry.get("name") or "").strip()
                reason = str(entry.get("error") or entry.get("reason") or "").strip()
            elif isinstance(entry, str):
                name, reason = entry.strip(), ""
            else:
                continue
            if name:
                datasets.setdefault(name.lower(), reason)
        exps = data.get("experiments")
        if isinstance(exps, dict):
            for exp_id, meta in exps.items():
                if isinstance(meta, dict) and str(meta.get("status", "")).lower() == "data_unavailable":
                    name = str(exp_id).strip()
                    if name:
                        datasets.setdefault(name.lower(), str(meta.get("reason") or "").strip())
        # --- models ---
        for m, mv in (data.get("per_model") or {}).items():
            if isinstance(mv, dict) and str(mv.get("status", "")).lower() in _MODEL_FAIL:
                key = _normalise_model_name(m)
                reason = str(mv.get("reason") or mv.get("error") or "").strip()
                # A per_model status failure is always a runtime failure, not an
                # operator-intended skip.  These are repairable code bugs unless the
                # operator explicitly de-scoped the model.
                is_bug = key not in op_skip
                models.setdefault(key, (reason, is_bug))
        for m in ((data.get("scope") or {}).get("models_skipped") or []):
            if isinstance(m, str) and m.strip():
                key = _normalise_model_name(m)
                # Distinguish: is this an operator-intended skip, or did the agent
                # quietly dump a load failure into models_skipped to avoid scoring?
                is_bug = key not in op_skip
                models.setdefault(key, ("scope reduction" if not is_bug else "", is_bug))
        for entry in data.get("model_load_failures") or []:
            if isinstance(entry, dict) and (entry.get("model") or entry.get("name")):
                key = _normalise_model_name(str(entry.get("model") or entry.get("name")))
                reason = str(entry.get("error") or entry.get("reason") or "").strip()
                is_bug = key not in op_skip
                models.setdefault(key, (reason, is_bug))
            elif isinstance(entry, str) and entry.strip():
                key = _normalise_model_name(entry)
                is_bug = key not in op_skip
                models.setdefault(key, ("", is_bug))
    gaps: list[str] = []
    for name, reason in sorted(datasets.items()):
        tail = f" ({reason[:160]})" if reason else ""
        gaps.append(f"{name}: dataset unobtainable{tail} — excluded from rubric score, not penalised")
    for name, (reason, is_repairable_bug) in sorted(models.items()):
        if is_repairable_bug:
            # Requested model whose load failed in agent code — surface as a
            # repairable failure so the repair loop sees it, NOT as a silent
            # scope-exclusion that would launder the bug into a 0.188 score.
            tail = f" ({reason[:160]})" if reason else ""
            gaps.append(
                f"{name}: model load failure (repairable code bug){tail}"
                f" — NOT excluded from rubric; fix the loader and re-run"
            )
        else:
            # Operator-intended scope reduction — legitimately excluded.
            tail = f" ({reason[:160]})" if reason else ""
            gaps.append(f"{name}: model unavailable{tail} — excluded from rubric score, not penalised")
    return gaps


def _merge_data_unavailable_gaps(
    scope: dict,
    project_dir: Path,
    operator_skip_models: list[str] | None = None,
) -> dict:
    """Merge auto-collected data-unavailable gaps into ``scope.gaps`` (deduped by
    leading dataset token), so unobtainable datasets are reported even when the
    root model did not declare them in scope.gaps itself.

    ``operator_skip_models`` is forwarded to ``_collect_data_unavailable_gaps``
    so the intentional-skip vs repairable-code-bug distinction is preserved.
    """
    auto = _collect_data_unavailable_gaps(project_dir, operator_skip_models=operator_skip_models)
    if not auto:
        return scope
    existing = list((scope or {}).get("gaps") or [])
    existing_tokens = {str(g).split(":", 1)[0].strip().lower() for g in existing}
    merged = list(existing)
    for g in auto:
        if g.split(":", 1)[0].strip().lower() not in existing_tokens:
            merged.append(g)
    return {**(scope or {}), "gaps": merged}


def _verify_scope_evidence(
    scope: dict,
    run_dir: Path,
) -> tuple[dict, str | None]:
    """Cross-check ``scope.ran`` against ``experiment_runs.jsonl`` model_id/eval_env tags.

    The root model self-attests ``scope.ran``; this function moves any item
    claimed in ``ran`` but lacking a successful ``run_experiment`` row with
    matching tags into ``scope.gaps``. Returns ``(new_scope, downgrade_reason)``
    where ``downgrade_reason`` is ``None`` on a clean cross-check.

    Evidence model:
      - Each successful experiment_runs.jsonl row carries ``model_id`` and
        ``eval_env`` tags (PR A).
      - For multi-model + multi-dataset scopes, the expected scope.ran ids
        are composite ``"<model>/<env>"`` strings; the cross-check accepts
        either composite or plain-model ids when only one dimension is
        multi.
      - When neither model_id nor eval_env tag is anywhere in the log
        (legacy / single-config runs), the cross-check is a no-op so old
        runs are not mis-flagged.
    """
    if not isinstance(scope, dict):
        return scope, None
    ran_claimed = set(scope.get("ran") or [])
    if not ran_claimed:
        return scope, None

    exp_log = run_dir / "experiment_runs.jsonl"
    if not exp_log.exists():
        return scope, None

    evidence_ids: set[str] = set()
    has_any_tag = False
    for line in exp_log.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        model_id = entry.get("model_id")
        eval_env = entry.get("eval_env")
        # has_any_tag is set from all rows (including failed) so legacy
        # detection works correctly: any row with a non-default tag means
        # this is a tagged run and enforcement applies.
        if model_id and model_id != "default":
            has_any_tag = True
        if eval_env and eval_env != "default":
            has_any_tag = True
        # Only successful runs contribute to evidence_ids.
        if not entry.get("success"):
            continue
        if model_id and model_id != "default":
            evidence_ids.add(str(model_id))
        if eval_env and eval_env != "default":
            evidence_ids.add(str(eval_env))
        if (
            model_id and eval_env
            and model_id != "default" and eval_env != "default"
        ):
            evidence_ids.add(f"{model_id}/{eval_env}")

    # Legacy runs: every row tagged "default" → cross-check is a no-op.
    if not has_any_tag:
        return scope, None

    unverified = sorted(ran_claimed - evidence_ids)
    if not unverified:
        return scope, None

    new_scope = {
        **scope,
        "ran": sorted(ran_claimed - set(unverified)),
        "gaps": list(scope.get("gaps") or []) + [
            f"{item}: claimed in scope.ran but no successful run_experiment found with matching tag"
            for item in unverified
        ],
    }
    return new_scope, (
        f"moved {len(unverified)} unverified item(s) from scope.ran to scope.gaps"
    )


def _reconcile_verdict(parsed: dict) -> str:
    """Return an honest verdict.

    Accepts the value from the parsed report if it is in the valid set,
    otherwise down-grades to `partial` (something came back but is suspect).
    """
    raw = parsed.get("verdict", "")
    if raw in _VALID_VERDICTS:
        return raw
    # Something was returned but verdict is missing or unknown → partial
    return "partial"


def _cost_dict(result: RLMChatCompletion, ctx: RunContext) -> dict:
    """Reconcile LLM cost from the RLMChatCompletion usage + cost_ledger.

    T7/M-BUDGET — honest cost reporting: ``rlm`` does not tag usage by tier
    (root vs sub-call), so the former ``root``/``sub`` split was always ``sub=0``
    and misleading.  We now report a single honest ``llm_usd`` total drawn from
    ``result.usage_summary`` (all rlm-tracked LLM spend, root + sub combined)
    plus ``primitives_usd`` from the cost ledger (primitive-internal LLM calls).

    When per-primitive usage tracking arrives in a future seam, ``primitives``
    will carry real values; the ``llm_usd`` field is always the authoritative
    total.
    """
    # All rlm-tracked LLM spend (root + sub-calls combined — rlm does not separate them).
    rlm_usd: float = 0.0
    if result.usage_summary is not None:
        for _model_key, summary in result.usage_summary.model_usage_summaries.items():
            rlm_usd += summary.total_cost or 0.0

    # Primitive-internal LLM cost from the ledger.
    primitives_usd: float = 0.0
    if ctx.cost_ledger is not None:
        primitives_usd = ctx.cost_ledger.total_usd()

    total = round(rlm_usd + primitives_usd, 8)
    return {
        "llm_usd": total,
        "primitives": round(primitives_usd, 8),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def reconcile_verdict_with_score(verdict: str, overall_score: float) -> str:
    """Cap a self-reported verdict at what the authoritative rubric score supports.

    Symptom this guards against: the `ftrl` run self-reported verdict
    "reproduced" while its post-run leaf score was 0.000 — an impossible
    combination that makes the benchmark leaderboard dishonest.

    Evidence ceiling derived from ``overall_score``:
      - >= _VERDICT_REPRODUCED_MIN_SCORE (0.60) → ceiling "reproduced"
      - >= _VERDICT_PARTIAL_MIN_SCORE    (0.15) → ceiling "partial"
      - else                                    → ceiling "failed"

    The function NEVER upgrades a verdict — it only downgrades.  If the
    incoming ``verdict``'s rank exceeds the ceiling's rank, the ceiling is
    returned; otherwise the original ``verdict`` is returned unchanged.

    An unrecognised ``verdict`` string is treated as rank ``partial``
    (consistent with ``_reconcile_verdict``, which also downgrades unknowns
    to ``partial``).

    Args:
        verdict: The self-reported verdict string from the final report.
        overall_score: The authoritative post-run rubric leaf score (0.0–1.0).

    Returns:
        The reconciled verdict string (one of "reproduced", "partial", "failed").
    """
    # Determine the evidence ceiling
    if overall_score >= _VERDICT_REPRODUCED_MIN_SCORE:
        ceiling = "reproduced"
    elif overall_score >= _VERDICT_PARTIAL_MIN_SCORE:
        ceiling = "partial"
    else:
        ceiling = "failed"

    # Treat unrecognised verdicts as "partial" (matches _reconcile_verdict logic)
    verdict_rank = _VERDICT_RANK.get(verdict, _VERDICT_RANK["partial"])
    ceiling_rank = _VERDICT_RANK[ceiling]

    # Only downgrade — never upgrade
    if verdict_rank > ceiling_rank:
        return ceiling
    return verdict


def _authoritative_primitive_trace(ctx: RunContext) -> dict[str, Any]:
    """Count primitive invocations from the cost ledger — the authoritative record.

    ``binding.wrap_primitive`` appends a ledger row on *every* primitive call,
    so the ledger reflects what actually ran. The root model also self-reports
    a ``primitive_trace`` in its report JSON, but that has been observed to
    undercount and to omit calls entirely — so the report uses this instead.
    """
    by_primitive: dict[str, int] = {}
    for entry in ctx.cost_ledger.entries:
        by_primitive[entry.agent_id] = by_primitive.get(entry.agent_id, 0) + 1
    return {"calls": sum(by_primitive.values()), "by_primitive": by_primitive}


def _best_recorded_rubric_score(project_dir: "Path") -> "float | None":
    """Max rubric ``overall_score`` recorded in this run's dashboard_events.jsonl.

    The run emits a ``rubric_score`` event per ``verify_against_rubric`` call, so
    the max is the run's true high-water mark. Reading from disk makes the
    best-of-run floor in ``build_final_report`` salvage-capable: re-running the
    builder on an already-degraded run dir recovers the best score. Returns None
    when no rubric score was ever recorded.
    """
    events = Path(project_dir) / "dashboard_events.jsonl"
    if not events.exists():
        return None
    best: float | None = None
    try:
        for line in events.read_text(encoding="utf-8").splitlines():
            if "rubric_score" not in line:
                continue
            try:
                d = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            payload = d.get("payload") if isinstance(d.get("payload"), dict) else d
            s = payload.get("overall_score")
            if s is None:
                s = payload.get("score")
            try:
                val = float(s)
            except (TypeError, ValueError):
                continue
            if best is None or val > best:
                best = val
    except OSError:
        return None
    return best


def _terminal_stop_reason_from_disk(project_dir: Path) -> dict | None:
    """Recover the last terminal ``stop_reason`` from ``experiment_runs.jsonl``.

    Fail-soft fallback for when ``ctx._terminal_stop_reason`` is unavailable (e.g.
    re-running the builder on a finalized run). Scans in order so the LAST recorded
    terminal stop wins; returns None on any error or when none was recorded.
    """
    path = Path(project_dir) / "experiment_runs.jsonl"
    if not path.is_file():
        return None
    found: dict | None = None
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "stop_reason" not in line:
                continue
            try:
                entry = json.loads(line)
            except (ValueError, TypeError):
                continue
            for candidate in (entry, entry.get("result") if isinstance(entry, dict) else None):
                if isinstance(candidate, dict) and isinstance(candidate.get("stop_reason"), dict):
                    if candidate["stop_reason"].get("kind"):
                        found = candidate["stop_reason"]
    except OSError:
        return None
    return found


def build_final_report(
    result: RLMChatCompletion,
    *,
    ctx: RunContext,
    root_model: Any = None,
) -> RLMFinalReport:
    """Convert an `RLMChatCompletion` into a validated `RLMFinalReport`.

    Parsing strategy (spec §11, Codex C1 resolution):
    1. `json.loads` on `result.response` (the root did `json.dumps`).
    2. `ast.literal_eval` fallback (recovers a repr-stringified dict).
    3. If both fail, return a `failed` report carrying `raw` in
       `reproduction_summary` — never crash.

    Cost reconciliation: sums `result.usage_summary` (root + sub LLM) and
    `ctx.cost_ledger` entries (primitive-internal LLM).

    Args:
        result: The completed RLM run result.
        ctx: Run-scoped context supplying project metadata and cost ledger.
        root_model: Optional root-model metadata (any type; passed as metadata
            only and never used for required logic — avoids a cross-module dep).

    Returns:
        A fully-validated `RLMFinalReport`; every field has an honest default.
    """
    raw = result.response or ""

    parsed = _parse_response(raw)

    if parsed is None:
        logger.error(
            "report: could not parse RLM response (len=%d); producing failed report",
            len(raw),
        )
        _excluded = {"verdict", "reproduction_summary", "cost", "iterations"}
        return RLMFinalReport(
            verdict="failed",
            reproduction_summary=f"[unparseable response] {raw[:1000]}",
            **{k: v for k, v in _HONEST_DEFAULTS.items() if k not in _excluded},
            cost=_cost_dict(result, ctx),
            iterations=_safe_int(result.metadata),
        )

    # Build kwargs, falling back to honest defaults for any missing field
    verdict = _reconcile_verdict(parsed)

    # The root assembles the report JSON itself, so its `primitive_trace` and
    # `baseline_metrics` are self-attested. Replace the trace with the
    # authoritative ledger count, then enforce the honesty invariant: a result
    # section must be backed by the primitive that produces it.
    trace = _authoritative_primitive_trace(ctx)
    baseline_metrics = parsed.get("baseline_metrics") or {}
    summary = str(parsed.get("reproduction_summary") or "")
    if baseline_metrics and not trace["by_primitive"].get("run_experiment"):
        # The root reported metrics it never measured — run_experiment never
        # ran. Drop them and say so; downgrade an over-claimed verdict.
        logger.warning(
            "report: dropping %d unbacked baseline metric(s) — run_experiment "
            "never ran (root-fabricated)",
            len(baseline_metrics),
        )
        baseline_metrics = {}
        summary = (
            summary
            + "\n\n[honesty guard] Baseline metrics were dropped: the "
            "run_experiment primitive never ran, so no metrics were measured."
        ).strip()
        if verdict == "reproduced":
            verdict = "partial"

    # NEW: evidence-based verdict reconciliation (T6 / P0-I9).
    verdict, downgrade_reason = _reconcile_verdict_against_evidence(
        verdict,
        baseline_metrics=baseline_metrics,
        rubric=parsed.get("rubric") or {},
        primitive_trace=trace,
    )
    if downgrade_reason:
        summary = (
            summary
            + f"\n\n[verdict guard] Downgraded to 'partial': {downgrade_reason}."
        ).strip()
        logger.warning(
            "report: verdict downgraded to partial — %s", downgrade_reason,
        )

    # PR B: cross-check scope.ran against experiment_runs.jsonl evidence.
    # The root attests scope.ran itself; this enforces the claim against the
    # primitive trace. Unverified items move to scope.gaps; if any unverified
    # remain AND verdict is "reproduced", downgrade to "partial".
    raw_scope = parsed.get("scope") or {"requested": "", "ran": [], "gaps": []}
    verified_scope, scope_downgrade_reason = _verify_scope_evidence(
        raw_scope, ctx.project_dir
    )
    if scope_downgrade_reason:
        summary = (
            summary
            + f"\n\n[scope guard] {scope_downgrade_reason}."
        ).strip()
        logger.warning("report: scope-evidence cross-check — %s", scope_downgrade_reason)
        if verdict == "reproduced":
            verdict = "partial"

    # Surface datasets the agent recorded as unobtainable as explicit scope.gaps,
    # even if the root never declared them — they were EXCLUDED from the rubric
    # score (data-unavailable-aware grader), so the report must say so plainly.
    # Pass the operator's skip_models so we can distinguish intentional scope
    # reductions from model load failures the agent silently laundered into
    # scope.models_skipped (a code bug that must NOT be auto-excluded).
    _op_skip = list(getattr(getattr(ctx, "scope_spec", None), "skip_models", None) or [])
    try:
        verified_scope = _merge_data_unavailable_gaps(
            verified_scope, ctx.project_dir, operator_skip_models=_op_skip
        )
    except Exception:  # noqa: BLE001 — gap surfacing augments the report, never fatal
        logger.exception("report: data-unavailable gap merge failed")

    kwargs: dict[str, Any] = {
        "verdict": verdict,
        "paper": parsed.get("paper") or {},
        "reproduction_summary": summary,
        "baseline_metrics": baseline_metrics,
        "paper_claims": parsed.get("paper_claims") or {},
        "rubric": parsed.get("rubric") or {
            "overall_score": None,
            "meets_target": None,
            "target_score": None,
            "degraded": None,
            "areas": [],
        },
        "improvements": list(parsed.get("improvements") or []),
        "scope": verified_scope,
        "primitive_trace": trace,
        "cost": _cost_dict(result, ctx),
        "iterations": _safe_int(parsed.get("iterations") or (result.metadata or {}).get("iterations")),
    }

    # comp 4b (2026-05-31): surface a terminal capacity/OOM stop so the report
    # explains WHY the run ended early. Prefer the in-process signal stashed on
    # ctx by run.py's tool wrapper; fall back to experiment_runs.jsonl so
    # re-running this builder on a finalized run still recovers it.
    _stop = getattr(ctx, "_terminal_stop_reason", None)
    if not (isinstance(_stop, dict) and _stop.get("kind")):
        _stop = _terminal_stop_reason_from_disk(ctx.project_dir)
    if isinstance(_stop, dict) and _stop.get("kind"):
        kwargs["stop_reason"] = _stop

    # Best-of-run floor (2026-05-30): the RLM loop can over-iterate into a degraded
    # final state and self-report a LOWER rubric than it actually achieved — the
    # SDAR run reported 0.0 despite scoring ~0.18 mid-run. Surface the best
    # overall_score the run recorded so a late regression can never bury a real
    # result. Reads dashboard_events.jsonl, so re-running this builder also
    # salvages an already-finalized degraded run.
    _best = _best_recorded_rubric_score(ctx.project_dir)
    if _best is not None:
        _r = dict(kwargs.get("rubric") or {})
        _cur = _r.get("overall_score")
        try:
            _cur_f = float(_cur) if _cur is not None else None
        except (TypeError, ValueError):
            _cur_f = None
        if _cur_f is None or _best > _cur_f:
            _r["overall_score"] = _best
            _r["best_of_run"] = True
            kwargs["rubric"] = _r

    return RLMFinalReport(**kwargs)


def _safe_int(value: Any) -> int:
    """Coerce value to int silently; return 0 on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def write_final_report_rlm(
    report: RLMFinalReport,
    project_dir: Path,
) -> tuple[Path, Path]:
    """Write `final_report.json` and `final_report.md` atomically.

    Both files are written via a temp-file + `os.replace` to avoid partial
    writes on crash or timeout. On non-failed verdicts (i.e. the run
    produced a real reproduction, partial or full), also stamp a
    ``.preserved`` marker file in ``project_dir`` — a tiny JSON manifest
    flagging this run as worth keeping. The marker is a GC signal: any
    cleanup script that prunes ``runs/`` MUST skip directories carrying
    this file. It survives process kill (it is on disk) and is written
    after the canonical artifacts, so a half-written final_report.json
    cannot leave behind a stale "preserved" claim.

    Args:
        report: The completed final report.
        project_dir: Run project directory (e.g. `runs/<project_id>/`).

    Returns:
        ``(json_path, md_path)`` — the paths of the written files.
    """
    project_dir.mkdir(parents=True, exist_ok=True)

    json_path = project_dir / "final_report.json"
    md_path = project_dir / "final_report.md"

    # --- Merge deep rubric evaluation (2026-05-26) -------------------------
    # binding.py persists the full verify_against_rubric payload to
    # rubric_evaluation.json on every successful verification. Merge the
    # per-leaf justifications + weak_leaves into the report's rubric block
    # so final_report.json carries the WHY-this-score evidence instead of
    # just the rolled-up area numbers. Idempotent — if the eval file doesn't
    # exist (failed run, never reached verify), the existing rubric block is
    # preserved untouched.
    try:
        eval_path = project_dir / "rubric_evaluation.json"
        if eval_path.exists():
            import json as _json
            deep = _json.loads(eval_path.read_text())
            current = dict(report.rubric or {})
            for key in ("leaf_scores", "weak_leaves", "leaf_count", "graded",
                        "rubric_source", "coverage_pct", "compute_adjusted_score",
                        "compute_scope"):
                if key in deep and deep[key] is not None and key not in current:
                    current[key] = deep[key]
            report.rubric = current
    except Exception:  # noqa: BLE001 — merge is best-effort, never crashes the write
        logger.exception("report: rubric_evaluation.json merge failed (non-fatal)")

    # --- JSON (canonical) ---
    json_content = report.model_dump_json(indent=2)
    _atomic_write(json_path, json_content)

    # --- Markdown (human-readable) ---
    # PR-ν.2: pass project_dir so the renderer can embed token usage + per-step
    # timing read from the run's own telemetry sidecars (tokens_total.json,
    # timing.json). The renderer fails soft on any missing/corrupt sidecar.
    md_content = _render_markdown(report, project_dir=project_dir)
    _atomic_write(md_path, md_content)

    # --- Preservation marker --------------------------------------------
    # Stamp a tiny manifest so future cleanup / GC scripts know this run
    # produced real reproduction artifacts and must not be pruned. We
    # stamp on every non-failed verdict — partial reproductions are still
    # valuable, and a future operator can grep ``.preserved`` files to
    # rebuild the leaderboard if other state is ever lost.
    try:
        if report.verdict != "failed":
            _write_preserved_marker(project_dir, report)
    except Exception:  # noqa: BLE001 — marker is best-effort; never crash the run
        logger.exception("report: failed to write .preserved marker (non-fatal)")

    # --- Token aggregate (PR-α.5) -----------------------------------------------
    # Write tokens_total.json alongside the final report so the 3-source
    # estimator (PR-ε) can read per-run token distributions without re-parsing
    # cost_ledger.jsonl on every calibration pass.
    try:
        write_tokens_total(project_dir)
    except Exception:  # noqa: BLE001 — aggregate is best-effort; never crash the run
        logger.warning("report: tokens_total.json write failed (non-fatal)")

    # --- Timing capture (PR-ε.1) -------------------------------------------------
    # Write timing.json so the k-NN estimator has empirical wall-clock data
    # for future estimates.  Only written on non-failed verdicts (same gate
    # as .preserved); always runs after the .preserved marker so backfill
    # can detect the pair.
    if report.verdict != "failed":
        try:
            from backend.services.pricing.timing import write_timing_json
            write_timing_json(project_dir)
        except Exception:  # noqa: BLE001 — timing capture is best-effort
            logger.warning("report: timing.json write failed (non-fatal)")

    # C4 — Auto-recompute calibration priors at every finalize (best-effort,
    # non-fatal).  Runs unconditionally on non-failed verdicts so each completed
    # run tightens token estimates for the next one.  The env-var opt-out
    # REPROLAB_UPDATE_CALIBRATION=false disables the auto-update (useful in
    # CI or when a single smoke run must not overwrite the historical corpus).
    _update_cal = os.environ.get("REPROLAB_UPDATE_CALIBRATION", "").lower()
    _cal_enabled = _update_cal not in {"false", "0", "no", "off"}
    if report.verdict != "failed" and _cal_enabled:
        try:
            from backend.services.pricing.calibration import recompute_calibration
            runs_root = project_dir.parent
            recompute_calibration(runs_root)
        except Exception:  # noqa: BLE001 — calibration update is non-critical
            logger.warning("report: calibration recompute failed (non-fatal)")

    logger.info(
        "report: wrote final_report.{json,md} to %s (verdict=%s)",
        project_dir,
        report.verdict,
    )
    return json_path, md_path


def _write_preserved_marker(project_dir: Path, report: RLMFinalReport) -> None:
    """Write the ``.preserved`` GC-protection marker.

    The marker is a small JSON manifest, intentionally smaller than the
    final report so it stays cheap to scan. Schema is forward-compatible
    (extra keys ignored by readers); the canonical fields are verdict,
    rubric_overall_score, paper_id, paper_title, preserved_at_utc.
    """
    rubric = getattr(report, "rubric", None)
    overall_score = None
    if isinstance(rubric, dict):
        overall_score = rubric.get("overall_score")
    paper = getattr(report, "paper", None) or {}
    paper_id = paper.get("id") if isinstance(paper, dict) else None
    paper_title = paper.get("title") if isinstance(paper, dict) else None
    manifest = {
        "verdict": report.verdict,
        "rubric_overall_score": overall_score,
        "paper_id": paper_id,
        "paper_title": paper_title,
        "iterations": getattr(report, "iterations", None),
        "preserved_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
    }
    _atomic_write(
        project_dir / ".preserved",
        json.dumps(manifest, indent=2, sort_keys=True),
    )


def _atomic_write(path: Path, content: str) -> None:
    """Write `content` to `path` atomically via a sibling temp file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _aggregate_tokens_total(project_dir: Path) -> dict:
    """Aggregate cost_ledger.jsonl entries into the tokens_total summary schema.

    Returns a dict with the shape documented in PR-α.5 spec:
      - schema_version: 1
      - by_primitive: {name: {input_tokens, output_tokens, calls}}
      - by_model: {model: {input_tokens, output_tokens}}
      - grand_total: {input_tokens, output_tokens, cache_read_input_tokens,
                      cache_creation_input_tokens, calls}
      - computed_at_utc: ISO timestamp
    """
    ledger_path = project_dir / "cost_ledger.jsonl"
    by_primitive: dict[str, dict] = {}
    by_model: dict[str, dict] = {}
    grand_total_input = 0
    grand_total_output = 0
    grand_total_cache_read = 0
    grand_total_cache_creation = 0
    grand_total_calls = 0

    if ledger_path.exists():
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Support both field names: the canonical field and the alias written by to_json()
            primitive = str(row.get("primitive") or row.get("agent_id") or "unknown")
            model = str(row.get("model") or "unknown")
            input_t = int(row.get("input_tokens") or row.get("tokens_in") or 0)
            output_t = int(row.get("output_tokens") or row.get("tokens_out") or 0)
            cache_read = int(row.get("cache_read_input_tokens") or 0)
            cache_creation = int(row.get("cache_creation_input_tokens") or 0)

            # by_primitive
            prim_rec = by_primitive.setdefault(
                primitive, {"input_tokens": 0, "output_tokens": 0, "calls": 0}
            )
            prim_rec["input_tokens"] += input_t
            prim_rec["output_tokens"] += output_t
            prim_rec["calls"] += 1

            # by_model
            model_rec = by_model.setdefault(
                model, {"input_tokens": 0, "output_tokens": 0}
            )
            model_rec["input_tokens"] += input_t
            model_rec["output_tokens"] += output_t

            # grand_total
            grand_total_input += input_t
            grand_total_output += output_t
            grand_total_cache_read += cache_read
            grand_total_cache_creation += cache_creation
            grand_total_calls += 1

    return {
        "schema_version": 1,
        "by_primitive": by_primitive,
        "by_model": by_model,
        "grand_total": {
            "input_tokens": grand_total_input,
            "output_tokens": grand_total_output,
            "cache_read_input_tokens": grand_total_cache_read,
            "cache_creation_input_tokens": grand_total_cache_creation,
            "calls": grand_total_calls,
        },
        "computed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_tokens_total(project_dir: Path) -> Path:
    """Aggregate cost_ledger.jsonl and write runs/<id>/tokens_total.json.

    Written atomically via tmp + os.replace. Safe to call concurrently with
    cost_ledger appends — reads the file at a point in time; concurrent
    writers may add rows after this read and before the next calibration
    pass (that is acceptable; the file is regenerated at end-of-run).

    Returns the path written.
    """
    path = project_dir / "tokens_total.json"
    data = _aggregate_tokens_total(project_dir)
    _atomic_write(path, json.dumps(data, indent=2, sort_keys=True))
    return path


def _render_markdown(report: RLMFinalReport, project_dir: Path | None = None) -> str:
    """Render a human-readable Markdown report.

    Structure:
    - Verdict banner (prominent)
    - Rubric score (+ weak leaves when available)
    - Reproduction summary
    - Baseline metrics vs. paper claims
    - Improvement candidates and outcomes
    - Cost
    - Token usage (PR-ν.2 — read from ``tokens_total.json`` if present)
    - Per-step timing (PR-ν.2 — read from ``timing.json`` if present)

    ``project_dir`` is optional for backwards compat — when None, the token /
    timing sections are silently skipped (no file lookups). Callers that own
    the run directory (``write_final_report_rlm``) should pass it.
    """
    lines: list[str] = []

    # --- Header ---
    verdict_label = {
        "reproduced": "REPRODUCED",
        "partial": "PARTIAL REPRODUCTION",
        "failed": "REPRODUCTION FAILED",
    }.get(report.verdict, report.verdict.upper())

    paper_title = report.paper.get("title", "Unknown Paper")
    paper_id = report.paper.get("id", "")

    lines.append(f"# {verdict_label}")
    lines.append("")
    if paper_title or paper_id:
        lines.append(f"**Paper:** {paper_title}" + (f" (`{paper_id}`)" if paper_id else ""))
        lines.append("")

    # --- Rubric ---
    # C2c (second pass): a run that was never scored carries
    # overall_score=None / meets_target=None — render as "not scored", never as
    # a fabricated 0.000 / "below target". This is the markdown counterpart of
    # the honest-null defaults in RLMFinalReport.rubric.
    rubric = report.rubric
    overall = rubric.get("overall_score")
    meets_target = rubric.get("meets_target")
    lines.append("## Rubric Score")
    lines.append("")
    if overall is None:
        lines.append("**Overall score:** not scored  (run did not reach the leaf scorer)")
    else:
        if meets_target is True:
            target_flag = "✔ meets target"
        elif meets_target is False:
            target_flag = "✘ below target"
        else:  # None — target unknown (rubric had no target_score)
            target_flag = "no target set"
        lines.append(f"**Overall score:** {overall:.3f}  ({target_flag})")
    # Provenance: after the post-run leaf scorer amends the report, surface the
    # rubric source + leaf coverage so a "generated" score is never mistaken for
    # a PaperBench-official one.
    source = rubric.get("rubric_source")
    leaf_count = rubric.get("leaf_count")
    if source or leaf_count:
        bits: list[str] = []
        if leaf_count:
            bits.append(f"{rubric.get('graded', 0)}/{leaf_count} rubric leaves graded")
        if source == "generated":
            bits.append("self-generated rubric — not PaperBench-official")
        elif source == "paperbench_bundle":
            bits.append("PaperBench bundle rubric")
        elif source:
            bits.append(f"rubric source: {source}")
        lines.append("")
        lines.append(f"_{' · '.join(bits)}_")
    lines.append("")

    areas = rubric.get("areas") or []
    if areas:
        lines.append("| Area | Score | Notes |")
        lines.append("|---|---|---|")
        for area in areas:
            # Areas can use either 'name' (legacy) or 'area' (current _rubric_areas
            # output) as the label — accept both so a backfilled report renders
            # cleanly with the canonical field.
            name = area.get("name") or area.get("area") or "—"
            score = area.get("score")
            score_str = f"{score:.3f}" if isinstance(score, (int, float)) else str(score or "—")
            notes = area.get("notes", "")
            lines.append(f"| {name} | {score_str} | {notes} |")
        lines.append("")

    # --- Weakest rubric leaves (PR-ν.2) -----------------------------------
    # Top lowest-scoring graded leaves with their justifications — the
    # actionable "where did points come off?" surface for the operator.
    # Pre-populated by verify_against_rubric (PR-κ). Skipped data-unavailable
    # leaves are already filtered out by the grader.
    weak = rubric.get("weak_leaves") or []
    weak = [w for w in weak if isinstance(w, dict) and w.get("score") is not None]
    if weak:
        lines.append("### Weakest rubric leaves")
        lines.append("")
        lines.append("| Score | Justification |")
        lines.append("|---|---|")
        for leaf in weak[:5]:
            score = leaf.get("score", "—")
            score_str = f"{score:.2f}" if isinstance(score, (int, float)) else str(score)
            just = (leaf.get("justification") or "").replace("|", "\\|")
            just = " ".join(just.split())  # collapse whitespace
            if len(just) > 220:
                just = just[:217] + "…"
            lines.append(f"| {score_str} | {just} |")
        lines.append("")

    # --- Reproduction summary ---
    lines.append("## Reproduction Summary")
    lines.append("")
    summary = report.reproduction_summary.strip()
    lines.append(summary if summary else "_No summary provided._")
    lines.append("")

    # --- Scope ---
    # Only render when the root populated at least one of requested/ran/gaps.
    # Default-empty scope is suppressed to keep older reports clean.
    scope = report.scope or {}
    requested = str(scope.get("requested") or "").strip()
    ran = list(scope.get("ran") or [])
    gaps = list(scope.get("gaps") or [])
    if requested or ran or gaps:
        lines.append("## Scope")
        lines.append("")
        if requested:
            lines.append(f"**Requested:** {requested}")
            lines.append("")
        if ran:
            lines.append("**Ran:**")
            for item in ran:
                lines.append(f"- {item}")
            lines.append("")
        if gaps:
            lines.append("**Gaps:** _(items requested but not reproduced; datasets marked "
                         "\"unobtainable\" were excluded from the rubric score, not penalised)_")
            for item in gaps:
                lines.append(f"- {item}")
            lines.append("")

    # --- Baseline metrics vs. paper claims ---
    lines.append("## Baseline Metrics vs. Paper Claims")
    lines.append("")
    if report.baseline_metrics or report.paper_claims:
        all_keys = sorted(set(report.baseline_metrics) | set(report.paper_claims))
        lines.append("| Metric | Reproduced | Paper Claim |")
        lines.append("|---|---|---|")
        for key in all_keys:
            repro_val = report.baseline_metrics.get(key, "—")
            claim_val = report.paper_claims.get(key, "—")
            lines.append(f"| {key} | {repro_val} | {claim_val} |")
        lines.append("")
    else:
        lines.append("_No metrics recorded. (Phase 5 will populate this table.)_")
        lines.append("")

    # --- Improvements ---
    lines.append("## Improvement Candidates")
    lines.append("")
    if report.improvements:
        for i, imp in enumerate(report.improvements, 1):
            name = imp.get("name") or imp.get("tag") or f"Candidate {i}"
            outcome = imp.get("outcome") or imp.get("status") or "pending"
            delta = imp.get("delta") or imp.get("rubric_delta") or ""
            delta_str = f" ({delta})" if delta else ""
            lines.append(f"**{i}. {name}** — {outcome}{delta_str}")
            description = imp.get("description") or ""
            if description:
                lines.append(f"> {description}")
        lines.append("")
    else:
        lines.append("_No improvement candidates recorded._")
        lines.append("")

    # --- Cost ---
    lines.append("## Cost")
    lines.append("")
    cost = report.cost
    lines.append(f"| Category | USD |")
    lines.append(f"|---|---|")
    lines.append(f"| Primitive-internal LLM | ${cost.get('primitives', 0.0):.6f} |")
    lines.append(f"| **Total LLM** | **${cost.get('llm_usd', 0.0):.6f}** |")
    lines.append("")
    lines.append(f"**Iterations:** {report.iterations}")
    lines.append("")

    # --- Token usage + per-step timing (PR-ν.2) ----------------------------
    # Pull the sidecars written alongside this report. Both are best-effort:
    # if either is missing or corrupt, the section is silently skipped.
    if project_dir is not None:
        _append_token_usage_section(lines, project_dir)
        _append_timing_section(lines, project_dir)

    # --- Footer ---
    lines.append("---")
    lines.append("_Generated by ReproLab RLM orchestrator (Issue #60)._")

    return "\n".join(lines) + "\n"


def _append_token_usage_section(lines: list[str], project_dir: Path) -> None:
    """Append the Token Usage section reading from ``tokens_total.json``.

    Silent no-op on missing or malformed sidecar — every code path here is
    best-effort instrumentation, never load-bearing on the report being
    written.
    """
    tokens_path = project_dir / "tokens_total.json"
    if not tokens_path.exists():
        return
    try:
        tokens = json.loads(tokens_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — see docstring
        return
    if not isinstance(tokens, dict):
        return
    grand = tokens.get("grand_total") or {}
    if not isinstance(grand, dict):
        return

    lines.append("## Token Usage")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Total LLM calls | {int(grand.get('calls', 0))} |")
    lines.append(f"| Input tokens | {int(grand.get('input_tokens', 0)):,} |")
    lines.append(f"| Output tokens | {int(grand.get('output_tokens', 0)):,} |")
    cache_create = int(grand.get("cache_creation_input_tokens", 0) or 0)
    cache_read = int(grand.get("cache_read_input_tokens", 0) or 0)
    if cache_create or cache_read:
        lines.append(f"| Cache creation (input) | {cache_create:,} |")
        lines.append(f"| Cache read (input, prompt-cache-billed) | {cache_read:,} |")
    lines.append("")

    by_primitive = tokens.get("by_primitive") or {}
    if isinstance(by_primitive, dict) and by_primitive:
        rows = sorted(
            ((name, stats) for name, stats in by_primitive.items() if isinstance(stats, dict)),
            key=lambda kv: -int(kv[1].get("output_tokens", 0) or 0),
        )
        # Suppress all-zero rows (heartbeat-style primitives that aren't LLM-billed).
        rows = [(n, s) for n, s in rows if any((s.get("input_tokens"), s.get("output_tokens")))]
        if rows:
            lines.append("### Per-primitive token usage")
            lines.append("")
            lines.append("| Primitive | Calls | Input | Output |")
            lines.append("|---|---|---|---|")
            for name, stats in rows:
                lines.append(
                    f"| {name} | {int(stats.get('calls', 0))} | "
                    f"{int(stats.get('input_tokens', 0) or 0):,} | "
                    f"{int(stats.get('output_tokens', 0) or 0):,} |"
                )
            lines.append("")


def _append_timing_section(lines: list[str], project_dir: Path) -> None:
    """Append the Per-Step Timing section reading from ``timing.json``.

    Same fail-soft policy as the token section.
    """
    timing_path = project_dir / "timing.json"
    if not timing_path.exists():
        return
    try:
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return
    if not isinstance(timing, dict):
        return

    total = timing.get("wall_clock_s")
    primitive_ts = timing.get("primitive_wall_clock_s") or {}
    counts = timing.get("primitive_call_counts") or {}
    gpu_hours = timing.get("gpu_hours")
    gpu_type = timing.get("gpu_type")
    gpu_count = timing.get("gpu_count")

    if total is None and not primitive_ts and gpu_hours is None:
        return

    lines.append("## Per-Step Timing")
    lines.append("")
    if isinstance(total, (int, float)) and total > 0:
        mm, ss = divmod(int(total), 60)
        hh, mm = divmod(mm, 60)
        lines.append(f"**Total wall clock:** {total:.1f}s ({hh}h {mm}m {ss}s)")
        lines.append("")

    if isinstance(primitive_ts, dict) and primitive_ts:
        rows = sorted(
            ((name, float(s)) for name, s in primitive_ts.items() if isinstance(s, (int, float))),
            key=lambda kv: -kv[1],
        )
        rows = [(n, s) for n, s in rows if s > 0.01]
        if rows:
            lines.append("| Primitive | Calls | Total time (s) |")
            lines.append("|---|---|---|")
            for name, secs in rows:
                n_calls = int(counts.get(name, 1)) if isinstance(counts, dict) else 1
                lines.append(f"| {name} | {n_calls} | {secs:.2f} |")
            lines.append("")

    if isinstance(gpu_hours, (int, float)) and gpu_hours > 0:
        lines.append(
            f"**GPU hours:** {gpu_hours:.3f}h on `{gpu_type or '?'}` × {int(gpu_count or 1)}"
        )
        lines.append("")
