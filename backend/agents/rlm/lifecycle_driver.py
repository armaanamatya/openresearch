"""Harness-driven lifecycle chain for degenerate-root recovery.

When the RLM root degenerates (calls FINAL_VAR repeatedly without doing
reproduction work), this driver executes the mandatory lifecycle primitives
directly, starting at whichever stage the root stalled at.

Design principles:
- Pure-ish: no globals, no env-var reads, no I/O outside the supplied tools.
- Fail-soft: every step is wrapped; no exception ever propagates out.
- Wall-clock aware: stops before each expensive step when remaining budget
  is below ``min_remaining_s``.
- Observability: emits a ``lifecycle_drive_step`` event before each step.

The ``tools`` dict has the shape ``{name: {"tool": callable, "description": str}}``.
Callables re-supply ``ctx`` internally (they are closed-over by
``binding.build_custom_tools``) — this module calls them WITHOUT ctx.
"""

from __future__ import annotations

__all__ = ["drive_lifecycle_chain", "run_lifecycle_primary"]

import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_emit(emit: Any, event: dict) -> None:
    """Fire the emit callable, swallowing any exception."""
    try:
        emit(event)
    except Exception:  # noqa: BLE001
        pass


def _get_tool(tools: dict, name: str):
    """Return the callable for *name*, or None if missing/None."""
    entry = tools.get(name)
    if entry is None:
        return None
    return entry.get("tool")


def _wallclock_ok(ctx: Any, min_remaining_s: float) -> bool:
    """True if there is enough wall-clock budget left (or no budget at all)."""
    try:
        rem = ctx.remaining_s()
    except Exception:  # noqa: BLE001
        return True  # fail-open: no budget info → allow
    if rem is None:
        return True
    return rem >= min_remaining_s


def _call(tool_fn, *args) -> dict:
    """Call *tool_fn* with positional *args*, always returning a dict."""
    result = tool_fn(*args)
    if not isinstance(result, dict):
        return {"_raw": result}
    return result


def _is_repairable(result: dict) -> bool:
    """True when run_experiment signals a repairable outcome (canonical label)."""
    return isinstance(result, dict) and result.get("outcome") == "repairable"


def _is_fatal(result: dict) -> bool:
    """True when a primitive result signals an unrecoverable fatal outcome."""
    return isinstance(result, dict) and result.get("outcome") == "fatal"


def _has_gradeable_evidence(result: dict) -> bool:
    """True when a run_experiment result contains metrics worth grading.

    A non-success result from a partial grid (some cells ok, others OOM'd or
    failed) still carries real per-model metrics that deserve a rubric grade —
    the run should ship a real score rather than ``None``.

    Conservative: any non-empty ``metrics`` dict is considered gradeable.
    If the dict contains a ``per_model`` sub-dict, at least one entry with
    ``status == "ok"`` is required for that stricter check; plain flat-scalar
    metrics are assumed gradeable (better to over-verify than under-verify).
    """
    if not isinstance(result, dict):
        return False
    metrics = result.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        return False
    per_model = metrics.get("per_model")
    if isinstance(per_model, dict) and per_model:
        # Has structured per_model — require at least one cell with status==ok.
        for model_data in per_model.values():
            if not isinstance(model_data, dict):
                continue
            # per_model[model] may be flat or nested under per_dataset/baseline.
            if model_data.get("status") == "ok":
                return True
            for env_data in model_data.values():
                if not isinstance(env_data, dict):
                    continue
                if env_data.get("status") == "ok":
                    return True
                for leaf in env_data.values():
                    if isinstance(leaf, dict) and leaf.get("status") == "ok":
                        return True
        # per_model present but no ok cell found — still grade on non-empty metrics
        # so a partial but non-empty aggregate doesn't silently produce None score.
        return True
    # Flat scalar metrics (non-SDAR papers) — any non-empty dict is gradeable.
    return True


def _is_explicit_failure(result: dict) -> bool:
    """Return True ONLY when the result has an explicit ``ok=False`` key.

    - A dict WITHOUT an ``ok`` key is treated as success (run_experiment,
      understand_section, detect_environment, plan_reproduction, and
      verify_against_rubric all return plain dicts with no ``ok``).
    - implement_baseline returns ``{"ok": True, "code_path": ...}`` on
      success or ``{"ok": False, "error": ...}`` on failure.
    """
    if "ok" not in result:
        return False
    return not result["ok"]


def _evidence_fingerprint(result: dict, ctx: Any) -> str:
    """Fingerprint the CURRENT result's evidence for repair-progress tracking.

    The primary signal is ``result.get("metrics")`` — the in-memory evidence
    THIS attempt actually produced — hashed via the same canonical
    ``external_validator.evidence_fingerprint`` helper the validator panel and
    ``report.py``'s finalize chokepoint already use, so a genuine metric
    change between repair attempts ALWAYS changes the fingerprint.

    Deliberately does NOT consult ``evidence_bundle.resolve_bundle``: that
    receipt is read from a PERSISTED on-disk file
    (``rlm_state/evidence_bundle.json``) which can be STALE — left behind by
    an earlier finalize in a resumed ``project_dir`` — so it stays constant
    across repairs even while the current attempt's metrics genuinely
    improve, which previously made two repairs with different metrics
    fingerprint IDENTICALLY and falsely tripped the "no progress" stagnation
    stop. ``ctx`` is accepted for call-site stability but is intentionally
    unused now that disk state plays no part in this fingerprint.

    Fail-soft by construction: this must never raise.
    """
    metrics = result.get("metrics") if isinstance(result, dict) else None
    try:
        from backend.agents.rlm.external_validator import evidence_fingerprint  # noqa: PLC0415 — optional, avoids import at module load

        return evidence_fingerprint(metrics)
    except Exception:  # noqa: BLE001 — fingerprinting must never break the driver
        pass
    if not isinstance(metrics, dict):
        metrics = {}
    try:
        return str(sorted(metrics.items()))
    except Exception:  # noqa: BLE001 — unorderable metric keys (rare) still fingerprint
        return str(metrics)


# ---------------------------------------------------------------------------
# Stage constants (mirrors root_progress.REQUIRED_STAGES)
# ---------------------------------------------------------------------------

_STAGE_NEED_BASELINE = "need_baseline"
_STAGE_NEED_ENVIRONMENT = "need_environment"
_STAGE_NEED_EXPERIMENT = "need_experiment"
_STAGE_NEED_VERIFICATION = "need_verification"
_STAGE_CAN_FINALIZE = "can_finalize"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def drive_lifecycle_chain(
    *,
    tools: dict,
    ctx: Any,
    paper_text: str,
    rubric_spec: dict,
    start_stage: str,
    emit: Any,
    min_remaining_s: float = 300.0,
    max_repair_iterations: int = 2,
) -> dict:
    """Execute the mandatory lifecycle chain starting at *start_stage*.

    Parameters
    ----------
    tools:
        The ``custom_tools`` dict, shape ``{name: {"tool": callable, ...}}``.
        Callables are called WITHOUT ``ctx`` (they close over it internally).
    ctx:
        A ``RunContext``-compatible object.  Must expose ``project_dir`` (Path)
        and ``remaining_s() -> float | None``.
    paper_text:
        Raw paper text supplied to understand_section when a fresh baseline is
        needed.
    rubric_spec:
        The rubric spec dict forwarded to verify_against_rubric.
    start_stage:
        One of the strings in ``root_progress.REQUIRED_STAGES``.
    emit:
        Observability callable — receives ``{"event": "lifecycle_drive_step",
        "stage": start_stage, "primitive": <name>}`` before each step.
        Exceptions from ``emit`` are swallowed.
    min_remaining_s:
        Stop before a step if ``ctx.remaining_s()`` is below this threshold.
    max_repair_iterations:
        Cap on bounded repair attempts after a repairable run_experiment.

    Returns
    -------
    dict with keys:
        ``driven``       – list of primitive names executed in order
        ``stopped_at``   – primitive name that failed/was skipped, or None
        ``stopped_reason`` – reason string, or None
        ``final_result`` – the last step's raw result dict, or None
        ``rubric_score`` – ``verify.get("overall_score")`` if verify ran, else None
        ``repaired``     – number of bounded repair attempts made (0 when none)
        ``last_run_ok``  – True when the final run_experiment was not repairable,
                           False when it was; None when run_experiment did not run
    """
    # Canonical no-op return shape.
    summary: dict = {
        "driven": [],
        "stopped_at": None,
        "stopped_reason": None,
        "final_result": None,
        "fatal_result": None,
        "rubric_score": None,
        "verify_result": None,
        "repaired": 0,
        "last_run_ok": None,
        "plan_result": None,
    }

    # "can_finalize" or any unknown stage → nothing to drive.
    if start_stage not in (
        _STAGE_NEED_BASELINE,
        _STAGE_NEED_ENVIRONMENT,
        _STAGE_NEED_EXPERIMENT,
        _STAGE_NEED_VERIFICATION,
    ):
        summary["stopped_reason"] = "already_finalizable"
        return summary

    # ------------------------------------------------------------------
    # Decide which steps to run based on start_stage.
    # ------------------------------------------------------------------
    run_understand = start_stage == _STAGE_NEED_BASELINE
    run_detect = start_stage == _STAGE_NEED_BASELINE
    run_plan = start_stage == _STAGE_NEED_BASELINE
    run_implement = start_stage == _STAGE_NEED_BASELINE
    run_experiment = start_stage in (
        _STAGE_NEED_BASELINE,
        _STAGE_NEED_ENVIRONMENT,
        _STAGE_NEED_EXPERIMENT,
    )
    run_verify = start_stage in (
        _STAGE_NEED_BASELINE,
        _STAGE_NEED_ENVIRONMENT,
        _STAGE_NEED_EXPERIMENT,
        _STAGE_NEED_VERIFICATION,
    )

    # ------------------------------------------------------------------
    # Accumulated intermediate results used to thread data between steps.
    # ------------------------------------------------------------------
    pcm: dict = {}
    env_spec: dict = {}
    contract: dict = {}
    plan: dict = {}
    try:
        code_path: str = str(ctx.project_dir / "code")
    except Exception:  # noqa: BLE001 — ctx may be a bare stub (tests) or None
        code_path = "code"
    verify_result: dict | None = None
    last_result: dict | None = None

    # ------------------------------------------------------------------
    # Helper: run one named step, handling wall-clock and fail-soft.
    # Returns (result, stop) — stop=True means the chain should halt.
    # ------------------------------------------------------------------
    def _step(name: str, *args) -> tuple[dict, bool]:
        # Wall-clock gate — checked BEFORE the tool is called.
        if not _wallclock_ok(ctx, min_remaining_s):
            summary["stopped_at"] = name
            summary["stopped_reason"] = "low_wallclock"
            return {}, True

        # Guard: missing or None callable.
        tool_fn = _get_tool(tools, name)
        if tool_fn is None:
            summary["stopped_at"] = name
            summary["stopped_reason"] = f"missing_tool:{name}"
            return {}, True

        # Advance the harness-owned iteration counter BEFORE the tool runs so
        # binding.wrap_primitive (which reads ctx.current_iteration for the
        # rubric_score event + ctx.latest_rubric_iteration policy state) records
        # real, non-zero iterations under lifecycle-primary. The root-loop logger
        # that normally advances it never runs here.
        try:
            _it = int(getattr(ctx, "current_iteration", 0) or 0) + 1
            setattr(ctx, "current_iteration", _it)
        except Exception:  # noqa: BLE001 — counter is best-effort, never blocks a step
            _it = int(getattr(ctx, "current_iteration", 0) or 0)

        _safe_emit(
            emit,
            {"event": "lifecycle_drive_step", "stage": start_stage,
             "primitive": name, "iteration": _it},
        )

        try:
            result = _call(tool_fn, *args)
        except Exception as exc:  # noqa: BLE001
            summary["stopped_at"] = name
            summary["stopped_reason"] = str(exc) or repr(exc)
            return {}, True

        summary["driven"].append(name)
        summary["final_result"] = result

        if _is_explicit_failure(result):
            summary["stopped_at"] = name
            summary["stopped_reason"] = result.get("error", "ok=False")
            return result, True

        if _is_fatal(result):
            summary["fatal_result"] = result
            summary["stopped_at"] = name
            summary["stopped_reason"] = result.get("error") or "fatal"
            return result, True

        return result, False

    # ------------------------------------------------------------------
    # Step 1: understand_section  (need_baseline only)
    # ------------------------------------------------------------------
    if run_understand:
        pcm, stop = _step("understand_section", paper_text)
        if stop:
            return summary

    # ------------------------------------------------------------------
    # Step 2: detect_environment  (need_baseline only)
    # ------------------------------------------------------------------
    if run_detect:
        env_spec, stop = _step("detect_environment", pcm)
        if stop:
            return summary

    # ------------------------------------------------------------------
    # Step 3: plan_reproduction  (need_baseline only)
    # ------------------------------------------------------------------
    if run_plan:
        contract, stop = _step("plan_reproduction", pcm, env_spec)
        if stop:
            return summary
        summary["plan_result"] = contract

    # ------------------------------------------------------------------
    # Step 4: implement_baseline  (need_baseline only), with a bounded
    # re-drive when the initial attempt itself returns a repairable result
    # (distinct from the run_experiment repair loop below — this covers a
    # repairable implement_baseline outcome, e.g. a contract-guard rejection,
    # BEFORE any run_experiment call is attempted).
    # ------------------------------------------------------------------
    if run_implement:
        plan = {
            "paper_claim_map": pcm,
            "environment_spec": env_spec,
            "reproduction_contract": contract,
        }
        impl_result, stop = _step("implement_baseline", plan)
        if stop and not _is_repairable(impl_result):
            return summary

        impl_repair_count = 0
        while _is_repairable(impl_result) and impl_repair_count < max_repair_iterations:
            impl_repair_count += 1
            plan["repair_context"] = impl_result
            impl_result, stop = _step("implement_baseline", plan)
            if stop and not _is_repairable(impl_result):
                return summary

        if _is_repairable(impl_result):
            # Repair budget exhausted and still repairable — stop honestly
            # rather than proceeding to run_experiment with no valid code_path.
            summary["stopped_at"] = "implement_baseline"
            summary["stopped_reason"] = "repair_exhausted"
            return summary

        # A repaired (or first-try) success clears any transient stop markers
        # _step set while the result still looked like an explicit failure.
        summary["stopped_at"] = None
        summary["stopped_reason"] = None
        summary["repaired"] += impl_repair_count

        # Extract code_path from impl result; fall back to default.
        if isinstance(impl_result.get("code_path"), str):
            code_path = impl_result["code_path"]

    # ------------------------------------------------------------------
    # Step 5: run_experiment (with bounded repair loop)
    # ------------------------------------------------------------------
    if run_experiment:
        last_result, stop = _step("run_experiment", code_path, "")
        if stop:
            # Even when run_experiment stopped the chain (fatal, ok=False, exception,
            # wall-clock), if the partial result carries gradeable metrics we must
            # still verify — otherwise a 3-of-6-cell partial grid ships None score.
            if run_verify and _has_gradeable_evidence(last_result):
                _safe_emit(
                    emit,
                    {
                        "event": "lifecycle_drive_step",
                        "stage": start_stage,
                        "primitive": "verify_against_rubric",
                        "note": "partial_evidence_verify",
                    },
                )
                _vfn = _get_tool(tools, "verify_against_rubric")
                if _vfn is not None:
                    try:
                        _vr = _call(_vfn, {}, rubric_spec)
                        summary["driven"].append("verify_against_rubric")
                        summary["rubric_score"] = _vr.get("overall_score")
                        summary["verify_result"] = _vr
                    except Exception:  # noqa: BLE001 — fail-soft; partial evidence, not fatal
                        pass
            return summary
        repair_count = 0
        prior_repair_fingerprint: str | None = None
        while _is_repairable(last_result) and repair_count < max_repair_iterations:
            repair_count += 1
            # Re-implement with the failed run as repair_context (the exact shape
            # implement_baseline consumes), then re-run. _step gates wall-clock +
            # fail-soft before each call.
            plan["repair_context"] = last_result
            impl_result, stop = _step("implement_baseline", plan)
            if stop:
                return summary
            if isinstance(impl_result.get("code_path"), str):
                code_path = impl_result["code_path"]
            last_result, stop = _step("run_experiment", code_path, "")
            if stop:
                # Same partial-evidence check inside the repair loop.
                if run_verify and _has_gradeable_evidence(last_result):
                    _safe_emit(
                        emit,
                        {
                            "event": "lifecycle_drive_step",
                            "stage": start_stage,
                            "primitive": "verify_against_rubric",
                            "note": "partial_evidence_verify",
                        },
                    )
                    _vfn2 = _get_tool(tools, "verify_against_rubric")
                    if _vfn2 is not None:
                        try:
                            _vr2 = _call(_vfn2, {}, rubric_spec)
                            summary["driven"].append("verify_against_rubric")
                            summary["rubric_score"] = _vr2.get("overall_score")
                            summary["verify_result"] = _vr2
                        except Exception:  # noqa: BLE001
                            pass
                return summary
            current_fingerprint = _evidence_fingerprint(last_result, ctx)
            if (
                prior_repair_fingerprint is not None
                and current_fingerprint == prior_repair_fingerprint
            ):
                # Two consecutive repairs produced identical evidence — another
                # repair is unlikely to help. Stop honestly instead of silently
                # grinding out the rest of the iteration cap.
                summary["stopped_reason"] = "repair_exhausted"
                break
            prior_repair_fingerprint = current_fingerprint
        summary["repaired"] += repair_count
        summary["last_run_ok"] = not _is_repairable(last_result)

    # ------------------------------------------------------------------
    # Step 6: verify_against_rubric
    # ------------------------------------------------------------------
    if run_verify:
        verify_result, stop = _step("verify_against_rubric", {}, rubric_spec)
        if stop:
            return summary

    # ------------------------------------------------------------------
    # Populate rubric_score from the verify result.
    # ------------------------------------------------------------------
    if verify_result is not None:
        summary["rubric_score"] = verify_result.get("overall_score")
        summary["verify_result"] = verify_result

    summary["iterations"] = int(getattr(ctx, "current_iteration", 0) or 0)
    return summary


def run_lifecycle_primary(
    *,
    tools: dict,
    ctx: Any,
    paper_text: str,
    rubric_spec: dict,
    emit: Any,
    target_score: float | None = None,
    max_repair_iterations: int = 2,
    max_improve_iterations: int = 2,
    min_remaining_s: float = 300.0,
) -> dict:
    """Harness-owned proactive reproduction: drive the full backbone (+repair) to a
    scored baseline, then climb toward target via propose_improvements.

    Fail-soft, wall-clock-aware. Returns a summary dict with keys:
        ``driven``          – aggregate list of primitive names executed
        ``rubric_score``    – best-of-climb score, or None when no baseline was scored
        ``verify_result``   – the latest verify result dict, or None
        ``improved``        – number of improvement iterations attempted
        ``stopped_reason``  – reason string, or None
    """
    summary: dict = {
        "driven": [],
        "rubric_score": None,
        "verify_result": None,
        "improved": 0,
        "stopped_reason": None,
        "fatal_result": None,
        "stopped_at": None,
        "reproduction_plan": None,
    }

    # ------------------------------------------------------------------
    # Phase 1: Backbone (understand → implement → run → verify) + repair.
    # ------------------------------------------------------------------
    base = drive_lifecycle_chain(
        tools=tools,
        ctx=ctx,
        paper_text=paper_text,
        rubric_spec=rubric_spec,
        start_stage=_STAGE_NEED_BASELINE,
        emit=emit,
        min_remaining_s=min_remaining_s,
        max_repair_iterations=max_repair_iterations,
    )

    summary["driven"].extend(base.get("driven") or [])
    summary["rubric_score"] = base.get("rubric_score")
    summary["verify_result"] = base.get("verify_result")
    summary["stopped_reason"] = base.get("stopped_reason")
    summary["fatal_result"] = base.get("fatal_result")
    summary["stopped_at"] = base.get("stopped_at")
    # Track iterations from the backbone; updated again at the terminal return
    # so any climb steps that advance ctx.current_iteration are also captured.
    summary["iterations"] = int(getattr(ctx, "current_iteration", 0) or 0)

    # E2: pre-commit the ordered plan_reproduction steps to disk (fail-soft) and
    # read them back — a durable, re-readable record of the dispatch order this
    # run committed to, independent of the in-memory value. No-op when the
    # backbone produced no plan/steps or ctx has no usable project_dir.
    plan_result = base.get("plan_result")
    steps = plan_result.get("steps") if isinstance(plan_result, dict) else None
    if steps is not None:
        try:
            plan_path = Path(ctx.project_dir) / "rlm_state" / "reproduction_plan.json"
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_path.write_text(
                json.dumps({"steps": steps}, indent=2, sort_keys=True), encoding="utf-8"
            )
            persisted = json.loads(plan_path.read_text(encoding="utf-8"))
            summary["reproduction_plan"] = persisted.get("steps")
        except Exception:  # noqa: BLE001 — persistence is advisory, never fatal
            pass

    # If the backbone never reached a score there is nothing to climb.
    if base.get("verify_result") is None:
        summary["stopped_reason"] = base.get("stopped_reason") or "no_baseline_score"
        return summary

    # ------------------------------------------------------------------
    # Phase 2: Improvement climb.
    # ------------------------------------------------------------------
    if max_improve_iterations <= 0:
        return summary

    verify_result: dict = base["verify_result"]
    score: float | None = verify_result.get("overall_score")
    target: float | None = verify_result.get("target_score") or target_score
    improved = 0

    while (
        score is not None
        and target is not None
        and score < target
        and improved < max_improve_iterations
    ):
        # Wall-clock gate.
        if not _wallclock_ok(ctx, min_remaining_s):
            summary["stopped_reason"] = "low_wallclock"
            break

        improved += 1

        # --- propose_improvements (fail-soft) ---
        hyps: list = []
        try:
            propose_fn = _get_tool(tools, "propose_improvements")
            if propose_fn is not None:
                raw = _call(propose_fn, verify_result, {"overall_score": score, "target_score": target})
                if isinstance(raw, list):
                    hyps = raw
                elif isinstance(raw, dict) and "_raw" in raw and isinstance(raw["_raw"], list):
                    hyps = raw["_raw"]
        except Exception:  # noqa: BLE001
            hyps = []

        valid = [h for h in hyps if isinstance(h, dict) and h.get("hypothesis") and h.get("success") is not False]
        if not valid:
            break

        chosen = valid[0]

        _safe_emit(
            emit,
            {"event": "lifecycle_drive_step", "stage": "improve", "phase": "improve",
             "primitive": "propose_improvements",
             "hypothesis": chosen.get("hypothesis"),
             "iteration": int(getattr(ctx, "current_iteration", 0) or 0)},
        )

        # --- implement_baseline with improvement patch (fail-soft) ---
        try:
            impl_fn = _get_tool(tools, "implement_baseline")
            if impl_fn is None:
                break
            impl_result = _call(impl_fn, {"repair_context": {"improvement": chosen, "prior_verify": verify_result}})
            if _is_explicit_failure(impl_result):
                break
        except Exception:  # noqa: BLE001
            break

        # --- run + repair + re-verify (reuse drive_lifecycle_chain at need_experiment) ---
        sub = drive_lifecycle_chain(
            tools=tools,
            ctx=ctx,
            paper_text=paper_text,
            rubric_spec=rubric_spec,
            start_stage=_STAGE_NEED_EXPERIMENT,
            emit=emit,
            min_remaining_s=min_remaining_s,
            max_repair_iterations=max_repair_iterations,
        )
        summary["driven"].extend(sub.get("driven") or [])

        # Best-of-climb: apply the sub-drive's score (if any) BEFORE deciding
        # whether to break — this ensures a partial-evidence verify from the
        # sub-drive (e.g. a partial OOM grid that still produced real metrics)
        # is captured even when last_run_ok is False or the sub-drive was fatal.
        sub_verify = sub.get("verify_result")
        if sub_verify is not None:
            sub_score = sub_verify.get("overall_score")
            if sub_score is not None and (score is None or sub_score >= score):
                score = sub_score
                verify_result = sub_verify

        if sub.get("fatal_result"):
            summary["fatal_result"] = sub.get("fatal_result")
            summary["stopped_at"] = sub.get("stopped_at")
            summary["stopped_reason"] = sub.get("stopped_reason") or "improve_fatal"
            break

        if sub_verify is None or sub.get("last_run_ok") is False:
            # No gradeable evidence from this climb step, or experiment still
            # repairable — do not continue improving with a broken implementation.
            summary["stopped_reason"] = sub.get("stopped_reason") or "improve_subdrive_no_progress"
            break

    summary["improved"] = improved
    # ``score`` and ``verify_result`` track the BEST outcome across the backbone
    # and all climb iterations.  The backbone's values were the starting point;
    # any climb iteration that produced a better score has already updated them
    # above. We therefore write these unconditionally (best-of-run is guaranteed
    # by the >= check in the sub_score update block and the non-regress loop).
    summary["rubric_score"] = score
    summary["verify_result"] = verify_result
    summary["iterations"] = int(getattr(ctx, "current_iteration", 0) or 0)
    return summary
