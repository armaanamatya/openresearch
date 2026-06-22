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

__all__ = ["drive_lifecycle_chain"]

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

    Returns
    -------
    dict with keys:
        ``driven``       – list of primitive names executed in order
        ``stopped_at``   – primitive name that failed/was skipped, or None
        ``stopped_reason`` – reason string, or None
        ``final_result`` – the last step's raw result dict, or None
        ``rubric_score`` – ``verify.get("overall_score")`` if verify ran, else None
    """
    # Canonical no-op return shape.
    summary: dict = {
        "driven": [],
        "stopped_at": None,
        "stopped_reason": None,
        "final_result": None,
        "rubric_score": None,
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
    code_path: str = str(ctx.project_dir / "code")
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

        _safe_emit(
            emit,
            {"event": "lifecycle_drive_step", "stage": start_stage, "primitive": name},
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

    # ------------------------------------------------------------------
    # Step 4: implement_baseline  (need_baseline only)
    # ------------------------------------------------------------------
    if run_implement:
        plan = {
            "paper_claim_map": pcm,
            "environment_spec": env_spec,
            "reproduction_contract": contract,
        }
        impl_result, stop = _step("implement_baseline", plan)
        if stop:
            return summary
        # Extract code_path from impl result; fall back to default.
        if isinstance(impl_result.get("code_path"), str):
            code_path = impl_result["code_path"]

    # ------------------------------------------------------------------
    # Step 5: run_experiment
    # ------------------------------------------------------------------
    if run_experiment:
        last_result, stop = _step("run_experiment", code_path, "")
        if stop:
            return summary

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

    return summary
