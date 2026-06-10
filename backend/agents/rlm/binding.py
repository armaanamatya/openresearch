"""Bind primitives to a RunContext and assemble the rlm `custom_tools` dict.

Phase 2 (issue #59). `build_custom_tools(ctx)` produces the dict
`rlm.RLM(custom_tools=...)` consumes: `{name: {"tool": callable, "description": str}}`.
Each wrapped callable emits a `primitive_call` SSE event (start + complete) and
appends a row to `cost_ledger.jsonl`.

Phase 6 (Task 13) additive wiring: after successful `propose_improvements` and
`verify_against_rubric` calls, and for `record_candidate_outcome`, emit the three
additional events described in the handoff spec
(docs/superpowers/specs/2026-05-21-rlm-phase4-backend-events-handoff.md §3–5).
All new events route through `ctx.emit` (the `make_emit`-produced thread-safe
closure) — never through `dashboard._emit` directly.
"""

from __future__ import annotations

import inspect
import json as _json
import logging
import sys
import threading
from concurrent.futures import Future, TimeoutError as FuturesTimeoutError
from typing import Any, Callable

from backend.agents.rlm.context import RunContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PR-γ.2 — Per-primitive wall-clock timeout table.
#
# Each entry gives a hard cap in seconds for that primitive's execution.
# On timeout the wrapper returns a retryable-outcome dict so PR-α's typestate
# routes the failure correctly.
#
# Intentionally EXCLUDES ``implement_baseline`` and ``run_experiment`` — both
# have existing, separately-designed caps (4h aclose watchdog and
# OPENRESEARCH_RUN_EXPERIMENT_TIMEOUT_S / ctx.remaining_s() respectively).
# The default for any primitive NOT in the table is 1800 s (30 min).
# ---------------------------------------------------------------------------

PRIMITIVE_TIMEOUT_S: dict[str, int] = {
    "understand_section": 300,
    "extract_hyperparameters": 300,
    "detect_environment": 600,
    "plan_reproduction": 600,
    "verify_against_rubric": 600,
    "propose_improvements": 300,
    "check_user_messages": 30,
    "respond_to_user": 30,
    "record_candidate_outcome": 60,
    "heartbeat": 30,
    # implement_baseline + run_experiment have their own existing caps — NOT here.
}

_DEFAULT_PRIMITIVE_TIMEOUT_S: int = 1800  # 30 min catch-all

# PR-μ.1 — the comment above is now ENFORCED. Before this fix, the
# `.get(name, _DEFAULT_PRIMITIVE_TIMEOUT_S)` line silently fell through to
# 1800s for `implement_baseline` and `run_experiment`, killing every max-mode
# `run_experiment` call at exactly 30 min regardless of PR-μ Solution B's
# inner 6h cap (the 0.305 Adam max-mode rerun was killed by this outer wrap,
# not by the inner resolver).  These primitives have their OWN internal
# caps (`resolve_experiment_timeout_s` for run_experiment; claude-agent-sdk
# session limits + 4h watchdog for implement_baseline).  The outer wrap here
# is a defensive long-tail bracket above the inner cap so it can never fire
# before the inner cap does.
_LONG_RUNNING_PRIMITIVES: dict[str, int] = {
    "run_experiment": 28800,      # 8h — bracket above 6h max-mode inner cap
    "implement_baseline": 21600,  # 6h — generous bracket; inner ~4h watchdog
}


def _coerce_args(fn: Callable[..., Any], args: tuple, kwargs: dict) -> tuple[tuple, dict, bool]:
    """Attempt one simple coercion pass when a positional arg has an obvious type mismatch.

    Inspects the function's signature and tries conservative type coercions:
      - str expected, non-string simple scalar given  → str(arg)
      - int expected, string-of-digits given          → int(arg)

    Does NOT attempt to fill missing required arguments or coerce dicts.
    Returns (new_args, new_kwargs, coerced_flag).  On any failure the original
    args/kwargs are returned unchanged with coerced=False.

    Note: primitives.py uses ``from __future__ import annotations``, so all
    annotations are stored as strings (PEP 563 lazy evaluation).  We read the
    raw annotation string from the signature directly and map "str"/"int" to the
    corresponding builtin types — no ``get_type_hints`` needed.
    """
    try:
        # primitives.py uses TYPE_CHECKING guards so RunContext cannot be resolved
        # via get_type_hints.  Read raw annotation strings from the signature and
        # map the simple builtins we care about.
        sig = inspect.signature(fn)
        params = [
            p for p in sig.parameters.values()
            if p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.POSITIONAL_ONLY,
            )
            and p.name != "ctx"
        ]
        if not params or not args:
            return args, kwargs, False

        new_args = list(args)
        coerced = False
        for i, param in enumerate(params):
            if i >= len(new_args):
                break
            raw_ann = param.annotation  # may be a string (PEP 563) or a type
            # Resolve the annotation: if it's already a type, use it;
            # if it's the string "str" / "int", map to the builtin.
            if raw_ann is inspect.Parameter.empty:
                continue
            if raw_ann is str or raw_ann == "str":
                ann = str
            elif raw_ann is int or raw_ann == "int":
                ann = int
            else:
                continue  # unknown annotation — skip, don't guess

            val = new_args[i]
            # str expected but not a str — coerce via str() for simple scalars only.
            # Dicts are NOT coerced (a dict where str is expected is almost certainly
            # a structural error the root should fix, not a trivial repr conversion).
            if ann is str and not isinstance(val, str):
                if isinstance(val, (int, float, bool)):
                    new_args[i] = str(val)
                    coerced = True
            # int expected but a string-of-digits given
            elif ann is int and isinstance(val, str) and val.strip().lstrip("-").isdigit():
                new_args[i] = int(val.strip())
                coerced = True

        return tuple(new_args), kwargs, coerced
    except Exception:  # noqa: BLE001 — never let coercion logic break a run
        return args, kwargs, False


def _summarize(args: tuple, kwargs: dict) -> dict:
    """A short, value-free summary of a primitive call's arguments."""
    out = {f"arg{i}": f"{type(a).__name__}[{len(a)}]" if hasattr(a, "__len__")
           else type(a).__name__ for i, a in enumerate(args)}
    out.update({k: type(v).__name__ for k, v in kwargs.items()})
    return out


def _result_summary(result: Any) -> str:
    """A short, value-free summary of a primitive's return value.

    When the result carries a ``_meta.hint`` (injected by understand_section
    and extract_hyperparameters for large slices), prepend ``[hint] `` so the
    SSE stream carries the signal to the UI and the root model sees it in the
    abbreviated event.  The rest of the summary is unchanged.
    """
    if isinstance(result, dict):
        base = f"dict[{', '.join(sorted(map(str, result))[:6])}]"
        if isinstance(result.get("_meta"), dict) and result["_meta"].get("hint"):
            return f"[hint] {base}"
        return base
    if isinstance(result, (list, str)):
        return f"{type(result).__name__}[{len(result)}]"
    return type(result).__name__


def _process_rss_bytes() -> int | None:
    """Best-effort current process RSS in bytes, without adding a dependency."""
    try:
        import resource
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if rss <= 0:
            return None
        if sys.platform == "darwin":
            return rss
        return rss * 1024
    except Exception:  # noqa: BLE001
        pass
    if sys.platform.startswith("win"):
        try:
            import ctypes
            from ctypes import wintypes

            class _ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            )
            if ok:
                return int(counters.WorkingSetSize)
        except Exception:  # noqa: BLE001
            return None
    return None


def _emit_primitive_resource(ctx: RunContext, *, primitive: str, boundary: str) -> None:
    """Emit RSS at a primitive boundary as a dashboard_event."""
    try:
        rss = _process_rss_bytes()
        payload: dict[str, Any] = {
            "primitive": primitive,
            "boundary": boundary,
        }
        if rss is not None:
            payload["rss_bytes"] = rss
            payload["rss_mb"] = round(rss / (1024 * 1024), 2)
        emit = getattr(getattr(ctx, "dashboard", None), "emit", None)
        if callable(emit):
            emit("primitive_resource", payload)
    except Exception:  # noqa: BLE001
        logger.exception("primitive %s: resource snapshot emit failed", primitive)


def _run_experiment_contract_guard(args: tuple[Any, ...]) -> tuple[tuple[Any, ...], dict[str, Any] | None]:
    """Unwrap valid implement_baseline envelopes and reject invalid code_path args."""
    if not args:
        return args, {
            "success": False,
            "metrics": {},
            "failure_class": "contract_guard",
            "source": "contract_guard",
            "error": "run_experiment: missing code_path argument",
            "contract_violations": [{
                "area": "Experiment execution and reproducibility",
                "detail": "code_path argument was omitted",
                "hint": "Call run_experiment only with a string path or an implement_baseline ok=true envelope.",
            }],
        }
    first = args[0]
    if isinstance(first, dict) and first.get("ok") is True:
        code_path = str(first.get("code_path") or "")
        if code_path.strip():
            return (code_path, *args[1:]), None
    if isinstance(first, str) and first.strip():
        return args, None
    return args, {
        "success": False,
        "metrics": {},
        "failure_class": "contract_guard",
        "source": "contract_guard",
        "error": (
            "run_experiment: code_path must be a non-empty string path or an "
            f"implement_baseline ok=true envelope, got {type(first).__name__}"
        ),
        "contract_violations": [{
            "area": "Experiment execution and reproducibility",
            "detail": f"code_path was {type(first).__name__!r}, not a usable baseline path",
            "hint": (
                "Never forward an implement_baseline ok=false/error dict to run_experiment. "
                "Call propose_improvements or retry implement_baseline instead."
            ),
        }],
    }


def _primitive_failure_details(result: Any) -> tuple[str | None, dict[str, Any]]:
    """Extract worker-report-safe failure fields from a primitive result."""
    if not isinstance(result, dict):
        return None, {}
    error = result.get("error")
    if error is None and result.get("error_code"):
        error = result.get("error_code")
    if error is None and result.get("success") is False:
        error = "primitive returned success=false"
    detail: dict[str, Any] = {}
    for key in ("failure_class", "contract_violations", "repairable", "source", "error_code", "missing_files"):
        if key in result:
            detail[key] = result.get(key)
    return (str(error) if error is not None else None), detail


# Chat-steering injection (2026-06-10). The steering contract asks the root to
# call check_user_messages() at every iteration start — but a non-cooperating
# root simply never does (both live Adam/All-CNN runs ignored time-critical
# operator steering for hours while burning GPU on work the messages would have
# prevented). Primitive results are the one channel the root ALWAYS reads, so
# unread messages ride along there. A separate injection cursor caps delivery
# at once per message; check_user_messages keeps its own cursor untouched so
# the formal flow (and respond_to_user) is unchanged.
_STEERING_INJECT_CURSOR = "_steering_inject_cursor.json"
_STEERING_INJECT_PRIMITIVES = frozenset({
    "run_experiment", "verify_against_rubric", "implement_baseline",
    "plan_reproduction", "propose_improvements",
})


def _inject_operator_messages(name: str, result: Any, ctx: "RunContext") -> Any:
    """Attach unread operator chat messages to a primitive's dict result.

    Fail-soft and flag-gated (``OPENRESEARCH_INJECT_STEERING=0`` disables). Returns
    ``result`` unchanged unless ``name`` is a steering-injection primitive, the
    result is a dict, and unread user messages exist.
    """
    import json as _json
    import os as _os
    from pathlib import Path as _Path

    try:
        if _os.environ.get(
            "OPENRESEARCH_INJECT_STEERING", ""
        ).strip().lower() in ("0", "false", "off"):
            return result
        if name not in _STEERING_INJECT_PRIMITIVES or not isinstance(result, dict):
            return result
        msgs_path = _Path(ctx.project_dir) / "user_messages.jsonl"
        if not msgs_path.exists():
            return result
        lines = [
            ln for ln in msgs_path.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        cursor_path = _Path(ctx.project_dir) / _STEERING_INJECT_CURSOR
        offset = 0
        if cursor_path.exists():
            try:
                offset = int(_json.loads(cursor_path.read_text(encoding="utf-8")).get("offset", 0))
            except Exception:  # noqa: BLE001 — corrupt cursor → start over (idempotent-ish)
                offset = 0
        fresh: list[str] = []
        for ln in lines[offset:]:
            try:
                d = _json.loads(ln)
            except Exception:  # noqa: BLE001 — tolerate a torn line
                continue
            if isinstance(d, dict) and d.get("role") == "user" and str(d.get("content", "")).strip():
                fresh.append(str(d["content"])[:600])
        if fresh:
            result = dict(result)
            result["operator_messages"] = fresh[-3:]
            result["operator_messages_note"] = (
                "Unread operator steering attached by the harness — read and act "
                "on it NOW, then acknowledge via respond_to_user()."
            )
        try:
            cursor_path.write_text(
                _json.dumps({"offset": len(lines)}), encoding="utf-8"
            )
        except OSError:
            logger.debug("steering injection: cursor write failed (non-fatal)")
        return result
    except Exception:  # noqa: BLE001 — injection must never break a primitive result
        logger.debug("steering injection skipped (non-fatal)", exc_info=True)
        return result


def wrap_primitive(name: str, fn: Callable[..., Any], ctx: RunContext) -> Callable[..., Any]:
    """Close `fn` over `ctx`, adding primitive_call emission and a cost-ledger row.

    Phase 6 (Task 13): after successful ``propose_improvements``,
    ``verify_against_rubric``, and ``record_candidate_outcome`` calls, emit
    the three new SSE events via ``ctx.emit`` (the ``make_emit``-produced
    thread-safe chokepoint).  When ``ctx.emit`` is None (e.g. old test fixtures
    that have not been updated), the additional emissions are silently skipped —
    the existing ``primitive_call`` events still fire via ``dashboard``.
    """

    def _ledger(outcome: str = "") -> None:
        """Append a cost-ledger row for this primitive invocation.

        Reads ctx.llm_client._last_usage (populated by ClaudeLlmClient.complete)
        to capture tokens for primitives that call the LLM via llm_client.
        For primitives that don't call the LLM (heuristics) or that use the
        agent engine path (implement_baseline — engine writes its own entry),
        _last_usage stays at the zeroed value and the entry records 0 tokens,
        which is correct.

        ``outcome`` is the per-row provenance stamp (audit 2026-06-10): "ok"
        (returned non-failure), "failed" (returned a failure-shaped dict), or
        "raised". The evidence gate consumes it via
        ``RunCostLedger.session_success_compatible_count`` so a real-but-FAILED
        run_experiment call can no longer back a forged success row.
        """
        usage: dict = {}
        llm_client = getattr(ctx, "llm_client", None)
        if llm_client is not None:
            last_usage = getattr(llm_client, "_last_usage", None)
            if isinstance(last_usage, dict):
                usage = last_usage
        from backend.agents.resilience.cost import CostLedgerEntry as _CLE
        ctx.cost_ledger.append(_CLE.from_usage(
            agent_id=name,
            attempt_index=0,
            provider=ctx.provider,
            model=ctx.model,
            usage=usage,
            outcome=outcome,
        ))

    def _emit_extra(event: dict) -> None:
        """Emit via the thread-safe chokepoint; skip if ctx.emit is unset."""
        if ctx.emit is not None:
            ctx.emit(event)

    # Primitives that should generate a worker report
    _REPORT_PRIMITIVES = {
        "implement_baseline", "build_environment", "run_experiment",
        "verify_against_rubric", "propose_improvements",
    }

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        kwargs.pop("ctx", None)  # the wrapper supplies ctx; never let a caller double it
        # Attempt one conservative coercion pass before calling the primitive.
        # This repairs obvious type mismatches (str expected but int/list passed;
        # int expected but string-of-digits passed) so the root does not burn
        # a full iteration cycle on a trivially fixable ValidationError.
        args, kwargs, coerced = _coerce_args(fn, args, kwargs)
        if coerced:
            logger.info("primitive %s: args auto-coerced", name)
        # Zero out llm_client._last_usage at the boundary of every primitive call
        # so we only capture tokens from THIS invocation, not carry over stale
        # usage from a previous primitive's LLM call.
        _llm_client = getattr(ctx, "llm_client", None)
        if _llm_client is not None and hasattr(_llm_client, "_last_usage"):
            _llm_client._last_usage = {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "reasoning_tokens": 0,
            }
        _emit_primitive_resource(ctx, primitive=name, boundary="start")
        ctx.dashboard.primitive_call(name, "start", args_summary=_summarize(args, kwargs))

        # Open a worker report for key primitives
        _wr_report = None
        _wr_start = None
        if name in _REPORT_PRIMITIVES:
            try:
                import time as _time
                from backend.agents.worker_reports import (
                    WORKER_TYPE_RLM_PRIMITIVE,
                    build_extended_worker_report,
                    build_worker_report_started_event,
                    open_worker_report,
                )
                _wr_start = _time.monotonic()
                _wr_report = build_extended_worker_report(
                    run_id=getattr(ctx, "project_id", None),
                    worker_type=WORKER_TYPE_RLM_PRIMITIVE,
                    agent_id=name,
                    project_dir=ctx.project_dir,
                    model=ctx.model,
                    provider=ctx.provider,
                    status="running",
                    assignment={"summary": f"RLM primitive: {name}"},
                )
                open_worker_report(ctx.project_dir, _wr_report)
                _emit_extra(build_worker_report_started_event(_wr_report))
            except Exception:  # noqa: BLE001
                _wr_report = None

        # PR-γ.2 — per-primitive wall-clock enforcement.
        # Resolution order:
        #   1. _LONG_RUNNING_PRIMITIVES (run_experiment, implement_baseline)
        #      — they have inner caps; outer wrap is a defensive long-tail.
        #   2. PRIMITIVE_TIMEOUT_S — explicit table.
        #   3. _DEFAULT_PRIMITIVE_TIMEOUT_S (1800s) — catch-all for everything else.
        if name in _LONG_RUNNING_PRIMITIVES:
            _timeout_s = _LONG_RUNNING_PRIMITIVES[name]
        else:
            _timeout_s = PRIMITIVE_TIMEOUT_S.get(name, _DEFAULT_PRIMITIVE_TIMEOUT_S)

        try:
            try:
                # Run the primitive in a daemon thread so a hung call does NOT
                # prevent the process from returning. Python cannot forcibly
                # interrupt a thread mid-execution, but a daemon thread is
                # silently abandoned when the main thread moves on — it won't
                # hold up the interpreter or the test runner.
                guard_result = None
                if name == "run_experiment":
                    args, guard_result = _run_experiment_contract_guard(args)

                if guard_result is not None:
                    result = guard_result
                else:
                    _prim_future: Future = Future()

                    def _runner() -> None:
                        try:
                            _prim_future.set_result(fn(*args, **{**kwargs, "ctx": ctx}))
                        except Exception as _e:  # noqa: BLE001
                            _prim_future.set_exception(_e)

                    _t = threading.Thread(
                        target=_runner,
                        name=f"prim-{name}",
                        daemon=True,
                    )
                    _t.start()
                    try:
                        result = _prim_future.result(timeout=_timeout_s)
                    except FuturesTimeoutError:
                        logger.warning(
                            "primitive %s timed out after %ss — marking retryable",
                            name, _timeout_s,
                        )
                        # Emit a run_warning SSE event so the UI and cost-ledger
                        # reflect the hung primitive.
                        try:
                            from backend.agents.rlm.primitives import (
                                _emit_dashboard_event as _emit_evt,
                            )
                            _emit_evt(ctx, event_type="run_warning", payload={
                                "code": "primitive_timeout",
                                "primitive": name,
                                "wall_clock_s": _timeout_s,
                                "message": (
                                    f"primitive `{name}` exceeded its wall-clock "
                                    f"cap of {_timeout_s}s and was interrupted. "
                                    f"The orchestrator can retry this primitive."
                                ),
                            })
                        except Exception:  # noqa: BLE001 — emit MUST NOT break the run
                            pass
                        result = {
                            "outcome": "retryable",
                            "error": "primitive_hung",
                            "primitive": name,
                            "wall_clock_s": _timeout_s,
                        }
            except Exception as exc:
                # Value-free event: an exception MESSAGE can carry raw LLM output,
                # paper text or paths, and result_summary is streamed to the UI.
                # Only the type + (for pydantic ValidationError) field paths +
                # pydantic error types go into the event. Field paths and types
                # come from the schema, not from user/LLM values, so they're safe
                # to surface. The full traceback stays server-side.
                #
                # Before 2026-05-23 this only emitted `type(exc).__name__`, so the
                # UI showed "Exception"/"ValidationError" with zero detail. That
                # made the user think the run was stuck when the root was just
                # adapting to a schema mismatch. See fix-plan §T2 / U2.
                summary = type(exc).__name__
                try:
                    from pydantic import ValidationError as _PydanticVE
                    if isinstance(exc, _PydanticVE):
                        parts: list[str] = []
                        for err in exc.errors()[:6]:  # cap at 6 for SSE payload bound
                            loc = ".".join(str(p) for p in err.get("loc", ()))
                            msg = str(err.get("msg", ""))[:80]
                            etype = err.get("type", "")
                            parts.append(f"{loc}: {msg} ({etype})")
                        if parts:
                            summary = "ValidationError: " + "; ".join(parts)
                            # bound at 500 chars regardless of error count
                            if len(summary) > 500:
                                summary = summary[:497] + "..."
                except Exception:  # noqa: BLE001 — defensive; pydantic import / .errors() must not break the wrapper
                    pass
                ctx.dashboard.primitive_call(name, "error", result_summary=summary)
                _emit_primitive_resource(ctx, primitive=name, boundary="end")
                _ledger("raised")
                logger.warning("primitive %s raised %s", name, type(exc).__name__)
                raise
            # Most primitives are fail-soft: on failure they RETURN a failure-shaped
            # dict instead of raising. Surface that as an `error` primitive_call and
            # a server-side WARNING — otherwise it is silently logged as a success
            # (a run-7 verify_against_rubric failure stayed invisible until traced
            # by hand).
            failed = isinstance(result, dict) and (
                result.get("success") is False or bool(result.get("error"))
            )
            ctx.dashboard.primitive_call(
                name, "error" if failed else "ok",
                result_summary=_result_summary(result),
                coerced=coerced,
            )
            _emit_primitive_resource(ctx, primitive=name, boundary="end")
            _ledger("failed" if failed else "ok")
            if failed:
                logger.warning(
                    "primitive %s returned a failure: %s",
                    name, result.get("error") or "(see dashboard_events.jsonl)",
                )
            else:
                # --- Phase 6 (Task 13): post-success supplemental event emission ---
                _emit_supplemental(name, result, ctx, _emit_extra)
                # PEEK-lite (OPENRESEARCH_CONTEXT_MAP): union orientation-primitive
                # outputs into rlm_state/context_map.json. Flag-gated + fail-soft
                # inside update_context_map; no-op for non-orientation primitives.
                try:
                    from backend.agents.rlm.context_map import update_context_map
                    update_context_map(ctx.project_dir, name, result)
                except Exception:  # noqa: BLE001 — orientation cache must never break a call
                    pass
            # Steering (main 2026-06-09): surface unread operator messages inside
            # primitive results so the root sees them without polling.
            result = _inject_operator_messages(name, result, ctx)
            return result
        finally:
            # Finalize the worker report if one was opened
            if _wr_report is not None:
                try:
                    import time as _time
                    from backend.agents.worker_reports import (
                        build_worker_report_completed_event,
                        build_worker_report_failed_event,
                        finalize_worker_report,
                    )
                    elapsed = int((_time.monotonic() - _wr_start) * 1000) if _wr_start else None
                    try:
                        _wr_result = result  # noqa: F841
                        is_failed = isinstance(_wr_result, dict) and (
                            _wr_result.get("success") is False or bool(_wr_result.get("error"))
                        )
                    except NameError:
                        is_failed = True
                    final_status = "failed" if is_failed else "completed"
                    _wr_error = None
                    _wr_details: dict[str, Any] = {}
                    try:
                        _wr_error, _wr_details = _primitive_failure_details(_wr_result)
                    except NameError:
                        _wr_error = "primitive raised before returning a result"
                        _wr_details = {"source": "exception"}
                    if _wr_details:
                        _wr_report.update(_wr_details)
                    finalize_worker_report(
                        ctx.project_dir, _wr_report,
                        status=final_status,
                        duration_ms=elapsed,
                        error=_wr_error if is_failed else None,
                        errors=[{
                            "message": _wr_error,
                            "source": _wr_details.get("source") or "primitive_result",
                            "recoverable": bool(_wr_details.get("repairable", False)),
                            "failure_class": _wr_details.get("failure_class"),
                            "contract_violations": _wr_details.get("contract_violations"),
                        }] if is_failed and _wr_error else None,
                    )
                    if is_failed:
                        _emit_extra(build_worker_report_failed_event(_wr_report))
                    else:
                        _emit_extra(build_worker_report_completed_event(_wr_report))
                except Exception:  # noqa: BLE001
                    pass
            # Flush any buffered cost-ledger entries at the end of every primitive
            # boundary so the file is always in a consistent state for inspection
            # after each primitive completes.  flush() is idempotent and lock-safe.
            ctx.cost_ledger.flush()

    wrapped.__name__ = name
    return wrapped


def _friendly_candidate_title(title: str) -> str:
    """Return a display-friendly condensed form of a candidate title.

    Rules (spec 2026-05-23):
      - Already short (≤5 words AND ≤40 chars) → return as-is.
      - Has a colon or em-dash → return the part before the separator (stripped).
      - Else → return the first 5 words joined by spaces.
    """
    title = title.strip()
    words = title.split()
    if len(words) <= 5 and len(title) <= 40:
        return title
    # Try splitting on colon or em-dash
    for sep in ("—", ": ", " - "):
        if sep in title:
            prefix = title.split(sep, 1)[0].strip()
            if prefix:
                return prefix
    # Fallback: first 5 words
    return " ".join(words[:5])


def _emit_supplemental(
    name: str,
    result: Any,
    ctx: RunContext,
    emit_extra: Callable[[dict], None],
) -> None:
    """Emit supplemental SSE events after a SUCCESSFUL primitive call.

    Three primitives produce dedicated events that carry richer data than the
    value-free ``primitive_call(ok)`` event:

    * ``propose_improvements`` → one ``candidate_proposed`` per hypothesis.
    * ``verify_against_rubric`` → one ``rubric_score`` (only on real success).
    * ``record_candidate_outcome`` → one ``candidate_outcome``.

    This function is a no-op for all other primitives.
    """
    from backend.agents.rlm.sse_bridge import (
        build_candidate_outcome_event,
        build_candidate_proposed_event,
        build_rubric_score_event,
    )

    if name == "propose_improvements" and isinstance(result, list):
        # Increment propose_round BEFORE the per-hypothesis loop so all events
        # in this fan share the same round number.
        ctx.propose_round += 1
        for hyp in result:
            if not isinstance(hyp, dict):
                continue
            raw_title = hyp.get("title") or hyp.get("path_id", "candidate")
            candidate = {
                "id": hyp.get("path_id", ""),
                "title": raw_title,
                "display_title": _friendly_candidate_title(raw_title),
                "category": hyp.get("category", ""),
                "description": hyp.get("hypothesis", ""),
                "reasoning": hyp.get("rationale", ""),
            }
            if any(
                not isinstance(candidate.get(key), str) or not candidate.get(key, "").strip()
                for key in ("id", "category", "description", "reasoning")
            ):
                continue
            emit_extra(build_candidate_proposed_event(
                iteration=ctx.current_iteration,
                round=ctx.propose_round,
                candidate=candidate,
            ))

    elif name == "verify_against_rubric" and isinstance(result, dict):
        # Only emit on a genuinely successful verification — failure returns
        # {"success": False, ...} and is already handled by the `failed` branch
        # above, so we never reach here for a failed verification.
        score = result.get("overall_score")
        target = result.get("target_score")
        areas = result.get("areas", [])
        if score is not None and isinstance(areas, list):
            # Lane H — stash the latest score so the FINAL_VAR interceptor can
            # decide whether to force another iteration before accepting the
            # root's bail-out. Set BEFORE emit so a downstream emit failure
            # cannot leave the policy reading stale state.
            try:
                ctx.latest_rubric_score = float(score)
                ctx.latest_rubric_target = float(target) if target is not None else 0.0
                ctx.latest_rubric_iteration = ctx.current_iteration
            except (TypeError, ValueError):
                # Malformed score — leave policy state untouched; the
                # interceptor treats None-score as "no rubric, accept".
                pass
            # Pass through the per-area `leaves` detail (id/label/score/status/why)
            # that `_rubric_areas` attached, plus the cross-area weak_leaves and
            # the last few failed-experiment rows, so the lab UI can show
            # leaf-level "which criteria fail + why" without a second fetch.
            # build_rubric_score_event re-derives every status and bounds sizes.
            _weak_leaves = [
                {
                    "id": w.get("id", ""),
                    "score": w.get("score"),
                    "why": w.get("justification") or w.get("why") or "",
                    "area": w.get("area", ""),
                }
                for w in (result.get("weak_leaves") or [])
                if isinstance(w, dict)
            ]
            try:
                from backend.agents.rlm.primitives import _recent_experiment_errors
                _recent_errors = _recent_experiment_errors(ctx.project_dir)
            except Exception:  # noqa: BLE001 — observability must never block emit
                _recent_errors = []
            emit_extra(build_rubric_score_event(
                iteration=ctx.current_iteration,
                score=float(score),
                target=float(target) if target is not None else 0.0,
                areas=[
                    {"area": a.get("area") or a.get("name") or "", "score": a.get("score", 0.0),
                     "weight": a.get("weight", 0.0),
                     "leaves": a.get("leaves") if isinstance(a.get("leaves"), list) else []}
                    for a in areas if isinstance(a, dict)
                ],
                weak_leaves=_weak_leaves,
                recent_errors=_recent_errors,
            ))

            # 2026-05-26: persist the full verify_against_rubric payload so
            # write_final_report_rlm can merge per-leaf justifications into
            # final_report.json::rubric. The SSE event surface stays minimal
            # (areas only); the deep data lives on disk. Atomic write so a
            # mid-write crash never leaves a half-baked rubric_evaluation.json.
            try:
                import os as _os
                from pathlib import Path as _Path
                eval_path = _Path(ctx.project_dir) / "rubric_evaluation.json"
                tmp = eval_path.with_suffix(".json.tmp")
                payload = {
                    "iteration": ctx.current_iteration,
                    "overall_score": float(score),
                    "target_score": float(target) if target is not None else None,
                    "meets_target": result.get("meets_target"),
                    "leaf_count": result.get("leaf_count"),
                    "graded": result.get("graded"),
                    "rubric_source": result.get("rubric_source"),
                    "degraded": result.get("degraded"),
                    "compute_adjusted_score": result.get("compute_adjusted_score"),
                    "compute_scope": result.get("compute_scope"),
                    "coverage_pct": result.get("coverage_pct"),
                    "areas": areas,
                    "leaf_scores": result.get("leaf_scores", []),
                    "weak_leaves": result.get("weak_leaves", []),
                }
                tmp.write_text(_json.dumps(payload, indent=2, default=str))
                _os.replace(tmp, eval_path)
            except Exception:  # noqa: BLE001 — persistence is best-effort
                logger.debug("binding: rubric_evaluation.json persist failed (non-fatal)")

    elif name == "record_candidate_outcome" and isinstance(result, dict):
        # 2026-05-23: only emit when the primitive returned success — bad input
        # (None, "None", invalid outcome) returns {"success": False, "error": ...}
        # and must NOT produce a candidate_outcome event with corrupted fields.
        # The 2026-05-23 bug: every outcome event had candidate_id="None" because
        # str(None) was emitted unconditionally.
        if not result.get("success"):
            return
        cid = result.get("candidate_id") or ""
        outc = result.get("outcome") or ""
        if not cid or not outc:
            return  # defensive: even if success=True somehow, do not emit garbage
        emit_extra(build_candidate_outcome_event(
            iteration=ctx.current_iteration,
            candidate_id=str(cid),
            outcome=str(outc),
            rubric_delta=None,  # root supplies outcome; delta is not computed here
        ))


def build_custom_tools(
    ctx: RunContext,
    *,
    registry: dict[str, Callable[..., Any]] | None = None,
    descriptions: dict[str, str] | None = None,
) -> dict[str, dict]:
    """Return the rlm `custom_tools` dict, every primitive closed over `ctx`.

    The consumer MUST instantiate `rlm.RLM(environment="local")`: `rlm`'s
    `DockerREPL` silently drops `custom_tools` (it absorbs the kwarg into
    `**kwargs` and never injects the tools), so under `environment="docker"`
    none of these primitives would exist in the REPL. `"local"` is also a
    security boundary — see the threat model in
    `docs/design/rlm-pivot-brief.md` §7.
    """
    if registry is None or descriptions is None:
        # Imported here, not at module scope, so a caller that passes both
        # `registry=` and `descriptions=` explicitly (e.g. tests) never imports
        # primitives.py at all. A kwarg left as None falls back to the
        # module-level PRIMITIVE_REGISTRY / PRIMITIVE_DESCRIPTIONS.
        from backend.agents.rlm import primitives as _p
        registry = registry if registry is not None else _p.PRIMITIVE_REGISTRY
        descriptions = descriptions if descriptions is not None else _p.PRIMITIVE_DESCRIPTIONS
    return {
        name: {"tool": wrap_primitive(name, fn, ctx),
               "description": descriptions.get(name, name)}
        for name, fn in registry.items()
    }
