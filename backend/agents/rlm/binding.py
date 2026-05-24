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
import logging
from datetime import datetime, timezone
from typing import Any, Callable

from backend.agents.resilience.cost import CostLedgerEntry
from backend.agents.rlm.context import RunContext

logger = logging.getLogger(__name__)


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


def wrap_primitive(name: str, fn: Callable[..., Any], ctx: RunContext) -> Callable[..., Any]:
    """Close `fn` over `ctx`, adding primitive_call emission and a cost-ledger row.

    Phase 6 (Task 13): after successful ``propose_improvements``,
    ``verify_against_rubric``, and ``record_candidate_outcome`` calls, emit
    the three new SSE events via ``ctx.emit`` (the ``make_emit``-produced
    thread-safe chokepoint).  When ``ctx.emit`` is None (e.g. old test fixtures
    that have not been updated), the additional emissions are silently skipped —
    the existing ``primitive_call`` events still fire via ``dashboard``.
    """

    def _ledger() -> None:
        # Phase 2 (D7): a zero-usage call entry; real token usage lands with run.py (#60).
        ctx.cost_ledger.append(CostLedgerEntry(
            timestamp=datetime.now(timezone.utc),
            agent_id=name,
            attempt_index=0,
            provider=ctx.provider,
            model=ctx.model,
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

        try:
            try:
                result = fn(*args, ctx=ctx, **kwargs)
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
                _ledger()
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
            _ledger()
            if failed:
                logger.warning(
                    "primitive %s returned a failure: %s",
                    name, result.get("error") or "(see dashboard_events.jsonl)",
                )
            else:
                # --- Phase 6 (Task 13): post-success supplemental event emission ---
                _emit_supplemental(name, result, ctx, _emit_extra)
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
                    finalize_worker_report(
                        ctx.project_dir, _wr_report,
                        status=final_status,
                        duration_ms=elapsed,
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
            emit_extra(build_rubric_score_event(
                iteration=ctx.current_iteration,
                score=float(score),
                target=float(target) if target is not None else 0.0,
                areas=[
                    {"area": a.get("area") or a.get("name") or "", "score": a.get("score", 0.0),
                     "weight": a.get("weight", 0.0)}
                    for a in areas if isinstance(a, dict)
                ],
            ))

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
