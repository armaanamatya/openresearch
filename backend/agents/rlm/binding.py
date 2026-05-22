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

import logging
from datetime import datetime, timezone
from typing import Any, Callable

from backend.agents.resilience.cost import CostLedgerEntry
from backend.agents.rlm.context import RunContext

logger = logging.getLogger(__name__)


def _summarize(args: tuple, kwargs: dict) -> dict:
    """A short, value-free summary of a primitive call's arguments."""
    out = {f"arg{i}": f"{type(a).__name__}[{len(a)}]" if hasattr(a, "__len__")
           else type(a).__name__ for i, a in enumerate(args)}
    out.update({k: type(v).__name__ for k, v in kwargs.items()})
    return out


def _result_summary(result: Any) -> str:
    """A short, value-free summary of a primitive's return value."""
    if isinstance(result, dict):
        return f"dict[{', '.join(sorted(map(str, result))[:6])}]"
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

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        kwargs.pop("ctx", None)  # the wrapper supplies ctx; never let a caller double it
        ctx.dashboard.primitive_call(name, "start", args_summary=_summarize(args, kwargs))
        try:
            result = fn(*args, ctx=ctx, **kwargs)
        except Exception as exc:
            # Value-free event: an exception MESSAGE can carry raw LLM output,
            # paper text or paths, and result_summary is streamed to the UI.
            # Only the type goes into the event. The WARNING and the `raise`
            # below are server-side only.
            ctx.dashboard.primitive_call(name, "error", result_summary=type(exc).__name__)
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

    wrapped.__name__ = name
    return wrapped


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
            candidate = {
                "id": hyp.get("path_id", ""),
                "title": hyp.get("title") or hyp.get("path_id", "candidate"),
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
                    {"area": a.get("area", ""), "score": a.get("score", 0.0),
                     "weight": a.get("weight", 0.0)}
                    for a in areas if isinstance(a, dict)
                ],
            ))

    elif name == "record_candidate_outcome" and isinstance(result, dict):
        emit_extra(build_candidate_outcome_event(
            iteration=ctx.current_iteration,
            candidate_id=str(result.get("candidate_id", "")),
            outcome=str(result.get("outcome", "")),
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
