"""Bind primitives to a RunContext and assemble the rlm `custom_tools` dict.

Phase 2 (issue #59). `build_custom_tools(ctx)` produces the dict
`rlm.RLM(custom_tools=...)` consumes: `{name: {"tool": callable, "description": str}}`.
Each wrapped callable emits a `primitive_call` SSE event (start + complete) and
appends a row to `cost_ledger.jsonl`.
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
    """Close `fn` over `ctx`, adding primitive_call emission and a cost-ledger row."""

    def _ledger() -> None:
        # Phase 2 (D7): a zero-usage call entry; real token usage lands with run.py (#60).
        ctx.cost_ledger.append(CostLedgerEntry(
            timestamp=datetime.now(timezone.utc),
            agent_id=name,
            attempt_index=0,
            provider=ctx.provider,
            model=ctx.model,
        ))

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
        return result

    wrapped.__name__ = name
    return wrapped


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
