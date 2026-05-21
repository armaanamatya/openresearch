"""Bind primitives to a RunContext and assemble the rlm `custom_tools` dict.

Phase 2 (issue #59). `build_custom_tools(ctx)` produces the dict
`rlm.RLM(custom_tools=...)` consumes: `{name: {"tool": callable, "description": str}}`.
Each wrapped callable emits a `primitive_call` SSE event (start + complete) and
appends a row to `cost_ledger.jsonl`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from backend.agents.resilience.cost import CostLedgerEntry
from backend.agents.rlm.context import RunContext


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
            ctx.dashboard.primitive_call(
                name, "error", result_summary=f"{type(exc).__name__}: {exc}"[:200])
            _ledger()
            raise
        ctx.dashboard.primitive_call(name, "ok", result_summary=_result_summary(result))
        _ledger()
        return result

    wrapped.__name__ = name
    return wrapped


def build_custom_tools(
    ctx: RunContext,
    *,
    registry: dict[str, Callable[..., Any]] | None = None,
    descriptions: dict[str, str] | None = None,
) -> dict[str, dict]:
    """Return the rlm `custom_tools` dict, every primitive closed over `ctx`."""
    if registry is None or descriptions is None:
        # Deferred import: keeps this module importable without loading
        # primitives.py at module-load time. A kwarg left as None falls back
        # to the module-level PRIMITIVE_REGISTRY / PRIMITIVE_DESCRIPTIONS.
        from backend.agents.rlm import primitives as _p
        registry = registry if registry is not None else _p.PRIMITIVE_REGISTRY
        descriptions = descriptions if descriptions is not None else _p.PRIMITIVE_DESCRIPTIONS
    return {
        name: {"tool": wrap_primitive(name, fn, ctx),
               "description": descriptions.get(name, name)}
        for name, fn in registry.items()
    }
