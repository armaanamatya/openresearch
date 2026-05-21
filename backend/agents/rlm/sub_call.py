"""Python callables exposed to the REPL: `llm_query` and `rlm_query`.

These are the paper's `sub_LLM` and `sub_RLM` — Python functions the root
model invokes from inside its code, NOT Anthropic/OpenAI `tool_use` blocks
on the root API call. The distinction is brief §13 FM#2 (pseudo-recursion):
if `sub_LLM` / `sub_RLM` ever appear in the root model's tools list, the
implementation has slipped back to Algorithm 2.

Phase 2 (#59) implementation. Reuses `RlmQueryTool._recursive_query` from
`backend/services/context/workspace/tools/rlm_query.py` as the `sub_RLM`
engine. See `docs/rlm-pivot-mapping.md` §3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Per-run caps (brief §13 FM#10 + correction #8). These are sub-call guards;
# the root-iteration cap lives in root_loop.py and is separate.
DEFAULT_MAX_SUB_CALLS_PER_RUN = 50
DEFAULT_MAX_SUB_CALL_COST_USD = 10.0

# Recursion depth (brief correction #1). depth=2 makes sub_RLM real;
# depth=1 means rlm_query degenerates to llm_query.
DEFAULT_SUB_RLM_DEPTH = 2


@dataclass
class SubCallBudget:
    """Run-wide budget shared across all sub-calls. Mutated in-place."""

    max_calls: int = DEFAULT_MAX_SUB_CALLS_PER_RUN
    max_cost_usd: float = DEFAULT_MAX_SUB_CALL_COST_USD
    calls_made: int = 0
    cost_accrued_usd: float = 0.0

    def can_spend(self, estimated_cost_usd: float) -> bool:
        """True if another sub-call is within budget."""
        return (
            self.calls_made < self.max_calls
            and self.cost_accrued_usd + estimated_cost_usd <= self.max_cost_usd
        )

    def record(self, cost_usd: float) -> None:
        self.calls_made += 1
        self.cost_accrued_usd += cost_usd


def bind_sub_calls(
    *,
    llm_client: Any,        # LlmClient Protocol
    budget: SubCallBudget,
    sub_rlm_depth: int = DEFAULT_SUB_RLM_DEPTH,
) -> dict[str, Any]:
    """Return a dict suitable for injection into the REPL globals.

    The returned dict contains `llm_query` and `rlm_query` callables that
    close over `llm_client` and `budget`. The REPL host injects these into
    the namespace so the root's `exec`-ed code can call them directly.

    Phase 2 (#59) implementation:
      - `llm_query(prompt, model="default")` — single leaf call via
        `llm_client.complete(system=..., user=prompt)`. Records cost.
      - `rlm_query(context, query)` — invokes the existing
        `RlmQueryTool._recursive_query(content=context, question=query, ...)`
        with `depth=0, max_depth=sub_rlm_depth`. At the depth cap, falls
        back to `llm_query` (paper's documented behavior).
      - Both emit `primitive_call` (for llm_query) / `sub_rlm_spawned`
        (for rlm_query) SSE events.
    """
    raise NotImplementedError("Phase 2 (#59) — bind llm_query/rlm_query to REPL namespace")


def llm_query(prompt: str, model: str = "default") -> str:
    """Placeholder — the actual callable is constructed by `bind_sub_calls()`.

    This module-level export exists so `from backend.agents.rlm import
    llm_query` works for type-checking. At runtime the REPL receives the
    closure produced by `bind_sub_calls()`, not this stub.
    """
    raise NotImplementedError("Phase 2 (#59) — call bind_sub_calls() to construct the REPL-bound version")


def rlm_query(context: str, query: str) -> str:
    """Placeholder — see `llm_query` docstring."""
    raise NotImplementedError("Phase 2 (#59) — call bind_sub_calls() to construct the REPL-bound version")
