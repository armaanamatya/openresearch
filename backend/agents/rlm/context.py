"""RunContext — run-scoped dependencies threaded into every RLM primitive.

Phase 2 (issue #59). The root RLM model passes only slices/specs as primitive
arguments (the Algorithm-2 guard). Everything else a primitive needs — paths,
the event emitter, the cost ledger, the LLM client, the agent runtime — lives
here and is closed over by `backend.agents.rlm.binding.build_custom_tools`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RunContext:
    """Everything a primitive needs that the root model does not pass.

    `llm_client` is the synchronous `LlmClient` protocol from
    `backend/services/context/workspace/tools/rlm_query.py` — `.complete(*,
    system, user) -> str`. `runtime` is an `AgentRuntime`; only
    `implement_baseline` needs it, so it defaults to None.
    """

    project_id: str
    project_dir: Path
    runs_root: Path
    dashboard: Any            # DashboardEmitter
    cost_ledger: Any          # RunCostLedger
    llm_client: Any           # LlmClient protocol: .complete(*, system, user) -> str
    provider: str             # "anthropic" | "openai"
    model: str
    runtime: Any = None       # AgentRuntime — only implement_baseline uses it
    workspace_service: Any = None
    workspace_id: str | None = None
