"""RunContext — run-scoped dependencies threaded into every RLM primitive.

Phase 2 (issue #59). The root RLM model passes only slices/specs as primitive
arguments (the Algorithm-2 guard). Everything else a primitive needs — paths,
the event emitter, the cost ledger, the LLM client, the agent runtime — lives
here and is closed over by `backend.agents.rlm.binding.build_custom_tools`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class RunContext:
    """Everything a primitive needs that the root model does not pass.

    `llm_client` is the synchronous `LlmClient` protocol from
    `backend/services/context/workspace/tools/rlm_query.py` — `.complete(*,
    system, user) -> str`. `runtime` is an `AgentRuntime`; only
    `implement_baseline` needs it, so it defaults to None. `agent_model` is the
    model a primitive-spawned agent runs on (the code-writing agent in
    `implement_baseline`); `None` falls back to the agent registry's default.

    `deadline_utc` is set by `run.py` from the wall-clock budget at run start
    (M-DEADLINE, WS-H Batch P).  Primitives call `remaining_s()` to get seconds
    left; `None` means no wall-clock budget was configured.
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
    agent_model: str | None = None  # model for primitive-spawned agents (implement_baseline)
    workspace_service: Any = None
    workspace_id: str | None = None
    deadline_utc: datetime | None = field(default=None)  # M-DEADLINE — set by run.py

    def remaining_s(self) -> float | None:
        """Seconds until `deadline_utc`, clamped ≥ 0; None if no deadline set.

        Always returns a timezone-aware comparison: if `deadline_utc` is naive
        it is treated as UTC.
        """
        if self.deadline_utc is None:
            return None
        dl = self.deadline_utc
        if dl.tzinfo is None:
            dl = dl.replace(tzinfo=timezone.utc)
        return max(0.0, (dl - datetime.now(tz=timezone.utc)).total_seconds())
