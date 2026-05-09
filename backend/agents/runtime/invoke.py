"""Shared helpers for one-shot agent runtime invocations."""

from __future__ import annotations

from pathlib import Path

from backend.agents.registry import AGENT_REGISTRY
from backend.agents.runtime.base import AgentRuntime, ProviderName, StreamText
from backend.agents.runtime.factory import make_runtime


async def collect_agent_text(
    agent_id: str,
    prompt: str,
    *,
    project_dir: Path,
    model: str | None = None,
    provider: ProviderName | str | None = None,
    runtime: AgentRuntime | None = None,
    max_turns: int | None = None,
) -> str:
    """Run one agent and return concatenated text output."""
    selected_runtime = runtime or make_runtime(provider)
    spec = AGENT_REGISTRY[agent_id].to_runtime_spec(
        selected_runtime.provider_name,
        model_override=model,
        working_directory=project_dir,
        max_turns=max_turns,
    )
    collected: list[str] = []
    async for event in selected_runtime.run_agent(agent=spec, user_input=prompt):
        if isinstance(event, StreamText):
            collected.append(event.text)
    return "\n".join(collected)


__all__ = ["collect_agent_text"]
