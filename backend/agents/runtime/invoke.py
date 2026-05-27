"""Shared helpers for one-shot agent runtime invocations."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from backend.agents.registry import AGENT_REGISTRY
from backend.agents.runtime.base import AgentRuntime, ProviderName, StreamText, StreamToolCall
from backend.agents.runtime.factory import make_runtime
from backend.agents.runtime.sdk_isolation import run_isolated
from backend.agents.worker_reports import (
    append_worker_report_instruction,
    build_worker_report,
    write_worker_report,
)


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
    started_at = datetime.now(timezone.utc).isoformat()
    spec = AGENT_REGISTRY[agent_id].to_runtime_spec(
        selected_runtime.provider_name,
        model_override=model,
        working_directory=project_dir,
        max_turns=max_turns,
    )
    collected: list[str] = []
    tool_calls: list[dict[str, object]] = []

    async def _do_sdk_call() -> tuple[list[str], list[dict[str, object]]]:
        # Inner coroutine so run_isolated can thread-isolate the SDK call and
        # contain its aclose race within the worker's event loop.
        _inner_collected: list[str] = []
        _inner_tool_calls: list[dict[str, object]] = []
        async for event in selected_runtime.run_agent(
            agent=spec,
            user_input=append_worker_report_instruction(prompt),
        ):
            if isinstance(event, StreamText):
                _inner_collected.append(event.text)
            elif isinstance(event, StreamToolCall):
                _inner_tool_calls.append({
                    "tool_id": event.tool_id,
                    "tool_name": event.tool_name,
                    "tool_input": event.tool_input,
                })
        return _inner_collected, _inner_tool_calls

    try:
        collected, tool_calls = await run_isolated(_do_sdk_call)
    except Exception as exc:
        raw_text = "\n".join(collected)
        report = build_worker_report(
            agent_id=agent_id,
            project_dir=project_dir,
            model=spec.model,
            provider=selected_runtime.provider_name,
            status="failed",
            started_at=started_at,
            finished_at=datetime.now(timezone.utc).isoformat(),
            raw_text=raw_text,
            tool_calls=tool_calls,
            error=str(exc),
        )
        write_worker_report(project_dir, report)
        raise

    raw_text = "\n".join(collected)
    report = build_worker_report(
        agent_id=agent_id,
        project_dir=project_dir,
        model=spec.model,
        provider=selected_runtime.provider_name,
        status="completed",
        started_at=started_at,
        finished_at=datetime.now(timezone.utc).isoformat(),
        raw_text=raw_text,
        tool_calls=tool_calls,
    )
    write_worker_report(project_dir, report)
    return raw_text


__all__ = ["collect_agent_text"]
