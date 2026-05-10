"""Claude Agent SDK adapter for the provider-agnostic agent runtime."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from backend.agents.runtime.base import (
    AgentRuntimeSpec,
    ProviderConfigurationError,
    ProviderName,
    RuntimeGuardViolation,
    StreamEvent,
    StreamText,
    StreamToolCall,
    StreamUsage,
)
from backend.agents.telemetry import coerce_usage


class ClaudeAgentRuntime:
    """AgentRuntime implementation backed by ``claude-agent-sdk``."""

    @property
    def provider_name(self) -> ProviderName:
        return "anthropic"

    async def run_agent(
        self,
        *,
        agent: AgentRuntimeSpec,
        user_input: str,
    ) -> AsyncIterator[StreamEvent]:
        try:
            from claude_agent_sdk import (
                AgentDefinition,
                AssistantMessage,
                ClaudeAgentOptions,
                ResultMessage,
                ToolUseBlock,
                query,
            )
        except ImportError as exc:  # pragma: no cover - depends on local install
            raise ProviderConfigurationError(
                provider=self.provider_name,
                reason="claude-agent-sdk is not installed",
            ) from exc

        sub_agents = {
            sub_agent.name: AgentDefinition(
                description=sub_agent.description or sub_agent.instructions[:200],
                prompt=_with_guard_prompt(sub_agent.instructions, sub_agent),
                tools=[tool.name for tool in sub_agent.tools] or None,
                model=sub_agent.model or None,
                maxTurns=sub_agent.max_turns,
                permissionMode=sub_agent.permission_mode,
            )
            for sub_agent in agent.sub_agents
        }

        options = ClaudeAgentOptions(
            model=agent.model or None,
            permission_mode=agent.permission_mode,
            max_turns=agent.max_turns,
            agents=sub_agents,
            cwd=str(agent.working_directory) if agent.working_directory else None,
            system_prompt=_with_guard_prompt(agent.instructions, agent),
            max_thinking_tokens=agent.thinking_budget_tokens,
        )

        async for message in query(prompt=user_input, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    text = getattr(block, "text", "")
                    if text:
                        yield StreamText(str(text))
                    elif isinstance(block, ToolUseBlock):
                        tool_input = _as_dict(getattr(block, "input", None))
                        blocked = agent.guard.find_blocked_term(
                            json.dumps(tool_input, sort_keys=True)
                        )
                        if blocked is not None:
                            raise RuntimeGuardViolation(
                                f"Claude tool call references blocked PaperBench resource: {blocked}"
                            )
                        yield StreamToolCall(
                            tool_id=str(getattr(block, "id", "")),
                            tool_name=str(getattr(block, "name", "")),
                            tool_input=tool_input,
                        )
            elif isinstance(message, ResultMessage):
                usage = coerce_usage(getattr(message, "usage", None))
                yield StreamUsage(
                    input_tokens=_int_value(usage, "input_tokens"),
                    output_tokens=_int_value(usage, "output_tokens"),
                    cache_read_input_tokens=_int_value(usage, "cache_read_input_tokens"),
                    cache_creation_input_tokens=_int_value(
                        usage, "cache_creation_input_tokens"
                    ),
                )


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {"value": dumped}
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {"value": value}


def _int_value(data: dict[str, Any], key: str) -> int:
    value = data.get(key, 0)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _with_guard_prompt(instructions: str, agent: AgentRuntimeSpec) -> str:
    if not agent.guard.blocked_terms:
        return instructions
    blocked = ", ".join(agent.guard.blocked_terms)
    return (
        instructions
        + "\n\nRuntime guardrail: do not access, fetch, clone, download, or copy "
        + f"from these blocked PaperBench resources: {blocked}."
    )


__all__ = ["ClaudeAgentRuntime"]
