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

    def __init__(self, *, default_model: str | None = None) -> None:
        """``default_model`` is used when an ``AgentRuntimeSpec`` carries no
        model of its own — it lets the caller pin a cost tier (e.g. Sonnet for
        a heavy code-writing agent) without every agent spec naming a model.

        Auth is independent of the model: the claude-agent-sdk resolves
        ``ANTHROPIC_API_KEY`` (production API mode) with priority, else the
        Claude Code subscription's OAuth login — so the same runtime serves
        both deployment modes.
        """
        self._default_model = default_model

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

        mcp_servers, mcp_tool_extensions = _resolve_mcp_servers()

        sub_agents = {
            sub_agent.name: AgentDefinition(
                description=sub_agent.description or sub_agent.instructions[:200],
                prompt=_with_guard_prompt(sub_agent.instructions, sub_agent),
                tools=_tools_for_sub_agent(sub_agent, mcp_tool_extensions),
                model=sub_agent.model or None,
                maxTurns=sub_agent.max_turns,
                permissionMode=sub_agent.permission_mode,
            )
            for sub_agent in agent.sub_agents
        }

        options = ClaudeAgentOptions(
            model=agent.model or self._default_model or None,
            permission_mode=agent.permission_mode,
            max_turns=agent.max_turns,
            agents=sub_agents,
            cwd=str(agent.working_directory) if agent.working_directory else None,
            system_prompt=_with_guard_prompt(agent.instructions, agent),
            max_thinking_tokens=agent.thinking_budget_tokens,
            **({"mcp_servers": mcp_servers} if mcp_servers else {}),
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


# MCP server registration name used as the tool prefix the Claude SDK
# exposes (Claude Code MCP convention: ``mcp__<server>__<tool>``). The
# server name lives in this module — not in Settings — because changing
# it changes the tool identifiers visible to agents and would require
# coordinated changes elsewhere.
_APIFY_ARXIV_SERVER_NAME = "apify-arxiv"


def _resolve_mcp_servers() -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Build the mcp_servers config and per-sub-agent tool extensions.

    Returns:
        servers: dict suitable for ``ClaudeAgentOptions(mcp_servers=...)``,
            empty when no MCP integrations are configured.
        tool_extensions: maps sub-agent id -> list of MCP tool prefix
            strings to append to that sub-agent's tools list, so the SDK
            allows the agent to call the MCP tools.
    """
    from backend.config import get_settings

    settings = get_settings()
    servers: dict[str, Any] = {}
    extensions: dict[str, list[str]] = {}

    if settings.apify_api_token and settings.apify_arxiv_mcp_url:
        servers[_APIFY_ARXIV_SERVER_NAME] = {
            "type": "sse",
            "url": settings.apify_arxiv_mcp_url,
            "headers": {"Authorization": f"Bearer {settings.apify_api_token}"},
        }
        prefix = f"mcp__{_APIFY_ARXIV_SERVER_NAME}"
        enabled = [
            name.strip()
            for name in settings.apify_arxiv_enabled_agents.split(",")
            if name.strip()
        ]
        for agent_id in enabled:
            extensions.setdefault(agent_id, []).append(prefix)

    return servers, extensions


def _tools_for_sub_agent(
    sub_agent: Any,
    extensions: dict[str, list[str]],
) -> list[str] | None:
    """Compute the tools list for a sub-agent, merging MCP extras.

    Returns ``None`` when there are no tools at all (SDK convention for
    "agent inherits parent tools"), preserving prior behavior. When MCP
    tools apply, they are appended to the explicit registry list.
    """
    base = [tool.name for tool in sub_agent.tools]
    mcp_extras = extensions.get(sub_agent.name, [])
    merged = base + [name for name in mcp_extras if name not in base]
    return merged or None


__all__ = ["ClaudeAgentRuntime"]
