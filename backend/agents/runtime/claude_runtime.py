"""Claude Agent SDK adapter for the provider-agnostic agent runtime."""

from __future__ import annotations

import json
import logging
import warnings
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)

# Suppress the harmless "aclose(): asynchronous generator is already running"
# RuntimeWarning that the claude-agent-sdk emits on every subprocess teardown.
# The warning fires from CPython's async-generator finalizer (Objects/genobject.c)
# when asyncio.shutdown_asyncgens() or GC tries to close the SDK's triple-nested
# async generator (InternalClient._process_query_inner) while it is still marked
# as running — a known SDK quirk documented in CLAUDE.md (search "aclose") and
# analysed in docs/superpowers/specs/2026-05-22-sdk-aclose-investigation.md.
# Option B (module-level filter) is used here because the warning fires in a GC
# finalizer / asyncio.shutdown_asyncgens() path that executes AFTER our `async for`
# loop exits, so a scoped `warnings.catch_warnings()` context at the call site would
# already be closed by the time the warning is emitted.  The filter is intentionally
# message-specific (regex match) so unrelated RuntimeWarnings are unaffected.
warnings.filterwarnings(
    "ignore",
    message=r"aclose\(\).*asynchronous generator is already running",
    category=RuntimeWarning,
)

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

        mcp_servers, mcp_tool_extensions = _resolve_mcp_servers()

        sub_agents = {
            sub_agent.name: AgentDefinition(
                description=sub_agent.description or sub_agent.instructions[:200],
                prompt=_with_guard_prompt(sub_agent.instructions, sub_agent),
                tools=_tools_for_agent(sub_agent, mcp_tool_extensions),
                model=sub_agent.model or None,
                maxTurns=sub_agent.max_turns,
                permissionMode=sub_agent.permission_mode,
            )
            for sub_agent in agent.sub_agents
        }

        options = ClaudeAgentOptions(
            agents=sub_agents,
            **_agent_options_kwargs(agent, mcp_servers, mcp_tool_extensions),
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
                # FIX-E (2026-05-30): surface the SDK's own error metadata. On a
                # transport wedge the SDK yields an is_error ResultMessage (carrying
                # api_error_status, e.g. 429/500/529) just before raising ProcessError.
                # Previously only usage was read, so the HTTP status was invisible and
                # diagnosing FM-001 required spelunking ~/.claude transcripts. Log it.
                if getattr(message, "is_error", False):
                    logger.warning(
                        "claude_runtime: SDK error ResultMessage subtype=%s "
                        "api_error_status=%s num_turns=%s",
                        getattr(message, "subtype", None),
                        getattr(message, "api_error_status", None),
                        getattr(message, "num_turns", None),
                    )
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


def _tools_for_agent(
    agent: Any,
    extensions: dict[str, list[str]],
) -> list[str] | None:
    """Compute the explicit tools list for an agent (root OR sub-agent), merging
    MCP extras. Shared so the Claude **root** enforces ``allowed_tools`` exactly
    like sub-agents already did — closing invariant 3 (the root previously
    inherited ALL default SDK tools while OpenAI restricted both providers).

    Returns ``None`` when there are no tools at all (SDK convention for "inherit
    all default tools"). The registry fail-closed guard (``to_runtime_spec``)
    ensures registered agents are never empty, so in practice this is non-None
    and the root is always restricted. MCP tools are appended to the registry list.
    """
    base = [tool.name for tool in agent.tools]
    mcp_extras = extensions.get(agent.name, [])
    merged = base + [name for name in mcp_extras if name not in base]
    return merged or None


def _hermetic_enabled() -> bool:
    """``REPROLAB_SDK_HERMETIC`` (default true). When on, the SDK runs hermetically
    — no ambient ``CLAUDE.md`` / ``.claude/settings.json`` / discovered-MCP
    leakage (``setting_sources=[]`` + ``strict_mcp_config=True``). Disable only
    for local debugging. Note this gates ONLY the hermetic config; the
    ``allowed_tools`` restriction is always on (a hatch would re-open invariant 3).
    """
    import os

    return os.environ.get("REPROLAB_SDK_HERMETIC", "true").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _agent_options_kwargs(
    agent: AgentRuntimeSpec,
    mcp_servers: dict[str, Any],
    mcp_tool_extensions: dict[str, list[str]],
) -> dict[str, Any]:
    """Build the ``ClaudeAgentOptions`` kwargs for the root agent (Gap A).

    Extracted as a pure function so the parity / permission-mode / hermetic
    contract is directly testable without the SDK. ``agents=sub_agents`` is added
    by the caller (it needs the SDK ``AgentDefinition`` type).
    """
    kwargs: dict[str, Any] = {
        "model": agent.model or None,
        # KEEP bypassPermissions: headless runs have no approver; the real
        # controls are allowed_tools + sandbox + RuntimeGuard (invariant 4 targets
        # a future external-CLI shell-out, not the harness's own sub-agents).
        "permission_mode": agent.permission_mode,
        "max_turns": agent.max_turns,
        "cwd": str(agent.working_directory) if agent.working_directory else None,
        "system_prompt": _with_guard_prompt(agent.instructions, agent),
        "max_thinking_tokens": agent.thinking_budget_tokens,
        # Always explicit (even {}) so MCP config is deterministic + strict.
        "mcp_servers": mcp_servers,
    }
    # Restrict the root to its declared tools (MCP-merged). Always on — no hatch.
    allowed_tools = _tools_for_agent(agent, mcp_tool_extensions)
    if allowed_tools:
        kwargs["allowed_tools"] = allowed_tools
    # Hermetic isolation — gated by REPROLAB_SDK_HERMETIC (default true).
    if _hermetic_enabled():
        kwargs["setting_sources"] = []
        kwargs["strict_mcp_config"] = True
    return kwargs


__all__ = ["ClaudeAgentRuntime"]
