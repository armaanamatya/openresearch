"""Agent registry — RLM-path ReproLab agent definitions.

Usage:
    from backend.agents.registry import get_agent_definitions
    opts = ClaudeAgentOptions(agents=get_agent_definitions())  # Anthropic adapter
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from backend.agents.prompts import (
    BASELINE_IMPLEMENTATION_PROMPT,
    IMPROVEMENT_ORCHESTRATOR_PROMPT,
    IMPROVEMENT_PATH_PROMPT,
    RUBRIC_VERIFIER_PROMPT,
)
from backend.agents.runtime.base import AgentRuntimeSpec, ProviderName, ToolSpec
from backend.config import get_settings


@dataclass(frozen=True)
class AgentSpec:
    """Internal registry entry for a ReproLab agent."""

    agent_id: str
    role: str
    description: str
    prompt: str
    tools: list[str] = field(default_factory=list)
    spawn_permissions: bool = False
    max_turns: int | None = None
    default_model_anthropic: str = ""
    default_model_openai: str = ""
    thinking_budget_tokens: int | None = None

    def to_runtime_spec(
        self,
        provider: ProviderName,
        *,
        model_override: str | None = None,
        max_turns: int | None = None,
        working_directory: Path | None = None,
        sub_agents: tuple[AgentRuntimeSpec, ...] = (),
    ) -> AgentRuntimeSpec:
        """Convert a registry entry into the provider-neutral runtime contract."""
        settings = get_settings()
        configured_model = _model_override_from_settings(
            self.agent_id,
            provider,
            settings.agent_provider_overrides,
        )
        model = (
            model_override
            or configured_model
            or _default_model_for_provider(self, provider)
        )
        return AgentRuntimeSpec(
            name=self.agent_id,
            description=self.description,
            instructions=self.prompt,
            model=model,
            tools=tuple(
                ToolSpec(
                    name=tool_name,
                    description=_TOOL_DESCRIPTIONS.get(tool_name, ""),
                )
                for tool_name in self.tools
                if tool_name != "Agent"
            ),
            sub_agents=sub_agents,
            max_turns=max_turns if max_turns is not None else self.max_turns,
            thinking_budget_tokens=self.thinking_budget_tokens,
            working_directory=working_directory,
        )


# ---------------------------------------------------------------------------
# The 14 PRD agents
# ---------------------------------------------------------------------------

AGENT_REGISTRY: dict[str, AgentSpec] = {
    # --- Builder agents (RLM path) ---
    "baseline-implementation": AgentSpec(
        agent_id="baseline-implementation",
        role="builder",
        description="Implements or adapts the paper baseline inside a Docker sandbox.",
        prompt=BASELINE_IMPLEMENTATION_PROMPT,
        tools=["Read", "Write", "Edit", "Bash"],
        default_model_anthropic="claude-opus-4-7",
        default_model_openai="gpt-4o",
    ),
    # --- Verifier agents ---
    "rubric-verifier": AgentSpec(
        agent_id="rubric-verifier",
        role="verifier",
        description="Derives or loads a PaperBench-style rubric and scores the reproduction against it.",
        prompt=RUBRIC_VERIFIER_PROMPT,
        tools=["Read", "Bash"],
        default_model_openai="o4-mini",
    ),
    # --- Improvement agents ---
    "improvement-orchestrator": AgentSpec(
        agent_id="improvement-orchestrator",
        role="improvement",
        description="Selects N improvement hypotheses from evidence and launches path agents.",
        prompt=IMPROVEMENT_ORCHESTRATOR_PROMPT,
        tools=["Read", "Bash", "Agent"],
        spawn_permissions=True,
        default_model_openai="gpt-4o",
    ),
    "improvement-path": AgentSpec(
        agent_id="improvement-path",
        role="improvement",
        description="Executes one improvement hypothesis in an isolated branch and sandbox.",
        prompt=IMPROVEMENT_PATH_PROMPT,
        tools=["Read", "Write", "Edit", "Bash"],
        default_model_anthropic="claude-opus-4-7",
        default_model_openai="gpt-4o",
    ),
}


def get_agent_definitions() -> dict[str, dict]:
    """Convert registry to claude-agent-sdk AgentDefinition dicts.

    Returns a dict suitable for ``ClaudeAgentOptions(agents=...)``.
    """
    from claude_agent_sdk import AgentDefinition

    defs: dict[str, AgentDefinition] = {}
    for name, spec in AGENT_REGISTRY.items():
        defs[name] = AgentDefinition(
            description=spec.description,
            prompt=spec.prompt,
            tools=spec.tools if spec.tools else None,
            maxTurns=spec.max_turns,
        )
    return defs


def _default_model_for_provider(spec: AgentSpec, provider: ProviderName) -> str:
    settings = get_settings()
    if provider == "openai":
        return spec.default_model_openai or settings.openai_default_model
    return spec.default_model_anthropic or settings.anthropic_default_model


def _model_override_from_settings(
    agent_id: str,
    provider: ProviderName,
    overrides: dict[str, str],
) -> str:
    return overrides.get(f"{agent_id}.{provider}") or overrides.get(agent_id, "")


_TOOL_DESCRIPTIONS = {
    "Read": "Read files from the current project workspace.",
    "Write": "Write files into the current project workspace.",
    "Edit": "Apply exact text edits to workspace files.",
    "Bash": "Run shell commands from the current project workspace.",
    "WebSearch": "Search for external paper artifacts and references.",
    "WebFetch": "Fetch external paper artifact metadata or documentation.",
    "Agent": "Delegate to registered specialist agents.",
}
