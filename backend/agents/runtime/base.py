"""Provider-agnostic agent runtime contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Literal, Protocol


ProviderName = Literal["anthropic", "openai"]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentRuntimeSpec:
    name: str
    instructions: str
    model: str
    description: str = ""
    tools: tuple[ToolSpec, ...] = ()
    sub_agents: tuple["AgentRuntimeSpec", ...] = ()
    max_turns: int | None = None
    thinking_budget_tokens: int | None = None
    cache_static_blocks: bool = True
    permission_mode: str = "bypassPermissions"
    working_directory: Path | None = None


@dataclass(frozen=True)
class StreamText:
    text: str


@dataclass(frozen=True)
class StreamToolCall:
    tool_id: str
    tool_name: str
    tool_input: dict[str, Any]


@dataclass(frozen=True)
class StreamUsage:
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    reasoning_tokens: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }


StreamEvent = StreamText | StreamToolCall | StreamUsage


class ProviderConfigurationError(RuntimeError):
    def __init__(self, *, provider: str, reason: str) -> None:
        super().__init__(f"Provider {provider!r} is not configured: {reason}")
        self.provider = provider
        self.reason = reason


class ProviderFeatureUnsupported(RuntimeError):
    def __init__(self, *, provider: str, feature_name: str) -> None:
        super().__init__(f"Provider {provider!r} does not support {feature_name!r}")
        self.provider = provider
        self.feature_name = feature_name


class AgentRuntime(Protocol):
    @property
    def provider_name(self) -> ProviderName: ...

    async def run_agent(
        self,
        *,
        agent: AgentRuntimeSpec,
        user_input: str,
    ) -> AsyncIterator[StreamEvent]: ...


__all__ = [
    "AgentRuntime",
    "AgentRuntimeSpec",
    "ProviderConfigurationError",
    "ProviderFeatureUnsupported",
    "ProviderName",
    "StreamEvent",
    "StreamText",
    "StreamToolCall",
    "StreamUsage",
    "ToolSpec",
]
