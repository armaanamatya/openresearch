"""Execution profile and sandbox policy for agent pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ExecutionMode(str, Enum):
    """User-facing speed/quality profile."""

    efficient = "efficient"
    max = "max"


class SandboxMode(str, Enum):
    """Experiment execution backend policy."""

    auto = "auto"
    docker = "docker"
    local = "local"
    simulate = "simulate"


@dataclass(frozen=True)
class ExecutionProfile:
    """Resolved pipeline execution settings.

    The efficient profile intentionally preserves current quality defaults
    while making the mode explicit. The max profile raises bounded budgets for
    slower, higher-confidence runs.
    """

    mode: ExecutionMode
    max_turns_per_agent: int | None
    heavy_agent_max_turns: int | None
    command_timeout_seconds: int
    sandbox_network_disabled: bool = True
    sandbox_memory_limit: str = "4g"
    sandbox_cpus: float = 2.0
    sandbox_platform: str | None = None

    @classmethod
    def from_mode(
        cls,
        mode: ExecutionMode | str,
        *,
        command_timeout_seconds: int | None = None,
        sandbox_network_disabled: bool | None = None,
        sandbox_memory_limit: str | None = None,
        sandbox_cpus: float | None = None,
        sandbox_platform: str | None = None,
    ) -> "ExecutionProfile":
        resolved_mode = ExecutionMode(mode)
        if resolved_mode is ExecutionMode.max:
            base = cls(
                mode=resolved_mode,
                max_turns_per_agent=None,
                heavy_agent_max_turns=None,
                command_timeout_seconds=7200,
                sandbox_network_disabled=True,
                sandbox_memory_limit="8g",
                sandbox_cpus=4.0,
            )
        else:
            base = cls(
                mode=resolved_mode,
                max_turns_per_agent=None,
                heavy_agent_max_turns=None,
                command_timeout_seconds=3600,
                sandbox_network_disabled=True,
                sandbox_memory_limit="4g",
                sandbox_cpus=2.0,
            )

        return cls(
            mode=base.mode,
            max_turns_per_agent=base.max_turns_per_agent,
            heavy_agent_max_turns=base.heavy_agent_max_turns,
            command_timeout_seconds=(
                command_timeout_seconds
                if command_timeout_seconds is not None
                else base.command_timeout_seconds
            ),
            sandbox_network_disabled=(
                sandbox_network_disabled
                if sandbox_network_disabled is not None
                else base.sandbox_network_disabled
            ),
            sandbox_memory_limit=sandbox_memory_limit or base.sandbox_memory_limit,
            sandbox_cpus=sandbox_cpus if sandbox_cpus is not None else base.sandbox_cpus,
            sandbox_platform=sandbox_platform,
        )


def resolve_sandbox_mode(
    requested: SandboxMode | str,
    *,
    pipeline_mode: str,
) -> SandboxMode:
    """Resolve auto sandbox policy from the high-level pipeline mode.

    Auto is intentionally Docker-first for user-triggered runs. Local host
    execution is never selected implicitly; callers must request it explicitly.
    """

    mode = SandboxMode(requested)
    if mode is not SandboxMode.auto:
        return mode
    return SandboxMode.docker


__all__ = [
    "ExecutionMode",
    "ExecutionProfile",
    "SandboxMode",
    "resolve_sandbox_mode",
]
