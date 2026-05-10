"""Execution profile and sandbox policy for agent pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ExecutionMode(str, Enum):
    """User-facing speed/quality profile."""

    efficient = "efficient"
    max = "max"


class GpuMode(str, Enum):
    """User-facing GPU acceleration policy."""

    off = "off"
    auto = "auto"
    prefer = "prefer"
    max = "max"


class SandboxMode(str, Enum):
    """Experiment execution backend policy."""

    auto = "auto"
    docker = "docker"
    local = "local"
    runpod = "runpod"
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
    max_tool_calls_per_agent: int | None
    # Hard upper bound on wall-clock time for ONE agent invocation. Catches
    # runs that get stuck thrashing on a small problem (e.g. an infinite
    # tool-call loop the SDK doesn't break) without relying on the user
    # to notice. Enforced via asyncio.timeout in the orchestrator.
    agent_wall_clock_seconds: int
    command_timeout_seconds: int
    sandbox_network_disabled: bool = True
    sandbox_memory_limit: str = "4g"
    sandbox_cpus: float = 2.0
    sandbox_platform: str | None = None
    gpu_mode: GpuMode = GpuMode.auto
    sandbox_environment: dict[str, str] | None = None

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
        gpu_mode: GpuMode | str = GpuMode.auto,
    ) -> "ExecutionProfile":
        resolved_mode = ExecutionMode(mode)
        resolved_gpu_mode = GpuMode(gpu_mode)
        if resolved_mode is ExecutionMode.max:
            base = cls(
                mode=resolved_mode,
                # No turn / tool-call cap. The SDK default of 15 turns is
                # far too low for any non-trivial paper. ``max`` mode is
                # explicit opt-in: bound the run with agent_wall_clock_seconds
                # and the agent's submit-when-done contract instead.
                max_turns_per_agent=None,
                heavy_agent_max_turns=None,
                max_tool_calls_per_agent=None,
                agent_wall_clock_seconds=3600,
                command_timeout_seconds=7200,
                sandbox_network_disabled=True,
                sandbox_memory_limit="12g" if resolved_gpu_mode is GpuMode.max else "8g",
                sandbox_cpus=6.0 if resolved_gpu_mode is GpuMode.max else 4.0,
                gpu_mode=resolved_gpu_mode,
                sandbox_environment=_gpu_environment(resolved_gpu_mode),
            )
        else:
            base = cls(
                mode=resolved_mode,
                # ``efficient`` is the default profile and bounds runaway
                # agents with three independent governors:
                #   * 30 turns / 60 for "heavy" code-writing agents
                #   * 80 tool calls / agent (heavy agents share the same cap;
                #     overrun raises AgentLimitExceeded for clean handling)
                #   * 20 minute wall-clock per agent invocation
                # Hitting any of these raises a typed AgentLimitExceeded with
                # partial output preserved, so the orchestrator can decide
                # whether to fail loudly or continue.
                max_turns_per_agent=30,
                heavy_agent_max_turns=60,
                max_tool_calls_per_agent=80,
                agent_wall_clock_seconds=1200,
                command_timeout_seconds=3600,
                sandbox_network_disabled=True,
                sandbox_memory_limit=(
                    "6g"
                    if resolved_gpu_mode in (GpuMode.prefer, GpuMode.max)
                    else "4g"
                ),
                sandbox_cpus=(
                    3.0
                    if resolved_gpu_mode in (GpuMode.prefer, GpuMode.max)
                    else 2.0
                ),
                gpu_mode=resolved_gpu_mode,
                sandbox_environment=_gpu_environment(resolved_gpu_mode),
            )

        return cls(
            mode=base.mode,
            max_turns_per_agent=base.max_turns_per_agent,
            heavy_agent_max_turns=base.heavy_agent_max_turns,
            max_tool_calls_per_agent=base.max_tool_calls_per_agent,
            agent_wall_clock_seconds=base.agent_wall_clock_seconds,
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
            gpu_mode=resolved_gpu_mode,
            sandbox_environment=base.sandbox_environment,
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


def ensure_sandbox_mode_available(mode: SandboxMode | str) -> None:
    """Fail fast when a selected sandbox backend is unavailable locally."""
    resolved = SandboxMode(mode)
    if resolved is SandboxMode.docker:
        from backend.services.runtime import ensure_local_docker_available

        ensure_local_docker_available()


def _gpu_environment(mode: GpuMode) -> dict[str, str]:
    env = {"REPROLAB_GPU_MODE": mode.value}
    if mode is GpuMode.off:
        env["CUDA_VISIBLE_DEVICES"] = ""
        return env
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    return env


__all__ = [
    "ExecutionMode",
    "ExecutionProfile",
    "GpuMode",
    "SandboxMode",
    "ensure_sandbox_mode_available",
    "resolve_sandbox_mode",
]
