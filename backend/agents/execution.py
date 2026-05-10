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
                max_turns_per_agent=None,
                heavy_agent_max_turns=None,
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
                max_turns_per_agent=None,
                heavy_agent_max_turns=None,
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
    "resolve_sandbox_mode",
]
