"""Execution profile and sandbox policy for agent pipelines."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

logger = logging.getLogger(__name__)


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
    brev = "brev"
    docker = "docker"
    local = "local"
    runpod = "runpod"
    simulate = "simulate"


DEFAULT_SANDBOX_MODE = SandboxMode.runpod


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
    # Maximum number of agent invocations to run concurrently (e.g. parallel
    # improvement paths).  Bounded by Claude subscription concurrency limits.
    max_concurrent_agents: int = 2
    max_improvement_rounds: int = 1
    improvement_convergence_pct: float = 1.0
    max_gate_retries: int = 0
    adaptive_selection: bool = False
    hypothesis_pool_multiplier: float = 1.0
    sandbox_network_disabled: bool = True
    sandbox_memory_limit: str = "4g"
    sandbox_cpus: float = 2.0
    sandbox_platform: str | None = None
    gpu_mode: GpuMode = GpuMode.auto
    sandbox_environment: dict[str, str] | None = None
    # Lane Q — minimize-compute mode. When True, the agent prompt gets a
    # "reproduce the CLAIM, not the recipe" block that swaps slow paper
    # recipes (SGD+linear-decay-from-10 × 3000 epochs) for modern fast
    # equivalents (Adam@lr=0.001 × 200-500 epochs) and annotates every
    # substitution in scope.declared_reductions so the scope-adjusted
    # rubric scores the metric claim, not the recipe-step count.
    minimize_compute: bool = False

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
        minimize_compute: bool = False,
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
                max_concurrent_agents=4,
                max_improvement_rounds=4,
                improvement_convergence_pct=0.5,
                max_gate_retries=2,
                adaptive_selection=True,
                hypothesis_pool_multiplier=2.0,
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
                max_improvement_rounds=2,
                improvement_convergence_pct=1.0,
                max_gate_retries=1,
                adaptive_selection=False,
                hypothesis_pool_multiplier=1.0,
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
            max_concurrent_agents=base.max_concurrent_agents,
            max_improvement_rounds=base.max_improvement_rounds,
            improvement_convergence_pct=base.improvement_convergence_pct,
            max_gate_retries=base.max_gate_retries,
            adaptive_selection=base.adaptive_selection,
            hypothesis_pool_multiplier=base.hypothesis_pool_multiplier,
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
            minimize_compute=minimize_compute,
        )


@lru_cache(maxsize=1)
def _is_wsl() -> bool:
    """Detect WSL — checks /proc/version for the Microsoft signature.

    Returns True on WSL1/WSL2; False on native Linux, macOS, Windows.
    Cached because this never changes within a process.
    """
    try:
        with open("/proc/version", "r", encoding="utf-8") as f:
            content = f.read().lower()
        return "microsoft" in content or "wsl" in content
    except (OSError, FileNotFoundError):
        return False


@lru_cache(maxsize=1)
def _docker_reachable() -> bool:
    """Best-effort: is the `docker` binary on PATH AND does `docker info` succeed?

    Used only as a tiebreaker for sandbox=auto. Failure means we shouldn't
    silently pick docker. False positive (docker reachable but the daemon
    rejects later) is acceptable — the explicit docker preflight will catch it.
    """
    if not shutil.which("docker"):
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def resolve_sandbox_mode(
    requested: SandboxMode | str,
    *,
    pipeline_mode: str,
) -> SandboxMode:
    """Resolve auto sandbox policy from the high-level pipeline mode.

    Resolution order for "auto":
      1. REPROLAB_FORCE_SANDBOX env override wins (applied upstream at the HTTP
         layer, but also checked here for CLI / direct callers).
      2. On WSL where docker is NOT verifiably reachable → "local"
         (avoids Docker Desktop dependency for local dev).
      3. Otherwise → DEFAULT_SANDBOX_MODE.

    Explicit non-auto values are returned as-is; the docker preflight is the
    right gate when the user has consciously chosen docker.
    """
    # Allow a direct env override for callers that bypass the HTTP layer
    # (e.g. the CLI).  REPROLAB_FORCE_SANDBOX is the canonical knob.
    force = os.environ.get("REPROLAB_FORCE_SANDBOX", "").strip()
    if force:
        return SandboxMode(force)

    mode = SandboxMode(requested)
    if mode is not SandboxMode.auto:
        return mode

    # WSL safety: docker might not be wired up via Docker Desktop; prefer
    # local unless we can verify the daemon is reachable.
    if _is_wsl() and not _docker_reachable():
        logger.info(
            "resolve_sandbox_mode: WSL detected without reachable docker — "
            "preferring 'local' (set REPROLAB_DEFAULT_SANDBOX=docker to override)"
        )
        return SandboxMode.local

    return DEFAULT_SANDBOX_MODE


def ensure_sandbox_mode_available(mode: SandboxMode | str) -> None:
    """Fail fast when a selected sandbox backend is unavailable locally."""
    resolved = SandboxMode(mode)
    if resolved is SandboxMode.auto:
        resolved = DEFAULT_SANDBOX_MODE
    if resolved is SandboxMode.docker:
        from backend.services.runtime import ensure_local_docker_available

        ensure_local_docker_available()
    elif resolved is SandboxMode.brev:
        from backend.services.runtime import ensure_brev_available

        ensure_brev_available()
    elif resolved is SandboxMode.runpod:
        from backend.services.runtime import ensure_runpod_available

        ensure_runpod_available()


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
    "DEFAULT_SANDBOX_MODE",
    "GpuMode",
    "SandboxMode",
    "_docker_reachable",
    "_is_wsl",
    "ensure_sandbox_mode_available",
    "resolve_sandbox_mode",
]
