from backend.agents.execution import (
    GpuMode,
    ExecutionMode,
    ExecutionProfile,
    SandboxMode,
    resolve_sandbox_mode,
)


def test_execution_profile_efficient_caps_at_30_turns_and_80_tool_calls() -> None:
    """Efficient mode bounds runaway agents with three governors:
      * 30 turns (60 for heavy code-writing agents)
      * 80 tool calls per agent
      * 20 minute wall-clock cap

    Hitting any of these raises AgentLimitExceeded with partial output
    preserved (see learn.md 2026-05-09)."""

    profile = ExecutionProfile.from_mode("efficient")

    assert profile.mode is ExecutionMode.efficient
    assert profile.gpu_mode is GpuMode.auto
    assert profile.max_turns_per_agent == 30
    assert profile.heavy_agent_max_turns == 60
    assert profile.max_tool_calls_per_agent == 80
    assert profile.agent_wall_clock_seconds == 1200
    assert profile.command_timeout_seconds == 3600
    assert profile.sandbox_network_disabled is True


def test_execution_profile_max_uncaps_per_call_governs_only_by_wall_clock() -> None:
    """Max mode is explicit opt-in to uncapped per-invocation budgets, but
    keeps a 1 hour wall-clock per agent so a stuck run still terminates."""

    profile = ExecutionProfile.from_mode(
        "max",
        sandbox_network_disabled=False,
        sandbox_platform="linux/amd64",
    )

    assert profile.mode is ExecutionMode.max
    assert profile.max_turns_per_agent is None
    assert profile.heavy_agent_max_turns is None
    assert profile.max_tool_calls_per_agent is None
    assert profile.agent_wall_clock_seconds == 3600
    assert profile.command_timeout_seconds == 7200
    assert profile.sandbox_network_disabled is False
    assert profile.sandbox_platform == "linux/amd64"


def test_execution_profile_gpu_modes_set_resource_intent() -> None:
    off = ExecutionProfile.from_mode("efficient", gpu_mode="off")
    prefer = ExecutionProfile.from_mode("efficient", gpu_mode="prefer")
    max_gpu = ExecutionProfile.from_mode("max", gpu_mode="max")

    assert off.gpu_mode is GpuMode.off
    assert off.sandbox_environment["REPROLAB_GPU_MODE"] == "off"
    assert prefer.gpu_mode is GpuMode.prefer
    assert prefer.sandbox_environment["REPROLAB_GPU_MODE"] == "prefer"
    assert prefer.sandbox_environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert max_gpu.gpu_mode is GpuMode.max
    assert max_gpu.sandbox_cpus >= prefer.sandbox_cpus
    assert max_gpu.sandbox_environment["REPROLAB_GPU_MODE"] == "max"


def test_sandbox_mode_auto_defaults_to_docker_for_all_user_modes() -> None:
    assert resolve_sandbox_mode("auto", pipeline_mode="sdk") is SandboxMode.docker
    assert resolve_sandbox_mode("auto", pipeline_mode="offline") is SandboxMode.docker


def test_sandbox_mode_preserves_explicit_simulation_request() -> None:
    assert resolve_sandbox_mode("simulate", pipeline_mode="sdk") is SandboxMode.simulate


def test_sandbox_mode_preserves_explicit_runpod_request() -> None:
    assert resolve_sandbox_mode("runpod", pipeline_mode="sdk") is SandboxMode.runpod
