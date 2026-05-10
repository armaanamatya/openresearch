from backend.agents.execution import (
    GpuMode,
    ExecutionMode,
    ExecutionProfile,
    SandboxMode,
    resolve_sandbox_mode,
)


def test_execution_profile_efficient_uses_uncapped_agent_turns() -> None:
    profile = ExecutionProfile.from_mode("efficient")

    assert profile.mode is ExecutionMode.efficient
    assert profile.gpu_mode is GpuMode.auto
    assert profile.max_turns_per_agent is None
    assert profile.heavy_agent_max_turns is None
    assert profile.command_timeout_seconds == 3600
    assert profile.sandbox_network_disabled is True


def test_execution_profile_max_raises_bounded_budgets() -> None:
    profile = ExecutionProfile.from_mode(
        "max",
        sandbox_network_disabled=False,
        sandbox_platform="linux/amd64",
    )

    assert profile.mode is ExecutionMode.max
    assert profile.max_turns_per_agent is None
    assert profile.heavy_agent_max_turns is None
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
