from backend.agents.execution import (
    ExecutionMode,
    ExecutionProfile,
    SandboxMode,
    resolve_sandbox_mode,
)


def test_execution_profile_efficient_uses_uncapped_agent_turns() -> None:
    profile = ExecutionProfile.from_mode("efficient")

    assert profile.mode is ExecutionMode.efficient
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


def test_sandbox_mode_auto_defaults_to_docker_for_all_user_modes() -> None:
    assert resolve_sandbox_mode("auto", pipeline_mode="sdk") is SandboxMode.docker
    assert resolve_sandbox_mode("auto", pipeline_mode="offline") is SandboxMode.docker


def test_sandbox_mode_preserves_explicit_simulation_request() -> None:
    assert resolve_sandbox_mode("simulate", pipeline_mode="sdk") is SandboxMode.simulate
