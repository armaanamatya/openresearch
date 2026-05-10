"""Provider runtime wiring tests for the root orchestrator."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator

from backend.agents.orchestrator import PipelineState, ReproLabOrchestrator
from backend.agents.runtime.base import (
    AgentRuntimeSpec,
    ProviderName,
    StreamEvent,
    StreamText,
    StreamUsage,
)


class FakeRuntime:
    def __init__(
        self,
        provider_name: ProviderName = "openai",
        output: str = '{"core_contribution": "ok"}',
    ) -> None:
        self.provider_name = provider_name
        self.output = output
        self.agent: AgentRuntimeSpec | None = None
        self.user_input = ""

    async def run_agent(
        self,
        *,
        agent: AgentRuntimeSpec,
        user_input: str,
    ) -> AsyncIterator[StreamEvent]:
        self.agent = agent
        self.user_input = user_input
        yield StreamText(self.output)
        yield StreamUsage(input_tokens=1, output_tokens=2, reasoning_tokens=1)


class FailingRuntime(FakeRuntime):
    def __init__(
        self,
        provider_name: ProviderName,
        *,
        message: str,
        partial_text: str = "",
    ) -> None:
        super().__init__(provider_name)
        self.message = message
        self.partial_text = partial_text

    async def run_agent(
        self,
        *,
        agent: AgentRuntimeSpec,
        user_input: str,
    ) -> AsyncIterator[StreamEvent]:
        self.agent = agent
        self.user_input = user_input
        if self.partial_text:
            yield StreamText(self.partial_text)
        raise Exception(self.message)


class FakeOpenAiRuntime(FakeRuntime):
    def __init__(self) -> None:
        super().__init__("openai")


class ConcurrentImprovementRuntime(FakeRuntime):
    def __init__(self) -> None:
        super().__init__("openai")
        self.active_path_calls = 0
        self.max_active_path_calls = 0

    async def run_agent(
        self,
        *,
        agent: AgentRuntimeSpec,
        user_input: str,
    ) -> AsyncIterator[StreamEvent]:
        self.agent = agent
        self.user_input = user_input
        if agent.name == "improvement-orchestrator":
            yield StreamText(
                json.dumps(
                    {
                        "hypotheses": [
                            {
                                "path_id": "path_1",
                                "hypothesis": "Tune learning rate",
                                "rationale": "Learning rate is underexplored.",
                                "expected_outcome": "Higher reward",
                            },
                            {
                                "path_id": "path_2",
                                "hypothesis": "Tune entropy",
                                "rationale": "Entropy changes exploration.",
                                "expected_outcome": "More stable reward",
                            },
                        ]
                    }
                )
            )
            yield StreamUsage(input_tokens=1, output_tokens=2)
            return
        if agent.name == "improvement-path":
            self.active_path_calls += 1
            self.max_active_path_calls = max(
                self.max_active_path_calls,
                self.active_path_calls,
            )
            try:
                await asyncio.sleep(0.05)
                path_id = "path_1" if "path_1" in user_input else "path_2"
                yield StreamText(
                    json.dumps(
                        {
                            "path_id": path_id,
                            "hypothesis": f"{path_id} hypothesis",
                            "diff_summary": "changed config",
                            "success": True,
                        }
                    )
                )
                yield StreamUsage(input_tokens=1, output_tokens=2)
            finally:
                self.active_path_calls -= 1
            return
        yield StreamText("{}")


def test_orchestrator_builds_provider_specific_runtime_spec(tmp_path: Path) -> None:
    runtime = FakeOpenAiRuntime()
    orchestrator = ReproLabOrchestrator(
        "prj_runtime",
        tmp_path,
        runtime=runtime,
    )

    output = asyncio.run(
        orchestrator._invoke_agent(
            "paper-understanding",
            "Analyze.",
            cwd=tmp_path / "work",
            max_turns=6,
        )
    )

    assert output == '{"core_contribution": "ok"}'
    assert runtime.agent is not None
    assert runtime.agent.name == "paper-understanding"
    assert runtime.agent.model == "gpt-4o"
    assert runtime.agent.max_turns == 6
    assert runtime.agent.working_directory == tmp_path / "work"
    assert runtime.agent.sub_agents
    assert "Structured Output Contract" in runtime.user_input

    telemetry = tmp_path / "prj_runtime" / "agent_telemetry.jsonl"
    assert '"provider": "openai"' in telemetry.read_text()


def test_improvement_path_agents_run_with_bounded_concurrency(tmp_path: Path) -> None:
    runtime = ConcurrentImprovementRuntime()
    orchestrator = ReproLabOrchestrator(
        "prj_parallel_paths",
        tmp_path,
        runtime=runtime,
    )
    orchestrator._audit_step = lambda *args, **kwargs: None  # type: ignore[method-assign]
    orchestrator._enrich_workspace = lambda *args, **kwargs: None  # type: ignore[method-assign]

    state = PipelineState(project_id="prj_parallel_paths")
    state = asyncio.run(orchestrator.run_improvements(state, n_paths=2))

    assert runtime.max_active_path_calls == 2
    assert [result.path_id for result in state.path_results] == ["path_1", "path_2"]


def test_orchestrator_uses_efficient_default_caps_for_heavy_agents(tmp_path: Path) -> None:
    """Heavy agents (experiment-runner, baseline-implementation) get 60
    turns + 80 tool calls + 20 min wall-clock by default. Hitting any
    cap raises a typed AgentLimitExceeded — see learn.md 2026-05-09."""

    runtime = FakeOpenAiRuntime()
    orchestrator = ReproLabOrchestrator(
        "prj_capped",
        tmp_path,
        runtime=runtime,
    )

    asyncio.run(
        orchestrator._invoke_agent(
            "experiment-runner",
            "Run the experiment.",
            cwd=tmp_path / "work",
        )
    )

    assert runtime.agent is not None
    assert runtime.agent.max_turns == 60
    assert runtime.agent.guard.max_tool_calls == 80


def test_orchestrator_propagates_run_metadata_and_guard(tmp_path: Path) -> None:
    runtime = FakeOpenAiRuntime()
    orchestrator = ReproLabOrchestrator(
        "prj_guarded",
        tmp_path,
        runtime=runtime,
        seed=123,
        attempt_id="attempt-a",
        run_group_id="group-a",
        blacklist_terms=("https://github.com/BartekCupial/finetuning-RL-as-CL",),
    )

    asyncio.run(
        orchestrator._invoke_agent(
            "paper-understanding",
            "Analyze.",
            cwd=tmp_path / "work",
            max_turns=6,
        )
    )

    assert runtime.agent is not None
    # max_tool_calls_per_agent default: 80 in efficient mode (None in max).
    # See test_execution_modes.py for the cross-reference.
    assert runtime.agent.guard.max_tool_calls == 80
    assert runtime.agent.guard.find_blocked_term(
        "git clone https://github.com/BartekCupial/finetuning-RL-as-CL.git"
    )
    assert "Use random seed 123" in runtime.user_input
    assert "attempt_id=attempt-a" in runtime.user_input


def test_orchestrator_routes_supervisor_to_verification_runtime(tmp_path: Path) -> None:
    builder_runtime = FakeRuntime("openai")
    verification_runtime = FakeRuntime(
        "anthropic",
        '{"gate": "gate_2", "status": "verified", "verifier_scores": []}',
    )
    orchestrator = ReproLabOrchestrator(
        "prj_review_runtime",
        tmp_path,
        runtime=builder_runtime,
        verification_runtime=verification_runtime,
    )

    output = asyncio.run(
        orchestrator._invoke_agent(
            "supervisor-verifier",
            "Verify.",
            cwd=tmp_path / "work",
        )
    )

    assert output == '{"gate": "gate_2", "status": "verified", "verifier_scores": []}'
    assert builder_runtime.agent is None
    assert verification_runtime.agent is not None
    assert verification_runtime.agent.name == "supervisor-verifier"
    assert verification_runtime.agent.sub_agents

    telemetry = tmp_path / "prj_review_runtime" / "agent_telemetry.jsonl"
    assert '"provider": "anthropic"' in telemetry.read_text()


def test_orchestrator_falls_back_to_openai_when_claude_limit_is_hit(
    tmp_path: Path,
) -> None:
    claude_runtime = FailingRuntime(
        "anthropic",
        message="Claude Code returned an error result: success",
        partial_text="You've hit your limit - resets 9:40pm (America/Chicago)",
    )
    openai_runtime = FakeRuntime("openai", '{"core_contribution": "fallback ok"}')
    orchestrator = ReproLabOrchestrator(
        "prj_claude_limit_fallback",
        tmp_path,
        runtime=claude_runtime,
        claude_limit_fallback_runtime=openai_runtime,
    )

    output = asyncio.run(
        orchestrator._invoke_agent(
            "paper-understanding",
            "Analyze.",
            cwd=tmp_path / "work",
        )
    )

    assert output == '{"core_contribution": "fallback ok"}'
    assert claude_runtime.agent is not None
    assert openai_runtime.agent is not None
    assert openai_runtime.agent.name == "paper-understanding"

    telemetry = (tmp_path / "prj_claude_limit_fallback" / "agent_telemetry.jsonl").read_text()
    assert '"provider": "openai"' in telemetry
    assert '"success": false' in telemetry
    assert '"success": true' in telemetry


def test_orchestrator_does_not_fallback_for_non_limit_claude_errors(
    tmp_path: Path,
) -> None:
    claude_runtime = FailingRuntime("anthropic", message="network exploded")
    openai_runtime = FakeRuntime("openai", '{"core_contribution": "fallback ok"}')
    orchestrator = ReproLabOrchestrator(
        "prj_no_fallback",
        tmp_path,
        runtime=claude_runtime,
        claude_limit_fallback_runtime=openai_runtime,
    )

    try:
        asyncio.run(
            orchestrator._invoke_agent(
                "paper-understanding",
                "Analyze.",
                cwd=tmp_path / "work",
            )
        )
    except Exception as exc:
        assert "network exploded" in str(exc)
    else:
        raise AssertionError("Expected non-limit Claude errors to propagate")

    assert openai_runtime.agent is None


def test_orchestrator_converts_sdk_turn_cap_message_to_typed_exception(
    tmp_path: Path,
) -> None:
    """Regression: the SDK throws Exception('...Reached maximum number of
    turns (15)') instead of a typed error. The orchestrator now extracts
    the cap value via _TURN_LIMIT_RE and re-raises as AgentLimitExceeded
    so callers can react programmatically — see learn.md 2026-05-09."""

    from backend.agents.runtime.base import AgentLimitExceeded

    claude_runtime = FailingRuntime(
        "anthropic",
        message="Claude Code returned an error result: Reached maximum number of turns (30)",
        partial_text="Halfway through analyzing the paper",
    )
    orchestrator = ReproLabOrchestrator(
        "prj_turn_cap",
        tmp_path,
        runtime=claude_runtime,
    )

    try:
        asyncio.run(
            orchestrator._invoke_agent(
                "paper-understanding",
                "Analyze.",
                cwd=tmp_path / "work",
            )
        )
    except AgentLimitExceeded as exc:
        assert exc.kind == "turns"
        assert exc.limit_value == 30
        assert exc.agent_id == "paper-understanding"
        assert "Halfway" in exc.partial_output
    else:
        raise AssertionError("Expected AgentLimitExceeded")
