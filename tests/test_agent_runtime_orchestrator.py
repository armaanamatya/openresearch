"""Provider runtime wiring tests for the root orchestrator."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator

from backend.agents.orchestrator import ReproLabOrchestrator
from backend.agents.runtime.base import AgentRuntimeSpec, ProviderName, StreamEvent, StreamText, StreamUsage


class FakeOpenAiRuntime:
    provider_name: ProviderName = "openai"

    def __init__(self) -> None:
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
        yield StreamText('{"core_contribution": "ok"}')
        yield StreamUsage(input_tokens=1, output_tokens=2, reasoning_tokens=1)


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


def test_orchestrator_does_not_cap_heavy_agents_by_default(tmp_path: Path) -> None:
    runtime = FakeOpenAiRuntime()
    orchestrator = ReproLabOrchestrator(
        "prj_uncapped",
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
    assert runtime.agent.max_turns is None
