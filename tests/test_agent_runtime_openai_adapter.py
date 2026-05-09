"""Tests for the OpenAI provider runtime adapter."""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from typing import Any

from backend.agents.runtime.base import AgentRuntimeSpec, StreamText, StreamToolCall, StreamUsage, ToolSpec
from backend.agents.runtime.openai_runtime import OpenAiAgentRuntime


def test_openai_runtime_normalizes_streamed_events(monkeypatch, tmp_path: Path) -> None:
    created_agents: list[Any] = []
    captured_run: dict[str, Any] = {}

    class Agent:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            created_agents.append(self)

    class WebSearchTool:
        pass

    def function_tool(func=None, **kwargs: Any):
        def decorate(inner):
            inner.tool_metadata = kwargs
            return inner

        return decorate(func) if func is not None else decorate

    class ItemHelpers:
        @staticmethod
        def text_message_output(item: Any) -> str:
            return item.text

    class Runner:
        @staticmethod
        def run_streamed(agent: Any, *, input: str, max_turns: int):
            captured_run.update(agent=agent, input=input, max_turns=max_turns)
            return FakeResult()

    class FakeResult:
        async def stream_events(self):
            yield types.SimpleNamespace(
                type="raw_response_event",
                data=types.SimpleNamespace(type="response.output_text.delta", delta="partial"),
            )
            yield types.SimpleNamespace(
                type="run_item_stream_event",
                item=types.SimpleNamespace(
                    type="tool_call_item",
                    id="call-1",
                    name="Read",
                    arguments='{"file_path": "paper.txt"}',
                ),
            )
            yield types.SimpleNamespace(
                type="run_item_stream_event",
                item=types.SimpleNamespace(type="message_output_item", text="partial"),
            )
            yield types.SimpleNamespace(
                type="raw_response_event",
                data=types.SimpleNamespace(
                    type="response.completed",
                    response=types.SimpleNamespace(
                        usage=types.SimpleNamespace(
                            input_tokens=13,
                            output_tokens=17,
                            output_tokens_details=types.SimpleNamespace(reasoning_tokens=5),
                        )
                    ),
                ),
            )

    fake = types.ModuleType("agents")
    fake.Agent = Agent
    fake.ItemHelpers = ItemHelpers
    fake.Runner = Runner
    fake.WebSearchTool = WebSearchTool
    fake.function_tool = function_tool
    monkeypatch.setitem(sys.modules, "agents", fake)

    runtime = OpenAiAgentRuntime()
    spec = AgentRuntimeSpec(
        name="artifact-discovery",
        instructions="system",
        model="gpt-test",
        tools=(ToolSpec(name="Read"), ToolSpec(name="WebSearch")),
        sub_agents=(
            AgentRuntimeSpec(
                name="method-verifier",
                instructions="verify",
                model="o-test",
            ),
        ),
        working_directory=tmp_path,
        max_turns=4,
    )

    async def collect():
        return [event async for event in runtime.run_agent(agent=spec, user_input="task")]

    events = asyncio.run(collect())

    assert captured_run["input"] == "task"
    assert captured_run["max_turns"] == 4
    assert created_agents[-1].kwargs["name"] == "artifact-discovery"
    assert created_agents[-1].kwargs["model"] == "gpt-test"
    assert len(created_agents[-1].kwargs["handoffs"]) == 1
    assert any(isinstance(tool, WebSearchTool) for tool in created_agents[-1].kwargs["tools"])

    assert isinstance(events[0], StreamText)
    assert events[0].text == "partial"
    assert isinstance(events[1], StreamToolCall)
    assert events[1].tool_name == "Read"
    assert events[1].tool_input == {"file_path": "paper.txt"}
    assert len([event for event in events if isinstance(event, StreamText)]) == 1
    assert isinstance(events[2], StreamUsage)
    assert events[2].input_tokens == 13
    assert events[2].output_tokens == 17
    assert events[2].reasoning_tokens == 5
