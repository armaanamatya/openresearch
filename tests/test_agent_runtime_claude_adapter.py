"""Tests for the Claude provider runtime adapter."""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from typing import Any

from backend.agents.runtime.base import AgentRuntimeSpec, StreamText, StreamToolCall, StreamUsage, ToolSpec
from backend.agents.runtime.claude_runtime import ClaudeAgentRuntime


def test_claude_runtime_normalizes_sdk_events(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    class AgentDefinition:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class ClaudeAgentOptions:
        def __init__(self, **kwargs: Any) -> None:
            captured["options"] = kwargs

    class TextBlock:
        text = "hello"

    class ToolUseBlock:
        id = "tool-1"
        name = "Read"
        input = {"file_path": "paper.txt"}

    class AssistantMessage:
        def __init__(self, content: list[Any]) -> None:
            self.content = content

    class ResultMessage:
        is_error = False
        usage = {
            "input_tokens": 7,
            "output_tokens": 11,
            "cache_read_input_tokens": 3,
        }

    async def query(prompt: str, options: Any):
        captured["prompt"] = prompt
        yield AssistantMessage([TextBlock(), ToolUseBlock()])
        yield ResultMessage()

    fake = types.ModuleType("claude_agent_sdk")
    fake.AgentDefinition = AgentDefinition
    fake.AssistantMessage = AssistantMessage
    fake.ClaudeAgentOptions = ClaudeAgentOptions
    fake.ResultMessage = ResultMessage
    fake.ToolUseBlock = ToolUseBlock
    fake.query = query
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake)

    runtime = ClaudeAgentRuntime()
    spec = AgentRuntimeSpec(
        name="paper-understanding",
        instructions="system",
        model="claude-test",
        tools=(ToolSpec(name="Read"),),
        sub_agents=(
            AgentRuntimeSpec(
                name="verifier",
                instructions="verify",
                model="claude-sub",
                tools=(ToolSpec(name="Bash"),),
            ),
        ),
        working_directory=tmp_path,
        max_turns=5,
    )

    async def collect():
        return [event async for event in runtime.run_agent(agent=spec, user_input="task")]

    events = asyncio.run(collect())

    assert captured["prompt"] == "task"
    assert captured["options"]["model"] == "claude-test"
    assert captured["options"]["system_prompt"] == "system"
    assert captured["options"]["max_turns"] == 5
    assert str(captured["options"]["cwd"]) == str(tmp_path)
    assert "verifier" in captured["options"]["agents"]

    assert isinstance(events[0], StreamText)
    assert events[0].text == "hello"
    assert isinstance(events[1], StreamToolCall)
    assert events[1].tool_name == "Read"
    assert events[1].tool_input == {"file_path": "paper.txt"}
    assert isinstance(events[2], StreamUsage)
    assert events[2].input_tokens == 7
    assert events[2].output_tokens == 11
    assert events[2].cache_read_input_tokens == 3


def test_claude_runtime_preserves_uncapped_turns(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    class AgentDefinition:
        def __init__(self, **kwargs: Any) -> None:
            captured["sub_agent"] = kwargs

    class ClaudeAgentOptions:
        def __init__(self, **kwargs: Any) -> None:
            captured["options"] = kwargs

    class ResultMessage:
        usage = {"input_tokens": 1, "output_tokens": 1}

    async def query(prompt: str, options: Any):
        yield ResultMessage()

    fake = types.ModuleType("claude_agent_sdk")
    fake.AgentDefinition = AgentDefinition
    fake.AssistantMessage = type("AssistantMessage", (), {})
    fake.ClaudeAgentOptions = ClaudeAgentOptions
    fake.ResultMessage = ResultMessage
    fake.ToolUseBlock = type("ToolUseBlock", (), {})
    fake.query = query
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake)

    runtime = ClaudeAgentRuntime()
    spec = AgentRuntimeSpec(
        name="experiment-runner",
        instructions="system",
        model="claude-test",
        sub_agents=(
            AgentRuntimeSpec(
                name="verifier",
                instructions="verify",
                model="claude-sub",
                max_turns=None,
            ),
        ),
        working_directory=tmp_path,
        max_turns=None,
    )

    async def collect():
        return [event async for event in runtime.run_agent(agent=spec, user_input="task")]

    asyncio.run(collect())

    assert captured["options"]["max_turns"] is None
    assert captured["sub_agent"]["maxTurns"] is None
