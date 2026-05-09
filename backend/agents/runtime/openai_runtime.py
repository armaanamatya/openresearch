"""OpenAI Agents SDK adapter for the provider-agnostic agent runtime."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from backend.agents.runtime.base import (
    AgentRuntimeSpec,
    ProviderConfigurationError,
    ProviderName,
    StreamEvent,
    StreamText,
    StreamToolCall,
    StreamUsage,
)


class OpenAiAgentRuntime:
    """AgentRuntime implementation backed by ``openai-agents``."""

    @property
    def provider_name(self) -> ProviderName:
        return "openai"

    async def run_agent(
        self,
        *,
        agent: AgentRuntimeSpec,
        user_input: str,
    ) -> AsyncIterator[StreamEvent]:
        try:
            from agents import Agent, ItemHelpers, Runner, function_tool
        except ImportError as exc:  # pragma: no cover - depends on local install
            raise ProviderConfigurationError(
                provider=self.provider_name,
                reason="openai-agents is not installed",
            ) from exc
        try:
            from agents import WebSearchTool
        except ImportError:  # pragma: no cover - depends on installed SDK version
            WebSearchTool = None

        root = (agent.working_directory or Path.cwd()).resolve()
        handoffs = [
            _build_openai_agent(sub_agent, root, Agent, function_tool, WebSearchTool)
            for sub_agent in agent.sub_agents
        ]
        openai_agent = _build_openai_agent(
            agent,
            root,
            Agent,
            function_tool,
            WebSearchTool,
            handoffs=handoffs,
        )

        result = Runner.run_streamed(
            openai_agent,
            input=user_input,
            max_turns=agent.max_turns,
        )
        saw_raw_text = False
        async for event in result.stream_events():
            event_type = _get(event, "type")
            if event_type == "raw_response_event":
                raw = _get(event, "data")
                delta = _raw_text_delta(raw)
                if delta:
                    saw_raw_text = True
                    yield StreamText(delta)
                usage = _raw_usage(raw)
                if usage is not None:
                    yield usage
                continue

            if event_type != "run_item_stream_event":
                continue

            item = _get(event, "item")
            item_type = _get(item, "type")
            if item_type == "message_output_item":
                if saw_raw_text:
                    continue
                text = _message_text(item, ItemHelpers)
                if text:
                    yield StreamText(text)
            elif item_type in {"tool_call_item", "function_call_item"}:
                yield StreamToolCall(
                    tool_id=str(_get(item, "id") or _get(item, "call_id") or ""),
                    tool_name=str(_tool_name(item)),
                    tool_input=_tool_input(item),
                )


def _build_openai_agent(
    spec: AgentRuntimeSpec,
    root: Path,
    agent_cls: type,
    function_tool: Callable[..., Any],
    web_search_tool_cls: type | None,
    *,
    handoffs: list[Any] | None = None,
) -> Any:
    return agent_cls(
        name=spec.name,
        instructions=spec.instructions,
        model=spec.model or None,
        tools=_build_tools(spec, root, function_tool, web_search_tool_cls),
        handoffs=handoffs or [],
    )


def _build_tools(
    spec: AgentRuntimeSpec,
    root: Path,
    function_tool: Callable[..., Any],
    web_search_tool_cls: type | None,
) -> list[Any]:
    factories: dict[str, Callable[[], Callable[..., Any]]] = {
        "Read": lambda: _read_tool(root),
        "Write": lambda: _write_tool(root),
        "Edit": lambda: _edit_tool(root),
        "Bash": lambda: _bash_tool(root),
        "WebSearch": _unsupported_web_search_tool,
        "WebFetch": _unsupported_web_fetch_tool,
    }
    tools: list[Any] = []
    for tool_spec in spec.tools:
        if tool_spec.name == "WebSearch" and web_search_tool_cls is not None:
            tools.append(web_search_tool_cls())
            continue
        factory = factories.get(tool_spec.name)
        if factory is None:
            continue
        tools.append(
            _decorate_tool(
                function_tool,
                factory(),
                name=tool_spec.name,
                description=tool_spec.description or _TOOL_DESCRIPTIONS[tool_spec.name],
            )
        )
    return tools


def _decorate_tool(
    function_tool: Callable[..., Any],
    func: Callable[..., Any],
    *,
    name: str,
    description: str,
) -> Any:
    try:
        return function_tool(
            func,
            name_override=name,
            description_override=description,
        )
    except TypeError:
        decorator = function_tool(
            name_override=name,
            description_override=description,
        )
        return decorator(func)


def _read_tool(root: Path) -> Callable[[str], str]:
    def read(file_path: str) -> str:
        path = _resolve_inside(root, file_path)
        return path.read_text(encoding="utf-8")

    return read


def _write_tool(root: Path) -> Callable[[str, str], str]:
    def write(file_path: str, content: str) -> str:
        path = _resolve_inside(root, file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Wrote {path.relative_to(root)}"

    return write


def _edit_tool(root: Path) -> Callable[[str, str, str], str]:
    def edit(file_path: str, old_string: str, new_string: str) -> str:
        path = _resolve_inside(root, file_path)
        content = path.read_text(encoding="utf-8")
        if old_string not in content:
            raise ValueError(f"Old string not found in {file_path!r}")
        path.write_text(content.replace(old_string, new_string, 1), encoding="utf-8")
        return f"Edited {path.relative_to(root)}"

    return edit


def _bash_tool(root: Path) -> Callable[[str, int], str]:
    async def bash(command: str, timeout_seconds: int = 120) -> str:
        timeout = max(1, min(int(timeout_seconds or 120), 600))
        proc = await asyncio.to_thread(
            subprocess.run,
            command,
            cwd=root,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return (
            f"exit_code={proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    return bash


def _unsupported_web_search_tool() -> Callable[[str], str]:
    def web_search(query: str) -> str:
        return (
            "WebSearch is not available in the local OpenAI runtime adapter yet. "
            f"Requested query: {query}"
        )

    return web_search


def _unsupported_web_fetch_tool() -> Callable[[str], str]:
    def web_fetch(url: str) -> str:
        return (
            "WebFetch is not available in the local OpenAI runtime adapter yet. "
            f"Requested URL: {url}"
        )

    return web_fetch


def _resolve_inside(root: Path, requested: str) -> Path:
    path = Path(requested)
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Path escapes working directory: {requested!r}")
    return resolved


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _message_text(item: Any, item_helpers: Any) -> str:
    try:
        return str(item_helpers.text_message_output(item))
    except Exception:
        pass

    parts: list[str] = []
    for content in _iter_content(_get(item, "content") or _get(item, "raw_item")):
        text = _get(content, "text") or _get(content, "output_text")
        if text:
            parts.append(str(text))
    return "\n".join(parts)


def _iter_content(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    raw_content = _get(value, "content")
    if isinstance(raw_content, list):
        return raw_content
    return [value]


def _raw_text_delta(raw: Any) -> str:
    raw_type = str(_get(raw, "type", ""))
    if raw_type in {"response.output_text.delta", "response.text.delta"}:
        return str(_get(raw, "delta", "") or "")
    return ""


def _raw_usage(raw: Any) -> StreamUsage | None:
    raw_type = str(_get(raw, "type", ""))
    if raw_type not in {"response.completed", "response.done"}:
        return None
    response = _get(raw, "response", raw)
    usage = _get(response, "usage")
    if usage is None:
        return None
    input_tokens = _int(_get(usage, "input_tokens") or _get(usage, "prompt_tokens"))
    output_tokens = _int(_get(usage, "output_tokens") or _get(usage, "completion_tokens"))
    details = _get(usage, "output_tokens_details") or _get(usage, "completion_tokens_details")
    reasoning_tokens = _int(_get(details, "reasoning_tokens"))
    return StreamUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def _tool_name(item: Any) -> str:
    raw = _get(item, "raw_item")
    return (
        _get(item, "name")
        or _get(item, "tool_name")
        or _get(raw, "name")
        or _get(raw, "tool_name")
        or ""
    )


def _tool_input(item: Any) -> dict[str, Any]:
    raw = _get(item, "raw_item")
    value = _get(item, "arguments") or _get(item, "input") or _get(raw, "arguments") or {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        import json

        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            return {"raw": value}
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {"value": dumped}
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {"value": value}


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


_TOOL_DESCRIPTIONS = {
    "Read": "Read a UTF-8 text file inside the current project workspace.",
    "Write": "Write a UTF-8 text file inside the current project workspace.",
    "Edit": "Replace the first exact text occurrence in a workspace file.",
    "Bash": "Run a shell command from the current project workspace.",
    "WebSearch": "Search the web for paper artifacts or documentation.",
    "WebFetch": "Fetch a URL needed for artifact or paper analysis.",
}


__all__ = ["OpenAiAgentRuntime"]
