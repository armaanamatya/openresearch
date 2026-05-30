#!/usr/bin/env python3
"""Smoke test: can a local Qwen (via vLLM) drive the OpenAI-agents tool loop?

Verifies executor-tier feasibility — whether `implement_baseline` could run on Qwen
instead of Sonnet. Builds a minimal agentic spec (Write + Bash tools), points
`OpenAiAgentRuntime` at the local vLLM endpoint, and asks Qwen to write a file and
run it. PASS = the file was created with the expected content (the model actually
*called* the Write tool) and at least one tool call was observed.

Usage:
    .venv/bin/python scripts/smoke_qwen_executor.py \
        [--base-url http://127.0.0.1:8001/v1] [--model Qwen/Qwen2.5-Coder-14B-Instruct]
"""
from __future__ import annotations

import argparse
import asyncio
import os
import tempfile
from pathlib import Path

from backend.agents.runtime.base import (
    AgentRuntimeSpec,
    StreamText,
    StreamToolCall,
    ToolSpec,
)
from backend.agents.runtime.openai_runtime import OpenAiAgentRuntime


async def _run(base_url: str, model: str) -> int:
    os.environ.setdefault("OPENAI_API_KEY", "local")  # SDK tracing safety
    tmp = Path(tempfile.mkdtemp(prefix="qwen_smoke_"))
    spec = AgentRuntimeSpec(
        name="smoke-executor",
        instructions=(
            "You are a coding agent working in a workspace. Use the Write tool to create "
            "files and the Bash tool to run shell commands. Be concise. When asked to "
            "create and run a file, you MUST call the tools — do not merely describe them."
        ),
        model=model,
        tools=(ToolSpec("Write", "write a file"), ToolSpec("Bash", "run a shell command")),
        working_directory=tmp,
        max_turns=10,
    )
    runtime = OpenAiAgentRuntime(base_url=base_url, api_key="local")
    tool_calls: list[str] = []
    text_chunks: list[str] = []
    print(f"[smoke] workspace={tmp}  model={model}  base_url={base_url}", flush=True)
    try:
        async for ev in runtime.run_agent(
            agent=spec,
            user_input=(
                "Create a file named hello.py whose ONLY content prints exactly: "
                "qwen executor works. Then run it with `python hello.py` and report the output."
            ),
        ):
            if isinstance(ev, StreamToolCall):
                tool_calls.append(ev.tool_name)
                print(f"[smoke] tool_call: {ev.tool_name} {str(ev.tool_input)[:120]}", flush=True)
            elif isinstance(ev, StreamText):
                text_chunks.append(ev.text)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] FAILED — runtime raised: {type(exc).__name__}: {exc}", flush=True)
        return 2

    hello = tmp / "hello.py"
    created = hello.exists()
    content_ok = created and "qwen executor works" in hello.read_text(errors="replace")
    print(f"[smoke] tool_calls={tool_calls}", flush=True)
    print(f"[smoke] hello.py created={created} content_ok={content_ok}", flush=True)
    if content_ok and tool_calls:
        print("[smoke] ✅ PASS — Qwen drove the agentic tool loop (wrote + ran a file).", flush=True)
        return 0
    print("[smoke] ❌ FAIL — Qwen did not reliably drive the tool loop.", flush=True)
    print(f"[smoke] final text (tail): {''.join(text_chunks)[-500:]}", flush=True)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-14B-Instruct")
    a = ap.parse_args()
    return asyncio.run(_run(a.base_url, a.model))


if __name__ == "__main__":
    raise SystemExit(main())
