import asyncio, sys
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, ResultMessage

CWD = "/Volumes/CS_Stuff/openresearch/runs/prj_09047604e591d969/code"
stderr_lines = []

def _stderr_cb(line: str):
    stderr_lines.append(line)

async def main():
    opts = ClaudeAgentOptions(
        model="claude-sonnet-4-6",
        permission_mode="bypassPermissions",
        max_turns=12,
        cwd=CWD,
        allowed_tools=["Bash", "Write", "Read"],
        mcp_servers={},
        setting_sources=[],
        max_thinking_tokens=4000,
        system_prompt="You are a coding agent. Complete the task using Write and Bash tools.",
        stderr=_stderr_cb,
    )
    texts, tools, result_info = [], [], {}
    try:
        async for m in query(
            prompt="Write a file train.py containing a 30-line PyTorch MLP training loop on random data, then run `python -c 'import ast; ast.parse(open(\"train.py\").read()); print(\"PARSE_OK\")'` and report the output.",
            options=opts,
        ):
            if isinstance(m, AssistantMessage):
                for b in m.content:
                    t = getattr(b, "text", "")
                    if t: texts.append(t)
                    n = getattr(b, "name", None)
                    if n: tools.append(n)
            elif isinstance(m, ResultMessage):
                result_info = {
                    "subtype": getattr(m, "subtype", None),
                    "is_error": getattr(m, "is_error", None),
                    "api_error_status": getattr(m, "api_error_status", None),
                    "num_turns": getattr(m, "num_turns", None),
                    "duration_ms": getattr(m, "duration_ms", None),
                    "stop_reason": getattr(m, "stop_reason", None),
                    "total_cost_usd": getattr(m, "total_cost_usd", None),
                }
    except Exception as e:
        print("HEAVY_EXCEPTION:", type(e).__name__, repr(str(e))[:300])
        for attr in ("api_error_status", "errors", "subtype"):
            if hasattr(e, attr):
                print(f"  exc.{attr} =", getattr(e, attr))
    print("TOOL_CALLS:", tools)
    print("TEXT_LEN:", len(" ".join(texts)))
    print("RESULT_INFO:", result_info)
    print("STDERR_TAIL:")
    for l in stderr_lines[-15:]:
        print("  |", str(l)[:200])

try:
    asyncio.run(asyncio.wait_for(main(), timeout=300))
except asyncio.TimeoutError:
    print("HEAVY_VERDICT: TIMEOUT_300s")
    print("STDERR_TAIL:")
    for l in stderr_lines[-15:]:
        print("  |", str(l)[:200])
