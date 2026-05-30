import asyncio, sys, os
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage

CWD = "/Volumes/CS_Stuff/openresearch/runs/prj_09047604e591d969"

async def main():
    opts = ClaudeAgentOptions(
        model=None,
        permission_mode="bypassPermissions",
        max_turns=3,
        cwd=CWD,
        allowed_tools=["Bash"],
        mcp_servers={},
        setting_sources=[],
        system_prompt="You are a shell assistant. When asked, use the Bash tool.",
    )
    got_text = []
    tool_calls = []
    try:
        async for message in query(
            prompt="Run the bash command `pwd && echo SMOKE_OK` using the Bash tool, then report exactly what it printed.",
            options=opts,
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    t = getattr(block, "text", "")
                    if t:
                        got_text.append(t)
                    name = getattr(block, "name", None)
                    if name:
                        tool_calls.append(name)
    except Exception as e:
        print("SMOKE_EXCEPTION:", type(e).__name__, str(e)[:300])
        return
    full = " ".join(got_text)
    print("TOOL_CALLS:", tool_calls)
    print("TEXT_LEN:", len(full))
    print("TEXT_HEAD:", full[:400].replace(chr(10), " "))
    print("SMOKE_VERDICT:", "PASS" if ("SMOKE_OK" in full or "Bash" in tool_calls) else "EMPTY_OR_NO_TOOL")

try:
    asyncio.run(asyncio.wait_for(main(), timeout=90))
except asyncio.TimeoutError:
    print("SMOKE_VERDICT: TIMEOUT (no completion in 90s)")
