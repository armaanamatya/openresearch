import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, ResultMessage

CWD = "/Volumes/CS_Stuff/openresearch/runs/prj_09047604e591d969/code"

async def one_call(i):
    errs = []
    opts = ClaudeAgentOptions(
        model="claude-sonnet-4-6", permission_mode="bypassPermissions", max_turns=4,
        cwd=CWD, allowed_tools=["Bash"], mcp_servers={}, setting_sources=[],
        system_prompt="Shell assistant.", stderr=lambda l: errs.append(l),
    )
    texts = []; info = {}
    try:
        async for m in query(prompt=f"Run `echo CALL_{i}_OK` via Bash and report output.", options=opts):
            if isinstance(m, AssistantMessage):
                for b in m.content:
                    t = getattr(b, "text", "");
                    if t: texts.append(t)
            elif isinstance(m, ResultMessage):
                info = {"is_error": getattr(m,"is_error",None), "subtype": getattr(m,"subtype",None),
                        "api_error_status": getattr(m,"api_error_status",None)}
        ok = ("CALL_%d_OK"%i) in " ".join(texts)
        ceref = [str(e)[:140] for e in errs if "onnection" in str(e) or "refused" in str(e).lower()]
        return {"i": i, "ok": ok, "info": info, "connrefused": ceref}
    except Exception as e:
        ceref = [str(x)[:140] for x in errs if "onnection" in str(x) or "refused" in str(x).lower()]
        return {"i": i, "ok": False, "exc": f"{type(e).__name__}: {str(e)[:180]}", "connrefused": ceref}

async def main():
    # 5 concurrent bundled-CLI children at once (mimics navigation fan-out + grader + implementer)
    results = await asyncio.gather(*[one_call(i) for i in range(1, 6)], return_exceptions=True)
    for r in results:
        print(r if not isinstance(r, dict) else f"CALL {r['i']}: ok={r['ok']} info={r.get('info')} exc={r.get('exc')} connrefused={r.get('connrefused')}")

asyncio.run(asyncio.wait_for(main(), timeout=180))
