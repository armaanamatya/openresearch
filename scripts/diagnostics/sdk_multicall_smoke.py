import asyncio, sys
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, ResultMessage

CWD = "/Volumes/CS_Stuff/openresearch/runs/prj_09047604e591d969/code"

async def one_call(i):
    errs = []
    opts = ClaudeAgentOptions(
        model="claude-sonnet-4-6",
        permission_mode="bypassPermissions",
        max_turns=4,
        cwd=CWD,
        allowed_tools=["Bash"],
        mcp_servers={},
        setting_sources=[],
        system_prompt="Shell assistant.",
        stderr=lambda l: errs.append(l),
    )
    texts = []
    info = {}
    try:
        async for m in query(prompt=f"Run `echo CALL_{i}_OK` via Bash and report output.", options=opts):
            if isinstance(m, AssistantMessage):
                for b in m.content:
                    t = getattr(b, "text", "")
                    if t: texts.append(t)
            elif isinstance(m, ResultMessage):
                info = {"is_error": getattr(m,"is_error",None), "subtype": getattr(m,"subtype",None),
                        "api_error_status": getattr(m,"api_error_status",None)}
        return {"i": i, "ok": ("CALL_%d_OK"%i in " ".join(texts)), "text_len": len(" ".join(texts)), "info": info, "err_tail": [str(e)[:120] for e in errs[-3:]]}
    except Exception as e:
        return {"i": i, "ok": False, "exc": f"{type(e).__name__}: {str(e)[:160]}", "err_tail": [str(x)[:160] for x in errs[-5:]]}

# Mimic the pipeline: each "primitive" does its own asyncio.run() in a worker thread.
import concurrent.futures
_fails = 0
for i in range(1, 61):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        try:
            r = pool.submit(asyncio.run, one_call(i)).result(timeout=120)
        except Exception as e:
            r = {"i": i, "ok": False, "outer_exc": f"{type(e).__name__}: {str(e)[:160]}"}
    if (not r.get("ok")) or i % 10 == 0 or i <= 2:
        print(f"CALL {i}: ok={r.get('ok')} text_len={r.get('text_len')} info={r.get('info')} exc={r.get('exc') or r.get('outer_exc')}")
    if r.get("err_tail"):
        for e in r["err_tail"]:
            if any(k in e for k in ("Connection","refused","ECONN")): print("    stderr|", e)
    if not r.get("ok"):
        _fails += 1
        print(f"  >>> DEGRADED at call {i} (total fails={_fails})")
        if _fails >= 3:
            print("STOP: reproduced degradation 3x"); break
print(f"DONE fails={_fails}")
