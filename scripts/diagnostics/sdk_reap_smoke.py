"""Confirming experiment: does reaping bundled-claude children AFTER each
successful call prevent the ~33-call ConnectionRefused degradation?

Control = sdk_multicall_smoke.py (no reap) degraded at call 34.
Treatment (this) = reap each call's children on success. If it survives to 50,
the reap-on-success hypothesis is confirmed.
"""
import asyncio, os, signal, sys, time
import concurrent.futures

sys.path.insert(0, "/Volumes/CS_Stuff/openresearch")
sys.path.insert(0, "/Volumes/CS_Stuff/openresearch/src")
from backend.services.context.workspace.tools.rlm_query import _bundled_claude_child_pids  # noqa: E402

from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, ResultMessage  # noqa: E402

CWD = "/Volumes/CS_Stuff/openresearch/runs/prj_09047604e591d969/code"


def _reap(pre_pids):
    try:
        wedged = _bundled_claude_child_pids() - pre_pids
    except Exception:
        return 0
    n = 0
    for pid in wedged:
        try:
            os.kill(pid, signal.SIGKILL)
            n += 1
        except (OSError, ProcessLookupError):
            pass
    if wedged:
        time.sleep(0.2)
    return n


async def one_call(i):
    opts = ClaudeAgentOptions(
        model="claude-sonnet-4-6", permission_mode="bypassPermissions", max_turns=4,
        cwd=CWD, allowed_tools=["Bash"], mcp_servers={}, setting_sources=[],
        system_prompt="Shell assistant.",
    )
    texts, info = [], {}
    async for m in query(prompt=f"Run `echo CALL_{i}_OK` via Bash and report output.", options=opts):
        if isinstance(m, AssistantMessage):
            for b in m.content:
                t = getattr(b, "text", "")
                if t: texts.append(t)
        elif isinstance(m, ResultMessage):
            info = {"is_error": getattr(m, "is_error", None), "subtype": getattr(m, "subtype", None),
                    "api_error_status": getattr(m, "api_error_status", None)}
    return {"ok": (f"CALL_{i}_OK" in " ".join(texts)), "info": info}


_fails = 0
for i in range(1, 51):
    pre = _bundled_claude_child_pids()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        try:
            r = pool.submit(asyncio.run, one_call(i)).result(timeout=120)
        except Exception as e:
            r = {"ok": False, "exc": f"{type(e).__name__}: {str(e)[:160]}"}
    killed = _reap(pre)  # <-- the fix under test: reap THIS call's children
    if (not r.get("ok")) or i % 10 == 0 or i <= 2:
        print(f"CALL {i}: ok={r.get('ok')} reaped={killed} info={r.get('info')} exc={r.get('exc')}", flush=True)
    if not r.get("ok"):
        _fails += 1
        print(f"  >>> DEGRADED at call {i} (fails={_fails})", flush=True)
        if _fails >= 3:
            print("VERDICT: reap-on-success did NOT prevent degradation"); break
if _fails == 0:
    print("VERDICT: reap-on-success PREVENTED degradation (50/50 ok)")
print(f"DONE fails={_fails}")
