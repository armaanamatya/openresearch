# Claude SDK aclose() deadlock — root-cause investigation

_2026-05-22 · Investigator: Sonnet sub-agent dispatched mid-session ·
Status: investigation complete; fix is **Workaround B** (~20 lines of wrapper code)._

## Root cause (two compounding defects)

**Defect 1 — SDK triple-nested async generator race:** The SDK creates three nested
async generators per call:
1. `query()` in `claude_agent_sdk/query.py:11` — yields messages from `process_query()`.
2. `process_query()` in `_internal/client.py:52` — yields messages from `_process_query_inner()`; its `finally` block explicitly `await inner.aclose()` (line 83).
3. `_process_query_inner()` in `_internal/client.py:91` — yields parsed messages; its `finally` calls `await query.close()` (line 229), which cancels `_read_task` and `transport.close()` to terminate the subprocess.

asyncio tracks **all three** via `sys.get_asyncgen_hooks()` (`BaseEventLoop._asyncgen_firstiter_hook`). When `asyncio.run()` (or a `wait_for` cancel) triggers `BaseEventLoop.shutdown_asyncgens()`, it does `asyncio.gather(*[ag.aclose() for ag in closing_agens], return_exceptions=True)` — closing all three concurrently. `aclose(process_query)` enters `process_query.finally` and awaits `inner.aclose()`, which sets `_process_query_inner.ag_running = True`. The concurrent `aclose(_process_query_inner)` from the gather sees `ag_running = True` and raises:
> `RuntimeError: aclose(): asynchronous generator is already running`

**Defect 2 — WSL2 futex deadlock:** `transport.close()` in `subprocess_cli.py:583-584` does `with suppress(Exception): await self._process.wait()` **with no timeout** after sending SIGKILL. On WSL2 (`Linux 6.6.114.1-microsoft-standard-WSL2`), SIGCHLD delivery for SIGKILL'd Node.js processes can be delayed/lost by the WSL2 compat layer. `process.wait()` uses asyncio's child-watcher (SIGCHLD-driven), so the event loop parks in `epoll_wait` / `futex_wait_queue` indefinitely. This is what wedged mech2 / mech3 controllers in `futex_wait_queue`.

## Verdict

**SDK bug.** Our `asyncio.wait_for(collect_agent_text(...), timeout=...)` is the correct pattern; `asyncio.run()`'s cleanup is the proximate trigger of the race. The SDK's three-level async generator chain violates the implicit ownership contract — `process_query` owns `_process_query_inner` and explicitly manages its lifecycle, but does not prevent asyncio from independently tracking and closing the inner generator at shutdown.

## Recommended fix (v1) — Workaround B: thread isolation

Run `collect_agent_text(...)` in a `threading.Thread` with its OWN `asyncio.run()` inside. The controller awaits a `concurrent.futures.Future` populated by the worker thread.

**Why this works:** the worker thread has its own event loop. Its `shutdown_asyncgens` is isolated from the main controller's loop. Even if the worker's loop hits the SDK's aclose RuntimeError internally, it does NOT propagate to the controller and does NOT deadlock the main process's `futex_wait_queue`.

**Implementation sketch (in `backend/agents/rdr/agent.py`, ~20 lines):**

```python
import concurrent.futures
import asyncio

def _run_sdk_in_thread(prompt, code_dir, model, provider, runtime, max_turns, timeout_s):
    """Run collect_agent_text in an isolated thread with its own event loop.
    Returns the agent's text output, or raises TimeoutError on watchdog.
    """
    def _worker():
        return asyncio.run(asyncio.wait_for(
            collect_agent_text(
                "baseline-implementation",
                prompt,
                project_dir=code_dir,
                model=model,
                provider=provider,
                runtime=runtime,
                max_turns=max_turns,
            ),
            timeout=timeout_s,
        ))
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_worker)
        return future.result(timeout=timeout_s + 30)  # +30s slack for thread teardown
```

In `_reproduce_inner`, replace the existing `asyncio.wait_for(collect_agent_text(...))` call with `await asyncio.to_thread(_run_sdk_in_thread, prompt, code_dir, model, provider, runtime, max_turns, timeout_s)`. (Wraps the wrapper in `asyncio.to_thread` so the controller's loop continues to run while the worker thread handles the SDK.)

The `_ACTIVE_CHILDREN` atexit hook in `subprocess_cli.py:47` still ensures the claude subprocess is SIGTERMed when the main process exits. Worst-case: an abandoned worker thread takes ~10s to exit (transport.close timeouts) while holding the subprocess; the subprocess is still killed within ~10s; no permanent orphan.

**Trade-offs:**
- Complexity: MEDIUM (~20 lines).
- Robustness: HIGH (main loop is isolated from SDK's nested-generator race).
- New-bug risk: LOW.

## Rejected alternatives

**Workaround A — separate Python subprocess:** highest robustness but HIGH complexity (full IPC serialization of `ClaudeAgentOptions` + StreamEvent results + two layers of child-process management). Defer to v2.

**Workaround C — signal-based watchdog:** SIGTERM to the SDK's subprocess does NOT fix the asyncio generator race, does NOT unblock `shutdown_asyncgens`, does NOT address SIGCHLD/futex hang. Treats symptom, not cause.

## Minimal SDK patch (v2 — file upstream)

**File:** `claude_agent_sdk/_internal/client.py` lines 82–83.

Current:
```python
try:
    await inner.aclose()
```

Patch (3 lines): wrap in `try: ... except RuntimeError as e: if "already running" not in str(e): raise`. When `aclose(_process_query_inner)` races with `shutdown_asyncgens`' concurrent close, the redundant call silently swallows the known-safe error.

**Second SDK fix:** `subprocess_cli.py:583-584` — replace `with suppress(Exception): await self._process.wait()` with a bounded `with anyio.fail_after(5): await self._process.wait()` so the final wait has a 5-second cap (covers the WSL2 SIGCHLD hang).

Both belong as upstream PRs to `claude-agent-sdk` (v0.1.80).

## Recommendation for this branch

**Apply Workaround B before the merge.** It removes the watchdog-as-only-defense story and makes the harness actually robust to long Claude SDK calls. ~20 lines of code, one new test, fits in one commit (one of the 10-15 noteworthy ones). After this, the `_ClusterWatchdog` becomes a defense-in-depth net rather than the primary mitigation.

## Open follow-ups

- File upstream SDK PRs (post-merge work).
- Verify Workaround B on the wedge-prone `mechanistic-understanding` paper — re-run mech end-to-end and see whether it now completes (no controller deadlock even if the SDK still hits the RuntimeError internally).
