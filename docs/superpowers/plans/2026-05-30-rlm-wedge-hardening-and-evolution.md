# RLM Wedge-Hardening + Evolution — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the silent sub-RLM wedge, make it visible in <2 min, stop zero-evidence reports, move the hot-volume sub-RLM path onto a timeout-bearing transport, and lay flag-gated foundations for cross-run learning — all on `claude-oauth` root + `claude-sonnet-4-6` sub-agents, no new paid services.

**Architecture:** Nine staged phases, each a standalone landable commit. Phases 1–4 are the high-confidence reliability/cost core (the audit's next-3-commits + fast-follow). Phases 5–7 are bounded optimizations. Phases 8–9 are flag-gated research prototypes (PEEK context-map, MUSE negative-lessons) that stay OFF by default. Every behavior change is gated by an env flag with a documented rollback. Source audit: `docs/audits/2026-05-30-infra-optimization-audit.md`.

**Tech Stack:** Python 3.14 / FastAPI / `rlms==0.1.1` / `claude-agent-sdk` (bundled CLI, OAuth) / `openai` (via rlm OpenAI sub-backend) / httpx / pytest (+asyncio +xdist) / Next.js 16 + React 19 (SSE dashboard).

**Conventions (verified):**
- Run one test: `.venv/bin/python -m pytest tests/path/test_x.py::test_name -xvs`
- Run all (parallel): `.venv/bin/python -m pytest tests/ -n auto`
- pytest config: `pyproject.toml:49-51` (`pythonpath=["src"]`, `testpaths=["tests"]`), plugins `pytest-asyncio`, `pytest-xdist`, `pytest-rerunfailures`.
- SDK mock idiom: monkeypatch `sys.modules["claude_agent_sdk"]` with a fake module exposing stub `AssistantMessage`/`ResultMessage` classes + an `async def query()` generator. (`tests/test_agent_runtime_claude_adapter.py:16-60`.)
- `ClaudeLlmClient.complete(*, system, user) -> str` (sync); internally `_async_complete -> tuple[str, dict]`.
- New SSE event types MUST be registered in `frontend/src/lib/events/rlm-events.ts` (`RLM_EVENT_TYPES` array + a TS interface + the union) or the frontend drops them.
- Fail-soft is the house rule (design decision D3): observability/cleanup must never raise into a run.

---

## Phase roadmap & dependency graph

| Ph | Title | Commit theme | Fixes | Flag (default) | Size | Depends on |
|----|----|----|----|----|----|----|
| 0 | Branch + green baseline | — | — | — | — | — |
| 1 | Read-idle timeout + typed sub-RLM failure + hardened kill | reliability | FM-001/002/007 | `REPROLAB_SUBRLM_READ_IDLE_S=120` | 200–500 | 0 |
| 2 | Run heartbeat + dangling-subcall detector + status reconciliation | observability | FM-003/005/006 | `REPROLAB_STALL_DETECT_S=120` | 200–500 | 1 (consumes stall sink) |
| 3 | Final-report evidence gate | report validity | FM-004 | `REPROLAB_EVIDENCE_GATE=1` | <200 | 0 |
| 4 | Route nav sub-RLM to gpt-5-mini (timeout + telemetry) | cost/latency | FM-001/008 | `REPROLAB_ACCELERATOR=off` | <200 | 1,2 (A/B telemetry) |
| 5 | Code-level sub-RLM concurrency cap | latency/reliability | FM-008 | `REPROLAB_SUBRLM_MAX_CONCURRENCY=4` | <200 | 1 |
| 6 | Prompt-cache instrumentation (measure first) | cost | opt #6 | `REPROLAB_LOG_CACHE_TOKENS=0` | <200 | 0 |
| 7 | Infra hardening (SSE disconnect, dashboard freshness, Docker/RunPod lifecycle) | reliability/observability | survey | per-fix flags | 200–500 | 2 |
| 8 | PEEK context-map prototype (deterministic, write-once) | latency/accuracy | research | `REPROLAB_CONTEXT_MAP=0` | 200–500 | 0 |
| 9 | MUSE negative-lessons prototype | accuracy | research | `REPROLAB_NEGATIVE_LESSONS=0` | 200–500 | 3,8 |

Phases 1→2→4 and 8→9 are ordered; 3, 5, 6, 7, 8 are otherwise independent. Land 1–3 first (they need no new spend), then 4, then the rest. **Phases 8–9 require a brainstorming pass before implementation** (see their headers) — they are prototypes, not locked designs.

### Implementation status (2026-05-30, branch `feat/rlm-wedge-hardening`)

| Item | Status | Commit / note |
|--|--|--|
| Phase 1 read-idle + sentinel + kill | ✅ done | `da7bdff` + `6da7545` (killpg fix) |
| Phase 2 stall detector + freshness | ✅ done | `f7c0ba9` |
| Phase 3 evidence gate | ✅ done | `7fa38d0` + `bef112d` (E2E contract) |
| Phase 4 gpt-5-mini route | ✅ done | `b73e730` |
| Phase 5 concurrency cap | ✅ done | committed |
| Phase 6.1 cache-ratio measurement | ✅ done | committed |
| Phase 7c uvicorn graceful-shutdown | ✅ done | committed |
| Phase 7e RunPod granular timeout | ✅ done | committed |
| Phase 7g run-replay harness | ✅ done | committed (verified on real runs) |
| Phase 7b SSE disconnect | ✅ done | committed |
| Phase 8 PEEK context-map | ✅ done | flagged off (`REPROLAB_CONTEXT_MAP`); **union-per-field** (not write-once) — design `2026-05-30-intra-run-context-map-design.md`, plan `2026-05-30-intra-run-context-map.md` |
| Phase 6.2 OAuth cache_control breakpoint | ⏸ deferred | needs probing whether `claude-agent-sdk` forwards a structured system block (real SDK) |
| Phase 7a frontend stale→"stalled" pill | ⏸ deferred | UI already shows a "no signal Ns" chip (`rlm-header.tsx:148-183`); pill-flip is polish; needs vitest/RTL + careful display-vs-backend status typing |
| Phase 7d Docker OOM detection | ⏸ deferred | exec-vs-container `OOMKilled` ambiguity (exit 137 ≠ container OOMKilled for an exec); needs real Docker to verify |
| Phase 7f provider request-id capture | ⏸ deferred | needs the real `ResultMessage` shape (which id field the bundled CLI exposes) |
| Phase 9 MUSE negative-lessons | ⏸ not started | flag-gated prototype; requires a brainstorm first |

The four deferred 6.2/7a/7d/7f items share a theme: each needs a **real environment** (real SDK / browser / Docker) to verify, so they were held back rather than shipped unverified to a hot path. They remain fully specified above for a focused follow-up.

---

## Phase 0: Branch + green baseline

**Files:** none (git + test run only)

- [ ] **Step 1: Create the feature branch off the current branch**

```bash
git checkout -b feat/rlm-wedge-hardening
git status   # expect: On branch feat/rlm-wedge-hardening, clean (docs/audits + docs/superpowers/plans untracked)
```

- [ ] **Step 2: Establish the green test baseline for the files we will touch**

```bash
.venv/bin/python -m pytest tests/ -k "rlm_query or claude_oauth or primitive_cache or sse_bridge or forced_iteration or report or accelerator" -q
```
Expected: PASS (record the count; this is the regression floor — no later phase may reduce it).

- [ ] **Step 3: Commit the audit + this plan (documentation only)**

```bash
git add docs/audits/2026-05-30-infra-optimization-audit.md docs/superpowers/plans/2026-05-30-rlm-wedge-hardening-and-evolution.md
git commit -m "docs: infra audit + wedge-hardening master plan"
```

---

## Phase 1: Read-idle timeout + typed sub-RLM failure + hardened kill

**Problem:** `ClaudeLlmClient.complete` (`rlm_query.py:556-637`) bounds a sub-RLM only by a **total** wall-clock cap (`future.result(timeout=600)`), wrapping an SDK stream (`_async_complete:699`) that has **no read-idle timeout** (Context7: `query()` "continues indefinitely if no ResultMessage"). On timeout it returns `""` — indistinguishable from a real empty answer — and `ex.shutdown(wait=False)` leaks the worker. Fixes FM-001/002/007.

**Architecture:** Add an application-level **read-idle** timeout that fires when no stream event arrives within `read_idle_s` (default 120s). On stall: salvage any partial assistant text; else kill descendants (hardened, fail-soft, cross-platform-guarded) and return a **non-empty self-describing sentinel** (not `""`) so the root sees a typed failure; optionally notify a `stall_event_sink` (Phase 2 wires the dashboard emit). Keep the existing total-time cap as a backstop, but it now also returns the sentinel.

> **Reuse, don't reinvent (Section E Gap-1):** a full typed-failure taxonomy already exists at `backend/agents/resilience/failures.py` (`WallClockExceeded`, `TransientError`, …) with a `classify.py` boundary. Use those types for internal classification/telemetry; the sentinel **string** is only the root-facing surface because rlm's `complete()` contract is `-> str` (raising would break the rlm sub-call machinery). Full unification onto the built-but-**unwired** `run_agent_with_resilience` engine is deferred (Gap-5) — it consumes an event-streaming runtime, an impedance mismatch with this `ThreadPoolExecutor`+`future.result` path.

**Files:**
- Modify: `backend/services/context/workspace/tools/rlm_query.py`
- Test: `tests/test_rlm_query_read_idle.py` (create)

- [ ] **Step 1: Write the failing test — read-idle stall returns a non-empty sentinel, not ""**

Create `tests/test_rlm_query_read_idle.py`:

```python
"""Phase 1 — ClaudeLlmClient read-idle timeout + typed sub-RLM failure."""
from __future__ import annotations

import asyncio
import sys
import types
from typing import Any

import pytest


def _install_fake_sdk(monkeypatch, query_fn) -> None:
    class TextBlock:
        def __init__(self, text: str): self.text = text
    class AssistantMessage:
        def __init__(self, content, usage=None): self.content = content; self.usage = usage
    class ResultMessage:
        def __init__(self, result, usage=None): self.result = result; self.usage = usage
    fake = types.ModuleType("claude_agent_sdk")
    fake.AssistantMessage = AssistantMessage
    fake.ResultMessage = ResultMessage
    fake.ClaudeAgentOptions = type("ClaudeAgentOptions", (), {"__init__": lambda self, **k: None})
    fake.query = query_fn
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake)
    return AssistantMessage, ResultMessage, TextBlock


def test_read_idle_stall_returns_sentinel_not_empty(monkeypatch):
    from backend.services.context.workspace.tools import rlm_query as rq

    async def hanging_query(prompt: str, options: Any):
        # Never yields an event — simulate a half-open socket / stalled stream.
        await asyncio.sleep(60)
        yield  # unreachable
    _install_fake_sdk(monkeypatch, hanging_query)
    # Avoid SIGKILLing real processes during the test.
    monkeypatch.setattr(rq, "_bundled_claude_child_pids", lambda: set())

    client = rq.ClaudeLlmClient(model="claude-test", max_turns=1)
    result = client.complete(system="s", user="u", read_idle_s=0.2)

    assert result != ""                       # NOT the old empty-string fallback
    assert rq.SUB_RLM_STALL_SENTINEL in result
```

- [ ] **Step 2: Run it — expect failure (no `read_idle_s` kwarg / no sentinel)**

Run: `.venv/bin/python -m pytest tests/test_rlm_query_read_idle.py::test_read_idle_stall_returns_sentinel_not_empty -xvs`
Expected: FAIL — `TypeError: complete() got an unexpected keyword argument 'read_idle_s'` (or `AttributeError: SUB_RLM_STALL_SENTINEL`).

- [ ] **Step 3: Add the sentinel, the exception, the env default, and the read-idle loop**

In `backend/services/context/workspace/tools/rlm_query.py`, near the top-level constants (after `_ZERO_USAGE` or before the class), add:

```python
# Phase 1 (BUG-NEW-044 follow-up): a non-empty, self-describing sentinel returned
# when a sub-RLM stalls. NOT "" — the root must distinguish a dead socket from a
# real empty answer. rlm treats this string as the sub-call's answer; the root
# system prompt teaches it to treat a SUB_RLM_STALL line as a retryable failure.
SUB_RLM_STALL_SENTINEL = "[SUB_RLM_STALL]"


def _stall_message(idle_s: float) -> str:
    return (
        f"{SUB_RLM_STALL_SENTINEL} the sub-query stalled (no stream bytes for "
        f"{idle_s:.0f}s) and was aborted. Retry with a smaller slice or fewer "
        f"concurrent sub-calls; do NOT treat this as the answer."
    )


class _SubRlmReadIdleTimeout(Exception):
    """Raised inside the worker when the SDK stream is idle past read_idle_s."""
    def __init__(self, idle_s: float):
        super().__init__(f"sub-RLM stream idle > {idle_s:.0f}s")
        self.idle_s = idle_s


def _read_idle_default() -> float:
    """Default per-event read-idle bound; 0 disables (env REPROLAB_SUBRLM_READ_IDLE_S)."""
    try:
        return float(os.environ.get("REPROLAB_SUBRLM_READ_IDLE_S", "120") or "120")
    except (TypeError, ValueError):
        return 120.0
```

Change `ClaudeLlmClient.__init__` to accept and store the sink + default:

```python
    def __init__(
        self,
        model: str | None = None,
        max_turns: int = 1,
        stall_event_sink: "Callable[[dict], None] | None" = None,
    ) -> None:
        self._model = model
        self._max_turns = max_turns
        self._last_usage: dict[str, int] = _ZERO_USAGE.copy()
        # Phase 2 wires this to the dashboard emit; default None = log only.
        self._stall_event_sink = stall_event_sink
```

(Add `from typing import Callable` if not already imported — it is via `Any` import; add `Callable` to that import line.)

Rewrite the `async for` loop in `_async_complete` to a read-idle loop. Replace the `try: async for event in query(...)` block (lines ~698-724) with:

```python
        read_idle_s = self._read_idle_s  # set by complete() before scheduling
        agen = query(prompt=user, options=options)
        aiter = agen.__aiter__()
        try:
            while True:
                try:
                    event = await asyncio.wait_for(aiter.__anext__(), timeout=read_idle_s)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError as exc:
                    # No bytes for read_idle_s — a stalled/half-open stream. Raise so
                    # complete() can kill the child and surface a typed failure. Salvage
                    # any partial assistant text first (partial > nothing).
                    raise _SubRlmReadIdleTimeout(read_idle_s) from exc
                if isinstance(event, AssistantMessage):
                    for block in (getattr(event, "content", None) or []):
                        txt = getattr(block, "text", None)
                        if txt:
                            assistant_parts.append(txt)
                    usage = getattr(event, "usage", None)
                    if usage is not None:
                        assistant_usages.append(usage)
                elif isinstance(event, ResultMessage):
                    result_text = event.result or ""
                    if event.usage is not None:
                        result_usage = event.usage
                    break
        except _SubRlmReadIdleTimeout:
            if assistant_parts:
                logger.warning(
                    "rlm_query: read-idle %.0fs but %d assistant part(s) salvaged",
                    read_idle_s, len(assistant_parts),
                )
            else:
                raise  # pure stall — propagate so complete() returns the sentinel
        except Exception as exc:  # noqa: BLE001 — salvage over crash (existing behavior)
            logger.warning(
                "rlm_query: claude-agent-sdk stream raised (%s); salvaging %d part(s)",
                exc, len(assistant_parts),
            )
```

In `complete()`, set `self._read_idle_s` before scheduling and handle the new exception. Replace the `_timeout_s = 600.0` / scheduling / except block so it reads:

```python
        _timeout_s = 600.0
        self._read_idle_s = read_idle_s if read_idle_s is not None else _read_idle_default()
        ...
        try:
            future = ex.submit(lambda: asyncio.run(coro_factory()))
            try:
                text, usage = future.result(timeout=_timeout_s)
            except _SubRlmReadIdleTimeout as stall:
                self._kill_wedged_children(_pre_pids)
                self._notify_stall(stall.idle_s)
                logger.warning(
                    "rlm_query: SUB_RLM_STALL read-idle %.0fs — killed wedged child(ren); "
                    "returning sentinel so the root can retry.", stall.idle_s,
                )
                return _stall_message(stall.idle_s)
            except concurrent.futures.TimeoutError:
                self._kill_wedged_children(_pre_pids)
                self._notify_stall(_timeout_s)
                logger.warning(
                    "rlm_query: SUB_RLM_STALL total %.0fs — killed wedged child(ren); "
                    "returning sentinel.", _timeout_s,
                )
                return _stall_message(_timeout_s)
            self._last_usage = usage
            return text
        finally:
            ex.shutdown(wait=False)
```

Add the `read_idle_s` parameter to `complete`'s signature: `def complete(self, *, system: str, user: str, read_idle_s: float | None = None) -> str:`.

- [ ] **Step 4: Run the test — expect PASS**

Run: `.venv/bin/python -m pytest tests/test_rlm_query_read_idle.py::test_read_idle_stall_returns_sentinel_not_empty -xvs`
Expected: PASS.

- [ ] **Step 5: Write the salvage test — partial text beats the sentinel**

Append to `tests/test_rlm_query_read_idle.py`:

```python
def test_read_idle_salvages_partial_text(monkeypatch):
    from backend.services.context.workspace.tools import rlm_query as rq

    async def partial_then_hang(prompt, options):
        # Build messages from the fake module installed below.
        import claude_agent_sdk as sdk
        yield sdk.AssistantMessage(content=[type("B", (), {"text": "partial answer"})()])
        await asyncio.sleep(60)  # then stall before ResultMessage
    _install_fake_sdk(monkeypatch, partial_then_hang)
    monkeypatch.setattr(rq, "_bundled_claude_child_pids", lambda: set())

    client = rq.ClaudeLlmClient(model="claude-test", max_turns=1)
    result = client.complete(system="s", user="u", read_idle_s=0.2)
    assert result == "partial answer"
    assert rq.SUB_RLM_STALL_SENTINEL not in result
```

- [ ] **Step 6: Run it — expect PASS** (no code change needed; salvage branch already returns partial)

Run: `.venv/bin/python -m pytest tests/test_rlm_query_read_idle.py -xvs`
Expected: both PASS.

- [ ] **Step 7: Write the stall-sink + kill-hardening tests**

Append:

```python
def test_stall_event_sink_receives_event(monkeypatch):
    from backend.services.context.workspace.tools import rlm_query as rq
    async def hanging(prompt, options):
        await asyncio.sleep(60); yield
    _install_fake_sdk(monkeypatch, hanging)
    monkeypatch.setattr(rq, "_bundled_claude_child_pids", lambda: set())
    sink: list[dict] = []
    client = rq.ClaudeLlmClient(model="m", max_turns=1, stall_event_sink=sink.append)
    client.complete(system="s", user="u", read_idle_s=0.2)
    assert len(sink) == 1
    assert sink[0]["event"] == "sub_rlm_stalled"
    assert sink[0]["idle_seconds"] >= 0.2 - 0.01


def test_kill_helpers_are_fail_soft(monkeypatch):
    from backend.services.context.workspace.tools import rlm_query as rq
    client = rq.ClaudeLlmClient(model="m", max_turns=1)
    # No children, broken pgrep — must not raise.
    monkeypatch.setattr(rq, "_bundled_claude_child_pids", lambda: {999999})
    client._kill_wedged_children(set())  # 999999 is "new" — kill attempt must swallow OSError
    client._notify_stall(1.0)            # sink None — must not raise
```

- [ ] **Step 8: Implement `_kill_wedged_children`, `_notify_stall`, hardened kill**

Add these methods to `ClaudeLlmClient` (refactor the inline kill in the old timeout path into `_kill_wedged_children`):

```python
    def _kill_wedged_children(self, pre_pids: set[int]) -> None:
        """SIGKILL bundled-claude children spawned by THIS call (fail-soft, cross-platform).

        The SDK spawns the CLI outside our process group, so we can't killpg the
        group we own; we diff post-vs-pre child PIDs and SIGKILL the new ones, with
        a best-effort killpg for any grandchildren. Non-POSIX or pgrep-less hosts
        no-op. Never raises (D3).
        """
        import os as _os
        import signal as _signal
        import time as _time
        if not hasattr(_os, "kill"):
            return
        try:
            wedged = _bundled_claude_child_pids() - pre_pids
        except Exception:  # noqa: BLE001
            return
        for pid in wedged:
            try:
                try:  # best-effort grandchildren via the child's own group
                    _os.killpg(_os.getpgid(pid), _signal.SIGKILL)
                except (OSError, ProcessLookupError, AttributeError):
                    pass
                _os.kill(pid, _signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
        if wedged:
            _time.sleep(0.2)  # let the OS reap zombies before the next snapshot

    def _notify_stall(self, idle_s: float) -> None:
        """Best-effort: push a sub_rlm_stalled event dict to the sink (Phase 2 wires it)."""
        if self._stall_event_sink is None:
            return
        try:
            self._stall_event_sink({
                "event": "sub_rlm_stalled",
                "model": self._model or "unknown",
                "idle_seconds": float(idle_s),
            })
        except Exception:  # noqa: BLE001 — observability never blocks
            logger.debug("rlm_query: stall_event_sink raised", exc_info=True)
```

- [ ] **Step 9: Run the full Phase-1 test file + the regression slice**

Run: `.venv/bin/python -m pytest tests/test_rlm_query_read_idle.py tests/ -k "rlm_query or claude_oauth" -q`
Expected: PASS, regression count ≥ Phase 0 floor.

- [ ] **Step 10: Teach the root to handle the sentinel (system prompt, 4 lines)**

In `backend/agents/rlm/system_prompt.py`, inside `_PRIMITIVES_SECTION` (after the `rlm_query` guidance ~line 159), add:

```python
"\nIf an `rlm_query` / `llm_query` result contains the marker `[SUB_RLM_STALL]`, the "
"sub-query stalled and was aborted — it is NOT an answer. Retry that one call with a "
"smaller slice, or reduce how many sub-calls you dispatch at once. Never write the "
"stall marker into the final report.\n"
```

- [ ] **Step 11: Commit**

```bash
git add backend/services/context/workspace/tools/rlm_query.py backend/agents/rlm/system_prompt.py tests/test_rlm_query_read_idle.py
git commit -m "feat(rlm): read-idle timeout + typed sub-RLM stall sentinel + hardened kill (FM-001/002/007)"
```

**Acceptance:** the root receives `[SUB_RLM_STALL]…` not `""` (FM-002, **guaranteed**); partial text is preferred; descendants are killed fail-soft **per-pid (never `killpg` — the SDK child shares our process group, so `killpg` would SIGKILL the backend)**; no orphaned child after the call. The **≤120s read-idle bound is best-effort**: `asyncio.wait_for` awaits the cancelled `__anext__`, which may hit the documented Defect-2 (`transport.close()` futex hang) cleanup path and degrade to the **600s hard backstop** (still bounded; the `future.result(600)` always fires). **Verification gate (do NOT claim "120s" until met):** run the E2E with an induced stall (a `tc`/proxy that half-opens the socket to api.anthropic.com against the *real* SDK) and confirm the run unsticks in ≈`read_idle_s`, not 600s. If it degrades, the fast bound needs a different mechanism (e.g. a separate idle-watchdog that abandons the worker), tracked as a follow-up. **Rollback:** `REPROLAB_SUBRLM_READ_IDLE_S=0` disables the read-idle path (reverts to total-time-only).

---

## Phase 2: Run heartbeat + dangling-subcall detector + status reconciliation

**Problem:** A wedge was invisible for 70 min — the SDAR run shows a 76-min gap with a dangling `sub_rlm_spawned` (5 spawned / 4 complete) and only 1 `iteration_heartbeat` in 89 min. `demo_status.updatedAt` is stamped only at lifecycle transitions (`run.py:601`); the cost loop (`run.py:992-1006`) rewrites the file every 30s but never touches `updatedAt`, so it reads false-stale (FM-005) and a wedge is undetectable (FM-003/006).

**Architecture:** (a) Register a new `sub_rlm_stalled` SSE event. (b) A `SubcallTracker` wraps the existing `on_subcall_start`/`on_subcall_complete` callbacks to record open sub-calls; a daemon timer thread emits `sub_rlm_stalled` + stamps `demo_status` when any open sub-call exceeds `STALL_DETECT_S` (default 120s). (c) The cost loop also stamps `updatedAt` so freshness advances every 30s. (d) Wire Phase 1's `stall_event_sink` to `emit` so a killed stall is also surfaced.

**Files:**
- Modify: `backend/agents/rlm/sse_bridge.py` (new event builder + `make_on_subcall_*` wrap with tracker)
- Modify: `frontend/src/lib/events/rlm-events.ts` (register event type + interface + union)
- Modify: `backend/agents/rlm/run.py` (instantiate tracker + timer thread; stamp `updatedAt` in cost loop)
- Test: `tests/agents/rlm/test_subcall_stall_detector.py` (create)

- [ ] **Step 1: Write the failing test — the detector flags a dangling sub-call**

Create `tests/agents/rlm/test_subcall_stall_detector.py`:

```python
"""Phase 2 — dangling sub_rlm_spawned detector emits sub_rlm_stalled."""
from __future__ import annotations
import time
import pytest


def test_tracker_emits_stalled_for_open_subcall():
    from backend.agents.rlm.sse_bridge import SubcallTracker
    emitted: list[dict] = []
    tracker = SubcallTracker(emit=emitted.append, stall_after_s=0.2)

    tracker.on_start(depth=1, model="claude-haiku-4-5", prompt_preview="navigate §3")
    # Simulate the watchdog tick BEFORE completion, after the stall window.
    time.sleep(0.25)
    tracker.check_once()                      # one synchronous poll
    assert any(e["event"] == "sub_rlm_stalled" for e in emitted)
    assert emitted[-1]["depth"] == 1
    assert emitted[-1]["idle_seconds"] >= 0.2

    # A second poll must NOT double-emit for the same open sub-call.
    emitted.clear()
    tracker.check_once()
    assert emitted == []


def test_tracker_no_stall_when_completed():
    from backend.agents.rlm.sse_bridge import SubcallTracker
    emitted: list[dict] = []
    tracker = SubcallTracker(emit=emitted.append, stall_after_s=0.2)
    tracker.on_start(depth=1, model="m", prompt_preview="x")
    tracker.on_complete(depth=1, model="m", duration=0.01, error=None)
    time.sleep(0.25)
    tracker.check_once()
    assert emitted == []
```

- [ ] **Step 2: Run it — expect ImportError (`SubcallTracker` missing)**

Run: `.venv/bin/python -m pytest tests/agents/rlm/test_subcall_stall_detector.py -xvs`
Expected: FAIL — `ImportError: cannot import name 'SubcallTracker'`.

- [ ] **Step 3: Add the event builder + `SubcallTracker` to `sse_bridge.py`**

Add the builder (after `build_run_warning_event`, ~line 716):

```python
def build_sub_rlm_stalled_event(*, depth: int, model: str, idle_seconds: float) -> dict:
    """A depth>=1 sub-call has made no progress for idle_seconds — a likely wedge."""
    return {
        "event": "sub_rlm_stalled",
        "timestamp": _now_iso(),
        "depth": depth,
        "model": model,
        "idle_seconds": float(idle_seconds),
    }
```

Add `"build_sub_rlm_stalled_event"` and `"SubcallTracker"` to `__all__`. Add the tracker class:

```python
class SubcallTracker:
    """Track open depth>=1 sub-calls; emit sub_rlm_stalled when one wedges.

    Wraps the rlm on_subcall_start/on_subcall_complete callbacks. A daemon poller
    (run.py) calls check_once() every poll_interval; any open sub-call older than
    stall_after_s emits a single sub_rlm_stalled event (dedup per open call).
    Thread-safe; fail-soft.
    """
    def __init__(self, emit: "Callable[[dict], None]", stall_after_s: float = 120.0) -> None:
        self._emit = emit
        self._stall_after_s = stall_after_s
        self._open: dict[int, tuple[float, str]] = {}     # depth -> (start_monotonic, model)
        self._flagged: set[int] = set()
        self._lock = threading.Lock()

    def on_start(self, depth: int, model: str, prompt_preview: str) -> None:
        with self._lock:
            self._open[depth] = (time.monotonic(), model)
            self._flagged.discard(depth)

    def on_complete(self, depth: int, model: str, duration: float, error: object) -> None:
        with self._lock:
            self._open.pop(depth, None)
            self._flagged.discard(depth)

    def check_once(self) -> None:
        now = time.monotonic()
        to_flag: list[tuple[int, str, float]] = []
        with self._lock:
            for depth, (start, model) in self._open.items():
                idle = now - start
                if idle >= self._stall_after_s and depth not in self._flagged:
                    self._flagged.add(depth)
                    to_flag.append((depth, model, idle))
        for depth, model, idle in to_flag:
            try:
                self._emit(build_sub_rlm_stalled_event(depth=depth, model=model, idle_seconds=idle))
            except Exception:  # noqa: BLE001 — observability never blocks
                logger.debug("SubcallTracker: emit failed", exc_info=True)
```

(Ensure `import threading`, `import time`, and `Callable` are imported at the top of `sse_bridge.py`.)

- [ ] **Step 4: Run the tests — expect PASS**

Run: `.venv/bin/python -m pytest tests/agents/rlm/test_subcall_stall_detector.py -xvs`
Expected: both PASS.

- [ ] **Step 5: Register the event type in the frontend contract**

In `frontend/src/lib/events/rlm-events.ts`: add `"sub_rlm_stalled"` to the `RLM_EVENT_TYPES` array (266-288); add the interface near `SubRlmSpawnedEvent`:

```typescript
export interface SubRlmStalledEvent {
  event: "sub_rlm_stalled";
  timestamp: string;
  depth: number;
  model: string;
  idle_seconds: number;
}
```

and add `| SubRlmStalledEvent` to the `RlmDashboardEvent` union.

- [ ] **Step 6: Verify the frontend still type-checks**

Run: `cd frontend && npx tsc --noEmit`
Expected: no new errors.

- [ ] **Step 7: Wire the tracker + poller + `updatedAt` stamp in `run.py`**

In `run.py` where the RLM is constructed (~1462-1463), replace the direct callback wiring:

```python
    from backend.agents.rlm.sse_bridge import SubcallTracker
    _stall_after_s = float(os.environ.get("REPROLAB_STALL_DETECT_S", "120") or "120")
    _subcall_tracker = SubcallTracker(emit=emit, stall_after_s=_stall_after_s) if _stall_after_s > 0 else None

    def _wrapped_subcall_start(depth, model, prompt_preview):
        if _subcall_tracker is not None:
            _subcall_tracker.on_start(depth, model, prompt_preview)
        make_on_subcall_start(emit)(depth, model, prompt_preview)

    def _wrapped_subcall_complete(depth, model, duration, error):
        if _subcall_tracker is not None:
            _subcall_tracker.on_complete(depth, model, duration, error)
        make_on_subcall_complete(emit)(depth, model, duration, error)
    # ... RLM(..., on_subcall_start=_wrapped_subcall_start, on_subcall_complete=_wrapped_subcall_complete)
```

Start a daemon poller alongside the existing watchdog arm (~1467), and stamp `demo_status` on stall:

```python
    _stall_poller_stop = threading.Event()
    def _poll_subcall_stalls():
        while not _stall_poller_stop.wait(min(_stall_after_s, 30.0)):
            if _subcall_tracker is not None:
                _subcall_tracker.check_once()
                _stamp_demo_status_updated(project_dir)  # advance updatedAt = liveness
    if _subcall_tracker is not None:
        threading.Thread(target=_poll_subcall_stalls, name="subcall-stall-poller", daemon=True).start()
    # In the finally block that cancels the watchdog: _stall_poller_stop.set()
```

Add the cost-loop `updatedAt` fix — in `_update_cost_summary_loop` (~run.py:1003) where it sets `existing["cost_summary"] = summary`, also add `existing["updatedAt"] = datetime.now(timezone.utc).isoformat()`. Add a small helper `_stamp_demo_status_updated(project_dir)` that atomically rewrites `demo_status.json` with a fresh `updatedAt` (mirror the `os.replace` pattern at `live_runs.py:407-415`), fail-soft.

- [ ] **Step 8: Write the integration test — `updatedAt` advances during a run**

Create `tests/agents/rlm/test_demo_status_freshness.py`:

```python
def test_stamp_demo_status_advances_updatedat(tmp_path):
    import json, time
    from backend.agents.rlm.run import _stamp_demo_status_updated
    p = tmp_path / "demo_status.json"
    p.write_text(json.dumps({"status": "running", "updatedAt": "2026-05-30T00:00:00+00:00"}))
    _stamp_demo_status_updated(tmp_path)
    after = json.loads(p.read_text())["updatedAt"]
    assert after != "2026-05-30T00:00:00+00:00"
```

- [ ] **Step 9: Run Phase-2 tests + regression**

Run: `.venv/bin/python -m pytest tests/agents/rlm/test_subcall_stall_detector.py tests/agents/rlm/test_demo_status_freshness.py tests/ -k "sse_bridge or run_watchdog" -q`
Expected: PASS.

- [x] **Step 10: Wire Phase 1's stall sink to `emit` — DEFERRED (rationale recorded).**

Decision (2026-05-30): NOT wired in this commit. `ClaudeOauthClient`/`ClaudeLlmClient` are built by rlm's backend factory (`_oauth_backend_patch.get_client`) from `backend_kwargs` with no `emit` in scope; reaching it needs either a thread-local/contextvar emit (uncertain propagation across rlm's sub-call worker threads) or an invasive factory change. The **Phase-2 poller is the robust primary signal** and uniquely handles the case that matters most — when Phase 1's read-idle degrades to the 600s backstop (advisor concern #2), the poller still emits `sub_rlm_stalled` at ~120s. The only gap left open: a read-idle stall that Phase 1 resolves cleanly at ~120s shows on the dashboard as a normal `sub_rlm_complete` (the `stall_event_sink` hook remains available to close this later via a contextvar emit). Tracked as a follow-up, not blocking.

- [ ] **Step 11: Commit**

```bash
git add backend/agents/rlm/sse_bridge.py backend/agents/rlm/run.py backend/agents/rlm/claude_oauth_client.py frontend/src/lib/events/rlm-events.ts tests/agents/rlm/test_subcall_stall_detector.py tests/agents/rlm/test_demo_status_freshness.py
git commit -m "feat(rlm): dangling sub-call detector + sub_rlm_stalled event + demo_status freshness (FM-003/005/006)"
```

**Acceptance:** an open sub-call >120s emits exactly one `sub_rlm_stalled` (dedup); `demo_status.updatedAt` advances ≥ every 30s while running; frontend type-checks with the new event. **Rollback:** `REPROLAB_STALL_DETECT_S=0` disables the poller.

---

## Phase 3: Final-report evidence gate

**Problem:** Every finished `/runs` report ships `baseline_metrics={}` + rubric 0 yet `verdict="partial"`; pb_…784 even claims "implemented and executed" with **no** `run_experiment` in its trace. The forced-iteration floor is iteration-**count** based and satisfied by 2 empty iterations (FM-004).

**Architecture:** A **write-time** gate in `write_final_report_rlm`: if `verdict ∈ {reproduced, partial}` AND `baseline_metrics == {}` AND there is no experiment evidence (`experiment_runs.jsonl` has no entry with non-empty `metrics`), downgrade `verdict` to `"failed"` and append an `evidence_gap` note to `reproduction_summary`. This is path-agnostic (covers clean FINAL_VAR, watchdog, and fatal-abort writers). Gated by `REPROLAB_EVIDENCE_GATE` (default on). Single concern — no forced-iteration changes here.

**Files:**
- Modify: `backend/agents/rlm/report.py` (`write_final_report_rlm` + a small evidence-check helper)
- Test: `tests/agents/rlm/test_evidence_gate.py` (create)

- [ ] **Step 1: Write the failing test — partial with no evidence is downgraded**

Create `tests/agents/rlm/test_evidence_gate.py`:

```python
"""Phase 3 — final-report evidence gate."""
from __future__ import annotations
import json
import pytest
from backend.agents.rlm.report import RLMFinalReport, write_final_report_rlm


def test_partial_without_evidence_downgrades_to_failed(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_EVIDENCE_GATE", "1")
    # No experiment_runs.jsonl at all.
    report = RLMFinalReport(verdict="partial", reproduction_summary="did stuff", baseline_metrics={})
    json_path, _ = write_final_report_rlm(report, tmp_path)
    data = json.loads(json_path.read_text())
    assert data["verdict"] == "failed"
    assert "evidence_gap" in data["reproduction_summary"]


def test_partial_with_metrics_evidence_is_kept(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_EVIDENCE_GATE", "1")
    (tmp_path / "experiment_runs.jsonl").write_text(
        json.dumps({"success": True, "metrics": {"accuracy": 0.41}}) + "\n"
    )
    report = RLMFinalReport(verdict="partial", reproduction_summary="ran it",
                            baseline_metrics={"accuracy": 0.41})
    json_path, _ = write_final_report_rlm(report, tmp_path)
    assert json.loads(json_path.read_text())["verdict"] == "partial"


def test_gate_disabled_keeps_verdict(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_EVIDENCE_GATE", "0")
    report = RLMFinalReport(verdict="partial", reproduction_summary="x", baseline_metrics={})
    json_path, _ = write_final_report_rlm(report, tmp_path)
    assert json.loads(json_path.read_text())["verdict"] == "partial"
```

- [ ] **Step 2: Run it — expect FAIL (gate not implemented)**

Run: `.venv/bin/python -m pytest tests/agents/rlm/test_evidence_gate.py -xvs`
Expected: `test_partial_without_evidence_downgrades_to_failed` FAILs (verdict stays "partial").

- [ ] **Step 3: Implement the gate in `report.py`**

Add a helper (reuse the logic from `run.py:724-749`, kept local to avoid a circular import):

```python
def _has_experiment_evidence(project_dir) -> bool:
    """True iff experiment_runs.jsonl has any entry with a non-empty metrics dict."""
    from pathlib import Path
    path = Path(project_dir) / "experiment_runs.jsonl"
    if not path.exists():
        return False
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            metrics = entry.get("metrics")
            if isinstance(metrics, dict) and metrics:
                return True
    except OSError:
        return False
    return False


def _apply_evidence_gate(report: "RLMFinalReport", project_dir) -> "RLMFinalReport":
    """Downgrade a success-ish verdict that has no experiment evidence (FM-004)."""
    import os
    if os.environ.get("REPROLAB_EVIDENCE_GATE", "1").strip().lower() in {"0", "false", "off"}:
        return report
    if report.verdict in {"reproduced", "partial"} and not report.baseline_metrics \
            and not _has_experiment_evidence(project_dir):
        report.verdict = "failed"
        note = (" [evidence_gap: downgraded to failed — no run_experiment produced metrics "
                "and baseline_metrics is empty].")
        report.reproduction_summary = (report.reproduction_summary or "").rstrip() + note
    return report
```

Call it at the top of `write_final_report_rlm(report, project_dir)`, before serialization:

```python
def write_final_report_rlm(report: RLMFinalReport, project_dir) -> tuple[Path, Path]:
    report = _apply_evidence_gate(report, project_dir)
    ...  # existing serialization unchanged
```

(`RLMFinalReport` is a pydantic BaseModel, so `report.verdict = "failed"` mutates in place — confirm mutation is allowed; if `model_config` forbids it, use `report = report.model_copy(update={"verdict": "failed", ...})`.)

- [ ] **Step 4: Run the tests — expect PASS**

Run: `.venv/bin/python -m pytest tests/agents/rlm/test_evidence_gate.py -xvs`
Expected: all three PASS.

- [ ] **Step 5: Run the report regression slice**

Run: `.venv/bin/python -m pytest tests/ -k "report" -q`
Expected: PASS (the watchdog/fatal-abort writers, which also call `write_final_report_rlm`, now inherit the gate — verify their tests still pass; if a test asserts a `partial` with no evidence, it documented the bug and should be updated to expect `failed`).

- [ ] **Step 6: Commit**

```bash
git add backend/agents/rlm/report.py tests/agents/rlm/test_evidence_gate.py
git commit -m "feat(rlm): final-report evidence gate — no success verdict without experiment metrics (FM-004)"
```

**Acceptance:** a pb_…784-shaped report (partial, empty metrics, no `run_experiment`) is written as `verdict="failed"` with an `evidence_gap` note; a report with real metrics is untouched; all write paths inherit the gate. **Rollback:** `REPROLAB_EVIDENCE_GATE=0`.

---

## Phase 4: Route nav sub-RLM to gpt-5-mini (timeout + telemetry)

**Problem:** The hot-volume depth-1 sub-RLM (`rlm_query`/`llm_query`) runs on Haiku via the bundled CLI — the no-read-idle, orphaned-child transport that wedged (FM-001), and the prompt-driven 8-way fan-out multiplies that risk (FM-008).

**Architecture:** Use the **existing** accelerator hook (`run.py:1436-1444`) with the generic `endpoint` provider to route nav to **OpenAI gpt-5-mini**, which rides the `openai`/httpx transport (raises `ReadTimeout`, auto-retries 2×, leaks no subprocess — DOCUMENTED). Two small code changes: (a) thread a read timeout into the sub-backend kwargs (flows to `openai.OpenAI(timeout=…)` via rlm `OpenAIClient`/`BaseLM` — verified `rlm/clients/openai.py:43`, `base_lm.py:11`); (b) a convenience fallback so `api.openai.com` picks up `OPENAI_API_KEY`. Grader/`verify_against_rubric` stay on Sonnet-OAuth (`scope=navigation`). OFF by default; opt-in via env. User has OpenAI credits.

**Files:**
- Modify: `backend/agents/rlm/run.py:1436-1444` (add `"timeout"` to `_other_backend_kwargs`)
- Modify: `backend/agents/rlm/accelerator.py` (`_resolve_endpoint` OPENAI_API_KEY fallback)
- Modify: `CLAUDE.md` (document the env recipe)
- Test: `tests/agents/rlm/test_accelerator_openai_route.py` (create)

- [ ] **Step 1: Write the failing test — endpoint resolves OpenAI key + timeout flows through**

Create `tests/agents/rlm/test_accelerator_openai_route.py`:

```python
"""Phase 4 — gpt-5-mini accelerator route."""
from __future__ import annotations
import pytest
from backend.agents.rlm import accelerator as acc


def test_endpoint_falls_back_to_openai_api_key(monkeypatch):
    monkeypatch.setenv("REPROLAB_ACCELERATOR", "endpoint")
    monkeypatch.setenv("REPROLAB_ACCELERATOR_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("REPROLAB_ACCELERATOR_MODEL", "gpt-5-mini")
    monkeypatch.delenv("REPROLAB_ACCELERATOR_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-xyz")
    monkeypatch.setattr(acc, "probe_endpoint", lambda *a, **k: True)  # skip network
    ep = acc.resolve_accelerator("endpoint")
    assert ep is not None
    assert ep.model == "gpt-5-mini"
    assert ep.api_key == "sk-test-xyz"      # fell back to OPENAI_API_KEY, not "local"
```

- [ ] **Step 2: Run it — expect FAIL (api_key would be "local")**

Run: `.venv/bin/python -m pytest tests/agents/rlm/test_accelerator_openai_route.py -xvs`
Expected: FAIL — `assert "local" == "sk-test-xyz"`.

- [ ] **Step 3: Add the OPENAI_API_KEY fallback in `_resolve_endpoint`**

In `accelerator.py::_resolve_endpoint`, after reading `api_key = os.environ.get("REPROLAB_ACCELERATOR_API_KEY", "local")`:

```python
    # Convenience: targeting OpenAI's public endpoint with no explicit accelerator key
    # falls back to OPENAI_API_KEY (the user's existing credits). Other hosts keep "local".
    if api_key in ("", "local") and "api.openai.com" in base_url:
        api_key = os.environ.get("OPENAI_API_KEY", api_key)
```

- [ ] **Step 4: Run it — expect PASS**

Run: `.venv/bin/python -m pytest tests/agents/rlm/test_accelerator_openai_route.py::test_endpoint_falls_back_to_openai_api_key -xvs`
Expected: PASS.

- [ ] **Step 5: Write the timeout-flow test**

Append:

```python
def test_other_backend_kwargs_carry_read_timeout(monkeypatch):
    """The sub-backend kwargs built for an active accelerator include a timeout."""
    from backend.agents.rlm.run import _build_accel_sub_backend_kwargs  # extract helper in Step 6
    ep = acc.AcceleratorEndpoint(base_url="https://api.openai.com/v1", model="gpt-5-mini",
                                 api_key="sk-x", kind="endpoint", is_azure=False)
    monkeypatch.setenv("REPROLAB_SUBRLM_OPENAI_TIMEOUT_S", "90")
    kwargs = _build_accel_sub_backend_kwargs(ep)
    assert kwargs["model_name"] == "gpt-5-mini"
    assert kwargs["timeout"] == 90.0
```

- [ ] **Step 6: Extract a tiny helper + add the timeout in `run.py`**

Refactor `run.py:1436-1444` to a testable helper and add the timeout:

```python
def _build_accel_sub_backend_kwargs(accel_ep) -> dict:
    """Sub-backend kwargs for an active (non-Azure) accelerator, with a read timeout
    that flows to openai.OpenAI(timeout=...) via rlm's OpenAIClient/BaseLM."""
    import os
    timeout_s = float(os.environ.get("REPROLAB_SUBRLM_OPENAI_TIMEOUT_S", "120") or "120")
    return {
        "model_name": accel_ep.model,
        "base_url": accel_ep.base_url,
        "api_key": accel_ep.api_key,
        "timeout": timeout_s,
    }
```

and at the call site:

```python
    if _accel_ep is not None and not _accel_ep.is_azure:
        _other_backends = ["openai"]
        _other_backend_kwargs = [_build_accel_sub_backend_kwargs(_accel_ep)]
```

- [ ] **Step 7: Run tests + regression**

Run: `.venv/bin/python -m pytest tests/agents/rlm/test_accelerator_openai_route.py tests/ -k "accelerator" -q`
Expected: PASS.

- [ ] **Step 8: Document the opt-in recipe in `CLAUDE.md`**

Under the accelerator/sandbox section, add:

```
### gpt-5-mini navigation route (Phase 4)
Route the hot-volume rlm_query/llm_query navigation off the bundled CLI to OpenAI gpt-5-mini
(keeps grader/verify on Sonnet-OAuth). Opt-in:
  REPROLAB_ACCELERATOR=endpoint
  REPROLAB_ACCELERATOR_BASE_URL=https://api.openai.com/v1
  REPROLAB_ACCELERATOR_MODEL=gpt-5-mini
  REPROLAB_ACCELERATOR_SCOPE=navigation        # default; do NOT use "all" (grader stays Sonnet)
  REPROLAB_SUBRLM_OPENAI_TIMEOUT_S=120          # read-idle bound on the openai transport
  # OPENAI_API_KEY is used automatically for api.openai.com.
Rollback: REPROLAB_ACCELERATOR=off.
```

- [ ] **Step 9: Commit**

```bash
git add backend/agents/rlm/run.py backend/agents/rlm/accelerator.py CLAUDE.md tests/agents/rlm/test_accelerator_openai_route.py
git commit -m "feat(rlm): gpt-5-mini nav route via accelerator endpoint + read timeout (FM-001/008)"
```

**Acceptance:** with the env recipe set, nav sub-calls go to gpt-5-mini over the openai/httpx transport with a 120s read bound and OPENAI_API_KEY auth; grader stays on Sonnet. **A/B (uses Phase 2 telemetry):** run SDAR with accelerator on vs off; compare `final_report.json::rubric.overall_score`, wall-clock, and `sub_rlm_stalled` incidence across ≥3 paired runs. **Rollback:** `REPROLAB_ACCELERATOR=off`.

---

## Phase 5: Code-level sub-RLM concurrency cap

**Problem:** The f848d7e system prompt tells the root to fan out up to 8 concurrent sub-RLMs (`system_prompt.py:295-317`), but the cap is prompt-only — unbounded in code, multiplying FM-001 and OAuth-slot pressure (FM-008). `rlm_query`/`llm_query` are pure rlm builtins (not wrapped in `binding.py`), so the cap cannot live there; the single chokepoint every sub-call crosses is `ClaudeLlmClient.complete`.

**Architecture:** A module-level bounded `threading.Semaphore` in `rlm_query.py`, acquired around the worker submission in `complete()`. Size from `REPROLAB_SUBRLM_MAX_CONCURRENCY` (default 4). When the accelerator route (Phase 4) is active, sub-calls go through the openai client instead — so the cap is a no-op there; document that it bounds the OAuth/bundled-CLI path specifically.

**Files:** Modify `backend/services/context/workspace/tools/rlm_query.py`; Test `tests/test_rlm_query_concurrency_cap.py`.

- [ ] Step 1: Failing test — N+2 concurrent `complete()` calls never exceed N in-flight (instrument an in-flight counter on a stubbed `_async_complete` that sleeps).
- [ ] Step 2: Run → FAIL.
- [ ] Step 3: Add module-level `_SUBRLM_SEMAPHORE = threading.BoundedSemaphore(int(os.environ.get("REPROLAB_SUBRLM_MAX_CONCURRENCY","4") or "4"))` (rebuilt lazily so tests can set the env); wrap the `future.result` block in `with _subrlm_semaphore():`. `0`/unset-large disables (use a no-op context manager when value ≤ 0).
- [ ] Step 4: Run → PASS.
- [ ] Step 5: Regression `pytest -k "rlm_query"`; commit `feat(rlm): bound concurrent sub-RLM calls (FM-008)`.

**Acceptance:** at most `N` `complete()` calls run concurrently; default 4. **Validation:** the in-flight-counter test. **Risk:** too-low cap slows legit fan-out → default 4 (≥ the prompt's typical batch). **Rollback:** `REPROLAB_SUBRLM_MAX_CONCURRENCY=0`.

---

## Phase 6: Prompt-cache leak — measure, then fix the OAuth path

**Problem (decisive data, Section E):** Across **every** `cost_ledger.jsonl` in `runs/`, `cache_read_input_tokens = 0` — the one real LLM entry shows `cache_create=59477, cache_read=0`. The ~32KB system prompt is **written to cache but never read back**, so it is re-billed every iteration. The API-key path has an explicit `cache_control` wrapper (`_oauth_backend_patch.py:103,113-186`), but D3 skips the default `claude-oauth`/SDK path — and the data shows nothing is landing there. This is a real cost leak, not a hypothetical.

**Architecture:** Two steps. (6.1) **Measure** — a per-run cache-hit ratio log to confirm the cause (TTL expiry across long inter-iteration gaps vs. the OAuth path never sending a cache breakpoint). (6.2) **Fix** — first verify `claude-agent-sdk` passes a structured system block through (probe the SDK message/options shape); if it does, add an explicit `cache_control:{type:"ephemeral"}` breakpoint on the OAuth system block (mirror `_wrap_system_with_cache_control`), D5 no-op-on-error. If the SDK does NOT accept structured system blocks, close 6.2 as not-actionable on the bundled-CLI path and record that finding.

**Files:** `backend/agents/rlm/run.py` (ratio log at finalize, flag `REPROLAB_LOG_CACHE_TOKENS`), `backend/services/context/workspace/tools/rlm_query.py` / OAuth system-block construction, `resilience/cost.py` (counter). Test `tests/agents/rlm/test_cache_ratio_log.py`.

- [ ] 6.1 Step 1: Failing test — `_cache_hit_ratio(project_dir)` over a stub `cost_ledger.jsonl` returns `cache_read/(cache_read+input_tokens)`. Step 2: Run→FAIL. Step 3: implement + log at finalize when flag set. Step 4: Run→PASS. Step 5: commit `chore(rlm): per-run prompt-cache hit ratio (opt #6 measurement)`.
- [ ] 6.2 Step 1: probe — does `claude-agent-sdk` accept a structured/list system block (vs plain str)? Write a unit that constructs `ClaudeAgentOptions(system_prompt=[{...cache_control...}])` and asserts it doesn't raise / is forwarded. Step 2: if yes, add the breakpoint behind D5 no-op; test that `cache_read_input_tokens>0` on iteration≥2 of a paired SDAR run. If no, document and stop. Commit `feat(rlm): cache_control breakpoint on OAuth system block (opt #6 fix)` or `docs: OAuth path cannot cache structured system block (opt #6 closed)`.

**Validation:** `cache_read_input_tokens > 0` on iteration ≥2 (currently 0 everywhere). **Risk:** low (D5 no-op-on-error established). **Rollback:** `REPROLAB_LOG_CACHE_TOKENS=0`; the breakpoint falls back to a plain string on any error.

---

## Phase 7g: Read-only run-replay / postmortem harness (Gap-3)

**Problem:** The entire `/runs` empirical audit was done by hand. There is no tool that ingests `dashboard_events.jsonl` + `demo_status.json` + `cost_ledger.jsonl` + `experiment_runs.jsonl` into a verdict. This harness makes the audit's north-star metric (time-to-detect a stalled subcall) measurable and repeatable, and gives a regression fixture for Phases 1–3.

**Architecture:** A read-only `scripts/replay_run.py <project_id>` (no LLM, no new persistence) that walks the per-run JSONL and prints: max inter-event gap, `sub_rlm_spawned`/`complete` pairing (dangling count), cache-read ratio, and any `verdict ∈ {partial,reproduced}` with empty `baseline_metrics` and no experiment evidence (the FM-004 mismatch).

**Files:** new `scripts/replay_run.py`; Test `tests/scripts/test_replay_run.py`.

- [ ] Step 1: Failing test — fixture dirs (copy the shapes of `prj_09047604e591d969` and `pb_…784`) ⇒ replay flags the 76-min gap + dangling spawn for the first and the empty-evidence `partial` for the second. Step 2: Run→FAIL. Step 3: implement the JSONL walkers + verdict printer. Step 4: Run→PASS. Step 5: commit `feat(scripts): read-only run-replay/postmortem harness (Gap-3)`.

**Acceptance:** `replay_run.py prj_09047604e591d969` reports the dangling spawn + 76-min gap; `replay_run.py pb_…784` reports the evidence/verdict mismatch. **Risk:** none (read-only). **Rollback:** delete the script (no runtime coupling).

---

## Phase 7: Infra hardening (six narrow, survey-grounded sub-commits)

**Problem & scope correction:** The survey (Section E) confirmed the repo **already** does most sandbox lifecycle correctly — `local_docker.py` bounds exec with `asyncio.wait_for` + `destroy()` (`stop(timeout=3)→remove(force=True)`) + `demux=True` log streaming; `runpod_backend.py` classifies capacity/balance/transient errors and shields the pod DELETE. So this phase is **six narrow gaps**, not a rewrite. Land each as an independent TDD sub-commit.

**7a — Frontend stale-"running" → "stalled" (FM-005 UI half).** Root cause OBSERVED: `rlm-header.tsx:83-90` computes `noSignalSecs` from heartbeat freshness but **never flips `status`** — a dead backend renders `running` with a pulsing dot forever. Fix is a pure render derivation (every primitive already exists: the `nowMs` 1s clock `rlm-lab.tsx:172-189`, `lastHeartbeatAt`, `HEARTBEAT_STALE_MS`):
```ts
const effectiveStatus =
  status === "running" && inFlightPrimitive === null && lastHeartbeatAt && heartbeatNowMs &&
  (heartbeatNowMs - new Date(lastHeartbeatAt).getTime()) > HEARTBEAT_STALE_MS
    ? "stalled" : status;
```
Add a `"stalled"` arm to `RlmRunStatus` + `statusTone` (warn tone, `pulse:false`); drive the pill from `effectiveStatus`. Gate on `inFlightPrimitive === null` so a legit long `implement_baseline` doesn't false-alarm. Files: `frontend/src/components/lab/rlm/rlm-header.tsx`. Test: vitest — stale heartbeat + no in-flight primitive ⇒ pill reads "stalled". **No flag** (pure UI, self-corrects on next heartbeat).

**7b — SSE disconnect handling.** `live_runs.py:696 stream_events` has no `is_disconnected()` check and no `try/except asyncio.CancelledError`; its signature doesn't receive the `Request`. Pass `request: Request` from `app.py:812` into `stream_events`; at the top of each `while` iteration `if await request.is_disconnected(): return`; wrap the body in `try/except asyncio.CancelledError: ... raise`. Frees the per-viewer file-poll loop when a tab closes. Test: a fake request whose `is_disconnected()` returns True ends the generator. **No flag.**

**7c — uvicorn graceful-shutdown timeout.** Add `--timeout-graceful-shutdown 30` to `docker/entrypoint.sh` + `start.sh` so a wedged in-flight SSE stream can't delay SIGTERM shutdown indefinitely (uvicorn waits forever with none set). Validation: SIGTERM during an open SSE stream terminates within ~30s. **No flag** (ops config).

**7d — Docker OOM detection.** The only Docker gap: after a container exits, `local_docker.py` does not read OOM state. After exit (esp. code 137), `container.reload(); oom = container.attrs["State"]["OOMKilled"]` and tag the result `failure_class="oom_killed"` (class already in `failure_classifier.py:31-51`) so `run_experiment` distinguishes OOM from a generic crash (feeds the GPU-escalation ladder). Files: `backend/services/runtime/local_docker.py`. Test: stub `attrs["State"]={"OOMKilled":True,"ExitCode":137}` ⇒ `failure_class=="oom_killed"`. **No flag.**

**7e — RunPod granular timeout + bounded retry.** Replace the scalar `httpx.AsyncClient(timeout=60)` (`runpod_backend.py:983-987`) with `httpx.Timeout(connect=10, read=30, write=10, pool=5)` so a hung connect fails fast; wrap the create `POST /pods` + status `GET` in a 3-attempt exp-backoff retry that fires **only** when the existing classifier returns `retryable=True` (never 401/403/balance). Files: `runpod_backend.py`. Test: a mocked transient-500-then-200 succeeds in ≤3 tries; a 403 does not retry. **No flag** (tightens existing behavior).

**7f — Provider request-id capture (Gap-2, observability).** No LLM call records a provider request id, so a wedge can't be correlated to Anthropic/OpenAI logs. Stamp whatever id the response exposes (`response._request_id` on the openai/anthropic SDK paths; `ResultMessage.session_id`/`uuid` on the bundled-CLI path — verify the SDK message shape) onto the telemetry record + `dashboard_events`; fall back to `messaging/envelope.new_correlation_id` when the provider gives none. Files: `claude_runtime.py`, `rlm_query.py`, `resilience/engine.py:328`. Test: a real-shaped `ResultMessage` ⇒ id present in the emitted record. **No flag** (additive field).

**Acceptance:** each sub-commit independently green + revertable; UI flips to "stalled" within `HEARTBEAT_STALE_MS` of silence with no in-flight primitive; a closed tab stops its poll loop; OOM is tagged distinctly; RunPod calls are bounded; every LLM call carries a correlatable id.

---

## Phase 8: PEEK context-map prototype — **✅ IMPLEMENTED (brainstorm superseded this sketch)**

> **SUPERSEDED (2026-05-30):** the brainstorm changed the keying from "write-once" to **union-per-field** — a deterministic heuristic called per paper section would clobber under write-once (SDAR's 3 model sizes / 3 environments collapse to the last one). Consumption is a `read_context_map()` primitive (not a REPL-injected dict); DELETE is dropped (the union model + the Phase 3 evidence gate cover contamination — see the design's §6). Locked design: `docs/superpowers/specs/2026-05-30-intra-run-context-map-design.md`; plan: `docs/superpowers/plans/2026-05-30-intra-run-context-map.md`. The sketch below is the pre-brainstorm version, kept for provenance.

**Problem:** The root re-issues `rlm_query`/`llm_query` navigation calls to rediscover the same paper facts each iteration (redundant LLM cost + drift). PEEK (arXiv 2605.19932, evaluated *on* RLM) caches a bounded orientation map and reports 93–145 fewer iterations.

**Architecture (smallest useful version — NO LLM distiller):** A bounded (~1.5KB) `runs/<id>/rlm_state/context_map.json` written **once** from the structured outputs primitives already return (`understand_section`, `extract_hyperparameters`, `detect_environment` — note `understand_section` is a pure heuristic, so writing the map is free). Exposed to the root as a REPL-readable dict + one system-prompt line: "check the context map before re-navigating." DELETE-capable (a wrong cached hyperparameter must be removable). OFF by default.

**Files:** `backend/agents/rlm/context_map.py` (new: write-once + bounded + DELETE), `backend/agents/rlm/primitives.py` (write hook after the 3 orientation primitives), `backend/agents/rlm/system_prompt.py` (reuse instruction), `backend/agents/rlm/run.py` (load/expose per run). Atomic write via the `tmp`+`os.replace` pattern (`primitives.py:943-945`).

- [ ] After brainstorm: TDD the map writer (bounded, write-once, validators mirror `primitive_cache._CACHE_VALIDATORS`), the DELETE path, and the reader injection. Flag `REPROLAB_CONTEXT_MAP` (default 0).

**Validation (against /runs):** count `primitive_call` events with `name ∈ {rlm_query-proxy, understand_section}` per run, SDAR, map-on vs map-off; target a measurable drop in redundant nav with **no `rubric.overall_score` regression**. **Risk (central):** a wrong cached fact poisons the final report → deterministic-first, DELETE-capable, flag-gated, never cache unverified single-query facts. **Rollback:** `REPROLAB_CONTEXT_MAP=0`.

---

## Phase 9: MUSE negative-lessons prototype — **REQUIRES BRAINSTORM**

> **Before implementing:** brainstorm the lesson schema, the promotion threshold, and the retirement policy. Prototype only — ship the *negative-lessons* half (the part MUSE does NOT do), not a full skill bank.

**Problem:** Failed runs leave no durable lesson; the same `failure_class` recurs across runs (e.g. prose-in-Dockerfile BUG-NEW-042, thin env spec). MUSE (arXiv 2605.27366) promotes/curates skills; the cheap, high-value half for ReproLab is a per-paper **negative-lessons** file mined post-run and injected into the next run's implementer prompt.

**Architecture:** A post-run distiller hook at `run.py:1747` (after `write_final_report_rlm`) reads `experiment_runs.jsonl` (`failure_class`, `suggested_fix`) + `final_report.verdict`/`rubric` and appends atomic, class-tagged lessons to `docs/lessons/<arxiv_id>-negative-lessons.json` (keyed by `ctx.arxiv_id`). On the next run, `baseline_implementation.py:~1708` (before the OPERATOR GUIDANCE block) injects the lessons as a guardrail string. Lessons are class-tagged (not free prose), capped in count/length, and retired when stale (a lesson whose `failure_class` has not recurred in N runs is dropped). OFF by default.

**Files:** `backend/agents/rlm/lesson_distiller.py` (new), `backend/agents/rlm/run.py:1747` (hook), `backend/agents/baseline_implementation.py:~1708` (injection). Per-paper key `ctx.arxiv_id` (`context.py:63`). Failure enum at `failure_classifier.py:31-51`.

- [ ] After brainstorm: TDD the distiller (lesson schema, dedup, cap, retire-on-stale), the post-run hook (runs after report write, fail-soft), and the prompt injection. Flag `REPROLAB_NEGATIVE_LESSONS` (default 0).

**Validation (against /runs):** does a class-tagged lesson reduce recurrence of its `failure_class` on the next SDAR run? Compare iterations-to-first-rubric-pass with/without, ≥3 paired runs; lesson earns its keep only if it cuts the failure without lowering `overall_score`. **Risk:** vague/stale lessons become prompt junk → class-tagged + capped + retire-on-stale; pytest-gate any positive skills before they ever ship (deferred — not this phase). **Rollback:** `REPROLAB_NEGATIVE_LESSONS=0`.

---

## Section E: Research + Context7 infra inputs (the evidence behind the phases)

### Table 1 — Paper-derived infra ideas

| Paper | Idea | ReproLab gap it addresses | Smallest impl | Validation | Verdict → Phase |
|--|--|--|--|--|--|
| **AutoScientists** (2605.28655) | Supervisor is **deterministic code in a heartbeat loop**, not an LLM | Wedge had no role to detect it (FM-003) | Code-level subcall stall watchdog | stall test catches a wedge an LLM peer can't | **adopt now → Phase 1/2** |
| **PEEK** (2605.19932) | Bounded ~1KB context map; evaluated **on RLM**; 93–145 fewer iters | Redundant `rlm_query`/`llm_query` nav + drift | Deterministic write-once map from primitive outputs, no LLM | `rlm_query` calls/run ↓ on SDAR, no rubric regress | **prototype → Phase 8** |
| **MUSE** (2605.27366) | Skill create→pytest-gate→register; `.memory.md` | Failures leave no durable lesson | Per-paper **negative-lessons** file → next implementer prompt | failure_class recurrence ↓ next run | **prototype (neg-lessons only) → Phase 9** |
| **BES** (2605.28814) | Recombination evolution; verifier = fitness | Cross-run strategy amnesia | Offline archive-as-prior, **no in-loop selection** | corr(rubric, invariant-pass) on /runs first | **reject in-loop; defer** |

**Why reject/defer:** an LLM peer cannot observe a sibling blocked inside a primitive (so multi-agent role-splitting would NOT have prevented the wedge); BES in-loop evolution multiplies ReproLab's documented-weak auto-rubric into confident fakery. `understand_section` is a pure heuristic (no LLM), so PEEK's savings are on `rlm_query`/`llm_query`, not `understand_section`.

### Table 2 — Context7-derived infra fixes (actual deps only)

| Dependency | Documented behavior (cited) | Current repo usage | Gap | Fix → Phase |
|--|--|--|--|--|
| `claude-agent-sdk 0.2.87` | No read-idle/transport timeout; `query()` "continues indefinitely if no ResultMessage"; no PID accessor | bundled-CLI stream, no per-event timeout (`rlm_query.py:699`) | unbounded stall | app-level `wait_for` per event → **Phase 1** |
| `openai` (via rlm) | `OpenAIClient(timeout=)` → `openai.OpenAI(timeout=)`; raises `ReadTimeout`; retries 2× | accelerator route only, default 300s | route nav off bundled CLI + 120s read bound | **Phase 4** |
| `httpx 0.27` | `Timeout(connect,read,write,pool)`; read = idle-between-bytes; `keepalive_expiry` | only RunPod REST, scalar `timeout=60` | granular timeout + bounded retry | **Phase 7e** |
| `fastapi/starlette/uvicorn` | `request.is_disconnected()` + `try/except CancelledError`; `--timeout-graceful-shutdown` | SSE gen has neither (`live_runs.py:696`); no shutdown flag | viewer-disconnect leak; SIGTERM hang | **Phase 7b/7c** |
| `next 16 / react 19` | derive-in-render; `useSyncExternalStore`; `setInterval`+cleanup | `noSignalSecs` computed but never flips `status` (`rlm-header.tsx:83-90`) | stale-"running" (FM-005 UI) | **Phase 7a** |
| `docker 7` | `container.reload(); attrs["State"]["OOMKilled"]`; `remove(force=True)` in finally | cleanup OK; **no OOM read** | OOM indistinguishable from crash | **Phase 7d** |
| `asyncio.subprocess` | cancel ≠ child death; need explicit kill+`wait`; `start_new_session`+`killpg` | pgrep+SIGKILL, no killpg | grandchildren/group | **Phase 1 (hardened kill)** |

### Ranked infra improvements (reliability+observability+latency+cost − risk − ops)

1. **Read-idle timeout + typed sub-RLM failure** (Phase 1) — fixes the wedge; highest reliability/risk ratio. *Reuses* `resilience.failures` (Gap-1), does not rebuild it.
2. **Heartbeat + dangling-subcall detector + `updatedAt` reconciliation** (Phase 2) — wedge visible <2 min; the north-star metric.
3. **Final-report evidence gate** (Phase 3) — kills the most-recurring /runs failure (zero-evidence `partial`).
4. **gpt-5-mini nav route** (Phase 4) — moves the hot path onto a timeout-bearing transport; cost/latency; A/B-gated.
5. **Run-replay/postmortem script + request-id capture** (Phase 7g/7f, Gap-2/3) — cheapest observability; makes the audit repeatable.
6. **Frontend stale-"running" flip + SSE disconnect + uvicorn shutdown** (Phase 7a/b/c) — UI truth + resource hygiene.
7. **Prompt-cache fix** (Phase 6) — real cost leak (`cache_read=0` everywhere), but measure-then-fix and SDK-shape-dependent.
8. **PEEK map / MUSE negative-lessons** (Phase 8/9) — flag-gated prototypes, last.

**Do NOT build (already exist — Section E discovery):** typed-failure taxonomy, provider fallback, circuit breaker, 429 backoff, `RunBudget` cost cap, config-safety validator, SQLite WAL, API-key prompt caching, structured file logging. **Reject:** tenacity (homegrown backoff exists), VCR/respx (can't see the bundled-CLI path). **Defer:** unifying the sub-RLM path onto the unwired `run_agent_with_resilience` engine (Gap-5 — impedance mismatch; do after Phase 1).

---

## Self-review (spec coverage)

- **FM-001..008** → FM-001/002/007 Phase 1; FM-003/005/006 Phase 2 (+7a UI); FM-004 Phase 3; FM-008 Phase 5. ✅ all mapped.
- **Audit next-3-commits + fast-follow** → Phases 1, 2, 3, 4. ✅
- **Four papers** → Section E Table 1 (AutoScientists→1/2, PEEK→8, MUSE→9, BES→reject). ✅
- **Expanded Context7 (10 deps)** → Section E Table 2 + Phases 1/4/6/7. ✅
- **Broad survey gaps** → Gap-1 Phase 1, Gap-2 Phase 7f, Gap-3 Phase 7g, Gap-4 Phase 6, Gap-5 deferred, Gap-6 rejected. ✅
- **Type/flag consistency:** every behavior change is env-flagged; event `sub_rlm_stalled` registered in `rlm-events.ts` (Phase 2); the Phase-1 sink dict is normalized through `build_sub_rlm_stalled_event` before emit (Phase 2 Step 10).
- **Known gap:** Phases 8–9 are intentionally outlined, not line-coded — they REQUIRE a `superpowers:brainstorming` pass first (marked in their headers). All other phases have complete TDD code.

---

## Execution handoff

Land order: **Phase 0 → 1 → 2 → 3** (no new spend, fixes the wedge + zero-evidence reports), then **4** (cost A/B once Phase 2 telemetry exists), then **5, 6, 7a–g** (independent), then brainstorm **8, 9**. Each phase is one reviewable commit; approve chunk-by-chunk.

Per the writing-plans skill, after this plan two execution modes are available — but **you already chose "Plan + start implementing Phase 1,"** so I proceed to implement Phase 1 now (TDD, on `feat/rlm-wedge-hardening`) and **pause at the Phase-1 commit checkpoint** for your review before Phase 2.

