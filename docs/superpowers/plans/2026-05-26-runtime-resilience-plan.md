# PR-μ Runtime Resilience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close four runtime resilience gaps discovered in the 0.305 Adam max-mode post-mortem — SDK aclose race in the implement_baseline path, undersized run_experiment wall-clock, single-mega-iteration shape, and dormant pod_sweeper.

**Architecture:** Four mostly-independent solutions land as ONE commit (per user "infrequent commits" preference). Solutions A/B/D have no file conflicts and execute in parallel; Solution C touches `primitives.py` (same as B) and runs after B. All four follow TDD: test first → run failing → minimal impl → run passing.

**Tech Stack:** Python 3.14 / pytest / FastAPI lifespan / claude-agent-sdk / RunPod REST API / Docker SDK

**Spec:** `docs/superpowers/specs/2026-05-26-runtime-resilience-design.md`

---

## File structure

```
backend/agents/runtime/
    sdk_isolation.py        [NEW, Task A]   — shared thread-isolation primitive
    invoke.py               [modify, Task A] — collect_agent_text uses sdk_isolation
backend/agents/rdr/
    agent.py                [modify, Task A] — refactor existing Workaround B to use shared helper
backend/agents/rlm/
    claude_oauth_client.py  [modify, Task A] — refactor existing Workaround B to use shared helper
    primitives.py           [modify, Task B+C] — EXPERIMENT_TIMEOUT_BY_MODE table + iteration_boundary_recommended emit
    system_prompt.py        [modify, Task C] — iteration discipline paragraph
    forced_iteration.py     [modify, Task C] — two-run_experiment policy extension
backend/services/runtime/
    runpod_backend.py       [modify, Task D] — atexit hook in acquire
    pod_sweep_scheduler.py  [NEW, Task D]    — periodic background sweep
backend/
    app.py                  [modify, Task D] — lifespan wires startup + periodic sweeps
tests/
    test_sdk_isolation.py            [NEW, Task A]
    test_sdk_direct_call_lint.py     [NEW, Task A] — CI guard
    rlm/test_experiment_timeout_resolution.py  [NEW, Task B]
    rlm/test_iteration_boundary_policy.py      [NEW, Task C]
    rlm/test_run_warning_emission.py           [NEW, Task C]
    test_pod_sweeper_atexit.py                 [NEW, Task D]
    test_pod_sweep_scheduler.py                [NEW, Task D]
```

---

## Task A: SDK aclose chokepoint

**Files:**
- Create: `backend/agents/runtime/sdk_isolation.py`
- Modify: `backend/agents/runtime/invoke.py` — wrap `collect_agent_text` SDK call with helper
- Refactor: `backend/agents/rdr/agent.py` (line ~504 — Workaround B uses helper)
- Refactor: `backend/agents/rlm/claude_oauth_client.py` — Workaround B uses helper
- Test: `tests/test_sdk_isolation.py`
- Test: `tests/test_sdk_direct_call_lint.py`

- [ ] **Step A1: Write failing test for sdk_isolation helper**

```python
# tests/test_sdk_isolation.py
import asyncio
import pytest
from backend.agents.runtime.sdk_isolation import run_isolated

@pytest.mark.asyncio
async def test_returns_value_from_isolated_coroutine():
    async def coro():
        return 42
    result = await run_isolated(coro())
    assert result == 42

@pytest.mark.asyncio
async def test_swallows_aclose_race_at_cleanup():
    """The known SDK race fires at generator cleanup; helper must not propagate."""
    async def coro_with_aclose_race():
        try:
            return "result"
        finally:
            raise RuntimeError("aclose(): asynchronous generator is already running")
    result = await run_isolated(coro_with_aclose_race())
    assert result == "result"

@pytest.mark.asyncio
async def test_propagates_genuine_exception_from_coroutine():
    """Real exceptions from the coroutine body itself must still propagate."""
    async def coro_that_raises():
        raise ValueError("real error")
    with pytest.raises(ValueError, match="real error"):
        await run_isolated(coro_that_raises())
```

- [ ] **Step A2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_sdk_isolation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.agents.runtime.sdk_isolation'`

- [ ] **Step A3: Implement sdk_isolation helper**

```python
# backend/agents/runtime/sdk_isolation.py
"""Thread-isolated execution of claude-agent-sdk async calls.

The SDK has a known nested-generator aclose race (see PR-μ runtime resilience
spec, 2026-05-26). When the SDK's internal generator is being closed while
another generator is mid-stream, Python raises RuntimeError("aclose():
asynchronous generator is already running") and the entire pipeline can
unwind catastrophically.

Existing call sites (rdr/agent.py:~504 and rlm/claude_oauth_client.py)
worked around this by running the SDK call in a dedicated thread with its
own event loop, so the race stays trapped in the worker. This module
extracts that pattern so EVERY SDK call inherits the workaround.

The aclose race is swallowed because it ALWAYS fires at cleanup AFTER the
coroutine produced its result — the result is already captured in
`captured_result` before the race fires. Genuine exceptions raised from
inside the coroutine body propagate normally.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any, Coroutine, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_ACLOSE_MARKERS = (
    "aclose(): asynchronous generator is already running",
    "aclose(): synchronous generator already running",
)


def _is_aclose_race(exc: BaseException) -> bool:
    msg = str(exc)
    return any(marker in msg for marker in _ACLOSE_MARKERS)


async def run_isolated(coro: Coroutine[Any, Any, T]) -> T:
    """Run ``coro`` inside a dedicated worker thread with its own event loop.

    The aclose race that fires at generator cleanup is swallowed because it
    cannot affect the value the coroutine already returned. Any other exception
    from the coroutine body propagates to the caller unchanged.

    Set ``REPROLAB_SDK_ISOLATION_DISABLED=true`` to bypass the workaround and
    run the coroutine directly on the calling event loop — useful when
    debugging a suspected isolation-induced issue.
    """
    if os.environ.get("REPROLAB_SDK_ISOLATION_DISABLED", "").lower() in {"true", "1", "yes"}:
        return await coro

    captured_result: list[T] = []
    captured_exception: list[BaseException] = []
    done = threading.Event()

    def worker() -> None:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(coro)
                captured_result.append(result)
            except BaseException as exc:
                if _is_aclose_race(exc) and captured_result:
                    # Race fired AFTER result was returned — safe to swallow.
                    logger.debug("sdk_isolation: aclose race swallowed at cleanup")
                else:
                    captured_exception.append(exc)
            finally:
                try:
                    loop.close()
                except RuntimeError as exc:
                    if not _is_aclose_race(exc):
                        raise
                    # The race can also fire DURING loop.close() — swallow there too.
                    logger.debug("sdk_isolation: aclose race swallowed at loop close")
        finally:
            done.set()

    thread = threading.Thread(target=worker, name="sdk-isolation", daemon=True)
    thread.start()
    # Yield the calling event loop while we wait — non-blocking from the caller's perspective.
    while not done.is_set():
        await asyncio.sleep(0.05)
    thread.join(timeout=1.0)

    if captured_exception:
        raise captured_exception[0]
    if not captured_result:
        raise RuntimeError("sdk_isolation worker completed without result or exception")
    return captured_result[0]


__all__ = ["run_isolated"]
```

- [ ] **Step A4: Verify isolation helper tests pass**

Run: `.venv/bin/python -m pytest tests/test_sdk_isolation.py -v`
Expected: PASS — all 3 tests green.

- [ ] **Step A5: Wire `collect_agent_text` through the helper**

Locate `backend/agents/runtime/invoke.py:18` (the `async def collect_agent_text` definition). Find where it calls `claude_agent_sdk.query()` and awaits the response stream. Wrap that block in `await run_isolated(...)`.

Concrete edit pattern (adapt to the actual function body):

```python
# At top of backend/agents/runtime/invoke.py:
from backend.agents.runtime.sdk_isolation import run_isolated

# Inside collect_agent_text, find the existing pattern like:
#   async def _do_sdk_call():
#       async for message in claude_agent_sdk.query(...):
#           ...
#       return collected_text
#   text = await _do_sdk_call()
# Replace with:
#   text = await run_isolated(_do_sdk_call())
```

If `collect_agent_text` does not currently have an inner `async def _do_sdk_call()`, extract one and pass it through `run_isolated`. Preserve all current behavior — same args, same return type, same exception types except the aclose race.

- [ ] **Step A6: Refactor existing Workaround B sites to use the helper**

`backend/agents/rdr/agent.py` line ~504: replace the inline thread-isolation block with `await run_isolated(_inner())`.
`backend/agents/rlm/claude_oauth_client.py`: same — replace inline isolation with `await run_isolated(...)`.
No behavior change; just deduplication. Preserve all current arguments and return shapes.

- [ ] **Step A7: Write codebase lint test**

```python
# tests/test_sdk_direct_call_lint.py
"""Codebase scanner: claude_agent_sdk.query() / Client() may only be called
from the sdk_isolation module. Any other caller risks the aclose race."""
import ast
from pathlib import Path

ALLOWED_FILES = {
    "backend/agents/runtime/sdk_isolation.py",
    "backend/agents/runtime/invoke.py",  # legacy — must be migrated by Task A5
}

def test_no_direct_claude_agent_sdk_query_calls():
    repo_root = Path(__file__).resolve().parent.parent
    violations: list[str] = []
    for py_file in (repo_root / "backend").rglob("*.py"):
        rel = str(py_file.relative_to(repo_root))
        if rel in ALLOWED_FILES:
            continue
        if "claude_agent_sdk" not in py_file.read_text():
            continue
        tree = ast.parse(py_file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # Match `claude_agent_sdk.query(...)` and `claude_agent_sdk.Client(...)`
                if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                    if func.value.id == "claude_agent_sdk" and func.attr in ("query", "Client"):
                        violations.append(f"{rel}:{node.lineno}: direct claude_agent_sdk.{func.attr}() call")
    assert not violations, (
        "Direct claude_agent_sdk calls must go through sdk_isolation.run_isolated:\n"
        + "\n".join(violations)
    )
```

- [ ] **Step A8: Run lint test + full Task-A test suite**

Run: `.venv/bin/python -m pytest tests/test_sdk_isolation.py tests/test_sdk_direct_call_lint.py -v`
Expected: PASS — all tests green. If lint test fails, the named files have direct SDK calls and need to be migrated (this is the gate that prevents regression).

---

## Task B: Mode-scaled `run_experiment` wall-clock cap

**Files:**
- Modify: `backend/agents/rlm/primitives.py` (around line 2168-2255 — timeout resolver)
- Test: `tests/rlm/test_experiment_timeout_resolution.py`

- [ ] **Step B1: Write failing test for the resolver**

```python
# tests/rlm/test_experiment_timeout_resolution.py
"""Resolution order: env override > mode default > catch-all default, clamped
by ctx.remaining_s() only when finite."""
import math
import os
import pytest
from unittest.mock import MagicMock
from backend.agents.rlm.primitives import resolve_experiment_timeout_s, EXPERIMENT_TIMEOUT_BY_MODE

@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("REPROLAB_RUN_EXPERIMENT_TIMEOUT_S", raising=False)

def _ctx(*, remaining_s=math.inf, execution_mode="efficient"):
    ctx = MagicMock()
    ctx.remaining_s.return_value = remaining_s
    ctx.execution_mode = execution_mode
    return ctx

def test_env_override_wins():
    os.environ["REPROLAB_RUN_EXPERIMENT_TIMEOUT_S"] = "1234"
    try:
        assert resolve_experiment_timeout_s(_ctx(execution_mode="efficient")) == 1234
    finally:
        del os.environ["REPROLAB_RUN_EXPERIMENT_TIMEOUT_S"]

def test_efficient_mode_default():
    assert resolve_experiment_timeout_s(_ctx(execution_mode="efficient")) == 7200

def test_max_mode_default():
    assert resolve_experiment_timeout_s(_ctx(execution_mode="max")) == 21600

def test_unknown_mode_falls_back():
    assert resolve_experiment_timeout_s(_ctx(execution_mode=None)) == 7200

def test_clamps_to_finite_remaining():
    """remaining_s=3600 with max mode (21600 default) should clamp to 3600."""
    result = resolve_experiment_timeout_s(_ctx(remaining_s=3600.0, execution_mode="max"))
    assert result == 3600

def test_does_not_clamp_to_infinite_remaining():
    """Infinite remaining_s means no --max-wall-clock; clamp must be a no-op."""
    assert resolve_experiment_timeout_s(_ctx(remaining_s=math.inf, execution_mode="max")) == 21600

def test_env_override_also_clamped_to_finite_remaining():
    os.environ["REPROLAB_RUN_EXPERIMENT_TIMEOUT_S"] = "100000"
    try:
        result = resolve_experiment_timeout_s(_ctx(remaining_s=3600.0, execution_mode="efficient"))
        assert result == 3600
    finally:
        del os.environ["REPROLAB_RUN_EXPERIMENT_TIMEOUT_S"]
```

- [ ] **Step B2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_experiment_timeout_resolution.py -v`
Expected: FAIL with `ImportError: cannot import name 'resolve_experiment_timeout_s'`.

- [ ] **Step B3: Implement resolver in primitives.py**

In `backend/agents/rlm/primitives.py`, near the existing `run_experiment` timeout block (around line 2168), add:

```python
# Near the top of the run_experiment-related primitive block, add module-level constants:
EXPERIMENT_TIMEOUT_BY_MODE: dict[str, int] = {
    "efficient": 7200,    # 2h per call
    "max":      21600,    # 6h per call
}
_DEFAULT_EXPERIMENT_TIMEOUT_S: int = 7200  # fallback when execution_mode is None/unknown


def resolve_experiment_timeout_s(ctx) -> int:
    """Resolve the wall-clock cap for a single run_experiment call.

    Order:
      1. REPROLAB_RUN_EXPERIMENT_TIMEOUT_S env override (if set and > 0)
      2. EXPERIMENT_TIMEOUT_BY_MODE[ctx.execution_mode]
      3. _DEFAULT_EXPERIMENT_TIMEOUT_S

    Then clamp to ctx.remaining_s() only when finite — an infinite remaining
    means no --max-wall-clock was set and we should honor the mode default.
    """
    import math as _math
    import os as _os

    _env = _os.environ.get("REPROLAB_RUN_EXPERIMENT_TIMEOUT_S", "").strip()
    if _env:
        try:
            override = int(_env)
            if override > 0:
                resolved = override
            else:
                resolved = EXPERIMENT_TIMEOUT_BY_MODE.get(ctx.execution_mode, _DEFAULT_EXPERIMENT_TIMEOUT_S)
        except ValueError:
            resolved = EXPERIMENT_TIMEOUT_BY_MODE.get(ctx.execution_mode, _DEFAULT_EXPERIMENT_TIMEOUT_S)
    else:
        resolved = EXPERIMENT_TIMEOUT_BY_MODE.get(ctx.execution_mode, _DEFAULT_EXPERIMENT_TIMEOUT_S)

    try:
        remaining = ctx.remaining_s()
    except Exception:
        remaining = _math.inf
    if _math.isfinite(remaining) and remaining > 0:
        resolved = min(resolved, int(remaining))
    return resolved
```

Then locate the existing inline timeout calculation in `run_experiment` (around line 2168) and replace it with a single call to `resolve_experiment_timeout_s(ctx)`. Preserve the surrounding error-handling and `{error, outcome, primitive, wall_clock_s}` failure shape.

- [ ] **Step B4: Run Task B tests**

Run: `.venv/bin/python -m pytest tests/rlm/test_experiment_timeout_resolution.py -v`
Expected: PASS — all 7 tests green.

- [ ] **Step B5: Verify no regression in adjacent timeout tests**

Run: `.venv/bin/python -m pytest tests/rlm/ -v -k "timeout or experiment" --tb=short`
Expected: PASS — all pre-existing timeout/experiment tests still pass.

---

## Task C: Hybrid iteration discipline

**Files:**
- Modify: `backend/agents/rlm/system_prompt.py` (insert iteration discipline paragraph)
- Modify: `backend/agents/rlm/primitives.py` (emit run_warning on repairable/partial from run_experiment)
- Modify: `backend/agents/rlm/forced_iteration.py` (refuse FINAL_VAR on two-run_experiment-with-failure pattern)
- Test: `tests/rlm/test_iteration_boundary_policy.py`
- Test: `tests/rlm/test_run_warning_emission.py`

**Sequencing: This task touches `primitives.py` like Task B does. Run Task C AFTER Task B has committed locally (no actual conflict on the same lines, but safer to serialize the file edits.)**

- [ ] **Step C1: Write failing test for forced-iteration extension**

```python
# tests/rlm/test_iteration_boundary_policy.py
"""ForcedIterationPolicy must refuse FINAL_VAR when the same iteration
contains TWO run_experiment calls with the second returning repairable/
partial_evidence/fatal — the 0.305 Adam anti-pattern."""
import pytest
from unittest.mock import MagicMock
from backend.agents.rlm.forced_iteration import ForcedIterationPolicy

def _policy(**overrides):
    defaults = dict(
        target_score=0.6,
        min_iterations=2,
        max_rlm_iterations=10,
        run_id="test-run",
        ctx=MagicMock(remaining_s=MagicMock(return_value=99999)),
    )
    defaults.update(overrides)
    return ForcedIterationPolicy(**defaults)

def test_refuses_final_var_when_two_run_experiments_with_repairable_latter():
    p = _policy()
    # Simulate: first run_experiment ok, second returned repairable
    p.record_run_experiment(outcome="ok")
    p.record_run_experiment(outcome="repairable")
    decision = p.should_refuse_final_var(current_score=0.8, iteration_count=1)
    assert decision.refuse is True
    assert "two run_experiment" in decision.reason.lower()

@pytest.mark.parametrize("second_outcome", ["repairable", "partial_evidence", "fatal"])
def test_refuses_on_any_failure_outcome_of_latter_experiment(second_outcome):
    p = _policy()
    p.record_run_experiment(outcome="ok")
    p.record_run_experiment(outcome=second_outcome)
    decision = p.should_refuse_final_var(current_score=0.9, iteration_count=1)
    assert decision.refuse is True

def test_allows_final_var_when_only_one_run_experiment_in_iteration():
    p = _policy()
    p.record_run_experiment(outcome="repairable")
    decision = p.should_refuse_final_var(current_score=0.8, iteration_count=2)
    assert decision.refuse is False

def test_allows_final_var_when_both_run_experiments_ok():
    p = _policy()
    p.record_run_experiment(outcome="ok")
    p.record_run_experiment(outcome="ok")
    decision = p.should_refuse_final_var(current_score=0.9, iteration_count=2)
    assert decision.refuse is False

def test_iteration_boundary_history_resets_on_iteration_advance():
    p = _policy()
    p.record_run_experiment(outcome="ok")
    p.record_run_experiment(outcome="repairable")
    p.on_iteration_advance()  # turn boundary
    # In a fresh iteration, history clean
    decision = p.should_refuse_final_var(current_score=0.9, iteration_count=2)
    assert decision.refuse is False
```

- [ ] **Step C2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_iteration_boundary_policy.py -v`
Expected: FAIL — `record_run_experiment` does not exist on `ForcedIterationPolicy`.

- [ ] **Step C3: Extend `ForcedIterationPolicy`**

In `backend/agents/rlm/forced_iteration.py`, add to the existing class:

```python
# New instance state in __init__:
#     self._experiments_in_iteration: list[str] = []

def record_run_experiment(self, outcome: str) -> None:
    """Track the per-iteration sequence of run_experiment outcomes.
    Called from the run_experiment primitive after computing its outcome."""
    self._experiments_in_iteration.append(outcome)

def on_iteration_advance(self) -> None:
    """Reset per-iteration trackers when a new REPL turn starts."""
    self._experiments_in_iteration = []
    # preserve existing iteration-advance behavior (repair counts etc.)
```

Then in `should_refuse_final_var(self, current_score, iteration_count)` (or whichever existing method is the FINAL_VAR gate), add this check at the top of the existing refusal logic:

```python
# PR-μ Solution C extension: refuse FINAL_VAR when the same iteration
# contained two run_experiment calls with the latter returning a
# failure outcome (repairable / partial_evidence / fatal).
_FAILURE_OUTCOMES = {"repairable", "partial_evidence", "fatal"}
if (
    len(self._experiments_in_iteration) >= 2
    and self._experiments_in_iteration[-1] in _FAILURE_OUTCOMES
):
    return PolicyDecision(
        refuse=True,
        reason=(
            f"two run_experiment calls in this iteration; the latter returned "
            f"'{self._experiments_in_iteration[-1]}'. End this iteration so the "
            f"failure surfaces as fresh next-turn context."
        ),
    )
# ... existing refusal logic continues ...
```

If `ForcedIterationPolicy` does not yet have a `PolicyDecision`-style return type, return `(True, reason_str)` and the caller will adapt. Inspect the file for the current signature before editing.

- [ ] **Step C4: Run Task C policy test**

Run: `.venv/bin/python -m pytest tests/rlm/test_iteration_boundary_policy.py -v`
Expected: PASS — all 5 tests.

- [ ] **Step C5: Write failing test for run_warning emission**

```python
# tests/rlm/test_run_warning_emission.py
"""run_experiment must emit a dashboard_event with code='iteration_boundary_recommended'
whenever its outcome is repairable or partial_evidence."""
import json
import pytest
from pathlib import Path
from unittest.mock import patch
from backend.agents.rlm.primitives import _emit_iteration_boundary_warning

def test_emits_warning_event_for_repairable(tmp_path: Path):
    events_file = tmp_path / "dashboard_events.jsonl"
    _emit_iteration_boundary_warning(
        run_dir=tmp_path,
        outcome="repairable",
        brief="preflight blocked: load_dataset('imdb')",
    )
    assert events_file.exists()
    events = [json.loads(l) for l in events_file.read_text().splitlines()]
    assert len(events) == 1
    assert events[0]["event"] == "run_warning"
    assert events[0]["code"] == "iteration_boundary_recommended"
    assert "preflight blocked" in events[0]["message"]

def test_emits_warning_event_for_partial_evidence(tmp_path: Path):
    _emit_iteration_boundary_warning(
        run_dir=tmp_path,
        outcome="partial_evidence",
        brief="wall_clock_s=21600",
    )
    events = [json.loads(l) for l in (tmp_path / "dashboard_events.jsonl").read_text().splitlines()]
    assert events[0]["code"] == "iteration_boundary_recommended"

def test_does_not_emit_for_ok_outcome(tmp_path: Path):
    _emit_iteration_boundary_warning(
        run_dir=tmp_path,
        outcome="ok",
        brief="success",
    )
    assert not (tmp_path / "dashboard_events.jsonl").exists()
```

- [ ] **Step C6: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/rlm/test_run_warning_emission.py -v`
Expected: FAIL — `_emit_iteration_boundary_warning` does not exist.

- [ ] **Step C7: Implement emit helper + wire into `run_experiment`**

In `backend/agents/rlm/primitives.py`, add near other primitive helpers:

```python
def _emit_iteration_boundary_warning(run_dir, outcome: str, brief: str) -> None:
    """Append an iteration_boundary_recommended run_warning to dashboard_events.jsonl.
    Pure file I/O — no LLM call. Fail-soft on write errors."""
    if outcome not in {"repairable", "partial_evidence"}:
        return
    try:
        from pathlib import Path as _Path
        import json as _json
        from datetime import datetime as _dt, timezone as _tz
        events_path = _Path(run_dir) / "dashboard_events.jsonl"
        event = {
            "event": "run_warning",
            "timestamp": _dt.now(_tz.utc).isoformat(),
            "code": "iteration_boundary_recommended",
            "message": (
                f"run_experiment returned {outcome}; end this iteration so the "
                f"failure surfaces as fresh next-turn context. ({brief})"
            ),
        }
        with open(events_path, "a") as f:
            f.write(_json.dumps(event) + "\n")
    except Exception:
        pass  # observability is best-effort
```

Then in `run_experiment`, after the outcome is classified and BEFORE the result is returned:

```python
# Existing outcome classification produces e.g. `result_dict["outcome"]`
_emit_iteration_boundary_warning(
    run_dir=ctx.run_dir,
    outcome=result_dict.get("outcome", "ok"),
    brief=str(result_dict.get("error") or result_dict.get("failure_class") or "")[:120],
)
# Also feed the policy:
if hasattr(ctx, "forced_iteration_policy") and ctx.forced_iteration_policy is not None:
    ctx.forced_iteration_policy.record_run_experiment(result_dict.get("outcome", "ok"))
```

Locate the actual variable names in the existing `run_experiment` body before editing; the names above are illustrative of the wiring required.

Also add a print-banner that the REPL captures and shows the root model:

```python
if result_dict.get("outcome") in {"repairable", "partial_evidence"}:
    print(
        "╔═ ITERATION BOUNDARY RECOMMENDED ═╗\n"
        f"║ run_experiment returned {result_dict['outcome']}; end this iteration\n"
        f"║ so the failure surfaces as fresh next-turn context.\n"
        "╚══════════════════════════════════╝",
        flush=True,
    )
```

- [ ] **Step C8: Run Task C emit test + policy test together**

Run: `.venv/bin/python -m pytest tests/rlm/test_run_warning_emission.py tests/rlm/test_iteration_boundary_policy.py -v`
Expected: PASS — both files green.

- [ ] **Step C9: Add system prompt paragraph**

In `backend/agents/rlm/system_prompt.py`, after the FORCED-ITERATION POLICY section (around line 200-227), add:

```python
_ITERATION_DISCIPLINE = """\
ITERATION DISCIPLINE — one run_experiment per iteration:
  After every `run_experiment` call, *return from the current iteration*.
  Do not write a follow-up propose_improvements → implement_baseline →
  run_experiment → verify_against_rubric chain in the same REPL turn — let
  the experiment result land as next-iteration context.

  This is MANDATORY when run_experiment returned `outcome="repairable"`
  or `outcome="partial_evidence"`. You will see a banner:

    ╔═ ITERATION BOUNDARY RECOMMENDED ═╗
    ║ run_experiment returned <outcome>; end this iteration ...
    ╚══════════════════════════════════╝

  Returning from the iteration immediately after this banner is the only
  way the forced-iteration policy can correctly bound the retry loop and
  cleanly surface the failure to the next root-turn's context window. The
  policy will REFUSE FINAL_VAR if you call run_experiment twice in one
  iteration with the latter failing — pack one experiment per iteration.
"""
```

Then add `_ITERATION_DISCIPLINE` to the prompt assembly in the same module (wherever the existing sections are concatenated into the final prompt string).

- [ ] **Step C10: Smoke-test that system prompt loads correctly**

Run: `.venv/bin/python -c "from backend.agents.rlm.system_prompt import SYSTEM_PROMPT; assert 'ITERATION DISCIPLINE' in SYSTEM_PROMPT; print('ok')"`
Expected: `ok`.

(If the prompt is assembled via a function rather than a constant, adapt the import to invoke that function.)

---

## Task D: Layered pod cleanup (atexit + startup + periodic)

**Files:**
- Modify: `backend/services/runtime/runpod_backend.py` — `atexit.register(self._cleanup_atexit)` in `acquire`
- Create: `backend/services/runtime/pod_sweep_scheduler.py` — periodic background sweep
- Modify: `backend/app.py` (`create_app` lifespan) — startup sweep + scheduler start/stop
- Test: `tests/test_pod_sweeper_atexit.py`
- Test: `tests/test_pod_sweep_scheduler.py`

- [ ] **Step D1: Write failing test for atexit cleanup**

```python
# tests/test_pod_sweeper_atexit.py
"""RunpodBackend.acquire registers an atexit cleanup; destroy() is
idempotent (safe to call from both atexit and the normal lifecycle's finally)."""
import atexit
import pytest
from unittest.mock import MagicMock, patch

@patch("backend.services.runtime.runpod_backend._create_pod_via_api")
def test_acquire_registers_atexit_cleanup(mock_create):
    mock_create.return_value = {"pod_id": "test-pod-id", "ssh_host": "h", "ssh_port": 22}
    from backend.services.runtime.runpod_backend import RunpodBackend
    backend = RunpodBackend(api_key="dummy", ssh_key_path="/dev/null")
    with patch("atexit.register") as mock_register:
        try:
            backend.acquire()
        except Exception:
            pass  # SSH may not actually connect in test — that's ok
        mock_register.assert_called_once()
        args, _ = mock_register.call_args
        # The atexit handler must reference the backend's destroy path
        assert callable(args[0])

def test_destroy_is_idempotent():
    """Calling destroy twice must not raise; second call is a no-op."""
    from backend.services.runtime.runpod_backend import RunpodBackend
    backend = RunpodBackend(api_key="dummy", ssh_key_path="/dev/null")
    backend._pod_id = "test-pod"
    with patch("backend.services.runtime.runpod_backend._delete_pod_via_api") as mock_delete:
        backend.destroy()
        backend.destroy()
        assert mock_delete.call_count == 1, "destroy() not idempotent"
```

- [ ] **Step D2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pod_sweeper_atexit.py -v`
Expected: FAIL — either `atexit.register` is not called, or `destroy()` is not idempotent.

- [ ] **Step D3: Wire atexit into RunpodBackend**

In `backend/services/runtime/runpod_backend.py`, inside `RunpodBackend.acquire()`, after the pod is successfully created and `self._pod_id` is set, add:

```python
import atexit  # at top of file

# Inside acquire(), AFTER self._pod_id assignment:
atexit.register(self._cleanup_atexit)
```

Then add an idempotent cleanup method:

```python
def _cleanup_atexit(self) -> None:
    """atexit handler: terminate the pod if still alive. Idempotent."""
    if self._pod_id is None:
        return
    try:
        self.destroy()
    except Exception as exc:
        logger.warning("atexit cleanup of pod %s failed: %s", self._pod_id, exc)
```

And make `destroy` idempotent — clear `self._pod_id` after successful termination so a second call is a no-op:

```python
def destroy(self) -> None:
    if self._pod_id is None:
        return  # already destroyed
    try:
        _delete_pod_via_api(self._api_key, self._pod_id)
    finally:
        self._pod_id = None
```

- [ ] **Step D4: Run atexit tests**

Run: `.venv/bin/python -m pytest tests/test_pod_sweeper_atexit.py -v`
Expected: PASS — both tests green.

- [ ] **Step D5: Write failing test for the periodic scheduler**

```python
# tests/test_pod_sweep_scheduler.py
"""Periodic scheduler runs sweep_stale_pods at the configured interval.
Fail-soft on exceptions; disabled when REPROLAB_POD_SWEEP_ENABLED=false."""
import asyncio
import pytest
from unittest.mock import MagicMock, patch

@pytest.mark.asyncio
async def test_scheduler_calls_sweep_at_interval(monkeypatch):
    monkeypatch.setenv("REPROLAB_RUNPOD_API_KEY", "dummy")
    monkeypatch.setenv("REPROLAB_POD_SWEEP_INTERVAL_S", "0.05")  # 50ms for fast test
    sweep_calls = []
    fake_sweep = MagicMock(side_effect=lambda **kw: sweep_calls.append(kw) or {"reaped": 0})
    with patch("backend.services.runtime.pod_sweep_scheduler.sweep_stale_pods", fake_sweep):
        from backend.services.runtime.pod_sweep_scheduler import PodSweepScheduler
        sched = PodSweepScheduler()
        await sched.start()
        await asyncio.sleep(0.2)
        await sched.stop()
    assert len(sweep_calls) >= 2, f"expected ≥2 sweeps, got {len(sweep_calls)}"

@pytest.mark.asyncio
async def test_scheduler_disabled_via_env(monkeypatch):
    monkeypatch.setenv("REPROLAB_POD_SWEEP_ENABLED", "false")
    monkeypatch.setenv("REPROLAB_RUNPOD_API_KEY", "dummy")
    fake_sweep = MagicMock(return_value={"reaped": 0})
    with patch("backend.services.runtime.pod_sweep_scheduler.sweep_stale_pods", fake_sweep):
        from backend.services.runtime.pod_sweep_scheduler import PodSweepScheduler
        sched = PodSweepScheduler()
        await sched.start()
        await asyncio.sleep(0.1)
        await sched.stop()
    fake_sweep.assert_not_called()

@pytest.mark.asyncio
async def test_scheduler_fail_soft_on_sweep_exception(monkeypatch):
    monkeypatch.setenv("REPROLAB_RUNPOD_API_KEY", "dummy")
    monkeypatch.setenv("REPROLAB_POD_SWEEP_INTERVAL_S", "0.05")
    calls = []
    def _flaky_sweep(**kw):
        calls.append(kw)
        if len(calls) == 1:
            raise RuntimeError("network blip")
        return {"reaped": 0}
    with patch("backend.services.runtime.pod_sweep_scheduler.sweep_stale_pods", _flaky_sweep):
        from backend.services.runtime.pod_sweep_scheduler import PodSweepScheduler
        sched = PodSweepScheduler()
        await sched.start()
        await asyncio.sleep(0.2)
        await sched.stop()
    assert len(calls) >= 2, "scheduler stopped after exception (should have continued)"
```

- [ ] **Step D6: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pod_sweep_scheduler.py -v`
Expected: FAIL — `ModuleNotFoundError: backend.services.runtime.pod_sweep_scheduler`.

- [ ] **Step D7: Implement scheduler**

```python
# backend/services/runtime/pod_sweep_scheduler.py
"""Periodic background sweep of stale RunPod pods.

Wired into the FastAPI app lifespan (see backend/app.py). Calls
backend.services.runtime.pod_sweeper.sweep_stale_pods on a configurable
interval. Fail-soft: any exception is logged but the scheduler keeps
running.

Disabled when:
  - REPROLAB_RUNPOD_API_KEY is unset (no RunPod usage)
  - REPROLAB_POD_SWEEP_ENABLED=false
"""
from __future__ import annotations

import asyncio
import logging
import os

from backend.services.runtime.pod_sweeper import sweep_stale_pods

logger = logging.getLogger(__name__)


class PodSweepScheduler:
    """Background asyncio task that runs sweep_stale_pods periodically."""

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._stop_event: asyncio.Event = asyncio.Event()

    def _enabled(self) -> bool:
        if not os.environ.get("REPROLAB_RUNPOD_API_KEY"):
            return False
        if os.environ.get("REPROLAB_POD_SWEEP_ENABLED", "true").lower() in {"false", "0", "no", "off"}:
            return False
        return True

    def _interval_s(self) -> float:
        try:
            return float(os.environ.get("REPROLAB_POD_SWEEP_INTERVAL_S", "1800"))
        except ValueError:
            return 1800.0

    def _max_age_s(self) -> int:
        try:
            return int(os.environ.get("REPROLAB_POD_SWEEP_MAX_AGE_S", "7200"))
        except ValueError:
            return 7200

    async def start(self) -> None:
        if not self._enabled():
            logger.info("pod_sweep_scheduler: disabled (no RUNPOD_API_KEY or REPROLAB_POD_SWEEP_ENABLED=false)")
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        interval = self._interval_s()
        max_age = self._max_age_s()
        logger.info("pod_sweep_scheduler: starting (interval=%.1fs, max_age=%ds)", interval, max_age)
        while not self._stop_event.is_set():
            try:
                summary = sweep_stale_pods(max_age_seconds=max_age, dry_run=False)
                logger.info("pod_sweep_scheduler: sweep complete: %s", summary)
            except Exception as exc:
                logger.warning("pod_sweep_scheduler: sweep failed: %s", exc)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass  # interval elapsed; loop continues

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                self._task.cancel()
        self._task = None


__all__ = ["PodSweepScheduler"]
```

- [ ] **Step D8: Run scheduler tests**

Run: `.venv/bin/python -m pytest tests/test_pod_sweep_scheduler.py -v`
Expected: PASS — all 3 tests green.

- [ ] **Step D9: Wire into FastAPI lifespan**

In `backend/app.py`, locate the `create_app` function and its lifespan handler. Add:

```python
# Near the lifespan handler (FastAPI's @asynccontextmanager pattern):
from backend.services.runtime.pod_sweep_scheduler import PodSweepScheduler
from backend.services.runtime.pod_sweeper import sweep_stale_pods

_pod_sweep_scheduler = PodSweepScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        if os.environ.get("REPROLAB_RUNPOD_API_KEY") and os.environ.get("REPROLAB_POD_SWEEP_ENABLED", "true").lower() not in {"false", "0", "no", "off"}:
            try:
                summary = sweep_stale_pods(max_age_seconds=7200, dry_run=False)
                logger.info("startup pod sweep: %s", summary)
            except Exception as exc:
                logger.warning("startup pod sweep failed: %s", exc)
        await _pod_sweep_scheduler.start()
    except Exception as exc:
        logger.warning("pod sweep init failed: %s", exc)
    yield
    # Shutdown
    try:
        await _pod_sweep_scheduler.stop()
    except Exception:
        pass
```

If `create_app` already has a lifespan, extend it; do not replace existing startup/shutdown logic.

- [ ] **Step D10: Smoke-test app boot loads scheduler module**

Run: `.venv/bin/python -c "from backend.app import create_app; app = create_app(); print('ok')"`
Expected: `ok` printed (any pod-sweep log lines are also fine).

---

## Cross-task: full test run before commit

- [ ] **Step F1: Run the full test suite to confirm no regressions**

Run: `.venv/bin/python -m pytest tests/ -n auto --tb=short 2>&1 | tail -40`
Expected: All tests pass. Baseline before this PR is 2701 tests passing; expect ~2701 + 22 new tests (~2723).

- [ ] **Step F2: Single commit of all four solutions**

```bash
git add docs/superpowers/specs/2026-05-26-runtime-resilience-design.md \
        docs/superpowers/plans/2026-05-26-runtime-resilience-plan.md \
        backend/agents/runtime/sdk_isolation.py \
        backend/agents/runtime/invoke.py \
        backend/agents/rdr/agent.py \
        backend/agents/rlm/claude_oauth_client.py \
        backend/agents/rlm/primitives.py \
        backend/agents/rlm/system_prompt.py \
        backend/agents/rlm/forced_iteration.py \
        backend/services/runtime/runpod_backend.py \
        backend/services/runtime/pod_sweep_scheduler.py \
        backend/app.py \
        tests/test_sdk_isolation.py \
        tests/test_sdk_direct_call_lint.py \
        tests/rlm/test_experiment_timeout_resolution.py \
        tests/rlm/test_iteration_boundary_policy.py \
        tests/rlm/test_run_warning_emission.py \
        tests/test_pod_sweeper_atexit.py \
        tests/test_pod_sweep_scheduler.py

git commit -m "Runtime resilience hardening: SDK isolation chokepoint, mode-scaled experiment cap, hybrid iteration discipline, layered pod cleanup (PR-μ)"
```

- [ ] **Step F3: Push to origin**

```bash
git push origin main
```

---

## Self-review (completed)

1. **Spec coverage:** All 4 spec solutions have tasks. Cross-cutting failure-class table is reference docs (not implementable). Out-of-scope items (resume-from-checkpoint, mid-training OOM, RDR refactor) explicitly excluded.
2. **Placeholder scan:** No TBD/TODO/fill-in-later. Each step has concrete code blocks where code is changed.
3. **Type consistency:** `resolve_experiment_timeout_s` is the same name across Task B; `_emit_iteration_boundary_warning` and `record_run_experiment` are the same names in Tasks C and tests.
4. **One-commit convention:** Per user memory "few substantial commits at major milestones" — single F2 commit, not per-task commits. Spec + plan included in the same commit as the implementation.
