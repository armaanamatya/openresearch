# Derived run-state contract — workflow-polish liveness design

> Locked 2026-05-27. Brainstormed in `/brainstorming` → "End-to-end workflow polish" → "Trustworthy liveness" → "Derived run-state contract, backend-computed, emitted as new SSE event."

## 1. Problem

The 2026-05-26 three-day audit found **4 of 12 runs in the audit window are zombie-status** (`status="running"`, dead PID, no `final_report.json`). The known-issues runbook (`docs/runbooks/known-issues-and-monitoring.md` §3.6) documents that the UI's "no signal Xs" chip false-alarms because a single heartbeat-age threshold cannot distinguish "Sonnet is thinking" from "actually stuck."

A researcher reproducing one paper cannot currently tell, from any single UI surface, whether their run is **working / waiting-on-LLM / stuck / dead**. The status pill, the "no signal" chip, the iteration counter, the primitive-call counter, and `dashboard_events.jsonl` each carry one slice of the truth. Cross-checking them requires reading a 470-line runbook.

PR-π Module B (`backend/services/events/run_liveness.py`) added an orphan-run sweeper that marks zombies as `status="interrupted"` 120s after PID death — but the sweeper **skips runs where `pid is None`** (line 268-272), which is exactly the failure mode for 2 of the 4 audit zombies (`prj_09047`, `prj_8819`). The sweeper as-deployed cannot close that loop.

## 2. Goal

One authoritative computed `run_state` field, written by the live process on every transition and by the sweeper on post-hoc detection, consumed identically by the lab UI, the CLI tail, and the leaderboard. Replaces the pill+chip+counter ambiguity with one signal the researcher can read.

## 3. Non-goals

- Sub-RLM lifecycle visualisation — covered by existing `sub_rlm_spawned` / `sub_rlm_complete`.
- Cost transparency for OAuth — separate spec.
- Multi-attempt archival UX — already landed via `attempt_isolation.py`; just consumed.
- Resume-button UI plumbing — surfaces from the new `interrupted` state but the action wire-up is a follow-up.

## 4. Architecture

```
primitive_call(start|ok|error)  ─┐
iteration_heartbeat              ─┼──▶  RunStateComputer.tick()  ─▶  SSE event `run_state`
code/ mtime (poller)             ─┤                                 │
PID liveness (live + sweeper)    ─┤                                 ▶  demo_status.json::run_state
crash try/finally                ─┘
```

**One new module:** `backend/agents/rlm/run_state.py` owns `RunStateKind`, `RunStateSubstate`, and `RunStateComputer`. It does not touch the corpus and emits via the existing thread-safe `emit` closure from `sse_bridge.make_emit`.

**Three writers, one field:**
1. The live process (`RunStateComputer.tick()`) emits on every transition.
2. The orphan sweeper writes `run_state={kind:"interrupted", ...}` into `demo_status.json` (no live SSE consumer).
3. A new outer `try/finally` in `run_pipeline_rlm` writes `run_state={kind:"failed", reason:...}` for any uncaught exception.

**Three consumers, one field:**
1. The lab UI reducer (`frontend/src/hooks/use-rlm-run.ts`) tracks the latest `run_state` and renders via a new `<RunStatePill>` component.
2. The leaderboard (`backend/routes/leaderboard.py`) surfaces `run_state.kind` for the status column when present, falling back to existing `status` for legacy rows.
3. CLI / screenshot tail (`scripts/lab_screenshot_tail.mjs`) reads the same field from `demo_status.json`.

## 5. The state machine

```python
class RunStateKind(StrEnum):
    INITIALIZING = "initializing"
    WORKING      = "working"
    IDLE         = "idle"
    STUCK        = "stuck"
    INTERRUPTED  = "interrupted"
    COMPLETED    = "completed"
    FAILED       = "failed"
```

| From | To | Trigger |
|------|----|---------|
| `initializing` | `working` | first `primitive_call(start)` |
| `working` | `idle` | mtime older than `RUN_STATE_IDLE_S=60` |
| `idle` | `stuck` | mtime older than `RUN_STATE_STUCK_S=240` AND no heartbeat in 60s |
| `working`/`idle`/`stuck` | `working` | new file mtime in `code/` OR new `primitive_call(start)` |
| `working`/`idle`/`stuck` | `idle` | `primitive_call(ok|error)` then no new mtime for 60s |
| any non-terminal | `interrupted` | sweeper write OR PID-dead observed by tick |
| any non-terminal | `completed` | `run_complete` with `status="completed"` |
| any non-terminal | `failed` | `run_complete` with `status in {failed,partial-failed}` OR outer try/finally caught exception |

Terminal states (`completed`, `failed`, `interrupted`) are absorbing — the computer rejects any further transition once entered.

The `RUN_STATE_STUCK_S` default matches the existing `_PRE_EMIT_STALL_S=240` in `primitives.py` so the UI's "stuck" badge fires at the same moment the backend's pre-emit watchdog escalates to repairable. Same env-var override path (`REPROLAB_PRE_EMIT_STALL_S`).

## 6. Substate schema

```python
@dataclass(frozen=True)
class RunStateSubstate:
    primitive: str | None             # last started primitive name
    seconds_active: int               # since current primitive start (0 if none)
    seconds_since_event: int          # since any dashboard event
    last_file_touched: str | None     # basename in code/; sentinel-redacted
    iteration: int                    # last completed root iteration
    pre_emit_stalled: bool            # mirrors primitives.py's stall flag
    reason: str | None                # only on interrupted/failed
```

Corpus invariant: `last_file_touched` is a basename only, run through `redact_corpus(sentinels)` defensively. The substate dict goes through `sse_bridge` so any future field reuses the same egress chokepoint.

## 7. SSE event shape

```python
def build_run_state_event(*, kind, substate, run_id) -> dict:
    return {
        "event": "run_state",
        "timestamp": _now_iso(),
        "run_id": run_id,
        "kind": kind,             # one of RunStateKind values
        "substate": {
            "primitive": ...,
            "seconds_active": ...,
            "seconds_since_event": ...,
            "last_file_touched": ...,
            "iteration": ...,
            "pre_emit_stalled": ...,
            "reason": ...,
        },
    }
```

Emitted on every transition. The computer suppresses emission when neither `kind` nor any substate field changed since the prior emit (de-dupe). A heartbeat tick that produces no change is silent.

## 8. PID gap fix (load-bearing)

`backend/agents/rlm/run.py:1024` calls `_write_demo_status("running")` at run start but does not stamp `os.getpid()`. The parent spawner in `live_runs.py:839` stamps it — but only for runs spawned through that path; CLI runs (`python -m backend.cli reproduce`) and script-spawned runs miss the stamp until the parent gets around to it.

**Fix:** `_write_demo_status` gains an optional `pid` kwarg defaulting to `os.getpid()`. On `status="running"` it stamps every call. This is a defensive double-write — the parent's stamp still wins on timing — and it closes the `pid is None` orphan gap in `run_liveness.py:266-272`.

## 9. Crash try/finally (Pattern P3 from 2026-05-26 audit)

Current `run_pipeline_rlm` wraps `rlm.completion` in try/except/finally but **does not wrap the whole function**. An exception in setup (e.g. paper-text precondition failure, `_build_llm_client`, `_resolve_agent_runtime`) bypasses the terminal `_write_demo_status` call and leaves the run as `status="running"`.

**Fix:** Outer `try/except BaseException/finally` around the entire body. The `finally` block ensures `_write_demo_status("failed", error=...)` lands and a terminal `run_state={kind:"failed"}` event is emitted, regardless of where the crash happened.

## 10. Files touched

| File | Change | Lines (est) |
|------|--------|-------------|
| `backend/agents/rlm/run_state.py` | NEW — enum, dataclass, computer | ~280 |
| `backend/agents/rlm/sse_bridge.py` | Add `build_run_state_event` + `__all__` entry | +30 |
| `backend/agents/rlm/run.py` | Wire computer, defensive pid stamp, outer try/finally | +60 |
| `backend/services/events/run_liveness.py` | Sweeper writes `run_state` field + event | +30 |
| `tests/rlm/test_run_state.py` | NEW — transition matrix + threshold tuning | ~250 |
| `tests/rlm/test_sse_bridge.py` | Add `build_run_state_event` test | +20 |
| `tests/services/events/test_run_liveness.py` | Add `run_state` field assertion | +20 |
| `frontend/src/lib/events/rlm-events.ts` | Add `RunStateEvent` interface | +30 |
| `frontend/src/hooks/use-rlm-run.ts` | Reducer tracks `runState` + `runSubstate` | +20 |
| `frontend/src/components/lab/rlm/run-state-pill.tsx` | NEW — plain-language renderer | ~100 |
| `frontend/src/components/lab/rlm/rlm-header.tsx` | Use RunStatePill, retire no-signal-chip | -15 +5 |

Total: ~835 LoC across 11 files (5 new, 6 modified).

## 11. Test plan

**Backend unit (`tests/rlm/test_run_state.py`):**
- Initial state is `INITIALIZING`.
- First `primitive_call(start)` → `WORKING`.
- mtime > 60s → `IDLE`.
- mtime > 240s + no heartbeat 60s → `STUCK`.
- New mtime resets to `WORKING`.
- `run_complete(completed)` → `COMPLETED` and locks.
- Transition into terminal state is absorbing — subsequent ticks are no-ops.
- De-dupe: equivalent substate produces no emission.
- Heartbeat-only updates that don't cross a threshold produce no emission.
- Wall-clock thresholds honor env-var overrides.

**Backend integration (`tests/services/events/test_run_liveness.py`):**
- Sweeper-written `run_state` carries `kind="interrupted"` and a `reason`.
- A run that was orphaned with `pid is None` is now eligible for sweep (after defensive pid stamp lands).

**Frontend (`use-rlm-run.test.ts`):**
- Reducer consumes `run_state` event and exposes `runState` / `runSubstate`.
- Terminal `run_state` does not get overwritten by subsequent events.

**Manual:**
- Start a run, kill the subprocess mid-`implement_baseline`. Within 240s the UI should show `STUCK`, then within 120s+threshold the sweeper marks `INTERRUPTED`.

## 12. Backward compatibility

- Legacy `status` field continues to be written verbatim. Consumers that don't yet know about `run_state` see no behavior change.
- `final_report.json` schema unchanged.
- `dashboard_events.jsonl` gains a new event type. Existing consumers that ignore unknown event types (the FE reducer does) are unaffected.
- The leaderboard projects `run_state.kind` only when present, falling back to `status` for historical rows. No migration needed.

## 13. Out of scope (next iterations)

- Resume button UI surfacing when `kind=interrupted` (follow-up).
- Cost-transparency banner ("Subscription-billed" vs "$x.xx") — separate spec.
- Per-model rubric breakdown for SDAR — separate spec.
- Stuck-state auto-recovery (the system flags, the operator decides; no auto-restart).
