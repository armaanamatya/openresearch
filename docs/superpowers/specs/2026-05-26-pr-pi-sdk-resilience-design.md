# PR-π — SDK call resilience + run liveness + ingestion precondition

Date: 2026-05-26
Status: Locked. Codex executes; Opus reviews diff. Sonnet fallback if Codex stalls >15min.
Author: Opus (design); Codex (impl); Opus (review).

## 1. Why this exists — the failure that motivated it

VAE re-dispatch on 2026-05-27T00:00:35 UTC died at 00:09 UTC inside `implement_baseline`. The stderr file shows 7 `aclose(): asynchronous generator is already running` errors followed by the CLI's `{"status": "partial"}` summary. The dashboard's last event was `worker_report_started agent=implement_baseline` at 00:05:34 with NO succeeded/failed counterpart. NO `commands.json`, NO `train.py`, NO `_openresearch_curated.py` were written — meaning the PR-ξ γ knowledge-channel mechanism never even executed. `demo_status.json` remained `status=running` (stale) because the Python process exited bypassing both `_mark_demo_status_stopped` (graceful) and `_mark_demo_status_failed` (uncaught exception).

The existing PR-μ defenses (`sdk_isolation.run_isolated`, the 120s stall watchdog at `primitives.py:1373-1472`, the `live_runs.py` stderr-tail watchdog) all assume specific failure shapes:

- **`sdk_isolation.py:64`**: only swallows aclose race AFTER `captured_result` is populated. If aclose fires DURING streaming (before `async for` collects all events), it propagates as a real exception and the worker thread re-raises it on the calling event loop.
- **`primitives.py:1421` watchdog**: only triggers when `commands.json` exists on disk. If the SDK deadlocks BEFORE the sub-agent emits commands.json, this gate is never crossed and the 4h `_timeout_for(ctx, 14400)` is the only safety net.
- **`live_runs.py:343` watchdog**: only flags `demo_status.degraded=True` after threshold; doesn't recover, doesn't write a terminal state, and only runs for FastAPI-server-spawned subprocesses (NOT CLI-spawned ones).
- **`_mark_demo_status_failed`**: only fires via Python's exception handler, never under SIGKILL / host suspend / OOM-killer.

There is also a precondition failure at the START of every VAE run: `parsed_full_text.txt missing — parser likely failed; falling back to workspace variable (lossy)`. This silent degradation means `paper_grounding.assert_paper_grounded` (PR-ξ Bug 1a) no-ops on the empty file path, defeating the PR-ξ grounding check.

## 2. Design principles

- **One canonical abstraction per failure class**, not scattered patches.
- **Fail visibly**: every run terminates with `status ∈ {completed, failed, interrupted, stopped}`. Never silent `status=running` with a dead PID.
- **Defensive narrowing**: every SDK call returns a structured outcome (success / aclose-pre-result-retry / aclose-post-result-swallow / real-error), not a bare `T`.
- **Modular boundaries**: SDK resilience lives in `runtime/`. Liveness lives in `services/events/`. Watchdog policy lives in `primitives.py`. No leakage.
- **Backwards compatible**: every change opt-in via parameter or config, with defaults that match prior behavior except where the prior behavior was broken (e.g., pre-emit watchdog).
- **YAGNI**: no abstractions for hypothetical future failure modes. Only what the VAE run + adjacent papers (Adam, Dropout, SDAR) need.

## 3. Scope — five modules

### Module A — `backend/agents/runtime/sdk_isolation.py` (extend)

**Add a structured outcome and retry-on-streaming-aclose.**

New types:
```python
class IsolationFailureKind(str, Enum):
    ACLOSE_PRE_RESULT = "aclose_pre_result"   # race during streaming, retryable
    ACLOSE_POST_RESULT = "aclose_post_result" # race at cleanup, swallowed
    REAL_EXCEPTION = "real_exception"         # actual error from coro body

@dataclass(frozen=True)
class IsolationOutcome:
    kind: Literal["ok", "aclose_post_result_swallowed", "aclose_pre_result_retried", "real_exception"]
    attempt_count: int                # 1 = no retry; 2 = retried once
    stderr_excerpt: str = ""           # last 4KB of worker stderr (best-effort)
```

New API:
```python
async def run_isolated(
    coro_factory: Callable[[], Coroutine[Any, Any, T]],   # FACTORY now, not coro
    *,
    max_retries: int = 1,
    name: str = "sdk-isolation",
) -> T:
    """Thread-isolated SDK call with aclose retry semantics.
    
    On ACLOSE_PRE_RESULT: discard the worker, call coro_factory() again (fresh
    coro on a fresh worker thread + event loop), up to max_retries times.
    On ACLOSE_POST_RESULT: swallow, return captured_result.
    On REAL_EXCEPTION: propagate.
    
    coro_factory MUST return a new coroutine each call — coroutines can only
    be awaited once. Existing callers that pass a coro directly will get a
    DeprecationWarning + auto-wrapping for one release cycle.
    """
```

**Why factory instead of coro**: a coroutine can only be awaited once. To retry, we need a way to recreate the coroutine. Factory pattern is cleanest. Backwards-compat shim wraps a bare coro in `lambda: coro` for one release with a deprecation warning, then we remove it.

**Detection of pre-vs-post-result aclose**: the worker thread sets `captured_result` BEFORE the asyncgen finalizer runs. If `_is_aclose_race(exc)` fires and `captured_result` is non-empty → POST_RESULT (existing logic). If `captured_result` is empty → PRE_RESULT (new — retry).

**Stderr capture**: worker thread redirects stderr to a `StringIO` buffer for the duration of the coro; last 4KB attached to outcome for diagnostics.

### Module B — `backend/services/events/run_liveness.py` (NEW)

**Orphan-run sweeper. Idempotent. Fail-soft.**

```python
@dataclass(frozen=True)
class OrphanReport:
    project_id: str
    last_status: str
    last_updated_at: datetime
    pid: int | None
    reason: str

def sweep_orphaned_runs(
    runs_root: Path,
    *,
    stale_after_s: float = 120.0,
    emit_event: bool = True,
) -> list[OrphanReport]:
    """Scan runs_root/*/demo_status.json. For each status=running:
      - If pid is missing OR _pid_alive(pid) is False AND
        (now - parse_iso(updatedAt)) > stale_after_s:
        → mark as orphan, write terminal state, return OrphanReport.
    
    Terminal state writes (atomic, idempotent):
      - demo_status.json: status=interrupted, degraded=True,
        degraded_reason='run process disappeared (host suspend / SIGKILL / OOM)',
        completedAt=now
      - dashboard_events.jsonl: append {event: run_interrupted, ...} if emit_event
      - final_report.json (only if missing): {
          status: 'interrupted',
          mode: <from demo_status>,
          paperId: <from demo_status>,
          projectId: <from demo_status>,
          startedAt: <from demo_status.startedAt>,
          completed_at: now (ISO),
          iterations: <count of rlm_state/iterations.jsonl lines>,
          rubric_score: <last rubric from rlm_state if any, else 0.0>,
          cost_usd: <sum of cost_ledger.jsonl entries, else 0.0>,
          reason: 'orphaned',
        }
    
    Returns the list of OrphanReports for caller logging. Idempotent: a run
    already marked terminal is skipped.
    """

def _pid_alive(pid: int) -> bool:
    """Linux: send signal 0 via os.kill. Windows: psutil if available, else
    fall back to True (conservative — don't false-positive on Windows)."""
```

**Invocation points**:
- `backend/cli.py:_module_main` — call once at startup, log orphans found.
- `backend/app.py` FastAPI lifespan startup — call once at startup.
- `backend/services/events/live_runs.py` — add a 60s periodic task that calls it (only when run via the FastAPI server; CLI doesn't need periodic since it's one-shot).

**PID instrumentation**: every code path that writes `demo_status.json status=running` must include `"pid": os.getpid()`. Audit and add:
- `backend/cli.py:_atomic_write_json` callers around line 1005-1013 (Lane U status write).
- `backend/services/events/live_runs.py` subprocess spawn — write pid into demo_status after `Popen`.
- Any other writer of demo_status.json with status=running.

### Module C — `backend/agents/rlm/primitives.py` (extend implement_baseline watchdog)

**Add pre-commands.json stall detection.**

Current logic at `primitives.py:1397-1463` only checks files in `code_dir` AFTER `commands.json` exists. Add a parallel branch for the pre-emit case:

```python
# Existing:
_ACLOSE_STALL_S = 120  # post-emit stall

# NEW:
_PRE_EMIT_STALL_S = 240   # 4 min of total SDK silence before commands.json
_PRE_EMIT_PROGRESS_FILES = ("commands.json", "train.py", "_openresearch_curated.py")

# In the polling loop:
if not commands_json.exists():
    # NEW pre-emit branch
    progress_files = [
        f for f in code_dir.iterdir()
        if f.is_file() and f.name in _PRE_EMIT_PROGRESS_FILES
    ]
    if not progress_files:
        # Track when implement_baseline started; if elapsed > _PRE_EMIT_STALL_S, escalate
        if _pre_emit_stall_start is None:
            _pre_emit_stall_start = _time.time()
        elif _time.time() - _pre_emit_stall_start > _PRE_EMIT_STALL_S:
            logger.warning(
                "implement_baseline: SDK silent for %ds with no code emission. "
                "Escalating to repairable error.",
                int(_time.time() - _pre_emit_stall_start),
            )
            _err = _with_outcome({
                "success": False,
                "error": (
                    f"implement_baseline: SDK pre-emit stall — no code written "
                    f"in {_PRE_EMIT_STALL_S}s. Likely SDK aclose deadlock pre-result."
                ),
            }, PrimitiveOutcome.repairable)
            # Emit SSE warning so the UI shows it
            _emit_run_warning(ctx, code="sdk_pre_emit_stall", message=...)
            _cache.put(...)
            return _err
    else:
        # Any progress file appeared — reset pre-emit timer, transition to post-emit logic
        _pre_emit_stall_start = None
```

Coupled to Module A: when the SDK retry from `run_isolated` succeeds on attempt 2, the pre-emit timer should reset (because the second attempt does make progress).

### Module D — `backend/cli.py` (orphan sweep + resume offer)

**At CLI startup (`_module_main` early in flow)**:
```python
from backend.services.events.run_liveness import sweep_orphaned_runs

orphans = sweep_orphaned_runs(runs_root)
if orphans:
    print(f"[orphan-sweep] marked {len(orphans)} interrupted run(s):")
    for o in orphans:
        print(f"  {o.project_id}  ({o.reason})")
```

**On `reproduce --project-id <existing>` with prior `interrupted` state**:
```python
prior_status_path = runs_root / args.project_id / "demo_status.json"
if prior_status_path.exists():
    prior = json.loads(prior_status_path.read_text())
    if prior.get("status") == "interrupted":
        last_iter = _count_iterations(runs_root / args.project_id)
        last_rubric = _read_last_rubric(runs_root / args.project_id)
        print(
            f"Detected interrupted prior run for {args.project_id} "
            f"(iter={last_iter}, last_rubric={last_rubric:.2f})."
        )
        if not args.yes:
            answer = input("Resume from last checkpoint? [Y/n] ").strip().lower()
            if answer in {"", "y", "yes"}:
                args.resume = True
```

The `--resume` flag already exists for RLM (it loads `rlm_state/iterations.jsonl` and `repl_state.pickle`). The CLI just needs to offer the prompt and set the flag.

### Module E — `backend/services/ingestion/parser/service.py` (precondition fail-fast)

Current at `service.py:161, 197`: parser writes `parsed_full_text.txt` on success, OR deletes it on failure. Downstream code (the workspace builder + paper_grounding) silently falls back to the lossy workspace variable when the file is missing.

**Change**: add a config-gated strict mode (default ON) where missing `parsed_full_text.txt` after parser cascade fails the run BEFORE the RLM loop starts. Override via `--allow-lossy-paper-text` for diagnostic runs.

```python
# In backend/agents/rlm/run.py, before starting the RLM loop:
parsed_path = project_dir / "parsed_full_text.txt"
if not parsed_path.exists() or parsed_path.stat().st_size < 1024:
    if not ctx.settings.allow_lossy_paper_text:
        raise PreconditionError(
            f"parsed_full_text.txt missing or <1KB at {parsed_path}. "
            f"Parser likely failed. Re-run ingestion or pass --allow-lossy-paper-text."
        )
    logger.warning("paper text degraded — proceeding with lossy workspace fallback")
```

Couples to Module B: if PreconditionError fires before any iteration runs, the CLI writes `status=failed` (via existing `_mark_demo_status_failed`), and no orphan sweep is needed.

## 4. Tests (all REQUIRED to pass before merging)

### `tests/agents/runtime/test_sdk_isolation_resilience.py`
- `test_aclose_post_result_swallowed_returns_result` — race fires after result, return T, kind="aclose_post_result_swallowed".
- `test_aclose_pre_result_retries_then_succeeds` — first attempt races during streaming (no captured_result), second attempt succeeds, kind="ok", attempt_count=2.
- `test_aclose_pre_result_exceeds_max_retries_raises` — both attempts race pre-result, raises `IsolationFailure` with kind="aclose_pre_result_exhausted".
- `test_real_exception_propagates_immediately` — coro raises ValueError, no retry, propagates as ValueError.
- `test_factory_called_fresh_each_retry` — factory called exactly attempt_count times.
- `test_stderr_excerpt_captured_on_failure` — stderr bytes show up in IsolationOutcome.stderr_excerpt.

### `tests/services/events/test_run_liveness.py`
- `test_sweep_marks_orphan_with_dead_pid` — fixture: runs/prj_X/demo_status.json with status=running, pid=99999 (dead), updatedAt=200s ago. After sweep: status=interrupted, final_report.json exists with status=interrupted.
- `test_sweep_skips_orphan_with_live_pid` — pid=os.getpid() (alive), even with stale updatedAt, NOT marked.
- `test_sweep_skips_recently_updated` — pid dead but updatedAt=30s ago (under threshold), NOT marked.
- `test_sweep_idempotent` — call twice, second is no-op (no double-write of final_report.json).
- `test_sweep_writes_final_report_with_iter_count_from_jsonl` — rlm_state/iterations.jsonl has 3 lines → final_report.iterations==3.
- `test_pid_alive_signal_0` — _pid_alive(os.getpid()) True; _pid_alive(99999) False.

### `tests/agents/rlm/test_implement_baseline_pre_emit_stall.py`
- `test_pre_emit_stall_returns_repairable_after_threshold` — code_dir stays empty for >_PRE_EMIT_STALL_S, returns repairable with code="sdk_pre_emit_stall".
- `test_pre_emit_progress_resets_timer` — train.py appears at t=120s, no escalation even though >_PRE_EMIT_STALL_S has elapsed total.

### `tests/cli/test_resume_offer.py`
- `test_resume_offer_on_interrupted_prior_run` — fixture: prior run with status=interrupted, iter=2, last_rubric=0.45. CLI invocation with --project-id and stdin "y" sets args.resume=True.
- `test_no_resume_offer_when_no_prior_run` — fresh project_id, no offer.

### `tests/services/ingestion/parser/test_precondition.py`
- `test_run_fails_fast_when_parsed_full_text_missing` — project_dir has no parsed_full_text.txt, RLM run fails with PreconditionError.
- `test_allow_lossy_paper_text_override` — same fixture + --allow-lossy-paper-text flag, run proceeds (logs warning).

## 5. Backwards compatibility

- `run_isolated` accepts a bare coroutine for one release (auto-wrap + DeprecationWarning). Migrate `backend/agents/runtime/invoke.py:_do_sdk_call` to factory pattern in this PR.
- Existing demo_status.json files without `pid` → orphan sweeper treats `pid=None` as "stale check via updatedAt only" (same path as dead pid).
- `--allow-lossy-paper-text` defaults to True for backwards compat in this PR; flip default to False in PR-ρ after observing prod for a week.

## 6. Non-goals (out of scope)

- Resumable mid-iteration RLM state. We resume from iteration boundaries, not mid-primitive.
- Auto-redispatch from CLI on orphan detection. CLI prints the orphan list and exits; redispatch is the user's call.
- Fixing the parser itself (task #126). PR-π only adds the precondition gate.
- Multi-paper batch resume. PR-π handles single-project resume.
- claude-oauth → API key fallback on aclose. The retry-on-pre-result handles transient aclose; persistent aclose under one auth surface is a separate diagnosis.
- Sub-agent OOM detection (deferred to PR-ρ — no current evidence of OOM in VAE failure mode).

## 7. File-by-file impact summary

| Path | Action | LoC |
|---|---|---|
| `backend/agents/runtime/sdk_isolation.py` | extend | +120 |
| `backend/services/events/run_liveness.py` | new | +200 |
| `backend/agents/rlm/primitives.py` | extend (pre-emit watchdog) | +60 |
| `backend/agents/rlm/run.py` | add precondition gate | +20 |
| `backend/cli.py` | orphan sweep + resume offer | +80 |
| `backend/services/ingestion/parser/service.py` | (no change — gate is in run.py) | 0 |
| `backend/config.py` | add `allow_lossy_paper_text` setting | +5 |
| `backend/app.py` | FastAPI lifespan startup hook | +10 |
| `backend/services/events/live_runs.py` | periodic sweep + pid into demo_status | +30 |
| `backend/agents/runtime/invoke.py` | migrate to factory pattern | +10 |
| `tests/agents/runtime/test_sdk_isolation_resilience.py` | new | +250 |
| `tests/services/events/test_run_liveness.py` | new | +200 |
| `tests/agents/rlm/test_implement_baseline_pre_emit_stall.py` | new | +180 |
| `tests/cli/test_resume_offer.py` | new | +120 |
| `tests/services/ingestion/parser/test_precondition.py` | new | +90 |

Total: ≈+1375 LoC.

## 8. Acceptance criteria

1. `pytest tests/` passes (all old + new).
2. `pytest tests/agents/runtime/test_sdk_isolation_resilience.py tests/services/events/test_run_liveness.py tests/agents/rlm/test_implement_baseline_pre_emit_stall.py tests/cli/test_resume_offer.py tests/services/ingestion/parser/test_precondition.py -v` shows ≥25 new test cases passing.
3. A simulated dead PID with stale updatedAt is converted to terminal state within one sweep.
4. A simulated streaming-phase aclose retries once and returns the captured result.
5. No existing tests break.
6. No new lint/type errors (where checked).

## 9. Risks + mitigations

- **Risk: factory pattern breaks existing callers.** Mitigation: auto-wrap shim with DeprecationWarning for one release.
- **Risk: orphan sweeper races with a slow-starting run.** Mitigation: 120s `stale_after_s` floor; runs that don't write demo_status within 120s of starting will be marked, but that's the correct behavior (something is broken).
- **Risk: pre-emit watchdog mis-fires on slow Sonnet code emission.** Mitigation: 240s threshold (4 min) is generous; if it consistently misfires, raise to 360s.
- **Risk: precondition gate breaks CLI runs on unparseable papers.** Mitigation: `--allow-lossy-paper-text` defaults True for one release, gives time to fix parser separately.
- **Risk: pid_alive false-positive on Windows.** Mitigation: conservative fallback to True; orphan sweep won't mark Windows runs aggressively (acceptable — fewer false positives matter more than fewer false negatives on Windows where Linux watchdog primary).

## 10. Sequencing

1. Codex implements Module A + tests A (most isolated).
2. Codex implements Module B + tests B.
3. Codex implements Module C + tests C (depends on A pattern but no direct dep).
4. Codex implements Module D + tests D (depends on B).
5. Codex implements Module E + tests E (independent).
6. Codex runs full `pytest tests/`, reports green.
7. Opus reviews diff, requests changes if needed.
8. Single substantial commit (per memory rule): "PR-π SDK resilience + orphan-run liveness + ingestion precondition — VAE-class failure preempt".
9. Redispatch VAE (task #130).
