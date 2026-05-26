# Runtime resilience design — PR-μ

**Date:** 2026-05-26
**Status:** design — pending user review
**Authors:** Opus (design), brainstormed from 0.305 Adam max-mode post-mortem

## Background

The 0.305 Adam max-mode run (`runs/prj_0a77e7ed00d0c9da`, 2026-05-26T03:48Z) revealed three independent failure modes that combined to drop the score from a recoverable ~0.535 to 0.305:

1. **Per-experiment wall-clock too tight.** The second `run_experiment` (post-repair) ran for ~33 minutes and timed out with the `{error, outcome, primitive, wall_clock_s}` shape from `binding.py:wrap_primitive`. The agent had likely already applied the `stanfordnlp/imdb` fix; training simply didn't finish.
2. **claude-agent-sdk aclose race.** After the timeout, the pipeline crashed with `RuntimeError: aclose(): asynchronous generator is already running` (status=3). `final_report.json` landed but the FINAL_VAR path never executed cleanly; forced-iteration policy never fired.
3. **Single mega-iteration shape.** The root packed 30 code blocks into ONE REPL turn including conditional repair logic. When iter-2's `run_experiment` died, there was no fresh root-turn to react.

Additionally: `pod_sweeper.py` exists as a dormant CLI — a crashed run can leak a billable RunPod pod indefinitely. The current code path has no atexit, no startup sweep, no periodic sweep.

Below: four solutions matching the four decisions taken in brainstorming. Sandbox-mode-aware throughout (local / docker / runpod).

---

## Solution A — Push SDK aclose Workaround B into `collect_agent_text`

**Decision:** Single chokepoint at `backend/agents/runtime/invoke.py`. Every SDK call inherits the workaround for free.

**Mechanism.** Workaround B is documented at `backend/agents/rdr/agent.py:104` and used in two existing call sites (`claude_oauth_client.py:41` for the root model; `rdr/agent.py:504` for the RDR coding agent). The pattern is: run the async SDK call inside a dedicated thread with its own event loop, so the SDK's nested-generator `aclose()` race stays trapped in the worker thread and never racing the parent's loop.

The 0.305 crash was in `baseline_implementation.py:1789` → `collect_agent_text(...)` — which lives in `backend/agents/runtime/invoke.py:18`. That function is the *only* path the RLM Sonnet sub-agents use for the SDK. If we wrap the workaround there, both `implement_baseline` and the patch-mode `patch_mode_run_with_sdk` get covered with one edit.

**Implementation surface:**
- `backend/agents/runtime/invoke.py` — convert the async-direct call into a thread-isolated call with its own event loop.
- New module `backend/agents/runtime/sdk_isolation.py` — extracts the thread-isolation primitive from `rdr/agent.py` so the three call sites share one implementation.
- `rdr/agent.py:504` and `claude_oauth_client.py` — refactored to call the new shared helper. Removes duplicated thread-management code (no behavior change).

**Failure modes still possible after this lands:**
- A future caller that constructs its own `claude_agent_sdk.query()` call directly without going through `collect_agent_text` would still race. Mitigation: a 1-line unit test that scans the codebase for direct `claude_agent_sdk.query(`/`Client(` instantiations outside the isolation module, fails if any are found.

**Tests:**
- Unit: `tests/test_sdk_isolation.py` — round-trip a fake-SDK call through the new helper; assert no exception propagates even when the fake raises `RuntimeError("aclose(): asynchronous generator is already running")`.
- Integration: run a paper end-to-end on local sandbox, kill the SDK mid-stream via signal, verify the pipeline lands `final_report.json` with `status="failed"` (not crashed) and exit code 0.

---

## Solution B — Mode-scaled `run_experiment` wall-clock cap

**Decision:** efficient=2h, max=6h per single `run_experiment` call. `REPROLAB_RUN_EXPERIMENT_TIMEOUT_S` env var continues to override. Total run still bounded by `--max-wall-clock`, `--max-usd`, `--max-pod-seconds`.

**Today.** The cap derives from `ctx.remaining_s()` (run-level budget) with `REPROLAB_RUN_EXPERIMENT_TIMEOUT_S` as override (`primitives.py:2168-2179`). If the operator doesn't set the env var AND doesn't set `--max-wall-clock`, the cap is essentially unbounded — but in practice it gets squeezed by other accumulated time, leading to the 33-min surprise termination on the 0.305 run.

**Mechanism.** A new `EXPERIMENT_TIMEOUT_BY_MODE` table in `backend/agents/rlm/primitives.py`:

```python
EXPERIMENT_TIMEOUT_BY_MODE: dict[str, int] = {
    "efficient": 7200,    # 2h per call
    "max":      21600,    # 6h per call
}
_DEFAULT_EXPERIMENT_TIMEOUT_S: int = 7200  # fallback when execution_mode unknown
```

Resolution order at `run_experiment` entry:
1. If `REPROLAB_RUN_EXPERIMENT_TIMEOUT_S` set and >0, use it.
2. Else if `ctx.execution_mode in EXPERIMENT_TIMEOUT_BY_MODE`, use that.
3. Else fall back to `_DEFAULT_EXPERIMENT_TIMEOUT_S`.
4. Clamp to `min(resolved, ctx.remaining_s())` — never exceed remaining run budget. If `remaining_s == ∞` (no `--max-wall-clock`), this clamp is a no-op.

**Coherence with the idle watchdog.** `PRIMITIVE_IDLE_BASELINE_S["run_experiment"]` is 2h (no-progress detection). With the mode-scaled cap:
- efficient mode: idle-watchdog (2h) ≈ wall-clock-cap (2h) — coherent. If training stalls, idle fires first; if training runs straight without log progress, wall-clock fires.
- max mode: idle-watchdog (2h) < wall-clock-cap (6h) — coherent. Idle fires on stalls; wall-clock is a long-tail safety net.

**Failure modes still possible:**
- A paper genuinely needs >6h per experiment (rare for reproductions). Operator escape hatch: `REPROLAB_RUN_EXPERIMENT_TIMEOUT_S=43200` (12h).

**Tests:**
- Unit: `tests/rlm/test_experiment_timeout_resolution.py` — table-driven over (env var, execution_mode, remaining_s) → expected cap.
- Integration: a slow-training stub on the local sandbox; verify the cap fires at the mode-scaled value, not 30 min default.

---

## Solution C — Hybrid iteration-boundary discipline

**Decision:** System-prompt nudge + REPL soft signal. Root retains agency; gets a clear cue.

**Today.** The root can write arbitrary numbers of code blocks per REPL turn. The 0.305 run had 30 blocks in iter-1 covering plan → implement → run → score → propose → re-implement → re-run → re-score → FINAL_VAR. When the second `run_experiment` died at block 26, blocks 27-30 still tried to execute but were swallowed by the SDK crash. Forced-iteration policy never fired because FINAL_VAR was never reached.

**Mechanism.**

1. **System prompt addition** (`backend/agents/rlm/system_prompt.py`):

   > **Iteration discipline.** After every `run_experiment` call, *return from the current iteration*. Do not write a follow-up `propose_improvements` / `implement_baseline` / `verify_against_rubric` chain in the same REPL turn. Let the experiment result land as next-iteration context. This is mandatory when `run_experiment` returned `outcome="repairable"` or `outcome="partial_evidence"` — the failure is loud enough that a fresh turn is the only way to react clearly. You can still write multiple `run_experiment` calls across multiple iterations; just one per iteration.

2. **REPL soft signal** (in the `run_experiment` primitive's return-path emit):
   When the primitive returns `outcome in {"repairable", "partial_evidence"}`, emit a `dashboard_events.jsonl` entry with type `run_warning` and code `iteration_boundary_recommended` carrying the failure summary, AND have the REPL's print-after-call inject a one-line banner:
   ```
   ╔═ ITERATION BOUNDARY RECOMMENDED ═╗
   ║ run_experiment returned repairable; end this iteration so the
   ║ failure ({brief}) surfaces as fresh next-turn context.
   ╚══════════════════════════════════╝
   ```

3. **Forced-iteration policy extension.** The existing policy (PR-α-followup) refuses FINAL_VAR when `score < target` AND `iteration_count < REPROLAB_MIN_RUBRIC_ITERATIONS`. Extend it: also refuse FINAL_VAR when the SAME iteration contains TWO `run_experiment` calls with the latter returning `repairable`/`partial_evidence`/`fatal`. This catches the "root chained both attempts into one turn and then tries to FINAL_VAR" anti-pattern.

**Failure modes still possible:**
- A future root model variant that genuinely benefits from multi-experiment turns (e.g., parameter sweeps across 3 hyperparameter values in one turn). For now this is a YAGNI; the forced-iteration policy can be relaxed when that use case lands.

**Tests:**
- Unit: `tests/rlm/test_iteration_boundary_policy.py` — synthesize an iteration with two `run_experiment` calls and a FINAL_VAR; assert policy refuses.
- Unit: `tests/rlm/test_run_warning_emission.py` — call `run_experiment` with a repairable failure; assert `dashboard_events.jsonl` contains the `iteration_boundary_recommended` warning.

---

## Solution D — Layered pod cleanup: atexit + startup + periodic

**Decision:** Three layers, defense-in-depth. atexit catches the common case, startup catches crashes during backend down-time, periodic is the safety net for everything else.

**Today.** `pod_sweeper.py` exists as a fully-formed CLI with `sweep_stale_pods(max_age_seconds, dry_run, ...)`. Nothing auto-triggers it. The 0.305 Adam crash's exit-status-3 means *any* pod that run created could in principle still be billing (we manually killed `jvu9ai82zf5o11` last session — that was an orphan from a different run, but the pattern is the same).

**Mechanism.**

1. **Layer 1 — atexit in run subprocess** (`backend/services/runtime/runpod_backend.py`):
   - On `RunpodBackend.acquire()`, register an `atexit.register(self._cleanup_atexit)` that calls `self.destroy()`.
   - `_cleanup_atexit` is idempotent (safe to call twice — once from atexit, once from the normal lifecycle's finally).
   - On normal exit, the lifecycle's `finally` block runs first and tears down; atexit's cleanup is a no-op.
   - On crash (uncaught exception, signal, hard exit), atexit runs and tears down.
   - Doesn't catch SIGKILL or OOM-killer or `os._exit()` — those bypass atexit.

2. **Layer 2 — startup sweep** (`backend/app.py` `create_app()` lifespan startup):
   - On backend boot, call `pod_sweeper.sweep_stale_pods(max_age_seconds=2*3600, dry_run=False)` in a background task.
   - 2h threshold means we never reap a pod that a still-active run might be using (run-level wall-clock cap is much tighter).
   - Fail-soft: any error in the sweep is logged but doesn't block backend startup.
   - Disabled when `REPROLAB_RUNPOD_API_KEY` is unset (no RunPod usage → no cleanup needed).

3. **Layer 3 — periodic background sweep** (new module `backend/services/runtime/pod_sweep_scheduler.py`):
   - Runs `sweep_stale_pods` every 30 minutes.
   - Started by the FastAPI lifespan together with the startup sweep.
   - Stoppable via lifespan shutdown.
   - Same fail-soft behavior.
   - Configurable: `REPROLAB_POD_SWEEP_INTERVAL_S` (default 1800), `REPROLAB_POD_SWEEP_MAX_AGE_S` (default 7200), `REPROLAB_POD_SWEEP_ENABLED` (default true, can disable for dev/test).

**Sandbox-mode coverage:**
- **runpod**: all three layers active.
- **docker**: atexit registers `docker rm -f <container>` for the spawned container. No periodic sweep needed (Docker doesn't bill by the hour). Local-only.
- **local**: atexit kills the child training process if alive (`os.killpg(pgid, SIGTERM)` then SIGKILL). No periodic sweep — local processes don't leak in a costly way.

**Failure modes still possible:**
- SIGKILL of the backend process itself — no atexit fires, no lifespan shutdown. Mitigation: startup sweep on next backend boot catches it. Worst case: 30 min of $/hr leak per orphan before the periodic sweep.
- Pods created by a different account / API key that the sweeper doesn't have credentials for — by design, we only sweep our own.
- Pods labeled `delete_on_destroy=false` (operator debug override) — by design, we honor that flag. Operator must clean up themselves.

**Tests:**
- Unit: `tests/test_pod_sweeper_atexit.py` — mock RunPod backend, register atexit, raise, assert pod terminated.
- Unit: `tests/test_pod_sweep_scheduler.py` — mock sweeper, fake clock, assert called at interval.
- Integration: spawn a real `START_FULL_SMOKE=1` run, SIGKILL the run subprocess mid-flight, assert the backend's startup sweep on next boot cleans up the orphan.

---

## Cross-cutting: failure-class → action table

A clarifying single-source table for what fires what (mostly already exists in code, this is documentation):

| Failure | Detection | Action | Layer |
|---|---|---|---|
| `run_experiment` exceeds wall-clock cap | `binding.py:wrap_primitive` ThreadPool timeout | Return `{outcome: partial_evidence, wall_clock_s}` | Solution B |
| `run_experiment` no log progress for 2h | `run_watchdog.py` idle detector | Kill child process, return `{outcome: repairable}` | Existing (PR-ι) |
| `run_experiment` returns repairable | `primitives.py` outcome classifier | Emit `iteration_boundary_recommended` warning | Solution C |
| SDK aclose race | `collect_agent_text` wrapper | Thread-isolate; swallow the race | Solution A |
| Run subprocess exits (any reason) | atexit | Tear down pod / container / local proc | Solution D.1 |
| Backend boots with prior-run pods alive | lifespan startup | Sweep pods older than 2h | Solution D.2 |
| Pod leaks past startup sweep | Periodic 30-min sweep | Reap stale pods | Solution D.3 |
| CUDA OOM mid-training | sandbox runner | Escalate GPU SKU (already in place) | Existing |
| Network blip on dataset download | retry in dataset loader | Try fallback mirror (PR-λ recipes) | Existing |
| Preflight catches code bug (e.g. `imdb`) | `primitives.py:run_experiment` preflight | Return `{outcome: repairable, contract_violations}` | Existing |

---

## Out of scope (and why)

- **Resume-from-checkpoint** — if `run_experiment` dies mid-training, persist the checkpoint so iter-2 can resume. *Why deferred:* requires a checkpoint-format contract with every model architecture; major design surface. Belongs in its own spec (PR-ν).
- **Mid-training OOM auto-escalation** — already exists via the dynamic-GPU escalation ladder per CLAUDE.md.
- **Cross-sandbox semantic differences** beyond pod/container cleanup — local sandbox doesn't billing-leak, docker sandbox doesn't billing-leak. Atexit covers the cleanup; nothing else to reconcile.
- **RDR coding agent SDK isolation** — already implemented at `rdr/agent.py:504`. Solution A refactors it to share the new helper but doesn't change its behavior.

---

## Acceptance criteria

The Adam max-mode rerun must:
1. Land `final_report.json` with non-zero `iteration_count`, even if the SDK races at cleanup. (Solution A)
2. Complete at least one `run_experiment` that runs for >30 min without spurious termination. (Solution B)
3. Show ≥2 iterations in `iterations.jsonl` if the first `run_experiment` returns repairable. (Solution C)
4. Leave zero RunPod pods alive after exit (verified via `gh-run-followup`-style log check on the RunPod account). (Solution D)

Test suite must:
- All existing tests pass (2701 currently).
- New tests from Solutions A-D pass (estimated +15 unit + 2 integration).
- An adversarial test that injects an SDK aclose race into `collect_agent_text` does not crash the pipeline.

---

## Rollout

- All four solutions land as one commit `PR-μ` per repo conventions (the user prefers infrequent, substantial commits).
- Environment-variable kill switches for each layer:
  - `REPROLAB_SDK_ISOLATION_DISABLED=true` — disables Solution A (bypass workaround if it causes other issues)
  - `REPROLAB_RUN_EXPERIMENT_TIMEOUT_S` — already exists; overrides Solution B
  - `REPROLAB_ITERATION_BOUNDARY_ENFORCEMENT=off|advise|enforce` — Solution C policy strength (default `advise` = hybrid; `enforce` = library-level forced boundary; `off` = no policy)
  - `REPROLAB_POD_SWEEP_ENABLED=true|false` — Solution D master switch

## Observability

New SSE event types: none (reuse `run_warning` with codes `iteration_boundary_recommended`, `sdk_aclose_swallowed`, `pod_sweep_summary`).

New dashboard event codes:
- `sdk_aclose_swallowed` — Solution A swallowed a race
- `iteration_boundary_recommended` — Solution C banner fired
- `pod_sweep_summary` — Solution D sweep ran; includes pods-reaped count + reaped-pod-ids list
- `atexit_cleanup` — Solution D.1 atexit cleanup fired with which resource (pod_id, container_id, pid)

---

## Estimated effort

| Solution | LOC | Risk | Reviewer time |
|---|---|---|---|
| A — SDK isolation | ~150 (refactor 2 existing call sites, new helper module) | Low — pattern proven in existing code | 20 min |
| B — Timeout cap | ~50 (lookup table + resolver) | Low — additive | 15 min |
| C — Iteration discipline | ~80 (system prompt + REPL emit + forced-iter extension) | Medium — touches root model behavior | 30 min |
| D — Pod cleanup | ~200 (atexit + scheduler + lifespan wire-in) | Medium — must not block startup if RunPod auth fails | 30 min |
| **Total** | **~480** | | **~95 min review** |
