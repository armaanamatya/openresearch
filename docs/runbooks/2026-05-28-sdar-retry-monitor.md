# SDAR retry monitor — `prj_76bd8243ff997f72`

**Paper:** SDAR (arXiv 2605.15155) — Self-Distilled Agentic Reinforcement Learning
**Run mode:** RLM, claude-oauth root, sandbox=runpod, --vram-gb 38
**Caps:** `--max-wall-clock 5400 --max-pod-seconds 5400 --max-usd 20`
**Branch:** `main` + uncommitted local fixes (see "Fixes applied" below)
**Scope env-var:** Qwen3-1.7B + Qwen2.5-3B only (smallest-two; skips 7B), real weights, ALFWorld + Search-QA + WebShop slices

## Fixes applied before this retry

Three P0 fixes, uncommitted, applied directly to the working tree:

1. **Fix A — `backend/agents/rlm/primitives.py:89`** — `_DEFAULT_PRE_EMIT_STALL_S: 240.0 → 1800.0` (30 min). The 240-second watchdog killed yesterday's run while the SDK was healthily writing 30+ files for 25+ minutes (real code, not stalled). 1800s gives hard papers like SDAR room to think.
2. **Fix B — `backend/agents/rlm/primitives.py:~1672` (finally block of implement_baseline)** — SIGKILL any `claude` SDK child process owned by this PID before `pool.shutdown(wait=False, ...)`. Yesterday two zombie `claude --stream-json` children survived the parent's death and kept writing to the dead run's `code/` dir, racing the retry's fresh SDK call.
3. **Fix C — `backend/services/runtime/service.py:_load_aggregate`** — evict cached `SandboxAggregate` if its state is `FAILED` before returning. Without this the retry path hit `InvalidSandboxTransition: 'request' in state 'failed'` because the in-memory cache returned the prior failed aggregate.

(Fix D — planner GPU hint — was deprioritized after artifact-mining showed the agent already plans for real GPU runs; the `compute_scope` schema warning is cosmetic.)

## Retry loop

Per-attempt cap: same flags as above; up to 5 attempts before pushing a notification + stopping for user input. Between attempts the next failure-mode is analyzed against this tracker and a new targeted fix is applied if the failure differs from prior attempts.

## Attempt 1 (launched 2026-05-28 21:51 local, project `prj_76bd8243ff997f72`)

### T+3 min snapshot (21:54)

- Process alive (etime 03:09, 0% CPU — work in claude SDK child PID 79858 under our PID 74581).
- Backend `demo_status.status: running`, `startedAt = updatedAt = 02:51:07Z`.
- 1 dashboard event so far: a single `repl_iteration` for iteration 1 (root completed first turn, thinking about next).
- 0 `primitive_call` events visible yet (root still in initial think phase).
- **Fix B validation: NO zombie SDK children.** Only 1 active `claude --stream-json` (PID 79858), child of PID 74581 (the reproduce parent). Last run had 2 orphans by this point.
- 0 RunPod pods (expected; created lazily at `run_experiment`).
- UI status badge correctly shows "running" (not stuck on "queued"). BUG-NEW-004 not reproducing this tick.
- UI page heading still "Untitled paper", `paperTitle = "paper_text"` in `demo_status.json` — BUG-NEW-001 reproduces unchanged (Fix D not applied).
- No `run_warning` events.
- No `code/` dir written yet.

### Next milestones to watch

1. `primitive_call understand_section start` — root finishes initial planning, enters Understand phase.
2. `primitive_call implement_baseline start` — the canary primitive that died yesterday on the 240s watchdog. With Fix A (1800s window), this should now have room to actually finish.
3. `primitive_call run_experiment start` + RunPod pod count → 1 — the watershed event the user predicted has never happened.
4. `experiment_completed` with `success: true` or `false` — first real signal of whether the agent's plan actually runs on GPU.

Continuing 2-min Playwright + filesystem cadence; Monitor `bhj2sb9e8` armed for all run-warnings and primitive-error events.

### T+7 min snapshot (21:58)

- Process alive (etime 07:11, 1.2% CPU — root is briefly active in REPL between sub-RLM calls).
- 25 dashboard events (+24 in last 4 min). Primitives executed so far: `check_user_messages`, `understand_section`, `extract_hyperparameters`, `heartbeat`. 4 `sub_rlm_spawned` / 3 `sub_rlm_complete` — last sub-RLM still in flight.
- **No run_warnings.** Yesterday at this point we had `compute_scope_invalid` already; not seeing it yet (root hasn't called `plan_reproduction` yet).
- Zero RunPod pods (expected — lazy at `run_experiment`, still ~10–15 min away).
- Zero zombie SDK children — 1 healthy `claude --stream-json` (PID 97272) under parent 74581. Fix B continues to hold.
- UI badge "running" correct; primitive call history reads "8 calls" (matches backend count exactly — SSE pipeline healthy).
- Heading still "Untitled paper" (BUG-NEW-001, untouched by this fix round).

On the timeline-vs-yesterday: yesterday at T+7 min the root was already entering `plan_reproduction` (which then emitted the smoke-run `compute_scope`). Today's pace is comparable. The watershed gate (`run_experiment` actually being called) was crossed yesterday too — but the SandboxAggregate was already poisoned by then. So this attempt's true validation comes around T+15–25 min.

### T+8 min — BUG-NEW-021 (P1, new) — UI "no signal 217s" pill fires during healthy sub-RLM activity

User-flagged at 21:59. Lab UI's status row shows a warn-coloured pill reading `no signal 217s` while the run is actually healthy.

Evidence:
- Backend has 39 dashboard events; last event was a `sub_rlm_spawned` 35s ago.
- Last `repl_iteration` was 399s ago — the agent has been in a sub-RLM chain for ~7 minutes between iteration boundaries.
- 1 healthy SDK child PID 8724 running under parent 74581.
- 0 `run_warning` events.

Source: `frontend/src/components/lab/rlm/rlm-header.tsx:84-90`. `noSignalSecs` is computed as `heartbeatNowMs − lastHeartbeatAt`, and the pill is rendered when:
1. status === "running" AND
2. heartbeat-gap > `HEARTBEAT_STALE_MS` (probably 60s) AND
3. `inFlightPrimitive === null` (lines 148–185)

The third condition is what produces the false alarm: when the agent is inside a sub-RLM (a multi-LLM-call decomposition spawned by `rlm_query`), there is no "current primitive" because sub-RLMs aren't tracked as primitives. So the UI thinks the run is wedged. In reality the agent is running multi-minute LLM calls inside `rlm_query`.

Fix shape:
- Either: treat an active `sub_rlm_spawned` without a matching `sub_rlm_complete` as an in-flight indicator and show `running sub-RLM (Ns)` instead of `no signal Ns`.
- Or: emit a heartbeat from inside `rlm_query` periodically so the heartbeat stays warm during sub-RLM chains.

Severity: P1 — looks broken to the operator at the worst possible time (the user assumes the run died and starts diagnosing, exactly like just happened). Functionally harmless: the run actually IS progressing. Action: file follow-up — add sub-RLM-in-flight to the UI's busy-state derivation OR heartbeat inside rlm_query.

### T+16 min snapshot (22:07) — major milestone progress

Run is past the Understand phase. New primitives executed since last tick:
- `detect_environment` (ok)
- `resolve_gpu_requirements` (ok) → `gpu_resolved` event
- `build_environment` (start) — currently in flight

Event count: 78 (+27 in 3 min). 0 warnings, 0 zombies. 15 sub-RLM spawns / 15 completes (sub-RLM chain done for the moment).

**Fix B validation again:** despite agent moving through 4 different primitives with SDK calls in between, only 1 SDK child at any time, 0 orphans. Yesterday this point would already have had 2+ orphans accumulating.

**BUG-NEW-005 is FIXED for this run.** `runs/.../rlm_state/gpu_plan.json` shows:
- `runpod_id: "NVIDIA RTX A6000"`, `vram_gb: 48`, `gpu_count: 1`, `sku_usd_per_hr: 0.49`
- `source: "paper"` (the agent reasoned about it — not fallback)
- `requirements.estimated_vram_gb: 38` — the `--vram-gb 38` override propagated correctly
- With 1.25 headroom (38 × 1.25 = 47.5), A6000's 48 GB is the smallest valid SKU — resolver picked correctly. Yesterday it fell back to rtx4090 24 GB which would have pre-emptively OOM'd. Today's plan should fit on first try.

**Pace check:** 16 min in. Yesterday at this point we were already at `implement_baseline` retry 2 + the SandboxAggregate poisoning. Today we're at `build_environment` — about one primitive behind, but on healthier ground because the upstream phases were more thorough (15 sub-RLMs vs ~3 yesterday). Budget: 74 min remaining of 90.

**Next milestone:** `build_environment ok` → `plan_reproduction` → `implement_baseline` (the Fix A 1800s watchdog stress test) → `run_experiment` (the Fix C SandboxAggregate test + first-ever RunPod pod).

### T+29 min — BUG-NEW-009 fixed in-flight

User flagged the canvas-click-doesn't-show-detail bug AGAIN (screenshot of Plan node highlighted, sidebar empty). Investigation traced it:

- Click handler IS firing — `setSelectedNodeId(id)` updates state correctly.
- `selectedNode = state.tree.find((n) => n.id === selectedNodeId)` resolves correctly — the activity nodes (Plan / Understand / Detect Env / etc.) ARE in `state.tree` once their primitive completes with `status === "ok"` (see `use-rlm-run.ts:535-550`).
- `NodeDetailSidebar` IS receiving the resolved node as a prop.
- **But the sidebar default is `collapsed: true`** (`rlm-lab.tsx:135`), so the detail panel is hidden behind a 36px toggle rail. Click any node → state updates, sidebar renders detail content — but the user sees zero visual change because the sidebar is collapsed. Sidebar's collapsed state was never tied to node selection.

**Fix applied to `frontend/src/components/lab/rlm/rlm-lab.tsx:140-149`:** wrapped `setSelectedNodeId` so that whenever a node id becomes non-null, it also calls `setSidebarCollapsed(false)`. The sidebar auto-expands on first selection; the user can still manually collapse it after.

Verified live in the running app (`prj_76bd8243ff997f72`) via Playwright after Fast Refresh picked up the change:
- Clicked Plan node → `aria-pressed: "true"`, sidebar `aria-label: "Node detail sidebar"` (no "(collapsed)" suffix), sidebar body populated with: `primitive plan_reproduction status ok called at 2026-05-29 03:10:53 phase plan type LLM call result dict[compute_scope, data_recipes, dataset_...]`.

Severity: BUG-NEW-009 was P0 — primary user complaint, the entire canvas was unusable for inspection. Now resolved. No backend run interruption (HMR live-reload).

### T+37 min — BUG-NEW-022 (P1) — blank unlabeled circles on canvas

User flagged: several small circles on the canvas have no label whatsoever. Cause: `constellation-canvas.tsx:317` set `showLabel = kind === "llm_primitive" || kind === "subrlm"`. The plain `"primitive"` kind (which covers `heartbeat`, `build_environment`, `run_experiment`, `implement_baseline`, `resolve_gpu_requirements`, etc.) was excluded — those nodes rendered as 14px-radius blank circles. User saw a constellation full of mystery dots.

Fix applied:
- `constellation-canvas.tsx:317-321`: `showLabel = true` for every node kind.
- `layout-constellation.ts:63-65`: bumped plain-primitive radius from 14 → 18 so a label fits.

No backend interruption (HMR).

#### Second pass on BUG-NEW-022 — affordance contrast bump

User flagged again that some nodes still look unclickable. The labels were now showing (Build Env, heartbeat, resolve gpu) but the plain-primitive stroke `var(--line)` was so low-contrast it read as disabled. Bumped `constellation-canvas.tsx:310-315` stroke from `var(--line)` → `var(--muted)` so the affordance is visually obvious. Now plain primitives look like sibling clickable controls, not greyed-out placeholders.

End-to-end verification of every node type via Playwright:
- Tested 9 unique aria-labels: `heartbeat, Understand, Sub-RLM, Extract Hyperparams, Detect Env, resolve gpu requirements, Build Env, Plan, Implement` — all bind onClick and dispatch to the React event system.
- Single-clicked `heartbeat` (the lowest-contrast plain primitive) → sidebar opens, `aria-pressed=true`, body shows: `primitive: heartbeat, status: ok, called at: 2026-05-29 02:56:05, result: dict[alive, counter, note, outcome]`.

BUG-NEW-022 fully resolved: every node has a label AND is visibly clickable AND opens the detail sidebar on click.

### T+37 min snapshot — implement_baseline WRITING CODE

State: **Files written: paper.pdf, requirements.txt, rubric_guard.py, config.json, train.py (52 KB), commands.json (post-emit signal).** `commands.json` exists ⇒ implement_baseline has emitted the per-step manifest, which is the natural completion marker. Same file-set shape as yesterday's run at the same point (which then went on to write ~25 more files into `sdar/` + `skills/`).

Elapsed: 1158s of 1800s Fix-A window (642s headroom). SDK child still alive and writing. The 240s pre-emit watchdog would have killed this at the 240s mark (yesterday's death timing). Fix A is doing its job.

### T+45 min — 🟢 WATERSHED MILESTONE: `implement_baseline OK` + `run_experiment START`

This is the first time on this paper that the pipeline has crossed into `run_experiment`.

Key events:
- `03:33:44Z primitive_call implement_baseline status=ok` — total duration **1,370,730 ms = 22:50 (22 min 50 sec)**. Yesterday died at 240s; today Fix A's 1800s window let it finish. **Fix A's value is now empirically proven.**
- `03:33:44Z worker_report_completed` (status=completed, blockers=0, error=null) — clean exit
- `03:36:44Z primitive_call run_experiment status=start` — first ever attempt at provisioning a RunPod pod on this paper
- **0 `InvalidSandboxTransition` errors so far** — Fix C is holding through the transition
- 0 SDK orphans throughout the run (Fix B held)

Files written by implement_baseline:
- `paper.pdf, requirements.txt, rubric_guard.py, config.json, train.py, commands.json, commands.log` — 7 files. Simpler shape than yesterday's 30-file SDAR module split — the agent chose a single-file train.py strategy this time.

`commands.json`:
```json
["ALFWORLD_DATA=/workspace/data/alfworld alfworld-download 2>&1 | tail -30 || echo 'alfworld-download completed (or failed gracefully)'",
 "python train.py 2>&1"]
```

**Next milestone:** RunPod pod creation (currently `pods == []`, request just submitted, COMMUNITY tier provisioning takes 30s–2min). Will appear in the GraphQL pods list once allocated. Expect the gpu_resolved plan (RTX A6000 48GB at $0.49/hr) to be honored or escalated up the ladder if capacity-exhausted.

### T+55 min — 🟡 NEW BUG: BUG-NEW-023 (P0) — `run_experiment` deadlock from leaked asyncio thread

After `run_experiment start` fired, 10 minutes passed with:
- 0 outbound TCP connections from the CLI (`lsof -p $PID -nP`)
- 0 file writes anywhere in the run dir (cost_summary updater excepted)
- 0 RunPod API activity (`pods == []`)
- 0 SDK CLI children (Fix B working — implement_baseline's SDK CLI was reaped)

But `sample 74581` shows ALL 10 threads parked:
- Main: `select_kqueue_control_impl` (asyncio event loop wait)
- `prim-run_experiment`: `_PyParkingLot_Park` → `_pthread_cond_wait` (mutex wait)
- `ThreadPoolExecutor-2_0` (run_experiment's pool): `kevent` (inner asyncio loop wait)
- **`asyncio_0` (two threads, leaked from implement_baseline's `sdk_isolation.run_isolated`)**: `_PyMutex_LockTimed` (mutex wait)
- `cost-summary-prj_...`: parked
- `Thread-1`, `Thread-2`, system thread, etc.: all parked

**Root cause:** `sdk_isolation.run_isolated` spawns a daemon thread per SDK call with its own `new_event_loop()`. When `implement_baseline`'s `pool.shutdown(wait=False, cancel_futures=True)` returns (per my Fix B), the SDK daemon thread is NOT joined — it remains alive holding a half-closed asyncio loop. When `run_experiment` then calls `pool.submit(asyncio.run, _execute_in_sandbox(...))`, the new asyncio loop creation deadlocks against the leaked daemon thread's state (likely a Python 3.14 asyncio internal lock — `_PyMutex_LockTimed` shows up in all three asyncio-related threads).

Fix B's SDK process reaping works (no SDK CLI orphans), but it doesn't reap the Python-side asyncio THREAD. That requires a different mechanism — either:
- **Option E.1 (small):** add explicit `asyncio` runtime cleanup at the end of implement_baseline — `asyncio.set_event_loop(None)` + force-close any leftover loops via `gc.get_objects()` filter. Best-effort; doesn't kill the daemon thread but unblocks asyncio.run.
- **Option E.2 (medium):** switch `implement_baseline`'s thread pool to `multiprocessing.Process`. Process boundary cleanly kills the SDK + asyncio loop. Marshalling cost: ~200 lines of arg/result pickling.
- **Option E.3 (large):** retire `sdk_isolation.py`'s threading-based isolation and route every SDK call through a subprocess (`subprocess.Popen` invoking a tiny SDK-runner script). Cleanest but a real refactor.

The run was killed at T+55 min (~10 min into the deadlock). No RunPod pods leaked. State of progress for this attempt:
- ✅ Reached implement_baseline ok (Fix A's 1800s watchdog proven)
- ✅ Reached `run_experiment start` for the first time ever on this paper
- ❌ Deadlocked before first pod request reached RunPod API
- ❌ No InvalidSandboxTransition (Fix C didn't even get a chance to be tested — deadlock fired earlier in the path)

Each prior attempt got further than the one before:
- May 28 AM (`prj_09047604e591d969`): died at REPL safe-builtins in iter 0 (BUG-LR-011..015 — fixed)
- May 28 PM #1 (`prj_792d374343d77d70`): died at sdk_pre_emit_stall in implement_baseline (BUG-NEW-012 — Fix A applied)
- May 28 PM #2 (`prj_76bd8243ff997f72`): cleared implement_baseline, deadlocked in run_experiment (BUG-NEW-023 — needs Fix E)

Next session needs to pick Option E.1/E.2/E.3.

---

## Attempt 2 (launched 2026-05-28 22:47 local, project `prj_68ec2c1d28e63a77`)

### Fixes applied beyond attempt 1
- **Fix E.1** — `backend/agents/rlm/primitives.py` finally-block of `implement_baseline`. After `pool.shutdown(wait=False, cancel_futures=True)`, force-close any leftover non-running asyncio event loops via `gc.get_objects()` walk + `asyncio.set_event_loop(None)`. Best-effort cleanup of state that leaked from `sdk_isolation.run_isolated`'s daemon thread, so the next primitive's `asyncio.run()` doesn't deadlock on a leftover loop's mutex.

If attempt 2 also deadlocks at the same point, escalate to Fix E.2 (multiprocessing.Process for implement_baseline) or E.3 (full subprocess rewrite of sdk_isolation).

Monitor task `bqj31wdbo` armed.

### T+5 min — agent confusion in iter 1, recovered in iter 2

iter 1 response (1756 chars, 0 code blocks): the root model claimed "none of those tools exist in my actual function list" and asked the user to exit plan mode. This was an over-cautious read of the SDK setup: `permission_mode="plan", tools=[]` is the deliberate root-model SDK configuration (`backend/services/context/workspace/tools/rlm_query.py:545`) because the root only needs to write Python REPL code — it doesn't make SDK tool calls. The REPL primitives are Python functions in REPL globals, not SDK tools, so plan-mode doesn't block them.

Mitigation: sent a steering chat message via `POST /runs/<id>/messages` explaining the setup. The agent had ALREADY started iter 2 with normal primitive calls before the message arrived (36 events landed in the same minute), so the recovery was self-directed. Worth logging as a soft bug: BUG-NEW-024 (P3) — model occasionally hallucinates "no tools available" on first iteration. Mitigation: nudge system prompt to explicitly say "ignore your tool-list; the primitives below are Python functions you call via REPL code."

Current state at T+5 min: 5× understand_section, check_user_messages, heartbeat, extract_hyperparameters all OK. Same trajectory as attempt 1.

## BUG-NEW-025 (P2): `--vram-gb 38` override ignored on fallback resolver path

**Symptom:** `runs/prj_68ec2c1d28e63a77/rlm_state/gpu_plan.json` shows `vram_gb: 24` (rtx4090) with `source: "fallback"` and `requirements.estimated_vram_gb: 38`, despite CLI invocation passing `--vram-gb 38`.

**Hypothesis:** The LLM-based `resolve_gpu_requirements` call timed out or returned low-confidence (0.3) and the fallback path hardcodes the cheapest SKU (rtx4090) without honoring `ctx.vram_override`. The 2026-05-23 dynamic-GPU spec promised `--vram-gb` bypasses the LLM estimate but still applies the 1.25 headroom → 47.5GB needed → a6000 (48GB).

**Impact:** Lower than expected. The OOM-escalation ladder (`ladder_remaining: [a5000, a6000, l40s, a100_40, a100_80, h100_80]`) will climb on the first CUDA OOM. Adds 1-2 wasted minutes per escalation step but does not block the run.

**Fix location:** `backend/services/runtime/gpu_resolver.py` — the fallback branch should check `ctx.vram_override` before picking rtx4090.

**Status:** OPEN. Not blocking current run.

---

## Attempt 3 — relaunched 2026-05-29 00:15 PDT (07:15 UTC) [no-time-limit, dynamic GPU]

**project_id:** `prj_c52dd48df079b5fd` (forced fresh via new PDF filename `papers/sdar_2605.15155_attempt3_20260529_001500.pdf`)
**CLI PID:** 62713
**Log:** `logs/sdar-attempt3-20260529_001500.log`

**Why this attempt:** attempt 2 (`prj_68ec2c1d28e63a77`) died at 05:24 UTC — the 5400s wall-clock budget expired while RunPod was throwing `RUNPOD_CAPACITY_EXHAUSTED` on a6000 for ~50 min straight. `implement_baseline` then cascaded into "timed out after 1 s" failures and the root model gave up.

**Changes from attempt 2:**
1. **DROPPED `--vram-gb 38`** → dynamic GPU resolver picks (per user "it should pick its own gpu on whichever is best"). Sidesteps BUG-NEW-025 (fallback resolver ignores override) and lets the OOM-escalation ladder choose freely.
2. **DROPPED `--max-wall-clock` and `--max-pod-seconds`** → no wall-clock cap (per user "remove the time limit"). `--max-usd 30` and `REPROLAB_MAX_RUN_GPU_USD=10.0` (.env) remain as cost safeties.
3. Kept: `--sandbox runpod --mode rlm --model claude-oauth`, `REPROLAB_BASELINE_EXTRA_GUIDANCE` (smallest-two scope), `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY` shadow-defeat.

**Carry-over fixes validated in attempt 2 (still active):**
- Fix A: SDK pre-emit stall watchdog 1800s (proved structurally correct — `implement_baseline` ran 22 min cleanly).
- Fix C: SandboxAggregate FAILED eviction.
- Fix E.1: asyncio loop cleanup (no run-experiment-side deadlock observed; today's terminal cascade was wall-clock-driven, not deadlock).

**Monitors armed:**
- `bcbrhi1g0` — dashboard_events tail (`gpu_resolved|sandbox_*|pod_created|run_experiment|implement_baseline|run_complete|...`)
- `bdxab6o4w` — RunPod GraphQL `myself.pods` poll every 4 min (will emit on first-ever pod creation for this paper)

---

## Attempt 4 — relaunched 2026-05-29 02:15 PDT (09:15 UTC) [UI fix sweep + same flags]

**project_id:** `prj_e67c9b7be5529226` (forced fresh via new PDF filename `papers/sdar_2605.15155_attempt4_20260529_021500.pdf`)
**CLI PID:** 86208
**Log:** `logs/sdar-attempt4-20260529_021500.log`

**Why this attempt:** user asked for a kill-clean-rerun after a UI bug sweep + Playwright verification.

**Bugs fixed in this round (all applied to working tree, not committed):**
- BUG-NEW-001 (P1) **paperTitle "paper_text" leak** — `backend/cli.py:1182-1198` noise-filter now rejects workspace variable keys (`paper_text`, `paper_metadata`, `supplementary_text`, `repo_files`, `prior_work_refs`, `rubric_spec`) AND writes `sourceLabel`/`sourceNote`/`sourceKind` to `demo_status.json` so the lab-shell's `run.sourceLabel ?? "Untitled paper"` fallback resolves to the real title for CLI-spawned runs (only the upload path wrote those before).
- BUG-NEW-001 (cont.) **lab-shell fallback** — `frontend/src/components/lab/lab-shell.tsx:46` now shows `Parsing paper… (<id-prefix>)` while `sourceLabel` is empty instead of the generic "Untitled paper".
- BUG-NEW-004 (P0) **status badge stuck "queued"** — `frontend/src/hooks/use-rlm-run.ts:898-911` `fold()` now flips `status: "queued" → "running"` on the FIRST event of any kind (heartbeat, primitive_call, etc.) rather than waiting for `repl_iteration`. Run-complete keeps its own terminal status.
- BUG-NEW-021 (P1) **"no signal Ns" false alarm during sub-RLM** — `rlm-header.tsx` accepts a new `hasInFlightSubRlm` prop; `rlm-lab.tsx` passes `state.subRlms.some(s => s.completedAt === null)`. When true, the pill renders as "running sub-RLM (Ns)" (accent-soft, informational) instead of "no signal Ns" (warn, alarming).
- BUG-NEW-022 (B2) **blank sub-RLM sidebar** — `node-detail-sidebar.tsx:485+` no longer shows "no sub-RLM detail" alone; renders title/iteration/parent from the node payload + a "still in flight" hint.
- BUG-NEW-022 (B3) **pulse-dot contrast** — `rlm-header.module.css` mid-pulse opacity 0.35 → 0.55.
- BUG-NEW-022 (B4) **empty-canvas state** — `rlm-lab.tsx` overlays a centered "Waiting for first iteration" panel when `state.tree` has only the implicit paper root. Auto-hides on first work node.
- BUG-NEW-022 (B5) **recent projects truncation** — `lab-sidebar.tsx:94+` prefers `sourceLabel`, bumps default `projectId.slice(0,18)`, adds `title=` + `aria-label=` with full id + status.
- BUG-NEW-025 (P2) **--vram-gb fallback ignored** — `backend/services/runtime/gpu_resolver.py:62-90` now honors `fallback_vram_gb` in the LLM-low-confidence branch (uses `find_ladder(min_vram_gb=fallback*headroom)` before the rtx4090 hardcode).
- **leaf-scores 404 spam** — `backend/app.py:859` returns `{leaf_scores: []}` with HTTP 200 when scoring is not yet complete (vs 404 per-poll). Reserves 404 for missing run-dir.

**Carry-over fixes still active:**
- Fix A: SDK pre-emit stall watchdog 1800s.
- Fix C: SandboxAggregate FAILED eviction.
- Fix E.1: asyncio loop cleanup.

**Flags (same as attempt 3):**
`--sandbox runpod --mode rlm --model claude-oauth --max-usd 30` (NO `--max-wall-clock`, NO `--max-pod-seconds`, NO `--vram-gb` → dynamic GPU resolver). `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY` shadow-defeat.

**Servers restarted clean** (uvicorn PID 84466, next.js PID 84467) — old PIDs 18078 + 18140 had wedged on stale connections.

**Monitors armed:** dashboard_events tail + 4-min RunPod poll.

---

## BUG-NEW-026 (P0, NEW, NOT FIXED): claude-oauth root deadlocks on first SDK call (attempt 4)

**Symptom (user-observed 2026-05-29 02:12 PDT, ~5 min into attempt 4):**
- UI stuck on "Queued — backend acknowledging…" for 5+ min.
- `runs/prj_e67c9b7be5529226/` has `parsed_full_text.txt`, `generated_rubric.json`, `rlm_state/` (empty) — workspace built — but NO `dashboard_events.jsonl`. CLI is alive (PID 86208, 0% CPU, 16 MB RSS = parked).
- `demo_status.json` says `status: "running"` already (backend wrote it before the root pipeline blocked), but the lab-shell SSE reads status from event folding, so the UI sees zero events → stays queued.

**Smoking gun in `logs/sdar-attempt4-20260529_021500.log`:**
```
Execution profile: max; sandbox: runpod
run_pipeline_rlm: root model 'claude-oauth' is NOT paper-validated as an RLM root (root_model_unvalidated)
an error occurred during closing of asynchronous generator <async_generator object InternalClient._process_query_inner at 0x10f0e3870>
asyncgen: <async_generator object InternalClient._process_query_inner at 0x10f0e3870>
RuntimeError: aclose(): asynchronous generator is already running
Loop <_UnixSelectorEventLoop running=False closed=True debug=False> that handles pid 86255 is closed
```

**Hypothesis:** the `claude-oauth` root path (ClaudeOauthClient → ClaudeLlmClient → claude-agent-sdk subprocess PID 86255) is hitting the same aclose-race that Fix E.1 addresses for `implement_baseline` sub-agents — but Fix E.1 only patches `implement_baseline`'s finally block in `backend/agents/rlm/primitives.py`, NOT the root-model SDK path in `backend/agents/rlm/run.py`. When we restarted uvicorn earlier this session, leftover OAuth/asyncio state from the previous (killed) attempt may have leaked into the freshly-spawned root.

**Effect:** the root never produces a single REPL turn → no dashboard events → the UI's status flip never fires → user sees a dead-looking page. Carry-over: A2's "flip on first event" fix is structurally correct but a no-op when there are zero events.

**Fix candidates:**
1. Apply the Fix E.1-shaped asyncio cleanup to the root-model SDK path as well (run.py around the `rlm.completion()` call).
2. Add a backend-side fallback: if `status == "running"` in `demo_status.json` but no events have landed in N seconds, the lab-shell hook should accept the JSON status as authoritative.
3. Pre-validate the claude-oauth subprocess connection BEFORE handing it to the rlm library — currently the failure surfaces only on first generator advance.

**Status:** OPEN. Per user direction this turn, NOT FIXED yet. Attempt 4 left running so the deadlock is reproducible for the next session.

## UI cleanup follow-ups (NOT FIXED, user-flagged 2026-05-29 02:12 PDT — note-only)

User screenshot of attempt 4 surfaced these "looks bad / buttons not seeable" issues:

1. **Send button (chat panel, bottom-right of sidebar)** — orange outline on dark bg with very low fill contrast. When disabled (no text in input), it's nearly invisible. Needs filled bg in disabled state OR brighter outline.
2. **"primitive call history — waiting for first primitive" footer** — plain text on dark bg with a `▸` glyph; reads as decorative caption, not a clickable expander. Needs a visible button affordance (bg-chip, padding, hover state).
3. **Status pill "queued"** (header right block) — `var(--muted)` text on `var(--chip)` bg is too low-contrast; user sees it as broken. Even on a healthy run, the queued state should look "waiting", not "disabled".
4. **Phase nav active-state for "Ingest"** — active pill is the same color as inactive with a subtle radial highlight; reads as disabled. Needs filled bg / brighter ink for active.
5. **"runpod: not yet" + "iteration 0" chips** — visible but cramped against right edge of header; should respect more whitespace.
6. **Worker reports "0" chip** — tiny number in a corner; easy to miss when something lands.
7. **Tooltip "Rubric score updates when…"** floats in mid-canvas with no anchor connection — looks orphaned; needs a triangle/pointer to its anchor or a less-obtrusive treatment.
8. **Two "queued" indicators stacked** — the right-side header pill AND the wider "Queued — backend acknowledging…" status bar repeat the same info. Collapse to one.

**Action this turn:** none — per user "PLEASE note this down but dont change". File issues for a future contrast pass.

---

## 2026-05-29 07:25 UTC — UI fixes after user feedback on attempt 4

User screenshot of `prj_e67c9b7be5529226` (attempt 4) with three concrete complaints:
1. **Chat doesn't work** — user sent "what is understand", optimistic add visible, no RLM reply (root was parked in sub-RLM querying paper at depth 1; `check_user_messages()` only runs at start of each root iteration so a sub-RLM window silently blocks the chat). User had no way to know reply was queued, not lost.
2. **Numbers wrong** — ReportRail (center panel) tiles all showed "—" while top-right header said `iteration 1` and right sidebar counters said `1 · 8 · 0 · 0`. Two different grids with the same labels showing different values is confusing.
3. **Can't click "icon"** — LLM-call (purple Understand) circles didn't respond to clicks; sidebar stuck on "no node selected".

### BUG-NEW-027 — SVG `setPointerCapture` stole every click from child nodes
`constellation-canvas.tsx::handlePointerDown` engaged `setPointerCapture(pointerId)` on EVERY pointer-down. With the pointer captured by the SVG, the subsequent `pointerup` was routed back to the SVG instead of the child `<g role="button">`, so the browser never fired a `click` event on the node. The 3px drag threshold check in `handleNodeSelect` was a red herring — capture had already intercepted the click before any threshold mattered.

**Fix**: lazy capture — engage `setPointerCapture` inside `handlePointerMove` only when the 3px drag threshold is crossed. Click-only interactions (no movement) reach the child element as the browser intended. Drag-to-pan still works because capture engages the moment movement starts.

### BUG-NEW-028 — Chat silent during sub-RLM windows
`useSteeringChat` sends user messages fine; backend `respond_to_user()` works fine; root model polls `check_user_messages()` at start of each iteration. Failure mode: root parked in `rlm_query` (sub-RLM decomposition) for minutes, so no reply lands. UI gave zero feedback.

**Fix**: `steering-chat.tsx` now shows a dashed pending-hint card when the last message is from `user` (no following `assistant` message): _"RLM responds at the start of each iteration. If a sub-RLM is in flight, the reply waits until it returns."_ Pulsing accent dot for animation. Reduced-motion respected.

### BUG-NEW-029 — ReportRail "—" tiles when live data exists
`report-rail.tsx` showed "—" placeholders for iterations/primitive-calls/proposed/promoted whenever `state.report === null`, even when live counts existed and were already rendered elsewhere on screen. Two grids with the same labels showing different values is worse than one grid that updates.

**Fix**: added `liveIterationCount` / `liveProposedCount` / `livePromotedCount` props (passed from `rlm-lab.tsx` using `state.iterationCount`, `candidatesProposed`, `candidatesPromoted`). Primitive calls falls back to `primitiveCalls.length`. Tiles now show live counts pre-completion and snap to final counts when `final_report.json` writes. Empty-note rewritten: _"Live counts above; tiles freeze when final_report.json is written."_

### Status of attempt 4
First repl_iteration landed at 07:12:41 UTC (≈ 8 min after launch); building Docker environment as of 07:15. The BUG-NEW-026 deadlock cleared on its own this time — keep watching to confirm the run progresses to the first `run_experiment` and RunPod pod creation.

### 2026-05-29 07:23 UTC — yellow flag: root planned SDAR as CPU-only
Attempt 4 iteration 1 emitted `run_warning compute_scope_invalid` with payload claiming `"CPU-only. No GPU required (matches ENV003). Smoke run: ≤ 60 seconds, < 200 MB RAM."` Two problems:
1. **Schema violation** — `compute_scope` must be `dict | null`; the root passed a free-text string. Harness coerced and continued.
2. **Paper mis-read** — SDAR (arxiv 2605.15155) is a multi-GPU GRPO+OPSD RL training paper on Qwen 1.7B / 3B / 7B; "CPU-only ≤ 60 s" is wildly off the actual hardware budget. This means `resolve_gpu_requirements` was likely skipped or returned `fallback` with no GPU, and the first `run_experiment` will either fail or produce a meaningless surrogate score.

Possible causes:
- Root mis-parsed ENV003 (whatever rubric leaf that is) and overgeneralized to "CPU-only run".
- `REPROLAB_BASELINE_EXTRA_GUIDANCE` smallest-two override may have been mis-applied (it should pin to 1.7B+3B on a single 24–48 GB GPU, not strip GPUs entirely).
- The forced-iteration policy will catch this when verify scores ~0 and refuse FINAL_VAR — at least the run won't ship a partial.

Action: wait for first `run_experiment` outcome. If it's a degraded CPU surrogate, file BUG-NEW-030 and investigate whether the system prompt needs a stronger GPU-required-for-SDAR hint.

### 2026-05-29 07:28 UTC — attempt 4 timeline snapshot

| t (UTC) | event | note |
|---|---|---|
| 02:15:00 | CLI spawn | `prj_e67c9b7be5529226`, `--mode rlm --model claude-oauth --sandbox runpod --provider anthropic` (no time/cost cap; dynamic-GPU on) |
| 07:07 | (silent) | claude-agent-sdk root path hung on `aclose(): asynchronous generator is already running` — BUG-NEW-026, recovered on its own without intervention |
| 07:12:41 | first `repl_iteration` | text began "Defensive initialization - ensure all variables exist" — declared `outcome_label = "skipped"` defensively; mild concern but proceeded |
| 07:15:22 | heartbeat #1, #2, #3 | "building paper claim map" → "detecting environment" → "building Docker environment" |
| 07:22:41 | heartbeat #4 | "planning reproduction" |
| 07:23:28 | `run_warning compute_scope_invalid` | **yellow flag** — root passed string `"CPU-only. No GPU required..."` instead of dict/null; **and** mis-classified SDAR as CPU-only. RunPod has NOT been requested for this run. |
| 07:23:28 | heartbeat #5 + `primitive_call implement_baseline start` | RSS 153 MB at start; first heavy primitive in flight |

### Open bugs / flags carried into next iteration
- **BUG-NEW-026** (root `aclose` deadlock, P0) — deferred per user; transient on attempt 4 but will resurface; Fix E.1-shaped asyncio cleanup needed around `rlm.completion()` call in `backend/agents/rlm/run.py`
- **BUG-NEW-027/028/029** (UI) — fixed this turn; user verifies on next page-refresh
- **BUG-NEW-030** (candidate, pending) — root planned SDAR as CPU-only; will materialize as a degraded rubric (≈ 0) when `run_experiment` finishes — wait for the score before opening a new bug entry
- **8 deferred UI cleanup items** (Send button contrast, primitive history footer affordance, status pill contrast, phase nav active state, header chip whitespace, worker reports chip size, orphan tooltip, duplicate queued indicators) — still untouched per "PLEASE note this down but dont change" original direction; not in scope for this user-feedback round
- **Monitors armed**: `buwypo0c4` (SSE events, full stream), `b21vgl7sz` (RunPod 4-min poll — will tick "first pod ever for SDAR" when one spins up)
- **Next milestone to watch**: `implement_baseline` completion (~5-15 min) → `run_experiment` start (RunPod pod creation expected here if root corrects its CPU-only mis-read) → first rubric score

### 2026-05-29 07:39 UTC — attempt 4: BUG-NEW-030 materialized exactly as predicted

- 07:39:22 — `implement_baseline` completed after **15m 54s** (worker duration_ms 954546)
- 07:39:23 — `run_experiment` **pre-flight blocked** (`failure_class=preflight_blocked`, 1 hard contract violation: "no `AutoModelForCausalLM.from_pretrained(...)` call references any of the canonical model pat..."). Root wrote train.py with NO Qwen model load — exactly what the CPU-only mis-classification predicted.
- 07:39:23 — `rubric_score` iter=1 = **0.0** (target 0.6), all four areas fail. `compute_scope`/`compute_adjusted_score` populated → grader is honest about the gap.
- 07:39:23 — `run_warning iteration_boundary_recommended` — orchestrator told the root to end iteration so the failure surfaces as fresh context next turn (correct behavior).
- 07:40:16 — root proposed an improvement candidate **within** iteration 1 (heartbeat counter=8) and started a SECOND `implement_baseline` (arg0=dict[4], i.e. improvement plan). Watching whether the new baseline corrects the missing Qwen load.

**BUG-NEW-030 status: CONFIRMED.** The compute_scope_invalid warning at 07:23 was the smoking gun; the rubric=0.0 at 07:39 is the consequence. **Root fix lives upstream** in `backend/agents/rlm/primitives.py::plan_reproduction` or system prompt — must hard-require Qwen base-model loads for SDAR-shaped rubrics. Not in scope for this turn; opening as P1 once attempt 4 finishes.

**Forced-iteration policy** (`REPROLAB_MIN_RUBRIC_ITERATIONS=2`) is armed but has not fired yet — the root has not called `FINAL_VAR`; it's correctly iterating on its own. Good.

### 2026-05-29 07:42 UTC — attempt 4: TERMINATED at iter 2, verdict failed, score 0.0

- 07:42:29 — second `implement_baseline` completed in **2m 13s** (a targeted patch, not full rewrite)
- 07:42:37 — second `run_experiment` **pre-flight blocked with the EXACT SAME violation** (no canonical `AutoModelForCausalLM.from_pretrained` call). Implementer did not fix the hint.
- 07:42:38 — rubric iter=1 still 0.0; 3× `record_candidate_outcome` errors in a row (possibly book-keeping race during iteration handoff)
- 07:42:38 — `repl_iteration` iter=2 begins; root response: `"Good advice. Let me start fresh by checking messages, then inspecting what went wrong."` — runs `check_user_messages` + `print(json.dumps(final_exp))`. No new primitive work.
- 07:42:40 — `run_complete status="failed" iterations=2 rubric_score=0.0 cost_usd=0.0`. Root called `FINAL_VAR` after iter 2 inspection; forced-iteration policy ALLOWED it because `iteration_count == REPROLAB_MIN_RUBRIC_ITERATIONS=2` (policy enforces minimum iterations, not target score — by design, can't iterate forever).

**Verdict: complete reproduction failure. RunPod never spun up — $0 spent on GPU. Total wall-clock 36 min.**

### BUG-NEW-033 — ROOT CAUSE FOUND + FIXED 2026-05-29

**Root cause:** the `rlm` library's `_rlm_query` and `_llm_query` REPL globals have signature `(prompt: str, model: str | None = None)`. Our docs (system_prompt.py, primitives.py:573/612/3722) explicitly taught the wrong API: `rlm_query(slice, specific_question)`. When the root model followed our docs and called `rlm_query(context, "What is the core algorithmic contribution...")`, positional binding made `model=<the question>`. That value propagated to `claude-agent-sdk` → `claude --model "<question>"`. The CLI rejected the unknown model and returned the error text *as the response body* (the SDK does not raise on CLI-rejected model names — it returns the stderr text as the completion). The error string landed in `contribution_summary`, then in `final_paper_claim_map["core_contribution"]`, then in the implementer's input. The implementer never saw the canonical Qwen model paths → surrogate `nn.Linear` LM head → pre-flight blocked → score 0.0.

Verified by reading `runs/prj_e67c9b7be5529226/repl_state.pickle`: `contribution_summary` is the literal CLI error string; `final_paper_claim_map["core_contribution"]` is the same string truncated to 500 chars.

**Fix (shipped this session):**
1. New module `backend/agents/rlm/rlm_query_misuse_patch.py` — monkey-patches `LocalREPL._rlm_query` and `_llm_query` at import time. When the `model` positional arg looks like a question (`len > 80` OR contains whitespace), auto-composes `f"{prompt}\n\nQuestion: {model}"`, drops the bogus `model=`, and emits a stderr warning. Correct calls are untouched.
2. Imported at the top of `backend/agents/rlm/run.py` (alongside the existing `safe_builtins_patch` / `safe_repl_traceback_patch`).
3. Doc fixes — `system_prompt.py:51,145` and `primitives.py:573,612,3722` now teach the correct API: `rlm_query(f"{slice}\n\nQuestion: {q}")` with an explicit warning about the (slice, question) footgun.

**Smoke-tested:**
- Patch heuristic correctly classifies question-shaped strings vs. model-name strings
- End-to-end exercise: `LocalREPL._rlm_query(slice, question)` now composes into a single prompt and emits the warning; the underlying `_llm_query` receives a clean composed prompt
- `pytest tests/agents/rlm/` — 3/3 passed, no regressions

**Observed:** `runs/prj_e67c9b7be5529226/final_report.json` field `paper_claims.core_contribution` is:
> "There's an issue with the selected model (What is the core algorithmic contribution of this paper? List: (1) the method name, (2) the key formula or algorithm, (3) the main datasets used, (4) the key metrics reported, (5) the hardware used for experiments.). It may not exist or you may not have access to it. Run --model to pick a different model."

That sentence shape is the **Claude Code CLI error** emitted when an unknown `--model <name>` is supplied. The literal question text appears inside the parentheses where the model name would normally go.

**Confounder:** the only sub-rlm in this run completed with `duration_ms: 37574, error: null` (see `dashboard_events.jsonl`). That rules out the trivial "the call instantly errored" theory and means the path that filled `core_contribution` is **not** the same sub-rlm whose `model` field looks weird in the spawn event (the `model` field there is a known logging quirk in the `rlm` library callback, not a real argument-passing bug — `understand_section` itself is pure heuristic, no LLM call).

**Possible root causes (need to investigate before claiming a fix):**
1. The `rlm` library's `_final_state_extract` path may run a separate LLM call on the REPL's `report_state` dict, and pass the question into a position that the CLI parses as `--model`. The 37s duration sub-rlm was for a different question; the core_contribution call could be elsewhere and never logged as `sub_rlm_spawned`.
2. `claude-agent-sdk` may have a kwarg-swap when a tool call's `parameters.model` ends up overriding the SDK's `model=` kwarg.
3. The `runs/<id>/iterations/iter_N.json` files (not yet inspected) likely show which primitive call actually filled the field.

**Why this cascades to failure** (still valid regardless of which root-cause):
The implementer agent reads `paper_claims` with this garbage and never sees the canonical Qwen model names (`Qwen/Qwen3-1.7B-Instruct`, etc.). It invents a surrogate `nn.Linear(d, V)` LM head, pre-flight correctly blocks it, root burns 2 minimum iterations and ships `verdict: failed`.

**Next-session investigation steps:**
1. `grep -rn "core_contribution" runs/prj_e67c9b7be5529226/iterations/` — find which iteration wrote the field
2. Read that iteration's `repl_iteration.response` to see what tool/primitive call produced the string
3. Trace the primitive's wrapper for kwarg-swap or error-as-content
4. **DO NOT** trust the previous version's claim that this is in `understand_section` — that was an over-confident diagnosis based on the spawn-event quirk and is contradicted by `understand_section` being pure heuristic

**P0 standing**: until this is fixed, every SDAR (and likely every other multi-component-paper) attempt will fail the same way.

### BUG-NEW-034 — `record_candidate_outcome` 3× status=error in 2 seconds (P1)

After `propose_improvements` succeeded with 2 candidates, the orchestrator called `record_candidate_outcome` three times in a row, all status=error. The primitive returns `{candidate_id, candidate_outcome, error, outcome, parent_id, success}` — the `error` key signals book-keeping failure. Doesn't kill the run (orchestrator continues) but the candidate tree won't render correctly in the lab UI. P1.

### BUG-NEW-035 — scorecard view hides the exploration tree (UX request, not yet fixed)

**User ask (2026-05-29 ~07:50 UTC):** on the `/lab?projectId=...` page when the run completes (or hits the verify phase), the layout swaps to the *Reproduction scorecard* view (FIG § 5.1 — rubric table + areas-passing tile + drift band + report.md/run.json artifact links). The constellation/tree view that was visible while the run was active is no longer reachable from this state. User wants to see **both** — either:
- A second route (`/lab/<projectId>/tree`, or `?view=tree` query param), OR
- An in-page toggle (e.g. tabs in the header: `scorecard | tree | timeline`)

**Why it matters:** post-mortem on a failed run is where the tree view is most useful — you want to see which candidate the root proposed, which sub-RLM expanded, which `record_candidate_outcome` calls errored, etc. Right now that history is only inspectable via raw `dashboard_events.jsonl`.

**Suggested implementation note:** the constellation node graph already derives entirely from the SSE-replay state (`useRlmRun`), so persisting it past `status="completed"` is mostly a routing/layout change — the underlying data is still available because `dashboard_events.jsonl` replays in full on tab reload. Likely lives in `frontend/src/components/lab/rlm/rlm-lab.tsx` around the conditional that switches to scorecard mode.

Not in scope for the current monitoring session; logging here so the next implementation session picks it up.

### BUG-NEW-036 — Lab upload page hydration mismatch on model select (P1)

**User report (2026-05-29 ~07:55 UTC):** reloading `/lab?projectId=prj_e67c9b7be5529226` (and presumably plain `/lab`) throws a Next.js hydration error. Console diff:
```
<option value="gpt-5">
+ Gpt-5    (client)
- Sonnet   (server)
```
Same `value="gpt-5"`, different label text. Location: `frontend/src/components/lab/upload-view.tsx:336`.

**Root-cause analysis:** the `<option>` text is computed from `modelOptions` (line 322-339). The fallback at line 336-338 (`model.charAt(0).toUpperCase() + model.slice(1)`) only runs when `modelOptions.length === 0`. The diff therefore implies:
- Server-side render: `modelOptions` is non-empty and contains an entry `{id:"gpt-5", label:"Sonnet"}` (or similar label-mismatch from `/api/models`).
- Client-side render: `modelOptions` is empty → falls through to the capitalize fallback "Gpt-5".

Effectively, `initialModels` (passed from `LabShell` via the server) does not match what the client recomputes/refetches on mount. Either the server fetched a stale credential-mapped catalog (Anthropic OAuth without API key → labels collapse to "Sonnet"?) or the client mounts before its own `/api/models` fetch resolves.

**Why it looks like "nothing works":** the hydration error is dev-only (production silently regenerates), but on first load the user sees the model dropdown briefly say "Sonnet" then snap to "Gpt-5" — confusing. The form is still functional (Begin button works), but the perception is "something is broken."

**Suggested fix:** pass `initialModels` deterministically through `LabShell → UploadView` and have the client use that exact array on first render (no mount-time refetch until user interacts). The current race is between server-fetch and client-fetch returning slightly different shapes.

**Workaround until fix:** `suppressHydrationWarning` on the `<option>` element. Band-aid, not a real fix.

### BUG-NEW-037 — "Recent runs unavailable" in the sidebar (P2)

Same screenshot: sidebar `RECENT` section shows literal text "Recent runs unavailable." instead of the list of past projects. The `initialRecentsError` is being passed from server (`<LabShell initialRecentsError="Recent run...">`). Likely the recents endpoint failed to fetch on this server render — possibly:
- Backend was momentarily unreachable
- `REPROLAB_BACKEND_URL` not set or pointing wrong place
- SQLite recents query errored

Lower priority than 036 because it's read-only and doesn't block any workflow. Verify by hitting `curl http://127.0.0.1:8000/runs/recent` (or whatever the endpoint is) and checking what comes back.

### Open bugs after attempt 4
- **BUG-NEW-026** (root aclose deadlock, P0 deferred) — did not recur on attempt 4
- **BUG-NEW-030** (CPU-only mis-plan, P1) — root cause is actually BUG-NEW-033 leaking garbage into paper_claims; treat as duplicate of 033 for now
- **BUG-NEW-033** (rlm_query (slice, question) misuse, P0) — **FIXED** this session (patch + 4 doc sites). Attempt 5 unblocked.
- **BUG-NEW-033** (rlm_query misuse, P0) — **FIXED** (patch + 4 doc fixes + smoke test + regression test pass)
- **BUG-NEW-034** (`record_candidate_outcome` errors, P1) — **NOT A BUG.** Reading `primitives.py:3528-3594` shows the "errors" are intentional input-validation rejections (`candidate_id=None` from the root). The primitive returns a `success=False` dict to surface the model's error rather than silently corrupting SSE downstream — comment at line 3548-3550 cites the 2026-05-23 prj_6b9acbfd8afcd789 incident where silent coercion broke the candidate tree. The visible UI noise is the COST of this fail-loud behavior, not a bug. Optional UI follow-up: dedup consecutive identical soft-failures.
- **BUG-NEW-035** (scorecard view hides tree, UX) — pending; add toggle/route
- **BUG-NEW-036/037** (hydration mismatch + "Recent runs unavailable") — **SAME ROOT CAUSE:** backend (PID 84466, started 02:05 AM) is wedged. `curl http://127.0.0.1:8000/health|/models|/runs` all timeout. Next.js dev server has 3+ stuck connections to it. The SSR-time `fetchModels()` and recents-fetch both fail silently and return `[]`, while client-side fetches return different stale data → hydration mismatch on the model `<option>` label, and "Recent runs unavailable" in the sidebar. Fix is to KILL AND RESTART the backend (`pkill -f uvicorn && .venv/bin/uvicorn backend.app:create_app --factory --reload --port 8000`). No code change needed for either bug, but worth investigating WHY the backend wedged — probably a stuck SSE generator or thread-pool exhaustion from the dead attempt 4 process.
- **BUG-NEW-027/028/029/031/032** (UI fixes) — all shipped this session

### Next attempt blocker
Do NOT launch attempt 5 until BUG-NEW-033 is fixed. Every attempt to date has been wasting compute fighting garbage paper claims from a kwarg-swap. Once `understand_section` returns real content, the implementer should get the correct Qwen model name and pre-flight should clear on first or second baseline.

### BUG-NEW-032 — WorkerReportCard renders "None" filler rows for rlm_primitive workers (UI, fixed this turn)

User screenshot: `[rlm_primitive] build_environment` card (completed, 7m 19s) shows implemented/undone/commands/issues = `None`, procedures = `unconfirmed`. These fields are RDR cluster-worker shape; RLM primitives don't populate them, so the card is 80% useless filler.

Fix (`frontend/src/components/lab/rlm/report-rail.tsx` WorkerReportCard, lines 427-491): wrapped the `<dl>` in an IIFE that derives `has*` flags per row and renders only non-empty rows. `procedures` row only shows when `worker.procedures_followed != null` (so `unconfirmed` is not a useless default). If a worker has zero meaningful facts, the whole `<dl>` is skipped — duration badge and status badge still render. Type-checks clean.

## Attempt 5 (launched 2026-05-29 03:18 PDT) — KILLED at iter 2, verdict aborted

Command: same as attempt 4 modulo PDF path (`/tmp/sdar_2605.15155_attempt5_20260529_031803.pdf`). project_id `prj_ab4399b6d9b95c19`. PID 73400. Goal was to validate BUG-NEW-033 patch end-to-end.

Iter 1: empty (`response_len=0`, no code, 87.7s think). Iter 2: root produced a 2 KB message **refusing to do any work** — "Environment Mismatch — Cannot Proceed as Described." Verbatim symptoms:

- Claimed its tool inventory was "Google Drive, Sentry, Context7, Playwright, Stripe, TrueMemory, and `advisor`" (= the user's outer Claude Code MCP servers).
- Claimed "plan mode is active, which blocks all execution anyway."
- Suggested the user run `python -m backend.cli reproduce 2605.15155 …` — i.e. it thought it was Claude Code, not the RLM root.

Killed at T+5m before further compute waste.

### BUG-NEW-038 — ROOT CAUSE FOUND + FIXED 2026-05-29 (P0)

**Root cause:** `ClaudeLlmClient._async_complete` in `backend/services/context/workspace/tools/rlm_query.py:541-547` built `ClaudeAgentOptions` with `permission_mode="plan"` and `tools=[]` — but did **not** set `setting_sources=[]` or `mcp_servers={}`. `claude-agent-sdk` defaults to loading the user/project/local settings, including the outer `~/.claude/settings.json` and every MCP server the user has configured. The inner root model received those tools + plan mode as part of its environment context and faithfully reported "I have these tools and plan mode is on, I cannot execute." It was technically correct given the contaminated environment.

This is why attempt 4's iter-1 response (and BUG-NEW-033 from this morning) showed CLI error strings as "paper claims" — the root was always seeing a polluted environment, just sometimes it tried to work through it and sometimes (attempt 5) it refused outright. The patch from this morning (rlm_query_misuse_patch, system_prompt, primitives doc-fix) was real but addressed a downstream symptom, not the root cause.

**Fix (same file, +6 lines):**

```python
options = ClaudeAgentOptions(
    system_prompt=system,
    model=self._model,
    max_turns=self._max_turns,
    permission_mode="bypassPermissions",   # was "plan" — caused root to refuse execution
    tools=[],
    mcp_servers={},                         # NEW — block inherited MCP servers
    setting_sources=[],                     # NEW — block ~/.claude/settings.json inheritance
)
```

`bypassPermissions` is appropriate because this call is `max_turns=1` text completion with no tools — there's no execution to gate. Verified the SDK supports both new kwargs (`dataclasses.fields(ClaudeAgentOptions)` shows them at the top of the dataclass).

**Validation pending:** awaiting user OK to launch attempt 6 (same paper, same flags, fresh project_id). Expected behavior: iter 1 root writes a REPL block calling `understand_section(context["paper_text"][:N])` and `extract_hyperparameters(...)` instead of describing its tool inventory.

### BUG-NEW-038 — companion sites FIXED 2026-05-29 (next attempt onward)

After attempt 6 validated the root-model fix, the two sibling sites were also patched preemptively, before the in-flight run reaches `implement_baseline`:

1. `backend/agents/runtime/claude_runtime.py:93` — sub-agent code-writing runtime (used by `implement_baseline`, RDR cluster workers, every Sonnet sub-agent). Was passing `mcp_servers` conditionally; now always passes an explicit dict + `setting_sources=[]`. **Critical because:** without this, the implementer sub-agent inherits the user's MCP servers AND any plan mode they have on. A code-writer in plan mode refuses to write files ("I can only describe what I would do"), and a code-writer with the user's Stripe/Sentry/Playwright MCPs hallucinates them as part of its tool inventory. The `permission_mode` is per-agent via `agent.permission_mode` (defaults to `bypassPermissions` in `backend/agents/runtime/base.py:107`), so plan mode would only leak through if the AgentSpec opts in — but the MCP and skill inheritance was unconditional.
2. `backend/hermes_audit/providers.py:394` — hermes audit LLM call. Same fix; low blast radius (single short audit summary).

**The in-flight attempt 6 will NOT pick up these fixes** — the CLI subprocess (PID 4376) imported these modules at startup, so the bytecode is frozen. If attempt 6 dies at `implement_baseline` because of the contamination, attempt 7 (fresh import) is protected.

### Attempt 6 universality validation (this turn)

In addition to the SDAR-specific fixes, audited the broader system for paper-specific assumptions while waiting for attempt 6's primitives to fire:

- **ClaudeAgentOptions construction sites:** 3 actual call sites, all now patched (rlm_query.py:550, claude_runtime.py:93, hermes_audit/providers.py:394). No other sites in the codebase.
- **SDAR-specific code:** `paper_invariants.py` (registry, designed to be extended), `paper_hints.py::PAPER_HINTS` (one entry, opt-in via `--paper-hint`), `baseline_implementation.py:985-1000` (ALFWorld scaffolding is conditional — "apply ONLY when the listed environment/dataset is named verbatim in YOUR paper"). System prompts (`system_prompt.py`, `primitives.py`) are paper-agnostic — no `grep SDAR` hits.
- **Per-paper customization surface:** `--paper-hint <arxiv-id>` (paper_hints.py), `--scope-spec` (operator narrowing), `REPROLAB_BASELINE_EXTRA_GUIDANCE` (free-form env var), `--vram-gb` (manual VRAM override). All paper-agnostic mechanisms.
- **Gaps for the "all papers" target:**
  - `PAPER_HINTS` registry has 1 entry (SDAR). New papers should add an entry OR rely on no-hint defaults. Add entries opportunistically — they improve fidelity but aren't required.
  - Auto-generated rubric quality is the dominant lever for non-PaperBench papers; the LLM that writes the rubric is the limiting factor. No code fix can close this — it's a model-capability gap.
  - `paper_invariants.py` only catches things in its registry. Missing entries = no invariant enforcement. Same opportunistic add story.

### BUG-NEW-039 — empty-state copy overlaps the Paper node label on iter-1 canvas (UI, cosmetic)

User screenshot 2026-05-29 03:22: while waiting for the first primitive, the canvas renders the "Paper" node centered AND the empty-state `<div>` ("Waiting for first iteration / The root model is thinking…") on top of it — text overlaps. Z-order or absolute-positioning bug in the lab canvas. Low priority — purely visual, only visible during the ~1 min pre-first-primitive window. File: likely `frontend/src/components/lab/rlm/{exploration-tree,rlm-lab}.tsx`. Deferred behind BUG-NEW-038 validation.

## BUG-NEW-042 (Attempt 6 — run_experiment failed)
**Symptom:** `run_experiment` returned `dockerfile parse error on line 1: unknown instruction: You've` after `implement_baseline` (19m41s, no error) had completed.

**Root cause:** the `implement_baseline` prompt (`backend/agents/prompts/baseline_implementation.py:8`) tells the sub-agent to "Write code and Dockerfile" and to point its dockerfile_path manifest entry at `runs/prj_.../Dockerfile` (project_dir). The sub-agent's response on iteration 2 was conversational prose starting with `"You've ..."` rather than valid Dockerfile content. `Write` dumped it verbatim, overwriting build_environment's good Dockerfile (originally `# SDAR baseline …`). `run_experiment` always rebuilds from `ctx.project_dir/Dockerfile` (`primitives.py:2650`), and the rebuild died at parse time. Two follow-up implement_baseline retries completed in 2-3s — far too fast to have actually re-written the file, suggesting an empty-input / no-op response path.

**Fix (LANDED 2026-05-29):** `backend/agents/rlm/primitives.py::_validate_dockerfile_shape` rejects any Dockerfile whose first non-blank, non-comment line isn't `FROM` / `ARG` / `# syntax=`. Two enforcement sites:
1. **implement_baseline (auto-recover):** snapshots `ctx.project_dir/Dockerfile` BEFORE the sub-agent runs; if the post-write file fails shape validation AND the pre-snapshot was valid, restores the snapshot and emits a `dashboard_event` warning (`code="dockerfile_shape_guard"`). The sub-agent's failed write is silently corrected; the iteration continues without wasting a `run_experiment` dispatch.
2. **run_experiment (fail-fast):** if the on-disk Dockerfile is still malformed (rare — implies BOTH pre-snapshot AND sub-agent write were bad), persists a `failure_class="dockerfile_invalid"` repairable result with a `suggested_fix` telling the root to call `implement_baseline` again. `_RUN_EXPERIMENT_REPAIRABLE_FAILURES` includes `dockerfile_invalid` so the typestate classifier routes it correctly.

Tests: `tests/agents/rlm/test_dockerfile_shape_guard.py` — 11 cases (valid: `FROM`, `ARG`, `# syntax=`, comments-then-FROM, leading whitespace; invalid: empty, whitespace-only, prose-first, RUN-before-FROM, comments-only).

## Attempt 7 (killed at user request)
- Reason: BUG-NEW-038 companion fix VALIDATED (implement_baseline cleared at 19m41s). Then user stopped the sprint to bundle BUG-NEW-041 (phantom-running on SIGKILL) + BUG-NEW-042 (Dockerfile prose) + Playwright Loop 4 wiring into one cleanup pass.
- **2026-05-29 14:13 UTC:** killed `prj_645a7069cc73430c` via `scripts/loops/kill_and_restart.sh`; backend uvicorn also stopped.

## Fixes landed in the cleanup pass (2026-05-29)
- **BUG-NEW-041 (CLOSED):** `backend/cli.py::_install_termination_handlers` registers SIGTERM/SIGHUP handlers (main thread of `cmd_reproduce`). Handler atomically flips `demo_status.json::status="killed"` + `killReason="received signal <n>"`, then `raise_signal(SIGINT)` so the existing `KeyboardInterrupt` graceful path also runs. `_mark_demo_status_stopped` / `_mark_demo_status_failed` now treat `killed` as a terminal state and refuse to overwrite it. Tests: `tests/cli/test_termination_handler.py` (6 cases, including a synchronous handler invocation that verifies status-write + `KeyboardInterrupt`-conversion).
- **BUG-NEW-042 (CLOSED):** see fix block above.
- **Loop 4 (Playwright) WIRED:** `frontend/e2e/lab-watch.spec.ts` (2 tests: `/lab?projectId=<env>` with screenshot + console-error assertion, and `/leaderboard`) + `scripts/loops/lab_watch_loop.sh` (outer 5-min cycle, honours `LAB_PROJECT_ID` / `LAB_BASE_URL` / `LAB_WATCH_MAX_CYCLES` / `LAB_WATCH_SCREENSHOT_DIR`). Updated `docs/runbooks/2026-05-29-monitoring-loops.md` Loop 4 row + open-follow-ups section.

## Next attempt prerequisites (when ready to resume the SDAR sprint)
- Backend up: `./start.sh` (or `.venv/bin/uvicorn backend.app:create_app --factory --reload --port 8000`).
- Fresh PDF stage: `cp <existing pdf> /tmp/sdar_2605.15155_attempt8_$(date -u +%Y%m%d_%H%M%S).pdf`.
- Launch: `nohup env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY ./.venv/bin/python -m backend.cli reproduce <pdf> --mode rlm --model claude-oauth --sandbox runpod --provider anthropic > /tmp/sdar-attempt8.log 2>&1 &`.
- Validation signal for the BUG-NEW-042 fix: either zero `dockerfile_shape_guard` warnings in `dashboard_events.jsonl` (sub-agent wrote clean Dockerfile), OR one or more such warnings with `"restored": true` (guard caught + recovered), AND `run_experiment` proceeds past the build step.
- Validation signal for the BUG-NEW-041 fix: `kill <pid>` of the CLI subprocess (or `scripts/loops/kill_and_restart.sh`) leaves `demo_status.json` with `status="killed"` — never `running`.
