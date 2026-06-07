# 2026-05-27 SDAR run issues + bug log

Run: `prj_09047604e591d969` — SDAR (arxiv 2605.15155), smallest-two scope,
`--mode rlm --sandbox runpod --model claude-oauth --vram-gb 38
--max-wall-clock 5400 --max-pod-seconds 5400 --max-usd 15`. Worktree off
`origin/main @ cb415ba` at
`/Volumes/CS_Stuff/openresearch/.claude/worktrees/sdar-main`.

This doc is a **living issue log** — appended to as the run progresses.

## Severity legend

- **P0** — blocks the run or destroys rubric score; fix before next launch.
- **P1** — degrades the run or the UX; fix soon.
- **P2** — visual polish or developer ergonomics.

## Open issues

### P1 — `compute_scope_invalid` warning at `plan_reproduction`

**T+6m30s.** The RLM root passed `compute_scope` as a **string** describing
"CPU-only, single process, no GPU, no distributed training, no external
network. Memory budget under 2 GB. Smoke run under 2 minutes; full run under
~30 minutes wall clock. No model weights are downloa[ded]" instead of a dict.

The orchestrator emitted `run_warning code=compute_scope_invalid` and coerced
the value, but the **intent** leaked into the agent's plan. SDAR requires GPU
(Qwen-3B + RL); a CPU-only baseline scores ~0 on the rubric. The cached
`rlm_state/gpu_plan.json` (A6000 48GB) still applies at `run_experiment`, but
`implement_baseline` may write training code that does not target CUDA, which
the rubric leaf scorer detects via static analysis + runtime checks.

**Likely root cause:** the system-prompt example or the schema hint for
`plan_reproduction` does not strongly signal that `compute_scope` is a dict;
the model defaulted to a free-form summary.

**Mitigation today:** send a chat-steering message via the lab sidebar before
next iteration: *"Your compute_scope was wrong — SDAR REQUIRES GPU. The pod
is RTX A6000 48GB. Write CUDA training code with `.to('cuda')` and
`torch.cuda.is_available()` assertions. Real Qwen weights from HuggingFace,
not surrogates."*

**Permanent fix candidates:**
1. Type-narrow the `compute_scope` parameter in
   `backend/agents/rlm/primitives.py::plan_reproduction` — refuse strings hard
   and return an error result the root must repair from.
2. Add a primer line in the system prompt: "`compute_scope` is a dict with
   keys {cpu, gpu, vram_gb, ...}; never pass a string."
3. Add a guard in the baseline-implementation prompt that re-reads the
   cached `gpu_plan.json` and forces CUDA codegen when `gpu_count > 0`.

### P0 — `sdk_pre_emit_stall` killed `implement_baseline`, cascaded to rubric=0

**T+~10m44s** (event ts 21:09:05). The earlier asyncgen `RuntimeError` was
NOT benign after all — it manifested at iteration 0 as a 240-second SDK
deadlock. Event:

```
run_warning code=sdk_pre_emit_stall
  message: "implement_baseline: SDK pre-emit stall — no file activity in 240s.
           Likely SDK aclose deadlock pre-result."
```

Cascade in the same 30ms:

1. `implement_baseline status=error` — returned a dict `{success: False,
   error: 'implement_baseline: SD...'}`, no `code_path`.
2. Root called `run_experiment(arg0=dict[4], arg1=str[46])` anyway — passing
   the error-dict as `code_path`.
3. `run_warning code=iteration_boundary_recommended` — orchestrator caught:
   "run_experiment: code_path must be a non-empty string path, got dict".
4. `run_experiment status=error` — `failure_class` + `contract_violations`.
5. `verify_against_rubric` ran on the empty result and scored **0.0 / 0.6
   target across all 6 areas** (Method 0.0/.38, Data 0.0/.13, Execution
   0.0/.18, Eval 0.0/.16, Match 0.0/.10, Artifacts 0.0/.05). Total fail.
6. Forced-iteration policy (`OPENRESEARCH_MIN_RUBRIC_ITERATIONS=2`) will refuse
   `FINAL_VAR` and force iteration 1.

**This is the Adam+VAE rubric=0 cascade shape PR-π.1 hotfix
(commit `022e3f4`, parent of current HEAD `cb415ba`) was meant to detect.**
The mtime-based pre-emit progress monitor DID detect the stall (it emitted
the warning), but `implement_baseline` returned an error dict rather than
raising — and the root model did not treat the error as terminal; it called
`run_experiment` on the error dict, which then violated the code_path
contract.

**Root causes (two distinct bugs):**

1. **claude-agent-sdk teardown race** (SDK version `0.2.82`, pinned in
   `backend/requirements.txt`). The `aclose(): asynchronous generator is
   already running` error from earlier (T+~2m, T+~6m) is the same race
   that locked up the SDK at T+~6m45s..T+~10m45s. Mechanism:
   - The SDK opens a subprocess to the `claude` CLI binary (we observed
     pid 25771 in `ps -ef` early in the run) and streams JSON results via
     `InternalClient._process_query_inner`, an async generator.
   - On completion or cancellation the SDK calls `aclose()` on that
     generator. If another coroutine is still iterating it (or the asyncio
     event loop is closing concurrently), `aclose()` raises
     `RuntimeError: aclose(): asynchronous generator is already running`
     and the cleanup hangs without ever yielding a result to the caller.
   - The outer `concurrent.futures` future in
     `backend/agents/rlm/primitives.py::implement_baseline` (line ~1419)
     therefore never resolves; the sub-agent has nothing to emit.
   - Two watchdogs guard against this in
     `primitives.py` (already in the codebase; both fired correctly here):
     - **Pre-emit stall watchdog** (240s, `_PRE_EMIT_STALL_S`, set by
       `OPENRESEARCH_PRE_EMIT_STALL_S` env, default 240). Fires if no file
       has been written to `runs/<id>/code/` since the primitive started.
       Returns a `repairable` error dict, **explicitly NOT cached** so a
       retry is fresh (line 1488-1490 comment). This is what fired in
       attempt #1.
     - **Post-emit aclose stall watchdog** (120s, `_ACLOSE_STALL_S`).
       Fires if `commands.json` has been written (real code produced) but
       no file has been touched for 120s — the SDK is hung on cleanup.
       Breaks out and harvests the on-disk files as the result.
   - The 240s pre-emit detector is the **PR-π.1 mtime-based hotfix**
     (commit `022e3f4`, parent of HEAD). It replaced an earlier
     name-list-based detector that false-positive-escalated when the
     sub-agent wrote `rubric_guard.py + config.json` but not yet
     `commands.json` (the 2026-05-27 Adam+VAE regression cited in the
     hotfix commit message).
   - **Underlying SDK fix needed:** the race lives in `claude-agent-sdk`
     itself; OpenResearch cannot patch it in user code. Options ranked by
     feasibility:
     1. Pin `claude-agent-sdk` to an older known-good version (check
        release notes for the asyncgen-teardown patch).
     2. Wrap the SDK call in `asyncio.wait_for(coro, timeout=...)` with a
        force-cancel pathway in `backend/agents/rlm/binding.py` — won't
        prevent the hang but lets the wrapper recover faster than the 240s
        watchdog.
     3. Vendor the SDK and apply a local patch in
        `_process_query_inner`: wrap `aclose()` in
        `await asyncio.wait_for(gen.aclose(), timeout=5.0)` with
        `except asyncio.TimeoutError: pass`.
2. **Root model treats `implement_baseline` error-dict as a code path.**
   When `implement_baseline` returns `{success: False, error: ...}`, the
   root should call `propose_improvements` next, not `run_experiment`.
   The system prompt's iteration-discipline block should be tightened, or
   the `run_experiment` primitive should *refuse* dict arguments earlier
   (it does — that's what the iteration_boundary warning is — but the
   refusal still costs a primitive call and a rubric=0 verify).

**Status at end of iteration 0:** rubric=0, $0 spent (no pod yet),
~10m44s elapsed of 90min cap. Plenty of budget for the forced retry, but
if the SDK deadlock is reproducible, every iteration will burn ~4m on the
stall detector and never produce code.

**Mitigation options to consider next:**
- Kill this run and restart after pinning `claude-agent-sdk` to a known-good
  version (check release notes for the asyncgen-teardown fix).
- Let it ride and see if iteration 1 reproduces the deadlock — confirms
  whether it's flaky-SDK or every-call.
- Send a chat-steering message hinting the root to retry `implement_baseline`
  fresh rather than chain into `run_experiment` on error.

### P1 — `RuntimeError: aclose(): asynchronous generator is already running`

**T+~2m, T+~6m** (recurring). The `claude-agent-sdk` internal client emitted:

```
an error occurred during closing of asynchronous generator
  <async_generator object InternalClient._process_query_inner at ...>
RuntimeError: aclose(): asynchronous generator is already running
Loop <_UnixSelectorEventLoop ... closed=True> that handles pid <subproc> is closed
```

Appears benign so far — the next sub-RLM completes successfully each time and
primitives keep flowing. Likely a teardown race in the SDK between the
asyncio event-loop close and the subprocess wait.

**Investigation TODO:** confirm in `claude-agent-sdk` release notes whether
this is fixed in a later version than what `backend/requirements.txt` pins.

**Status update (T+~10m45s):** NOT benign. Promoted to P0 above — this
race deadlocked `implement_baseline` for 240s and produced rubric=0.

## Setup / environment issues (already worked around)

### P1 — `npm ci` fails on macOS arm64 because lockfile pins linux/x64 deps

```
npm error notsup Valid os:   linux
npm error notsup Actual os:  darwin
npm error notsup Valid cpu:  x64
npm error notsup Actual cpu: arm64
```

`package-lock.json` carries a `@next/swc-linux-x64-gnu` (or similar) under
`optionalDependencies` and refuses to install on darwin/arm64. Workaround
used: `npm install --no-audit --no-fund --force` (526 packages installed in
~6s). The `--force` flag is recommended-protections-disabled; not ideal.

**Permanent fix candidates:**
1. Regenerate `package-lock.json` on macOS so cross-platform optional deps
   are honored, OR
2. Drop the platform-pinned dep into `optionalDependencies` with the right
   metadata so `npm ci` skips it gracefully on non-Linux.

### P2 — Turbopack rejects symlinked `node_modules`

```
TurbopackInternalError: Symlink [project]/node_modules is invalid,
it points out of the filesystem root
```

When a worktree symlinks `frontend/node_modules → main-repo/frontend/node_modules`
to save install time, Turbopack refuses to start. Workaround: install a real
`node_modules` in the worktree.

**Permanent fix:** none in our code; this is upstream Turbopack behavior.
Document in `CLAUDE.md` so future worktree-based runs don't hit it.

### P2 — Stale `next-server v14.2.35` lingers on :3000 between sessions

A previous `next start` from the main repo was still bound to `:3000` when
this session started. It silently absorbed `curl :3000` health checks while
the worktree's Next 16 dev server crashed on Turbopack. Workaround: `kill`
the stale PID.

**Permanent fix candidate:** ship a `scripts/lab-up.sh` that checks
`lsof -iTCP:3000,8000 -sTCP:LISTEN` first and refuses to start if a foreign
server is bound.

## UI bugs

### P1 — `runpod: executing` pill shown when NO pod exists

**T+~22m** (UI watcher mismatch, 16:20:45 local). `GET /runs/<id>/runpod-status`
returned:

```json
{
  "status": "executing",
  "label": "runpod: executing",
  "detail": "run_experiment has started. The RunPod pod should be provisioning or executing commands.",
  "source": "events",
  "pod": null,
  "updated_at": "2026-05-27T21:17:08.061288+00:00"
}
```

`pod: null` — yet the label says **"executing"**. Cross-checked with
`runpodctl pod list` → `[]`. No pod exists; the endpoint derives `executing`
purely from the `run_experiment` start event. This is a category error:
the label should distinguish:

| Actual state | Correct label |
|---|---|
| no pod, run_experiment not yet started | `runpod: idle` |
| run_experiment start emitted, no pod_id yet | `runpod: provisioning` |
| pod_id known, pod RUNNING | `runpod: running` |
| pod RUNNING, command executing | `runpod: executing` |
| run_experiment ok | `runpod: stopping` |
| run_experiment error before pod | `pre-pod failure` (NOT runpod) |

**Fix:** `backend/routes/...` (the runpod-status route — likely
`backend/routes/runs.py` or similar) should join the `pod_id` from
`runs/<id>/rlm_state/` or the latest `pod_created`/`pod_started` event
before claiming "executing".

**Same root cause** as the earlier P2 about "runpod: last experiment failed" —
both labels treat `run_experiment` lifecycle as pod lifecycle without
checking pod existence.

### P3 — `/favicon.ico` 404s

**T+~24m.** Headless Chromium requests `/favicon.ico` and the backend
returns 404. Cosmetic. Either serve a favicon from `frontend/public/` (the
Next.js convention) or add a redirect rule. Already filtered out of the
UI watcher's `seen404s` dedup so it won't fire again.

### P2 — Frontend logs 404s from `/leaf-scores` as console errors during in-flight runs

**T+~22m** (UI watcher caught 3 console errors). The lab page repeatedly
polls `/api/demo/runs/<id>/leaf-scores`, which the backend (`backend/app.py:854`)
**intentionally 404s** until `final_report.json` is on disk —
*"Returns 404 when the run dir or final_report.json do not exist yet"* is
literally in the docstring. The frontend treats every poll as an error and
spams the browser console with `Failed to load resource: 404`.

By the time the run completes, several dozen 404 entries are in the console.
This makes real errors hard to spot during the run.

**Fix:** in
`frontend/src/app/api/demo/runs/[projectId]/leaf-scores/route.ts` (or the
caller that consumes it), swallow 404 silently and treat as "not ready
yet"; only surface non-404 failures to the console. Or, better, have the
backend return `200 {"status":"not_ready"}` instead of 404 — keeps the
HTTP semantics aligned with "the resource is known, the data isn't ready".

### P2 — UI watcher status_mismatch false-positives during normal runs

**T+~22m** (recurring). My `ui_watch.mjs` flags
`status_mismatch backend=running ui_pills=["runpod: executing",...]`
on every check. Cause: the UI uses **descriptive labels**
("runpod: executing", "running implement_baseline (Ns)") in the visible
pills, not the raw backend `status` string ("running"). There IS a
green "running" pill in the header (visible in the user's screenshot),
but my watcher's broad CSS selector
`[class*="pill"], [class*="badge"], [class*="chip"]` is matching too many
elements including the rubric area chips ("✗ Method and code fidelity")
and missing or losing the small status pill in the ordering.

**Fix in the watcher** (not the app): narrow the status-pill selector to
the specific status-pill component or accept any pill containing the
backend status as a substring of a descriptive label
("execut", "running", "complet"). Will land when I restart the
watcher; for now treating these as benign noise.

### P2 — "runpod: last experiment failed" badge fires even when no pod was provisioned

**T+~14m** (visible in user screenshot at 4:12 PM local). The header pill row
shows `runpod: last experiment failed` in red even though zero pods were
ever provisioned in this run (the failure was a `run_experiment` contract
violation — bad `code_path` arg — that errored *before* the RunPod
backend was invoked). The badge conflates "run_experiment failed" with
"runpod pod failed."

**Fix candidate:** in `frontend/src/components/lab/rlm/` (likely
`use-rlm-run.ts` or the header pill component), gate the runpod-status
badge on actual `gpu_resolved` + first pod-id in
`runs/<id>/dashboard_events.jsonl`. Don't tint the badge red for failures
that happened *before* a pod existed; use a neutral "pre-pod" state, or
phrase as "implement_baseline failed."

### FIXED — phase-strip current-phase text unreadable

**Visually confirmed in user screenshot T+~14m:** the active "Implement"
phase label reads bright orange against the dark panel, matching its dot.
Done/pending phases read as bright gray. Reads cleanly.

`frontend/src/components/lab/rlm/pipeline-phase-strip.module.css::.current`
used `color: var(--accent-ink)` (oklch 0.20 — dark text intended for use *on*
an accent background) on the dark panel, making the active phase label
nearly invisible. Changed to `color: var(--accent)` (oklch 0.80) so the
active label reads bright orange to match its dot.

### P1 — Constellation bubbles show no context on what each call did

User-reported via screenshot at T+~5m. The exploration tree on the lab
canvas renders bubbles labelled only by primitive name ("Understand",
"Sub-RLM", "Extract Hyp…") — no indication of *which* section was read, what
sub-query was asked, or what was extracted. Specifics:

- "Understand" labels overlap adjacent bubbles.
- "Extract Hyp…" truncated mid-word.
- Sub-RLM bubbles do not show the actual sub-query text (the data is in
  `sub_rlm_spawned`/`sub_rlm_complete` events).
- Comprehension/Paper container rects show only the title, no aggregate
  state.
- Empty ghost circles ring the perimeter with no labels.
- Orphan orange dash pip next to the "Paper" container rect.
- Purple-on-dark contrast for labels is borderline.

**Queued action:** dispatch a frontend-design subagent against
`frontend/src/components/lab/rlm/` after this run completes (task #1 in
the session task list). Source of truth for available payload fields is
`runs/<id>/dashboard_events.jsonl` plus
`backend/agents/rlm/sse_bridge.py::sanitize_iteration`.

### P1 — Tree (constellation) not visible alongside rubric/scorecard view

**T+~14m** (visible by absence in user screenshot at 4:13 PM). The
exploration-tree canvas shown earlier (bubbles for Sub-RLM, Understand,
Paper, Comprehension, Extract Hyp…) is not present on the same scroll
position as the rubric/scorecard panel. The operator has to scroll or
toggle a view to see what the agent is doing structurally.

**Fix candidate:** in `frontend/src/components/lab/rlm/rlm-lab.tsx` layout,
either (a) make the constellation canvas a left/right pane that sits next
to the rubric scorecard at the same scroll position, or (b) add a fixed
mini-map of the tree in the header band so it's always visible. The 360px
right sidebar (`NodeDetailSidebar`) is already lifted to a single source of
truth; reuse that selection state to drive the mini-map.

### P1 — Operator needs more live insight into what the agent is doing

**T+~14m** (user request via screenshot). The 149-second `Running
implement_baseline` banner conveys no progress. The error toasts say
*"Recovery: The root receives the full traceback and will retry, choose a
fallback, or repair the next primitive on the next REPL turn"* — generic;
they don't show *what* the agent is about to try, what it just tried, or
the reasoning thread.

**Specific insight surfaces missing:**
1. **Current step explainer:** "implement_baseline (149s) — sub-agent
   `claude-agent-sdk`, repair_context = path_1 (fix async SDK shutdown)".
2. **Last sub-RLM result preview:** show the answer text from the most
   recent `sub_rlm_complete`, not just its existence.
3. **Recent primitive ledger:** a tight 10-row strip of `[ts, primitive,
   status, 1-line result_summary]` updating live. Already partially
   implemented in `live-activity-strip.tsx`; verify it's mounted on this
   view and showing rich content (slice ranges, sub-query text, result keys).
4. **Decision trail:** when the root picks a candidate and marks it
   `running`, show the *full candidate description* (not just id/category)
   above the implement spinner. Currently those descriptions exist in the
   `candidate_proposed` event payload but are only surfaced inside the
   NodeDetailSidebar after click.
5. **Why-failed callout:** when `implement_baseline` errors, show the
   actual error code/message from the returned dict (`SDK pre-emit stall`
   would be far more useful than "dict[code, error, outcome, success]").

All these surfaces have data already in `dashboard_events.jsonl` and
`cost_ledger.jsonl`; the gap is presentation, not data plumbing.

## Observability gaps that surfaced

### P2 — UI status badge stays "queued" until first `primitive_call`

`demo_status.json` flipped to `status: "running"` at T+~10s, but the lab UI
header kept showing "queued — backend acknowledging…" until the first
`primitive_call` event landed at T+~1m20s. The header derives from the
SSE/event stream, not the polled status file, so there is a ~70s "dead"
window during which the run is alive but the UI looks frozen.

**Mitigation candidates:**
1. Emit a synthetic `run_started` event from the orchestrator as soon as
   `rlm.completion()` is dispatched, so the SSE stream has *something* to
   tick the UI off "queued".
2. Or: make the header read `demo_status.json::status` directly during the
   bootstrap window.

## Informational — `.venv-gepa` neighbor run (intentional)

A parallel `python -m backend.cli reproduce 2512.24601` is running from
`.venv-gepa` (PID 85574, started 4:23 PM, paper = RLM itself). The user
confirmed this is their own intentional background experiment, not an
abandoned daemon. Not a bug — noted here so future readers of this doc
don't mistake the parallel CPU/network activity (or any shared Claude OAuth
quota pressure) for an SDAR-run issue.

## Self-healing observed (informational)

### T+~11m25s — root proposed and ran candidate `path_1` to fix the P0

After the rubric=0 cascade, `propose_improvements` returned 3 candidates that
diagnose the run's own failure modes — meta-correct triage:

- **`path_1` infrastructure_stability** (chosen, marked `outcome=running`):
  *"Replace the async SDK shutdown path in implement_baseline with a sync
  emit-then-exit pattern (write result file before calling aclose, or wrap
  aclose in asyncio.wait_for with a 30s timeout and force-cancel). This
  should let implement_baseline return a valid code_path instead of stalling
  for 240s."*
- **`path_2` orchestration_contract**: *"Add an isinstance(code_path, str)
  and pathlib.Path.exists() guard between implement_baseline and
  run_experiment. On dict/error return, short-circuit to
  propose_improvements + retry instead of forwarding the error dict
  downstream."* — direct fix for the cascade-into-`run_experiment` shape.
- **`path_3` baseline_simplification**: deterministic template fallback.

`implement_baseline` restarted at 21:09:47 with the new plan
(`arg0=dict[5]`, the original plan was `dict[3]` — extra repair context
attached). This is iteration 0 -> 1 boundary; the forced-iteration policy
will accept this as a real attempt regardless of outcome.

**Note:** candidates `path_2` and `path_3` are excellent *codebase* fixes
worth landing in the repo after the run, even if `path_1` succeeds in
unblocking this run. They harden the orchestration contract against a known
SDK failure mode.

## 5-min snapshot ledger

| Tick | T+ | Status | Iter | Cost | Warnings | Pods |
|---|---|---|---|---|---|---|
| #1 | 12m24s | running | 0 | $0 | 3 | 0 |
| #2 | 17m24s | running | 0 | $0 | 3 | 0 |

**T+~18m47s — BREAKTHROUGH.** `implement_baseline` attempt #2 returned OK
(`result_summary: str[89]`, the code path string). `run_experiment` started
at 21:17:08 UTC with the real code_path. The candidate-`path_1` plan
worked: the sub-agent wrote `commands.json` + `config.json` +
`requirements.txt` + `rubric_guard.py` to `runs/<id>/code/`, and either
the SDK aclose finished cleanly OR the 120s post-emit watchdog harvested
the files. Either way, we have a code path for the first time.

**T+~28m — Real reproduction code is on disk.** `runs/<id>/code/commands.log`
shows the sub-agent wrote a full reproduction project:

- `train.py` — **1694 lines** of training script
- `requirements.txt` — Python deps (torch pre-installed in image)
- `commands.json` — ALFWorld download + `python train.py`
- `config.json` — all hyperparameters
- `rubric_guard.py` — copied from `backend/agents/rlm/rubric_guard.py`
  (the Lane G post-training schema validator) and verified to resolve all
  11 required dotted metric paths
- `README.md` — reproduction documentation
- `Dockerfile` (root) — updated to RunPod base image

The agent ran self-checks: `python -c "import ast; ast.parse(open('train.py').read())"` → Syntax OK; rubric_guard contract resolves all metrics. **This is a serious, real attempt at reproducing SDAR.**

## RESOLVED — the "silence" was pre-flight validation

**Closure at T+33m.** `run_experiment` was not hung. It was running the
**pre-flight contract validator** (static analysis of `train.py` against
the rubric contract), which finally emitted:

```
failure_class: preflight_blocked
error: "pre_flight: 1 hard violation(s) — see contract_violations"
contract_violations: [{
  severity: "hard",
  area: "Data and preprocessing fidelity",
  detail: "train.py:766: `load_dataset('hotpot_qa')` is a deprecated bare
           short name. The modern HuggingFace Hub requires `owner/name`
           and rejects this with `HfUriError: Repository id must be
           'namespace/name', got 'hotpot_qa'`. The 2026-05-25 Adam run
           died here.",
  hint: "Replace `load_dataset('hotpot_qa')` with
         `load_dataset('hotpotqa/hotpot_qa')` — the canonical owner/name
         on the modern Hub."
}]
```

**This is the system working as designed.** A previously-encountered
runtime failure (the 2026-05-25 Adam run) was codified into the
pre-flight contract, so this run was blocked **before** spending RunPod
money on code that would have crashed at the same line. Cost saved:
~$0.49/hr × pod-uptime ≈ $1-3.

**What was an actual P0 bug, though:** the pre-flight validator took
~14 minutes with **zero progress events** emitted between
`run_experiment status=start` (21:17:08) and `run_experiment status=error`
(21:31:29). To the operator and the UI, this looked identical to a hang.

**Fix candidates:**
1. Emit a `pre_flight_started` SSE event when the validator begins, and
   periodic `pre_flight_progress` events as it scans files.
2. Surface "running pre-flight contract validation on N files…" in the
   live activity strip, so the 14-minute silence becomes a 14-minute
   visible status.
3. Investigate why pre-flight is so slow — 14 min for one ~1700-line
   `train.py` suggests it's doing per-line LLM-graded checks, not pure
   static analysis. If LLM-graded, it should cache and parallelize.

## RESOLVED downgrade — `run_experiment` silent for 10+ min, NO pod ever visible in `runpodctl`

**T+~28m.** Since `run_experiment status=start` at 21:17:08 there have been
zero `pod_created`, `pod_started`, `gpu_escalated`, or
`run_experiment status=ok|error` events. The python process is alive
(`ps -p 17768`: 28m39s elapsed, 0.1% CPU, state=S sleeping → blocked on
I/O). **`runpodctl pod list --all`** returns `[]` consistently. **No pod
exists for this run, but `run_experiment` has not errored.**

Possible causes:
1. **RunPod API credentials mismatch.** `runpodctl` may be using a different
   account/key than `OPENRESEARCH_RUNPOD_API_KEY` in `.env`. Backend may have
   provisioned a pod on a different account that `runpodctl user` can't see.
2. **`run_experiment` is blocked on capacity wait** with no event emission.
   The capacity-escalation ladder (PR `aae89ad`) should emit
   `gpu_escalated`/`gpu_fallback`, but those haven't fired either —
   suggesting we are stuck *before* the escalation loop, maybe in the
   initial API auth or SSH key step.
3. **Local pre-pod prep** (rsync code, docker build push, etc.) is taking
   minutes silently. The `run_experiment` code does emit progress events
   for the pod-create step, but nothing for the pre-create steps.
4. **The parallel `2512.24601` neighbor run** is holding global locks
   (SSH session, RunPod auth) that block our run.

The python process being in state=S (sleep) with 0.1% CPU strongly
suggests it is **blocked on a network read** — either RunPod API or SSH
to a pod. That points to (1) or (2).

**Mitigation options to consider:**
- `runpodctl ssh list-keys` to confirm the SSH key in `.env` matches the
  account `runpodctl` is logged into. If they differ, we're authenticating
  for pod-create against a different account than `runpodctl user` shows.
- Add stdout-flushed log lines to `backend/services/runtime/runpod_backend.py`
  around the pod-create call so silence becomes visible.
- Add a `run_experiment` "pre-provision" SSE event so the dashboard can
  show "waiting on RunPod API…" instead of "executing".

## Worker-reports schema bug (P2)

`worker_reports.jsonl` entries since T+~18m carry `phase=null, summary=null,
ts=null` — only the `status` field is populated. The schema expects all
four. Likely a missing `**kwargs` plumbing or a default at the emit site.

**Snapshot #2 note:** `implement_baseline` retry has been running ~6 min,
past the 240s stall threshold that killed attempt #1. No new
`sdk_pre_emit_stall` warning — first encouraging sign. Either the
candidate `path_1` plan worked, or the SDK race is non-deterministic and
we got lucky. Will know which when attempt #2 emits a primitive other
than `heartbeat`.

(Schema bugs in the snapshot script — keeping for next runbook:
`last=` field uses `.data.primitive + "/" + .data.status`, but flat
`primitive_call` events put `primitive`/`status` at the top level —
should be `.primitive // .data.primitive`. Also `pods=` prints twice
because `grep -c ... || echo 0` triggers the fallback when grep matches
nothing AND exits non-zero. Trivial fixes, not worth restarting the
monitor mid-run.)

## Cost / capacity events (informational)

- **T+~4m** — `gpu_resolved` event: NVIDIA RTX A6000 48GB, 1×, COMMUNITY,
  $0.49/hr. Source=paper (paper mentioned hardware clues that resolved
  cleanly). Cached to `rlm_state/gpu_plan.json`.

---

*Appended live during the run. See `runs/prj_09047604e591d969/` for raw
event log and rlm_state checkpoints.*

## Final run summary

| Field | Value |
|---|---|
| project_id | `prj_09047604e591d969` |
| Paper | arXiv 2605.15155 (SDAR) — smallest-two scope (Qwen 1.7B + 3B) |
| Mode | rlm |
| Models | planner=`claude-oauth`, executor=`claude-sonnet-4-6` |
| Started | 2026-05-27T20:58:21Z |
| Completed | 2026-05-27T21:31:30Z |
| Wall-clock | ~33m07s (of 90min budget) |
| Verdict | **partial** |
| Rubric score | **0.0 / 0.6 target** (all 6 areas failed) |
| Iterations | 1 |
| Primitive calls | 37 |
| Pods provisioned | 0 |
| **Cost (USD)** | **$0.00** |

### Primitive call distribution
- `heartbeat` × 19
- `understand_section` × 3
- `extract_hyperparameters` × 1
- `resolve_gpu_requirements` × 1 (→ A6000 48GB COMMUNITY)
- `detect_environment` × 1 → `build_environment` × 1 → `plan_reproduction` × 1
- `implement_baseline` × 2 (first stalled in SDK aclose, second produced
  real 1694-line train.py + rubric_guard + Dockerfile + config)
- `run_experiment` × 2 (first errored on contract_violation — the
  cascade; second errored on pre_flight_blocked — caught
  `load_dataset('hotpot_qa')` deprecated path)
- `verify_against_rubric` × 2 (both 0.0)
- `propose_improvements` × 1 (returned 3 high-quality candidates targeting
  exactly the right bugs)
- `record_candidate_outcome` × 2 (path_1 marked `running` → `marginal`)
- `check_user_messages` × 1

### What we learned that's worth landing in the repo
1. **`compute_scope` schema needs type-narrowing** — refuse strings hard
   in `plan_reproduction` so the LLM can't leak "CPU-only" intent.
2. **`implement_baseline` SDK aclose race** is reproducible — landing
   `asyncio.wait_for` with a 5s timeout around the SDK's `aclose()` in
   `backend/agents/rlm/binding.py` would convert 240s hangs into 5s
   recoveries. Candidate `path_1` from this very run described the fix
   accurately.
3. **`run_experiment` contract guard for error-dict code_path** — root
   model passed an error dict where a string was expected; the orchestrator
   caught it but only after a primitive_call cycle. Candidate `path_2`
   described it.
4. **Pre-flight validation needs SSE progress events** — 14 minutes of
   silence felt like a hang; could be much faster too.
5. **`/api/demo/runs/<id>/leaf-scores`** semantics: change 404 → 200 with
   `{status: not_ready}` to stop console-error spam during runs.
6. **`runpod-status` endpoint** must check actual pod existence before
   labelling "executing" / "failed".
7. **Worker reports schema bug** — `phase=null, summary=null, ts=null`
   for most entries; the emit site isn't populating all fields.
8. **Constellation tree + live insight surfaces** — queued for the
   post-run UI subagent (task #1).

### The good news
- System self-healed from the cascade: root proposed 3 well-targeted
  candidates, picked the right one, retried successfully.
- Pre-flight contract validator saved ~$1-3 in pod cost by blocking
  known-broken code before provisioning.
- No false positives — the rubric=0 is honest; the code genuinely had a
  bug that would have crashed on the pod.
- Zero cost burn for a deep-debug session that produced a comprehensive
  bug log.

## Post-run audit findings (added after log re-scan)

### P1 — `timing.json` reports nonzero GPU hours even though zero pods existed

Post-run audit found `runs/prj_09047604e591d969/timing.json` contains:

```json
{
  "gpu_hours": 0.2393,
  "gpu_type": "a6000",
  "gpu_count": 1
}
```

This contradicts every other source:

- `runpodctl pod list --all` returned `[]` during and after the run.
- `cost_ledger.jsonl` totals `cost_usd=0.0`.
- `demo_status.json.cost_summary.usd_total=0.0`.
- `run_complete.cost_usd=0.0`.
- No `pod_created` / `pod_started` events exist.

Likely bug: the timing aggregator estimates `gpu_hours` from planned GPU type
and `run_experiment` wall time, not actual pod uptime. For pre-flight-blocked
runs, GPU hours must be `0`.

**Fix:** compute GPU hours from actual pod lifecycle events or runtime backend
metadata only. A cached `gpu_plan.json` should never imply spend or uptime.

### P1 — Final report loses candidate outcome state

`dashboard_events.jsonl` records candidate `path_1` as:

```text
path_1 -> running
path_1 -> marginal
```

But `final_report.json.improvements` marks every candidate as `declined`,
including `Bypass SDK aclose Deadlock`.

This is misleading: `path_1` was selected and produced real code on attempt #2.
It should be `marginal` (or a more explicit `attempted_then_preflight_blocked`),
not `declined`.

**Fix:** final-report assembly should join against `candidate_outcome` events by
`candidate_id`. Do not infer outcomes only from rubric delta.

### P1 — Final report metadata/scope is internally inconsistent

The final artifacts disagree on the paper and scope:

- `demo_status.json.paperTitle` and `final_report.json.paper.title` are
  `"paper_text"` instead of the SDAR title.
- `final_report.scope.requested` says "one environment (ALFWorld)", but the
  generated `train.py` targets ALFWorld + WebShop + Search-QA.
- The scope says `Qwen2.5-1.7B`; the worker report says
  `Qwen/Qwen3-1.7B-Instruct` + `Qwen/Qwen2.5-3B-Instruct`.
- The top-level run request was "smallest-two scope (Qwen 1.7B + 3B)", not
  ALFWorld-only.

This makes the report hard to trust even when the underlying code attempt is
real.

**Fix:** derive final-report paper metadata from arXiv metadata, and derive
`scope.ran` from generated `commands.json` / `config.json` / worker report
rather than a stale template.

### P1 — `data/calibration.json` was rewritten from 30-run history to this single failed run

The worktree now shows `data/calibration.json` changed from:

```text
schema_version: 1
based_on_n_preserved_runs: 30
many primitives with historical sample counts
```

to:

```text
schema_version: 2
based_on_n_preserved_runs: 1
only plan_reproduction / propose_improvements / run_experiment
```

This appears to have been generated from `prj_09047604e591d969` alone. Since
this run failed before any pod execution, overwriting calibration with it would
poison expected-cost / expected-token estimates.

**Fix:** calibration updates must be opt-in and should merge into the historical
corpus, not replace it. Failed pre-flight-only runs should either be excluded
from cost calibration or tagged separately.

### P1 — UI subagent changes were left unverified; lint currently fails

The post-run UI cleanup subagent created screenshots and tests pass, but it hit
the monthly spend-limit boundary before producing a usable final report.
Verification results from the worktree:

```text
npm --prefix frontend test
  36 files passed, 267 tests passed

npm --prefix frontend run lint
  failed: 2 errors, 5 warnings
```

Relevant new error in the edited UI path:

```text
frontend/src/components/lab/rlm/rlm-lab.tsx:180
react-hooks/set-state-in-effect:
setNowMs(completedAtMs) is called synchronously in an effect
```

There is also an existing/ref-scope lint error in `frontend/src/hooks/use-run.ts`
around `providerOptionsRef.current = providerOptions` during render.

**Fix:** clean up `rlm-lab.tsx` before merging the UI subagent output. The hook
lint error in `use-run.ts` should be handled separately if it predates this run.

### P2 — `leaf-scores` proxy fix masks backend 5xx as `not_ready`

The UI subagent changed
`frontend/src/app/api/demo/runs/[projectId]/leaf-scores/route.ts` to translate
both backend `404` and backend `>=500` into:

```json
{ "leaf_scores": [], "status": "not_ready" }
```

Suppressing the intentional pre-final-report `404` is good. Suppressing `5xx`
is risky: a real backend bug will now look identical to "not ready yet" and
could hide broken artifact generation.

Runtime logs also show the proxy now returns `200` for a nonexistent project:

```text
GET /api/demo/runs/prj_doesnotexist/leaf-scores 200
```

That means the route no longer distinguishes "valid run, leaf scores not ready"
from "run id does not exist".

**Fix:** only translate `404` to `not_ready`. Preserve `5xx` as an error, or
return `{status:"backend_error"}` with the original status code. If 404 is kept
as a sentinel, include enough backend context to distinguish missing
`final_report.json` from missing `run_dir`.

### P2 — `package-lock.json` was polluted by platform-specific optional dependency rewrite

The worktree now moves `@rolldown/binding-linux-x64-gnu` from
`optionalDependencies` into normal `dependencies` and removes its `optional`
flag in the lockfile.

This matches the earlier local workaround for `npm install --force` on macOS
arm64, but should not be committed as-is. It can recreate the exact
cross-platform install failure that blocked setup.

**Fix:** regenerate the lockfile cleanly on the intended platform or restore the
optional dependency metadata before committing.

### P2 — `next-env.d.ts` was changed by local dev-server state

`frontend/next-env.d.ts` changed:

```diff
- import "./.next/types/routes.d.ts";
+ import "./.next/dev/types/routes.d.ts";
```

This is a generated Next.js environment file mutation from local dev mode, not
a product change. It should not be part of a UI cleanup commit unless the repo
intentionally tracks dev-mode route types.

**Fix:** restore or regenerate consistently via the project's normal Next.js
workflow before committing.

### P2 — Cost ledger provider labels `claude-oauth` as `provider=openai`

`cost_ledger.jsonl` entries for `claude-oauth` show:

```json
{ "model": "claude-oauth", "provider": "openai" }
```

The dollar cost remains correct at zero, but the provider field is wrong and
will confuse dashboards that group usage by vendor.

**Fix:** cost ledger provider inference should map `claude-oauth` to
`anthropic` / `claude_oauth`, not `openai`.
