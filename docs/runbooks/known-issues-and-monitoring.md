# Known issues and monitoring runbook

_Last updated: 2026-05-23. This doc is the canonical "what looks wrong but isn't" plus "what's actually broken" register for the OpenResearch / ReproLab stack. Add new entries here when you discover a confusing failure mode; remove entries when the underlying bug is fixed AND the fix has shipped to `main`._

If you are triaging a run **right now**, start at §1 (UI signal interpretation) then §3 (active issues). For background context on the architecture see `CLAUDE.md` and `system_overview.md`; for the day-to-day UI walkthrough see `docs/runbooks/e2e-testing.md`; for preflight gates see `docs/runbooks/readiness.md`.

---

## 0. Quick-reference status board

| # | Issue | Severity | Status | Where fixed / Where to look |
|---|---|---|---|---|
| 1 | SDK `aclose()` deadlock kills RLM root (Defect 1) | blocker | **resolved 2026-05-23** | `backend/services/context/workspace/tools/rlm_query.py::ClaudeLlmClient.complete` — always thread-isolates now |
| 2 | SDK `transport.close()` futex hang on WSL2 (Defect 2) | blocker on WSL | mitigated | Same fix as #1 contains it. See `docs/superpowers/specs/2026-05-22-sdk-aclose-investigation.md`. |
| 3 | SSR↔client hydration mismatch in resize handles | dev-mode warning | **resolved 2026-05-23** | `frontend/src/hooks/use-resizable-panels.ts` — `useState` seeds with `defaultSizes`, storage hydrated in post-mount `useEffect` |
| 4 | UI "no signal Xs" chip false-alarms during long primitives | UX | resolved 2026-05-23 | `RlmHeader` now shows the in-flight primitive instead |
| 5 | Sub-RLM root strategy queries the same paper slice repeatedly | run quality | **open** | `rlms` PyPI library — root prompt or `primitives.py` cache |
| 6 | `REPROLAB_DEFAULT_SANDBOX=docker` disagreed with `start.sh` | gotcha | resolved 2026-05-23 | config, `.env`, and `.env.example` now match `start.sh` (`runpod`) |
| 7 | Missing `REPROLAB_FORCE_SANDBOX` silently forced Docker | gotcha | resolved 2026-05-23 | `backend/config.py` default is now empty; explicit env still overrides |
| 8 | Anthropic API-key with no credit balance dies at first Sonnet sub-call | blocker on stale keys | documented in CLAUDE.md | `.env` — leave empty if subscription is the source |
| 9 | macOS Keychain OAuth not detected by `factory.has_provider_credentials` | blocker | resolved 2026-05-23 | `backend/agents/.../factory.py::_has_claude_subscription_oauth` |
| 10 | `demo_status.json::updatedAt` lags far behind real progress | confusion | **open** (architectural) | Backend writes status snapshots far less often than the SSE event stream |
| 11 | `/api/demo/runs/<id>/leaf-scores` 404 spam in console for non-RDR runs | dev-mode noise | resolved 2026-05-23 | `frontend/src/hooks/use-rdr-artifacts.ts` stops after mixed missing cycles |
| 12 | Orphan `claude-agent-sdk` subprocesses persist after killed runs | quota leak | **open** (operational) | `pkill -9 -f "claude_agent_sdk/_bundled/claude"` between iterations |
| 13 | Orphan Playwright MCP Chrome locks `SingletonLock` indefinitely | dev workflow | **open** (operational) | Kill the Chrome process by PID, remove `~/Library/Caches/ms-playwright/mcp-chrome-*/SingletonLock` |
| 14 | `final_report.json` missing required keys → readiness tier 6 fails | bug shape | resolved 2026-05-23 | `backend/agents/rlm/report.py` — writes `mode`, `models`, `started_at`, `completed_at` |
| 15 | Hybrid controller dies if no PaperBench bundle (`prj_*` uploads) | blocker | resolved (F1) | Hybrid falls back to pure RLM when no bundle present |
| 16 | Path normalization for Windows ↔ WSL ↔ macOS | gotcha | resolved (F3) | `backend/services/paths.py` |
| 17 | `resolve_root_model` rejected `sonnet` / `claude-sonnet-4-6` UI aliases | blocker | resolved (F4) | `backend/agents/rlm/models.py::resolve_root_model` |
| 18 | `use-rdr-artifacts` polled forever on non-RDR runs (proxy 502/404 spam) | UX | resolved (F2/F7) | `frontend/src/app/api/demo/runs/[projectId]/...` route + hook |

§3 covers each open issue in depth. §4 is the family of fixes ("F1"-"F8") referenced in `e2e-testing.md`.

---

## 1. UI signal interpretation — what's a real alarm vs a false one

The lab UI surfaces several state indicators. Each has a specific source and known failure modes. Understanding them prevents 90% of "is it stuck?" questions.

### 1a. Status pill (top-right of header)

- **Green `running`** — the run subprocess is alive and `demo_status.json::status == "running"`. Says nothing about whether iterations are advancing.
- **Yellow `queued`** — subprocess spawned but no first event landed. Normal for the first 5-30 seconds after run start.
- **Red `failed`** — subprocess exited non-zero, OR a fatal exception was caught. Check `runs/<id>/runner.stderr.log` for the traceback.
- **Gray `completed`** — `final_report.json` was written, subprocess exited 0.

### 1b. "no signal Xs" chip

Shows up next to the status pill only when there is no heartbeat and no known
primitive in flight. If a long primitive is active, the header should show
`running <primitive> (Xs)` instead. Long-running primitives (especially
`implement_baseline`, which spawns a Sonnet sub-agent with tools) execute
between heartbeats and may legitimately consume 5-15 minutes with no event
traffic.

**Confirm if it's a real wedge:**
```bash
PROJECT=prj_xxx
find runs/$PROJECT/code -mmin -2 -type f                  # any file touched in last 2 min?
pgrep -fa "claude_agent_sdk/_bundled" | head -5           # are SDK workers alive?
ps -p <pid> -o pid,etime,pcpu                             # is the worker consuming CPU/wall?
```
A worker with `etime` increasing, `pcpu > 0`, AND files being touched in
`runs/<id>/code/` is working, not stuck.

### 1c. Iteration counter

Increments only when the rlm library emits a `repl_iteration` event. Iteration 0 → 1 transition can take 5+ minutes because the root reads the paper, plans, and (in iteration 1) typically spawns sub-RLMs before yielding back. **An "empty" first iteration is normal.**

### 1d. Primitive call counter

Counts `primitive_call` events. Updates immediately when a primitive returns. The 12 domain primitives are listed in `backend/agents/rlm/primitives.py`. The most common high-value names to watch:

| primitive | typical wall-clock | what it produces |
|---|---|---|
| `check_user_messages` | <100ms | file-I/O only, advances cursor |
| `respond_to_user` | <100ms | file-I/O only, emits `user_message_response` SSE |
| `understand_section` | 5-20s per call | datasets, metrics, training_recipe extraction |
| `extract_hyperparameters` | 10-30s | structured dict of hyperparams |
| `detect_environment` | 30-90s | `EnvironmentSpec` (Dockerfile spec) |
| `build_environment` | 1-5 min | builds + tags Docker image |
| `plan_reproduction` | 30-60s | `ReproductionContract` |
| `implement_baseline` | **5-15 min** | full code directory, dockerfile, train.py |
| `run_experiment` | 1-30 min depending on baseline | metrics, logs |
| `verify_against_rubric` | 30-90s | scored leaves |
| `propose_improvements` | 30-60s | candidate improvements |
| `record_candidate_outcome` | <100ms | bookkeeping |

`implement_baseline` is the single biggest wall-clock consumer and the most common source of "no signal" false alarms.

### 1e. Sub-RLM nodes in the canvas

Each sub-RLM is a recursive `rlm_query` or `llm_query` call against a slice of `context`. The root model dispatches them when it wants to read part of the paper without loading the whole thing into its turn. Multiple sub-RLMs in flight is normal; see §3.5 for the known quality issue (duplicate slice queries).

### 1f. REPL state rail (left side)

Shows the variables currently set in the rlm root's Python REPL. The size suffix (`str 137k`, `dict 563 B`) is the actual byte size of the offloaded variable, not a token estimate. Variables only appear here once they're explicitly assigned by the root's emitted code. `paper_text` is rarely listed because the rlm library binds it to the special `context` name, and the SSE bridge filters that.

### 1g. Rubric score strip + RubricBreakdown panel

- **RubricStrip** (the bar at the top): a single roll-up across all rubric leaves. Updates on `rubric_score` SSE events.
- **RubricBreakdown** (cluster grid): populates from `cluster_*` events; only emitted on RDR/hybrid runs against a bundle paper. On arXiv / PDF uploads (no bundle), this stays empty and the polling hook stops after ~15s of all-404s (F2).

---

## 2. Run-lifecycle event reference

Every event the SSE stream emits, where it comes from, and what it means.

| event | source | meaning |
|---|---|---|
| `repl_iteration` | rlm core loop | one full iteration of `RLM.completion()` completed |
| `iteration_heartbeat` | rlm core loop | iteration is alive (fires periodically inside long iterations) |
| `primitive_call` | `primitives.py` decorator | a domain primitive returned (success or error) |
| `sub_rlm_spawned` | `rlm_query.py` | a recursive `llm_query` / `rlm_query` call started |
| `sub_rlm_complete` | `rlm_query.py` | a recursive call returned |
| `candidate_proposed` | `propose_improvements` | the root proposed an improvement |
| `candidate_outcome` | `record_candidate_outcome` | the root recorded an accept/reject for a candidate |
| `rubric_score` | `verify_against_rubric` | rubric was scored against current results |
| `user_message` | `messages.py` route | a user message was POSTed via the chat panel |
| `user_message_response` | `respond_to_user` primitive | root replied to a user message |
| `run_warning` | various | non-fatal anomaly logged |
| `run_complete` | run shutdown | terminal event; `final_report.json` should exist after this |
| `cluster_started` | RDR controller | bundle paper only; one rubric work-cluster started |
| `cluster_artifact_emitted` | RDR controller | bundle paper only; cluster wrote an artifact |
| `cluster_scored` | RDR controller | bundle paper only; cluster's leaves scored |
| `repair_dispatched` | hybrid controller | bundle paper only; RLM repair task dispatched for a low-scoring leaf |

The append-only log at `runs/<id>/dashboard_events.jsonl` is the source of truth. `sse_bridge.sanitize_iteration` is the single egress point that strips REPL locals and bounds stdout/stderr.

---

## 3. Active issues — diagnosis and remediation

### 3.1 SDK aclose deadlock — **FIXED 2026-05-23**

**Symptom:** stderr fills with
```
an error occurred during closing of asynchronous generator
RuntimeError: aclose(): asynchronous generator is already running
Loop <_UnixSelectorEventLoop running=False closed=True> that handles pid N is closed
```
and the run never produces any `repl_iteration` events. Reproducible on macOS 25 / claude-agent-sdk current bundled version. Originally documented as "racy" in `docs/runbooks/e2e-testing.md` §4f; turned out to be reliably reproducible on macOS once we exercised the `claude-oauth` root model path.

**Root cause:** `ClaudeLlmClient.complete` in `backend/services/context/workspace/tools/rlm_query.py` had a guard that only thread-isolated SDK calls if the caller was already inside a running asyncio loop. The RLM root calls `completion()` synchronously from a worker thread with no outer loop, so it fell through to a bare `asyncio.run(...)` which left the SDK's nested async generators bound to a loop that was torn down before generator finalization could run.

**Fix:** `ClaudeLlmClient.complete` now always runs the SDK call inside a dedicated `ThreadPoolExecutor` with `shutdown(wait=False)`. The SDK's loop-bound async generators are created and finalized inside the worker's own loop; the caller's loop never sees them. `aclose()` warnings still appear in stderr as the abandoned worker GCs, but they are now non-fatal — the result text has already returned via `future.result()`.

**Verification:** run a fresh OAuth-based run:
```bash
curl -sS -X POST http://127.0.0.1:8000/runs/arxiv \
  -H 'content-type: application/json' \
  -d '{"url":"https://arxiv.org/abs/2512.24601","mode":"rlm","sandbox":"docker","model":"sonnet"}'
# Then watch:
PROJECT=<projectId>
until grep -q '"event":"repl_iteration"' runs/$PROJECT/dashboard_events.jsonl; do sleep 5; done
# First iteration should land within 2-4 minutes
```

**Cross-reference:** the rdr path's `_run_sdk_in_thread` in `backend/agents/rdr/agent.py` already had the same thread-isolation pattern. The fix brings the RLM path into parity.

### 3.2 Hydration mismatch in resize handles — **FIXED 2026-05-23**

**Symptom:** Next.js dev overlay dialog:
```
A tree hydrated but some attributes of the server rendered HTML didn't match the client properties.
...
+ aria-valuenow={200}
- aria-valuenow="280"
+ style={{width:200}}
- style={{width:"280px"}}
```
Stack points to `src/components/lab/rlm/resize-handle.tsx` rendered by `RlmLab`.

**Root cause:** `frontend/src/hooks/use-resizable-panels.ts` initialized state with `useState(() => readStorage())`. On the server, `readStorage` returned defaults (no `localStorage`); on the client's first render, it read the persisted value. The mismatch wasn't patched up by React, so the rail rendered at 280 then jumped to 200 on first effect cycle.

**Fix:** seed `useState` with `defaultSizes` (deterministic across SSR/CSR), then hydrate from localStorage in a post-mount `useEffect` via `queueMicrotask`. There is a one-frame visible "snap" from default → stored width — acceptable for a dev workflow surface.

**Test coverage gap:** `vitest` runs under jsdom with no SSR layer, so hydration mismatches cannot be detected there. The fix should ideally be guarded by a Playwright check that opens `/lab` and watches for the React hydration warning. See §5 (coverage gaps).

### 3.3 macOS Keychain OAuth detection — **FIXED 2026-05-23**

**Symptom:** `validate_provider_credentials` / `has_provider_credentials` returned False on every macOS dev machine with Claude Code logged in. The sub-agent runtime resolved as `unresolved`; `implement_baseline` died at first Sonnet sub-call with `cost_usd=0.0`.

**Root cause:** modern Claude Code on macOS stores OAuth credentials in the Keychain (`security find-generic-password -s "Claude Code-credentials"`), not in `~/.claude/.credentials.json`. The credential-detection helpers only checked the file.

**Fix:** both helpers now route through `_has_claude_subscription_oauth()`, which probes the Keychain on `darwin` and the file on other platforms.

**Cost implication:** the cheapest local-dev cost model is now: OpenAI for the root (~$1/run via `--model gpt-5`), OAuth subscription for Sonnet sub-agents ($0), RunPod COMMUNITY for the GPU sandbox (~$0.34/run). Zero Anthropic API balance required. See CLAUDE.md §"Fixed 2026-05-23".

### 3.4 Wedge indicator false alarms — resolved 2026-05-23

**Symptom:** `rlm-header.tsx` shows an orange `no signal Xs` chip while a legitimate long primitive (especially `implement_baseline`) is in flight. The screenshot tail's `wedge-log.tsv` will record steadily climbing values like `234s → 268s → 300s → 331s` even though the agent is actively writing code under `runs/<id>/code/`.

**Root cause:** the chip is driven purely by `Date.now() - new Date(lastHeartbeatAt).getTime()`. `iteration_heartbeat` is emitted by the rlm core loop, not by individual primitive calls. A primitive that does a single 5-10 min Sonnet sub-agent invocation produces no heartbeats *and* no other intermediate events.

**Workaround for users:** confirm a primitive call is in flight before assuming a wedge:
```bash
PROJECT=prj_xxx
.venv/bin/python - <<'PY'
import json, pathlib, datetime
events = [json.loads(l) for l in pathlib.Path(f'runs/{PROJECT}/dashboard_events.jsonl').read_text().splitlines()]
last_prim = next((e for e in reversed(events) if e['event'] == 'primitive_call'), None)
print('last primitive:', last_prim and last_prim.get('name'))
print('time since:', (datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(last_prim['timestamp'].replace('Z','+00:00'))).total_seconds() if last_prim else 'n/a')
PY
```
Cross-check with `find runs/$PROJECT/code -mmin -2 -type f` — if files were touched in the last 2 min, the agent is alive.

**Fix:** `RlmLab` derives the in-flight primitive from start/ok/error
`primitive_call` events and passes it to `RlmHeader`. The header renders
`running <primitive> (Xs)` while a primitive is active and keeps `no signal Xs`
for true no-primitive wedges.

### 3.5 Sub-RLM root strategy queries the same slice repeatedly — **OPEN**

**Symptom:** of N sub-RLMs spawned in iteration 1, several have nearly identical `prompt_preview` payloads — typically all targeting the paper's opening section.
Example from `prj_7674099e252837e1`: 5 of 11 sub-RLMs all started with `## paper_text  Recursive Language Models  Alex L. Zhang  MIT CSAIL …` (i.e., the paper title block).

**Root cause:** the `rlms` PyPI library's default root strategy. The root model is free to call `understand_section` (or `llm_query` on a slice) repeatedly with overlapping payloads if it judges that useful. There is no slice-content dedupe.

**Impact:** wasted tokens on the root surface (each `llm_query` is a fresh model call), inflated wall-clock, and a visually cluttered exploration tree.

**Possible mitigations** (none shipped yet, listed in order of invasiveness):

1. **System-prompt addendum** in `backend/agents/rlm/system_prompt.py`: explicitly tell the root *"avoid querying the same paper slice in consecutive iterations; if you already have an answer for a section, reference it from REPL state instead of re-asking."* Lowest-risk change.
2. **Cache by slice-content hash** in `backend/agents/rlm/primitives.py::understand_section`: keep a per-run dict keyed on `hashlib.sha256(text_slice).hexdigest()` and short-circuit returning the prior result. Risk: legitimate re-queries with different prompts on the same slice are also short-circuited.
3. **Slice-overlap detection** in the SSE bridge: emit a `run_warning` event when a sub-RLM's prompt has high textual overlap with a recent sibling. UI surfaces a dim badge but the call still proceeds. Information-only.

**Open question:** is the root's repetition a real strategy choice (e.g., "verify section X from a different angle") or a planning failure? Reading the actual `understand_section` results across the duplicates would tell us. Worth investigating before changing the prompt.

### 3.6 `demo_status.json::updatedAt` lags far behind real progress — **OPEN**

**Symptom:** when polling `demo_status.json` you'll see `updatedAt` frozen at e.g. `16:57:28` while the dashboard_events.jsonl shows events flowing through `17:18:16`. Looks like the run is stuck if you only read `demo_status.json`.

**Root cause:** `demo_status.json` is updated by atomic-write on a coarse cadence (status transitions, error captures, terminal events). The continuous progress signal lives in `dashboard_events.jsonl`. The two file-write loops are intentionally decoupled so a high-throughput event stream doesn't thrash the atomic write.

**Triage rule:** when diagnosing "is it stuck?", always cross-check both files:
```bash
PROJECT=prj_xxx
echo "--- demo_status ---"
.venv/bin/python -c "import json; d=json.load(open('runs/$PROJECT/demo_status.json')); print(f'status={d[\"status\"]} updatedAt={d[\"updatedAt\"]}')"
echo "--- last 3 events ---"
tail -3 runs/$PROJECT/dashboard_events.jsonl
```

**Why not fix:** writing `updatedAt` on every event would multiply IO by 100x without a user-facing benefit, since the SSE stream is the user-facing surface anyway. The right fix is to teach docs/tooling that `dashboard_events.jsonl` is authoritative for liveness.

### 3.7 Orphan `claude-agent-sdk` subprocesses persist — **OPEN** (operational)

**Symptom:** after a backend restart or a `pkill -9 -f "backend.cli reproduce"`, processes matching `claude_agent_sdk/_bundled/claude` remain with PPID=1. They consume OAuth subscription quota silently.

**Detection:**
```bash
ps -eo pid,etime,cmd | grep "claude_agent_sdk/_bundled/claude" | grep -v grep
```

**Remediation:**
```bash
pkill -9 -f "claude_agent_sdk/_bundled/claude"
```
Run between test iterations if you're iterating on the codebase. Already documented in `docs/runbooks/e2e-testing.md` §4g; included here so this file is a single discoverable source.

### 3.8 Orphan Playwright MCP Chrome holds the SingletonLock — **OPEN** (operational)

**Symptom:** any new Playwright MCP browser call fails with
```
Browser is already in use for /Users/.../Library/Caches/ms-playwright/mcp-chrome-XXX, use --isolated to run multiple instances
```
even though no MCP browser window is visibly open.

**Detection:**
```bash
ls -la ~/Library/Caches/ms-playwright/mcp-chrome-*/SingletonLock 2>/dev/null
# Symlink target is "<hostname>-<pid>"; check whether that PID exists:
ps -p <pid_from_symlink_target>
```

**Remediation:**
```bash
# Identify the orphan Chrome with the MCP user-data-dir:
pgrep -fa "Google Chrome.*user-data-dir=.*mcp-chrome"
# Kill it:
kill -9 <pid>
# Remove the stale lock symlink:
rm -f ~/Library/Caches/ms-playwright/mcp-chrome-*/SingletonLock
```

This is why our `scripts/lab_screenshot_tail.mjs` (see §6.2) launches its own headless Chromium via `playwright-core` directly rather than going through the MCP — it sidesteps the singleton lock entirely.

---

## 4. The "F1-F8" deployment-blocker family (reference)

Numbered fixes referenced throughout `docs/runbooks/e2e-testing.md`. Reproduced here for completeness; the runbook covers each in depth.

| ID | Commit | Subject |
|---|---|---|
| F1 | `8d534d8` | `fix(hybrid): no-bundle paper_id falls back to pure RLM` |
| F2 | `45e60df` | `fix(frontend): quiet useRdrArtifacts polling + null-render RubricBreakdown` |
| F3 | `9da8646` | `feat(paths): cross-platform input path normalization` |
| F4 | `65ee5a2` | `fix(rlm): resolve_root_model aliases (sonnet/opus → registry keys)` |
| F5 | in flight | sandbox auto-detect WSL → prefer local |
| F6 | in flight | cross-mount Claude OAuth detection on WSL |
| F7 | _post-F6_ | RDR polling resilience — proxy timeouts + broaden hook early-exit |
| F8 | _post-F7_ | chat steering + collapsible right sidebar with kind-specific node detail |

---

## 5. Test coverage gaps in `scripts/readiness.sh`

The current readiness script gates seven tiers (env, deps, static, tests, boot, run-smoke, deploy) but does not catch every failure class. Document gaps here so future contributors know what to add.

| Failure class | Does readiness catch it? | What would catch it |
|---|---|---|
| SSR↔CSR hydration mismatch (e.g., #3 above) | **No** | Playwright test that opens `/lab` and asserts no `console.error` containing "did not match" |
| `console.error` flooding from a hook | **No** | Same Playwright test, asserts error count under a threshold |
| SDK aclose deadlock with `claude-oauth` root | Partial — tier 6 would FAIL but only if `READINESS_RUN_SMOKE=1` is set, and only on the `ftrl` smoke (single paper) | Add a per-root-model smoke matrix |
| Stale OAuth credentials (revoked subscription) | No | Tier 7 already checks for credentials presence but not validity |
| Orphan SDK subprocess accumulation | No | Add `pkill -f "claude_agent_sdk/_bundled" || true` to tier 2 setup |
| Frontend bundle size regression | No | `npm run build` size diff against a baseline manifest |
| RubricStrip count-up tween jank under devtools | No (subjective) | Visual regression via Playwright + pixelmatch |

**Concrete additions to consider** (priority order):

1. Add a `tier4b: playwright /lab smoke` step that opens the page, waits 5s, fails on any React-warning or `error`-level console message.
2. Make tier 6 iterate over a matrix of `{paper_id, root_model}` rather than a single combination, so SDK-bound failures surface before deploy.
3. Add a tier 2 line that kills SDK orphans and warns if any were found (indicates a leaked run from a prior session).

---

## 6. Monitoring tools

Tools shipped in-repo for tailing a live run. Use the screenshot loop (§6.2) when you want a record; use the CLI poll (§6.1) for fast interactive checks.

### 6.1 CLI status poll (one-shot)

```bash
PROJECT=prj_xxx
.venv/bin/python - <<PY
import json, pathlib, datetime
from collections import Counter
d = json.load(open(f'runs/{PROJECT}/demo_status.json'))
p = pathlib.Path(f'runs/{PROJECT}/dashboard_events.jsonl')
events = [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
ctr = Counter(e.get('event','?') for e in events)
final = pathlib.Path(f'runs/{PROJECT}/final_report.json').exists()
print(f'status={d.get("status")} updatedAt={d.get("updatedAt","")[11:19]} final_report={final}')
print(f'events ({len(events)}): {dict(ctr)}')
last = events[-1] if events else None
if last:
    ts = last.get('timestamp','')
    age = (datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(ts.replace('Z','+00:00'))).total_seconds()
    print(f'last event @ {ts[11:19]} ({age:.0f}s ago): {last.get("event")}')
PY
```

### 6.2 Periodic screenshot tail — `scripts/lab_screenshot_tail.mjs`

Headless Chromium navigates `/lab?projectId=<id>` every N seconds, saves a full-page PNG, scrapes the wedge indicator + iteration label, captures console errors, appends a row to a TSV log. Auto-stops when the run terminates.

```bash
# From the repo root:
node /Volumes/CS_Stuff/openresearch/scripts/lab_screenshot_tail.mjs <projectId> [intervalSec]

# Default 30s, runs in foreground. To background it:
mkdir -p screenshots
nohup node scripts/lab_screenshot_tail.mjs <projectId> 30 > screenshots/tail.log 2>&1 &
```

**Artifacts produced in `screenshots/`:**
- `lab-<ISO timestamp>.png` — one screenshot per cycle.
- `console-errors-<ts>.json` — flushed if any errors landed in the cycle.
- `wedge-log.tsv` — append-only `ts | no_signal_secs | iteration | status | notes`.
- `tail.log` — stdout: human-readable summary including the `⚠ no_signal=Xs` flag when triggered.

**Stop conditions:** `runs/<id>/final_report.json` exists, OR `runs/<id>/demo_status.json::status` is `failed` or `completed`. The script self-terminates cleanly in either case.

**Implementation note:** the script resolves `playwright-core` from `frontend/node_modules` via an explicit path import. Node's ESM resolver does not honor `NODE_PATH`, and `@playwright/test` is not installed at the repo root. If you ever move the script, update the path computation.

### 6.3 Cycle-correlated stderr tail

```bash
PROJECT=prj_xxx
# Watch stderr alongside the events stream:
tail -F runs/$PROJECT/runner.stderr.log runs/$PROJECT/dashboard_events.jsonl
```

`aclose()` lines in stderr are now non-fatal (see §3.1). Genuine failures look like `RuntimeError: Pipeline exited with status N`, `FileNotFoundError`, `Error code: 401`, etc. — search those patterns specifically.

### 6.4 Run-state at a glance — single command

```bash
PROJECT=prj_xxx
echo "=== $(date +%H:%M:%S) ===" && \
.venv/bin/python -c "
import json, pathlib
d = json.load(open('runs/$PROJECT/demo_status.json'))
p = pathlib.Path('runs/$PROJECT/dashboard_events.jsonl')
events = [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
from collections import Counter
print(f'status={d[\"status\"]} final={pathlib.Path(\"runs/$PROJECT/final_report.json\").exists()}')
print(f'events: {dict(Counter(e[\"event\"] for e in events))}')
" && \
echo "code/ last modified:" && \
ls -lt runs/$PROJECT/code/ 2>/dev/null | head -3 && \
echo "SDK workers: $(pgrep -fa 'claude_agent_sdk/_bundled' | wc -l)"
```

---

## 7. Triage flowchart

```
                  ┌──────────────────────────┐
                  │ User reports "stuck"     │
                  └──────────────┬───────────┘
                                 │
              ┌──────────────────▼─────────────────┐
              │ Read runs/<id>/demo_status.json    │
              │   → status field                   │
              └──────────────────┬─────────────────┘
                                 │
        ┌────────────────────────┼─────────────────────────┐
        │                        │                         │
   status=failed           status=running             status=queued
        │                        │                         │
        ▼                        ▼                         ▼
 Read runner.stderr.log    Read dashboard_events.jsonl   Has subprocess
   → look for traceback     → time since last event       spawned?
   → see §3.1 (aclose),                                    │
     CLAUDE.md auth,                                       ▼
     §4 F1-F8 fixes                                Check ps -fa
                              ┌─────────────────────┐    runner.cli
                              │ event_age < 60s     │      |
                              │ → healthy, wait     │      ▼
                              │                     │   no → log subprocess
                              │ event_age > 60s     │   stderr; run died
                              │ → check §3.4        │   at startup
                              │   (long primitive   │
                              │    in flight?)      │
                              │ → check code/       │
                              │   mtime within 2m   │
                              └─────────────────────┘
                                 │
                          ┌──────┴──────┐
                          │             │
                       working      truly wedged
                       (write       (no file touches,
                        files,       workers idle,
                        workers      stderr shows
                        alive)       new aclose)
                          │             │
                          ▼             ▼
                    let it run    pkill + restart
                                  with §3.7 cleanup
```

---

## 8. When you discover something new

1. Add an entry to §0 status board (id, severity, status, location).
2. Add a numbered section under §3 (active issues) with symptom / root cause / fix or workaround / verification.
3. If a readiness gate doesn't catch the failure, add a row to §5.
4. If you ship a fix, change the status board cell from "open" to "resolved YYYY-MM-DD" and add the fix location.
5. Keep the cross-references between this doc, `e2e-testing.md`, `readiness.md`, and `CLAUDE.md` consistent. None of these should contradict the others. When they do, this doc is canonical for "what fails and why."

---

## 9. Cross-references

- **Architecture overview:** `system_overview.md`, `CLAUDE.md`
- **UI walkthrough + happy path:** `docs/runbooks/e2e-testing.md`
- **Preflight gates:** `docs/runbooks/readiness.md`
- **RLM pivot rationale:** `docs/design/rlm-pivot-brief.md`
- **Locked launch decisions:** `docs/superpowers/specs/2026-05-23-cleanup-condensation-leaderboard-design.md`
- **SDK aclose deep-dive (Defect 1 + 2):** `docs/superpowers/specs/2026-05-22-sdk-aclose-investigation.md`
- **Source of truth for events:** `runs/<project_id>/dashboard_events.jsonl`
- **Source of truth for status snapshot:** `runs/<project_id>/demo_status.json` (lossy; see §3.6)
- **SSE egress chokepoint:** `backend/agents/rlm/sse_bridge.py`
