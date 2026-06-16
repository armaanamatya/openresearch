<!-- doc-meta: status=archived; archived=2026-06-03; superseded-by=docs/superpowers/specs/ -->
> ⚠️ **ARCHIVED — historical UI-work log, not current-state documentation.**
> Frozen 2026-06-03 (last entry 2026-05-23). For the current UI architecture see
> [`README.md`](../../README.md) (UI Pages) and
> [`CLAUDE.md`](../../CLAUDE.md) ("UI surfaces"). Kept for provenance only.

# UI Progress Log

_Append-only log of UI-facing work. Newest first. Each entry: **what shipped → why → guardrail**. Distill failure modes here so the next session doesn't re-learn them._

---

## How to read / extend this doc

- **Format**: one entry per landed PR/commit cluster, dated, with a fixed shape (`Shipped` / `Why it broke` / `Root cause` / `Fix` / `Guardrail`).
- **Append-only**: never rewrite past entries; if a fix later regresses, add a NEW entry that references the old one.
- **Token-thrifty**: one-liners over paragraphs. Code snippets only when the rule is non-obvious. Tables for parallel facts.
- **No status theatre**: only landed work. WIP belongs in `docs/superpowers/plans/`.
- **Cross-link**: `(see learn.md §X)` for full RCAs, `CLAUDE.md` for live architecture, `docs/runbooks/e2e-testing.md` for ops.

---

## 2026-05-23 (late evening) — Parallel 4-paper sweep + 6 reliability commits (`19e87ee…f8546d1`)

User goal: "fan out sub agents, try multiple papers in parallel (last 6 months ML), debug
relentlessly until reproduction-with-improvements works end to end. /iterate, no codex.
Opus plans+reviews everything, Sonnet executes. Make every fix work for both API and OAuth."

### Shipped (6 commits)

| # | SHA | Subject |
|---|---|---|
| 1 | `19e87ee` | The 'stuck Running' bug — runs that finish now actually flip to Completed |
| 2 | `7970506` | Parallel paper sweep was killing the second ingest with 'database is locked' — BEGIN IMMEDIATE + 30s busy_timeout fixes it |
| 3 | `1f72e07` | Promoted-candidate gate stops getting blocked by a wire-contract bug — candidate_id=None corrupted every outcome event |
| 4 | `9dd7c6d` | The root was declining every candidate to 'save cost' — now told to try a scoped-down subset before declining anything |
| 5 | `991517a` | Leaderboard stopped 500ing on legacy final_report shapes — defensive coerce to {} for the four header fields |
| 6 | `f8546d1` | run_experiment cap was 2 hours — B2 of the paper sweep wedged for it; now 30 min with env-var escape hatch |

### Paper-sweep results (4 papers, OAuth surface, docker sandbox)

| Run | Paper | Iters | Rubric | Promoted | Verdict |
|---|---|---|---|---|---|
| A1 | 2512.24601 (RLM) | 3 | 9.6% | 0/3 | partial (old prompt) |
| A2 | 2512.18131 (LLM CodeGen) | 3 | 14.4% | 0/3 | partial (old prompt) |
| **B1** | **2602.01785 (CodeOCR)** | **6** | **31.88%** | **1/3 ✓** | **partial (new prompt, hit gate)** |
| B2 | 2602.17186 (Visual Info Gain) | 2 | n/a | n/a | stopped manually (CPU-bound VLM training looped) |

**B1 hit the user's success gate.** First E2E run to produce a verified `outcome=promoted` candidate, validating the full chain (stuck-Running fix + SQLite fix + candidate_id wire fix + anti-decline prompt) end-to-end.

### New failure modes recorded (F18–F23)

| # | Symptom | Root cause | Rule |
|---|---|---|---|
| F18 | `prj_6b9acbfd8afcd789` reached "partial" rubric 0.244 + wrote `final_report.json`, but `demo_status.json` stuck on `running` forever | Demo wrapper template called `finalize_benchmark()` then `write_status("completed")`. atexit cleanup hit claude-agent-sdk's `subprocess.wait()` which hangs on WSL2 after SIGKILL — wrapper never reached `write_status` line | Side-effect ordering at the wrapper layer: **the visible state transition must come BEFORE any post-pipeline computation that could hang**. Also add `os._exit(...)` in `finally:` to bypass atexit. Pin via wrapper-string compile test (`tests/services/events/test_live_runs_status_ordering.py`) |
| F19 | Triggering 2 `/runs/arxiv` in parallel killed the second ingest at SQLite write: `AppendError: SQLite error during append: database is locked` | Writers used `BEGIN` (= BEGIN DEFERRED), upgraded SHARED→RESERVED on first INSERT under WAL. Upgrade can SQLITE_BUSY without honoring `busy_timeout` | All write paths: `BEGIN IMMEDIATE` (acquires RESERVED upfront, serializes cleanly under busy_timeout). Also bumped `busy_timeout` 5s → 30s. Pinned via 3 concurrency tests in `tests/test_eventstore_sqlite_concurrent.py` |
| F20 | Every `candidate_outcome` SSE event had `candidate_id="None"` — UI couldn't match outcomes to proposed candidates → promoted-candidate gate unreachable | `record_candidate_outcome` accepted `None`; `binding.py` ran `str(None)` → string `"None"`; system prompt didn't tell root WHICH IDs to use | Defense in depth: (1) primitive validates `candidate_id` (rejects `None`/`'None'`/`'null'`/`''`); (2) binding skips emit unless `success=True` + both fields present; (3) system_prompt explicitly says "use the `id` from the most recent `propose_improvements` result". 4 new tests in `test_binding.py` |
| F21 | Root model declined every improvement candidate citing "save cost" — never tried even a scoped-down version | System prompt rewarded declining: "A candidate declined early saves Docker build time, experiment wall-clock, and LLM cost. The goal is a verified reproduction, not exhaustive exploration." | Anti-bias framing: "Success target is at least one PROMOTED candidate per run, not 'every candidate declined for cost reasons'. If all candidates look too big, IMPLEMENT A SCOPED-DOWN SUBSET." Validated end-to-end: B1 hit the gate on first run with the new prompt |
| F22 | `GET /leaderboard` returned HTTP 500 `'list' object has no attribute 'get'` | 3 legacy fixtures had `rubric` as list-of-areas (old shape). Route did `rubric = data.get('rubric') or {}` — list is truthy so it stayed, next `.get()` crashed. One bad file killed the whole endpoint | Defensive coerce at the read boundary: `_as_dict(v)` helper applied to all dict-shaped header fields. Single bad row gets None score + defaults; aggregation continues. Pinned via 1 new test in `test_leaderboard_http.py`. Also archived the 3 legacy fixtures to `runs/_archived_legacy_fixtures_20260523/` |
| F23 | B2's `run_experiment` wedged for ~30 min on a CPU-bound VLM training that the model wrote in iter 2. Default aggregate cap was 7200 s (2h) — pipeline would have waited the full 2 hours | Cap chosen for multi-command experiments at 1hr-per-command × 2 commands. Way too long for the common case where a single bad train.py spins forever on CPU | Default cap reduced 7200s → 1800s (30 min). New env var `OPENRESEARCH_RUN_EXPERIMENT_TIMEOUT_S` for callers who genuinely need longer. Invalid values fall back silently. Pinned via 3 new tests in `test_run_experiment_timeout.py` |

### Architecture additions

- **Wrapper template invariant**: `write_status("completed")` MUST appear before `finalize_benchmark()` in the success branch. `finally:` MUST end with `os._exit(0 if completed else 1)` to bypass atexit. Compiled-string tests guarantee no future refactor regresses these (5 tests).
- **SQLite write contract**: every transaction starts with `BEGIN IMMEDIATE`, never bare `BEGIN`. 30s busy_timeout. This is the entire concurrency story — no per-table locks, no manual retry, no application-level mutex.
- **Candidate-outcome wire contract**: emitter REFUSES to publish events with empty/None/'None' candidate_id. Validation lives at THREE layers (primitive returns success=False, binding skips emit, prompt instructs). Belt + suspenders + 3rd belt.
- **Leaderboard defense pattern**: `_as_dict(v)` is the right shape for read-time defensive coercion — keeps the row but blanks the broken field. Better than dropping the row (operator loses visibility).
- **CPU sandbox + agent baseline mismatch**: B2 demonstrated that the implement_baseline agent does NOT know it's running in a CPU sandbox. Future fix: pass `ctx.sandbox_mode` into the prompt so the agent picks `--smoke-test` automatically. Tracked.

### Manual-intervention runbook (added)

When a docker-sandbox `run_experiment` wedges on CPU-bound training:
```bash
CID=$(docker ps --format '{{.Names}}' | grep prj_<id>)
# Container-namespace PIDs (host PIDs from `docker top` are NOT usable from inside)
docker exec $CID ls /proc | grep -E '^[0-9]+$' | sort -n
# Kill via container-namespace PID (typically sh=7, python=16)
docker exec $CID python -c "import os, signal; [os.kill(p, signal.SIGKILL) for p in (7, 16)]"
# Pipeline will see run_experiment failure within ~5 s and iterate
```
Also: `DELETE /runs/{project_id}` cleanly stops a runaway run; status flips to `stopped`.

### Best practices distilled from this sprint

1. **Side-effects in order**: visible state changes BEFORE cleanup. If a "finalize" step can hang, write the "completed" status first.
2. **`os._exit(0)` in `finally:` for any subprocess wrapper** that spawns third-party processes that might leak past atexit. The 2-hour wait is a real failure mode.
3. **Defensive coercion at READ boundaries beats defensive write contracts**. `_as_dict(v)` survives any upstream data shape regression. Tests pin happy path AND each known-bad shape.
4. **Wire-contract bugs need defense at every seam they touch**. candidate_id=None hit 3 layers; the fix lives in all 3.
5. **Prompt biases are real bugs** — "save cost" framing in a system prompt was the entire reason B1's 3 candidates went 0/3 promoted with the old prompt and 1/3 promoted with the new one.
6. **Container debugging is namespace-scoped**: PIDs from `docker top` (host view) are NOT the PIDs you `os.kill` from inside the container (container PID namespace).
7. **Per-step timeouts must be sized for the worst common case, not the worst conceivable case**: 7200s was paranoid; 1800s covers 99% of real experiments and surfaces regressions in time to iterate.
8. **API/OAuth parity is "below the auth layer" by construction**: today's 6 fixes all sit at wrapper template / eventstore / data validation / prompt — auth-agnostic. Verify by grepping for provider branches in the edit surface.

### Known issues (not blocking, tracked for next session)

- **Zombie Claude subprocesses on long runs**: every Sonnet sub-call leaks a defunct child on WSL2 (SDK atexit subprocess.wait hangs after SIGKILL). B2 accumulated 72. Doesn't affect correctness but file-descriptor limits could matter eventually. Tracked.
- **implement_baseline agent doesn't know sandbox mode**: B2's repeated CPU-infeasible baselines are the symptom. Fix candidate: thread `ctx.sandbox_mode` into the agent's prompt so it picks `--smoke-test` automatically when sandbox is CPU.
- **Codex-companion plugin auto-invoked from sub-agent**: `/home/abheekp/.claude/plugins/cache/openai-codex/codex/1.0.3/scripts/codex-companion.mjs` ran for ~30 min during A2's implement_baseline. This is the system-being-tested's choice (via claude-agent-sdk plugin system), not Claude Code's agent dispatch. If "no codex" must extend to the sub-agent layer, uninstall the plugin or filter at the SDK layer.

### Live URLs from this session (operator can inspect)

- A1 (RLM, completed): http://localhost:3000/lab?projectId=prj_f4cc5fa917c27ef1
- A2 (LLM CodeGen, completed): http://localhost:3000/lab?projectId=prj_390202710d0f994b
- **B1 (CodeOCR, completed, promoted ✓)**: http://localhost:3000/lab?projectId=prj_7b7b34eb9d623b75
- B2 (Visual Info Gain, stopped): http://localhost:3000/lab?projectId=prj_77b7294aed1bf872
- Leaderboard: http://localhost:3000/leaderboard

### Test count check

`tests/rlm/ tests/rdr/ tests/routes/ tests/services/events/ tests/agents/ tests/test_eventstore_sqlite{,_concurrent}.py`: **729 passed, 1 xfailed** after all 6 commits. Zero regressions across today's edit surface.

---

## 2026-05-23 (evening) — Schema-coercion + failed-run-UX + doc maintenance (`b290449…f624d33`)

### Shipped (5 commits)

| # | SHA | Subject |
|---|---|---|
| 1 | `b290449` | A 30-minute run was crashing at second-39 because paper_claims came back as a list |
| 2 | `8f9b07a` | Clicking a failed run no longer shows a blank lab — it shows the failure and offers a one-click rerun |
| 3 | `c930117` | CHANGELOG: today's session — 4 sections (Added / Changed / Fixed / Documentation) |
| 4 | `4c1d2bb` | learn.md: three new postmortems — paper_claims coercion, aclose visibility, terminal-state UI seeding |
| 5 | `f624d33` | progress.md: 2026-05-23 reliability + production-path sprint — current status above the older 05-22 entry |

### New failure modes recorded (F15–F17)

| # | Symptom | Root cause | Rule |
|---|---|---|---|
| F15 | 30-min runs crashed at `_finalize` with Pydantic ValidationError on `paper_claims` | Schema `dict`-only; root sometimes returns `list[dict]` (perfectly reasonable shape) | Schemas at long-computation seams: liberal about shape, strict about types. `@field_validator(mode="before")` coercer keyed by identity field, fallback to index |
| F16 | Clicking failed run in Recent → blank lab UI | `useRun` opens SSE only for queued/running; for terminal runs `dashboardEvents` stayed `[]`, tree empty, no error shown | When live channel is status-gated, the static snapshot (`/api/demo?projectId=` returns `payload.events`) must seed the reducer for terminal runs |
| F17 | No fast path to re-run a failed paper | Required navigate-back-to-upload + repick | `POST /runs/<id>/rerun` reads `demo_status.json::sourcePdf.runPath` and delegates to the same upload codepath; `useRerun` hook owns the POST + router.replace |

### Architecture additions

- **`useRerun(projectId)`** is reusable — future surfaces (Recent right-click, leaderboard "re-evaluate" button, CLI dashboard) can use the same hook.
- **`LiveDemoRunPayload`** type narrows the previously-too-loose `payload: null` → `LiveDemoRunPayload | null` with explicit `events?: unknown[]`. The terminal-state seed reads from there.
- **Rerun endpoint NEVER reuses the old project_id** — every rerun is a fresh `prj_*`. Old run state preserved for postmortem.

### Live E2E status

`http://localhost:3000/lab?projectId=prj_6b9acbfd8afcd789` (this session's run-after-schema-fix).
Earlier run `prj_d118333894223202` is the one that crashed at finalize (now the failed-state UI gives it a visible error banner + Rerun button instead of blank).

---

## 2026-05-23 (afternoon) — Stability + behavior-quality + Azure-path sprint (`30d243f…ad460ff`)

### Shipped (8 commits, in dependency order)

| # | SHA | Subject | Lines |
|---|---|---|---|
| 1 | `30d243f → 1316b8a` | Rubric area key contract aligned — no more duplicate empty `key=""` React warnings | +95/-14 |
| 2 | `f7c43c1` | uiprogress.md created as the regression-prevention log | +134 |
| 3 | `0cc2e77` | SSR-safe elapsed clock — `nowMs=null` until mount kills the hydration mismatch | +13/-3 |
| 4 | `ad32198` | Upload nav item above Lab/Library — one-shot `?new=1` reset | +72/-1 |
| 5 | `c6511be` | Aggregate counter strip + enriched `subrlm`/`baseline` node detail in sidebar | +343/-8 |
| 6 | `9d7f8e9` | Counter strip layout pinned + wrap_primitive arg coercion + rlm_query nudge | +157/-4 |
| 7 | `8c3371e` | The root was ignoring rlm_query — primitives now ask for it themselves on big slices | +229/-17 |
| 8 | `4e3bd38` | Two stability levers so SDK-deadlock runs stop pretending to be alive | (W1a + W1b) |
| 9 | `ad460ff` | Azure OpenAI joins the root-model lineup — the path to Azure deployment opens | (W2c) |

### New surfaces (afternoon)

| Feature | Component / endpoint | Source of truth |
|---|---|---|
| `heartbeat(note)` primitive (13th) | `backend/agents/rlm/primitives.py` | System prompt instructs root to call before any long op |
| `iteration_heartbeat` SSE event | `sse_bridge.build_iteration_heartbeat_event` | Reducer stores `lastHeartbeatAt` on RlmRunState |
| Stderr watchdog (asyncio task per run) | `backend/services/events/live_runs.py:_stderr_watchdog` | Detects `aclose()` loop ≥3× in 30s |
| `run_warning` SSE event | `sse_bridge.build_run_warning_event` | Reducer accumulates into `warnings[]` |
| `degraded: True` + `degraded_reason` on demo_status.json | watchdog atomic write | UI surfaces via run_warning chip |
| Amber "no signal Ns" chip | `rlm-header.tsx` | When running AND last heartbeat >60s stale |
| Red warning chip | `rlm-header.tsx` | When `warnings.length > 0` |
| `azure-gpt-4o` root model + aliases (`azure`, `azure-openai`, `gpt-4o-azure`) | `models.py` + `factory.py` + `run.py` | Reads AZURE_OPENAI_API_KEY/ENDPOINT/DEPLOYMENT |
| `AzureOpenAILlmClient` (primitives-layer wrapper) | `backend/services/context/workspace/tools/azure_openai_client.py` | Wraps `openai.AzureOpenAI` for the primitive `complete(system=, user=)` protocol |
| In-band `_meta.hint` on understand_section / extract_hyperparameters | `primitives.py` | Threshold 10K chars; result_summary prefixed `[hint]`; sidebar amber dot |
| Aggregate counter strip (iterations / primitive calls / proposed / promoted) | `node-detail-sidebar.tsx` | Always-visible in expanded sidebar |
| Enriched `subrlm` panel | `node-detail-sidebar.tsx` | Matches `subrlm-N` → `subRlms[N-1]`; shows model + depth + duration + prompt_preview |
| Enriched `baseline` panel | `node-detail-sidebar.tsx` | Rubric score + pass/partial/fail chip |
| Real-time elapsed clock | `rlm-lab.tsx` | `startedAt` + `setInterval(1000)` — SSR-safe via `nowMs=null` initial state |
| Upload nav item (first in sidebar) | `lab-sidebar.tsx` | One-shot `?new=1` clears localStorage + sets upload view |

### Failures we burned time on (F9–F14)

| # | Symptom seen | Root cause | Rule for next time |
|---|---|---|---|
| F9 | 6 React duplicate-key `=""` chips in rubric strip + breakdown | Silent key mismatch: `_rubric_areas` emitted `"name"`, `binding.py` read `"area"` → both became `""` for blank-requirement rubrics | Pin wire-contract field names at the seam test. **Defensive React keys**: always `key={x.id \|\| `__idx_${i}`}` — never trust upstream non-empty |
| F10 | SSR hydration mismatch on elapsed tile (24m 16s vs 24m 21s) | `useState(() => Date.now())` ran server-side and client-side with different values | Client-side dynamic values (Date.now / Math.random / locale-formatted) MUST be initialized to `null` and populated in useEffect — server + first-client render must match |
| F11 | Sidebar counter strip values ran into labels ("19primitive calls") | CSS missing `display: block; white-space: nowrap` on counter value + label spans; tile lacked min-width/overflow guards | Always pin tile geometry: `flex-shrink: 0; min-width: 0; overflow: hidden` and explicit `display: block` on stacked value/label |
| F12 | Backend wedged in `do_wai`; UI showed running + ticking elapsed indefinitely (no signal to user that SDK was dead) | aclose deadlock blocks repl_iteration emission but leaves the worker alive | **Heartbeat primitive** (root calls between ops) + **stderr watchdog** (detects aclose loop pattern) + UI chips for both. Operator now sees the difference between "thinking", "no signal", and "SDK wedged". |
| F13 | Root underused `rlm_query` for big slices (sometimes 0 sub-RLM spawns on a 9MB paper) | System-prompt-only nudges fade after a few iterations; model attention drops on early prompt | **In-band feedback** > upfront prompting. Primitives now return `_meta.hint` on >10K slices that travels via `result_summary` — root re-encounters the nudge every iteration |
| F14 | Run subprocess accumulated transient ValidationErrors costing root iterations on schema-repair | wrap_primitive raised immediately on type mismatch with no coercion attempt | `wrap_primitive` now attempts ONE conservative coercion pass (int↔str-of-digits, scalar→str when str expected); never coerces dicts (would invent values) — events carry `coerced: bool` so the operator sees when it fired |

### Architecture decisions worth preserving (afternoon additions)

- **Heartbeat is observational, not enforcing.** It does not abort or retry — it just emits a signal. The operator decides what to do. RLM autonomy intact.
- **Watchdog flags but never kills.** When `sdk_aclose_loop` is detected, the run subprocess keeps running (it may still progress; aclose RuntimeError is post-call generator cleanup noise). The user gets the warning and can `pkill` if they want.
- **In-band hints carry through SSE.** A `[hint] ` prefix on `result_summary` is the seam — backend writes it, frontend recognizes it. Don't introduce a separate "hints" field; cheap embedding is enough and survives schema evolution.
- **Azure provider plugs through the same abstraction as the others.** No conditional code in the orchestrator; the dispatch is purely on `root_model.rlm_backend`. Adding a 5th provider (Bedrock, Gemini, etc.) follows the same shape.

### Files-of-record (afternoon additions)

| File | Owns | Don't break |
|---|---|---|
| `backend/agents/rlm/primitives.py:heartbeat` | The "alive signal" primitive | Must remain side-effect-free; only emits one event |
| `backend/services/events/live_runs.py:_stderr_watchdog` | Aclose-loop detection | Must NEVER terminate the subprocess; idempotent (flag once per run) |
| `backend/services/context/workspace/tools/azure_openai_client.py` | Primitive-layer Azure wrapper | Mirror the `LlmClient` protocol used by Anthropic + OpenAI wrappers |
| `backend/agents/rlm/models.py:_inject_azure_kwargs` | Azure endpoint/deployment env injection | Both AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_DEPLOYMENT required; raise ValueError if either missing |
| `frontend/src/components/lab/rlm/rlm-header.tsx` | Status + heartbeat-staleness + warnings chips | Three distinct chip states must coexist visually |

### Production-readiness scoring (May 2026)

| Goal | Status | Notes |
|---|---|---|
| Single-user local dev | ✅ Excellent | All recent work refines this |
| OAuth subscription path | ✅ Works | F12 detection makes hangs survivable |
| Local GPU sandbox | ✅ Works | F12 + F13 make demos more reliable |
| Azure OpenAI provider | ✅ Wired | W2c — backend abstraction ready; needs end-to-end test on real Azure deployment |
| Vercel frontend hosting | 🟡 Works as static / proxy | Confirmed since the frontend is plain Next.js 16 |
| Vercel backend hosting | ❌ Not viable | File-state + long-lived subprocesses + 30+ min SSE conflict with Vercel Functions; needs Azure Container Apps or similar (W2 full migration) |
| Multi-tenant SaaS | ❌ Not yet | W5 (per-user auth boundary) + W3 (distributed state) outstanding |
| Concurrent run scaling | 🟡 Single-worker uvicorn | W2 (gunicorn + N workers) is 2 days work |
| Telemetry stack | ❌ Not yet | W10 outstanding |
| Cost ledger | ⚠️ API-key only | W6 — OAuth doesn't surface tokens; documented limitation |

### Open / deferred (afternoon additions)

- **W4 — Tool router primitive**: queued; conflicts with W1a's primitives.py edit; do next turn after rebase
- **Subprocess restart-on-degraded recovery**: only ship if E2E loop shows we're losing too many runs to aclose. Heartbeat + watchdog detection is the prerequisite; recovery is a separate decision.
- **Azure real-deployment smoke test**: wire works in tests; needs one actual `AZURE_OPENAI_API_KEY` smoke run before prod cutover.
- **`degraded` field plumb through `/runs/<id>` response**: backend writes it to demo_status.json, but `LiveDemoRunState` pydantic excludes it. Cosmetic; UI gets the warning via SSE event regardless. Patch later if the UI needs to render the chip on a refresh-without-SSE.

### Best practices distilled (additions)

9. **Detect, don't enforce.** Observational telemetry (heartbeat, watchdog) preserves autonomy. Save enforcement (timeouts, kills) for cases where the cost of inaction is catastrophic.
10. **Wire contracts are seams; field names cross them.** Every cross-process payload field gets at least one contract test that pins its exact name. F9 cost an hour because we didn't.
11. **Hydration mismatches are silent until prod.** Any dynamic value (Date.now, locale, random) in a client component MUST defer to useEffect; initial state must be SSR-stable.
12. **In-band feedback beats upfront prompting for frontier models.** When you want the model to learn a behavior, attach the hint to the *result* of the operation you want it to change, not the system prompt header.

### Live-link smoke after this sprint

`http://localhost:3000/lab?projectId=prj_d118333894223202` (active run) — shows:
- `● running` + `iteration N` (heartbeat-driven)
- Real-time elapsed clock (SSR-safe)
- Tree: Paper → Comprehension → Environment (work) → sub-RLM (depth=1)
- Sidebar counters: `iterations / primitive_calls / proposed / promoted` (not crushed)
- Red `run_warning` chip with "Backend SDK aclose deadlock detected — run is degraded; consider restarting"
- No console errors, no duplicate keys, no hydration mismatch

---

## 2026-05-23 — Lab UI hardening sprint (commits `78f0870…1316b8a`)

### Shipped (7 commits, in dependency order)

| # | SHA | Subject | Lines |
|---|---|---|---|
| 1 | `78f0870` | RDR polling resilience — proxy timeouts + broader hook early-exit | +99/-31 |
| 2 | `a07c336` | Chat steering + collapsible right sidebar with kind-specific node detail | +1.4k/-83 |
| 3 | `f664659` | Runbook + CLAUDE.md + system_overview + start_backend.sh | +134/-11 |
| 4 | `290c4d0` | Preset arXiv chip row + API/OAuth chat-steering parity test | +302/0 |
| 5 | `e4c30a0` | Live status flip on primitive_call + real-time elapsed clock + RDR empty-200 stop | +90/-8 |
| 6 | `30d243f` (rebased to `1316b8a`) | Align rubric_area key contract + defensive empty-name handling | +95/-14 |

### New surfaces

| Feature | Component / endpoint | Source of truth |
|---|---|---|
| Right-docked node-detail sidebar (360px / 36px collapsed) | `components/lab/rlm/node-detail-sidebar.tsx` | Lifted `selectedNodeId` in `rlm-lab.tsx` |
| Steering chat panel (bottom of sidebar) | `components/lab/rlm/steering-chat.tsx` + `hooks/use-steering-chat.ts` | SSE events `user_message` / `user_message_response` |
| Backend chat endpoint | `POST /runs/<id>/messages` | `backend/routes/messages.py` |
| RLM primitives `check_user_messages()` / `respond_to_user()` | `backend/agents/rlm/primitives.py` | System prompt instructs polling at iteration start |
| Preset arXiv chip row | `components/lab/upload-view.tsx` | `lib/demo/preset-papers.ts` (4 papers) |
| Real-time elapsed clock | `components/lab/rlm/rlm-lab.tsx` | `runMeta.startedAt` + `setInterval(1000)` |
| RDR polling early-exit on empty | `hooks/use-rdr-artifacts.ts` | `isMissing(res, arrKey)` helper |
| New SSE event types | `lib/events/rlm-events.ts` | `user_message`, `user_message_response` |

### Failures we burned time on (and the rule they leave behind)

| # | Symptom seen | Root cause | Rule for next time |
|---|---|---|---|
| F1 | Lab UI stuck "Queued" + "0s elapsed" while backend run was clearly progressing | `useRlmRun.foldPrimitiveCall` only flipped status to "running" on `repl_iteration` events; SDK aclose deadlock blocked iteration completion → no event → no flip. Elapsed used event-span (also 0 with 0-1 events). | **Status/elapsed must derive from the earliest possible signal**: status = first event of any kind; elapsed = `startedAt` + ticking clock. Never gate UI liveness on a single optional event type. |
| F2 | Endless 404 spam on `/leaf-scores`, no convergence | F2-era fix stopped polling on 3× consecutive 404, but `/clusters` and `/repair-iterations` return 200 with empty arrays for non-RDR runs → `allMissing` stayed false. | **"Missing" means missing semantically, not just HTTP-404**: `status === 0 || status >= 400 || (status === 200 && empty array)`. Use a helper, not an inline expr. |
| F3 | 502 Bad Gateway on `/clusters` while backend hung | Proxy `fetch` had no timeout; Node's default ~5 min elapsed before the catch-all returned 502. | **Every proxy fetch needs an `AbortController` with a tight timeout (4s)**. On timeout/error, return a semantically-correct status (404 = "no data") not 502, so callers can converge. |
| F4 | React "duplicate key=`""`" warnings on rubric chip/breakdown rows (6 chips, all empty) | `_rubric_areas()` emitted `{"name": ...}` but `binding.py` read `a.get("area", "")` — silent key mismatch + auto-generated rubric had blank `requirements`. | **Pin wire-contract field names at the seam test**. Whenever a payload crosses a process boundary (primitive → SSE → React), add ONE contract test that asserts the exact field name. Defensive React keys: always `key={x.id \|\| `__idx_${i}`}` — never trust upstream to be non-empty. |
| F5 | `NodeDetailSidebar` was dead code (component existed but not wired into rlm-lab) | Sonnet sub-agent reported "all gaps closed" but had only created files, not integrated them. | **Trust the diff, not the summary**: every sub-agent claim is verified by direct file inspection. The probe agent that grep'd for the import caught it. Add `grep <import-name> <consumer-file>` to every sidebar/integration acceptance check. |
| F6 | Sonnet C marked work complete while tests for new components didn't exist | Async dispatch + "Already complete" status was a false signal. | **Acceptance criteria must include a verifiable command** (`ls test_file.tsx`). "Tests added" without the file present = task not done. |
| F7 | 4 tests pinned the primitive count = 10 and broke when we added `check_user_messages` + `respond_to_user` | Brittle pin tests freezing a constant that legitimately changes. | **Test contracts, not constants.** Prefer "every primitive resolves to a callable" over "exactly N primitives exist". Pinning the count is a regression magnet. |
| F8 | Backend hung in `do_wai` syscall, all endpoints timed out, run state file said "running" | SDK `aclose()` nested-generator deadlock under WSL2 futex. Workaround B exists in `claude_oauth_client.py` + `rlm_query.py` but doesn't catch every code path. | **UI must be resilient to backend hangs** (F1+F2+F3 jointly do this). Don't fix the backend by making the UI fragile. Document the kill+restart recipe in runbook §4f. |

### Architecture decisions worth preserving

- **Selection state lives in `rlm-lab.tsx`**, not in `ExplorationCanvas`. The canvas signals via `onSelectNode(id)`, the sidebar consumes the same `selectedNode` ref. One source of truth, no prop-drilling cascades.
- **Chat is pure file I/O** at the primitive layer (`user_messages.jsonl` + cursor). Works identically under API-key and OAuth auth. Auth-surface-agnostic; lock-in via `tests/rlm/test_chat_steering_auth_parity.py` (6/6 pass).
- **Sidebar collapse state is internal to the component** (`useState`), grid track stays a fixed `360px`. Don't lift collapse state up — the parent doesn't care, and lifting creates re-render cascades.
- **`runMeta.startedAt`** is the canonical clock anchor. Event-span elapsed is a legacy fallback for fixtures/replays only.
- **`isMissing(res, arrKey)`** is the canonical "no artifact" predicate for any polling hook. Don't reinvent the check inline — factor it.

### Files-of-record (don't break these without an entry here)

| File | Owns | Don't break |
|---|---|---|
| `lib/events/rlm-events.ts` | SSE event type union + `RLM_EVENT_TYPES` | Compile-time `_RlmEventTypesInSync` guard enforces the union ↔ array match |
| `hooks/use-rlm-run.ts:fold()` | Pure reducer; exhaustive switch over `RlmDashboardEvent["event"]` | Adding a new event type requires a switch case (default = no-op is fine; absence = TS error) |
| `hooks/use-rdr-artifacts.ts` | RDR artifact polling + early-exit | Counter MUST stay monotonic; "missing" predicate MUST include empty 200 |
| `components/lab/rlm/node-detail-sidebar.tsx` | Kind-dispatch for node detail (paper / work / candidate / subrlm / baseline / declined-group) | All 6 `TreeNode["kind"]` values must have a branch or a fallback |
| `app/api/demo/runs/[projectId]/{clusters,leaf-scores,repair-iterations}/route.ts` | Polling proxies | All three MUST have a 4s `AbortController` + normalize timeout/5xx → 404 |
| `backend/agents/rlm/binding.py:_emit_extra` | Event payload construction at the seam | Defensive key reads: `a.get("area") or a.get("name") or ""` survives upstream rename |

### Auth-mode parity invariants (locked in by `test_chat_steering_auth_parity.py`)

1. System prompt build with `["claude", "claude-oauth"]` must always include `check_user_messages` AND `respond_to_user`.
2. Primitive registry binds both new primitives under either alias.
3. `POST /runs/<id>/messages` MUST never gate on root-model config.
4. Both new primitives execute with zero LLM credentials present (file I/O only).

**Implication for prod cutover (API key)**: nothing in the chat surface needs changing. Just set `ANTHROPIC_API_KEY` with credits and pick `--model claude`.

### Open / deferred (don't claim "done")

- **SDK aclose deadlock**: documented (runbook §4f) but not fixed. UI is resilient; backend still needs upstream-SDK fix OR a primitive-side `repl_iteration` heartbeat track.
- **Cost ledger reports $0 for OAuth runs**: SDK doesn't surface token counts. Documented in CLAUDE.md. Use `max_wall_clock` not `max_usd` for OAuth.
- **Frontend vitest blocked on Node 21**: needs Node 22.12+. Document & enforce via `package.json` engines OR a preflight in `start_backend.sh`. (User has 21.7.3 in their shell; coworkers will hit this.)
- **Sub-RLM nodes**: the tree builds them from `sub_rlm_spawned` events, but the root model doesn't always spawn sub-RLMs. Not a bug; behavior depends on the run.
- **Node detail "now" block for `baseline`/`declined-group` kinds**: falls back to a "no iteration detail" string. Could be richer (last few primitive calls).
- **Backend watchdog timer**: kill the aclose-hung worker after N seconds with no event progress. Tracked, not built.

### Best practices distilled from this sprint

1. **Verify the diff, not the summary.** Every sub-agent claim → `grep`/`Read` confirm. The probe-agent pattern caught two cases of "completed but not really".
2. **Pin contracts at seams, not constants in tests.** Field names crossing process boundaries get a one-line contract test; primitive *counts* don't.
3. **Defensive React keys: always have an index fallback.** `key={x.id || `__i_${i}`}`. Costs one parameter, prevents a class of bug.
4. **Every fetch in a proxy needs an AbortController + tight timeout.** No exceptions; reuse the 4s pattern.
5. **UI liveness derives from the earliest available signal, not the most specific one.** Status flips on first event; elapsed ticks from `startedAt`.
6. **Commit infrequently at meaningful milestones, not per-fix.** This sprint = 6 commits for ~20 logical changes. Trade fewer commits for richer commit messages.
7. **Push `origin/main` directly; pull --rebase before push.** Coworker branches don't deadlock us; `--autostash` keeps the working tree intact.
8. **The runbook (`docs/runbooks/e2e-testing.md`) is the deploy bible.** Every new failure mode adds a `§4x` entry there.

### Regression checklist before any UI-affecting PR

```bash
# Backend
.venv/bin/python -m pytest tests/rlm/ tests/rdr/ tests/routes/ -q   # must pass clean
# Frontend
cd frontend && npx tsc --noEmit                                      # must be clean
node --version | grep -qE "v(22\\.|23\\.|24\\.)" && npm test         # if Node 22+

# Live smoke
./start_backend.sh &  # term 1
cd frontend && npm run dev   # term 2
# - open /lab → pick preset paper → Begin
# - confirm: status flips queued→running within 5s
# - confirm: elapsed ticks every 1s
# - confirm: no React duplicate-key console warnings
# - confirm: /clusters /leaf-scores /repair-iterations stop after ~15s for non-RDR run
# - confirm: clicking Paper / work / candidate nodes populates sidebar
# - confirm: typing in chat panel → POST 202 → user msg appears in panel
```

### Cross-references

- Architecture: `CLAUDE.md` §`RLM orchestrator`, `Chat steering surface (2026-05-23)`, `Collapsible right sidebar (2026-05-23)`
- Operations: `docs/runbooks/e2e-testing.md` §`4f Backend hangs`, §`6b Chat-steering walkthrough`
- Auth surfaces (API ↔ OAuth): `CLAUDE.md` §`RLM auth — two surfaces, billed separately`
- Past bugs + rules: `learn.md` (especially 2026-05-22 SDK aclose entry — Workaround B)

---

_End of 2026-05-23 entry. Append new entries above this line, leaving "How to read" at the top._
