# ReproLab Infra + Optimization Audit — baseline `f848d7e`

Forensic audit of the RLM sub-call path after the 2026-05-29 SDAR wedge. No code changed.
Labels: **OBS** observed · **INF** inferred · **SPEC** speculative(+verify). Anthropic liveness at audit: `401` in `0.13s` (not an outage).

## A. Failure-mode catalogue

Cols: ID · mode · sources · trigger · blast · sev · freq · current mitigation · why insufficient · best fix · validation. (Evidence cited once here; later sections reference IDs.)

| ID | Mode | Sources | Trigger | Blast | Sev/Freq | Mitigation | Why insufficient | Fix | Validate |
|--|--|--|--|--|--|--|--|--|--|
| FM-001 | Sub-RLM stream stall: bundled CLI has **no read-idle timeout**; orphaned child holds OAuth slot | wedge,repo,Ctx7,/runs | half-open TCP / stalled stream | whole run stalls | P0/obs | 600s **total** cap + SIGKILL child (`rlm_query.py:589,607,608-633`) | total-time ≠ read-idle; wedge run's 76-min subcall **ended in `sub_rlm_complete`** → cap never bounded it; SDK `query()` "continues indefinitely if no ResultMessage" (Ctx7) | per-event `asyncio.wait_for` read-idle | inject stalled gen; kill ≤ idle+grace |
| FM-002 | Empty-string masks failure as success | wedge,repo | any FM-001 timeout | bad data → report | P0/obs | D3 fail-soft `return ""` (`rlm_query.py:633`) | `on_subcall_complete` fires `error=None`; root can't tell stall from empty answer | typed `SubRlmTimeout` + `sub_rlm_stalled` event | unit: timeout→sentinel |
| FM-003 | Process watchdog **off by default**; `rlm.max_timeout` only between iterations | wedge,repo,/runs | no `--max-wall-clock`; mid-iter wedge | unbounded run | P0/obs(3/3) | `_arm_watchdog` (`run.py:824`) but `wall_clock_s=None` default (`:116,1157,845`); `:833` | mid-iteration wedge is unbounded | always-on code-level subcall watchdog | run w/o ceiling; assert caught |
| FM-004 | Report-without-evidence: `verdict=partial` w/ `baseline_metrics={}`, rubric 0, no `run_experiment` | /runs,repo | floor (2 iters) met by empty iters | false success on leaderboard | P1/rec(2/2) | `MIN_RUBRIC_ITERATIONS=2` | floor is iteration-**count**; BUG-LR-013 only guards `None` below floor | evidence gate (no non-`failed` verdict w/o successful `run_experiment`) | replay pb_…784 → `failed` |
| FM-005 | Status mismatch; `updatedAt` lifecycle-only | /runs,repo | kill/stall mid-run | UI shows false "completed/alive" | P1/obs | mtime fallback `live_runs.py:956` | pb_…083 `status=completed` w/ `killReason` set; cost loop bumps mtime not `updatedAt` (`run.py:992-1006`) | stamp `updatedAt` per event; status from last event | assert advances/event |
| FM-006 | Heartbeat advisory (prompt-dependent) | wedge,repo,/runs | root forgets pre-op call | wedge invisible | P1/obs | system prompt `:370-383`; `heartbeat()` no-op `primitives.py:4096` | wedge run fired **1** heartbeat in 89 min | code-emitted per-subcall heartbeat | ≥1/interval w/o prompt help |
| FM-007 | Kill is Darwin pgrep + substring; no `killpg`; thread leaks | wedge,Ctx7 | timeout on non-Darwin / renamed bundle | stale children | P2/obs | best-effort fail-soft (`rlm_query.py:65-122`) | asyncio doc: cancel≠child death; need `start_new_session`+`killpg`; `ex.shutdown(wait=False)` leaks worker (`:637`) | own spawn in process group; else harden | kill grandchildren in test |
| FM-008 | Prompt-only parallelism → 8 concurrent sub-RLMs over fragile transport | wedge,repo,Ctx7 | iter-0 fan-out | 8× OAuth slots; ↑FM-001 prob | P1/likely | advice only (`system_prompt.py:295-317`) | unbounded in code; OAuth limit unknown | code-level concurrency cap + route nav off CLI | load test vs OAuth 429 |

## B. Infra audit — sub-RLM path

```
understand_section(slice)  primitives.py:658  PURE heuristic, NO LLM (note) + primitive_cache
root REPL: rlm_query/llm_query  rlm builtins (not custom_tools)  routed via other_backends[0]
   → ClaudeOauthClient.completion  claude_oauth_client.py:82  model=haiku-4-5
       timeout_s NEVER passed to complete() (:115) → f848d7e change here is INERT (OBS)
   → ClaudeLlmClient.complete  rlm_query.py:556  ThreadPoolExecutor(1); future.result(600) TOTAL
       on timeout: SIGKILL new PIDs, return "" (:608-633); finally shutdown(wait=False) leaks thread
   → _async_complete → claude_agent_sdk.query()  rlm_query.py:699  async for: NO per-event timeout
   → bundled claude CLI  anyio.open_process, NO timeout/keepalive (Ctx7); not in own process group
   → api.anthropic.com  streaming HTTP; macOS TCP keepalive ~2h
```

Routing (OBS): depth-1 `rlm_query`/`llm_query` → `other_backends[0]=sub_backend` = **Haiku** (`run.py:1434-1457`, `models.py:195-198`), never root Sonnet. Accelerator override (`run.py:1436-1444`) redirects nav to an OpenAI-compatible endpoint when `REPROLAB_ACCELERATOR≠off`, scope default `navigation` (`:1200`) — grader/verify stay on Sonnet.

Edges carrying new risk: (1) **complete→SDK** — 600s *total*, no retry, no read-idle, worker leaks, child SIGKILL'd. (2) **SDK→CLI** — no timeout, hangs forever absent ResultMessage, only `interrupt()` (keeps session). (3) **`on_subcall_complete`** — fires `error=None` on a `""` timeout → masks stall as success. Observability gaps: `updatedAt` lifecycle-only (FM-005), watchdog off (FM-003), heartbeat advisory (FM-006).

**Weakest link:** `rlm_query.py:556-637` — a **total-time** cap wrapping a **no-read-idle** stream whose only failure signal is `""` and whose cleanup leaks the worker thread. It cannot distinguish a slow-but-healthy stream from a dead socket, and hides the difference from the root.

## C. /runs empirical audit

3 dirs — directional, not statistical.

| metric | n |
|--|--|
| inspected / with events.jsonl | 3 / 3 |
| completed / killed | 2 / 1 |
| with `final_report.json` | 2 (SDAR has none) |
| reports w/ **empty metrics + rubric 0** | **3/3** |
| event gaps >2 min | 2 (76.1, 7.1 min — SDAR) |
| dangling `sub_rlm_spawned` | 1 (5 spawned / 4 complete) |
| demo_status ⟂ events | 1 (pb_…083) |

Clusters: **(1) Silent wedge/no-heartbeat** — `prj_09047604e591d969` (SDAR): 76.1+7.1-min spawn→complete gaps, 1 dangling spawn, **1** heartbeat/89min, no report → FM-001/003/006 → Cmt 1+2. **(2) Report-without-evidence** — pb_…784 ships `partial`, prose "implemented and executed", but trace has **no** `run_experiment`/`verify_against_rubric`, `baseline_metrics={}`; pb_…083 `partial` rubric 0.0 iters=1 → FM-004 → Cmt 3. **(3) Status mismatch** — pb_…083 `status=completed` while `killReason` set → FM-005 → Cmt 2.

Recurring/systemic: empty-evidence reports (every finished run), lifecycle-only `updatedAt`. One-off: the 76-min wedge (1 run, but architecturally reachable any run; not over-fit). Top-3 recurring → all covered (zero-evidence→Cmt 3; status mismatch→Cmt 2; silent wedge→Cmt 1+2).

## D. Optimization opportunities (effort→impact)

Impact = reliability+latency+observability+cost − risk − ops.

| # | Cat | Recommendation | Ev | Wall-clock Δ | Cost Δ | Risk | Validate |
|--|--|--|--|--|--|--|--|
| 1 | Reliab | Read-idle timeout on SDK stream + typed `SubRlmTimeout` | FM-001/002 | wedge ∞→idle+grace | 0 | Low | stalled gen test |
| 2 | Observ | Code heartbeat + dangling-spawn detector + `updatedAt`/event | FM-003/005/006 | wedge visible <2min | 0 | Low | gap>120s→warn |
| 3 | Cost+Rel | Route nav sub-RLM to OpenAI **gpt-5-mini** via `endpoint` accelerator | FM-001/008,`run.py:1436-1444` | nav off no-timeout transport | − (user has credits) | Med | A/B vs Haiku |
| 4 | Reliab | Evidence gate on `final_report` | FM-004 | 0 | 0 | Med (may trap stuck) | replay pb_…784 |
| 5 | Latency | Enforce sub-RLM concurrency cap in code | FM-008 | bounds 429 bursts | 0 | Low | concurrent load |
| 6 | Latency | Freeze 31.9KB prompt prefix for cache reuse (measure first) | `system_prompt.py` | input-token tax ↓ | − | Low | log cache-read tokens |

**Cost — gpt-5-mini (explicit answers).** **Yes**, **experiment flag**, **navigation scope first**. Mechanism: generic `endpoint` provider — `REPROLAB_ACCELERATOR=endpoint`, `_BASE_URL=https://api.openai.com/v1`, `_MODEL=gpt-5-mini`, `_API_KEY=$OPENAI_API_KEY` (`accelerator.py:31,205,213-214`) — moves the highest-volume call off the bundled CLI onto the openai/httpx transport that **raises `ReadTimeout`, auto-retries 2× (DOCUMENTED), leaks no subprocess**. Verified caveats: accelerator was built for self-hosted vLLM (default Qwen2.5-Coder-32B) so cloud-OpenAI use is **untested**; `build_accelerator_client` not confirmed to set a tight `httpx.Timeout`, so idle bound is likely openai default `read≈600s` — set ≈120s. Effort: **config + small timeout wiring**, not a pure flag flip. Keep grader on Sonnet-OAuth (scope `navigation`). Likely regression: weaker nav summaries → mitigated (nav feeds primitives, not the rubric). Telemetry before default: latency, `ReadTimeout` rate, `response._request_id`, rubric parity ≥3 paired SDAR runs. Rollback: `=off`.

**Observability — surface wedge <2 min.** Per-subcall start/heartbeat/elapsed/model/child_pid/last_token_ts; "no event for N s" detector over `dashboard_events.jsonl` (the 76-min gap is exactly this signal); `updatedAt` per event; status from last event.

## E. Research-inspired ideas (ranked)

| # | Paper | Mechanism | Pain | Smallest version | Risk | Verdict |
|--|--|--|--|--|--|--|
| 1 | **AutoScientists** (2605.28655) | self-organizing teams, but **supervisor is plain code in a heartbeat loop**, not an LLM | wedge had no role to detect it | per-subcall stall watchdog (code) | low | **adopt now** (= Cmt 1/2) |
| 2 | **PEEK** (2605.19932) | bounded ~1024-tok context map (distiller/cartographer/evictor); **evaluated on RLM**; 93–145 fewer iters | redundant `rlm_query`/`llm_query` nav, drift | deterministic write-once map from primitive outputs, no LLM | stale-fact contaminates report | **prototype behind flag** |
| 3 | **MUSE** (2605.27366) | skill create→pytest-gate→register; `.memory.md` | recipes not promoted; failures leave no lesson | post-run **negative-lessons** file → next implementer prompt | curation on unstable system | **revisit later** (neg-lessons first) |
| 4 | **BES** (2605.28814) | recombination evolution; verifier=fitness | cross-run strategy amnesia | offline archive-as-prior, no in-loop selection | **evolution games weak auto-rubric** | **reject** in-loop; borrow backward-decomposition for verifier |

Required answers: PEEK reduces repeated **navigation** sub-calls and curbs drift; it does **not** improve prompt-cache ratio (map is small+trailing) nor cut `understand_section` cost — **correction: `understand_section` is a pure heuristic, no LLM call** (`primitives.py:658`); redundant-LLM cost lives in `rlm_query`/`llm_query`. Contamination (a wrong cached β/λ/model-size poisons the report) → deterministic-first, DELETE-capable, flag-gated. **Rejected as premature:** BES in-loop evolution (multiplies a documented-weak fitness into confident fakery) and multi-agent role-splitting — AutoScientists' own supervisor is deterministic code, and an LLM peer cannot observe a sibling blocked inside a primitive call, so role-splitting would **not** have prevented the wedge.

## F. Risks of `f848d7e`

| Fix | Risk | Sev/Likely | Detect | Verdict |
|--|--|--|--|--|
| PID snapshot + SIGKILL | substring match, no `killpg`, misses grandchildren, non-Darwin, thread leaks (FM-007) | Med/Med | unit on synthetic tree | keep, harden in Cmt 1 |
| 1200→600s cap | total ≠ read-idle; healthy long call killed; premature→forced-iter loop | Med/Med | log cap-trips vs durations | keep, supersede Cmt 1 |
| `_DEFAULT_TIMEOUT_S 1800→600` | **inert** — never passed to `complete()` (`:115`); false confidence | Low/High(dead) | grep call sites | modify (remove false signal) |
| Parallelism prompt | root fans out 8 over fragile transport → OAuth saturation, ↑FM-001; may be ignored | High/Med | count concurrent spawns | keep but cap |
| `sourceKind` widen | masks schema drift; symptom fix | Low/Low | schema test | keep |

`f848d7e` is **safe as baseline** (reduces, not removes, the wedge; fail-soft) but 3/5 fixes are partial/inert and must be superseded by Cmt 1/2.

## G. Executive verdict

1. **Biggest risk:** a sub-RLM stall is bounded only by a **total-time** cap that returns `""` and leaks worker/child — no read-idle (FM-001).
2. **Highest-leverage fix:** per-event read-idle timeout + typed failure (Cmt 1).
3. **Biggest false lead:** multi-agent role-splitting and in-loop evolutionary search — neither addresses the wedge.
4. **`f848d7e` safe baseline?** Yes; tactical — supersede the 3 weak fixes.
5. **Next priority:** reliability, then observability; cost (gpt-5-mini) follows once telemetry exists.
6. **North-star metric:** time-to-detect a stalled subcall (<120s = max inter-event gap in `dashboard_events.jsonl`).
7. **Obvious in 2 min via:** a "no event 120s + dangling `sub_rlm_spawned`" detector emitting `run_warning` + stamping `demo_status` (Cmt 2).
8. **Recurring /runs failure:** every finished report ships `baseline_metrics={}` + rubric 0 yet `verdict=partial` (FM-004 → Cmt 3).

## H. Next-3-commits

**Cmt 1 — Read-idle timeout + typed sub-RLM failure + hardened kill** (<200–500 LOC). FM-001/002/007; cluster 1.
Files: `rlm_query.py`, `sse_bridge.py`. Change: wrap each `await anext(stream)` in `asyncio.wait_for(read_idle≈120s)`; on idle → cancel + kill child (process-group where we own spawn, pgrep fallback) + `await wait()`; return typed `SubRlmTimeout` (not `""`) + emit `sub_rlm_stalled`. Accept: stall detected ≤ idle+grace; root gets typed failure; no orphaned child. Test: stub stalled async-gen; manual proxy-induced stall on SDAR. Risk: healthy slow streams → tune idle (idle, not total). Rollback: `REPROLAB_SUBRLM_READ_IDLE_S=0`.

**Cmt 2 — Heartbeat + dangling-subcall detector + status reconciliation** (200–500 LOC). FM-003/005/006; clusters 1+3.
Files: `run_watchdog.py`, `live_runs.py`, `run.py`. Change: always-on watchdog tracks last-event ts + open `sub_rlm_spawned`; gap>N or unmatched spawn>N → `run_warning code="subcall_stall"` + stamp `demo_status`(`updatedAt`); stamp `updatedAt` every event; status from last event. Accept: 120s artificial gap → warning + fresh `updatedAt` within one interval; killed run never reads alive. Test: replay SDAR events (76-min gap)→warning; dashboard stalled badge. Risk: spam → debounce. Rollback: `REPROLAB_STALL_DETECT_S=0`. **Provides the A/B telemetry the gpt-5-mini route needs — do that as the immediate fast-follow.**

**Cmt 3 — Final-report evidence gate** (<200 LOC). FM-004; cluster 2.
Files: `run.py` writer / `forced_iteration.py`. Change: refuse `verdict∈{reproduced,partial}` when no `run_experiment` succeeded AND `baseline_metrics=={}`; downgrade to `failed` w/ `evidence_gap` (mirror Lane O), respecting `remaining_s()<=60s` bypass. Accept: pb_…784-shaped run → `failed`/refused. Test: synthetic empty-trace report → gated; replay pb_…784. Risk: trapping an impossible paper → bypass on wall-clock + emit `evidence_gap` (honest, not hung; ties BUG-LR-013). 

All three hit the wedge and/or recurring /runs failures; none is a rewrite, multi-agent, new service, or new persistence layer. gpt-5-mini routing (D#3) is **yes**, as the fast-follow after Cmt 2 (needs its A/B telemetry), not bundled in.

## I. Validation plan

Per commit: see H (unit+integration+manual+rollback). Telemetry expected: `sub_rlm_stalled`, `run_warning code="subcall_stall"`, `evidence_gap`. Dashboard: stalled badge; `updatedAt` advancing per event.

**E2E next SDAR:** `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY python -m backend.cli reproduce 2605.15155 --model claude-oauth --sandbox runpod --max-wall-clock <n>`. Expect paired `sub_rlm_spawned`→`sub_rlm_complete`, `iteration_heartbeat` ≥1/interval (code-emitted), `gpu_resolved`. Stalled stream → `sub_rlm_stalled` + typed failure ≤ idle+grace, child reaped, run continues. Terminate-ignoring child → process-group/SIGKILL, next subcall gets clean OAuth slot. OpenAI route → `ReadTimeout` retried 2×, request-id logged, rubric parity. Report after failed experiments → evidence gate → `failed`, never silent `partial`. status ⟂ events → reconciler stamps truth within one interval; empty subcall → typed not `""`; forced-iteration → bounded; stale map/skill → flag-gated off. Targets: wedge visible <2 min; killed < idle+grace; no OAuth slot held indefinitely; failures typed+observable; no success claim without artifact. Cover: normal/stalled/killed/terminate-ignoring subcall, OpenAI route, malformed `final_report.json`, stale `demo_status`, missing events.jsonl, forced-iteration loop, /runs replay.

## J. Rejected recommendations

| Rejected | Tempting | Wrong/premature | Instead |
|--|--|--|--|
| Replace `claude-oauth` root | CLI wedge-prone | user-preferred, free; only *sub* path is hot | keep root; route only nav off CLI |
| All inference → local Qwen/Featherless/friend GPUs | avoid Anthropic transport | user lacks keys/GPUs (hard constraint) | gpt-5-mini for nav (has OpenAI credits) |
| Opus sub-agents | "smarter" | hard pref: sub-agents stay `claude-sonnet-4-6` | unchanged |
| Async rewrite / K8s / Celery / Redis / Temporal | "scalable" | single-machine model not shown unsalvageable | in-process watchdog |
| Only TCP keepalive | OS handles it | macOS ~2h; read timeout is the right tool | app read-idle timeout |
| Only `future.result(timeout)` | already present | total≠read-idle; leaks thread; returns `""` | per-event `wait_for` + typed |
| Prompt-only concurrency | zero-cost, shipped | unbounded, ignorable, ↑FM-001 | code-level cap |
| Empty-string fallback for failures | keeps loop alive | hides stall as success (FM-002) | typed sentinel |
| Adopt 4 papers as architecture | exciting | most don't map to a failure | PEEK flag; MUSE neg-lessons later; AutoScientists=watchdog only |
| Broad skill system now | self-improving | curation surface on unstable system; too few runs | negative-lessons file first |
| Full multi-agent team | "robust" | wouldn't prevent wedge; LLM can't observe blocked sibling | deterministic watchdog |
| /runs anecdotes as proof | only 3 runs | over-fits | cluster, mark directional |
| Every sub-RLM → OpenAI by default | "cleaner" | no A/B yet; grader must stay Sonnet | navigation scope, flag, A/B |

### Punchlist
**Cmt 1** read-idle timeout + typed sub-RLM failure + hardened kill · **Cmt 2** heartbeat + dangling-subcall detector + `updatedAt` reconciliation · **Cmt 3** final-report evidence gate. Fast-follow: `REPROLAB_ACCELERATOR=endpoint`→gpt-5-mini (navigation) with Cmt-2 A/B telemetry + a tight read timeout. North-star: stalled-subcall detect <120s.
