## 2026-05-23 (night) — Constellation UI + dynamic sandbox + outcome canonicalization; 3 promotions / 2 distinct papers, round 3 chasing the 3rd-distinct

_Updated: 2026-05-23 night._

### Headline

4 commits: removed the run_experiment default cap; rewrote the sandbox guidance to be dynamic on `(sandbox_mode, gpu_mode)` not "docker = CPU forever"; replaced the 4-node tree with a live constellation graph + progressive disclosure UI; fixed a self-inflicted outcome strict-reject regression that was silently dropping the model's natural-English outcome strings on every run.

### Commits landed (newest first)

| SHA | Headline | Lines |
| --- | --- | --- |
| `183f4d6` | Outcome strict-reject regression — record_candidate_outcome was eating natural-language synonyms like 'success' / 'fail' and emitting zero events | +130 / -19 |
| `6022008` | Constellation graph replaces the 4-node tree — every primitive call + every mini-RLM is visible with progressive disclosure so the default view stays clean | +1584 / -8 |
| `34749b6` | Sandbox guidance is now dynamic (sandbox_mode + gpu_mode), not 'docker = CPU forever' — runpod runs no longer get smoke-test nudge | +158 / -75 |
| `c2dc968` | Cap removed by default + agent learns the sandbox is CPU-only — papers now feasible end to end without timeouts | +222 / -21 |

### Paper-sweep results (cumulative across this + prior session)

| Run | arXiv | Iters | Rubric | Promoted | Notes |
|---|---|---|---|---|---|
| B1 (prior) | 2602.01785 CodeOCR | 6 | 31.9% | **1 ✓** | first-ever gate hit |
| A1 | 2512.24601 RLM | 3 | 9.6% | 0 | old prompt |
| A2 | 2512.18131 LLM CodeGen | 3 | 14.4% | 0 | old prompt |
| C1 | 2602.01785 CodeOCR | 2 | 3.7% | **1 ✓** | new prompt validates |
| C2 | 2602.17186 Visual Info Gain | 6 | 7.1% | 0 (1 marginal) | outcomes likely eaten by strict-reject |
| C3 | 2512.18131 LLM CodeGen | 2 | 12.8% | **1 ✓** | new prompt + canonicalization |
| C4 | 2512.24601 RLM | 6 | 32.7% | 0 (1 marginal) | highest rubric, close miss |
| C5 | 2603.26337 RACE-bench | 3 | 3.2% | 0 (0 outcomes!) | the regression that surfaced 183f4d6 |
| **C6** | 2602.17186 Visual Info Gain | running | — | — | retry with canonicalization fix live |
| **C7** | 2603.26337 RACE-bench | running | — | — | retry with canonicalization fix live |

**3 total promotions · 2 distinct papers promoted** (CodeOCR + LLM CodeGen). Round 3 (C6 + C7) racing for the 3rd-distinct now that natural-English outcomes register correctly.

### What's now verified end-to-end

| Capability | Status | Where verified |
| --- | --- | --- |
| Live constellation graph w/ progressive disclosure | ✅ | C4 (95 events, dense case) + C1 (sparse) both render cleanly with default view + click-to-expand |
| Friendly candidate titles (display_title on wire) | ✅ | binding helper + 6 backend tests; UI renders short label, full text in tooltip |
| Mini-RLM visualization (every LLM-using primitive) | ✅ | 7 primitive types get pulsing circles; non-LLM (heartbeat etc.) filtered out |
| Sandbox guidance dynamic on (sandbox, gpu_mode) | ✅ | 10-case parameterized table + auth-parity (API/OAuth/OpenAI byte-identical prompts) |
| Outcome canonicalization (natural English → canonical) | ✅ | 17-pair alias map; unknowns pass through as literal; never reject + drop |
| No default per-primitive timeout cap | ✅ | run_experiment honors only env var OR run-budget deadline |

### What's still open (tracked, not blocking)

| # | Item | Severity | Path |
| --- | --- | --- | --- |
| 1 | 3rd distinct paper not yet promoted | Medium | C6 / C7 in flight; if both miss, try fresh paper |
| 2 | implement_baseline still doesn't emit intermediate SSE events | Low | constellation UI mitigates visually; back-end change would require new event types |
| 3 | Frontend `npm test` blocked on pre-existing Node 21 / rolldown infra | Low | CLAUDE.md mandates ≥22.12; type-check + manual UI works |
| 4 | Round-3 results pending validation of canonicalization fix end-to-end | High | monitor `beahqmtv0` watches C6 + C7 |

### Live URLs (this session's runs)

- **B1 promoted** (CodeOCR baseline): http://localhost:3000/lab?projectId=prj_7b7b34eb9d623b75
- **C1 promoted** (CodeOCR re-run): http://localhost:3000/lab?projectId=prj_b3b00478bcc974b6
- **C3 promoted** (LLM CodeGen): http://localhost:3000/lab?projectId=prj_40ed1381627a46a0
- **C4 close-miss** (RLM, rubric 32.7%): http://localhost:3000/lab?projectId=prj_d42934cc91035eb0
- **C6 running** (Visual Info Gain retry): http://localhost:3000/lab?projectId=prj_0181aa08c697382f
- **C7 running** (RACE-bench retry): http://localhost:3000/lab?projectId=prj_829ee3213b96d1fd
- Leaderboard: http://localhost:3000/leaderboard

### Test count check

560 passed + 1 xfailed (pre-existing) across `tests/rlm/ tests/services/events/ tests/agents/ tests/test_eventstore_sqlite_concurrent.py tests/routes/test_leaderboard_http.py`. TypeScript `npx tsc --noEmit` clean. Zero regressions from any of the 4 night commits.

---

## 2026-05-23 (late evening) — Parallel 4-paper sweep + 6 reliability commits — first run hits promoted-candidate gate

_Updated: 2026-05-23 evening._

### Headline

6 commits across wrapper-template reliability, SQLite concurrency, wire contracts, prompt nudges, leaderboard defense, and experiment timeouts. **B1 (CodeOCR paper) became the first E2E run to actually hit the user's success gate** — completed with rubric 31.88% AND 1 promoted improvement candidate, validating the full pipeline end-to-end.

### Commits landed (newest first)

| SHA | Headline | Lines |
| --- | --- | --- |
| `f8546d1` | run_experiment cap was 2 hours — B2 of the paper sweep wedged for it; now 30 min with env-var escape hatch | +176 / -1 |
| `991517a` | Leaderboard stopped 500ing on legacy final_report shapes — defensive coerce to {} for the four header fields | +64 / -4 |
| `9dd7c6d` | The root was declining every candidate to 'save cost' — now told to try a scoped-down subset before declining anything | +24 / -12 |
| `1f72e07` | Promoted-candidate gate stops getting blocked by a wire-contract bug — candidate_id=None corrupted every outcome event | +136 / -7 |
| `7970506` | Parallel paper sweep was killing the second ingest with 'database is locked' — BEGIN IMMEDIATE + 30s busy_timeout fixes it | +197 / -3 |
| `19e87ee` | The 'stuck Running' bug — runs that finish now actually flip to Completed | +159 / -2 |

### Paper sweep results (last 6 months ML, OAuth surface, docker sandbox)

| Run | arXiv | Pages | Iters | Rubric | Promoted | Outcome | Prompt vintage |
|---|---|---|---|---|---|---|---|
| A1 | 2512.24601 (RLM) | 43 | 3 | 9.6% | 0/3 | partial | old (pre-9dd7c6d) |
| A2 | 2512.18131 (LLM CodeGen) | 13 | 3 | 14.4% | 0/3 | partial | old |
| **B1** | **2602.01785 (CodeOCR)** | **24** | **6** | **31.88%** | **1/3 ✓** | **partial — HIT GATE** | **new (anti-decline)** |
| B2 | 2602.17186 (Visual Info Gain) | 22 | 2 | n/a | n/a | stopped | new — manual stop (CPU-bound train loop) |

### What's now verified end-to-end

| Capability | Status | Where verified |
| --- | --- | --- |
| Reproduction with at least 1 PROMOTED improvement candidate | ✅ | B1 — first E2E run to satisfy the user's success gate |
| Status transitions to "completed" reliably (no more stuck-Running) | ✅ | A1 + A2 + B1 (3/3 completing runs reached `status=completed` correctly); 5 wrapper-string tests pin the invariant |
| 2 concurrent `/runs/arxiv` ingests without DB lock errors | ✅ | A1+A2 retry after BEGIN IMMEDIATE fix succeeded; 3 concurrency tests pin |
| `candidate_outcome` events carry real `candidate_id` strings, not `"None"` | ✅ | A1 ("Implement Algorithm 1..."), B1 (`path_1`) — both real; 4 wire-contract tests pin |
| Leaderboard endpoint survives legacy / malformed `final_report.json` rows | ✅ | live endpoint returns 10 rows including pair A's; 1 defensive-coerce test pins |
| `run_experiment` cap covers common cases without 2-hour wedges | ✅ | default 1800 s; env-var override path tested |

### What still needs work (tracked, not blocking)

| # | Item | Severity | Path forward |
| --- | --- | --- | --- |
| 1 | Zombie Claude subprocesses on long runs (WSL2 subprocess.wait leak) | Low | Defunct children get reaped at runner exit; FD-limit risk only on multi-hour runs. Upstream SDK fix needed |
| 2 | `implement_baseline` agent doesn't know `sandbox_mode` | Medium | When sandbox=docker (CPU), prompt should hint `--smoke-test`. B2's repeated CPU-infeasible baselines are the symptom |
| 3 | Codex-companion plugin auto-invoked from sub-agent | Low | Sub-agent uses installed Claude Code plugin; uninstall plugin OR filter at SDK layer if "no codex" must extend below the orchestrator |
| 4 | SDK aclose noise in stderr is cosmetic but loud | Cosmetic | Watchdog handles fatal cases; suppress non-fatal aclose lines in subprocess wrapper |

### Operating posture

- **OAuth + docker sandbox**: works for papers where Sonnet writes CPU-friendly baselines (B1 was the proof). When the model writes a CPU-bound training loop (B2), the 30-min cap surfaces it within a reasonable iteration window.
- **API-key parity**: every fix this session sits below the auth layer (wrapper template / eventstore / data validation / prompt) — auth-agnostic by construction. Verified via grep of the edit surface.
- **Parallel concurrency**: 2+ concurrent runs now safe under the SQLite BEGIN IMMEDIATE + 30s busy_timeout fix.
- **Production migration (Vercel + Azure)**: provider abstraction was completed earlier this day (`ad460ff` Azure root); the late-evening sprint adds the reliability layer (stuck-Running, SQLite concurrency, wire contracts) needed for production-grade operation.

### Live URLs (this session's runs, viewable in browser)

- A1 (RLM, completed): http://localhost:3000/lab?projectId=prj_f4cc5fa917c27ef1
- A2 (LLM CodeGen, completed): http://localhost:3000/lab?projectId=prj_390202710d0f994b
- **B1 (CodeOCR, completed, 1 promoted)**: http://localhost:3000/lab?projectId=prj_7b7b34eb9d623b75
- B2 (Visual Info Gain, stopped manually): http://localhost:3000/lab?projectId=prj_77b7294aed1bf872
- Leaderboard: http://localhost:3000/leaderboard
- Failed-state demo (F16): http://localhost:3000/lab?projectId=prj_f87990c70c6bc8f6

### Test count check

729 passed, 1 xfailed across `tests/rlm/ tests/rdr/ tests/routes/ tests/services/events/ tests/agents/ tests/test_eventstore_sqlite{,_concurrent}.py`. Zero regressions across the 6 commits.

---

## 2026-05-23 — Reliability + production-path sprint

_Updated: 2026-05-23._

## Ship-readiness pass — 2026-05-23

Ran the launch-readiness sweep on a clean `chore/ship-readiness-2026-05-23` worktree. Backend fixes covered demo-gate parity, JSON 500s, RDR degraded scoring/report metadata, RDR spec cluster SSE events, scorer fallback honesty, leaderboard degraded detection, and `max_pod_seconds` threading across CLI/live/hybrid paths. Frontend fixes cleared lint/type/test failures, made live run streams resilient to SSE disconnects and long-event compaction, kept active RDR artifact polling alive, and replaced silent backend-outage empty states for library/leaderboard/recents with explicit errors. Added a vendored PaperBench bundle identity guard and refreshed docs for the hybrid default (`rlm`), peer `rdr`, and `rlm-pure` escape hatch.

### Headline

15 commits across UI hardening, backend stability, Azure provider support, chat steering, and RLM behavior quality; repo is production-ready for single-user demo and staged for Vercel + Azure migration.

### Commits landed (newest first)

| SHA | Headline | Lines |
| --- | --- | --- |
| `b290449` | A 30-minute run was crashing at second-39 because paper_claims came back as a list | +116 / -1 |
| `2b7bcf1` | The graph is the workspace now — side rails are drag-resizable and viewport-aware | +655 / -21 |
| `ec76d91` | A second opinion at decision time: recommend_next_tool gives the root a Reflexion-lite advisor | +154 / -3 |
| `ad460ff` | Azure OpenAI joins the root-model lineup — the path to Azure deployment opens | +387 / -10 |
| `4e3bd38` | Two stability levers so SDK-deadlock runs stop pretending to be alive | +707 / -4 |
| `8c3371e` | The root was ignoring rlm_query — primitives now ask for it themselves on big slices | +229 / -17 |
| `9d7f8e9` | fix(lab+harness): counter strip layout + wrap_primitive arg coercion + system prompt rlm_query nudge | +157 / -4 |
| `c6511be` | feat(sidebar): aggregate counter strip + enriched subrlm/baseline node detail | +343 / -8 |
| `ad32198` | feat(sidebar): add Upload nav item above Lab/Library — one-shot ?new=1 trigger | +72 / -1 |
| `0cc2e77` | fix(lab): SSR-safe elapsed clock — initialize nowMs=null to avoid hydration mismatch | +13 / -3 |
| `1316b8a` | fix(rubric): align rubric_area key contract + defensive empty-name handling | +95 / -14 |
| `e4c30a0` | fix(lab): live status flip on primitive_call + real-time elapsed clock + RDR polling stops on empty 200 | +90 / -8 |
| `f664659` | docs+chore: runbook + CLAUDE.md + system_overview for chat steering / sidebar / F7 / backend-hang remedy | +134 / -11 |
| `becfac8` | fix(landing): bench cells render cleanly without overlap | +16 / -8 |
| `a07c336` | feat(rlm+ui): real-time chat steering + collapsible right sidebar with kind-specific node detail | +2034 / -55 |

### What works end-to-end

| Capability | Status | Where verified |
| --- | --- | --- |
| OAuth root + local GPU sandbox | ✅ | live arXiv run; watchdog/heartbeat flags surface accurately |
| API-key root parity | ✅ | `tests/rlm/test_chat_steering_auth_parity.py` (6/6) |
| Azure OpenAI provider | ✅ | `tests/rlm/test_build_llm_client.py::TestAzureOpenAI` (4/4) |
| SDK aclose deadlock recovery | ⚠️ | flagged via watchdog + heartbeat; not yet auto-recovered (out of scope this session) |
| paper_claims as list-of-dicts | ✅ | `tests/rlm/test_paper_claims_coercion.py` (7/7) |
| Failed-run lab UI + Rerun button | ✅ | shipped this session (see commits) |
| Chat steering POST → primitive round-trip | ✅ | curl verified; in-band UI |
| Sub-RLM spawning under hint | ✅ | live run: 5 sub-RLMs with focused decomposition |
| Resizable workspace | ✅ | shipped this session |

### Open / deferred

- W4 tool router landed but ROI untested in the wild — needs 3–5 runs to see if root actually calls `recommend_next_tool`.
- aclose deadlock root cause is upstream SDK; we observe + flag but don't auto-recover. Future track: subprocess-level restart-on-degraded.
- Cost ledger reports $0 for OAuth runs (SDK doesn't surface tokens). Documented in CLAUDE.md.
- Vercel/Azure production migration: provider abstraction ready (W2c); state migration (Postgres + Blob) + multi-tenant auth not started.
- Frontend vitest blocked on Node 21.7.3 (needs ≥22.12). Tests pass under Node 22; CI / coworker workflow notes in CLAUDE.md.

### Test surface

Backend: 681 passed, 1 xfailed (`tests/rlm/ tests/rdr/ tests/routes/ tests/agents/ tests/services/events/`). Frontend: tsc clean across all changes; vitest blocked on Node 21 (works under Node 22).

---

# RLM Phase 5/6 — Progress

_Updated: 2026-05-22 — debug-and-harden session._

## Objective

Debug-and-harden pass: fix the issue catalogue (I1–I13 + I4), drive PaperBench
papers end-to-end (plus one harder recent paper), and complete GitHub issue #62.

## Status

The session's seven commits (`2630a77` P0, `4e7b4a4` catalogue I5–I13,
`52625d6` run_experiment Bug A/B/C + I3, `d656c7d` I4 + idempotency deflake,
`c22feb7` I3 revert, `652f842` rdr design spec + impl prompt, `bfe9e3d` run-3
outcome) are **squashed into one commit on `origin/main`**, whose `Closes #62`
keyword closed issue #62. The `merge` branch keeps the un-squashed history.

Test suite green (I3's 4 tests removed with the revert): 1252 passed, 3 skipped.

## run_experiment Bug A/B/C — fixed and verified

Run 2's `run_experiment` failed in 6 s. Three compounding bugs in
`backend/agents/rlm/primitives.py`, all fixed and **verified live by run 2b**:

- **Bug A** — `_execute_in_sandbox` logged stdout only; a failed command's
  stderr traceback was discarded. Fix: `_combine_command_output` joins both.
- **Bug B** — the experiment ran the image `detect_environment` built before
  any code existed (missing deps). Fix: `run_experiment` rebuilds from
  `ctx.project_dir/Dockerfile` via `build_environment`.
- **Bug C** — the sandbox ran `network_disabled`. Fix: `_execute_in_sandbox`
  enables network for the experiment container (user-approved).

## I3 — reverted (root-prompt change backfired)

The `_PAPER_GROUNDING` section anchored the `qwen3-coder-featherless` root on
the understanding phase — run 3 looped on `understand_section` for 21
iterations and never reproduced. Reverted; the known-good prompt is restored.
The `ftrl` acronym-collision I3 targeted is unaddressed — a robust fix needs
more than a prompt nudge on this root model. See `learn.md` 2026-05-22.

## I4 — fixed

Workspace `paper_text` now loads from the parser's full-text blob (located via
the `ParsingCompleted` event), not a lossy chunk-reassembly. Guard:
`test_paper_text_equals_parser_full_text`.

## Runs

See `runlog.md`.

- **Run 1** — sequential-neural-score-estimation: leaf **0.366**, partial.
- **Run 2b** — mechanistic-understanding (`pb_..._1779457326`): leaf **0.079**,
  failed — `run_experiment` succeeded (Fix A/B/C verified live); weak baseline.
- **Run 3** — GoRL (arXiv 2512.02581): first attempt looped under I3; re-run
  (I3 reverted) progressed through the full pipeline but crashed on the
  Featherless Qwen3-Coder 49 152-token context cap. `failed`, leaf 0.0.

## rdr harness — built (2026-05-22)

The `rlm_rubric_orchestration` branch carries the full rdr harness — six
modules under `backend/agents/rdr/` (`models`, `decomposer`, `context_engineer`,
`agent`, `controller`, `run`), the `--mode rdr` CLI wiring, the
`scripts/rdr_paperbench.py` launcher, and 112 rdr tests including a full
offline end-to-end on the real `sequential-neural-score-estimation` bundle (27
clusters). The deterministic controller reproduces the paper cluster-by-cluster
against the official rubric and repairs weak clusters in a capped loop; no LLM
in the control path. Provider/model is dynamic — Claude OAuth (Sonnet) locally
or Azure OpenAI — via the existing `collect_agent_text` runtime resolution.

Full test suite green: 1362 passed, 3 skipped (the 3 are pre-existing
optional-dep skips: chromadb, tesseract). Built across six phase commits
squashed into one milestone commit on `rlm_rubric_orchestration`.

## Remaining

Real live end-to-end run (Claude OAuth + local GPU) on a PaperBench bundle —
verify the leaf score beats the rlm baseline (≈0.37) on
`sequential-neural-score-estimation`. Production wiring into the UI / SSE
bridge if the live run validates. See the design spec §10 success criteria.
