# Session handoff — 2026-05-22 (live-test + merge prep)

_Self-contained continuation doc written immediately before compaction.
Read this first, then `docs/superpowers/specs/2026-05-22-rdr-merge-plan.md`,
then resume. Everything you need is on disk or in git._

---

## 1. Where we are (commit + branch state)

- Working branch: **`rlm_rubric_orchestration`** (NOT main).
- Latest commit pushed: **`c928fb7`** ("rdr: pre-merge fixes per Codex adversarial review (2 CRITICAL + 4 IMPORTANT)").
- All commits since `main` (origin/main is at `2e1ce37`):

```
c928fb7  rdr: pre-merge fixes per Codex adversarial review (2 CRITICAL + 4 IMPORTANT)
3a73903  rdr fix: controller-level watchdog + agent CWD guard
96b0b63  rdr fix: run score_reproduction off the event loop (asyncio.to_thread)
3e95752  rdr fix: remove stray tsnpe_neurips submodule reference
5b53a10  rdr fix: exclude ephemeral/cache dirs from snapshot; defensive write
ee51f02  rdr — rubric-driven paper-reproduction harness (--mode rdr)
```

Plus, **uncommitted in the working tree**: this handoff doc. Commit + push after writing.

- Full suite: **1380 passed, 3 skipped** (the 3 are pre-existing optional-dep skips: chromadb, tesseract).
- rdr-suite: **138 tests** (all green).

---

## 2. Binding merge bar (user's choice)

**Real scores + beat rlm baseline (0.37) on at least one paper.**

`--mode rlm` previously scored 0.366 on `sequential-neural-score-estimation`.
The rdr harness must score **> 0.37** on at least one PaperBench bundle paper
before merging to `main`.

Additional binding decisions (user-confirmed via AskUserQuestion):

| | Decision |
|---|---|
| **Merge strategy** | **NEW branch** (do NOT force-push), with **10-15 significant noteworthy commits** cherry-picked / structured cleanly. Original branch preserved. |
| **Mech wedge follow-up** | **Retry-on-watchdog** — bash wrapper restarts the rdr command with `--resume` after a watchdog kill (124). Add `--resume` to the CLI + resume logic to the controller. ~2h work. |
| **Cost cap (`--max-usd`)** | **Ship as-is** — no `--max-usd` enforcement for rdr; do not call it out. |
| **Frontend timing** | **ONE combined merge** — rdr backend + frontend wiring together, not two PRs. |
| **Frontend location** | **On `main` already** (the existing `frontend/` Next.js app); needs wiring to the new rdr backend. |
| **Frontend backend needs** | All three: reuse existing `/api/demo` flow + new rdr endpoints (cluster progress, repair stats) + **full live SSE of cluster checkpoints + repair-pass events**. |
| **Auth** | All three dynamically: existing X-Demo-Secret HMAC + proper auth (OAuth/JWT) + per-user (Claude OAuth identity). Configurable. |

---

## 3. Live runs / background tasks

When you resume, check these:

- **seqnn3** (background bash task `b2v5nt48b`) — the binding live e2e on
  `sequential-neural-score-estimation` with ALL fixes (Codex CRITICAL + IMPORTANT
  + watchdog + CWD guard + scorer-off-loop + snapshot exclusion). Launched
  ~22:14. Log: `/tmp/rdr_seqnn3.log`. Run dir:
  `runs/pb_sequential-neural-score-estimation_*` (latest by mtime).
  ETA: 2-3h. **Check `final_report.json` for `rubric.overall_score`.**

- **SDK aclose investigator** (agent `a4c0d8038906e84c9`, background) —
  deep-dive into the `claude_agent_sdk` async-generator close bug.
  Returns when done; results in the agent transcript.

- **Monitor `bcb7daxuv`** (persistent) — emits state changes for seqnn3
  iterations + grep'd errors. Stop it when no longer needed (TaskStop).

Check live state:
```bash
ls -dt runs/pb_sequential-neural-score-estimation_* | head -1   # latest seqnn dir
ls "$(ls -dt runs/pb_sequential-neural-score-estimation_* | head -1)/iterations/" | wc -l
cat "$(ls -dt runs/pb_sequential-neural-score-estimation_* | head -1)/final_report.json" | python3 -m json.tool | head -20
ps -ef | grep "reproduce sequential-neural.*--mode rdr" | grep -v grep
```

---

## 4. Key files / docs

- `docs/superpowers/specs/2026-05-22-rubric-driven-harness-design.md` — the design spec (committed earlier session).
- `docs/superpowers/specs/2026-05-22-rubric-driven-harness-implementation.md` — the implementation prompt.
- `docs/superpowers/specs/2026-05-22-rdr-merge-plan.md` — the merge plan (gate criteria, rebase strategy, risks).
- `docs/superpowers/specs/2026-05-22-session-handoff-compaction.md` — this doc.
- `backend/agents/rdr/{models,decomposer,context_engineer,agent,controller,run}.py` — the 6 harness modules.
- `backend/cli.py` `_cmd_reproduce_rdr` — CLI dispatch + new flags.
- `scripts/rdr_paperbench.py` — standalone launcher.
- `tests/rdr/` — 138 tests across 7 files (`test_rdr_models`, `test_decomposer`, `test_context_engineer`, `test_agent`, `test_controller`, `test_rdr_run`, `test_rdr_offline_e2e`).
- `runlog.md` / `progress.md` / `CHANGELOG.md` / `learn.md` / `CLAUDE.md` — all updated through `ee51f02`.

---

## 5. Bug-fix history (chronological — what shipped and why)

| Commit | Bug | Fix |
|---|---|---|
| `ee51f02` | (the initial harness) | 6-phase squash of Phases 0-5 + design+impl prompts. |
| `5b53a10` | mech1 PermissionError on Docker-created `hf_cache/.locks/*.lock` files during the repair-pass snapshot. | `_snapshot_code_dir` skips ephemeral / cache dirs (`__pycache__`, `hf_cache`, `.locks`, `outputs`, `sbi-logs`, `.git`, `wandb`, `mlruns`, …) and pyc/so/lock suffixes; controller merge-writes wrapped in defensive try/except. |
| `3e95752` | Stray `tsnpe_neurips/` submodule reference accidentally committed (an agent's `git clone` landed at repo root). | `git rm --cached`; repo cleanup. |
| `96b0b63` | `score_reproduction` deadlocks the loop via `asyncio.run()` nesting (live seqnn log showed "Batch N LLM call failed (asyncio.run() cannot be called from a running event loop)" → all leaves defaulted to 0.0). | `scores = await asyncio.to_thread(score_reproduction, ...)` at both call sites. |
| `3a73903` | Claude SDK `aclose()` deadlock wedges controller after subprocess exits unexpectedly; agent's `git clone` escaped `project_dir`. | `_ClusterWatchdog` (threading.Timer + `os._exit(124)`, configurable via `RDR_CLUSTER_WATCHDOG_S`, default 900); agent CWD guard (`_snapshot_repo_root_entries` + `_cleanup_repo_root_escape`, auto-cleans top-level repo-root entries). |
| `c928fb7` | Codex adversarial review (2 CRITICAL + 4 IMPORTANT): scorer exception propagation; path-traversal in merge-write; event-loop-blocking primitives; process-global `os.chdir` racing; watchdog no postmortem; CLI args unregistered. | try/except around scorer thread (zero-score fallback); `dest.resolve().is_relative_to(code_dir.resolve())` guard on merge-write; wrapped `detect_environment`/`build_environment`/`run_experiment` in `asyncio.to_thread`; removed `os.chdir` (use SDK's `cwd=` path via `ClaudeAgentOptions`); watchdog writes emergency `final_report.json` with `status="watchdog_killed"`; registered `--max-repair-iterations` / `--repair-target` flags on the CLI subparser. |

---

## 6. Known limitations / deferred items

- **Claude SDK aclose() deadlock** — investigated; the watchdog hard-bounds
  it via `os._exit(124)` but the run dies abruptly. Retry-on-watchdog is on
  the work-list. Real fix (SDK patch or subprocess wrapper) is v2.
- **Agent escape outside `code_dir`** — top-level repo-root escapes are
  detected+cleaned; subdirectory escapes (e.g. into `backend/`), tmp,
  $HOME are NOT covered. v2 fix: containment (docker / chroot).
- **Mech-class papers wedge** — observed in mech1, mech2, mech3 around
  cluster 15-16 (likely a specific cluster type triggers the SDK aclose).
  With the watchdog they die in 15 min; without resume they don't complete.
- **Agent tool surface (design §4.4 deferred)** — v1 reuses
  `baseline-implementation`'s Read/Write/Edit/Bash; "9 primitives as SDK
  tools + paper_search" deferred to v2. Functional escape hatch is bash on
  `paper_full.md` in the cluster's CWD.
- **Azure provider path** — code is dynamic (reads `OPENAI_BASE_URL` /
  `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_API_KEY`), never live-tested.
- **No SSE / dashboard wiring (yet)** — see §8 (frontend scope expansion).

---

## 7. Outstanding work BEFORE the combined merge

Order matters; this is the work-queue.

### 7.1 Bind the live score (in-progress)

- [ ] **seqnn3 completes** with `rubric.overall_score > 0.37`.
      Background task `b2v5nt48b`. ETA 2-3h from 22:14.
- [ ] If it scores ≤ 0.37, iterate: examine `runs/pb_..._<ts>/final_report.json`
      + `leaf_scores`, identify under-performing clusters, fix root causes,
      restart.

### 7.2 Retry-on-watchdog (user-confirmed scope)

- [ ] Add `--resume` flag to `--mode rdr` in `backend/cli.py`.
- [ ] In `backend/agents/rdr/controller.py` `run_rdr`: when starting,
      check `ctx.project_dir/iterations/cluster_*.json` — for each
      checkpoint, load it into the `done: dict[str, Artifacts]` and skip
      that cluster in the cluster loop.
- [ ] In `scripts/rdr_paperbench.py` + the bash wrapper for ad-hoc runs:
      wrap in a `while`-loop that retries up to 3× when the exit code is
      124 (watchdog kill), passing `--resume` on retries.
- [ ] Tests: `test_resume_skips_completed_clusters`,
      `test_resume_handles_missing_iterations_dir`.

### 7.3 SDK aclose investigation result

- [ ] When agent `a4c0d8038906e84c9` returns, decide between workarounds A/B/C
      from its report. **Likely choice:** workaround B (run SDK in a worker
      thread) or workaround A (separate Python subprocess) depending on
      complexity score.

### 7.4 Adversarial follow-up (user-requested)

- [ ] After all the above land, dispatch another Codex adversarial review on
      the FINAL branch state. Block merge on its CRITICAL findings.

### 7.5 Backward-compat smoke

- [ ] Verify `--mode offline / sdk / rlm` still work end-to-end on at least
      one paper. Quick (~30 min). The full test suite is already green so this
      is belt-and-braces.

### 7.6 Frontend integration (NEW SCOPE — see §8)

Substantial. Plan first, then implement, then merge with rdr in ONE PR.

### 7.7 Commit cleanup (10-15 noteworthy commits)

- [ ] Create a NEW branch (e.g., `rdr_merge_clean` or similar) off `main`.
- [ ] Cherry-pick or re-author commits from `rlm_rubric_orchestration`,
      arranging them as 10-15 noteworthy commits (NOT a 1-commit squash, NOT
      a force-push of the original branch).
- [ ] Each commit should be a "noteworthy" chunk: a feature, a bug class, a
      design doc, a fix family.
- [ ] Suggested grouping (final shape to be chosen with the user):
      1. Phase 0 scaffolding + models.py contracts
      2. Phase 1 — Rubric Decomposer
      3. Phase 2 — Context Engineer
      4. Phase 3 — Reproduction Agent
      5. Phase 4 — Controller + run entry
      6. Phase 5 — CLI `--mode rdr` + scripts/rdr_paperbench.py
      7. Phase 6 — Docs (CHANGELOG, system_overview, CLAUDE, learn, progress)
      8. Snapshot exclusion + defensive merge writes
      9. Scorer off the event loop (asyncio.to_thread)
      10. Watchdog + CWD guard
      11. Codex pre-merge hardening (path traversal, exception propagation, env-block)
      12. Retry-on-watchdog (`--resume`)
      13. Frontend integration (routes + SSE + auth)
      14. Final docs + scoreboard

      That's 14 commits — within the 10-15 window.

### 7.8 Final pre-merge checklist

See `2026-05-22-rdr-merge-plan.md` §7. All items must be checked.

---

## 8. Frontend integration scope (NEW)

User's binding decisions:
- Frontend on `main` already (the existing `frontend/` next.js app).
- Combined merge with rdr.
- Need ALL THREE of:
  - **Reuse existing `/api/demo` flow** — frontend POSTs `mode='rdr'`; backend dispatches `run_pipeline_rdr` instead of the current SDK/RLM path.
  - **New rdr-specific endpoints** — at minimum:
    - `GET /runs/<id>/clusters` — per-cluster status (id, title, leaf_ids, done, score).
    - `GET /runs/<id>/repair-iterations` — repair-pass count + per-pass clusters.
    - `GET /runs/<id>/leaf-scores` — per-leaf score with justification (post-scoring).
  - **Full live SSE of cluster checkpoints + repair events** — emit `dashboard_event` SSE frames from `run_rdr`:
    - `cluster_started` / `cluster_completed` per cluster.
    - `experiment_started` / `experiment_completed` for env+experiment phases.
    - `score_started` / `score_completed`.
    - `repair_pass_started` / `repair_cluster_completed` / `repair_pass_completed`.
- **Auth dynamic** — three modes selectable:
  - Existing X-Demo-Secret HMAC (current).
  - Proper auth: OAuth / signed JWT.
  - Per-user via Claude OAuth identity (when set).
  - Config via env vars (e.g., `REPROLAB_AUTH_MODE=demo|jwt|claude-oauth`).

### Frontend integration plan (high-level)

1. **Read** `frontend/src/components/lab/lab-shell.tsx`,
   `frontend/src/app/api/demo/*` (proxy routes), and the existing SSE
   consumer to understand current wiring.
2. **Backend changes:**
   - Add `dashboard_event` emission in `run_rdr` (controller.py) at each
     cluster transition. Existing SSE infrastructure
     (`backend/services/events/live_runs.py`) is reused; corpus-redaction
     applied at egress.
   - Add new GET routes in `backend/app.py` for cluster / repair / leaf-score
     introspection.
   - Modify `/api/demo` proxy to dispatch `--mode rdr` when `mode='rdr'` in
     the request body.
   - Add `REPROLAB_AUTH_MODE` config + the three auth paths
     (demo / jwt / claude-oauth).
3. **Frontend changes:**
   - Add `rdr` to the mode selector in `lab-shell.tsx`.
   - Add a rdr-specific run visualizer: cluster graph (Code-Dev → Code-Exec
     → Result-Analysis lanes), repair-pass timeline, leaf-score heatmap.
   - Consume the new SSE event types in `coalesceRunState`.
   - Add auth-mode awareness (login UI for the non-demo paths).
4. **Tests:**
   - Backend: route tests for the new endpoints; SSE emission tests.
   - Frontend: Playwright e2e for the rdr-mode happy path (offline e2e to
     avoid live LLM costs in CI).

### Frontend grilling — outstanding questions for the user

These need answers before frontend implementation begins:

- Visual layout for the cluster graph: lanes-by-category vs DAG vs flat list?
- Repair-pass UI: surface per-leaf score deltas vs just per-cluster?
- Auth-mode selection UI: a dropdown in the run-launcher? An admin setting?
- JWT-vs-Claude-OAuth: which provider? Auth0, custom, GitHub OAuth?
- SSE backpressure: how many cluster events per second is acceptable?

---

## 9. How to resume after compaction

1. **Read this doc first.**
2. **Read** `2026-05-22-rdr-merge-plan.md` (merge gate criteria).
3. **Check seqnn3** state — `runs/pb_sequential-neural-score-estimation_<latest>/` —
   `iterations/` count, presence of `final_report.json`, the `rubric.overall_score`.
4. **Check** background agents that may have completed (SDK aclose investigator
   `a4c0d8038906e84c9`).
5. **Resume the work queue** in §7 in order: seqnn3 score → retry-on-watchdog →
   frontend → adversarial review → commit cleanup → merge.
6. **Grill the user** on the open frontend questions in §8 before implementing.

### Standing constraints (always honor)

- Git remote: `origin` (`armaanamatya/openresearch`), NEVER `replix`.
- Commit messages: **no** `Co-Authored-By` / AI-attribution trailer.
- Branch: `rlm_rubric_orchestration` for current work; FINAL merge via a NEW
  clean branch (NOT force-push of the existing branch).
- Runs are serial (Claude OAuth + Featherless concurrency limits).
- Corpus-leak redaction at every egress (events, SSE, pickle, logs).
- Tests must be green before commits.
- The harness uses Claude OAuth (Sonnet) locally + dynamic provider routing
  for Azure OpenAI (untested live).
- "Always grill the user relentlessly" — surface every architectural worry,
  ask AskUserQuestion at every fork.
