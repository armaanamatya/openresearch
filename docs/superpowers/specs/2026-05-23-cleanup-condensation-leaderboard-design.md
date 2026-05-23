# OpenResearch — Cleanup, Condensation & Leaderboard — Design Spec

_Date: 2026-05-23 · Status: **design approved** (brainstorming) · Next: writing-plans._

This spec consolidates the cleanup pass with the next product surface (a persistent leaderboard with per-role model configuration). It is self-contained: §1–§8 are the design; §9 captures session context and the locked decisions that drove the design.

---

## 1. Motivation

The 2026-05-23 audit (see conversation transcript; not committed as a doc per §6.1) found `main` substantially healthier than the prior handoff assumed — the 28-task debug-and-harden delivery, Phase 6 cleanup, macOS Keychain OAuth fix, static-values audit salvage, the Phase-1 infra plan (`max_pod_seconds`), and incremental RunPod artifact sync are all merged and verified. The honesty cap is restored at `backend/evals/paperbench/leaf_scorer.py:92` (`DEGRADED_LEAF_CEILING = 0.35`). The exploration tree framing is restored at `frontend/src/components/lab/rlm/exploration-canvas.tsx:237-255`. Test suite is 1083 passed / 0 failed / 1 xfailed.

Two loads remain:

1. **`rdr` is implemented but unmerged.** `origin/rlm_rubric_orchestration` (10 commits ahead of `main`) carries a rubric-driven reproduction harness — a deterministic Python controller dispatching Claude reproduction agents per rubric cluster, in response to four observed RLM failure modes (wandering, degenerate baselines, under-scoping, rubric-blind reasoning). Abheek confirmed in chat (2026-05-23 ~02:00 local): *"The rdr stuff been done just not with the new frontend"* and *"the merge failed so I'm testing it."* He has already added `--mode {rlm,rdr}` to his branch's `backend/cli.py`.
2. **The product North Star, surfaced in the same chat**, is a **persistent leaderboard** that records every run with paper × mode × **per-role model config** (e.g. *"Claude plans, Kimi-K2.5 swarm executes, GPT-5 verifies"*), allows side-by-side comparison, and supports re-running any row with a swapped model assignment. Phrased by Abheek: *"a leader board mode that's persistent and allows us to use and configure different models for diff parts. Like Claude plans and Kimi swarm executed with recursive workers."*

A Microsoft VP review is imminent, raising the bar on demo readiness and honest scoring.

This spec covers landing rdr, closing the honesty queue, condensing the repository surface, and shipping the leaderboard — in one coherent plan.

## 2. North Star

A single `main` carries two clean reproduction modes (`rlm`, `rdr`) integrated with one lab UI, plus a persistent **Leaderboard** surface that records every run with paper × mode × per-role model config, exposes side-by-side comparison, and lets a viewer re-run any row with a different model assignment — all on top of a condensed ≤12-doc, zero-stale-branch repo.

Measured as:
- `find docs -name "*.md" | wc -l` ≤ 12.
- `git branch -a | wc -l` ≤ 3.
- All 5 PaperBench papers have at least one honest score per mode visible in `/leaderboard`.
- `gh issue list --label rlm-pivot --state open` returns 0.
- Working tree clean (`git status` empty) before each phase's PR is opened.

## 3. Locked decisions

| # | Question | Decision |
|---|---|---|
| 1 | Default `--mode` post-Phase-1 | **`rlm` (now meaning the hybrid: RDR phase 1 + RLM phase 2, as landed by PR #80 on 2026-05-23).** `--mode rlm-pure` is the escape hatch to the pre-hybrid pure-RLM path; `--mode rdr` runs the rdr controller standalone. Phase 2 honesty re-score still gates *empirical* validation of hybrid vs the others. |
| 2 | PR #74 (wow-factor poll) fate | **Shrink to ≤2 files (`docs/wow-factor-poll.html` + the Discord post text) and merge.** VP review is imminent; the poll is useful, the demo-track PNGs and other 26 files are session noise. |
| 3 | CLI naming for per-role models | **Keep `--model <id>` as shorthand for `planner` role; add `--models planner=…,executor=…,verifier=…,grader=…` for full per-role control.** Both flags coexist; `--models` overrides if both are passed. |
| 4 | Leaderboard re-run default budget | **Orchestrator-estimated, user-accepted, safety-ceiling-enforced.** At run start the orchestrator (rlm root / rdr controller) proposes `{usd, wall_clock_s, pod_seconds}` from the paper + rubric; user accepts or overrides via the UI; accepted value is the hard cap. Safety ceiling: `$20 / 2h / 7200 pod-seconds` — the maximum the orchestrator can request. Leaderboard re-runs inherit the original run's accepted budget. |
| 5 | Doc cleanup strategy | **`git rm` outright, no archive directory.** History retains everything. Archives accumulate decay (the audit just flagged that). |
| 6 | Leaderboard route name | **`/leaderboard`.** |
| 7 | Mode disposition | **Three modes kept indefinitely as peers.** `rlm` (hybrid, default), `rlm-pure` (pure-RLM escape), `rdr` (pure-rdr peer). Leaderboard is the comparison surface; none of the three is deprecated. Updated 2026-05-23 to reflect PR #80's hybrid landing. |

## 4. Phasing

Five phases, each a self-contained PR family. Phases 1–3 are cleanup + integration; Phases 4–5 are the new product surface. Phase 3 can run in parallel with Phase 1 reconciliation (different files, no shared state).

| Phase | What lands | Wall-clock | Blocks |
|---|---|---|---|
| **0** | This spec saved + writing-plans handoff | 30 min | precondition |
| **1** | rdr backend merged + lab UI extended for rdr events; both modes work end-to-end | 2–3 sessions | 2, 4 |
| **2** | T7 (ftrl), T24 (ThreadPool), re-score 5 disk-runs, verify rdr honesty cap | 0.5 session | honest leaderboard data |
| **3** | Docs 50+ → ≤12 canonical; branches; issues; working tree | 0.5 session | independent of 1 |
| **4** | Leaderboard: schema → backend → API → UI → per-role model config + dynamic budget | 3–4 sessions | needs 1 + 2 |
| **5** | Demo polish: pinned demo run, README rewrite, PR #74 shrunk + merged | 0.5 session | needs 4 |

Total: ~7–9 sessions serially. Phase 3 in parallel with Phase 1 backend conflict reconciliation saves ~0.5 session.

## 5. Phase specifications

### 5.1 Phase 0 — Pre-work

- **0.1** This spec saved to `docs/superpowers/specs/2026-05-23-cleanup-condensation-leaderboard-design.md`.
- **0.2** Spec self-review (placeholder scan, internal consistency, scope check) per the brainstorming skill — fixed inline.
- **0.3** User reviews the spec; on approval, hand off to `writing-plans` for the executable per-task plan.
- **0.4** Async coordination with Abheek: confirm he's paused on `rlm_rubric_orchestration` so the merge isn't a moving target; obtain his exact known-good rdr invocation (`python -m backend.cli reproduce ... --mode rdr ...`) for post-merge regression checking.

**Acceptance.** Spec committed; user approves; Abheek confirms branch is stable.

### 5.2 Phase 1 — Land rdr on main, integrate with the lab UI

#### 5.2.1 Pre-merge survey (1.1)

- Pull `origin/rlm_rubric_orchestration` into `git worktree add ../or-rdr rlm_rubric_orchestration`.
- Run `git diff origin/main..origin/rlm_rubric_orchestration --stat` and group conflicts into: `backend-rlm-overlap`, `frontend-overlap`, `cli-overlap`, `docs-overlap`, `infra-overlap`, `rdr-additive`.
- Inspect `backend/cli.py` on the rdr branch to read Abheek's exact `--mode` wiring (already merged into his branch — see §3 decision #3).
- Run `pytest tests/` on the rdr branch to capture his baseline; record the count.
- **Output**: a working note (not a committed doc) listing each conflicted file + the resolution rule that applies.

**Acceptance.** Conflict map written; no merge attempted yet.

#### 5.2.2 Reconcile backend conflicts (1.2)

For each conflict, apply this rule set in order:

1. **Honesty/correctness lines** (the C2 cap in `leaf_scorer.py:92,300`; `_is_degraded_run` at `leaf_scorer.py:101`; the in-loop `degraded` predicate at `primitives.py:787`; the C2c `meets_target` computation at `leaf_scorer.py:395`) → **keep `main`'s version.** These are the load-bearing fixes from PR #75; not negotiable.
2. **rdr-additive lines** (any new file under `backend/agents/rdr/`; the `--mode rdr` dispatch in `backend/cli.py`; any new SSE event type) → **keep rdr's version.** Pure additions.
3. **Overlapping fixes to `backend/agents/rlm/primitives.py`** (Abheek's `1581cad fix(rlm): verify_against_rubric returns no areas, root fabricates blank ones` and `7696d5d fix(rlm): bundle paper id/title not reaching context["paper_metadata"]` vs. main's T11/T20 work) → **merge by intent**: re-apply Abheek's fixes on top of main's structure. Both authors agreed the fix existed; the question is mechanical.
4. **Infra layer** (`backend/services/runtime/runpod_backend.py`, `RunBudget`, `Sandbox.created_at`, the incremental artifact sync from `ea95b0f`) → **keep `main`.** The `max_pod_seconds` work is finished on main; rdr inherits it for free.

**Acceptance.** `pytest tests/` returns ≥1083 passed / 0 failed / 1 xfailed (T24 still xfailed until Phase 2.2). No new failures introduced by the merge. `npm run lint && npx tsc --noEmit && npm test` green in `frontend/`.

#### 5.2.3 Frontend integration — one canvas, two layouts (1.3)

Design call: the existing `ExplorationCanvas` (`frontend/src/components/lab/rlm/exploration-canvas.tsx`) renders a dynamic candidate-spawning tree shaped by rlm's `candidate_proposed` events. rdr emits a static rubric tree shaped by the PaperBench `rubric.json` (clusters as inner nodes, leaves as terminals).

**Approach.** Keep one canvas, two layout modes. `useRlmRun` reads a `mode` from the run metadata; the layout function (`layoutTree`) picks `dynamicCandidateLayout` for rlm or `rubricClusterLayout` for rdr. Empty regions stay empty honestly when an event type doesn't apply to the active mode.

**Concrete steps.**

- Define 4 new SSE event types with strict shapes; bridge via `backend/agents/rlm/sse_bridge.py` (kept name despite serving both modes — internal-only):
  - `cluster_started`: `{ cluster_id, cluster_title, leaves: [{id, weight, requirements}], iteration: int }`
  - `cluster_artifact_emitted`: `{ cluster_id, artifact_path, byte_size, language: str | null }`
  - `cluster_scored`: `{ cluster_id, score: float, leaf_scores: {leaf_id: float}, degraded: bool }`
  - `repair_dispatched`: `{ cluster_id, attempt: int, prior_score: float, failed_leaves: [leaf_id] }`
- Extend `RlmRunState` (`frontend/src/components/lab/rlm/use-run.ts` or equivalent) with `clusterStatus: Record<clusterId, "pending" | "running" | "scored" | "repaired" | "failed">` and `clusterScores: Record<clusterId, number>`.
- `NodeDetailPopup` adds a cluster-mode branch: shows cluster title, leaves grading the cluster, current artifacts list, score breakdown.
- Region 2 (Rubric Score Bar) is mode-agnostic; reuse as-is.
- Region 4 (Exploration Tree) gets the dual-layout treatment.
- Region 5 (Rubric Timeline + Root Reasoning) — for rdr, "root reasoning" is the controller's deterministic next-cluster transition; rendered as a timeline of cluster-state changes rather than LLM reasoning text.
- Fixture: add `frontend/src/components/lab/rlm/fixtures/rdr-run.fixture.ts` mirroring `rlm-run.fixture.ts`; wire `?rdrFixture=1` query param in `/lab`.

**Acceptance.**
- `npm test` green (target ≥85 tests).
- `?rdrFixture=1` renders all six regions without console errors.
- A real `--mode rdr` run (using Abheek's known-good invocation from §5.1.0.4) drives the lab end-to-end without diverging from the fixture's contract.

#### 5.2.4 CLI surface (1.4) — already mostly done

Abheek's branch already adds `--mode {rlm,rdr}` to `backend/cli.py`. Verify the wiring survives the merge:

- `python -m backend.cli reproduce <paper> --mode rlm` dispatches to `backend/agents/rlm/run.py::run_pipeline_rlm`.
- `python -m backend.cli reproduce <paper> --mode rdr` dispatches to `backend/agents/rdr/run.py::run_pipeline_rdr` (or whatever entrypoint Abheek named).
- `--help` shows supported modes cleanly. Remove any leftover deleted-mode references in `--help`, README, CLAUDE.md.
- Default mode remains `rlm` per §3 decision #1.

**Acceptance.** Both modes run end-to-end against `2512.24601` (an arXiv ID that ingests cleanly).

#### 5.2.5 Phase 1 PR (1.5)

- Branch: `feat/rdr-merge-and-frontend`.
- Title: *"Land rdr + extend lab UI to render rubric-cluster layout."*
- Closes #63 (Phase 6 follow-up) and ticks the #64 Phase 6 checkbox.

**Risks.**
- **Merge math >30 conflicted files** → fall back to cherry-picking rdr's 10 commits one-at-a-time onto a fresh branch off `main`.
- **Abheek pushes more rdr fixes mid-merge** → he holds his branch until the reconciliation PR is open; he rebases on top of the PR, not on `main`.
- **Lab UI dual-layout is structurally larger than estimated** → split Phase 1 into 1a (backend merge) and 1b (frontend integration, deferred).

### 5.3 Phase 2 — Close honesty queue

#### 5.3.1 T7 — `ftrl` reproduces wrong method (2.1)

- Inspect `third_party/paperbench/ftrl/paper.md` against the actual ftrl arXiv abstract (*Fine-tuning RL Models is Secretly a Forgetting Mitigation Problem*, ICML 2024).
- **If the bundle's `paper.md` is the RLM paper text under the wrong path** (vendoring bug): re-fetch the real ftrl bundle from `openai/preparedness` via LFS; add `tests/test_paperbench_bundle_identity.py` that asserts each bundle's `paper.md` first-100-chars matches its directory name's expected title (table of expected titles maintained in the test).
- **If the bundle is correct and the root mis-grounded**: add a comprehension guard in `backend/agents/rlm/primitives.py::understand_section` that on the first call records the inferred paper title and re-asserts it on every subsequent call (catches drift mid-run).

**Acceptance.** A fresh `--mode rdr` run on ftrl either produces an honest non-zero score OR a `degraded=True` outcome capped at 0.35 — never a silent 0.000.

#### 5.3.2 T24 — `build_environment` ThreadPoolExecutor shutdown hang (2.2)

Three sites in `backend/agents/rlm/primitives.py:296, 437, 729`. Replace each:
```python
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
    ...
```
with:
```python
pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
try:
    ...
finally:
    pool.shutdown(wait=False, cancel_futures=True)
```

Mirror the fix in any equivalent pool sites added by rdr.

Remove `@pytest.mark.xfail` from `tests/rlm/test_build_environment_timeout.py:11`.

**Acceptance.** Full suite has 0 xfailed; no new failures.

#### 5.3.3 Re-score existing disk-runs under the restored C2 cap (2.3)

One-shot script `scripts/rerun_amend_2026-05-23.py`:
- Walks `runs/pb_*/final_report.json`.
- For each, calls `leaf_scorer.amend_final_report(run_dir)`.
- Logs before/after `overall_score` + `degraded` flag to `progress.md` as a single table.

**Acceptance.** One-page table committed; ftrl confirmed `degraded=True` capped ≤0.35; any other run with `baseline_metrics={}` honestly downgraded.

#### 5.3.4 Verify rdr respects the cap (2.4)

rdr reuses `leaf_scorer.score_reproduction` and `amend_final_report` (per the rdr design doc §4.7) — the cap should apply for free, but verify:

- Add `tests/rdr/test_rdr_honesty_cap.py`: runs rdr's controller against a stub bundle where the executor returns `metrics={}`; asserts the final score is ≤ 0.35.

**Acceptance.** Test passes; cap is no longer rlm-specific.

#### 5.3.5 Phase 2 PR (2.5)

- Branch: `fix/honesty-queue-cleanup`.
- Title: *"Close T7/T24/re-score; verify rdr honesty cap."*

### 5.4 Phase 3 — Aggressive condensation

#### 5.4.1 Docs: 50+ files → ≤14 tracked docs (3.1)

**Keep (the canonical 8):**

| Path | Purpose |
|---|---|
| `README.md` | Lead with what the system does. |
| `CLAUDE.md` | Day-to-day reference (already canonical). |
| `system_overview.md` | "Why and how it fits together." |
| `CHANGELOG.md` | Release log. |
| `progress.md` | Project journal. |
| `learn.md` | Post-mortem log. |
| `docs/design/rlm-pivot-brief.md` | RLM architecture reference and design rationale. |
| `docs/superpowers/specs/2026-05-23-rubric-climb-leaderboard.md` | Leaderboard and rubric-climb spec. |

**Delete (via `git rm`, no archive):**

- Retired Phase 2 design files (Phase 2 merged via #68).
- Retired Phase 3-5 review findings (closed by #75).
- Retired spike reports.
- The 2026-05-22 project-state audit artifact (superseded by the 2026-05-23 audit conversation).
- The 2026-05-23 project-state analysis prompt artifact (working artifact).
- The 2026-05-22 audit screenshots artifact (working artifact).
- `docs/design/cross-platform-encoding-{audit,fix}.md` (resolved).
- Retired observability notes superseded by current observability.
- The 2026-05-22 static-values audit artifact (merged via #78).
- All 12 `docs/superpowers/plans/2026-05-09-*.md` (issues #17–#21 all merged).
- All 3 `docs/superpowers/plans/2026-05-14-*.md` (rebuild + polish merged).
- All `docs/superpowers/plans/2026-05-2[12]-*.md` (Phase 4/5/6 + infra plan + debug-harden plan + static-values plan — all delivered).
- All `docs/superpowers/specs/2026-05-09-*.md`.
- All `docs/superpowers/specs/2026-05-2[12]-*.md` **except**:
  - `2026-05-22-rubric-driven-harness-design.md` and `2026-05-22-rubric-driven-harness-implementation.md` — both consumed into `architecture.md` then deleted at the end of Phase 4.
  - This spec (`2026-05-23-cleanup-condensation-leaderboard-design.md`) — deleted at the end of Phase 5.
- `docs/superpowers/specs/pr-body-*.md` (PR bodies — git history has them).

**Acceptance.** `find docs -name "*.md" | wc -l` ≤ 12.

#### 5.4.2 Branches (3.2)

- After Phase 1 merges: delete `origin/rlm_rubric_orchestration` (locally and on origin).
- Delete `origin/claude/great-tesla-xor5z` (PR #74's branch — after Phase 5 disposes of it).
- Verify with `git branch -r --merged origin/main | grep -v main`.

**Acceptance.** `git branch -a` returns `main` + at most one in-flight feature branch.

#### 5.4.3 PR #74 — deferred to Phase 5 (3.3)

PR #74 is the wow-factor poll. The merge / shrink happens in Phase 5 alongside the demo polish, not here, because the VP-review timing influences scope. Mention only — no action in Phase 3.

#### 5.4.4 Issues (3.4)

- `gh issue close 63 --comment "Closed by #72 (Phase 6) + this cleanup PR."`
- `gh issue close 64 --comment "All 6 phases delivered; superseded by rdr + leaderboard work tracked in docs/superpowers/specs/2026-05-23-cleanup-condensation-leaderboard-design.md."`

**Acceptance.** `gh issue list --state open --label rlm-pivot` returns 0.

#### 5.4.5 Working tree (3.5)

- Commit `frontend/next-env.d.ts` Next codegen drift (harmless, regenerable, but blocks clean merges).
- Delete `.claire/` if present (typo of `.claude/` from prior audit).

**Acceptance.** `git status` returns clean.

#### 5.4.6 Phase 3 PR (3.6)

- Branch: `chore/condense-docs-branches-issues`.
- Title: *"Condense 50+ docs to 8 canonical; close #63/#64; clean working tree."*

### 5.5 Phase 4 — Leaderboard mode

The new product surface. Per §3 decision #4, the budget model is orchestrator-estimated, user-accepted, safety-ceiling-enforced — not a fixed cap.

#### 5.5.1 Data model (4.1)

Extend `runs/<id>/final_report.json` with:
```json
{
  "models": {
    "planner":  "claude-opus-4-7",
    "executor": "kimi-k2.5",
    "verifier": "claude-sonnet-4-6",
    "grader":   "claude-sonnet-4-6"
  },
  "mode": "rdr",
  "models_hash": "<sha256 of the ordered (role,model) pairs>",
  "budget": {
    "estimated":  { "usd": 3.50, "wall_clock_s": 1800, "pod_seconds": 1200 },
    "accepted":   { "usd": 3.50, "wall_clock_s": 1800, "pod_seconds": 1200 },
    "consumed":   { "usd": 2.87, "wall_clock_s": 1456, "pod_seconds":  988 }
  }
}
```

_Field name `models` (not `model_config`) — Pydantic v2 reserves `model_config` as a BaseModel attribute. Live since `a02ccd9` (`RLMFinalReport.models`). The rubric-climb subset spec made the call first; this spec adopts it._

New SQLite table `leaderboard_runs`, written via projection from `final_report.json` on `_finalize`:

| Column | Type | Index |
|---|---|---|
| `run_id` | TEXT PK | |
| `project_id` | TEXT | |
| `paper_id` | TEXT | (paper_id, mode), (paper_id, finished_at_utc DESC) |
| `mode` | TEXT | |
| `models_hash` | TEXT | (models_hash, paper_id) |
| `models_json` | TEXT | |
| `overall_score` | REAL | |
| `leaf_score` | REAL | |
| `degraded` | INTEGER (0/1) | |
| `cost_usd` | REAL | |
| `wall_clock_s` | REAL | |
| `pod_seconds` | REAL | |
| `finished_at_utc` | TEXT (ISO-8601) | (finished_at_utc DESC) |
| `rubric_source` | TEXT (`bundle` / `generated`) | |
| `status` | TEXT | |

**Acceptance.** Schema migration applied; a fresh `--mode {rlm,rdr}` run lands a row in the table; the projection is idempotent.

#### 5.5.2 Per-role model configuration (4.2)

Roles: `planner` (rlm root / rdr controller LLM), `executor` (rlm sub-agent / rdr reproduction agent), `verifier` (`verify_against_rubric` primitive), `grader` (leaf_scorer's LLM).

- **rlm**: `--model <id>` keeps current behavior (sets the root) — see §3 decision #3. Add `--sub-model <id>` for the executor role. Plumb both into `RootModel` resolution.
- **rdr**: `backend/agents/rdr/agent.py` generalizes its `claude_model` kwarg to a `models: dict[str, str]` param. Controller passes the per-role model on each agent invocation.
- New CLI flag: `--models planner=<id>,executor=<id>,verifier=<id>,grader=<id>` — full per-role control. `--models` overrides `--model` and `--sub-model` if both are passed.
- Default `models` per mode lives in `backend/agents/<mode>/defaults.py`.

**Acceptance.** `python -m backend.cli reproduce <paper> --mode rdr --models planner=claude-opus-4-7,executor=claude-opus-4-7,verifier=claude-sonnet-4-6,grader=claude-sonnet-4-6` runs; `final_report.json` shows the actual IDs used per role.

#### 5.5.3 Dynamic budget — orchestrator-estimated, user-accepted (4.3)

New flow:

1. At run start, the orchestrator (rlm root / rdr controller) reads `paper.md` + `rubric.json` (or `generated_rubric.json` for arXiv runs) and proposes a budget: `{usd, wall_clock_s, pod_seconds}`.
2. The proposal is bounded by a **safety ceiling**: `usd ≤ 20`, `wall_clock_s ≤ 7200`, `pod_seconds ≤ 7200`. Proposals exceeding the ceiling are clamped with a warning logged to the run.
3. The UI (lab page and leaderboard re-run modal) shows the proposal and gives the user **Accept** / **Override** / **Cancel**.
4. The accepted budget becomes the hard cap for `RunBudget` and is recorded in `final_report.json` under `budget.accepted`.
5. The actual consumption is recorded under `budget.consumed` on run completion.
6. Leaderboard re-runs default to inheriting the original run's `budget.accepted`.

For CLI / non-UI runs, the budget is either passed via flag (`--max-usd $X --max-wall-clock <s> --max-pod-seconds <s>`) or accepted automatically up to a per-mode default ceiling (rlm: `$5`, rdr: `$3`).

**Acceptance.**
- An orchestrator estimate appears in `demo_status.json` within 10s of run start.
- The UI surfaces the proposal and the user can override before the first sub-agent call.
- Safety ceiling enforced at the budget-construction boundary (test that proposing `$50` clamps to `$20`).

#### 5.5.4 Backend API (4.4)

- `GET /leaderboard?paper=...&mode=...&models_hash=...&order_by=score|cost|time&limit=N` — returns ranked rows.
- `GET /leaderboard/runs/{run_id}` — full run detail (proxies existing `/runs/{id}`).
- `POST /leaderboard/rerun` — body: `{ paper_id, mode, models, budget_override? }`, returns the new `run_id`; spawns a new run via the existing subprocess path.

All endpoints under the existing FastAPI app; gated by `REPROLAB_DEMO_SECRET` if set (same as current demo gate).

**Acceptance.** Each endpoint has a Pydantic schema, a unit test, and an integration test that exercises a 2-row leaderboard.

#### 5.5.5 Frontend (4.5)

- New route `/leaderboard` (Next.js page at `frontend/src/app/leaderboard/page.tsx`).
- Table columns: paper, mode, planner, executor, verifier, grader, leaf score, cost, time, finished_at. Sortable by any column.
- Filters: paper (multi-select), mode (rlm/rdr), score range, degraded (yes/no).
- Row click → existing `/lab?run=<id>` (read-only since the run is finished).
- "Re-run with config" button per row → opens a modal with the model picker (one dropdown per role, defaults pre-filled from the row); on submit, fetches the orchestrator estimate, shows it for accept/override, then calls `POST /leaderboard/rerun` and navigates to `/lab?run=<new_id>` (live).
- 3 named model presets in the picker: **Default**, **Cheap**, **Mixed-Vendor** — plus a **Custom…** option that opens the full picker.
- Reuses the existing `frontend/src/components/library/` table primitives (LibraryFilters, etc.) — do not build a parallel table component.

**Acceptance.** `/leaderboard` shows ≥2 runs across ≥2 papers across both modes; re-run button spawns a new run that appears in the table on completion; a Playwright e2e drives the flow end-to-end.

#### 5.5.6 Cost discipline (4.6)

Per §3 decision #4, leaderboard re-runs default to `--sandbox local` when no sandbox flag is passed via the rerun UI. RunPod sandbox requires explicit opt-in in the modal.

**Acceptance.** A re-run with default settings does not allocate a RunPod pod.

#### 5.5.7 Phase 4 PRs (4.7)

Split into two PRs to keep diffs reviewable:

- `feat/leaderboard-backend`: schema + projection + API + per-role model config + dynamic budget + tests.
- `feat/leaderboard-frontend`: `/leaderboard` page + re-run modal + e2e.

### 5.6 Phase 5 — Demo polish

#### 5.6.1 Pin a canonical demo run (5.1)

Pick the highest-scoring honest run from the leaderboard (post-Phase-2 re-scoring), pin its `run_id` in `README.md` and as the default-highlighted row on `/leaderboard`.

#### 5.6.2 README rewrite (5.2)

Lead with: *"OpenResearch reproduces research papers end-to-end and scores them against PaperBench rubrics. Pick a paper, pick a reproduction mode, pick the models for each role, and run."* Two screenshots: the lab view of the canonical run; the leaderboard with that run pinned at top. Short setup section; the rest moves to `system_overview.md`.

#### 5.6.3 PR #74 (5.3)

Shrink PR #74's branch to ≤2 files: `docs/wow-factor-poll.html` + the Discord post text. Drop the demo-track PNGs and any other auto-generated content. Merge.

**Acceptance.** PR #74 merged with ≤2-file diff; `claude/great-tesla-xor5z` branch deleted.

#### 5.6.4 Demo gate (5.4)

Confirm `REPROLAB_DEMO_SECRET` gating is enabled by default in the Docker compose file. Note in README that local dev runs unsecured.

#### 5.6.5 Phase 5 PR (5.5)

- Branch: `chore/demo-polish`.
- Title: *"Demo polish: pin canonical run; README rewrite; shrink+merge PR #74."*

## 6. Out of scope

- **CI workflow.** Recommended; valuable; separate plan.
- **Test corpus expansion** (3 → 10 PaperBench papers). Defer; parallel-track effort.
- **Persistent pod reuse / verdict caching** (infra plan B.2, B.5). Defer until rdr stabilizes.
- **`--mode rlm` deprecation.** Both modes stay indefinitely per §3 decision #7.
- **New paper-reproduction harnesses** beyond rlm and rdr. Two is enough until the leaderboard surfaces clear comparison data.
- **Authentication / user accounts.** Demo gate remains the only auth layer.

## 7. Risks

1. **Phase 1 merge math** > 30 conflicted files → fallback: cherry-pick rdr's 10 commits one-at-a-time onto a fresh branch off `main`.
2. **rdr honesty regression discovered in Phase 2** → cap fix lands as a Phase 1 amendment; Phase 2 acceptance waits.
3. **Leaderboard cost explosion** — even with dynamic budgets, a user re-running every paper × every mode × N configs burns money. Mitigated by: default to local sandbox; orchestrator estimate visible before commit.
4. **Frontend dual-layout complexity** (5.2.3) underestimated → split Phase 1 into 1a (backend merge) and 1b (frontend integration, deferred one session).
5. **Per-role model picker UX** — four dropdowns can become a wall. Mitigated by named presets (Default / Cheap / Mixed-Vendor).
6. **rdr's frontend integration assumes the lab is the primary entry point**, but the leaderboard may become primary in Phase 4. Mitigated by treating the lab as a per-run drill-down view from Phase 4 onward; no UI work is wasted, just reframed.
7. **Orchestrator budget estimate is wildly wrong** — e.g. proposes $20 for a trivial paper. Mitigated by safety ceiling (`$20` max) and the user-accept gate. Track estimate-vs-consumed in `learn.md` over the first 10 runs to recalibrate the estimator prompt.

## 8. Acceptance criteria (rolled up from §2)

A reviewer can confirm the plan is complete by checking:

1. `find docs -name "*.md" | wc -l` ≤ 12.
2. `git branch -a | wc -l` ≤ 3.
3. `gh issue list --label rlm-pivot --state open` returns 0.
4. PR #74 either merged with ≤2 files or closed.
5. `pytest tests/` returns ≥1083 passed / 0 failed / 0 xfailed.
6. `npm test && npm run lint && npx tsc --noEmit` green in `frontend/`.
7. `/leaderboard` displays ≥2 runs across ≥2 papers across both modes.
8. `final_report.json` of every run after Phase 4 lands contains `models`, `mode`, and `budget` keys.
9. A "Re-run with config" action on `/leaderboard` produces a new live run that appears in the table on completion.
10. `git status` clean.
11. ftrl's `final_report.json` either shows a non-zero honest score or `degraded=True` with `overall_score ≤ 0.35`.

## 9. Session context (carry forward)

**Working mode.** `/iterate` — PhD-researcher + principal-engineer rigor, root-level elegant solutions, fix the invariant with one canonical abstraction + a guard test, not scattered patches. Quality bar: precise, accurate, 100% reliable, elegant.

**Delegation.** Opus owns design, contracts, numerical correctness, and reviewing every diff. Sonnet sub-agents implement against tight specs. Codex is review/adversarial/diagnosis only — never implementation code. Parallel sub-agents for disjoint file sets.

**Commits.** Infrequent — one commit per major milestone, not per fix. Never a `Co-Authored-By` / AI-attribution trailer. Branch in `feat/…` / `fix/…` / `chore/…` style, fast-forward `main`, push to `origin` (`armaanamatya/openresearch`) — never `replix`.

**Documentation.** `progress.md` current and succinct. `learn.md` is the post-mortem log (Symptom → Root cause → Fix → Lesson → Guardrail). `CHANGELOG.md` `[Unreleased]`. Keep `system_overview.md` and `CLAUDE.md` in sync with architectural drift.

**Operational facts (carried from 2026-05-22).** RLM runs are serial — Featherless `feather_pro_plus` caps at 4 concurrent units; one RLM run saturates it. Re-running a paper needs both stores purged — run dir + `event_store_events` aggregates in `reprolab.db`. macOS Keychain stores Claude OAuth creds (not `~/.claude/.credentials.json`); both `validate_provider_credentials` and `has_provider_credentials` probe the Keychain on darwin (fixed 2026-05-23 in #77).

**Audit baseline (2026-05-23).**
- `main` HEAD: `ecb7931` (PR #78 merge).
- Test suite: 1083 passed / 0 failed / 1 xfailed (`tests/rlm/test_build_environment_timeout.py`) / 5 skipped (optional-dep gated).
- Honesty cap restored at `backend/evals/paperbench/leaf_scorer.py:92`.
- Phase 6 (#72) merged; 14-stage pipeline fully deleted from `backend/`.
- Deleted legacy CLI modes are gone.
- Five real `runs/pb_*/final_report.{md,json}` exist on disk but were scored pre-fix; Phase 2.3 re-scores them.
- `origin/rlm_rubric_orchestration` is 10 commits ahead of `main` with `--mode rdr` already implemented + `--mode {rlm,rdr}` flag added to `backend/cli.py` by Abheek.
- Open issues: #63 (Phase 6 — merged via #72 but issue still open), #64 (RLM pivot umbrella — Phase 6 checkbox unticked).
- Open PRs: #74 (wow-factor poll, draft, CONFLICTING, 28 files / +6258, demo prep for Microsoft VP review).

---

## Next step

Hand off to `writing-plans` to produce the executable per-task plan (file paths, test commands, commit boundaries, TDD steps) — starting with Phase 0 (commit this spec, async Abheek coordination, conflict map) and Phase 1 (rdr merge + frontend integration).
