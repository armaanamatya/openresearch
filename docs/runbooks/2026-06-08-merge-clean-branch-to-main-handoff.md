# Handoff — Merge the clean Azure line → `origin/main` (2026-06-08)

**Objective.** Land our work (Azure AKS GPU backend + the full `5.30.26_sdar` line)
onto `origin/main` as one clean, conflict-resolved merge.
**Significant (code) conflicts are resolved MANUALLY by the operator** — the agent
explains each diff and proposes a resolution, the human decides and commits. The
agent may auto-resolve only the *trivial* tier (docs/run-artifacts/.gitignore),
and only after listing exactly what it did.

---

## 1. State at handoff

| Thing | Value |
|---|---|
| Deliverable branch (pushed) | `feat/azure-aks-gpu-clean` — 6 commits, all `lolout1 <lolout1@users.noreply.github.com>`, **tree byte-identical** to `feat/azure-aks-gpu` (`bde6bcf`) |
| Merge target | `origin/main` |
| Fork point (merge-base) | `1522579` |
| `main` ahead of fork | **146 commits** (135 `lolout1 <appradhann@gmail.com>` + teammate merges #96/#97 by Aayush/Armaan) |
| our line ahead of fork | **205 commits** (= `5.30.26_sdar` 179 + 26 azure/scoring) |
| `main` ↔ `5.30.26_sdar` | **independent forks of `1522579`, never merged** |

`main` and our line are **parallel re-integrations of largely the same project.**
- **`main` uniquely has:** ReproLab→OpenResearch rename, CLAUDE.md compaction (41→25 KB), `.mailmap`, teammate merge-PRs, the azure *scope* doc (`docs/superpowers/specs/2026-06-02-azure-gpu-compute-backend-scope.md`) — **but no azure implementation.**
- **Our line uniquely has:** the **Azure AKS GPU backend** (impl of main's scope doc), scoring-fairness layer, env-reliability fixes (`env_pin`/libcupti), and the BES/full-scope work folded into `5.30.26_sdar`.
- **Shared & integrated on BOTH lines (the conflict source):** harden/root-harness, F-03…F-35 fixes, recent-runs panel + leaderboard, SDAR core.

---

## 2. Conflict surface (measured — isolated trial merge of `origin/main` into the clean branch)

- 214 files changed on both sides → **87 actually conflict** (127 auto-merge cleanly).
- **Azure backend = purely additive on main → ZERO azure conflicts.** All 87 are in the shared RLM core/tests/docs both lines evolved.

| Tier | Count | Who resolves |
|---|---|---|
| **Significant — backend code** | 25 | **operator, manually** |
| **Significant — scripts / frontend** | 3 / 3 | **operator, manually** |
| Tests | 33 | operator-guided (resolve *after* the source they cover) |
| Trivial — docs / run-artifacts / `.gitignore` | 23 | agent may auto-resolve (union / pick-newer), then report |

### Significant code conflicts (resolve by hand)
```
backend/agents/baseline_implementation.py      backend/agents/rlm/report.py
backend/agents/rlm/accelerator.py              backend/agents/rlm/rl_scaffold.py
backend/agents/rlm/cell_matrix.py              backend/agents/rlm/run.py
backend/agents/rlm/claude_oauth_client.py      backend/agents/rlm/sdar_env_base.py
backend/agents/rlm/context.py                  backend/agents/runtime/base.py
backend/agents/rlm/executor.py                 backend/agents/runtime/claude_runtime.py
backend/agents/rlm/failure_classifier.py       backend/agents/runtime/invoke.py
backend/agents/rlm/gpu_cell_runner.py          backend/cli.py
backend/agents/rlm/preflight_ast.py            backend/evals/paperbench/leaf_scorer.py
backend/agents/rlm/primitives.py               backend/routes/replay.py
backend/services/context/workspace/tools/rlm_query.py   backend/services/events/live_runs.py
backend/services/ingestion/parser/service.py   backend/services/runtime/gpu_capacity.py
backend/services/runtime/local_process.py
scripts/batch_reproduce.py  scripts/reserve_and_run_sdar.py  scripts/serve_local_llm.py
frontend/src/app/api/demo/replay-events/route.ts
frontend/src/lib/leaderboard/server-fetch.ts   frontend/src/lib/user-prefs.ts
```
> `primitives.py`, `run.py`, `report.py`, `cell_matrix.py`, `gpu_cell_runner.py`,
> `leaf_scorer.py`, `gpu_capacity.py` are the hot ones — they carry **both** the
> azure threading (ours) and main's parallel fixes. Resolve these first and slowly.

---

## 3. Sharp edges (read before touching a conflict)

1. **Rename `REPROLAB_` → `OPENRESEARCH_`.** `main`'s `backend/config.py` sets
   `env_prefix="OPENRESEARCH_"` with backward-compat `REPROLAB_` aliases. Our azure
   code reads both Pydantic settings (`settings.azure_*`) **and** raw
   `REPROLAB_CELL_*`/`REPROLAB_AZURE_*` env vars across the cross-process cell seam.
   **After merge, prove both still resolve** (test below). Keep the seam env-var
   names identical on the runner-inject and entrypoint-read ends (that bug already
   bit us once — see `tests/agents/rlm/test_azure_env_contract.py`).
2. **Duplicated shared work** → a conflict here is usually *two versions of the same
   fix*. Resolution = **take the superset / later fix**, never hand-write new logic.
   The F-numbering (F-03…F-35) and the audit backlog
   (`docs/audits/2026-05-31-backend-core-opportunity-backlog.md`) tell you which
   line carries which fix.
3. **CLAUDE.md** — take `main`'s compacted structure, then re-add our sections
   (Azure backend, scoring-fairness, env-pin/local-torch-core, one-GPU-per-cell).
4. **Concurrent scoring-fairness session is still live on this worktree** (it adds
   commits onto `feat/azure-aks-gpu`). **Re-sync `feat/azure-aks-gpu-clean` from the
   latest `feat/azure-aks-gpu` tree before merging** if it has advanced past `bde6bcf`.
5. **Already IN our line (verified — content present in the tree via merge commits,
   even though the branch refs still show "unique commits" from pre-squash granular
   history / post-merge drift):** `full-scope-envs` (real ALFWorld/WebShop/Search-QA +
   **E5 dense retrieval**, merges `64b1ba5`/`2266fcc`), **BES** (`e643e4c`/`dbcd7f5`/
   `2702ee8`), **rubric-ui-clarity** (`fdad8f9`), harden, recent-runs/leaderboard,
   F-fixes, RL-scaffold, accelerator. **The clean branch's tree already contains all
   of these — do NOT try to re-merge the stale branch refs.**
   **Genuinely OUT — only `gepa`** (`origin/gepa` / `feat/gepa-integration` /
   `gepa-phase2-5`: no merge, zero content in tree) → decide in D2. Misc refs
   (`perf-accelerator`/`efficiency`/`wedge`/cleanup) are per memory
   `worktree-consolidation-plan` mostly subsumed or deferred — verify per-branch only
   if 100% completeness is required.

---

## 4. Recommended plan (robust; operator resolves manually)

```bash
# 0. Sync + sanity (main worktree has a live concurrent session — work in a fresh worktree)
git fetch origin
git worktree add ../openresearch-merge feat/azure-aks-gpu-clean
cd ../openresearch-merge
# (if feat/azure-aks-gpu advanced past bde6bcf, rebuild the clean branch first — see §5)

# 1. Integration branch based ON main, so the PR shows our 6 commits + the merge
git checkout -b integrate/clean-to-main origin/main

# 2. Merge our work in (expect 87 conflicts)
git merge feat/azure-aks-gpu-clean

# 3. Resolve per §5 policy. ORDER: significant code (operator) -> its tests -> trivial.
#    For each significant file the agent runs:  git diff --merge -- <file>
#    explains main-side vs ours-side, proposes, operator edits + `git add <file>`.

# 4. Verify (§6) BEFORE committing the merge.
git commit                      # only after a clean verify

# 5. Push + PR
git push -u origin integrate/clean-to-main
gh pr create --base main --head integrate/clean-to-main   # loop in Aayush/Armaan
```
Commit identity stays `lolout1 <lolout1@users.noreply.github.com>`, **no Claude
co-author trailer** (memory `commit-attribution-preference`).

---

## 5. Resolution policy table

| Category | Default resolution |
|---|---|
| Our-unique (azure / scoring / env-pin / bes / full-scope) | **keep ours** (additive — should not conflict; if it does, ours) |
| Naming / rename (`OpenResearch`, `env_prefix`, paths) | **take main's canonical name**, re-apply our functional additions on top |
| Duplicated shared fix (harden, F-xx, recent-runs, SDAR) | **take the superset / later fix**; diff both, pick the more complete; never invent |
| `CLAUDE.md` | main's structure + re-add our sections |
| Tests | match the resolved source behavior; **keep both** where each side added new tests |
| `docs/specs` | additive — keep both (fix any renamed paths) |
| `runs/prj_*` artifacts | trivial — pick either side consistently, or `git rm` (diagnostic logs only) |
| `.gitignore` | **union both** |

### Re-sync the clean branch (only if `feat/azure-aks-gpu` moved past `bde6bcf`)
```bash
# rebuild the 6-commit clean branch from the new tip, race-immune (see how it was built):
B=15225796b01ca678fbb11f7e7299a4b3aa310309
git checkout feat/azure-aks-gpu && git branch -D feat/azure-aks-gpu-clean
git checkout -b feat/azure-aks-gpu-clean && git reset --soft $B && git checkout -- .
# then the 6 grouped `git commit -- <pathspec>` from the prior session, gate:
git diff --quiet feat/azure-aks-gpu HEAD && echo IDENTICAL
git push -f origin feat/azure-aks-gpu-clean   # only if you own this clean ref
```

---

## 6. Post-merge verification (all must pass before pushing)

```bash
# Python core
pip install -r backend/requirements.txt
.venv/bin/python -m pytest tests/ -n auto            # expect green, 0 new failures
.venv/bin/python -m pytest tests/agents/rlm/test_azure_env_contract.py -v   # the seam
# env-prefix sanity (the rename hazard)
.venv/bin/python -c "from backend.config import get_settings as g; s=g(); print(s.azure_gpu_skus)"
# Frontend (3 conflicts there)
cd frontend && npm ci && npx tsc --noEmit && npm run lint && npm test
```
Spot-check: azure backend still imports (`python -c "import backend.services.runtime.aks_job_backend"`),
`REPROLAB_CELL_*` seam names unchanged on both runner + `docker/aks-cell-base/aks_cell_entrypoint.py`.

---

## 7. Decisions the operator must make

- **D1.** Confirm `main` is the canonical target (recommended — it has the rename +
  teammate work). If instead `5.30.26_sdar` should become canonical, the strategy flips.
- **D2.** GEPA — include it or not? It is the **only** named feature genuinely out of
  our line (full-scope-envs / BES / rubric-ui / harden / recent-runs / RL-scaffold /
  accelerator are **already in** — verified by merge history + content). Default:
  exclude GEPA from this merge and fold it separately.
- **D3.** Duplicated-fix tie-breaks where both lines diverged on the *same* file and
  neither is a clean superset — operator picks per-file.
- **D4.** Coordinate with Aayush/Armaan: `main` carries their merges; PR, don't force-push.

---

## 8. Reference docs & memory

**Project:** `CLAUDE.md` · `system_overview.md` · `docs/design/rlm-pivot-brief.md` · `docs/runbooks/running-the-project.md`
**Azure (this deliverable):** `docs/superpowers/specs/2026-06-03-azure-aks-gpu-backend-design.md` (§1–§14) · `docs/superpowers/specs/2026-06-02-azure-gpu-compute-backend-scope.md` (on `main`) · memory `azure-aks-gpu-backend`
**Concurrent / adjacent workstreams:** `docs/superpowers/specs/2026-06-07-rubric-scoring-harness-fairness-design.md` + memory `scoring-fairness-spec` · memory `harness-env-reliability-fixes` · memory `bes-integration-design` · memory `full-scope-agentic-envs`
**Shared-fix provenance (conflict tie-breaks):** `docs/audits/2026-05-31-backend-core-opportunity-backlog.md` + memory `backend-core-audit` · memory `root-harness-hardening` · `docs/superpowers/specs/2026-05-31-oom-gpu-capacity-remediation-design.md` · `docs/runbooks/2026-06-01-backend-core-merge-and-continuation-handoff.md` (prior merge precedent)
**Consolidation big-picture:** memory `worktree-consolidation-plan`
**Conventions:** memory `commit-attribution-preference` (lolout1, no Claude trailer) · `run-walltime-preference` · `best-runs-coworker-showcase`
**Memory dir:** `/home/sww35/.claude/projects/-home-sww35-openresearch/memory/` (`MEMORY.md` index)

---

## 9. Kickoff prompt (paste into the fresh session)

> Read `docs/runbooks/2026-06-08-merge-clean-branch-to-main-handoff.md` and the
> docs/memory it references. Goal: merge `feat/azure-aks-gpu-clean` into
> `origin/main` as one clean, conflict-resolved merge, committed as
> `lolout1 <lolout1@users.noreply.github.com>` with no Claude co-author trailer.
>
> Work in a fresh worktree (`../openresearch-merge`); the main worktree has a live
> concurrent session — do not disturb it. First re-run the §2 trial merge to confirm
> the 87-conflict surface is still current (re-sync the clean branch per §5 if
> `feat/azure-aks-gpu` advanced past `bde6bcf`).
>
> **I resolve all significant (code/script/frontend) conflicts manually.** For each,
> show me `git diff --merge` for that file, explain main-side vs our-side, propose a
> resolution, then wait for me to edit + `git add` before moving on. You may
> auto-resolve only the trivial tier (docs / `runs/` artifacts / `.gitignore`) and
> must list exactly what you changed. Follow the §5 policy table; respect the §3
> sharp edges (esp. the `REPROLAB_→OPENRESEARCH_` rename). Run the full §6
> verification before we commit the merge. Then push and open a PR to `main`
> (loop in Aayush/Armaan) — no force-push to shared refs.
>
> `/grill-me` me on D1–D4 before starting.
