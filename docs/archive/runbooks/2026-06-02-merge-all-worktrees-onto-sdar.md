# Merge all feature worktrees onto `5.30.26_sdar` — handoff (2026-06-02)

**Goal:** fold every worktree that carries unique work onto `5.30.26_sdar` and push to
`origin/5.30.26_sdar`. This supersedes the conflict *estimates* in
`docs/runbooks/2026-06-02-worktree-consolidation-handoff.md` — those were pre-trial guesses
(it feared 6 conflict files). The numbers below are **ground truth from real `git merge`
trials** (git 2.25.1 can't do in-memory `merge-tree --write-tree`, so trial-merge-and-abort
in a throwaway worktree was used).

## TL;DR

Three merges onto sdar, in order, with **exactly one trivial conflict**:

```
sdar ── merge c606748 (harden+abheek+F-04/05/11) ──► CLEAN
     ── merge feat/full-scope-envs              ──► 1 add/add hunk in context.py (union both fields)
     ── merge feat/rubric-ui-clarity            ──► CLEAN
```

Then gate with the test suite (GPU-hidden) and push as a **fast-forward** (no rewrite, no
force-push). origin/5.30.26_sdar is currently 1 *behind* local, so the push is clean.

---

## 0. Constraints — READ FIRST

- **A live run is active.** Two `train_cell.py` PIDs (the 7B SDAR + GRPO ALFWorld cells) are
  training out of `/home/sww35/openresearch-fullscope/`, driven by `batch_reproduce.py`
  (`--scope-spec /tmp/scope_sdar_7b.json`). **Do not** touch that worktree, the shared base
  venv (`/home/sww35/openresearch/.venv` — no `pip install` into it), or the GPUs. All merge
  work happens in a **fresh isolated worktree**; the test suite runs with `CUDA_VISIBLE_DEVICES=""`.
- **No history rewrite.** `origin/5.30.26_sdar` is published. Merge with real ancestry and
  fast-forward push. Do **not** force-push. (The rewritten `integrate/harden-into-sdar`
  `bf64b6f` is the cautionary tale — see §5.)
- **Commit identity:** author `lolout1` (the repo default `lolout1 <lolout1@users.noreply.github.com>`
  is fine — history already mixes that with `appradhann@gmail.com`, both as `lolout1`),
  **no `Co-Authored-By: Claude` trailer**.
- **Disk:** `/home` is at ~100% (≈99 GB free on 11 TB). Worktree checkouts are safe — the
  tracked working tree is only **20 MB** (the 9.8 TB is gitignored `runs/`/HF caches).
- **The main worktree (`/home/sww35/openresearch`) has uncommitted changes** (`data/calibration.json`,
  `runs/prj_09047604e591d969/*`). The consolidation touches none of those paths, so it can't
  collide — but build in a *separate* worktree so the main checkout is never disturbed.

---

## 1. Worktree inventory & verdict

Unique-commit counts and diffstats are **vs `5.30.26_sdar`** (HEAD `8928b4d`).

| Worktree | Branch / HEAD | Unique | Verdict | Why |
|---|---|---|---|---|
| `openresearch` (main) | `5.30.26_sdar` `8928b4d` | — | **TARGET** | push lands here |
| `openresearch-fullscope` | `feat/full-scope-envs` `8edeb48` | 16 commits, 53 files +10267/-282 | **MERGE** | real agentic envs; **live run runs here** |
| `openresearch-ui` | `feat/rubric-ui-clarity` `d12a1bc` | 1 commit, 10 files +1236/-25 | **MERGE** | lab-ui rubric clarity |
| `openresearch-harden` | `harden/root-harness` `fbea44d` | (in c606748) | **MERGE via c606748** | backend-core audit fixes |
| `openresearch-integrate` | `integrate/harden-into-sdar` `bf64b6f` | — | **IGNORE** | rewritten history; use the backup below |
| — (ref) | `backup/integrate-pre-rewrite-20260601` `c606748` | 48 commits, 88 files +8895/-335 | **MERGE** | real-ancestry harden+abheek+F-04/05/11 |
| `openresearch-fullscope-envcache` | `feat/full-scope-envs-envcache` `21d439e` | 0 | **DROP** | subsumed by `feat/full-scope-envs` |
| `openresearch-rl-trl` | `feat/rl-trl-grpo-engine` `89795b0` | 0 | **DROP** | already an ancestor of sdar |
| — (remote only) | `feat/rlm-perf-multienv-accelerator` `0e67c8f` | 0 | **DROP** | already an ancestor of sdar |
| — (remote only) | `feat/rlm-wedge-hardening` | 55 vs old `main` | **DEFER** | 5 add/add parallel impls; dedicated session |
| — (remote only) | `abheek_recent_runs_panel` | (in c606748) | **DONE** | already folded into c606748 |

> `harden/root-harness` and `integrate/harden-into-sdar` are **not merged directly**.
> `c606748` (`backup/integrate-pre-rewrite-20260601`) already contains the *full* harden track
> **plus** abheek's recent-runs UI **plus** F-04/05/11, all with real ancestry (merge-base with
> sdar = `c19b78c`). Merging the one backup branch brings the whole backend track in one clean
> step. (`c606748` already contains sdar's `e791c60` paper-title work, so no regression there.)

---

## 2. What's on the worktrees but NOT on sdar (the feature delta)

### A. `feat/full-scope-envs` — SDAR restored to full scope (16 commits)
- **Real multi-turn agentic envs**: ALFWorld, WebShop, Search-QA (replacing closed-book/faked
  surrogates that floored the rubric at ~0.36).
- **Dense E5 retrieval** at wiki-18 scale (mmap'd faiss index shared across cells) with BM25/lexical fallback.
- **Multi-GPU cells**: `OPENRESEARCH_GPUS_PER_CELL` device_map-shards one model across N GPUs per cell.
- **Never-die-without-a-report**: always-on watchdog hard-ceiling (14h default) + bounded matrix
  timeout + SIGTERM finalizer → wedged/killed runs still ship a partial report.
- **Verified operator-scope**: `ScopeSpec.skip_datasets` + `--scope-spec`; env-setup failures
  become *Exclusions* (rubric **excludes**, not zeroes — the fairness principle).
- **Cell-level resume**: fingerprint-based skip of unchanged ok cells; `--rerun-env`/`--rerun-cell`.
- New files: `agentic_rollout.py`, `alfworld_env.py`, `webshop_env.py`, `search_qa_env.py`,
  `exclusion.py`, `cell_fingerprint.py`, `services/runtime/env_cache.py`, `scripts/batch_reproduce.py`,
  13 test files, 3 runbooks.
- New env vars: `OPENRESEARCH_GPUS_PER_CELL`, `OPENRESEARCH_WATCHDOG_HARD_CEILING_S`,
  `OPENRESEARCH_RESUME_CELLS`, `OPENRESEARCH_RESUME_FORCE_CELLS`, `OPENRESEARCH_PROVISION_ENVS`,
  `OPENRESEARCH_ENV_CACHE_DIR`, `OPENRESEARCH_SEARCH_QA_{DENSE,ENCODER,INDEX_DIR,INDEX_REPO,...}`,
  `OPENRESEARCH_ALFWORLD_*`, `OPENRESEARCH_WEBSHOP_*`.

### B. `c606748` — backend hardening + recent-runs UI + F-04/05/11 (48 commits)
- **Harden / root-harness P0–P3 + audit Phase A/B** (11 findings): F-27 `<math>` alttext rescue,
  F-28 arXiv transient-HTML retry, F-29 degraded-paper-text warning, F-31 validated-cache reuse,
  F-32/F-34 placeholder-regex tightening, F-03 ENOSPC config co-signal, F-06 two-experiment guard
  reset, F-07 disk-exhausted detector, F-08 HF gated-repo + NCCL-timeout classification; plus the
  P1 RuntimeGuard blocklist plumbing, P2 metric↔artifact manifest binding, P3 metric projection +
  adversarial-grader stance. ~1500 LOC new tests.
- **Abheek recent-runs UI** (merge `be49356`, `bb9a58c`): recent-runs home panel + sidebar
  Leaderboard link + best-attempt scoring + run diagnostics + codex-subagent repair framework.
- **F-04** (GPU-ladder escalation on watchdog-killed OOM), **F-05** (surface masked code bug on
  already-failed runs), **F-11** (best-of-run floor *before* verdict reconciliation).

### C. `feat/rubric-ui-clarity` — lab UI (1 commit)
- Best-of-run rubric headline + expandable rubric breakdown; touches `primitives.py`,
  `sse_bridge.py`, `binding.py` and several `frontend/src/components/lab/rlm/*` files.

---

## 3. Merge plan & rationale

Merge order is chosen so the **only** conflict surfaces in the smallest, simplest step:

1. **`c606748` first.** sdar's only post-`c19b78c` commits are docs/`best_runs` — disjoint from
   every code file the backend track touches → this merge is **clean** (verified 3 ways).
2. **`feat/full-scope-envs` second.** Its one real overlap with the harden track is a single
   add/add line in `context.py`. Everything else (primitives.py, run.py, cli.py, leaf_scorer.py,
   schemas.py) auto-merges — the two tracks edit disjoint regions of those files.
3. **`feat/rubric-ui-clarity` last.** Even against full-scope's heavy `primitives.py` changes,
   git auto-merges it — **clean**.

---

## 4. Conflict map — GROUND TRUTH (real trial merges)

| Step | Merge | Result | Resolution |
|---|---|---|---|
| 1 | `c606748` → sdar | **CLEAN** (0 files) | none |
| 2 | `feat/full-scope-envs` → sdar+harden | **1 file, 1 hunk** — `backend/agents/rlm/context.py` | union (below) |
| 3 | `feat/rubric-ui-clarity` → sdar+harden+full | **CLEAN** (0 files) | none |

**The one conflict** is a pure add/add — both branches append a *different* new `RunContext`
field at the same spot. Keep **both** (both `field`/`Any` are already imported above):

```python
    # Benchmark-integrity blocklist (2026-05-31, #7): ... RuntimeGuard ...
    blocked_terms: tuple[str, ...] = ()                              # <- from c606748 (harden)
    # Verified env-setup exclusions (2026-06-01 full-scope): ... metrics.scope ...
    env_setup_exclusions: list[Any] = field(default_factory=list)   # <- from full-scope
```

Because it's add/add-keep-both, the resolution is mechanically just "drop the 3 conflict
markers" (the `sed` in §5 does exactly this). Verified: `context.py` parses (`ast.parse`) after.

---

## 5. Execution (copy-paste)

### Phase 1 — build the consolidation in an isolated worktree (zero touch to main / live run)

```bash
cd /home/sww35/openresearch
git fetch origin                                   # make sure refs are current
git worktree add -b integrate/consolidated /home/sww35/openresearch-consolidated 5.30.26_sdar
WT=/home/sww35/openresearch-consolidated

# 1) harden track — CLEAN
git -C "$WT" merge --no-ff \
  -m "merge: backend-core hardening + recent-runs UI + F-04/05/11 (c606748)" \
  backup/integrate-pre-rewrite-20260601

# 2) full-scope — one add/add hunk in context.py; union-resolve by dropping the markers
git -C "$WT" merge --no-ff --no-commit feat/full-scope-envs || true
sed -i '/^<<<<<<< HEAD$/d;/^=======$/d;/^>>>>>>> feat\/full-scope-envs$/d' \
  "$WT/backend/agents/rlm/context.py"
test "$(grep -cE '^(<<<<<<<|=======|>>>>>>>)' "$WT/backend/agents/rlm/context.py")" = 0 \
  || { echo "UNEXPECTED extra conflict markers — STOP and inspect"; exit 1; }
/home/sww35/openresearch/.venv/bin/python -c \
  "import ast; ast.parse(open('$WT/backend/agents/rlm/context.py').read()); print('context.py OK')"
git -C "$WT" add backend/agents/rlm/context.py
git -C "$WT" commit --no-edit

# 3) rubric-ui — CLEAN
git -C "$WT" merge --no-ff \
  -m "merge: lab-ui best-of-run rubric headline + expandable breakdown (rubric-ui-clarity)" \
  feat/rubric-ui-clarity

git -C "$WT" log --oneline -4
```

> If any merge stops with an *unexpected* conflict (history moved since this doc), DON'T force
> it: `git -C "$WT" merge --abort`, re-run the §Appendix trial to re-derive the conflict map,
> and resolve by keeping **both** sides' behavior (union) unless it's a genuine semantic clash —
> escalate those.

### Phase 2 — gate (GPU-hidden, nice'd so the live 7B run is undisturbed)

```bash
cd /home/sww35/openresearch-consolidated
CUDA_VISIBLE_DEVICES="" nice -n 19 /home/sww35/openresearch/.venv/bin/python -m pytest tests/ -q
```

Expected: green at the current baseline (~3577–3610 passed). A handful of **load-induced**
timeout/Dockerfile flakes (e.g. the `implement_baseline` pre-emit 20s test) are *not*
regressions — re-run any failure in isolation to confirm. Optionally `cd frontend && npm run
build && npx tsc --noEmit` for the UI merges.

### Phase 3 — land on sdar + push (fast-forward, no force)

```bash
# origin/5.30.26_sdar is 1 behind local, and integrate/consolidated descends from it,
# so this is a clean fast-forward of the remote sdar branch:
git push origin integrate/consolidated:5.30.26_sdar

# bring the LOCAL sdar ref up to match when convenient (touches only code files, not the
# dirty data/calibration.json or runs/* — those are outside the merge delta):
git -C /home/sww35/openresearch fetch origin
git -C /home/sww35/openresearch merge --ff-only origin/5.30.26_sdar
```

> Alternative if you prefer to land it through the local branch first: from a *clean* main
> worktree, `git -C /home/sww35/openresearch merge --ff-only integrate/consolidated` then
> `git push origin 5.30.26_sdar`. Only do this if the main worktree's dirty files are stashed
> or you've confirmed they're outside the delta (they are: code-only delta).

### Phase 4 — cleanup (optional)

```bash
git worktree remove /home/sww35/openresearch-consolidated   # after push; or keep for follow-ups
git branch -d integrate/consolidated                        # the commits now live on sdar
# Optional remote tidy (separate, deliberate): delete the stale rewritten branch
#   git push origin --delete integrate/harden-into-sdar
```

---

## 6. Verification checklist

- [ ] All three merges applied; `git log --oneline -4` shows the two merge commits + the
      full-scope merge (with the context.py resolution commit).
- [ ] `grep -rIl '^<<<<<<<\|^>>>>>>>' backend/ frontend/ scripts/` → **nothing**.
- [ ] Test suite green at baseline (GPU-hidden); any red re-passes in isolation.
- [ ] Feature presence sanity on the merged tree:
      `ls backend/agents/rlm/{alfworld_env,search_qa_env,exclusion,cell_fingerprint}.py`,
      `grep -n env_setup_exclusions backend/agents/rlm/context.py`,
      `grep -n blocked_terms backend/agents/rlm/context.py`,
      `ls frontend/src/components/lab/rlm/rubric-detail.tsx`.
- [ ] `git push` reported a **fast-forward** (no `+ forced update`).
- [ ] Live run still healthy (`ps aux | grep train_cell`), GPUs untouched.

---

## 7. Drops & deferrals (nothing to merge)

- **DROP** `feat/rl-trl-grpo-engine` (`89795b0`), `feat/rlm-perf-multienv-accelerator`
  (`0e67c8f`): both are already ancestors of sdar (`git merge-base --is-ancestor … 5.30.26_sdar`
  → 0). `feat/full-scope-envs-envcache` (`21d439e`): subsumed by `feat/full-scope-envs`.
- **DEFER** `feat/rlm-wedge-hardening`: 55 commits off *old* `main` with 5 add/add parallel
  reimplementations (`local_gpu_allocator`, `batch_reproduce`, `serve_local_llm`, accelerator).
  Needs its own session; not part of "merge the worktrees."
- **IGNORE** `integrate/harden-into-sdar` (`bf64b6f`, on origin): its history was rewritten by an
  identity filter-branch → new SHAs, severed ancestry → merging it conflict-storms. Its *content*
  is fully reproduced by `c606748` (the pre-rewrite backup). Delete the remote branch later if desired.

---

## 8. Rollback

- During Phase 1: `git -C "$WT" merge --abort` (mid-merge) or
  `git -C "$WT" reset --hard 5.30.26_sdar` (after commits) — the worktree is throwaway.
- Whole thing: `git worktree remove --force /home/sww35/openresearch-consolidated;
  git branch -D integrate/consolidated`. Nothing on sdar or origin changes until Phase 3.
- After push (only if needed): the previous remote tip is the pre-push `origin/5.30.26_sdar`
  (= parent of `8928b4d`); restore with an explicit `git push origin <old_sha>:5.30.26_sdar`
  (coordinate — it rewrites the remote).

---

## 9. Open decision (one) — history model

This doc assumes **real merges, no rewrite, fast-forward push** (safe; abheek's commits keep
their real authorship; reversible). The *alternative* — a `git filter-branch` to unify all
authors under `lolout1` over `origin/main..HEAD`, then force-push — was the old plan's last
step, designed for a clean **PR into `main`**. It is **out of scope here** because it
force-rewrites the published `origin/5.30.26_sdar` (the exact move that produced the broken
`bf64b6f`). If a single-author history is wanted for an eventual `main` PR, do it later on a
*throwaway* branch with a backup + tree-diff verification — never in-place on sdar.

---

## Appendix — how the conflict map was derived (reproducible)

git 2.25.1 lacks `merge-tree --write-tree`, so the map came from real trial merges in a
throwaway worktree (then aborted/removed):

```bash
git worktree add -b scratch/mergetest /tmp/wt 5.30.26_sdar
git -C /tmp/wt merge --no-ff --no-commit backup/integrate-pre-rewrite-20260601   # → 0 conflicts
git -C /tmp/wt commit --no-edit
git -C /tmp/wt merge --no-ff --no-commit feat/full-scope-envs                    # → context.py only
git -C /tmp/wt diff --name-only --diff-filter=U
# resolve context.py (drop markers), commit, then:
git -C /tmp/wt merge --no-ff --no-commit feat/rubric-ui-clarity                  # → 0 conflicts
git worktree remove --force /tmp/wt && git branch -D scratch/mergetest
```

Inventory/feature characterizations were produced by two read-only `Explore` agents over the
commit ranges; drop/subsumption claims were proven with `git merge-base --is-ancestor`.
