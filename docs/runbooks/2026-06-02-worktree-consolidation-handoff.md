# Worktree consolidation → one merge-ready branch for `main`

**Date:** 2026-06-02 · **Author:** lolout1 · **Status:** PLAN — ready to execute (no merges run yet)

> Goal: fold every *live* worktree/branch into **one** branch that is clean, green, and ready
> to PR into `main`. This doc is a self-contained execution prompt — a fresh agent (or you)
> can run it end-to-end. Every claim below was measured with read-only `git` on 2026-06-02;
> the exact commands are in the [Verification appendix](#10-verification-appendix-reproducible).

---

## 0. TL;DR (the whole plan in one paragraph)

The fog of "7 worktrees" collapses to **two real code merges**. The integration spine is the
**pre-rewrite backup `c606748`** (`backup/integrate-pre-rewrite-20260601`) — it already contains
`origin/main` + `harden/root-harness` + `abheek_recent_runs_panel` + F‑04/F‑05/F‑11 with *real*
git ancestry. Merge **`feat/full-scope-envs`** (11 unique commits — the real agentic envs; subsumes
`-envcache`) and **`feat/rubric-ui-clarity`** (1 commit — UI) into it. Three branches are already
**subsumed** (proven ancestors) → drop them. One branch — **`feat/rlm-wedge-hardening`** (55 commits,
74-file overlap) — is genuinely hard → **deferred, your decision**. After the two merges go green,
apply the identity rewrite (`.mailmap` + scoped `filter-branch`) as the **final** step, then PR.

```
spine c606748 (= main + harden + abheek + F-04/05/11)
   ├─ merge feat/full-scope-envs   (11 commits · 6 conflict-candidate files)
   ├─ merge feat/rubric-ui-clarity ( 1 commit  · 2 conflict-candidate files)
   ├─ (optional) fold sdar docs    ( 2 doc commits: best_runs + handoff)
   ├─ full-suite green gate
   └─ identity rewrite (LAST) → push → PR → main
DROP (subsumed): rl-trl-grpo-engine · rlm-perf-multienv-accelerator · full-scope-envs-envcache
DEFER (hard):    feat/rlm-wedge-hardening   ← decision needed
```

---

## 1. Goal & deliverable

One branch — proposed name **`integrate/consolidated`** — built off the real spine, containing all
*unique* features from every active worktree, full-suite green, with clean lolout1-only history, ready
to open as a single PR against `main`. The currently-pushed `integrate/harden-into-sdar` (`bf64b6f`,
rewritten) is **superseded** by this branch (see §2).

---

## 2. ⚠ CRITICAL: resume from the *backup*, not the pushed branch

The pushed `integrate/harden-into-sdar` (`bf64b6f`, on `origin`) had its **history rewritten**
(`git filter-branch` over `main..integrate` for the lolout1 identity cleanup). The rewrite gave every
commit a **new SHA**, which severs git's ancestry link to the source branches. Consequence:

- A `git merge feat/full-scope-envs` **into `bf64b6f`** finds merge-base = *old `main`* and replays
  sdar's history a second time → a conflict-storm across ~all shared files (this is the same artifact
  that showed every branch as "139 behind" the rewritten tip).
- The **pre-rewrite backup `c606748`** has the *original* SHAs → real ancestry → merges are bounded
  to the handful of genuinely-overlapping files (6 and 2, measured below).

**Rule: do all merging on the backup's real history. Apply the identity rewrite ONCE, at the very end.**
A rewrite is a *finishing* step, never a *mid-integration* step.

```
spine for merging   = backup/integrate-pre-rewrite-20260601   (c606748)  ← USE THIS
do NOT merge into    = integrate/harden-into-sdar / origin     (bf64b6f)  ← rewritten, superseded
```

The backup is verified to already contain: `origin/main` (1522579) ✓, `harden/root-harness` ✓,
`abheek_recent_runs_panel` ✓, and F‑04/F‑05/F‑11. It is **0 behind `origin/main`**.

---

## 3. Branch inventory & verdicts

Worktrees on disk (`git worktree list`):

| worktree | branch | tip |
|---|---|---|
| `/home/sww35/openresearch` | `5.30.26_sdar` *(LIVE RUN — do not disturb)* | `3c9cfa8` |
| `/home/sww35/openresearch-fullscope` | `feat/full-scope-envs` | `2d63b7f` |
| `/home/sww35/openresearch-fullscope-envcache` | `feat/full-scope-envs-envcache` | `21d439e` |
| `/home/sww35/openresearch-harden` | `harden/root-harness` | `fbea44d` |
| `/home/sww35/openresearch-integrate` | `integrate/harden-into-sdar` | `bf64b6f` |
| `/home/sww35/openresearch-rl-trl` | `feat/rl-trl-grpo-engine` | `89795b0` |
| `/home/sww35/openresearch-ui` | `feat/rubric-ui-clarity` | `d12a1bc` |

Verdict table (unique-commit counts are vs the **backup spine `c606748`**):

| branch | unique | conflict files | verdict | why |
|---|---:|---:|---|---|
| **`feat/full-scope-envs`** | **11** | **6** | **MERGE** | real multi-turn ALFWorld/WebShop/Search-QA envs, dense E5 retrieval, always-on watchdog, bounded matrix, env fail-soft + rollout-mask fix. **Subsumes `-envcache`.** High value. |
| **`feat/rubric-ui-clarity`** | **1** | **2** | **MERGE** | best-of-run rubric headline + expandable breakdown (lab UI). Low risk, mostly frontend. |
| `5.30.26_sdar` | 2 (docs) | 0 | FOLD (optional) | only `3c9cfa8` (best_runs showcase) + `adffe90` (session handoff) are not in the spine — pure docs. |
| `feat/full-scope-envs-envcache` | 0* | — | **DROP** | tip `21d439e` lives **inside** `feat/full-scope-envs` → merging full-scope covers it. |
| `feat/rl-trl-grpo-engine` | 0 | — | **DROP** | `89795b0` is a proven **ancestor** of sdar/spine — fully subsumed (TRL spec already landed). |
| `feat/rlm-perf-multienv-accelerator` | 0 | — | **DROP** | `0e67c8f` is a proven **ancestor** of sdar/spine — fully subsumed. |
| `harden/root-harness` | 0 | — | **already in spine** | merged (ancestor of backup). |
| `abheek_recent_runs_panel` | 0 | — | **already in spine** | merged (ancestor of backup). |
| **`feat/rlm-wedge-hardening`** | **55** | **74** | **DEFER → your call** | forked at *old* `main` (`2acbc8d`); 5 add/add parallel impls. Genuinely hard. See §6. |

\* `-envcache` is "1 unique vs spine" only because its single design-doc commit `21d439e` is the same
commit that sits at the base of `feat/full-scope-envs`; once full-scope is merged it is 0. DROP.

---

## 4. The consolidation plan

- **Target branch:** `integrate/consolidated`
- **Base / spine:** `backup/integrate-pre-rewrite-20260601` (`c606748`)
- **Merge order:** full-scope-envs → rubric-ui-clarity → (optional) sdar docs → gate → rewrite → push.
- **Workspace:** a **new worktree** so the live-run checkout (`/home/sww35/openresearch`) is never touched.

---

## 5. Step-by-step execution (exact commands)

### 5.1 — Create the consolidation worktree off the real spine

```bash
cd /home/sww35/openresearch
git worktree add /home/sww35/openresearch-consolidated \
  -b integrate/consolidated backup/integrate-pre-rewrite-20260601
cd /home/sww35/openresearch-consolidated
git log --oneline -1            # expect c606748
```

### 5.2 — Merge `feat/full-scope-envs` (the substantive one)

```bash
git merge --no-ff feat/full-scope-envs
```

**Conflict-candidate files (6)** — both sides changed these; expect to hand-resolve, **keeping all
features from both**:

| file | spine side owns | full-scope side owns |
|---|---|---|
| `backend/agents/rlm/primitives.py` | F‑04 `_is_oom_escalation_trigger`, F‑05 `_surface_masked_bug_on_failed_run` | agentic-env primitives / rollout wiring |
| `backend/agents/rlm/run.py` | harden F‑29 + forced-iteration seam | watchdog / bounded-matrix / SIGTERM |
| `backend/cli.py` | sdar `_NOISE_TITLES` + P1 `_resolve_blocked_terms` | full-scope guidance flags |
| `backend/evals/paperbench/leaf_scorer.py` | harden scoring guards | env-fidelity leaf checks |
| `backend/agents/rlm/context.py` | (spine baseline) | env context plumbing |
| `backend/agents/schemas.py` | (spine baseline) | env/rollout schema fields |

Resolution rule (per the standing instruction *"keep unique features from all"*): **union, not
replace.** If a true semantic conflict exists (same function, incompatible logic), **stop and ask the
user** — do not silently pick a side. The other ~27 files full-scope touches don't overlap → auto-merge.

### 5.3 — Merge `feat/rubric-ui-clarity` (UI)

```bash
git merge --no-ff feat/rubric-ui-clarity
```

**Conflict-candidate files (2):** `backend/agents/rlm/primitives.py`, `backend/agents/rlm/sse_bridge.py`.
Small; the rest are frontend. Union-resolve.

### 5.4 — (Optional) fold sdar's 2 doc commits

Only if you want the `best_runs/` showcase + the 2026-06-01 session handoff on the consolidated branch:

```bash
git cherry-pick adffe90 3c9cfa8        # docs only — trivial, or skip
```

### 5.5 — Acceptance gate (GPU-safe; the live run must stay untouched)

The live SDAR run owns the GPUs. Run the suite **GPU-hidden + niced** so it can't contend:

```bash
cd /home/sww35/openresearch-consolidated
CUDA_VISIBLE_DEVICES="" nice -n 19 /home/sww35/openresearch/.venv/bin/python \
  -m pytest tests/ -q -p no:cacheprovider
```

- Baseline to beat: **3577 passing** (the spine's count after harden+abheek+F‑04/05/11).
  Full-scope adds its own tests; expect the count to rise.
- **Known load-induced flakes** while the live run saturates CPU: a few timeout/shutdown +
  Dockerfile-wheel tests may fail under load. Re-run any failure **in isolation** before treating it as
  a regression (they passed solo last session). Cleanest signal: gate **after** the live run finishes.
- A genuinely failing env/scorer test after the full-scope merge is a **real** signal — fix it before
  proceeding (red → minimal fix → green → re-gate).

Commit the merges as lolout1 (merges auto-author from config; set it explicitly to be safe):

```bash
git -c user.name=lolout1 -c user.email=appradhann@gmail.com commit --no-verify   # if a merge paused for resolution
```

### 5.6 — Final identity rewrite (LAST step, once green)

This is the *only* place a rewrite happens. It folds maintainer aliases → `lolout1 <appradhann@gmail.com>`
and strips Claude trailers, over the `origin/main..HEAD` range **only**. The `.mailmap` (already on the
branch) is the display-layer twin; the rewrite makes it real in the objects.

**Back up first**, then rewrite, verify byte-identical tree, force-push with lease:

```bash
cd /home/sww35/openresearch-consolidated
git branch backup/consolidated-pre-rewrite-$(git rev-parse --short HEAD) HEAD

# Author/committer identity rewrite over the PR range only:
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f \
  --env-filter '
    CORRECT_NAME="lolout1"; CORRECT_EMAIL="appradhann@gmail.com"
    case "$GIT_AUTHOR_EMAIL" in
      sww35@sun.cs.txstate.edu|sww35@txstate.edu|lolout1@users.noreply.github.com|163784642+lolout1@users.noreply.github.com|appradhann@gmail.com)
        export GIT_AUTHOR_NAME="$CORRECT_NAME"; export GIT_AUTHOR_EMAIL="$CORRECT_EMAIL";;
    esac
    case "$GIT_COMMITTER_EMAIL" in
      sww35@sun.cs.txstate.edu|sww35@txstate.edu|lolout1@users.noreply.github.com|163784642+lolout1@users.noreply.github.com|appradhann@gmail.com)
        export GIT_COMMITTER_NAME="$CORRECT_NAME"; export GIT_COMMITTER_EMAIL="$CORRECT_EMAIL";;
    esac
    # Preserve genuine external contributors (Aayush, Armaan) — do NOT fold them.
  ' \
  --msg-filter '
    grep -v -E "^(Co-Authored-By: Claude|Co-authored-by: Claude|🤖 Generated with \[Claude Code\]|Generated with Claude Code)" || true
  ' \
  -- origin/main..HEAD

# Verify the tree is byte-identical to the pre-rewrite backup (content unchanged, identity only):
git diff --stat backup/consolidated-pre-rewrite-* HEAD     # expect EMPTY
git log --format='%an <%ae>' origin/main..HEAD | sort -u   # expect only lolout1 + Aayush/Armaan
```

> `.mailmap` content (already committed on the branch; keep it — it also fixes the contributor graph
> for any non-rewritten refs):
> ```
> lolout1 <appradhann@gmail.com> <sww35@sun.cs.txstate.edu>
> lolout1 <appradhann@gmail.com> <sww35@txstate.edu>
> lolout1 <appradhann@gmail.com> <lolout1@users.noreply.github.com>
> lolout1 <appradhann@gmail.com> Abheek Pradhan <163784642+lolout1@users.noreply.github.com>
> lolout1 <appradhann@gmail.com> Abheek Pradhan <appradhann@gmail.com>
> ```

### 5.7 — Push + open the PR

```bash
git push -u origin integrate/consolidated
# PR base = main. Compare URL:
#   https://github.com/armaanamatya/openresearch/compare/main...integrate/consolidated?expand=1
```

After this lands, the older pushed `integrate/harden-into-sdar` (`bf64b6f`) is **superseded** — close
its PR / delete the remote branch to avoid confusion.

---

## 6. Deferred: `feat/rlm-wedge-hardening` — DECISION NEEDED

- **Tip:** `b63e16a` (`origin/feat/rlm-wedge-hardening`) · **55 unique commits** · merge-base with spine =
  **`2acbc8d`** (*old* `main`) · **74 overlapping files**.
- It forked before `origin/main`'s last 4 commits **and** before sdar/harden/abheek/full-scope, so it
  re-touches almost everything. Memory notes **5 add/add parallel implementations**
  (`local_gpu_allocator`, `batch_reproduce`, `serve_local_llm`, accelerator, …) — the kind of conflict
  where both branches wrote *different* code for the *same* concept.
- This is **not** a "resolve 6 files" merge; it needs a dedicated, supervised session where you decide
  each parallel-impl winner. **Recommendation: ship the consolidated branch without wedge first**
  (clean, fast, mergeable), then merge wedge as a follow-up PR on top.

**Open question for you:** wedge *in* this consolidation (slower, big conflict session) or *after* (PR
the clean branch now, wedge as PR #2)? Default if you don't answer: **after**.

---

## 7. Dropped / subsumed (with proof)

All three are proven **ancestors** of the spine/sdar (nothing to merge):

```
feat/rl-trl-grpo-engine        89795b0  → ancestor of sdar & spine & full-scope   (TRL spec already in)
feat/rlm-perf-multienv-accel.  0e67c8f  → ancestor of sdar & spine & full-scope   (already absorbed)
feat/full-scope-envs-envcache  21d439e  → contained in feat/full-scope-envs       (covered by the merge)
```

(Their worktrees can be removed after: `git worktree remove <path>` — optional cleanup, not required.)

---

## 8. Hard constraints (do not violate)

1. **Do not disturb the live SDAR run** `prj_09047604e591d969` (training in `/home/sww35/openresearch`).
   All work happens in `/home/sww35/openresearch-consolidated`. Never touch `runs/`, never run an
   unhidden-GPU job, never `pip install` into the shared `.venv`. Tests run `CUDA_VISIBLE_DEVICES="" nice -n 19`.
2. **Attribution:** every commit authored **`lolout1 <appradhann@gmail.com>`**, **no** `Co-Authored-By:
   Claude` / "Generated with Claude Code" trailer. (See `.mailmap` + the rewrite in §5.6.)
3. **Keep unique features from all** — union-resolve conflicts; bring genuine semantic conflicts to the
   user, never silently drop a feature.
4. **Rewrite is the last step**, applied once, over `origin/main..HEAD` only, with a backup + tree-diff
   verification.

---

## 9. Open decisions for the user

1. **Wedge:** in now, or as a follow-up PR? (default: follow-up)
2. **sdar docs** (`best_runs/` + handoff): fold into the consolidated branch, or leave on sdar? (default: fold)
3. **Branch name:** `integrate/consolidated` ok, or prefer reusing `integrate/harden-into-sdar`? (default: new name)
4. **Backups:** keep the two existing `backup/integrate-*` refs + add the new pre-rewrite backup, then
   prune all once the PR merges? (default: keep until merged)

---

## 10. Verification appendix (reproducible)

Every fact above came from these read-only commands on 2026-06-02 (`cd /home/sww35/openresearch`):

```bash
git worktree list ; git branch -vv
# Honest ahead/behind use the REAL spine (rewritten bf64b6f poisons the numbers):
SPINE=backup/integrate-pre-rewrite-20260601
for b in feat/full-scope-envs feat/rubric-ui-clarity feat/rl-trl-grpo-engine \
         feat/rlm-perf-multienv-accelerator feat/full-scope-envs-envcache \
         origin/feat/rlm-wedge-hardening ; do
  git rev-list --left-right --count $SPINE...$b ; done
# Subsumption (ancestor => drop):
git merge-base --is-ancestor 89795b0 5.30.26_sdar && echo rl-trl-IN
git merge-base --is-ancestor 0e67c8f 5.30.26_sdar && echo perf-IN
git merge-base --is-ancestor 21d439e feat/full-scope-envs && echo envcache-IN-fullscope
# origin/main containment:
git merge-base --is-ancestor origin/main $SPINE && echo main-in-spine
# Conflict-candidate files (overlap of both sides' changed files vs merge-base):
for b in feat/full-scope-envs feat/rubric-ui-clarity ; do
  base=$(git merge-base $SPINE $b)
  comm -12 <(git diff --name-only $base..$b|sort) <(git diff --name-only $base..$SPINE|sort) ; done
```

**Measured results (locked):** spine `c606748` is 0-behind `origin/main`; full-scope = 11 unique / 6
overlap; rubric-ui = 1 unique / 2 overlap; rl-trl + perf-accel + envcache = subsumed; wedge = 55 unique
/ 74 overlap (merge-base old-main). See the tables in §3.
