# Consolidation finish-line — post-#115 handoff (OWNER actions)

> **Status as of 2026-06-21:** the consolidation is DONE, green, and durable on the
> remote. **10 open PRs → 5.** Everything below is downstream of one decision —
> merging **#115** to `main` — which the operator deliberately deferred. Until that
> hinge moves, **nothing else here is safe to run** (every close/prune is contingent
> on #115 landing, and most touch `lolout1`'s work). This file is the ready-to-run
> sequence for when you ARE ready to land it.

## Where things stand (verified)
- Trunk `feat/bes-conversion-correctness` @ `555354f0` carries all five workstreams:
  Foundry (#111), GKE (#112), hallucination Fix-1 (#114), the #110 grounded-harness
  integration + ADR + mega-prompts + cleanup design (#116), and the rename-safety
  tests (#117). Those 5 PRs are **MERGED**. Full suite **7036 passed / 0 failures**.
- **#115** (`consolidate/main-trunk` → `main`): **MERGEABLE/CLEAN, 7036-green** — it now
  carries the ENTIRE consolidated world (trunk + all 5 + the `main` fold). It is the
  single artifact that lands everything on `main`.
- Remaining open: #115, #107 (trunk PR), and `lolout1`'s #104/#109/#110 (CONFLICTING,
  all superseded once #115 lands).

## THE HINGE — one merge lands everything
```bash
# owner: you. Merge ONLY when you're ready to update main.
gh pr merge 115 --merge
```
After this, `main` contains the trunk + all three workstreams + the #110 integration +
the cleanup foundation. Everything below becomes safe.

## Why the 3 CONFLICTING PRs must be SUPERSEDED, not "fixed" individually
#104, #109, #110 (all `lolout1`'s) show CONFLICTING only because they target the diverged
`main` (2-ahead / 173-behind the trunk). **Do not resolve their conflicts one-by-one** —
verified 2026-06-21:
- **#104 (grader-fidelity)** and **#109 (azure-bicep)** are **literal ancestors of the
  trunk** → 100% of their content is already in #115.
- **#110 (grounded-self-improvement)** was squash-integrated via #116 → in #115; its only
  residual vs the trunk is **104 lines / 50 files**, which are #110's *older* versions of
  shared files (`CLAUDE.md`, `accelerator.py`, `grader_transport.py`, `gke_cell_entrypoint.py`,
  `start.sh`…) that the trunk now holds in *newer* form (superseded by workstreams #111/#112
  + the #116 trunk-canonical integration). #110's 76 net-new feature modules are all in the
  trunk. **Nothing is lost.**

Resolving these against `main` individually would (a) redo #115's work 3×, (b) risk
**reintroducing the stale file versions**, and (c) create three competing main-landing PRs.
The correct fix is the single #115 merge below; then close all three as superseded.

## After #115 lands — close the 4 superseded PRs
Their content is now in `main`; they are redundant. (Do NOT close before #115 merges —
their content currently lives only in unmerged branches.)
```bash
gh pr close 107 --comment "Superseded — trunk content landed via #115."     # owner: you
gh pr close 109 --comment "Superseded — azure-bicep landed via #115."       # owner: lolout1
gh pr close 110 --comment "Superseded — integrated via #116 -> #115."       # owner: lolout1
gh pr close 104 --comment "Superseded — grader-fidelity landed via #115."   # owner: lolout1
```
→ **0 open PRs.**

## After #115 lands — prune merged branches
**Caveat:** branch deletion is recoverable only short-term (whoever holds the SHA). Run
this AFTER #115 is on `main`, never speculatively — "merged into the trunk" only becomes
"in permanent history" once the trunk is in `main`.
```bash
# owner: lolout1 (these are lolout1's merged branches)
git push origin --delete feat/azure-aks-gpu feat/gcp-gke-backend integrate/grader-fidelity-to-main
# ambiguous handle — confirm ownership first
git push origin --delete bes
```
Plus the now-merged feature/consolidation branches once #115 lands (yours):
`feat/foundry-provider-unification feat/gke-firstclass-backend feat/hallucination-harness-fixes
feat/grounded-harness-integration chore/cleanup-foundation consolidate/main-trunk feat/bes-conversion-correctness`
→ remote branches **17 → ~3–4** (`main` + anything you keep, e.g. `feat/gepa-integration`).

## Then: Phase B cleanup (off the new `main`)
Per `docs/superpowers/specs/2026-06-21-project-cleanup-design.md`: CLAUDE.md de-drift
(7 code-verified contradictions), env canonicalize (4 reads + tests, lockstep), dead-code
removal (verify-first), runbook archiving (34 docs, 5 update-citing-line first), and the
`best_runs/` → Run/Test Bank migration (`docs/superpowers/prompts/2026-06-21-run-test-bank-megaprompt.md`).

## Still open from the original brief
- **The private "other repo"** (history cleanup) — UNTOUCHED; still needs its URL + what's
  wrong. That half of the original request is not done.
