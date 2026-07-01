# Cleanup Phase A — ready-to-run handoff (OWNER actions)

> **Why this is a handoff, not done automatically.** Every Phase-A item touches a
> COLLABORATOR's artifacts (`lolout1 <appradhann@gmail.com>`) or an ambiguous-owner
> branch. Deleting someone else's remote branches, closing their PR, or moving their
> tracked working files are unilateral outward-facing acts on another person's work —
> so they are left for the owner(s) to run. Commands are copy-paste ready; each is
> annotated with its owner and a safety note. Design: `docs/superpowers/specs/2026-06-21-project-cleanup-design.md`.

## A1 — prune merged branches (no open PR)
All three are merged into the trunk `feat/bes-conversion-correctness`. **Safety:** their
content survives in the trunk; deletion is recoverable short-term. **Do NOT run until the
trunk's path to `main` is settled** (#115 vs the #109→#107 stack) — "merged into the
unmerged trunk" is not yet "in permanent history."

```bash
# owner: lolout1
git push origin --delete feat/azure-aks-gpu
git push origin --delete feat/gcp-gke-backend
# owner: ambiguous "Aayush C Baniya" handle — confirm it's yours before deleting
git push origin --delete bes
```
Local mirrors (whoever has them): `git branch -D feat/azure-aks-gpu feat/gcp-gke-backend bes`

**Do NOT prune:** every branch backing an open PR (#104/#107/#109/#110/#111/#112/#114/#115/#116),
and `feat/gepa-integration` (operator's own, 3-week unmerged unique work — an `archive/gepa-*`
tag set already exists if you want to delete it after tagging).

## A2 — close superseded PR #110 (lolout1's)
#110 (`feat/grounded-self-improvement-harness-reliability`, by **lolout1**) was
squash-integrated into **#116** (`feat/grounded-harness-integration`). **Safety caveat:**
#116 is itself UNMERGED — if it reworks or never lands, #110 is the canonical PR. **Close
#110 only after #116 is merged**, not before.

```bash
# owner: lolout1 — run AFTER #116 merges
gh pr close 110 --comment "Superseded by #116 (squash-integrated there). Closing post-merge."
```

## A3 — archive 3 loose root .md (lolout1's files)
`bes_integration.md`, `claude_gcp_sdar_handoff.md`, `issues.md` are tracked working notes
authored by **lolout1** (3–5 days ago). **Owner's call** whether to keep at root, archive,
or delete. If archiving:

```bash
# owner: lolout1
mkdir -p docs/archive
git mv bes_integration.md claude_gcp_sdar_handoff.md issues.md docs/archive/
git commit -m "docs: archive point-in-time working notes to docs/archive/"
```
(`gcp_info.md` is gitignored already — leave it.)

## What WAS done automatically (solo-safe, no one else's work touched)
- **Two rename-safety characterization tests** added — `tests/config/test_legacy_env_and_db_compat.py`
  (7 tests, green): the `_apply_legacy_env_aliases` bridge + the `_fall_back_to_legacy_sqlite_db`
  fallback, the two mechanisms the research found UNTESTED. These guard the non-breaking contract
  before any Phase-B canonicalize.
- This work is on branch **`chore/cleanup-foundation`**, kept **separate from `main`** (not merged).

## Phase B (after the stack merges to main) — see the design doc
CLAUDE.md de-drift (7 code-verified contradictions), env canonicalize (4 reads + tests, lockstep),
dead-code removal (verify-first), runbook archiving (34 docs, 5 update-citing-line first), and the
`best_runs/` → Run/Test Bank migration.
