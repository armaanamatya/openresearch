<!-- doc-meta: status=current; last-verified=2026-07-05 -->
# Handoff — Merge `reconcile/grounded-self-improvement-on-main` → `main` (elegant, conflict-free)

> **Date:** 2026-07-05 · **Status:** Current · action doc for landing this branch on the
> `deepinvent` mainline via a PR. Self-contained.

## 0. TL;DR

The reconcile branch is a **clean forward-superset** of the mainline — a merge has **no
conflicts**. This is not a one-off 295-commit dump; it is the **next increment in an already
established PR cadence** (`deepinvent/main` has merged this same branch three times: PRs #4/#5/#6).
The new PR (#7) brings the commits added since the last merge, all **default-OFF / byte-identical
when off**. Land it, then rebase the sibling worktrees.

- **PR:** `deepinvent/main` ← `reconcile/grounded-self-improvement-on-main` on
  `Deepinvent/scientific_article_generator`.
- **Blockers:** `gh` is not authenticated here (need `gh auth login` or a token, or use the web
  compare URL); the uncommitted **external-runs** workstream + `runs/` + logs must stay OUT.
- **Author/contributor:** local git config = `lolout1 <appradhann@gmail.com>` — correct; no
  `Co-Authored-By`/AI trailer, never a `-c user.email=…` override.

## 1. Branch topology (verified 2026-07-05)

| Ref | SHA | Note |
|---|---|---|
| local `main` | `cf60903` | tracks `origin` — **stale**, ignore for the PR |
| `origin/main` | `776ff08` | `armaanamatya/openresearch` — do NOT push here (memory rule) |
| **`deepinvent/main`** | `c178f519` | the real mainline; already carries PRs #4/#5/#6 of this branch |
| branch HEAD | `8f4944bf` | our work, pushed to `deepinvent/reconcile/...` already |
| **merge-base**(deepinvent/main, HEAD) | `c87ddd36` | shared history |

- Branch is **3 commits ahead** of `deepinvent/main` (new since the last PR):
  `5daaff5b` (Anthropic-Foundry provider + lifecycle-primary/execute-mode foundation),
  `ddc82113` (SDAR-fix + autonomous-reproduction handoff), `8f4944bf` (OpenScience skill library
  Release-1) — **plus** the two uncommitted-at-write-time commits this session adds: the **CLAUDE.md
  lean-root + nested-docs restructure** and **this handoff**.
- `deepinvent/main` is 3 ahead of the merge-base, but those are only **merge-PR commits** of this
  same branch — no independent code → **the PR is conflict-free** (confirmed: `git log
  deepinvent/main ^HEAD` shows only merge commits).

## 2. What the PR lands (committed work only)

All flag-gated, default-OFF, byte-identical when unset — the mainline behavior does not change until
an operator flips a flag (each flip gated by ≥3 paired A/B runs + the grader-σ gate). Grouped:

- **OpenScience skill library — Release 1** (`8f4944bf`): 40 vendored `SKILL.md` playbooks,
  `consult_skill` (19th primitive), per-provider prompt tails, literature grounding + claim gate,
  evidence-report section. Flags `OPENRESEARCH_{SKILLS,PROVIDER_PROMPTS,LITERATURE_*,EVIDENCE_REPORT_SECTION}`.
- **Anthropic-Foundry + lifecycle-primary/execute-mode foundation** (`5daaff5b` + `a9cbb32b..ddc82113`):
  `opus-foundry`/`sonnet-foundry` reliable-root path, `OPENRESEARCH_LIFECYCLE_PRIMARY`,
  `OPENRESEARCH_REPRODUCTION_MODE=execute`.
- **CLAUDE.md restructure** (this session): lean ~2k-token root + nested `backend/agents/rlm/`,
  `backend/services/runtime/`, `frontend/`, `tests/` `CLAUDE.md` files; fidelity test extended to
  the root+nested set. Docs-only.
- Older on-branch history already partially merged (SDAR execute-mode + cell services, reproduction
  campaign Phase B/C, multi-cloud ComputeProvider phases 1a–1f, env adapters, grader-fidelity /
  evidence-gate reliability, grounded self-improvement).

## 3. What must stay OUT (uncommitted / experimental / junk)

Do **not** `git add` these into the PR — they are a separate, incomplete workstream or run debris:

- **External-runs feature (uncommitted):** `backend/app.py`, `backend/config.py`,
  `frontend/src/components/lab/lab-sidebar.tsx`, `backend/routes/external_runs.py`,
  `backend/services/external_monitor/`, `frontend/src/{app/api/external-runs,app/external-runs,components/lab/external,lib/external-runs}/`,
  `configs/external_runs.json.example`, `tests/routes/test_external_runs_http.py`,
  `tests/services/external_monitor/`.
- **Loose run-specs / scratch:** `configs/{autonomous_reproduction,canary_scratch,sdar_execute_cells_grid}_run_spec.json`,
  the modified `configs/sdar_execute_run_spec.json`.
- **Junk:** `campaign_validation.log`, `error.log`, `changes.md`, `image.webp`, and the **~136
  untracked `runs/` directories**.

Keep them in the working tree (or stash) — the PR ships only the committed 295 + the two new doc
commits.

## 4. Pre-merge verification (run on the branch before pushing)

```bash
cd /home/abheekp/openresearch
.venv/bin/python -m pytest tests/ -q                     # expect: pre-existing/env failures only
                                                          #   (host .env AZURE_FOUNDRY creds + missing
                                                          #    tesseract eng + OAuth/WSL) — zero NEW failures
.venv/bin/python -m pytest tests/test_claude_md_fidelity.py -q   # the restructure guard (root+nested)
uvx ruff@0.15.16 check backend/ tests/                   # touched-file clean
(cd frontend && npm run build && npx tsc --noEmit)       # if the external-runs UI is stashed
```

The 11 known failures on this host are pre-existing/environmental (proven by stash-isolation in the
skills-integration session) — they are NOT introduced by this branch.

## 5. The PR steps

```bash
cd /home/abheekp/openresearch
# 1. confirm the tree carries ONLY the intended commits (external-runs etc. uncommitted/stashed)
git status --short | grep -vE '^\?\? runs/'      # eyeball: nothing from §3 is staged

# 2. push the branch to deepinvent (SSH; branch already exists there → fast-forward push)
git push deepinvent reconcile/grounded-self-improvement-on-main

# 3a. open the PR with gh (needs `gh auth login` first — interactive, operator-run)
gh pr create --repo Deepinvent/scientific_article_generator \
  --base main --head reconcile/grounded-self-improvement-on-main \
  --title "Reconcile grounded-self-improvement onto main: Anthropic-Foundry foundation + OpenScience skill library (R1) + lean-root CLAUDE.md restructure" \
  --body-file docs/runbooks/2026-07-05-reconcile-to-main-merge-and-pr-handoff.md

# 3b. …or open it in the browser (no gh auth needed):
#   https://github.com/Deepinvent/scientific_article_generator/compare/main...reconcile/grounded-self-improvement-on-main?expand=1
```

- **Merge strategy — recommend a merge commit** (the established cadence; preserves provenance and
  keeps the flag-by-flag history that the A/B-gate discipline references). A squash would collapse
  295 commits and lose that trail.
- The branch is conflict-free against `deepinvent/main`, so the PR is green on merge.

## 6. Contributor / author check

```bash
git log -3 --format='%h  %an <%ae>'        # every commit must read: lolout1 <appradhann@gmail.com>
git log HEAD~3..HEAD --format='%B' | grep -i 'co-authored-by' && echo "TRAILER FOUND — remove" || echo "clean"
```

If a stray non-`lolout1` author appears on a session commit, fix with
`git commit --amend --reset-author` (local config only — never `-c user.email=sww35`).

## 7. Post-merge

- Rebase the sibling feature worktrees onto the new `deepinvent/main`:
  `feat/autonomous-upload-ui`, `feat/gcp-gke-backend`, `feat/repo-first-reproduction`,
  `feat/lab-kimik2-featherless`, `codex/lab-ui-demo-trace-fix` (see `git worktree list`).
- The external-runs workstream (§3) becomes its own branch + PR when it's complete.
- Update `project_openscience_skill_port` / the reconcile memory once merged (add the merge SHA).
