# OpenResearch — Cleanup / Condensation / Leaderboard — Implementation Kickoff

_Date: 2026-05-23 · Companion to `docs/superpowers/specs/2026-05-23-cleanup-condensation-leaderboard-design.md`. Copy §0 into the new session prompt; the rest are reference._

---

## §0 — Kickoff prompt (copy this into the new session)

````
/iterate — Begin executing the OpenResearch cleanup, condensation, and leaderboard
plan. Read this prompt in full first, then the spec, then begin Phase 0.

YOU ARE: the lead engineer on a multi-session implementation. Opus designs and
reviews every diff. Sonnet sub-agents implement against tight specs. Codex is
review/adversarial/diagnosis only — never implementation. Quality bar: precise,
accurate, 100% reliable, elegant. This is demo-grade work — a Microsoft VP
review is imminent.

YOUR MISSION: take the design spec at
`docs/superpowers/specs/2026-05-23-cleanup-condensation-leaderboard-design.md`
from "approved design" to "shipped across 5 PR families." The spec is canonical;
where it conflicts with anything you find in the codebase, the spec wins (it was
written against the 2026-05-23 audit baseline; the codebase is `main` at
`ecb7931` plus whatever has merged since — check `git log origin/main` first).

REQUIRED READING — in this order, before any tool call beyond `git log`:

1. `docs/superpowers/specs/2026-05-23-cleanup-condensation-leaderboard-design.md`
   — the spec. Sections §3 (locked decisions), §5 (phase specs), §8 (acceptance
   criteria), §9 (carried session context). Do not relitigate §3.
2. `CLAUDE.md` — repo conventions, two-process Docker model, RLM auth surfaces,
   sandbox config gotchas. The auth pitfall on line 14–18 of `.env` is real.
3. `system_overview.md` — why the system fits together the way it does.
4. `learn.md` — post-mortem log; read top 10 entries for recent root causes.
5. `progress.md` — the user's running journal.

DO NOT READ the deleted-as-of-Phase-3 docs (`docs/design/phase2-*`, the
`audit-2026-05-22-*` files, etc.) unless you need a specific historical fact —
they're being removed in Phase 3 because they're noise. The spec subsumes them.

LOCKED DECISIONS (from spec §3; do not re-ask the user):

1. Default `--mode` after Phase 1 = `rlm`. Flip to `rdr` only after Phase 2 honesty re-score proves rdr is at least as honest as rlm.
2. PR #74 (wow-factor poll, branch `claude/great-tesla-xor5z`, draft, conflicting) → shrink to ≤2 files (`docs/wow-factor-poll.html` + Discord post text) and merge in Phase 5. Drop the demo-track PNGs.
3. CLI per-role models: keep `--model <id>` as shorthand for `planner` role; add `--models planner=…,executor=…,verifier=…,grader=…` for full per-role control. `--models` overrides `--model` / `--sub-model` if both passed.
4. Budget = orchestrator-estimated, user-accepted, safety-ceiling-enforced. Ceiling: `$20 / 7200s wall-clock / 7200s pod-seconds`. UI shows accept/override/cancel before sub-agents fire. Leaderboard re-runs inherit the original run's accepted budget.
5. Docs cleanup = `git rm`, no archive directory. History retains everything.
6. Leaderboard route = `/leaderboard`.
7. Both `--mode rlm` and `--mode rdr` stay indefinitely. rdr is not a replacement; it's a peer.

STANDING INSTRUCTIONS (from spec §9):

- Commits: infrequent. One commit per major milestone, not per fix. NEVER a
  `Co-Authored-By` / AI-attribution trailer. Branch in `feat/…` / `fix/…` /
  `chore/…` style.
- Push to `origin` (`armaanamatya/openresearch`). NEVER `replix`.
- Documentation: keep `progress.md` current and succinct; append to `learn.md`
  on every non-obvious bug fix (Symptom → Root cause → Fix → Lesson → Guardrail).
  Update `CHANGELOG.md` `[Unreleased]` on every shipped phase.
- Delegation: parallel sub-agents for disjoint file sets. Spawn `Explore`
  subagents for read-only investigation. Use `feature-dev:code-architect` for
  per-phase architecture; `feature-dev:code-reviewer` before opening each PR.
- When in doubt — ask the user. Surface forks with concrete options.
- Evidence before assertions. Run the verification, read the output, then
  claim done. A passing self-test is not evidence — observable output is.

PHASE EXECUTION ORDER:

Phase 0 → Phase 1 → (Phase 2 || Phase 3 in parallel) → Phase 4 → Phase 5.

For each phase: invoke the `writing-plans` skill to produce a per-task
implementation plan with file paths, test commands, TDD steps, and commit
boundaries; show the plan to the user; on approval, invoke
`subagent-driven-development` (or `executing-plans` if you prefer inline) to
execute. After each phase's PR merges, update `progress.md`, append a
`learn.md` entry if there's a non-obvious lesson, and confirm the spec's §8
acceptance criteria for that phase.

FIRST ACTIONS (Phase 0):

1. `git log origin/main -5` and `git status` — confirm the baseline matches the
   spec's §9 (HEAD = `ecb7931` or descendants; test count = 1083+ / 0 / 1 / 5).
   If it doesn't, STOP and report drift to the user before continuing.
2. `git add docs/superpowers/specs/2026-05-23-cleanup-condensation-leaderboard-design.md`
   and commit with message:
     "docs(plan): cleanup + condensation + leaderboard design spec"
   Then `git add` this kickoff prompt (`2026-05-23-cleanup-implementation-kickoff.md`)
   and commit it as:
     "docs(plan): implementation kickoff prompt for the cleanup spec"
   Do NOT push.
3. Send the user a draft message to forward to Abheek (the rdr author) — see
   §A below for the template. Get the user's sign-off before sending; the user
   will deliver it. Block here on Abheek's response (or the user's explicit
   "skip the sync, just merge").
4. Once Abheek confirms his branch is stable: invoke `writing-plans` for Phase
   1. The Phase 1 plan should produce 5 tasks (1.1 pre-merge survey, 1.2
   backend reconcile, 1.3 frontend integration, 1.4 CLI verification, 1.5 PR).

FORBIDDEN ACTIONS:

- Do NOT modify the spec without explicit user approval. Surface changes as
  questions.
- Do NOT `git push --force` to `main`. Never.
- Do NOT skip pre-commit hooks (`--no-verify` etc.) without explicit user
  authorization.
- Do NOT use `Co-Authored-By` trailers on commits.
- Do NOT touch the `replix` remote.
- Do NOT delete docs in Phase 3 without first inspecting their last-modified
  date and verifying their listed parent issue/PR is merged. If unsure for any
  single file, ask before deleting.
- Do NOT add features outside the spec's §5. The spec is the scope contract.
  Surface discoveries to the user.
- Do NOT relitigate the §3 locked decisions. They're answered.
- Do NOT batch all 5 phases into one PR. Each phase = its own reviewable PR
  family (Phases 1 and 4 may split into 2 PRs each per the spec).

WHEN TO ASK THE USER:

- A spec §8 acceptance criterion is ambiguous in the current code reality.
- A merge conflict touches §3 decision lines (e.g. the C2 honesty cap).
- A sub-agent reports a finding that suggests the spec's assumption is wrong.
- Test counts drift more than 2% from the spec §9 baseline without an obvious
  cause.
- The budget orchestrator estimate (§5.5.3) lands wildly wrong on the first 3
  runs (>2× off from consumed).

DEFAULT WHEN BLOCKED:

Read more code, not more docs. The codebase is authoritative on what exists;
the spec is authoritative on what should exist. If they conflict on a
non-§3-locked decision, the codebase wins.

GO.
````

---

## §A — Abheek sync message (template; the user delivers, not the agent)

Subject / first line: `OpenResearch — rdr merge restart, quick sync`

Body:

> Hey — picking up the rdr merge. Two things:
>
> 1. Could you push your latest `rlm_rubric_orchestration` and pause new commits on it for ~24h while I reconcile against `main`? My branch will rebase on top of yours, not the other way around.
> 2. Can you send the exact CLI invocation you've been testing rdr with on each of the 5 PaperBench bundles (`ftrl`, `sequential-neural-score-estimation`, `mechanistic-understanding`, `RLM`, `IOI`)? E.g. `python -m backend.cli reproduce <bundle> --mode rdr --models ...`. I'll use those as the post-merge smoke-test commands.
>
> Plan is to land your rdr backend + extend the lab UI to render its cluster events (one canvas, two layouts), keep both `--mode rlm` and `--mode rdr` indefinitely, then build the leaderboard on top. Default mode stays `rlm` for one cycle while we re-score the existing runs honestly. Spec: `docs/superpowers/specs/2026-05-23-cleanup-condensation-leaderboard-design.md` on `main`.

---

## §B — Phase-by-phase entry conditions (for the agent to self-check)

| Phase | Entry condition | Verification command |
|---|---|---|
| 0 | Spec committed; kickoff prompt committed; Abheek pinged | `git log -2 --oneline` |
| 1 | Abheek confirms branch stable + invocation list received | (out-of-band) |
| 2 | Phase 1 PR merged; both modes run; lab UI green | `python -m backend.cli reproduce 2512.24601 --mode rdr` + `npm test` |
| 3 | Independent of 2; can run in parallel after Phase 1 merges | (none — start anytime) |
| 4 | Phase 1 + Phase 2 both merged; honest scores in disk-runs | Check the latest honesty-cap entry in `progress.md` |
| 5 | Phase 4 backend + frontend merged; ≥2 runs in `/leaderboard` | `curl http://localhost:3000/leaderboard` |

---

## §C — Phase 0 → Phase 1 hand-off contract

The new session should produce these artifacts before declaring Phase 0 done:

1. Commit `<hash>`: spec committed.
2. Commit `<hash>`: kickoff prompt committed.
3. A draft Abheek message ready (or user's explicit waiver).
4. A Phase 1 merge plan reviewed by the user and ready for execution.

---

## §D — Phase 1 → Phase 2 hand-off contract

1. PR titled "Land rdr + extend lab UI to render rubric-cluster layout" merged.
2. `python -m backend.cli reproduce 2512.24601 --mode rdr` runs end-to-end and writes `runs/<id>/final_report.{md,json}`.
3. `?rdrFixture=1` on `/lab` renders six regions without console errors.
4. `pytest tests/` ≥ 1083 passed / 0 failed / 1 xfailed (T24 still xfailed).
5. `progress.md` + `CHANGELOG.md [Unreleased]` updated.
6. `git push origin main`.

---

## §E — Standing risks (carried from spec §7)

1. Phase 1 merge math > 30 conflicted files → cherry-pick rdr's 10 commits one-at-a-time.
2. rdr honesty regression in Phase 2 → cap fix lands as a Phase 1 amendment; Phase 2 acceptance waits.
3. Leaderboard cost explosion → default sandbox = local for re-runs.
4. Frontend dual-layout (§5.2.3) larger than 1 session → split Phase 1 into 1a (backend) + 1b (frontend).
5. Per-role model picker UX bloat → 3 named presets (Default / Cheap / Mixed-Vendor) + Custom.
6. Lab vs. leaderboard primary entry point ambiguity → lab is a drill-down view from Phase 4 onward.
7. Orchestrator budget estimate wrong → safety ceiling clamps; recalibrate after 10 runs.

---

## §F — When the kickoff is over

This document and the spec are both deletable in Phase 3 (per spec §5.4.1). They survive as the historical record of the implementation plan. By the time Phase 5 lands, they should be the only `2026-05-2[123]-*` docs left in the repo.
