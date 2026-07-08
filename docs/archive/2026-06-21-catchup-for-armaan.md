# Catch-up for Armaan — start here (written to assume nothing)

> You've been away (~since **2026-06-14**), a LOT changed, and the version of the
> project you see on GitHub's default page is **out of date**. This doc explains
> everything from scratch — the project, the words we use, what happened, and the
> one button you need to press. Read top to bottom; it's written so you don't need
> any prior context.

---

## 1. What is this project, in one paragraph

**OpenResearch reproduces research papers automatically.** You hand it a machine-
learning paper (a PDF or an arXiv link). An AI agent reads the paper, writes the
code to recreate the paper's experiments, runs that code on a GPU, and then scores
how faithfully it reproduced the paper's results against a checklist ("rubric").
The output is a report saying "here's what I rebuilt and how close it got." That's
the whole product. Everything else is plumbing to make that work reliably and
honestly (i.e., not let the AI *fake* good results).

## 2. The 30-second status (the only thing you must understand)

Think of the code as living on parallel tracks ("branches"):

- **`main`** = the track GitHub shows by default. **It is stale** — it's missing
  ~6 weeks of work.
- **the "trunk"** (a branch literally named `feat/bes-conversion-correctness`) =
  where all the real work actually happened. It's **173 commits ahead of `main`**.

So when you look at `main`, you're seeing an old version of the project. **All the
new work is real, finished, tested (the full test suite passes — 7,036 tests), and
waiting** in one pull request.

**One action catches `main` up to everything:** merge **Pull Request #115**. That's
the decision at the end of this doc. We did NOT do it for you — it's your call.

*(This catch-up doc is the only thing we pushed to `main` so far, so you'd see it.)*

## 3. Words you'll see (plain-English glossary)

| Term | What it means here |
|---|---|
| **branch / `main` / trunk** | Parallel copies of the code. `main` is the "official" one; the *trunk* is the working one that's ahead. |
| **PR (pull request)** | A request to merge one branch's changes into another. Reviewed, then merged. |
| **merge / fast-forward** | Combining branches. "Fast-forward" = a clean catch-up with no conflicts. |
| **RLM** | "Recursive Language Model" — the AI agent design that reads the paper and writes/runs code. The core engine. |
| **primitive** | One tool the agent can call (e.g. `implement_baseline`, `run_experiment`, `verify_against_rubric`). There are 17. |
| **rubric / leaf** | The grading checklist. A "leaf" is one checkable item (e.g. "did it use the right learning rate?"). The "leaf scorer" grades them. |
| **sandbox** | Where experiments actually run: `local` (this machine), `docker`, `runpod`, `aks` (Azure), `gke` (Google). |
| **SDAR** | The hard test paper we benchmark against (arXiv 2605.15155). If the agent can reproduce SDAR, it can do most papers. It's the "North Star." |
| **BES** | "Best-of-N" — the agent tries several implementations and keeps the best one (judged on real evidence, not the AI's own opinion). |
| **evidence gate / hallucination** | Guards that stop the agent from *claiming* a result it didn't actually produce on disk. Core to keeping scores honest. |
| **default-OFF flag** | A new feature shipped turned off (behind an `OPENRESEARCH_*` env var). Off = the system behaves exactly as before. We add features "dark" first, then turn them on after testing. |

## 4. Why `main` and the trunk drifted apart (and that it's fixed)

Back around 2026-06-14, two batches of work (`#106` grader-fidelity, `#108`
azure-gcp) got merged **straight into `main`**. Meanwhile the bulk of development
kept going on the trunk. The two tracks drifted so far they could no longer be
combined automatically — a genuine mess.

**That mess is now cleaned up.** We rebuilt a clean combination of "everything on the
trunk + the two things that were only on `main`" into **PR #115**. Merging it makes
`main` the single, correct, complete version. No work is lost from either side.

## 5. What actually changed since you left (in plain terms)

Grouped by *why it matters*, not by commit:

**Making the agent honest (so a high score is real, not faked):**
- Guards that veto any result the agent claims but didn't actually compute on disk
  ("evidence gate"). A returning theme — there's now a whole layer of these.
- The grader (which assigns scores) was made less random: it can take several
  samples and use the median, and it cross-checks claims against real output files.

**Making the agent reliable (so long GPU runs don't silently die):**
- Experiments now stream their output live, detect when they're truly stuck, and —
  if they time out — still score the partial work instead of throwing it away.
- GPU work runs **one experiment per GPU** with automatic retry when a GPU runs out
  of memory, instead of cramming everything onto one card and crashing.

**More places to run (cloud):**
- Added **Azure** and **Google Cloud (GKE)** as first-class places to run GPU jobs,
  alongside the existing RunPod. (`--sandbox aks` / `--sandbox gke`.)

**This week's specific additions (all OFF by default — zero risk to current behavior):**
- **One config path for "Azure AI Foundry" models** — use any Foundry-hosted model
  everywhere.
- **Google GKE** promoted to a fully-supported, tested option.
- **A diagnosis of where the agent can hallucinate**, plus one guard for it.
- **Your own work landed** (see §6).
- **A "be honest" architecture writeup** that ties the evidence/validator guards
  together into one design.

## 6. What happened to YOUR work specifically

You had a big branch (`feat/grounded-self-improvement-harness-reliability`, PR **#110**)
— things like an evidence-auditing critic, an external validator, a recipe memory,
etc. **All of it is now on the trunk** (we folded it in cleanly as PR #116). It's all
shipped **turned off by default**, so it's there and ready but changes nothing until
someone flips it on after testing.

Your PR #110 itself shows as "closed" — but **only because its content is now in the
trunk**, not because it was thrown away. **Your branch still exists and the PR can be
reopened** if you want to look at the original. Same story for two other PRs that were
closed as already-included (#104 grader-fidelity, #109 azure-bicep) — branches intact,
nothing lost.

## 7. What YOU need to decide (the one button)

**Merge PR #115 into `main`.**

- **If you do:** `main` instantly becomes the complete, correct project — the trunk,
  all this week's work, your #110 work, everything. It's already tested green (7,036
  tests pass) and has no conflicts. After that, a couple of now-redundant PRs and old
  branches get cleaned up (exact commands are written down for you — see §9).
- **If you don't:** nothing breaks; the work just stays staged on the trunk and `main`
  stays old. There's no rush and no risk to waiting.

We intentionally left this for you instead of merging it ourselves.

## 8. The 5 docs worth reading (after #115 merges, they'll be on `main`)
Right now they live on the trunk; they arrive on `main` when you merge #115.
1. **SDAR plan** — how to push the benchmark paper to the best honest score.
2. **Run/Test Bank plan** — a shared archive of every past run's results & logs (also
   fixes a 12 MB bloat problem in the repo).
3. **Evidence-first architecture** — how all the "be honest" guards fit together.
4. **Cleanup plan** — the tidy-up still to do (and a list of 7 contradictions in our
   own docs, with the correct answers).
5. **Finish-line handoff** — the exact step-by-step to finish the consolidation.

## 9. The cleanup we already did (so the numbers make sense)
- **Pull requests: 10 → 2.** Only #115 (the catch-up merge) and #107 (a clean,
  reviewed piece) remain open. The rest were either merged into the trunk or closed
  because they were already included.
- **Branches: 18 → 13.** We deleted only branches that were fully merged and ours to
  delete. Your branches were left untouched.
- We did **not** touch `main` (except this doc) and did **not** delete any of your
  branches — those are your calls.

## 10. Where to look / two things still open
- **`CLAUDE.md`** = the day-to-day "how this repo works" guide (on the trunk, current).
  **`system_overview.md`** = the "why it's built this way" guide.
- **Still open:** a separate **private repo** with its own history cleanup was requested
  but never started (we need its link). And a round of **doc/code tidy-up** is planned
  for after #115 merges.

---
*Written 2026-06-21. Bottom line: `main` is old, everything new is finished and tested
in PR #115, and merging it is the one decision that's yours.*
