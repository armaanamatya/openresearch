# OpenResearch / ReproLab — Project-State Analysis Prompt (2026-05-23)

> **What this is.** A self-contained brief for a fresh agent (you) doing an
> end-to-end audit of the repo to answer one question: **what is the right
> next thing to work on?** Not "fix this bug" — *what should we be doing
> next, all surfaces considered*. The repo is in a state where many
> overlapping efforts have landed in parallel; the work that was queued has
> been partly absorbed by adjacent commits. Cross-check before you recommend.
>
> Mode: PhD-researcher + principal-engineer rigor. Evidence before assertions.
> Quote files/lines. Distinguish "documented as done" from "verified done."

---

## 0. How to use this prompt

1. Read §2's canonical sources directly. Do NOT trust paraphrases.
2. Run the surveying commands in §4 yourself — repo state shifts hourly.
3. Build a picture of (a) what is genuinely done, (b) what is in flight, (c)
   what is queued, (d) what is broken.
4. Recommend the next 1–3 things to do, ranked, with explicit dependencies
   and disagreements with prior plans called out.

---

## 1. Project, in one paragraph

OpenResearch / ReproLab is an agent pipeline that reproduces research papers
end-to-end and scores the result against a PaperBench-style rubric. It is
pivoting from a fixed 14-stage `PipelineStage` state machine to an **RLM
(Recursive Language Model) orchestrator** built on the `rlms` PyPI library
(arXiv 2512.24601 — Zhang, Kraska, Khattab). The pivot's canonical plan is
`docs/design/rlm-pivot-brief.md`. The pivot has gone through six numbered
phases (#58–#63), with **Phase 6 cleanup recently merged (#72)** — the
14-stage pipeline code is being / has been removed; `main` is now the
RLM-only path.

The user's working mode is `/iterate`: delegate implementation to Sonnet
sub-agents against tight specs; Opus reviews every diff; commit infrequently
at major milestones; **no `Co-Authored-By` / AI-attribution trailer**; surface
forks as concrete options.

---

## 2. Canonical sources — read these, do not paraphrase

- **`docs/design/rlm-pivot-brief.md`** — the canonical pivot plan (architecture, threat model, fidelity invariants, build order, success criteria). Where it conflicts with code, the brief is direction and code is present.
- **`docs/superpowers/specs/2026-05-22-rlm-debug-harden-handoff.md`** — the team's standing instructions (§1) and the original issue catalogue (§4, P0-I1 … P2-I13) that drove the most recent fix cycle.
- **`docs/superpowers/plans/2026-05-22-rlm-debug-harden-plan.md`** — the 31-task implementation plan that consolidates the handoff with the independent review. Most tasks are now merged (see §3); cross-check each against actual code.
- **`docs/design/phase3-5-review-findings.md`** — independent Codex review covering the Phase 3 orchestrator + Phase 5 e2e work (Critical C1-C2, Important I1-I9, Minor M1-M8). Drove most of the 31-task plan.
- **`CLAUDE.md`** — repo conventions, the two-process Docker model, the **RLM auth-surfaces** section (the 2026-05-22 pitfall about `ANTHROPIC_API_KEY` with no credits), the **sandbox config gotcha** about `REPROLAB_FORCE_SANDBOX`.
- **`docs/design/project-state-audit-2026-05-22.md`** (if present) — the team's own state audit from one day ago. Read first if it exists — it's the closest thing to a current snapshot.
- **`progress.md`** and **`learn.md`** — running progress + post-mortem log. The user maintains these; check what they've recorded.
- **`CHANGELOG.md`** `[Unreleased]` — what shipped recently.

Any handoff or plan doc whose filename starts `2026-05-2x-` is a session artifact — read each you find under `docs/superpowers/specs/` and `docs/superpowers/plans/` before deciding what is "queued vs done."

---

## 3. What this prompt adds — current-state distillation (the part you can't get from the docs alone)

This is the agent-author's summary of the state as of the last working session
(branch `fix/leaf-scorer-honesty` merged via PR #75). Verify each item from
git history — claims here can age fast.

**The 31-task debug-and-harden plan (P0-I1..P2-I13 + Critical C1-C2 + Important I1-I9 + Minor M1-M8):**

- **28 of 31 tasks landed and merged via PR #75** (`fix/leaf-scorer-honesty` → `main`). Task list with status:
  - **P0 done:** T1 real metric extraction (closes the 0.35 cap); T2 verify_against_rubric tree path; T3 leaf-scorer reads correct keys; T4 leaf-scorer degraded 0.35 cap (incl. failed-verdict branch); T5 leaf-scorer meets_target + areas preservation; T6 verdict reconciliation against evidence.
  - **P1 done:** T9 RunContext.deadline_utc wired; T10 corpus redaction in response; T11 propose_improvements failsoft; T12 sandbox_mode threading (note: Runpod backend was wired *after* T12 — see below); T13 reproduce --fresh; T14 429-aware retry/backoff in LLM clients; T15 REST RLM bridge + project-id threading; T16 OCR-skip quality gate; T17 arXiv HTML fetch cap + iterative HTML walker; T18 parsed_full_text invalidation on parse failure; T19 IterationCheckpointer restart safety; T20 stub-primitive shapes derived from real schemas; T21 stub-run observability (primitive_provider/degraded in RLMFinalReport + demo_status.json).
  - **P2 done:** T22 tesseract-ocr-eng install; T23 has_provider_credentials probes claude session (closed one of three originally-tracked pre-existing failures); T25 dead-code _extract_references guard; T26 leaf_scorer + rubric_gen reuse _extract_json; T27 render_markdown graded default = 0; T28 arxiv response.info() inside with-block; T29 OcrPaperParser.version cached; T30 fsync after iterations.jsonl append; T31 bs4 + pytesseract moved to required deps.

- **T7 — `ftrl` reproduced wrong method (handoff P0-I3):** **DEFERRED, no investigation.** The `ftrl` run wrote a 13 KB `train.py` whose summary said "the RLM system was implemented" — leaf score 0.000. Either `third_party/paperbench/ftrl/paper.md` is wrong (the RLM paper vendored under `ftrl/` by accident) or the root mis-grounded. The fix branches into two paths depending on which it is.
- **T8 — workspace `paper_text` content loss in SDK mode (handoff P0-I4):** **DEFERRED, no investigation.** For some papers (IOI was the canary) the workspace `paper_text` variable reassembled from indexed chunks is degraded/empty. RLM mode now bypasses the workspace variable (reads `parsed_full_text.txt`), but SDK mode still consumes it. Needs a trace through parser → indexer → chunker → workspace.
- **T24 — `build_environment` ThreadPoolExecutor timeout (handoff P2-I12):** **XFAILED with honest reason.** The guard test exposed an incomplete A2-C3 hardening: `with ThreadPoolExecutor(...) as pool:`'s `__exit__` blocks on shutdown while the worker thread is still wedged in `asyncio.sleep`. The cap fires in the `try/except TimeoutError` but the function never returns. Fix is `pool.shutdown(wait=False, cancel_futures=True)` or a non-context-manager pool lifecycle. **This is a real production gap, not a test issue.**

**Adjacent work that may already cover or supersede parts of the plan:**

These commits landed on `main` after the 31-task plan was written. Each one
may have resolved an open item or introduced new territory the analysis must
account for. **Verify each from git log:**

- `feat(rlm): wire RunpodBackend into RLM path for sandbox_mode=runpod` — the plan's T12 only threaded `sandbox_mode` through; full backend wiring for `runpod` mode was a noted follow-up. This commit may close that follow-up.
- `feat(rlm): thread run_budget through RunContext + _execute_in_sandbox` — overlaps with T9 (deadline_utc) but adds a separate run-wide spend / pod-time budget.
- `feat(budget): add max_pod_seconds field with check_pod_seconds method` and `feat(cli): add --max-pod-seconds flag and REPROLAB_MAX_POD_SECONDS env var` — new pod-time budget surface.
- `feat(runpod): incremental artifact sync replaces tar-stream` — performance / robustness improvement on the Runpod artifact path.
- `fix(runtime): detect macOS Keychain OAuth for claude-agent-sdk` (#77) — adjacent to T23 (`has_provider_credentials`); the OAuth detection on macOS now goes through Keychain. Check whether T23's `claude auth status` probe coexists cleanly with this.
- `fix(rlm): recover honesty guard + binding test fixtures lost in #72 rebase` — Phase 6 cleanup PR #72 dropped some test infrastructure during rebase; this commit recovered it. Worth knowing such churn exists.
- `docs+test: preserve static-values audit + e2e guard salvaged from #76` (#78, the salvage of the closed #76) — frontend static-values audit landed.

**Test-suite state (last verified):**
- **1285 passed, 2 failed (documented pre-existing), 1 xfailed (T24), 5 skipped.** Re-verify on `main`.
- The 2 documented pre-existing failures: `tests/test_issue17_runtime.py::test_local_process_backend_executes_commands_with_artifact_env` and `tests/test_issue26_experiment_runner.py::TestRunWithRuntime::test_local_process_mode_runs_without_dockerfile`. Both fail with `/bin/sh: python: command not found` — a local-process-backend PATH issue. Neither has been addressed in the queued plan; the recent commit `f850a0a test(runtime): use sys.executable, not bare 'python', for the artifact-env exec test` may have fixed one or both — verify.

**Open PRs (as of writing — re-run `gh pr list --state open --json ...`):**
- **#74 — `docs: interactive wow-factor / context-engineering poll`** (branch `claude/great-tesla-xor5z` → `main`), **draft**. Investigate purpose, owner, and merge readiness.

**Recently merged PRs that frame the current state:**
- #75 (`fix/leaf-scorer-honesty`) — the 28-task plan delivery.
- #72 (Phase 6 cleanup — 14-stage pipeline deletion).
- #70 (Phase 4 frontend redesign).
- #67 (Phase 3 orchestrator).
- #77 (macOS Keychain OAuth).
- #78 (static-values audit salvage).

**Branches that exist but may be stale:**
- `rlm_rubric_orchestration` (remote-only?). Check its last commit date and whether it's superseded.
- `claude/great-tesla-xor5z` (the open #74 branch).

---

## 4. Survey commands — run these to ground your analysis in fresh data

```bash
# Repo + branches
cd /Volumes/CS_Stuff/openresearch
git fetch --all --prune
git branch -a
git log origin/main --oneline -50
git log origin/main --since="2026-05-22" --pretty=format:"%h %ad  %s" --date=short

# PRs
gh pr list --state open --json number,title,headRefName,baseRefName,isDraft,mergeable,createdAt,updatedAt --limit 30
gh pr list --state all --limit 30 --json number,title,state,mergedAt --jq '.[] | "\(.number) [\(.state)] \(.title) — merged \(.mergedAt // "—")"'
gh pr view <N> --json title,body,state,reviewDecision,mergeable,additions,deletions,changedFiles,commits

# Issues — open + recently closed
gh issue list --state open --limit 30
gh issue list --state closed --limit 20 --search "closed:>2026-05-15"

# Tests — confirm the baseline
.venv/bin/python -m pytest tests/ --tb=no -q 2>&1 | tail -15
# Confirm T24's xfail is still the only one:
.venv/bin/python -m pytest tests/ --tb=no -q -rxX 2>&1 | tail -25
# Confirm whether the local-process-backend failures persist after f850a0a:
.venv/bin/python -m pytest tests/test_issue17_runtime.py tests/test_issue26_experiment_runner.py -v 2>&1 | tail -20

# Code health on the RLM core
git grep -n "TODO\|FIXME\|XXX" backend/agents/rlm/ | head -30
git grep -n "noqa:" backend/agents/rlm/ | head -10

# Docs — find the most recent state audits / handoffs
ls -la docs/superpowers/specs/ docs/superpowers/plans/ docs/design/
git log --oneline -20 -- docs/superpowers/specs/ docs/superpowers/plans/ docs/design/

# Working tree
git status --short --branch
```

**Verify each "documented as done" claim against the actual code** — don't trust this brief or the plan doc, trust `git show <commit>`.

---

## 5. Questions to answer

Group your investigation around these. Cite specific files/lines/commits/PRs as evidence.

### A. Which planned items are actually done vs. still queued?
- For each of the 31 tasks (T1-T31): is it present in `main`? Is its guard test present and passing? Is its symptom verifiable as fixed?
- For each of the 3 originally-deferred items (T7, T8, T24): what is the current state? Has any adjacent commit addressed them inadvertently?
- For the 2 documented pre-existing test failures: are they still failing? Did `f850a0a` (the `sys.executable` fix) close either of them?

### B. What is broken / unfinished right now on `main`?
- Run the full test suite. Capture exact pass/fail/xfail/skip counts.
- For every failure or xfail: what's the symptom? Is it a known issue (catalogued in the handoff or in `learn.md`) or new?
- Cross-reference against the project-state audit doc (if present) and `progress.md`.

### C. What changes have landed in parallel since the 31-task plan was written?
- List every commit on `main` since `2026-05-22`. For each, classify as:
  - resolves a plan task (which? cite the task ID)
  - resolves a deferred or xfailed item (T7 / T8 / T24)
  - opens new territory not covered by the plan
  - is purely cosmetic / docs

### D. What's the state of the open PR #74?
- Read its body, diff, and any review comments. Is it ready to merge? Stale? Conflicting with anything on `main`?

### E. What does the project NEED next, ranked?
Three buckets to consider — they compete for time:

1. **Honesty / correctness** (top-priority by handoff §3):
   - T7 (`ftrl` paper-grounding investigation) — fast to start, may reveal a vendoring bug.
   - T8 (workspace `paper_text` content loss) — affects SDK mode only; how much of the project still uses SDK mode after Phase 6 cleanup?
   - T24 follow-up (`build_environment` ThreadPoolExecutor shutdown) — small, contained, has a one-line fix.
   - The 2 local-process-backend test failures — environmental, but they've been ignored for sessions.

2. **Demo readiness** (the project is "a demo for employers" per the handoff):
   - End-to-end runs on real papers — are 5 papers still the canon? Have new papers been added?
   - Frontend polish — what did #70 land vs. what's left?
   - Documentation — `progress.md`, `system_overview.md`, `CHANGELOG.md` accurate?

3. **New infra investments** (the recent run_budget / Runpod work suggests a push here):
   - More backends? More budgets? Resilience hardening?
   - The infra plan referenced in `0a69a42 docs(infra): infra improvement plan` — is there a fresh plan that supersedes the 31-task one?

Where these compete: pick the one the user will most regret leaving undone. The handoff says quality bar is "precise, accurate, 100% reliable, elegant." Honesty wins over surface area when in doubt.

---

## 6. Cross-cutting checks (don't skip)

- **RLM fidelity invariants (brief §8):** are they still upheld on the current `main`? Especially: paper never enters root context (#1); primitives take slices, not corpus (#2); recursion is programmatic (#3); output is a REPL variable (#4); different papers → different trajectories (#9).
- **Threat model (brief §7):** has the `environment="local"` / host-RCE risk been resolved or explicitly accepted? Search for any threat-model decision in recent commits or `progress.md`.
- **Honesty guards:** the metric-fabrication guard + verdict-evidence reconciliation (T6) + leaf-scorer degraded cap (T4) — are they all still firing as intended on a fresh run?
- **Pre-existing failures:** anyone who claims "no new failures" against a moving baseline is suspect. Pin the baseline before you call something a regression.

---

## 7. Output format

Produce:

### Verdict (one paragraph)
What's the actual state of the repo? Stable / fragile / mid-pivot / done-with-followups? One paragraph, no hedging.

### Confirmed state
Bullet list. Each bullet: a specific, evidence-backed fact about the current repo. File:line, commit hash, or test name for every item.

### Active gaps (ranked)
For each gap, give:
- **What:** one-sentence symptom or unfinished work.
- **Severity:** P0 / P1 / P2 + rationale.
- **Evidence:** file:line or commit/PR reference.
- **Cost to fix:** small / medium / large (with the deciding factor).
- **Depends on / blocks:** other gaps.

### Recommended next moves
**Three concrete options**, ranked. For each:
- The one-sentence action.
- Why this and not the alternatives.
- The first concrete step (specific file or commit to start from).
- A "done" condition.
- Rough effort (in handoff terms: minutes / hours / sessions).

### Disagreements with this brief or the plan
If anything in §3 of this prompt or the 31-task plan is *wrong* given the current state, call it out plainly. Better a contradicted brief than a wrong recommendation.

### Open questions the user must answer
List anything that genuinely needs human judgment — trust boundaries, scope decisions, demo-vs-rigor trade-offs. Don't bury these in the verdict; make them their own section so they're easy to spot.

---

## 8. Ground rules

- Verify every "done" claim from primary source. If it's not in `git log` or `git show`, it isn't done.
- Quote `file:line`, commit hashes, and PR numbers in every claim.
- **Don't propose work that already landed.** Cross-check every recommendation against recent commits before naming it.
- **Don't recommend small polish if a load-bearing gap exists.** P0 honesty/correctness items beat P2 cleanups, always.
- If you find a gap not captured in this brief or the 31-task plan, NAME IT — that's high-value signal.
- Be specific. "Improve test coverage" is not actionable; "add a guard test pinning `_extract_json_array`'s behavior on a top-level array containing a dict" is.
- This is a demo for technical reviewers (handoff §1). The bar is "precise, accurate, 100% reliable, elegant." Apply that bar to your own output.
