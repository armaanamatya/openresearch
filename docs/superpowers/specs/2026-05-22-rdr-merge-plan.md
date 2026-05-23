# Merge plan — `rlm_rubric_orchestration` → `main`

_Date: 2026-05-22 · Author: live debug-and-harden session ·
Status: **draft, blocked on fresh seqnn3 producing real scores > 0.37**._

The `rdr` (rubric-driven reproduction) harness is on `rlm_rubric_orchestration`.
This document is the merge plan to land it on `main`.

---

## 1. Gate criteria — all must be ✅ before merge

| # | Gate | Status | Notes |
|---|---|---|---|
| 1 | Full test suite green | ✅ (1380+ passed, 3 skipped) | 138 rdr tests + the pre-existing 1242. |
| 2 | Codex CRITICAL findings fixed | ✅ (commit `c928fb7`) | Scorer exception propagation + path traversal in merge-write. |
| 3 | Codex IMPORTANT findings fixed | ✅ (commit `c928fb7`) | Event-loop blocking, os.chdir, watchdog emergency-report, CLI args. |
| 4 | Watchdog + CWD guard | ✅ (commit `3a73903`) | Controller-level threading.Timer + agent CWD guard with auto-cleanup. |
| 5 | Snapshot exclusion (HF cache, etc.) | ✅ (commit `5b53a10`) | Defense-in-depth + container write failures stay contained. |
| 6 | Scorer off-loop (`asyncio.to_thread`) | ✅ (commit `96b0b63`) | All three event-loop-blocking primitives now off-loop. |
| 7 | **Fresh seqnn3 completes with score > 0.37** | 🟡 in progress | Started 22:14; 2-3h budget; this is the binding gate. |
| 8 | Final docs commit (CHANGELOG, learn, progress) | ⏳ to do | After gate #7 lands; record the live-run score + the bug iteration story. |
| 9 | Interactive rebase to 5-10 commits | ⏳ to do | Reorganize the branch so main gets a tractable history. |
| 10 | Backward-compat smoke (rlm/sdk/offline still work) | ⏳ to do | `python -m backend.cli reproduce <paper> --mode offline` + `--mode rlm --max-wall-clock 60` smoke. |

The merge cannot proceed until **gate #7** binds. If seqnn3 scores ≤ 0.37 we
iterate (debug → fix → re-run) until we beat the rlm baseline.

---

## 2. Branch state at draft time

```
$ git log --oneline origin/main..rlm_rubric_orchestration
c928fb7  rdr: pre-merge fixes per Codex adversarial review (2 CRITICAL + 4 IMPORTANT)
3a73903  rdr fix: controller-level watchdog + agent CWD guard
96b0b63  rdr fix: run score_reproduction off the event loop (asyncio.to_thread)
3e95752  rdr fix: remove stray tsnpe_neurips submodule reference
5b53a10  rdr fix: exclude ephemeral/cache dirs from snapshot; defensive write
ee51f02  rdr — rubric-driven paper-reproduction harness (--mode rdr)
```

6 commits. The big `ee51f02` is itself a 6-phase squash. The other 5 are
live-debug fixes.

---

## 3. Commit cleanup strategy — target 5-7 commits on main

The user wants "5-10 commits total" — squash bug fixes but preserve meaningful
milestones. Proposed final shape after interactive rebase:

| # | Commit subject | Combined from | Rationale |
|---|---|---|---|
| 1 | `rdr — rubric-driven paper-reproduction harness (--mode rdr)` | `ee51f02` (unchanged) | The harness itself. |
| 2 | `rdr fix: snapshot excludes cache/ephemeral dirs + defensive merge writes` | `5b53a10` + `3e95752` | Squash the snapshot fix with its tsnpe_neurips cleanup. |
| 3 | `rdr fix: score_reproduction off the event loop (asyncio.to_thread)` | `96b0b63` (unchanged) | Scorer fix is its own story. |
| 4 | `rdr: production hardening — watchdog + CWD guard` | `3a73903` (unchanged) | Watchdog + CWD guard belong together. |
| 5 | `rdr: pre-merge hardening per Codex adversarial review` | `c928fb7` (unchanged) | The Codex findings sweep. |
| 6 | `rdr: docs + final-run scores` | new docs commit (gate #8) | CHANGELOG / learn.md / progress.md / runlog.md update + the score. |

6 commits on `main` post-merge. Tells the iterative debug-and-harden story
cleanly.

**Mechanics:** because the branch is already pushed to `origin`, the rebase
requires a force-push. If the user prefers to avoid force-push (per the prior
denial on `main`), an alternative is to land the merge as a single squash on
main (matching the prior rdr-milestone style at `2e1ce37`) and keep the per-fix
history on the branch only.

---

## 4. Backward-compat smoke (gate #10)

The merge must not break the three existing run modes. Pre-merge smoke:

```bash
# offline — deterministic, no LLM, no Docker
.venv/bin/python -m backend.cli reproduce sequential-neural-score-estimation \
    --mode offline --max-wall-clock 30

# sdk smoke — short cap
.venv/bin/python -m backend.cli reproduce sequential-neural-score-estimation \
    --mode sdk --max-wall-clock 60

# rlm smoke — short cap, allow it to time out fail-soft
.venv/bin/python -m backend.cli reproduce sequential-neural-score-estimation \
    --mode rlm --max-wall-clock 60
```

Each must exit with status `partial` or `completed` (NOT crash) and write a
`final_report.json`. The rdr mode is additive; no existing path is touched.

---

## 5. Top-3 residual risks if we merge as-is today

1. **Mech-class papers wedge on the Claude SDK aclose() deadlock.** The
   controller-level watchdog (`_ClusterWatchdog`) hard-bounds it to ~15 min
   per cluster, but the run dies with `os._exit(124)` rather than completing.
   Mitigation: the emergency final_report.json is now written; operator sees
   `status="watchdog_killed"` and knows what happened. v2 work: a proper
   subprocess wrapper around the SDK so we can `kill -9` cleanly without async
   cleanup.

2. **Agent CWD guard is detect-and-clean, not contain.** Top-level repo-root
   escapes are caught; subdirectory escapes (`./backend/agents/...`),
   `~/something`, `/tmp/...` are not. Real fix is sandbox containment — defer
   to v2 by running agents inside the docker sandbox.

3. **Live-test sample is one paper.** seqnn3 (sequential-neural-score-estimation)
   beats 0.37; mech-class is wedged by the SDK bug above. We have N=1 success
   on the rubric-driven approach. The harness is designed for any PaperBench
   bundle, but only one bundle has been e2e-validated live. Post-merge work:
   try `ftrl` and other PaperBench bundles in a sandbox env.

---

## 6. Rollback plan

If something breaks after merge:

```bash
# Revert the merge on main (creates a revert commit, doesn't rewrite history)
git checkout main
git revert -m 1 <merge-commit-sha>
git push origin main
```

`--mode rdr` is additive: even if its code is buggy, the other three modes
keep working. No data migration to roll back.

---

## 7. Pre-merge checklist (operator)

- [ ] gate #7: fresh seqnn3 final_report.json shows `rubric.overall_score > 0.37`.
- [ ] gate #8: docs updated (`CHANGELOG.md` cites the score; `learn.md`
      enumerates the bugs surfaced live + their guardrails; `progress.md`
      points at the merge).
- [ ] gate #10: backward-compat smoke (offline/sdk/rlm) clean.
- [ ] gate #9: rebase or squash decided + executed.
- [ ] Confirm `origin/rlm_rubric_orchestration` matches local before the
      merge command.
- [ ] Final `pytest tests/ -n auto -q` green on the cleaned-up branch.
- [ ] Tag the merge commit on main: `rdr-v1` (or similar).

---

## 8. Open questions / known unknowns

- **Azure path is code-correct but live-untested.** The provider abstraction
  reads `OPENAI_BASE_URL` / `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_API_KEY`
  and routes through `OpenAILlmClient(model, base_url, api_key)`. No live
  Azure run has been executed. Acceptable for merge; verifying on Azure is
  post-merge work.
- **No SSE / dashboard wiring.** The controller does not emit
  `dashboard_event` frames (unlike `--mode rlm`). The UI won't show rdr
  progress; rdr is CLI-only for v1. UI integration is post-merge.
- **`--mode rdr` is bundle-only for v1.** arXiv papers with a generated
  rubric (`rubric_gen.py`) are supported by the leaf scorer, but the rdr CLI
  path expects a vendored bundle. Generated-rubric support is post-merge.

---

## 9. After the merge

- Tag `rdr-v1` on the merge commit.
- Open follow-up tickets for:
  - Replace agent SDK subprocess with a wrapper we can `kill -9` cleanly
    (fixes the aclose deadlock — mech-class papers).
  - Sandbox containment so agents truly cannot escape `code_dir`.
  - Wire `dashboard_event` SSE frames for UI parity.
  - Add a `paper_search` SDK tool + (some of) the 9 RLM primitives as tools
    per design spec §4.4 (currently deferred to v2 — the agent uses
    `baseline-implementation`'s Read/Write/Edit/Bash + bash on `paper_full.md`).
  - Live-validate the Azure provider path.
  - Add Phase-5 smoke e2e to CI on a small bundle.
