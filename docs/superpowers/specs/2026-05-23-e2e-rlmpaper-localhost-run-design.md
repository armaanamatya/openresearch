# E2E localhost run of the RLM paper — design

**Date:** 2026-05-23
**Author:** Claude (Opus 4.7) + user
**Status:** approved, in execution
**Related:** `docs/runbooks/known-issues-and-monitoring.md`, `docs/runbooks/e2e-testing.md`, `CLAUDE.md`

## Goal

Run arXiv 2512.24601 (the rlms paper itself) through the ReproLab stack on localhost to terminal state — a written `final_report.json` — exercising the full UI surface and fixing every backend/UI defect found along the way.

This is a meta-reproduction: the orchestrator built on the `rlms` library is asked to reproduce the rlms paper.

## Configuration

| dimension | value | rationale |
|---|---|---|
| paper | arXiv 2512.24601 | the rlms paper |
| mode | `rlm` | hybrid; pure-RDR path would require a PaperBench bundle, none exists for this paper |
| root model | `claude-oauth` | $0 LLM cost via the local `claude` CLI subscription |
| sub-agent auth | OAuth subscription | $0; works after the 2026-05-23 Keychain fix |
| sandbox | `runpod` | GPU access on RTX 4090 COMMUNITY (~$0.34/hr) for `run_experiment` |
| ingest path | UI upload via `/lab` → `POST /api/demo/runs/arxiv` | exercises the user-facing flow, not the CLI |
| success criterion | `runs/<id>/final_report.json` exists, status=`completed` | full reproduction to terminal state |
| budget | wall-clock 90 min target / 180 min hard cap, RunPod spend ≤ $3 | |

## Prerequisite cleanup and fixes

These must be done before kickoff or the run will fail in predictable ways already documented in `known-issues-and-monitoring.md`.

1. **Kill all stale processes**
   - `kill <pid>` for existing uvicorn on :8000 and Next dev on :3000.
   - `pkill -9 -f "claude_agent_sdk/_bundled/claude"` — orphan SDK workers (known-issues §12).
   - Kill orphan Playwright Chrome + remove its `SingletonLock` (known-issues §13).
   - `docker ps -q | xargs -r docker stop` — stale build/exec containers.

2. **Commit two working-tree fixes** (referenced as "resolved 2026-05-23" in the runbook but not in HEAD):
   - `backend/services/context/workspace/tools/rlm_query.py` — SDK aclose deadlock fix (known-issues §3.1, blocker).
   - `frontend/src/hooks/use-resizable-panels.ts` — SSR hydration mismatch fix (known-issues §3.2).

3. **RunPod SSH key:** generate `~/.ssh/reprolab_runpod_ed25519` keypair, wire path + public key into `.env`'s `REPROLAB_RUNPOD_SSH_KEY_PATH` and `REPROLAB_RUNPOD_SSH_PUBLIC_KEY` (currently empty — would cause `run_experiment` to fail).

4. **Preflight:** `scripts/runpod_check.sh` must pass. Abort kickoff on failure.

5. **Boot fresh stack:** `./start.sh` (backend on :8000, frontend on :3000). The launcher runs the RunPod preflight when sandbox is runpod.

## Execution sequence

1. Kickoff via UI: open `/lab`, paste arXiv URL, configure model + sandbox, submit. Capture `projectId` from URL.
2. Start the screenshot tail in background: `node scripts/lab_screenshot_tail.mjs <projectId> 30 &`.
3. Enter the monitoring loop (below).

## Monitoring loop

Two-layer monitoring — heavy lifting is autonomous, agent re-engagement is sparser to stay cache-aware.

**Background (every 30 s, autonomous):** existing `scripts/lab_screenshot_tail.mjs` writes `screenshots/lab-<ts>.png`, `screenshots/wedge-log.tsv`, `screenshots/console-errors-<ts>.json`. Auto-terminates on `final_report.json` or `status=failed|completed`.

**Agent loop (every ~5 min, via `ScheduleWakeup`):** at each firing, read:
- newest `screenshots/lab-*.png`
- `runs/<id>/dashboard_events.jsonl` tail (new events since last check)
- `runs/<id>/runner.stderr.log` tail
- `runs/<id>/demo_status.json`
- `screenshots/wedge-log.tsv` tail
- any new `screenshots/console-errors-*.json`

Then triage. Cadence rationale: 30 s would burn the 5-minute prompt cache twelve times an hour for no signal change; 5 min costs one cache miss but matches the rhythm of real backend events (`implement_baseline` alone is 5-15 min wall-clock).

**One new script** (small, written in prep step): `scripts/health_probe.sh <projectId>` — single-shot health snapshot, called from each loop firing. Returns non-zero if `dashboard_events.jsonl` hasn't grown in 10 min AND no SDK worker process is alive (per the §1b false-alarm check).

## Bug-fix policy

User said "fix everything." Concretely:

| failure class | response |
|---|---|
| Frontend console error or layout regression | fix in-flight, dev-server hot-reload picks it up |
| Backend bug, no restart needed | edit, dev-reload via uvicorn `--reload` |
| Backend bug requiring restart | log finding, restart backend, verify UI re-attaches to live `dashboard_events.jsonl` SSE; the run subprocess survives because state is file-backed |
| Run-killer bug, fix < 15 min, restart cheap | fix, restart run, note the lost iterations |
| Run-killer bug, fix > 15 min | document in findings doc, evaluate workaround, escalate to user |

Every fix gets a one-line entry in a per-session findings doc at `docs/runbooks/2026-05-23-e2e-rlmpaper-run-findings.md`.

## Termination and audit

**Done states (any one):**
- `runs/<id>/final_report.json` exists and `demo_status.json.status == "completed"`.
- `demo_status.json.status == "failed"` AND no recoverable fix exists.
- User cancels.
- Hard wall-clock 180 min cap reached.

**Deliverables on done:**
- Findings doc with every issue + fix, linked to commits.
- One-paragraph summary: did the reproduction land, what was the rubric verdict, how many iterations, total RunPod spend.
- Updated `docs/runbooks/known-issues-and-monitoring.md` §0 status board if any open issues were closed or new ones opened.

## Out of scope

- Tuning the run's quality (rubric score, iteration count) — we judge the *pipeline*, not the *answer*.
- Adding new SSE events, primitives, or sandbox backends.
- Multi-paper sweeps. One paper, one run.
- Modifying the `rlms` PyPI library.
