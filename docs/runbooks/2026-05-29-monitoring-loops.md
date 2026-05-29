# Monitoring loops — SDAR retry sprint and beyond

**Established:** 2026-05-29 during SDAR attempt-6 launch (post BUG-NEW-038 fix).
**Owner:** Claude Code, while a paper reproduction is being driven from this session.
**Goal:** every meaningful failure-mode and visual regression of the system is observed automatically while a run is in flight, so the human user never has to babysit the terminal or the lab UI.

This document is the *single source of truth* for what loops exist, what each one watches, how to arm and stop each, and the convention that turns observation into a documented bug.

---

## Loop catalog

| # | Loop | What it watches | Emits on | Arm | Stop |
|---|---|---|---|---|---|
| 1 | **Run-log monitor** | `runs/<project>/dashboard_events.jsonl` | `repl_iteration`, `primitive_call`, `sub_rlm_complete`, `run_warning`, `rubric_score`, `candidate_proposed`, `run_complete`, `gpu_*`, `preflight_blocked`, `misuse_patch`, `from_pretrained`, errors, `FINAL_VAR` | `Monitor` with `tail -F … | grep --line-buffered -E '<filter>'` | `TaskStop <task-id>` |
| 2 | **CLI launch-log** | `/tmp/sdar-attempt<N>.log` (stdout/stderr of the `python -m backend.cli` subprocess) | Traceback, `ERROR`, `Error:`, `misuse_patch`, `forced_iteration`, `preflight_blocked`, `from_pretrained`, `RubricGuard`, `verdict`, `score`, `FATAL`, `Killed`, `aclose`, `deadlock` | `Monitor` with `tail -F /tmp/sdar-attempt<N>.log` | `TaskStop` |
| 3 | **Backend health probe** | `GET http://127.0.0.1:8000/health` every 60s | line beginning `BACKEND_WEDGED` if `/health` doesn't respond in 3s (BUG-NEW-036/037 wedge signal) | `Monitor` with `while true; do curl …; sleep 60; done` | `TaskStop` |
| 4 | **UI Playwright loop** | `http://localhost:3001/lab?projectId=<LAB_PROJECT_ID>` + `/leaderboard` | new console error, React hydration mismatch, network 5xx, screenshot per cycle | `LAB_PROJECT_ID=<pid> scripts/loops/lab_watch_loop.sh &` (defaults: 5-min interval, screenshots in `/tmp/playwright-ui-loop/`, log in `/tmp/lab-watch-loop.log`); requires running Next server | Ctrl-C the script, or `pkill -f lab_watch_loop` |
| 5 | **Kill+restart loop** | run_complete verdict=failed OR wall-clock exceeded OR backend wedge OR phantom-state (loop 3 fires for >5 min) | `scripts/loops/kill_and_restart.sh <project_id> <next_n> <pdf>` — SIGKILL → patch demo_status → fresh PDF copy → relaunch with `env -u` | `Bash` ad-hoc | n/a |
| 6 | **Heartbeat loop** | nothing (idle tick) | every 1200–1800s the loop wakes itself via `ScheduleWakeup` to re-check state when no Monitor has fired | `ScheduleWakeup` with `<<autonomous-loop-dynamic>>` sentinel | omit the next `ScheduleWakeup` call |

### Filter discipline (silence ≠ success)
Per Monitor tool guidance: a filter that matches only the happy path stays silent through a crash. Loop 1's filter explicitly includes `preflight_blocked`, errors, `FINAL_VAR` (early-exit signal), and warnings — broadening the grep alternation is *always* preferred over narrowing it. If you can't enumerate every failure marker, err on noise.

---

## The doc loop — observation → bug entry → memory

Every loop above is useless if findings don't land in durable storage. The convention:

1. **A loop emits an unexpected event** (notification fires in chat).
2. **Triage in-session**: read the run state, decide if this is a real bug or noise.
3. **If a real bug**: assign the next `BUG-NEW-NNN` number (currently up to 041 as of 2026-05-29) and append a section to the active runbook (`docs/runbooks/2026-05-28-sdar-retry-monitor.md` for the SDAR sprint).
4. **Each entry contains**: (a) symptom (verbatim event or quoted root-model text), (b) root cause once known, (c) fix or "deferred" with reason, (d) validation path.
5. **If the bug is load-bearing across sessions** (will affect future runs in non-obvious ways), also create a memory file at `~/.claude/projects/-Volumes-CS-Stuff-openresearch/memory/<date>-<slug>.md` and add an index line to `MEMORY.md`. The `feedback`/`project` types use the `**Why:** … **How to apply:**` structure.
6. **Cross-link** in both directions: runbook entry links to memory file and vice versa.

This loop is enforced by the autonomous-loop ScheduleWakeup heartbeat — when nothing new is happening, the tick is "scan recent Monitor events, are any of them un-triaged?"

---

## The retest+fix+update loop

After a fix lands during a sprint:

1. Identify which fix and which failure mode it addresses (cite BUG-NEW-NNN).
2. Verify the fix in-process where possible (`python -c "from <module> import <thing>; print(inspect.getsource(...))"` — confirm code path is loaded). For BUG-NEW-038, the verification was `dataclasses.fields(ClaudeAgentOptions)` showing `setting_sources` exists in the SDK.
3. Stage a fresh PDF copy under `/tmp/sdar_<arxiv>_attempt<N+1>_<timestamp>.pdf` so the orchestrator gets a new `project_id` (the orchestrator hashes the PDF path).
4. Launch via `nohup env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY ./.venv/bin/python -m backend.cli reproduce <pdf> --mode rlm --model claude-oauth --sandbox runpod --provider anthropic > /tmp/sdar-attempt<N>.log 2>&1 &`. Always `env -u` to defeat shell-shadow of `.env` credentials.
5. Re-arm loops 1, 2 on the new `project_id`. Loop 3 (backend health) keeps running across attempts.
6. Watch for the validation signal specific to the fix (e.g. for 038: iter 1 root writes a real `understand_section(…)` call instead of describing its tool inventory).

If the fix didn't take, escalate: re-read the symptom, re-check the code, consider an `advisor()` call.

---

## The kill+restart loop

Triggers — any one of:
- `run_complete` with `verdict=failed`
- wall-clock exceeded (`--max-wall-clock` hit; rare with current settings)
- Backend health loop fires `BACKEND_WEDGED` for >5 min straight
- Manual: I observe the root model in a refusal loop, hallucinated-environment loop, or repeated-error spiral

Procedure:
1. `pgrep -f 'prj_<id>' | xargs -r kill` (escalate to `-9` after 5s if it doesn't die).
2. Update `runs/<id>/demo_status.json`: `status='killed'`, add `killedAt`, `killReason`, bump `updatedAt`. This is the BUG-NEW-041 manual patch — until the watchdog fix lands, the UI shows a phantom "running" state otherwise.
3. Increment attempt counter in the runbook; add a new `## Attempt <N+1>` section with the new project_id.
4. Apply the next fix if one was queued.
5. Relaunch via the retest+fix+update loop above.

Per-attempt cap: **5 attempts before I push a notification and stop for user input**. The runbook tracks the count.

---

## Arming loops at the start of a sprint

Boilerplate (run once per fresh session that has an active reproduction):

```bash
# Verify backend is up
curl -s --max-time 3 http://127.0.0.1:8000/health
```

Then (via the Monitor tool, not bash):

- Loop 1 (run-log): `tail -F runs/<current-project>/dashboard_events.jsonl | grep --line-buffered -E '"event":"(repl_iteration|primitive_call|sub_rlm_complete|run_warning|rubric_score|candidate_proposed|run_complete|gpu_resolved|gpu_escalated|gpu_fallback)"|preflight_blocked|misuse_patch|from_pretrained|"error"|"failed"|"FINAL_VAR"'`
- Loop 2 (launch log): `tail -F /tmp/sdar-attempt<N>.log | grep --line-buffered -E 'Traceback|ERROR|Error:|misuse_patch|forced_iteration|preflight_blocked|from_pretrained|RubricGuard|verdict|score|FATAL|Killed|aclose|deadlock'`
- Loop 3 (health): `while true; do if ! curl -s --max-time 3 http://127.0.0.1:8000/health > /dev/null 2>&1; then echo "$(date -u +%H:%M:%SZ) BACKEND_WEDGED"; fi; sleep 60; done`

When you rotate to a new project (next attempt), `TaskStop` the old loop-1 monitor and arm a new one against the fresh `dashboard_events.jsonl`.

---

## Tools the loops have used so far

- `Monitor` — primary event source. Persistent + grep alternation = one notification per real event.
- `Bash` — orientation, one-shot kill commands, demo_status patches.
- `Read` / `Edit` — runbook + memory edits.
- `ScheduleWakeup` — heartbeat when no Monitor has fired in a while.
- `PushNotification` — escalation to mobile/terminal when the user needs to act (`BUG-NEW-038 found`, etc.).
- `TaskCreate` / `TaskUpdate` — sprint progress.
- `advisor()` — second opinion when a root cause is unclear or a fix path is risky.

---

## Open follow-ups for the loop infrastructure itself

- **Loop 4 (Playwright)** — **wired 2026-05-29**: `frontend/e2e/lab-watch.spec.ts` + `scripts/loops/lab_watch_loop.sh`. Cycle = one `npx playwright test` invocation; default 5-min interval. Honors `LAB_PROJECT_ID` (skips the `/lab` half when empty), `LAB_BASE_URL` (defaults to `http://localhost:3001`), `LAB_WATCH_SCREENSHOT_DIR` (defaults to `/tmp/playwright-ui-loop`), `LAB_WATCH_MAX_CYCLES` (0 = unlimited).
- **Loop 5 (kill+restart)** — wired as `scripts/loops/kill_and_restart.sh`.
- **BUG-NEW-041** — **fixed 2026-05-29**: `backend/cli.py:_install_termination_handlers` registers SIGTERM/SIGHUP handlers that flip `demo_status.json::status='killed'` and re-raise as SIGINT (so the existing `KeyboardInterrupt` graceful path runs). `_mark_demo_status_stopped` / `_mark_demo_status_failed` now preserve `killed` instead of overwriting it. The manual patch loop 5 used is still in `kill_and_restart.sh` as defense in depth.
- **BUG-NEW-042** — **fixed 2026-05-29**: `backend/agents/rlm/primitives.py::_validate_dockerfile_shape` validates the first non-blank, non-comment line is `FROM` / `ARG` / `# syntax=`. Used in two places: (a) `implement_baseline` snapshots the pre-run Dockerfile and restores it when the sub-agent's Write tool stomps the file with prose (auto-recover + `dashboard_event` warning); (b) `run_experiment` rejects a malformed Dockerfile fail-fast with `failure_class="dockerfile_invalid"` so the root's next iteration sees a clear `repair_context`.
- **Cross-run aggregator** — currently each loop watches one run. A future "watch all runs" loop would `tail -F` whichever `dashboard_events.jsonl` was most-recently modified, auto-rotating on new runs.
