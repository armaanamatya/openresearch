# ReproLab Live Service Validation — 2026-05-30

> Living doc for the long-run service-validation + monitoring session. Updated each
> operator heartbeat (~5 min). Append history; do not erase. No secrets, no `.env`
> values, no API keys.

## Current Status

- **overall status:** healthy
- **current branch:** `feat/rlm-wedge-hardening`
- **current HEAD:** `f6afc2d` (working tree dirty — pre-existing evidence-gate + codex work, plus this session's BUG-NEW-045 fix)
- **backend URL:** http://127.0.0.1:8000
- **frontend URL:** http://localhost:3000
- **active run id:** none (no live reproduction; 3 historical run dirs)
- **active paper:** none
- **started at (this session):** 2026-05-30 09:55 CDT (backend restarted to apply BUG-NEW-045)
- **last updated at:** 2026-05-30 09:57 CDT
- **current operator loop:** initial discovery + fix + first verify complete; entering monitor cadence
- **current risk:** dirty working tree carries unrelated half-edits; backend now runs that code (boot-verified clean). Background audit found an unfixed HIGH-sev evidence-gate forge loophole (flagged, not fixed).

## Localhost Links

- **Backend health:** http://127.0.0.1:8000/health → `{"status":"ok","version":"0.1.0"}`
- **Frontend home:** http://localhost:3000
- **Lab (RLM dashboard):** http://localhost:3000/lab
- **Leaderboard:** http://localhost:3000/leaderboard
- **Latest run (API):** http://127.0.0.1:8000/runs/latest
- **Run listing (API):** http://127.0.0.1:8000/runs

## Process Table

| service | pid | port | command | status | last checked |
|---|---:|---:|---|---|---|
| backend (uvicorn) | 72628 | 8000 | `.venv/bin/uvicorn backend.app:create_app --factory --port 8000` | healthy (restarted by operator @09:55) | 10:00 |
| frontend (next-server v16.2.6) | 84507 | 3000 | `next dev` (running ~1d7h) | healthy (pre-existing, reused) | 10:00 |
| monitor loop | (bg b6ujkvak2) | — | `scripts/loops/service_monitor.sh` (90s) | running → `logs/service-validation/operator.log` | 10:00 |

> The prior backend (PID 353, manually started 7:35PM, no `--reload`) was SIGTERM-ed at
> 09:55 and replaced so the BUG-NEW-045 fix took effect. Logs now at
> `logs/service-validation/backend.log` (harness-tracked → operator notified on crash).

## Test/Check Loop

| time | check type | result | evidence | action |
|---|---|---|---|---|
| 09:50 | backend route enumerate | 28 paths, `/health` present | openapi.json | — |
| 09:51 | backend probes | `/health` 200, `/leaderboard` 200, `/models` 200, `/auth-status` 200, **`/runs/latest` 500** | urllib | investigate 500 |
| 09:51 | frontend probes | `/` `/lab` `/leaderboard` `/api/demo/leaderboard` all 200 | urllib | — |
| 09:52 | root-cause `/runs/latest` 500 | `RunStatus` Literal missing `killed` → `LiveRunState(**status)` ValidationError | response body + code read | fix (BUG-NEW-045) |
| 09:53 | boot-safety gate | `create_app()` imports clean (34 routes, 0.3s) with dirty tree | python | restart allowed |
| 09:54 | TDD red | new test reproduces exact 500 at `_load_run:898` | pytest | apply fix |
| 09:55 | TDD green + regression | 5/5 new + 98/98 surrounding live-runs tests pass | pytest | restart backend |
| 09:56 | post-fix live verify | `/runs/latest` **200**, `/runs/prj_09047604e591d969` **200**, `/runs` 200, `/health` 200 | urllib + backend.log | fix confirmed live |

## Playwright Loop Results

| time | page | status | console errors | network errors | screenshot/path | notes |
|---|---|---|---|---|---|---|
| 09:58 | /lab | loads (title OK) | 1 (404) | 1: `GET /api/demo?projectId=prj_e67c9b7be5529226` → 404 | `loop-0957-lab.png` | stale localStorage auto-resume pointer to a non-existent run (UI-1, low) |
| 09:58 | /leaderboard | loads, table renders | 0 | 0 | `loop-0958-leaderboard.png` | 2 runs ranked, killed run correctly excluded; both show `partial`/0.00/$0.00 (UI-2) |

## Run Monitor

| time | run id | status | last event | event gap | report? | metrics? | action |
|---|---|---|---|---:|---|---|---|
| 09:57 | prj_09047604e591d969 | killed | (historical) | n/a | check | — | death-spiral run; now parses via /runs |
| 09:57 | pb_mechanistic-understanding_1780068083 | completed | (historical) | n/a | yes | empty | pre-gate evidence-leak (replay fixture source) |
| 09:57 | pb_mechanistic-understanding_1780068784 | completed | (historical) | n/a | yes | empty | pre-gate evidence-leak (replay fixture source) |

## Logs

| time | source | severity | excerpt | interpretation |
|---|---|---|---|---|
| 09:56 | backend.log | INFO | `Uvicorn running on http://127.0.0.1:8000` | fresh backend up |
| 09:56 | backend.log | INFO | `GET /runs/latest HTTP/1.1 200 OK` | fix live |

## Kill / Restart Decisions

| time | target | reason | evidence | command | result |
|---|---|---|---|---|---|
| 09:55 | backend PID 353 | apply BUG-NEW-045 fix (no `--reload`, won't hot-reload) | `/runs/latest` 500 + root cause + boot gate clean | `kill 353` (SIGTERM, single PID — NOT killpg) | gone in 3s, port freed, restarted as 72628, verified 200 |

## Bugs Found

| id | severity | symptom | root cause | fix | test |
|---|---|---|---|---|---|
| BUG-NEW-045 | medium | `GET /runs/latest` (and `/runs/{id}`) → HTTP 500 whenever a run dir has `status="killed"` | `RunStatus = Literal[...]` missing `killed` (BUG-NEW-041 writes it) and `interrupted` (orphan sweep writes it); `LiveRunState(**status)` raised ValidationError | added both terminal states to `RunStatus` (`live_runs.py:45`) | `tests/services/events/test_live_runs_terminal_status.py` (5 tests) |
| AUDIT-FORGE | high (flagged, NOT fixed) | evidence gate defeatable by forging a `success+metrics` row in `experiment_runs.jsonl` | REPL keeps `open`/`__import__` live; root writes the row directly → gate checks content not provenance | deferred — provenance/nonce check in `report.py`; see memory `2026-05-30-evidence-gate-forge-row-loophole` | audit `wfq3uop9l` (not in CI yet) |
| UI-1 | low | lab logs a 404 console error on landing | stale `localStorage` auto-resume pointer to non-existent run `prj_e67c9b7be5529226`; UI should clear pointer on 404 instead of erroring | not fixed (cosmetic; page works) | — |
| UI-2 | medium (honesty) | leaderboard shows 2 runs as `partial` with score 0.00 / $0.00 / no evidence | on-disk `final_report.json` for `pb_mechanistic-understanding_*` was written BEFORE the evidence gate tightened (these are the replay-fixture sources the gate now downgrades to `failed`); historical reports were not backfilled | not fixed (decision needed: backfill historical reports vs leave as-is) | replay fixtures `pb_784`/`pb_083` already encode the corrected `failed` verdict |

## Update — 2026-05-30 10:00 CDT

**Status: healthy.** Initial discovery + fix + first verify + first Playwright pass complete; monitor loop running.

- **Backend:** PID 72628, port 8000, `/health` 200. Restarted at 09:55 to apply BUG-NEW-045 (the old PID 353 ran pre-fix code with no `--reload`). `/runs/latest` now 200 (was 500).
- **Frontend:** PID 84507, port 3000. `/lab` + `/leaderboard` load; leaderboard 0 console errors.
- **Monitor:** bg `b6ujkvak2`, 90s cadence → `operator.log` (first line: `backend[port=up health=200] frontend[port=up] active_runs[none]`).
- **Runs:** no active run. 3 historical dirs; the 2 leaderboard rows are pre-gate hollow `partial`s (UI-2).
- **Issues this pass:** BUG-NEW-045 (fixed, code+test, live-verified); UI-1 (cosmetic), UI-2 (honesty, decision needed), AUDIT-FORGE (high, flagged).
- **Next:** heartbeat ~270s → re-Playwright + refresh this doc; await user decisions on commit + UI-2 backfill + AUDIT-FORGE track.

## Update — 2026-05-30 10:06 CDT

**Status: healthy — no change.** Heartbeat 1. Monitor ran 4 clean cycles (0 ALERTs): `backend[port=up health=200] frontend[port=up] active_runs[none]`. Backend `/health` 200, `/runs/latest` 200 (BUG-NEW-045 fix holding). No active/wedged runs (still 3 historical dirs). Playwright `/lab` **0 console errors** (UI-1 stale-pointer 404 was a one-time initial-load artifact, did not recur) and `/leaderboard` **0 console errors**. Screenshots `loop-1006-lab.png`, `loop-1006-leaderboard.png`. Pending user decisions unchanged (BUG-NEW-045 commit, UI-2 backfill, AUDIT-FORGE track).

## Update — 2026-05-30 10:11 CDT

**Status: healthy — no change.** Heartbeat 2. Monitor clean (0 ALERTs through 10:10:55). Backend `/health` 200, `/runs/latest` 200; no active/wedged runs (3 dirs). Playwright `/lab` + `/leaderboard` both **0 console errors**. Screenshots `loop-1011-*.png`. Pending user decisions unchanged.

## Update — 2026-05-30 10:16 CDT

**Status: healthy — no change.** Heartbeat 3. Monitor clean (0 ALERTs through 10:15:26). Backend `/health` + `/runs/latest` 200; idle (3 dirs, no active runs). Playwright `/lab` + `/leaderboard` both **0 console errors**. Screenshots `loop-1016-*.png`. Pending user decisions unchanged.

## Update — 2026-05-30 10:56 CDT

**Status: healthy — survived host sleep.** Heartbeat 4 (fired late + clock jumped 10:27→10:55 mid-turn → host suspended/slept between interactions).

- **Investigated an operator.log gap** (10:16:56 → next line 10:28:53 → 10:55:01): NOT a failure. All three processes verified alive across the sleeps — backend PID 72628 (:8000), frontend PID 84507 (:3000), monitor PID 78152. The monitor resumes writing on each wake; the gaps are pure host-suspend intervals. **Positive result: the full stack (backend + frontend + monitor loop) survives macOS sleep/wake and resumes cleanly.**
- **Backend:** `/health` 200, `/runs/latest` 200 (fix still holding post-sleep).
- **UI:** `/lab` auto-resumed the **killed run** (`?projectId=prj_09047604e591d969`) and rendered it with **0 console errors** — an end-to-end confirmation of BUG-NEW-045 (this exact path 500'd + broke the page before the fix). `/leaderboard` 0 console errors. Screenshots `loop-1056-*.png`.
- Pending user decisions unchanged.

## Update — 2026-05-30 11:01 CDT

**Status: healthy — no change.** Heartbeat 5. Host awake this interval (monitor cadence normal 90s, 0 ALERTs). Backend `/health` + `/runs/latest` 200; idle (3 dirs). Playwright `/lab` (auto-resumes killed run, renders clean) + `/leaderboard` both **0 console errors**. Screenshots `loop-1101-*.png`. Pending user decisions unchanged.

## Update — 2026-05-30 11:06 CDT

**Status: healthy — no change.** Heartbeat 6. Monitor cadence normal (0 ALERTs through 11:05:47). Backend `/health` + `/runs/latest` 200; idle (3 dirs). Playwright `/lab` + `/leaderboard` both **0 console errors**. Screenshots `loop-1106-*.png`. Pending user decisions unchanged.

## Update — 2026-05-30 11:11 CDT

**Status: healthy — no change.** Heartbeat 7. Monitor cadence normal (0 ALERTs through 11:10:17). Backend `/health` + `/runs/latest` 200; idle (3 dirs). Playwright `/lab` + `/leaderboard` both **0 console errors**. Screenshots `loop-1111-*.png`. Pending user decisions unchanged.

## Update — 2026-05-30 11:16 CDT

**Status: healthy — no change.** Heartbeat 8. Monitor cadence normal (0 ALERTs through 11:14:48). Backend `/health` + `/runs/latest` 200; idle (3 dirs). Playwright `/lab` + `/leaderboard` both **0 console errors**. Screenshots `loop-1116-*.png`. Pending user decisions unchanged.

## Update — 2026-05-30 11:21 CDT

**Status: healthy — no change.** Heartbeat 9. Monitor cadence normal (0 ALERTs through 11:20:49). Backend `/health` + `/runs/latest` 200; idle (3 dirs). Playwright `/lab` + `/leaderboard` both **0 console errors**. Screenshots `loop-1121-*.png`. Pending user decisions unchanged.

## Update — 2026-05-30 11:26 CDT

**Status: healthy — no change.** Heartbeat 10. Monitor cadence normal (0 ALERTs through 11:25:19). Backend `/health` + `/runs/latest` 200; idle (3 dirs). Playwright `/lab` + `/leaderboard` both **0 console errors**. Screenshots `loop-1126-*.png`. Pending user decisions unchanged. (10 consecutive clean heartbeats; ~90 min uptime across sleep/wake cycles.)

## Update — 2026-05-30 11:31 CDT

**Status: healthy — no change.** Heartbeat 11. Monitor cadence normal (0 ALERTs through 11:29:50). Backend `/health` + `/runs/latest` 200; idle (3 dirs). Playwright `/lab` + `/leaderboard` both **0 console errors**. Screenshots `loop-1131-*.png`. **Note:** frontend worker PID changed 84507 → 2037 (Next.js dev worker recycle; port 3000 never dropped, 0 ALERTs, no outage — monitored by port not PID). Pending user decisions unchanged.

## Update — 2026-05-30 11:36 CDT

**Status: healthy — no change.** Heartbeat 12. Monitor cadence normal (0 ALERTs through 11:35:50). Backend `/health` + `/runs/latest` 200; idle (3 dirs). Playwright `/lab` + `/leaderboard` both **0 console errors**. Screenshots `loop-1136-*.png`. Pending user decisions unchanged.

## Update — 2026-05-30 11:41 CDT

**Status: healthy — no change.** Heartbeat 13. Monitor cadence normal (0 ALERTs through 11:40:21). Backend `/health` + `/runs/latest` 200; idle (3 dirs). Playwright `/lab` + `/leaderboard` both **0 console errors**. Screenshots `loop-1141-*.png`. Pending user decisions unchanged.

## Update — 2026-05-30 11:46 CDT

**Status: healthy — no change.** Heartbeat 14. Monitor cadence normal (0 ALERTs through 11:44:51). Backend `/health` + `/runs/latest` 200; idle (3 dirs). Playwright `/lab` + `/leaderboard` both **0 console errors**. Screenshots `loop-1146-*.png`. Pending user decisions unchanged.

## Update — 2026-05-30 11:51 CDT

**Status: healthy — no change.** Heartbeat 15. Monitor cadence normal (0 ALERTs through 11:50:52). Backend `/health` + `/runs/latest` 200; idle (3 dirs). Playwright `/lab` + `/leaderboard` both **0 console errors**. Screenshots `loop-1151-*.png`. Pending user decisions unchanged.

## Update — 2026-05-30 12:01 CDT

**Status: healthy — no change.** Heartbeat 16 (completed alongside a user "update" request). Monitor cadence normal (0 ALERTs through 11:55:23). Backend `/health` + `/runs/latest` 200; idle (3 dirs). Playwright `/lab` + `/leaderboard` both **0 console errors**. Screenshots `loop-1201-*.png`. Pending user decisions unchanged. (~2h of continuous healthy uptime since the 09:55 backend restart.)

## Update — 2026-05-30 12:06 CDT

**Status: healthy — no change.** Heartbeat 17. Monitor cadence normal (0 ALERTs through 12:05:54). Backend `/health` + `/runs/latest` 200; idle (3 dirs). Playwright `/lab` + `/leaderboard` both **0 console errors**. Screenshots `loop-1206-*.png`. Pending user decisions unchanged.

## Update — 2026-05-30 12:11 CDT

**Status: healthy — no change.** Heartbeat 18 (user engaged mid-cycle re: run links). Monitor 0 ALERTs through 12:10:24. Backend `/health` + `/runs/latest` 200. `GET /runs` returns all 3 runs (killed + 2 partial). Playwright `/lab` 0 console errors (leaderboard nav deferred — verified clean prior cycle). Screenshot `loop-1211-lab.png`.

**UI-2 confirmed live in the UI:** user opened `pb_mechanistic-understanding_1780068784` — the lab detail view shows a red **`suspicious_partial: run completed`** badge, `2 total / 2 failed` workers, blocker `SDK success-with-no-text (claude_agent_sdk)`, and `implement_baseline`/`build_environment`/`plan_reproduction` all FAILED. So the `partial` verdict has **no successful experiment behind it** — the UI's `suspicious_partial` flag is the honest surface of the same hollow-verdict issue (UI-2). Reinforces the case for the evidence-gate backfill decision.

## Next Actions

- [x] First Playwright pass: /lab + /leaderboard — console/network errors + screenshot; killed run renders honestly (excluded from leaderboard, correct).
- [x] Stand up the background monitor loop (process/port/runs/event-gap every ~90s).
- [x] BUG-NEW-045: root-cause, TDD fix, restart, live-verify.
- [ ] Heartbeat cadence (~270s) to re-run Playwright + refresh this doc.
- [ ] Decide with user: BUG-NEW-045 fix + test — leave dirty or commit surgically (only `live_runs.py` + its test).
- [ ] Decide with user: UI-2 — backfill historical `final_report.json` through the tightened gate, or leave as documented.
- [ ] Decide with user: open a remediation track for the AUDIT-FORGE loophole (A/B-gated).
