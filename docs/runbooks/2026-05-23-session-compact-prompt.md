# Compact session bootstrap — paste the fenced block, then /clear

Usage: copy everything between the `---` lines into a fresh session. This file is NOT committed; regenerate per session with the agent.

---

```
You are resuming work on openresearch/ReproLab (WSL, /home/abheekp/openresearch).

## Operating rules
- Opus plans, analyses, reviews diffs. Sonnet executes (writes code). Never reverse.
- Verify the diff itself, not the agent summary. Read every changed file.
- Commit at meaningful milestones (each unique major change = its own commit).
- Headlines descriptive, no Conventional Commits prefixes.
- Git author: lolout1 / appradhann@gmail.com — use git config defaults, no -c overrides.
- No Co-Authored-By trailer. Push to origin (openresearch), never replix.
- Don't break unique features. Refactor + resolve elegantly.
- "no codex" applies to my agent dispatch (don't use codex:rescue or similar during development).
- Memory rules live at /home/abheekp/.claude/projects/-home-abheekp-openresearch/memory/.

## Where you are (verify: git log --oneline -10)
Today's recent commits (newest first):
  f8546d1  run_experiment cap was 2 hours — B2 of the paper sweep wedged for it; now 30 min with env-var escape hatch
  991517a  Leaderboard stopped 500ing on legacy final_report shapes — defensive coerce to {} for the four header fields
  9dd7c6d  The root was declining every candidate to 'save cost' — now told to try a scoped-down subset before declining anything
  1f72e07  Promoted-candidate gate stops getting blocked by a wire-contract bug — candidate_id=None corrupted every outcome event
  7970506  Parallel paper sweep was killing the second ingest with 'database is locked' — BEGIN IMMEDIATE + 30s busy_timeout fixes it
  19e87ee  The 'stuck Running' bug — runs that finish now actually flip to Completed

The 4-paper E2E sweep run today:
  A1 prj_f4cc5fa917c27ef1  2512.24601 RLM        completed  rubric 9.6%   0/3 promoted  (old prompt)
  A2 prj_390202710d0f994b  2512.18131 LLM CodeGen completed  rubric 14.4%  0/3 promoted  (old prompt)
  B1 prj_7b7b34eb9d623b75  2602.01785 CodeOCR     completed  rubric 31.88% 1/3 PROMOTED ✓ (new prompt — hit user's success gate!)
  B2 prj_77b7294aed1bf872  2602.17186 Visual Info Gain  stopped manually (CPU-bound train loop, killed twice, model kept regenerating same)

Live URLs:
  - Lab: http://localhost:3000/lab?projectId=<id>
  - Leaderboard: http://localhost:3000/leaderboard

Backend: PID 84423 alive on :8000 with REPROLAB_RLM_ROOT_MODEL=claude-oauth + new live_runs.py module already loaded.
Restart command (if backend killed): REPROLAB_RLM_ROOT_MODEL=claude-oauth nohup .venv/bin/uvicorn backend.app:create_app --factory --port 8000 > /tmp/backend.log 2>&1 &
Frontend: PID 91988 next-server :3000 (started by user, leave alone).

## What's solid (don't break)
- Wrapper template invariant: write_status("completed") BEFORE finalize_benchmark + finally os._exit. (Pinned: tests/services/events/test_live_runs_status_ordering.py × 5)
- SQLite eventstore: all writers BEGIN IMMEDIATE + 30s busy_timeout. (Pinned: tests/test_eventstore_sqlite_concurrent.py × 3)
- candidate_id wire contract: primitive rejects None/'None'/'null'/''; binding skips emit on failure; system_prompt mandates path_N. (Pinned: tests/rlm/test_binding.py × 4)
- Anti-decline prompt: IMPROVEMENT_LOOP section frames "promoted candidate" as the goal, "scoped-down subset" as the fallback.
- Leaderboard defensive coerce: _as_dict(v) at read boundary. (Pinned: tests/routes/test_leaderboard_http.py × 1)
- run_experiment default cap 1800s; REPROLAB_RUN_EXPERIMENT_TIMEOUT_S env var for override. (Pinned: tests/rlm/test_run_experiment_timeout.py × 3)
- 729 tests passing (1 xfailed pre-existing) across rlm/rdr/routes/services/events/agents + sqlite test files.

## Open issues (tracked, not blocking)
- SDK aclose deadlock: watchdog flags it (sets degraded=True), wrapper os._exit bypasses atexit hangs. No upstream SDK fix yet.
- implement_baseline agent doesn't know sandbox_mode. B2's CPU-infeasible baselines are the symptom. Fix candidate: thread ctx.sandbox_mode into the agent's prompt so it picks --smoke-test for CPU sandboxes.
- Zombie Claude subprocesses on long runs (B2 accumulated 72). WSL2 subprocess.wait leak. Cosmetic unless multi-hour runs.
- Codex-companion plugin auto-invoked from sub-agent (claude-agent-sdk plugin). Sub-agent's autonomous choice, not Claude Code's agent dispatch.
- Vercel + Azure production migration: provider abstraction done (Azure root), state migration + multi-tenant auth still pending.
- Manual container kill recipe (when run_experiment wedges): see uiprogress.md "Manual-intervention runbook" section.

## First action when continuing
1. Read uiprogress.md late-evening entry (search for "2026-05-23 (late evening)") for the full F18–F23 rule set.
2. Run: git log --oneline -10
3. Ask the user what they want to tackle. Do not assume continuation of any prior task.
   Likely next-task candidates the user mentioned but didn't ask for yet:
   - Thread ctx.sandbox_mode into implement_baseline prompt (would have unblocked B2)
   - Suppress cosmetic SDK aclose stderr lines
   - Add a CLI command to trigger a paper-sweep across N papers via one command

## Acceptance criteria for any change this session
- Any change survives pkill + restart of the backend (./start_backend.sh).
- Test suite passes:
    .venv/bin/python -m pytest tests/rlm/ tests/rdr/ tests/routes/ tests/services/events/ tests/agents/ tests/test_eventstore_sqlite{,_concurrent}.py -q
- Frontend type-clean:
    cd frontend && npx tsc --noEmit
- Final answer is honest: state what works AND what remains open.
- API/OAuth parity: any new fix must work for both auth surfaces (verify with grep — should be auth-agnostic by construction at our edit layer).
```

---

Notes:
- Paste everything between the `---` lines (the fenced block inclusive) into a fresh Claude Code session after `/clear`.
- The paste content above is intentionally terse — the full architecture is in CLAUDE.md, system_overview.md, and docs/design/.
- This file is NOT committed to git and will show as `??` in git status. Regenerate each session with: "create compact session prompt (don't commit)".
