# Next-session bootstrap prompt — paste this verbatim after `/clear`

_Authored 2026-05-23. Use this to bring a fresh Claude session up to speed on the openresearch E2E debug + test campaign without replaying the entire prior context._

---

## PASTE THIS PROMPT INTO YOUR NEW SESSION

You are resuming a multi-session debug + harden campaign on the openresearch ReproLab codebase. Before doing anything else, read **`docs/runbooks/e2e-testing.md`** end to end. That doc is the canonical reference for the system architecture, the recent fixes, what to test, where to look when things break, and the WSL-specific concerns. Treat it as source of truth.

### Operating rules — non-negotiable

These come from the user's standing instructions and memory. Violate any one and you've burned the user's time.

1. **You are Opus. You plan, analyze, review, and debug.** You **do not** write production code yourself except for the smallest single-line surgical fixes you authored the spec for. **Dispatch a Sonnet sub-agent** (model=`sonnet`, subagent_type=`general-purpose`) for any non-trivial implementation, with a tight, complete spec. The user has explicitly said: "sonnet should never review opus should do planning review and analysis sonnet just writes code." Adhere to that.

2. **Do not use Codex / codex-rescue for any purpose in this session.** The user explicitly said "without using codex." If the auto-classifier offers Codex as a substitute, decline.

3. **Follow `/iterate` working discipline** (the skill loaded at session start — re-read it). The four failure modes it kills: silent assumptions, overcomplication, orthogonal edits, weak success criteria. Specifically:
   - Think before coding — surface uncertainty, ask before guessing
   - Simplicity first — no abstraction for a single use, no speculative flexibility
   - Surgical changes — every line you change traces to the user's request
   - Goal-driven execution — define the verifiable goal before you run, then run it
   - Verify the diff, not the summary — read the code you reviewed, not the agent's report

4. **Memory rules** (already saved):
   - Push to `origin/main` (`armaanamatya/openresearch`). Never `replix`.
   - No `Co-Authored-By` trailer on any commit.
   - Commit infrequently — substantial commits at milestones, not per-fix.

5. **Delegation pattern** (memory: `feedback_delegation.md`, `feedback_review_role.md`):
   - Opus authors the spec → dispatches Sonnet → Opus reviews the diff.
   - For parallel work: dispatch multiple Sonnets in a single response (different files = no merge risk).
   - When a Sonnet reports back, **read the diff yourself**. Trust no summary.

### Where you are in the campaign

Recent commits on `origin/main` (newest first — verify with `git log --oneline -10`):

| Track | SHA prefix | Subject |
|---|---|---|
| Runbook + F6 (test+code) | `e74deca` | docs(runbook) + WSL OAuth credential discovery on factory.py |
| F5 | `49e63e7` | sandbox auto-detect WSL → prefer local when docker unreachable |
| F4 | `65ee5a2` | resolve_root_model aliases (sonnet/opus/claude-sonnet-4-6 → registry keys) |
| F3 | `9da8646` | cross-platform path normalization Win ↔ WSL ↔ macOS |
| F2 | `45e60df` | quiet useRdrArtifacts polling after 3 all-404 cycles + null-render RubricBreakdown |
| F1 | `8d534d8` | hybrid no-bundle paper_id falls back to pure RLM (deployment unblock) |

What each fix does, what its tests cover, and how to verify it locally: §1 + §6 of `docs/runbooks/e2e-testing.md`.

### The current open issue (start here)

A live test (project_id `prj_43c5ed84ae68c19c`) is running on the user's WSL backend. As of the prior session's last check:

- Backend `/runs/<id>` returns `status: "running"` correctly — the run IS executing
- The pipeline reached `run_pipeline_rlm`, resolved `claude-oauth` (F4 worked), spawned the SDK subprocess (alive at `40069`, parent `38099`)
- 1 `repl_iteration` event was emitted after 35.9s (normal — first SDK call takes time for the RLM root prompt)
- BUT the lab UI shows **"Queued"** instead of "Running"
- The frontend proxy `GET /api/demo/runs/<id>` returned empty when curl'd, while the direct backend `GET /runs/<id>` returned the correct state

**The bug**: state transition `queued → running` isn't propagating to the lab UI. Either:
- SSE stream broken (frontend not subscribing to `/api/demo/events`)
- Proxy route `/api/demo/runs/[projectId]` has a bug
- The state-mapper that maps backend `status: "running"` to the UI's "queued"/"running"/etc. is wrong

**Triage approach**:
1. `curl -i http://localhost:3000/api/demo/runs/prj_43c5ed84ae68c19c` — what's the actual response? 200 with empty body? 500? 404?
2. Read `frontend/src/app/api/demo/runs/[projectId]/route.ts` (if it exists) or the SSE handler
3. Check `frontend/src/lib/demo/server-run.ts:45` — that's where the frontend fetches run state
4. If proxy returns the right JSON but UI doesn't update, check the state mapping in `frontend/src/lib/demo/`

If you find the bug, **dispatch Sonnet with the fix spec** following the rules above. Don't fix inline unless it's a 1-line typo.

### How to test end-to-end yourself

The user runs the backend + frontend in tmux on WSL. You have read access to the running stack. To trigger a fresh run for testing:

```bash
# Verify both services are up:
ps -eo pid,etime,cmd | grep -E "uvicorn|next dev" | grep -v grep

# Curl the API directly with a PDF upload (avoid the UI's race conditions):
curl -i -X POST http://localhost:8000/runs/upload \
    -F "file=@runs/prj_43c5ed84ae68c19c/raw_paper.pdf" \
    -F "mode=rlm" -F "model=sonnet" -F "sandbox=local"

# Watch the run progress (no UI needed):
PROJECT=<the project_id returned by the curl>
watch -n 5 "cat runs/$PROJECT/demo_status.json | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d[\"status\"], d.get(\"error\"))'"

# Tail dashboard events as they accumulate:
tail -f runs/$PROJECT/dashboard_events.jsonl | jq -c '{e: .event, t: .iteration, m: .timing}'

# Inspect cluster artifacts (post-Phase-1 if hybrid runs; none for pure RLM):
ls runs/$PROJECT/iterations/ 2>/dev/null
```

### Hard limits

- **No live smoke tests as the primary validation strategy.** The user said "we need real tests only it is failing in deployment." Real tests = pytest + e2e with the live UI. Smoke tests that consume OAuth quota without producing test artifacts are a waste.
- **Cherry-pick to `main` is the deploy mechanism** (the user previously said "push directly to upstream main"). Don't open PRs from feature branches; commit directly to `main` (Sonnet should `git push origin main` at end of work).
- **No force-push to `main`** — ever.
- **OAuth quota is finite** — kill orphan `claude_agent_sdk/_bundled/claude` subprocesses between attempts (`pkill -9 -f "claude_agent_sdk/_bundled/claude"`).

### Open WSL-specific concerns to be aware of

- F5 already auto-degrades `sandbox=auto` to `local` on WSL when docker is unreachable. The user's frontend currently sends `sandbox=docker` explicitly (default in the frontend's `route.ts`), so F5 doesn't fire. Consider: should the frontend's default sandbox also be `local` on WSL, or detected at the API layer? Discuss with user before changing.
- F6 detects Windows-side OAuth credentials and logs a symlink hint, but does NOT auto-symlink (security boundary). User must `ln -s /mnt/c/Users/<x>/.claude/.credentials.json ~/.claude/.credentials.json` themselves if needed.
- `_DEFAULT_TIMEOUT_S = 1800.0` in `ClaudeOauthClient.completion` — 30-minute ceiling per SDK call. The first RLM root call can take 30–90s legitimately. Don't lower this without understanding why.

### Where to find context

- **`docs/runbooks/e2e-testing.md`** — full E2E test surface, diagnostic playbook, log locations
- **`learn.md`** — past bugs and the rules learned from them (especially the 2026-05-22 SDK aclose entry — Workaround B)
- **`CHANGELOG.md`** — what landed when
- **`CLAUDE.md`** — repo conventions, mode flags, RLM auth model (two surfaces — root vs sub-agent)
- **`docs/superpowers/specs/2026-05-22-rubric-driven-harness-design.md`** — RDR design spec

### Acceptance criteria for this session

You're done when **all four** hold:

1. The lab UI shows the run state correctly through the lifecycle (queued → running → completed/partial/failed) — no stuck "Queued" while backend is running.
2. A PDF upload through the UI runs to completion (verdict `completed` or `partial`) — same paper as the user has been testing (`arxiv_2512.24601.pdf`).
3. The RubricBreakdown panel populates ONLY for runs that have RDR artifacts (bundle paper_ids) — and the polling does stop at ~15s on non-RDR runs (no 404 spam on console).
4. Any new fixes added in this session are: committed to `main`, tests pass, runbook updated if a new failure mode discovered.

### Default first action

Pull main, restart your view of the backend logs:

```bash
cd /home/abheekp/openresearch
git pull origin main
git log --oneline -10
# Check whether the prior session's run still exists
ls -dt runs/prj_* | head -3
# Read the runbook BEFORE doing anything else
cat docs/runbooks/e2e-testing.md
```

Then triage the UI-state-stuck bug per the "current open issue" section above.

Good luck. The user has been patient through ~36 hours of debugging — be precise and quick, but don't skip the verify-the-diff step. The /iterate skill's red flags are exactly the ones that would tempt you to skip steps under time pressure. Don't.

---

## END OF PROMPT — paste everything between the horizontal rules above
