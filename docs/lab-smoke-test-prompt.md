# Lab pipeline smoke test — prompt for next session

Copy-paste the block below into a fresh Claude Code session at
`/home/abheekp/openresearch`. The session will exercise the full UI/UX flow
end-to-end and report what works vs. what doesn't.

---

You are validating an end-to-end fix for the OpenResearch lab pipeline. The
prior session bridged a botched merge that had left the `/lab` workflow graph
permanently stuck. Read `docs/lab-ui-pipeline-bridge.md` first — it explains
the architecture in one page.

**Your task: run the actual UI in a browser and confirm every claim below.
Report a pass/fail line for each. Do NOT change code unless you find a
genuine bug; if you do, fix it surgically and re-verify.**

## Setup

```bash
# Verify static checks still pass before touching the UI
cd /home/abheekp/openresearch/frontend
PATH=/home/abheekp/.nvm/versions/node/v22.14.0/bin:$PATH npx tsc --noEmit
PATH=/home/abheekp/.nvm/versions/node/v22.14.0/bin:$PATH npx vitest run \
  src/app/api/demo src/lib/demo src/components/lab src/app/api/lab

cd /home/abheekp/openresearch
.venv/bin/python -m pytest tests/test_live_run_api.py tests/test_live_run_source_artifacts.py -q
```

Expected: tsc clean, 34/34 frontend tests pass, 7/7 backend tests pass.

## Start the stack

In two separate terminals (or as background jobs):

```bash
# Backend
cd /home/abheekp/openresearch
.venv/bin/uvicorn backend.app:create_app --factory --host 127.0.0.1 --port 8000

# Frontend
cd /home/abheekp/openresearch/frontend
PATH=/home/abheekp/.nvm/versions/node/v22.14.0/bin:$PATH npm run dev
```

Wait for "Ready in" from Next.js. Open http://localhost:3000/lab in
the Chrome MCP browser (`mcp__claude-in-chrome__navigate`).

## Test matrix — report PASS / FAIL for each

### A. Fixture run (no PDF upload)

1. Click the "Begin a fixture run" button (or whatever the start button is on
   the upload screen — the screenshots in `Screenshot 2026-05-10 *.png` show
   the UI before changes).
2. **PASS criteria within 30 seconds:**
   - Workflow graph renders with 12 nodes (src, read, env, plan, impl,
     opt, bb, aug, hor, div, audit, report).
   - `src` immediately shows `done`.
   - `read` advances to `running` then `done` as backend stage moves through
     `paper_understood` → `artifacts_discovered`.
   - `env` and `plan` advance.
   - The header counter (`N/12 agents complete`) increments visibly past 1.
   - Right rail "Live agents" shows agent_started/completed entries flowing in.
   - Right rail "Reasoning" populates with paper-understanding reasoning steps.
   - Gate 1 chip on the plan→impl edge transitions from `pending` →
     `checking` → `passed`.

### B. PDF upload run (the one that "used to not work")

1. Return to the upload screen (click "New paper" if mid-run).
2. Drag any PDF (e.g. `demo_paper.pdf` from repo root if present, or any
   small PDF) onto the upload zone, OR click and select a PDF.
3. **PASS criteria:**
   - Upload accepts the PDF and transitions to the workflow view (no errors).
   - The Source PDF and benchmark fields appear in the report node's
     "Script panel" (Preview PDF + Download buttons).
   - Backend logs (`runs/<projectId>/runner.stderr.log`) show the pipeline
     advancing through stages — confirm with `tail -f`.
   - Workflow graph nodes animate exactly like the fixture run.
   - Browser console shows NO uncaught exceptions.

### C. Navigation

1. Click the brand/logo (top-left). Should reset back to the upload screen.
2. Click into Library / Hermes nav items. Should navigate without crashing.
3. Return to Lab. The previously-running run should still be active (state
   restored from FastAPI).

### D. Per-backend-signal accuracy

For an in-flight fixture run, click each node and verify:

- **read / env / plan / impl**: panel shows backend stage, telemetry rows
  for the matching agents (paper-understanding for read; environment-detective
  for env; reproduction-planner for plan; baseline-implementation +
  experiment-runner for impl). Multi-line "Latest log" tail (≥2 lines, NOT
  one-at-a-time as before).
- **opt / bb / aug / hor / div**: each animates independently. With the
  default `n_improvement_paths=1`, exactly one of the 5 should reach `done`;
  the other 4 should display `skipped` after `gate_3_passed`.
- **audit**: HermesAuditPanel renders per-stage step + checkpoint reports
  with status badges. Empty-state copy ("No audit findings yet") shows if the
  pipeline hasn't reached audit.
- **report**: ScriptPanel shows source PDF preview, benchmark numbers (or
  "Pending" for uploaded runs), and Final report link.

### E. SSE health check

In a third terminal:
```bash
curl -N http://127.0.0.1:8000/runs/<projectId>/events
```
Expect `event: run_state` (initial) + `event: agent_log` + `event: dashboard_event`
+ `event: heartbeat` (every 15 ticks). Should not stall.

Then check the proxy:
```bash
curl -N http://127.0.0.1:3000/api/demo/events?projectId=<projectId>
```
Expect the same events PLUS injected `event: run_state` frames with
`id: synth-N` (synthetic enriched frames). The `data` JSON should now have
`payload.summary.stage` populated (was `null` pre-fix).

## Bug class to watch for

- Workflow graph stuck at "1/12 agents complete" → enrichment broken; check
  `pipeline_state.json` exists and is readable.
- Right rail empty → `dashboard_event` listener not firing or backend not
  emitting events.
- EventSource keeps reconnecting (visible in browser DevTools Network tab as
  repeated `eventsource` requests) → useEffect dep regression.
- Synthetic `run_state` frames flooding the stream every 1s with identical
  payload → `stableEnrichedHash` stripping insufficient volatile keys.
- HermesAuditPanel crashes the audit node panel → null guard regression.

## Report format

Reply with a markdown table:

| Section | Status | Notes |
|---|---|---|
| Pre-flight tests | PASS/FAIL | |
| A. Fixture run | PASS/FAIL | |
| B. PDF upload | PASS/FAIL | |
| C. Navigation | PASS/FAIL | |
| D.read / D.env / ... | PASS/FAIL each | |
| E. SSE | PASS/FAIL | |

Then for any FAIL: paste the exact symptom + the file:line you suspect + the
fix you applied (if any).
