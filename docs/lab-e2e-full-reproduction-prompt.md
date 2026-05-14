# Full end-to-end PDF reproduction test — prompt for a fresh session

Copy the block below into a new Claude Code session at
`/home/abheekp/openresearch`. It drives the **entire** PDF-upload research-paper
reproduction pipeline from upload to `complete`, and verifies the lab UI stays
in lock-step with the backend at every stage.

---

You are validating the OpenResearch lab pipeline **end to end**: a real PDF is
uploaded, the full agent pipeline runs to completion, and the `/lab` UI must
reflect every backend stage transition accurately, in real time, with no
desync, no console errors, and a correct final report.

Read these first — they are the ground truth for architecture and prior work:
- `docs/lab-ui-pipeline-bridge.md` — the FastAPI ↔ Next.js data flow.
- `docs/lab-smoke-test-results.md` — the prior validation pass and the
  `advance_stage` fix that made `pipeline_state.json` write on every stage
  transition (not just gates). That fix is the reason this test can now pass.

**Your job: run the real pipeline in a real browser, all the way to `complete`,
and confirm every claim below. Report PASS/FAIL per row. Only change code if you
find a genuine bug; if you do, fix it at the root (one canonical change + a
guard test), not with a patch, and re-verify.**

## Environment (important — non-default ports)

Ports 3000 and 8000 are occupied by an unrelated "SmartFall Simulator" app on
this machine. Use **8001** (backend) and **3001** (frontend). The Next.js proxy
must be told where the backend is via `REPROLAB_BACKEND_URL`.

```bash
# 0. Pre-flight static checks
cd /home/abheekp/openresearch/frontend
PATH=/home/abheekp/.nvm/versions/node/v22.14.0/bin:$PATH npx tsc --noEmit
PATH=/home/abheekp/.nvm/versions/node/v22.14.0/bin:$PATH npx vitest run \
  src/app/api/demo src/lib/demo src/components/lab src/app/api/lab
cd /home/abheekp/openresearch
.venv/bin/python -m pytest tests/test_live_run_api.py \
  tests/test_live_run_source_artifacts.py tests/test_pipeline_state_persistence.py -q
# Expect: tsc clean, vitest 34/34, pytest 12/12.

# 1. Backend (FastAPI) on 8001
.venv/bin/uvicorn backend.app:create_app --factory --host 127.0.0.1 --port 8001 &
#   wait for: curl -sf http://127.0.0.1:8001/openapi.json

# 2. Frontend (Next.js) on 3001, pointed at the 8001 backend
cd /home/abheekp/openresearch/frontend
PATH=/home/abheekp/.nvm/versions/node/v22.14.0/bin:$PATH \
  REPROLAB_BACKEND_URL=http://127.0.0.1:8001 npm run dev &
#   wait for: "Ready in" in the dev log; it will bind :3001
```

Use Playwright (already installed: `frontend/playwright.config.ts`,
`frontend/e2e/lab-smoke.spec.ts`) headless-chromium. You may extend that spec or
add `frontend/e2e/lab-e2e-full.spec.ts`. Do NOT use the Chrome MCP browser.

## The run under test

- Upload `demo_paper.pdf` (repo root) through the `/lab` upload zone — the
  hidden input is `input[type="file"][aria-label="Upload paper PDF"]`. This is
  the `startUploadedRun` path (POST `/api/demo` multipart).
- A real run uses live Anthropic agent calls. **Budget 25-40 min of wall time**
  to reach `complete`. Per-stage cost on this machine: paper-understanding
  ~3-5 min, artifact-discovery ~4 min, each later stage 1-5 min. Set Playwright
  per-test timeout accordingly (e.g. 45 min) and poll on a 3-5 s interval.
- Capture the `projectId` from the workflow header eyebrow
  (`.workflow-header .eyebrow` → `workflow - <projectId>`). Everything below is
  keyed off it.

## The contract: 14 backend stages → 12 UI nodes

Backend `PipelineStage` (orchestrator.py) advances:
`ingested → paper_understood → artifacts_discovered → environment_built →
plan_created → gate_1_passed → baseline_implemented → baseline_run →
gate_2_passed → improvements_selected → improvements_run → gate_3_passed →
research_map_generated → complete`.

Each transition now calls `PipelineState.advance_stage(...)`, which writes
`runs/<projectId>/pipeline_state.json` atomically. The Next.js layer
(`server-payload.ts`) reads that file and populates `payload.summary.stage`;
`stateMapForRun` (repro-lab-client.tsx) maps it onto the 12 workflow nodes
(`src read env plan impl opt bb aug hor div audit report` /
`Paper Reader Forge Architect Builder Vesta Athena Orion Lyra Pyxis Hermes
Scribe`).

## Test matrix — report PASS / FAIL for each

### 1. Upload + handoff
- Upload accepted, transitions from upload zone to workflow view, 12 node cards
  render, projectId matches `^prj_`.
- `runs/<projectId>/` is created and `demo_status.json` shows `status: running`.
- Browser console: zero uncaught exceptions on upload.

### 2. Live stage sync (the core of this test)
Poll the workflow header counter (`.workflow-meta .mono` → `N/12 agents
complete`) AND `runs/<projectId>/pipeline_state.json` in parallel. For the whole
run, assert they never desync by more than one poll interval:
- Within ~10 s of `pipeline_state.json` showing a new `stage`, the UI counter /
  node states reflect it. (This is the regression the `advance_stage` fix
  targets — verify it holds for *every* transition, not just gate_1.)
- Node animation order is correct: `src` done immediately; `read` runs then
  done at `paper_understood`; `env`+`plan` progress through
  `environment_built`/`plan_created`; `impl` runs at `baseline_implemented` and
  completes by `gate_2_passed`; the path nodes (`opt/bb/aug/hor/div`) animate
  after gate 2; `audit` runs at `research_map_generated`; `report` + all nodes
  done at `complete`.
- **Watch for a known state-map quirk:** in `stateMapForRun`, `plan_created`
  marks `env`+`plan` as `running` again after `environment_built` already
  showed `env` done — so the counter can appear to go 3 → 2 → 4 across
  `environment_built → plan_created → gate_1_passed`. Decide if this is a real
  UI bug (non-monotonic progress is bad UX) and, if so, fix `stateMapForRun` so
  progress is monotonic.
- Final state: counter reaches `12/12`, every node shows the done check,
  `demo_status.json` shows `status: completed`, `pipeline_state.json` shows
  `stage: complete`.

### 3. Gate chips
- Gate 1 / 2 / 3 chips on the plan→impl / impl→audit edges transition
  `pending → checking → passed` (or `caveat`) as the backend writes
  `gate_1`/`gate_2`/`gate_3` into `pipeline_state.json`. No chip stuck on
  `pending` after its gate has a decision on disk.

### 4. Right rail (agent timeline)
- "Live agents" populates and updates as agents start/complete.
- "Reasoning", "Context handoffs", "Decisions" sections all populate during the
  run (not stuck on empty-state copy after the pipeline is well underway).

### 5. Per-node panels — click each of the 12 nodes once it is non-`upcoming`
- `read/env/plan/impl`: panel shows the matching backend stage, telemetry rows
  for the matching agents, and a multi-line "Latest log" tail (≥2 lines).
- `opt/bb/aug/hor/div`: each animates independently; with the run's
  `n_improvement_paths`, the unmatched path nodes end as `skipped` after
  `gate_3_passed` — confirm they don't hang on `running`.
- `audit`: `HermesAuditPanel` renders per-stage step + checkpoint reports with
  status badges once the run reaches audit; empty-state copy shows before that.
- `report`: `ScriptPanel` shows the source PDF (Preview + Download buttons whose
  hrefs include the projectId and actually return `application/pdf` bytes),
  benchmark numbers, and a working Final report link once `complete`.

### 6. Final report integrity
- `GET /api/demo/source-pdf?projectId=<id>` returns 200 + `application/pdf`.
- `GET /api/demo/final-report?projectId=<id>` returns 200 with the generated
  markdown report once the run is `complete`.
- Benchmark card shows real numbers (not "Pending"/"n/a") for a completed run.

### 7. SSE health
```bash
curl -N http://127.0.0.1:8001/runs/<projectId>/events       # backend
curl -N http://127.0.0.1:3001/api/demo/events?projectId=<id> # proxy
```
- Backend stream: `run_state` + `agent_log` + `dashboard_event` + `heartbeat`,
  never stalls.
- Proxy stream: same, PLUS injected `id: synth-N` enriched `run_state` frames.
  With the `advance_stage` fix, a synth frame should now appear on **each stage
  transition** (the enriched hash changes when `pipeline_state.json` updates) —
  confirm synth frames actually show up during an active run, not just once.
- `payload.summary.stage` in the proxy's `run_state` frames tracks the live
  stage, never `null` once `pipeline_state.json` exists.
- Browser DevTools / Playwright network: the EventSource is opened once and not
  reconnect-looping (the useEffect dep is `[run?.projectId, run?.status]`).

### 8. Robustness / regression
- No uncaught browser exceptions for the entire run.
- No `Pipeline exited with status [non-zero]` or `Traceback` in
  `runs/<projectId>/runner.stderr.log`.
- `pipeline_state.json` is always valid JSON when read mid-run (atomic write —
  no truncated reads); no `pipeline_state.json.tmp` left behind at the end.
- Re-run `tests/test_pipeline_state_persistence.py` — the `ast` guard must still
  pass (no bare `state.stage =` assignments crept in).

## Report format

```
| Section | Status | Notes |
|---|---|---|
| 0. Pre-flight        | PASS/FAIL | |
| 1. Upload + handoff  | PASS/FAIL | |
| 2. Live stage sync   | PASS/FAIL | reached stage=____, counter peaked at __/12 |
| 3. Gate chips        | PASS/FAIL | |
| 4. Right rail        | PASS/FAIL | |
| 5. Per-node panels   | PASS/FAIL | tested __/12 nodes |
| 6. Final report      | PASS/FAIL | |
| 7. SSE health        | PASS/FAIL | synth frames seen: __ |
| 8. Robustness        | PASS/FAIL | |
```

For any FAIL: exact symptom + suspected `file:line` + the root-cause fix you
applied (if any) + the command output proving it now passes. If the run does
not reach `complete` within the time budget, report the furthest stage reached
and whether the UI stayed in sync up to that point — partial sync verification
still counts.

## Notes / known caveats (don't re-discover these)
- Return-to-Lab after navigating away does NOT restore an in-flight run — the UI
  has no localStorage/URL rehydration. Out of scope unless you're asked to fix
  it.
- `upcoming` nodes are intentionally non-clickable (opacity 0.4, no-op onClick).
  Only click a node once it is `running`/`done`.
- `tests/test_issue22_orchestrator.py::test_pipeline_stages_are_ordered` fails
  on `main` independently (stale `composition_tested` enum expectation) — not
  your bug, ignore it.
