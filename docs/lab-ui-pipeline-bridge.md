# Lab UI ↔ FastAPI pipeline bridge

This doc explains the data path that drives the new lab workflow graph at
`/lab` and the contract between the FastAPI backend and the Next.js frontend.

## Why this exists

A May 2026 merge re-routed `/api/demo` from the Node-side runner
(`frontend/src/lib/demo/node-runner.ts`) — which used to assemble the rich
`payload` object the UI needs — to a thin proxy of the FastAPI live-run
service. The FastAPI service spawns the pipeline subprocess but does **not**
build `payload`; `_load_run` returns `payload = None`. Symptom: the workflow
graph stayed permanently at "src done / read running".

The fix bridges `runs/<id>/pipeline_state.json` → `LiveDemoPayload` in the
**Next.js API layer**, then expands the UI to surface every backend signal.

## Data flow

```
FastAPI (uvicorn :8000)
  POST /runs            spawns python subprocess
  GET  /runs/{id}       returns LiveRunState (payload always null)
  GET  /runs/{id}/events SSE: run_state | agent_log | dashboard_event
                                | agent_failed | heartbeat
  GET  /runs/{id}/source-pdf
  GET  /runs/{id}/final-report

  ↓ subprocess writes
runs/<projectId>/
  pipeline_state.json    (PipelineState.advance_stage in orchestrator.py:139, every stage transition)
  demo_status.json       (status: queued | running | completed | failed)
  dashboard_events.jsonl (one JSON event per line; live agent activity)
  runner.stderr.log      (raw log tail)

Next.js (:3000)  — proxies all backend traffic and ENRICHES on the way out
  /api/demo                 → FastAPI POST/GET/DELETE; GET enriches body with payload
  /api/demo/events          → FastAPI SSE; TransformStream enriches run_state frames
  /api/demo/source-pdf      → FastAPI passthrough
  /api/demo/final-report    → FastAPI passthrough

  enrichRunStateWithPayload(state)  (server-payload.ts)
    reads pipeline_state.json (last-good cache w/ LRU+TTL)
    + log tail + telemetry tail
    → buildLiveDemoDashboard(...) → state.payload populated
```

## Files

| Path | Role |
|---|---|
| `frontend/src/lib/demo/server-fs.ts` | FS-only helpers (runDir, readPipelineState, readLogTail, metaFromStatus, readTelemetryTail). `"server-only"`. |
| `frontend/src/lib/demo/server-payload.ts` | `enrichRunStateWithPayload`. Bounded LRU cache (64 entries × 30 min TTL) of last-good pipeline_state to survive non-atomic backend writes. |
| `frontend/src/lib/demo/pipeline-dashboard.ts` | `buildLiveDemoDashboard` returns a `LiveDemoPayload` with: `summary.stage`, `pathStates`, `decisionLog`, `assumptionCount`, `gates`, `hermes`, `events`, `initialSnapshot`. |
| `frontend/src/app/api/demo/route.ts` | GET enriches the FastAPI `/runs/{id}` response (750ms timeout fallback). |
| `frontend/src/app/api/demo/events/route.ts` | SSE proxy with TransformStream that (a) enriches every backend `run_state` frame, (b) emits a synthetic enriched `run_state` after each `dashboard_event` or 3s idle if the stable hash changed. 250ms enrichment timeout, fire-and-forget so the live stream never backpressures. |
| `frontend/src/components/lab/repro-lab-client.tsx` | Workflow graph + 3-column layout. `stateMapForRun` covers all 14 PipelineStage values. `pathStateMap` (in `pipeline-dashboard.ts`) drives opt/bb/aug/hor/div independently. SSE listener for `dashboard_event` populates `dashboardEvents` state slice. EventSource useEffect depends ONLY on `[run?.projectId, run?.status]` to avoid reconnect loops. |
| `frontend/src/components/lab/agent-timeline-rail.tsx` | Always-visible right rail: Live agents · Reasoning · Context handoffs · Decisions. |

## Backend → UI signal mapping

| Backend produces | UI surface |
|---|---|
| `PipelineStage` (orchestrator.py:92) | `stateMapForRun` covers all 14 values: ingested → paper_understood → artifacts_discovered → environment_built → plan_created → gate_1_passed → baseline_implemented → baseline_run → gate_2_passed → improvements_selected → improvements_run → gate_3_passed → research_map_generated → complete. |
| `PathResult[]` (`schemas.py:227`) | Keyword-bucketed into 5 nodes (opt/bb/aug/hor/div) by `pathStateMap`; round-robin fallback for unmatched. |
| `gate_1` / `gate_2` / `gate_3` `GateDecision` | `GateChips` overlay on edges. Backend `GateStatus` enum (verified, verified_with_caveats, partial_reproduction, failed_reproduction, blocked_requires_human, invalid_claim) is normalized to `passed/caveat/failed/running/pending` by `normalizeGateStatus` in pipeline-dashboard.ts. |
| `hermes_step_reports` / `hermes_checkpoint_reports` / `hermes_interventions` | `HermesAuditPanel` rendered when audit node is selected. |
| `decision_log[]`, `assumption_ledger[]` | Right rail "Decisions"; assumption count in plan-node panel. |
| `dashboard_event`s (`agent_started`/`agent_completed`/`agent_failed`/`agent_reasoning_step`/`shared_state_updated`/`context_enrichment`/`hermes_check_updated`) | Right rail (`agent-timeline-rail.tsx`) and per-node panel via `agentMatchers`. |
| `agent_log` SSE | Multi-line tail in agent panel; full log in side-panel "Live activity" section. |

## Gotchas

- **`pipeline_state.json` is non-atomic** (`save_checkpoint` in orchestrator.py uses direct `path.write_text`). `readPipelineState` returns `null` on a half-written read; `server-payload.ts` falls back to the last-good cache.
- **Every stage transition writes the checkpoint** via `PipelineState.advance_stage(stage, runs_root)` (orchestrator.py:139). Earlier versions only wrote at gate boundaries (gate_1/2/3/complete), which left `pipeline_state.json` absent for 5–15 min on fresh runs and stranded the UI counter at `1/12`. Always go through `advance_stage`, not bare `state.stage = X`.
- **Backend only emits `run_state` when `demo_status.json` changes**, but stage transitions are written to `pipeline_state.json`. That's why the SSE proxy injects synthetic `run_state` frames on `dashboard_event` arrival or 3 s idle.
- **`generatedAt` in `LiveDemoPayload` is volatile** — the SSE dedupe hash uses `stableEnrichedHash` which strips `generatedAt`/`lastUpdated`/`timestamp` before hashing, otherwise every tick would emit a synthetic frame.
- **EventSource useEffect deps narrowed to `[run?.projectId, run?.status]`** intentionally. Depending on the full `run` would reconnect on every state update.
- **Path bucket heuristic only matches keywords in `path_id` + `hypothesis`** (lr/optimizer/scheduler → `opt`, backbone/architecture/encoder → `bb`, augmentation/dropout → `aug`, horizon/rollout/n_steps → `hor`, diffusion/ddim/ddpm → `div`). With `n_improvement_paths=1` (the demo default) only one of the 5 nodes will animate per round; the others go to `skipped` after `gate_3_passed`.
- **Pre-existing `landing-page.test.tsx` failures are unrelated** to the lab pipeline — they break with or without these changes.
- **`server-only` import** breaks vitest unless tests `vi.mock("server-only", () => ({}))` at the top (see `node-runner.test.ts` for the pattern).

## Quick start (dev)

```bash
# Backend (FastAPI on :8000)
cd /home/abheekp/openresearch
.venv/bin/uvicorn backend.app:create_app --factory --host 127.0.0.1 --port 8000 --reload

# Frontend (Next on :3000)
cd /home/abheekp/openresearch/frontend
PATH=/home/abheekp/.nvm/versions/node/v22.14.0/bin:$PATH npm run dev
```

Open http://localhost:3000/lab.

## Verification

```bash
# Frontend
cd frontend && PATH=/home/abheekp/.nvm/versions/node/v22.14.0/bin:$PATH \
  npx tsc --noEmit && \
  npx vitest run src/app/api/demo src/lib/demo src/components/lab src/app/api/lab && \
  npx eslint src/lib/demo src/app/api/demo src/components/lab && \
  npx next build

# Backend
.venv/bin/python -m pytest tests/test_live_run_api.py tests/test_live_run_source_artifacts.py -q
```
