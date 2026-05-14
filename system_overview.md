# System Overview — OpenResearch / ReproLab

> Architecture orientation for future Claude Code / Codex sessions. Subsystem-altitude,
> not a file inventory. Keep it current — see *Maintaining this doc* at the end.

## Goal

OpenResearch (ReproLab) is an **agent pipeline that reproduces research papers**. Given
a paper (arXiv link or uploaded PDF) it ingests it, understands the claimed results,
builds an environment, implements and runs a baseline, gates the result, explores
improvement paths, and emits a computed scientific benchmark report comparing the
reproduction against the paper's claims.

## Topology

One Docker image, two processes (`docker/entrypoint.sh`, under `tini`):

- **Backend** — FastAPI (`backend/app.py`, uvicorn), internal `:8000`. Stateless HTTP
  layer; each run is a **long-lived subprocess** it spawns and tracks.
- **Frontend** — Next.js 16 (`frontend/`), public `:$PORT` (3000 default). The "lab"
  UI. Reaches the backend **server-side only** via `/api/demo/*` proxy routes — no CORS.

Run state is **file-backed**, not a service: `runs/<project_id>/` holds
`demo_status.json`, `pipeline_state.json`, `final_report.{json,md}`, `*.jsonl` event
logs, `runner.std{out,err}.log`, the reproduced `code/`, and the Hermes audit chain.
SQLite (`REPROLAB_DATABASE_URL`) is the event/persistence store.

## The pipeline — 14 stages

`backend/agents/orchestrator.py` defines `PipelineStage` — the single source of truth
for stage order:

`ingested → paper_understood → artifacts_discovered → environment_built → plan_created
→ gate_1_passed → baseline_implemented → baseline_run → gate_2_passed →
improvements_selected → improvements_run → gate_3_passed → research_map_generated →
complete`

Each stage is driven by an agent under `backend/agents/`; the orchestrator advances the
stage and checkpoints `pipeline_state.json` after each.

## UI ↔ backend sync

1. User starts a run from the lab UI (`repro-lab-client.tsx`) — arXiv link (fixture) or
   uploaded PDF → `POST /api/demo` → backend `POST /runs` or `/runs/upload`.
2. Backend spawns the pipeline subprocess, writes `demo_status.json`, returns run state.
3. UI switches from `UploadView` to `WorkflowView` (a graph of stage nodes) and opens an
   **SSE** stream: `/api/demo/events` → backend `/runs/<id>/events`.
4. SSE emits `run_state` (full state + current stage), `agent_log` (incremental log),
   and `dashboard_event` frames. `stateMapForRun()` maps the backend stage → graph node
   states; the agent timeline and log panel render the rest.
5. On `complete`, the computed `final_report` replaces the placeholder benchmark.

## Backend subsystems

| Path | Role |
|---|---|
| `backend/app.py` | FastAPI factory; run-start + SSE + report endpoints; the demo-secret gate |
| `backend/config.py` | Pydantic `Settings` (`REPROLAB_` env prefix) — single config source |
| `backend/cli.py` | CLI entrypoint for pipeline runs outside the lab UI |
| `backend/agents/orchestrator.py` | `PipelineStage` state machine; stage advance + checkpoint |
| `backend/agents/pipeline.py` | Pipeline runner — SDK (real agents) and offline (deterministic) modes |
| `backend/agents/*.py` | Per-stage agents: paper understanding, environment detective, baseline implementation, experiment runner, improvement, verification, report generator |
| `backend/agents/execution.py` | Sandbox selection / `ensure_sandbox_mode_available` |
| `backend/agents/runtime/` | LLM provider factory (Anthropic / OpenAI), API-key resolution |
| `backend/services/events/live_runs.py` | `FileLiveRunService` — subprocess spawn, run state, SSE bridge |
| `backend/services/ingestion/` | PDF parsing (PyMuPDF) → paper text |
| `backend/services/runtime/` | Sandbox runtimes incl. `runpod_backend.py` (pod lifecycle) |
| `backend/services/context/` | Knowledge graph + cross-project memory |
| `backend/hermes_audit/` | Hermes audit chain — per-stage oversight checkpoints |
| `backend/persistence/`, `backend/eventstore/` | SQLite schema, repositories, event store |
| `backend/evals/` | Evaluation harness + PaperBench scoring |
| `backend/services/{approval,comparison,diagnostics,scoring,verification,orchestration,worktrees}` | Supporting services |

## Frontend parts

| Path | Role |
|---|---|
| `frontend/src/app/page.tsx` | Landing page (`StellarHero`), links to `/lab` |
| `frontend/src/app/lab/page.tsx` → `components/lab/repro-lab-client.tsx` | The lab UI — upload view + workflow graph + agent timeline + logs |
| `frontend/src/app/api/demo/*` | Server-side proxy routes to the backend (run, events SSE, reports) |
| `frontend/src/proxy.ts` | Next 16 proxy — the whole-app unlock gate (active when `REPROLAB_DEMO_SECRET` is set) |
| `frontend/src/app/unlock/` + `api/unlock/` | Unlock screen + session-cookie issuance |
| `frontend/src/lib/events/` | SSE event adapters → the dashboard event contract |

## Key conventions

- **Run modes** — `sdk` (real LLM agents) vs `offline` (deterministic, no keys). The
  refactored lab UI is SDK-only.
- **Sandbox modes** — `local` (host), `docker` (daemon), `runpod` (remote GPU). The
  deployed UI requests `runpod`; `REPROLAB_FORCE_SANDBOX` pins it deployment-wide.
- **The demo gate** — `REPROLAB_DEMO_SECRET`, when set, gates the whole frontend (proxy
  + unlock screen) and the backend run-start endpoints. Unset = fully open (local dev).
- **Ports** — backend `8000` (internal), frontend `$PORT`. Locally 8000/3000 may be
  taken — use 8001/3001.
- **Python** — always the project venv (`.venv/bin/python`).

## Doc index

- [`learn.md`](learn.md) — post-mortems: bugs shipped + the guardrail for each.
- [`docs/deployment.md`](docs/deployment.md) — production + Railway deployment runbook.
- [`docs/reprolab-agent-prd.md`](docs/reprolab-agent-prd.md) — product requirements doc.
- [`docs/agent-lifecycle.md`](docs/agent-lifecycle.md), [`docs/lab-ui-pipeline-bridge.md`](docs/lab-ui-pipeline-bridge.md) — agent + UI-bridge detail.
- [`docs/setup-guide.md`](docs/setup-guide.md) — local dev setup.
- `README.md` — quick start.

## Maintaining this doc

This file is the architecture orientation for new sessions. When a change makes a
statement here wrong — a renamed subsystem, a changed stage list, a new top-level part,
a moved boundary — **update this doc in the same change**. If you notice drift between
the code and this doc, fix the doc; stale orientation is worse than none.
