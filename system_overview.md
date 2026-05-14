# System Overview — OpenResearch / ReproLab

> Orientation for new Claude Code / Codex sessions: the non-obvious "why" and "how it
> fits together." For *what's where*, read the code — it's named by function. Keep this
> current; see *Maintaining this doc*.

## Goal

OpenResearch (ReproLab) is an **agent pipeline that reproduces research papers**: given
a paper (arXiv link or uploaded PDF) it ingests it, understands the claimed results,
builds an environment, implements and runs a baseline, gates the result, explores
improvement paths, and emits a computed benchmark report comparing the reproduction
against the paper's claims.

## How it fits together

One Docker image, two processes (`docker/entrypoint.sh` under `tini`):

- **Backend** — FastAPI (`backend/app.py`), internal `:8000`. Stateless HTTP layer; each
  run is a **long-lived subprocess** it spawns and tracks.
- **Frontend** — Next.js 16 (`frontend/`), public `:$PORT`. The "lab" UI; reaches the
  backend **server-side only** via `/api/demo/*` proxy routes (no CORS).

Run state is **file-backed**, not a service — `runs/<project_id>/` holds
`demo_status.json`, `pipeline_state.json`, `final_report.*`, `*.jsonl` event logs, the
reproduced `code/`, and the Hermes audit chain. SQLite (`REPROLAB_DATABASE_URL`) is the
event/persistence store.

The pipeline is a **14-stage state machine** — `PipelineStage` in
`backend/agents/orchestrator.py` is the source of truth for stage order; the orchestrator
advances the stage and checkpoints `pipeline_state.json` after each. Each stage has an
agent under `backend/agents/`.

## The run lifecycle (UI ↔ backend)

This is the part worth knowing up front — the rest you can read directly.

1. Lab UI (`repro-lab-client.tsx`) starts a run — arXiv link or uploaded PDF →
   `POST /api/demo` → backend `POST /runs` or `/runs/upload`.
2. Backend spawns the pipeline subprocess, writes `demo_status.json`, returns run state.
3. UI switches to the workflow graph and opens an **SSE** stream
   (`/api/demo/events` → backend `/runs/<id>/events`).
4. SSE frames: `run_state` (full state + stage), `agent_log` (incremental log),
   `dashboard_event`. `stateMapForRun()` maps the backend stage → graph node states.
5. On `complete`, the computed `final_report` replaces the placeholder benchmark.

## Where to look

- **Backend entry points** — `app.py` (HTTP), `cli.py` (CLI / non-UI runs),
  `agents/orchestrator.py` (pipeline state machine), `agents/pipeline.py` (run modes),
  `services/events/live_runs.py` (subprocess spawn + SSE bridge). The rest of
  `backend/{agents,services}/` is named by function — read it directly.
- **Frontend entry points** — `app/lab/page.tsx` → `components/lab/repro-lab-client.tsx`
  (the lab UI), `app/api/demo/*` (backend proxy routes), `proxy.ts` (the unlock gate),
  `app/unlock/` + `app/api/unlock/` (unlock screen + session cookie).

## Conventions worth knowing

- **Run modes** — `sdk` (real LLM agents) vs `offline` (deterministic, no keys). The
  refactored lab UI is SDK-only.
- **Sandbox** — `local` / `docker` / `runpod`. The UI requests `runpod`;
  `REPROLAB_FORCE_SANDBOX` pins it deployment-wide. `REPROLAB_DEFAULT_SANDBOX` does *not*
  override an explicit request — only `FORCE_SANDBOX` does.
- **The demo gate** — `REPROLAB_DEMO_SECRET`, when set, gates the whole frontend
  (`proxy.ts` + unlock screen) and the backend run-start endpoints. Unset = fully open.
- **Local dev** — ports 8000/3000 may be taken; use 8001/3001. Always `.venv/bin/python`.

## Docs

- `learn.md` — post-mortems: bugs shipped + the guardrail for each.
- `docs/deployment.md` — production + Railway deployment runbook.
- `docs/reprolab-agent-prd.md` — product requirements.
- `docs/agent-lifecycle.md`, `docs/lab-ui-pipeline-bridge.md` — agent + UI-bridge detail.
- `docs/setup-guide.md`, `README.md` — local setup + quick start.

## Maintaining this doc

Orientation only — keep it at the "why / how it fits" altitude, never an inventory of
files (those drift and are cheap to rediscover). When a change makes a statement here
wrong, fix the doc in the same change.
