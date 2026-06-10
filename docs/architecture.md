<!-- doc-meta: status=current; last-verified=2026-06-09 -->
# Architecture

OpenResearch is a single-user paper-reproduction system: ingest a paper, let an
RLM root model orchestrate domain primitives, run generated experiments in a
sandbox, score the output, and write a report.

## Topology

```text
Browser
  -> Next.js frontend (:3000)
      -> server-side /api/demo/* proxy
          -> FastAPI backend (:8000)
              -> long-lived run subprocess
                  -> RLM / RDR controller
                      -> local, docker, runpod, or azure sandbox
```

The browser does not call FastAPI directly. The frontend proxy target is
`OPENRESEARCH_BACKEND_URL`, defaulting to `http://127.0.0.1:8000`.

## Backend

`backend/app.py:create_app` is the FastAPI factory. It registers run, upload,
leaderboard, report, message, and health routes. The app is mostly a stateless
HTTP layer over file-backed run state.

Run creation paths spawn subprocesses via `backend/services/events/live_runs.py`.
Status and liveness are reconciled from `demo_status.json`, process state, and
terminal report files. `backend/services/events/run_liveness.py` contains orphan
sweeps and `periodic_liveness_sweep`; `backend/app.py` wires the periodic sweep
into app lifespan.

## Frontend

`frontend/` is a Next.js 16 app. The lab UI lives under
`frontend/src/app/lab` and `frontend/src/components/lab/rlm`. API proxy routes
live under `frontend/src/app/api/demo`. Read-only run browsing lives at
`/leaderboard` and `/library`.

Live updates use SSE from the backend event stream. The frontend also polls
snapshot endpoints so a refresh or reconnect can recover from a dropped stream.

## RLM / Agent Flow

`backend/agents/rlm/run.py` builds the RLM run context, writes
`run_config.json`, initializes ledgers, and invokes the `rlms` engine. The root
model writes Python against a persistent REPL. The paper text is offloaded as a
REPL variable; the root navigates it through library calls and domain
primitives rather than receiving the full corpus in every prompt.

Core primitives live in `backend/agents/rlm/primitives.py` and are bound through
`backend/agents/rlm/binding.py`. Important primitives include:

- `understand_section`
- `extract_hyperparameters`
- `detect_environment`
- `build_environment`
- `plan_reproduction`
- `implement_baseline`
- `run_experiment`
- `verify_against_rubric`
- `propose_improvements`
- `record_candidate_outcome`
- `check_user_messages`
- `respond_to_user`

RDR mode (`backend/agents/rdr/`) is the deterministic rubric-driven controller.
Hybrid `--mode rlm` runs RDR decomposition first, then RLM repair of weak
clusters.

## Sandboxes

| Sandbox | Status | Behavior |
|---|---|---|
| `local` | Implemented | Host subprocess execution; no local Docker build |
| `docker` | Implemented | Local Docker build and container execution; requires daemon |
| `runpod` | Implemented | Remote GPU pod over SSH; no local Docker build after 2026-06-09 hardening |
| `azure` | Partial | AKS GPU backend and Terraform/Helm exist under `infra/azure`, but this is not documented as production-ready |
| `auto` | Implemented | Chooses available backend; Docker daemon availability matters |

## State and Artifacts

`runs/<project_id>/` is the operational source of truth:

- `demo_status.json`: current UI/status snapshot, written atomically
- `run_config.json`: relaunchable launch-parameter snapshot, secrets excluded
- `dashboard_events.jsonl`: append-only UI/SSE event log
- `rlm_state/`: RLM checkpoints and GPU plan
- `cost_ledger.jsonl`: model and compute spend records
- `experiment_runs.jsonl`: sandbox execution attempts and metrics
- `code/`: generated reproduction code and outputs
- `final_report.json` / `final_report.md`: final benchmark report

SQLite is the event/persistence store. The local default is
`sqlite:///openresearch.db`; compose uses
`sqlite:////app/runs/openresearch.db` so the DB lives on the mounted `runs`
volume.

## Background Model

The backend is not a distributed worker platform. Runs are subprocesses managed
by one backend process. Background capabilities are a mix of:

- app-lifespan liveness sweep (`periodic_liveness_sweep`)
- per-run watchdogs and heartbeat files
- shell/Playwright monitoring loops in `scripts/loops/`
- manual cleanup via `scripts/prune_runs.py`

There is no durable external scheduler, queue, or multi-node supervisor yet.

## Real vs Aspirational

Real and verified locally: backend factory, CLI parsing, full pytest suite,
frontend lint/type/test, Docker build, compose health, and the file-backed run
shape.

Partial or operationally manual: Azure/Kubernetes production readiness, scheduled
artifact generation, run retention policy, network-hermetic test enforcement,
and automated restart policy outside the local monitoring scripts.

