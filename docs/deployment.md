# Production Deployment

## Overview

A full ReproLab deployment consists of four components:

1. **Backend service** — FastAPI (uvicorn), stateless request handler + pipeline subprocess spawner
2. **Frontend app** — Next.js, served via `next start`
3. **Database** — SQLite (default) or any SQLAlchemy-compatible store; see database note below
4. **Execution sandbox** — Docker (local daemon) or RunPod GPU pods (remote, no Docker required on the host)

The simplest production path is the included multi-stage `Dockerfile` + Docker Compose, which packs both services into a single image. For separated deployments, each component can be run independently.

---

## Backend

### Container

A production-ready three-stage `Dockerfile` is at the repo root. It produces a slim `python:3.12-slim` image with the pre-built Python venv, Node 20, and the compiled Next.js frontend bundled together.

```bash
docker build -t openresearch/app:latest .
```

If deploying the backend separately (without the bundled frontend), the same image works — just don't start the `next start` process.

### Required environment variables / secrets

All settings use the `REPROLAB_` prefix (see `backend/config.py` for defaults and aliases). The unprefixed forms `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `RUNPOD_API_KEY`, and `APIFY_API_TOKEN` are also accepted.

| Variable | Required | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes (if using Anthropic) | Claude provider key |
| `OPENAI_API_KEY` | Yes (if using OpenAI) | OpenAI provider key |
| `OPENAI_ADMIN_KEY` | No | Admin-scoped OpenAI credentials |
| `REPROLAB_DATABASE_URL` | No | Defaults to `sqlite:///reprolab.db`; set to an absolute path in production, e.g. `sqlite:////app/runs/reprolab.db` |
| `REPROLAB_DEFAULT_SANDBOX` | No | `auto`, `local`, `docker`, or `runpod`; defaults to `runpod` |
| `REPROLAB_LLM_PROVIDER` | No | `anthropic` (default) or `openai` |
| `RUNPOD_API_KEY` | Yes (if sandbox=runpod) | RunPod REST API key |
| `REPROLAB_RUNPOD_SSH_KEY_PATH` | Yes (if sandbox=runpod) | Path to SSH private key for pod access |
| `REPROLAB_RUNPOD_SSH_PUBLIC_KEY` | Auto-derived | Derived from `SSH_KEY_PATH` at startup if not set |
| `REPROLAB_RUNPOD_GPU_TYPE` | No | Defaults to `NVIDIA GeForce RTX 4090` |
| `REPROLAB_RUNPOD_IMAGE` | No | Defaults to `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04` |
| `REPROLAB_RUNPOD_NETWORK_VOLUME_ID` | No | Persistent RunPod volume ID for cross-run caching |
| `APIFY_API_TOKEN` | No | Enables arXiv MCP server for artifact discovery agents |
| `REPROLAB_PYTHON_BIN` | No | Override interpreter path; Docker image sets this to `/opt/venv/bin/python` |

### Run command

```bash
# Production (no --reload)
/opt/venv/bin/python -m uvicorn backend.app:create_app \
    --factory \
    --host 0.0.0.0 \
    --port 8000 \
    --proxy-headers \
    --forwarded-allow-ips "*"
```

The Docker entrypoint (`docker/entrypoint.sh`) runs both backend and frontend under `tini` with signal forwarding.

---

## Frontend

### Build

```bash
cd frontend
npm ci
npm run build       # emits to frontend/.next
```

Node version constraint: **>=20.19.0 and <21, OR >=22.12.0**. Node 21 is excluded by the `engines` field in `frontend/package.json`.

### Serve

```bash
# Start the Next.js production server
npx next start --hostname 0.0.0.0 --port 3000
```

### Environment

```bash
REPROLAB_BACKEND_URL=https://your-backend-host:8000
```

The frontend API routes (`frontend/src/app/api/demo/`) proxy all pipeline requests to `REPROLAB_BACKEND_URL`. If unset, it defaults to `http://127.0.0.1:8000` — correct for local dev but must be set explicitly in production.

---

## Database

ReproLab currently uses **SQLite** as its event store and persistence layer. There is no Postgres dependency. The database file location is controlled by `REPROLAB_DATABASE_URL` (default: `sqlite:///reprolab.db`).

> **IMPORTANT — Schema consolidation required before production**
>
> The SQLite schema has accumulated tables across several development phases. Multiple schema definitions exist across `backend/persistence/database.py`, `backend/eventstore/sqlite_store.py`, `backend/evals/store.py`, `backend/messaging/idempotency.py`, `backend/services/diagnostics/service.py`, `backend/services/orchestration/`, `backend/services/approval/service.py`, and `backend/services/datasets/service.py`. Several of these tables are unused in current pipeline paths. Additionally, there are storage-handling issues (the `reprolab.db.corrupt-*` and `reprolab.db.offline_backup` files in the repo root indicate past write failures).
>
> **Schema consolidation and storage hardening are prerequisites for a production deployment.** Specifically:
> - Audit all `CREATE TABLE` sites and remove unused tables
> - Consolidate schema initialization into a single migration entrypoint
> - Ensure the database file resides on a persistent volume (not the container's ephemeral FS)
> - Validate WAL mode and `PRAGMA foreign_keys` settings under concurrent pipeline load

In Docker Compose, the database is stored inside the `runs/` volume:

```yaml
environment:
  - REPROLAB_DATABASE_URL=sqlite:///app/runs/reprolab.db
```

---

## Execution Sandbox

### Docker (local)

The backend spawns `docker run` via the Python `docker` SDK. The host daemon socket must be accessible:

```bash
docker run -v /var/run/docker.sock:/var/run/docker.sock ...
```

**Security note** (from the Dockerfile): mounting the host docker socket gives the container effective root on the host daemon. Acceptable for local dev; use RunPod for production GPU workloads.

Network policy: sandbox containers are launched with network disabled by default (`--sandbox docker`). Pass `--allow-sandbox-network` in the CLI or set `gpuMode` in the API to override.

### RunPod (remote GPU)

The `RunPodBackend` in `backend/services/runtime/runpod_backend.py` manages pod lifecycle (create, SSH, destroy). Required credentials:

- `RUNPOD_API_KEY` — REST API key from the RunPod console
- `REPROLAB_RUNPOD_SSH_KEY_PATH` — path to an SSH private key registered with your RunPod account
- Optional: `REPROLAB_RUNPOD_NETWORK_VOLUME_ID` — attach a persistent network volume to avoid re-downloading datasets per run

Pod lifecycle: by default (`REPROLAB_RUNPOD_DELETE_ON_DESTROY=true`), a new pod is created for each pipeline run and destroyed on completion. Set `REPROLAB_RUNPOD_POD_ID` to reuse an existing pod across runs (the backend will never delete pods not in its `_owned_pod_ids` allowlist).

---

## Persistence

The `runs/` directory holds all per-run artifacts and must be on a **persistent volume**:

```
runs/
  <project_id>/
    pipeline_state.json     — full pipeline state (checkpoint per stage)
    final_report.json       — computed benchmark report
    final_report.md         — human-readable report
    assumption_ledger.json
    agent_telemetry.jsonl
    cost_ledger.jsonl
    Dockerfile
    code/                   — reproduced baseline code
    hermes/                 — audit chain checkpoints
    raw_paper.pdf
  .lab_uploads/             — temporary uploaded PDFs (can be purged)
  .hermes_adapter_memory.json
```

In Docker Compose this is already wired:

```yaml
volumes:
  - ./runs:/app/runs
```

For Kubernetes or cloud deployments, mount a persistent volume at `/app/runs`.

---

## Scaling and Operations

**Subprocess model**: each pipeline run is a long-lived subprocess spawned by the FastAPI backend via `asyncio.create_subprocess_exec`. The backend tracks the PID in `demo_status.json` and polls liveness. This model works well for single-server deployments but has limits:

- Multiple concurrent runs are unbounded — the host must have sufficient memory/CPU/disk
- Runs do not survive backend process restarts (the PID becomes orphaned; the frontend reconciles to `failed`)
- For higher concurrency, a dedicated worker queue (Celery, ARQ, or similar) in front of the pipeline runner is recommended

**Logs**: each run writes `runner.stdout.log` and `runner.stderr.log` to its project directory. The `agent_telemetry.jsonl` file records per-invocation timing and token usage. `cost_ledger.jsonl` records USD spend per agent turn.

**Telemetry**: no external telemetry is sent. All observability data stays local in `runs/`.

---

## Railway deployment

Railway runs the bundled single-image build (backend + frontend in one container).
`railway.json` at the repo root pins the build to the `Dockerfile` and healthchecks
`/health`.

### One-time setup

1. **Volume** — create a Railway volume mounted at `/app/runs`. It holds both the
   SQLite DB and all run artifacts. Railway allows one volume per service and blocks
   two deployments mounting it at once, so redeploys cause a brief outage and the
   service must stay at `numReplicas: 1` (already set in `railway.json`).
2. **Target port** — set the service variable `PORT=3000` and the public networking
   target port to `3000`. The entrypoint binds the frontend to `$PORT`; the backend
   stays internal on `8000`.

### Environment variables

In addition to the variables in the *Required environment variables* section above:

| Variable | Value | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | your key | LLM provider — runs default to real SDK mode |
| `REPROLAB_DATABASE_URL` | `sqlite:////app/runs/reprolab.db` | DB on the persistent volume (four slashes — absolute path) |
| `REPROLAB_DEMO_SECRET` | a long random string | the access gate — see below |
| `REPROLAB_BACKEND_URL` | `http://127.0.0.1:8000` | frontend → backend, same container |
| `PORT` | `3000` | public frontend port |
| `REPROLAB_FORCE_SANDBOX` | *unset* (RunPod) or `local` | compute mode — see below |

### Compute: RunPod (preferred) vs local

The lab UI requests `sandbox=runpod` for every run. `REPROLAB_FORCE_SANDBOX` is the
deployment-level override that pins the sandbox regardless of what the client asks for:

- **RunPod-backed (preferred)** — leave `REPROLAB_FORCE_SANDBOX` unset and provide the
  RunPod credentials from the *RunPod (remote GPU)* section above. Real GPU compute;
  handles heavy papers.
- **`local` (no GPU, $0 compute)** — set `REPROLAB_FORCE_SANDBOX=local`. The
  reproduction pipeline then runs inside the Railway container itself — CPU-only,
  memory/disk-limited. Fine for the built-in fixture and lightweight papers; a heavy
  uploaded paper can OOM or run long.

`REPROLAB_DEFAULT_SANDBOX` does **not** help here — it is only a fallback for requests
that omit the sandbox, and the UI never does. `REPROLAB_FORCE_SANDBOX` is the real lever.

### The access gate — `REPROLAB_DEMO_SECRET`

When `REPROLAB_DEMO_SECRET` is set, the deployment is gated end to end:

- **Front door** — a Next.js proxy (`frontend/src/proxy.ts`) requires a valid session
  for every page and API route. Visitors land on an unlock screen (`/unlock`), enter
  the secret, and `POST /api/unlock` validates it (constant-time) and sets an HttpOnly,
  12-hour cookie holding `sha256(secret)` — never the raw value.
- **Back door** — `POST /runs` and `/runs/upload` independently require a matching
  `X-Demo-Secret` header (`backend/app.py: _enforce_demo_gate`); the `/api/demo` proxy
  attaches it for authenticated sessions. Defense in depth: the backend can't be driven
  even if reached directly.

When `REPROLAB_DEMO_SECRET` is **unset**, both gates are disabled — the local-dev
default, unchanged.

**Using it:**

```bash
openssl rand -hex 24   # generate a high-entropy secret, then set it in Railway
```

The variable is shared by both processes in the container. Share the Railway URL and
the secret with the demo owner (ideally via separate channels); they visit the URL →
unlock screen → paste the secret → 12-hour session. Rotate by changing the variable
(invalidates all existing sessions). Use a long, random value — `/api/unlock` adds a
fixed failure delay, but entropy is the real defense.

### Verify

- `GET /health` → `200` (the frontend proxies the backend probe; `503` if the backend is down)
- `GET /lab` with no session → redirect to `/unlock`
- `POST /api/unlock` with the wrong code → `401`; correct → `200` + `Set-Cookie`
- a run started from the UI reaches `POST /runs` with `X-Demo-Secret` and executes in the configured sandbox

### Limitations

- SQLite on a single volume → `numReplicas: 1`, brief redeploy downtime. Postgres is
  the scaling path (see the *Database* section).
- The unlock cookie is a static `sha256(secret)` token (no rotation/expiry signing) —
  adequate for an owner-only demo, not a general auth system.
- `local` sandbox capacity is bounded by the Railway container; see *Compute* above.

---

## Pre-Production Checklist

- [ ] **Schema consolidation** — audit and remove unused SQLite tables; consolidate migration init; harden WAL behavior (see Database section above)
- [ ] **Node version pinning** — pin Node to `20.x` or `22.x` in CI and deployment images; Node 21 is explicitly excluded by `package.json` `engines`
- [ ] **Persistent volume** — mount `runs/` on a persistent volume; do NOT rely on container ephemeral storage
- [ ] **Secrets management** — store API keys in a secrets manager (Vault, AWS Secrets Manager, Doppler, etc.) rather than in `.env` files on disk; the `.env` mount in Docker Compose is development-only
- [ ] **`REPROLAB_BACKEND_URL`** — set explicitly in the frontend deployment environment; the `http://127.0.0.1:8000` fallback is wrong for separated deployments
- [ ] **RunPod SSH key** — use a dedicated deployment SSH key pair; do not share with developer keys
- [ ] **Docker socket access** — for production GPU workloads, use `--sandbox runpod` and do not mount the host Docker socket
- [ ] **Backend `--reload` flag** — omit in production (the provided `docker/entrypoint.sh` and run command above already omit it)
- [ ] **Concurrent run limits** — if running multiple pipelines simultaneously, evaluate a worker queue approach and set `--max-usd` / `--max-wall-clock` budget guards on each run
- [ ] **Log rotation** — `runs/` grows unboundedly; implement a retention policy for old run directories
