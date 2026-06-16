<!-- doc-meta: status=current; last-verified=2026-06-09 -->
# Infrastructure

This document describes what the repository actually ships today. It is not a
production-readiness claim.

## Dockerfile

The root `Dockerfile` is a three-stage build:

1. `python-deps`: `python:3.12-slim`, creates `/opt/venv`, installs
   `backend/requirements.txt`.
2. `frontend`: `node:20-bookworm-slim`, runs `npm ci` and `npm run build`.
3. `runtime`: `python:3.12-slim`, copies the Python venv, copies Node/npm from
   the frontend stage, copies backend and frontend build output, then boots
   `docker/entrypoint.sh` under `tini`.

The image runs both FastAPI and Next.js as the non-root `app` user
(uid 10001, since 2026-06-10) and installs `docker.io` so local Docker
sandbox operations can talk to a mounted host socket (granted via compose
`group_add`). Socket access is still effective host-root by design; the
processes themselves are unprivileged.

## Entrypoint

`docker/entrypoint.sh`:

- parses `/app/.env` as key/value data instead of sourcing shell code
- keeps compose or `docker run -e` environment variables ahead of `.env`
- starts uvicorn on internal `:8000`
- starts Next on `$PORT` or `3000`
- forwards TERM/INT to both child processes
- tears down the surviving child when either child exits

## docker compose

`docker-compose.yml` is a local-dev and smoke-run configuration:

- publishes the frontend on `0.0.0.0:3000`
- publishes the backend only on `127.0.0.1:8000`
- mounts `/var/run/docker.sock`
- mounts `./runs:/app/runs`
- mounts `./third_party:/app/third_party:ro`
- mounts `./.env:/app/.env:ro`
- sets `OPENRESEARCH_DATABASE_URL=sqlite:////app/runs/openresearch.db`

The Docker socket mount gives the container effective control over the host
Docker daemon. Do not expose this compose stack on an untrusted network.

## Persistence

Run artifacts and the compose SQLite DB live under `runs/`. The path is
mostly gitignored via a whitelist idiom in `.gitignore`: everything under
`/runs/**` is ignored, but the small high-value per-run diagnostics are
deliberately re-included and tracked in git (`final_report.json`,
`final_report.md`, `demo_status.json`, `batch_child.log`,
`experiment_runs.jsonl`, `cost_ledger.jsonl`, `tokens_total.json`). Heavy
artifacts (per-run `.venv/`, `code/`, `rlm_state/` checkpoints) and the
SQLite DB itself stay ignored. Preserve `runs/` when debugging or comparing
runs; prune it intentionally with `scripts/prune_runs.py`.

SQLite URL gotcha:

- local relative DB: `sqlite:///openresearch.db`
- container absolute DB: `sqlite:////app/runs/openresearch.db`

Three slashes in the container path are wrong because they produce a path
relative to `/app`.

## Ports

| Surface | Default |
|---|---|
| Frontend dev | `localhost:3000` |
| Backend dev | `localhost:8000` |
| Compose frontend | host `3000` to container `3000` |
| Compose backend | host `127.0.0.1:8000` to container `8000` |

The backend is loopback-only in compose because direct backend access can bypass
the frontend unlock wall when `OPENRESEARCH_DEMO_SECRET` is empty.

## RunPod

RunPod is the intended remote GPU path. It needs:

- `OPENRESEARCH_RUNPOD_API_KEY`
- `OPENRESEARCH_RUNPOD_SSH_KEY_PATH`
- an image such as the default
  `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`

The default image is the CUDA `devel` variant because common ML dependencies
compile against CUDA headers. A lighter `runtime` image can work for papers that
do not JIT-compile CUDA packages, but it is not the safe default.

After the 2026-06-09 hardening, `build_environment` short-circuits for RunPod:
the local machine does not need to build a Docker image for `--sandbox runpod`.

## Kubernetes / Azure

Azure AKS GPU infrastructure exists under `infra/azure/` with Terraform and Helm
material. The backend also has an `azure` sandbox path and Azure dependencies.

Current status: partial. It is not the canonical local reproduction path and is
not documented here as production-ready. Before using it for production, verify
Terraform state, secrets handling, image publishing, blob persistence,
per-job cleanup, cost limits, and rollback from a fresh cloud account.

## Not Supported Yet

- ~~non-root hardened runtime image~~ — done 2026-06-10: both servers run as
  uid 10001 (`app`); the docker socket is granted via compose `group_add`
  (`OPENRESEARCH_DOCKER_GID`, default 0 for macOS Docker Desktop). Socket
  group membership remains root-equivalent on the host by design
  (LocalDockerBackend needs it); the hardening is that the PROCESSES are
  no longer uid 0.
- ~~digest-pinned base images~~ — done 2026-06-10 (python:3.12-slim and
  node:20-bookworm-slim pinned by sha256; bump deliberately)
- production-grade queue/scheduler
- multi-node run ownership
- ~~automatic run retention policy~~ — done 2026-06-10, opt-in:
  `OPENRESEARCH_RUNS_RETENTION_DAYS=N` prunes terminal runs older than N days
  hourly (honors `.preserved`; unset/0 = off; manual path stays
  `scripts/prune_runs.py`)
- compose resource limits for every sandbox child
- ~~frontend healthcheck in compose~~ — done 2026-06-10 (the compose
  healthcheck now curls BOTH :8000/health and :3000)

