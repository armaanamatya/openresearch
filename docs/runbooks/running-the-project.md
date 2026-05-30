# Running ReproLab — the full workflow, prerequisites, and sandbox matrix

> **Read this first if a run "completes" but produced no real experiment.** The most
> common cause is a missing prerequisite the system does not loudly check — usually
> **the local Docker daemon being down**, which fails `build_environment` even for
> RunPod runs. This doc is the end-to-end "what actually has to be running, and why."

## TL;DR prerequisites by sandbox

| You are running with… | Local **Docker daemon** (OrbStack / Docker Desktop) | RunPod creds | Local NVIDIA GPU |
|---|:---:|:---:|:---:|
| `--sandbox local` | **not needed** (host subprocess) | no | yes (for GPU papers) |
| `--sandbox docker` | **REQUIRED** | no | optional |
| `--sandbox runpod` (the repo default) | **REQUIRED** — for the `build_environment` step (see below) | yes | no (runs on the pod) |
| `--sandbox auto` / unset / anything else | **REQUIRED** — falls back to `LocalDockerBackend` | no | optional |

**The non-obvious one:** `REPROLAB_DEFAULT_SANDBOX=runpod` is the repo default, so most people run on RunPod — but **`build_environment` still does a *local* `docker build`** (only `--sandbox local` skips it). If OrbStack/Docker is down, the run dies at `build_environment` with `SandboxRuntimeError(backend_unavailable)` long before it reaches the GPU pod. Start your Docker engine before launching, even for RunPod.

## What each sandbox actually does

`backend/agents/rlm/primitives.py::_backend_for_sandbox_mode` maps the mode to a backend:

- **`local` → `LocalProcessBackend`.** `build_environment` short-circuits to a no-op (`primitives.py:1113`, returns `skipped: True`); `run_experiment` runs the code as a host subprocess with a per-run venv. No Docker, no RunPod. Needs the paper's deps + (for GPU papers) a local NVIDIA GPU.
- **`docker` → `LocalDockerBackend`.** `build_environment` builds a local image `reprolab/<project>:env-<digest>`; `run_experiment` runs it in a local container (network/memory/CPU bounded). Needs the Docker daemon up.
- **`runpod` → `RunpodBackend`.** `build_environment` **still builds a local image** (the short-circuit is `local`-only), but `run_experiment` boots a remote GPU pod (image = `REPROLAB_RUNPOD_IMAGE`, default `runpod/pytorch:…runtime`) and runs the code over SSH in a per-run venv. **The locally-built image is not used on the pod** — so the local build is, today, a Dockerfile-validation step that nonetheless requires a live Docker daemon. Needs Docker up **and** `REPROLAB_RUNPOD_API_KEY` + `REPROLAB_RUNPOD_SSH_KEY_PATH`.
- **`auto` / unknown / `None` → `LocalDockerBackend`** (with a WARNING for unknown modes). Needs Docker up.

> **Known rough edge (flagged 2026-05-30, not yet changed):** `build_environment` doing a
> local `docker build` under `--sandbox runpod` is wasted work — the pod runs its own image.
> A future change could short-circuit `build_environment` under `runpod` the way it already
> does under `local`. Until then: keep Docker up for RunPod runs, or use `--sandbox local`.

## End-to-end run workflow (and where each step can fail on a prerequisite)

1. **Prereqs up** — Docker daemon (unless `--sandbox local`), backend `:8000`, frontend `:3000`, `.env` with an LLM key (and RunPod creds for `--sandbox runpod`).
2. **Ingest** — `ResolvingParser` (HTML > PDF > OCR). Fails on: unreachable arXiv / unparseable PDF.
3. **Understand** — `understand_section`, `extract_hyperparameters`. LLM sub-calls — fails on: bad/empty root-model credentials.
4. **Environment** — `detect_environment` → `build_environment`. **`build_environment` needs the local Docker daemon for every sandbox except `local`.** Fails on: Docker down (`backend_unavailable`), malformed Dockerfile (`dockerfile_invalid`, BUG-NEW-042).
5. **Plan & Implement** — `plan_reproduction` → `implement_baseline` (Claude Sonnet via `claude-agent-sdk`). Fails on: sub-agent returning an empty completion (`SDK success-with-no-text`, the FM-001 wedge) — an **auth/SDK** problem, not Docker.
6. **Execute** — `run_experiment` on the chosen backend. Fails on: Docker down (docker/auto), RunPod auth/quota/OOM (runpod), missing deps/GPU (local).
7. **Score** — `verify_against_rubric`. 8. **Improve/iterate** — `propose_improvements`. 9. **Report** — `final_report.{json,md}` (subject to the evidence gate — a run with no successful `run_experiment` is downgraded to `failed`, see CLAUDE.md "Run-status enum / evidence gate").

**Diagnosing a hollow `partial` / `suspicious_partial`:** open the run in `/lab`, read the detail panel's blockers. `SDK success-with-no-text` = step 5 (auth/SDK), `backend_unavailable` / docker errors = step 4 or 6 (Docker), RunPod errors = step 6 (pod/creds).

## Startup sequence

```bash
# 0. Start your Docker engine (OrbStack or Docker Desktop) — even for RunPod.
#    Verify: `docker info` must succeed.
docker info >/dev/null 2>&1 && echo "docker up" || echo "START ORBSTACK/DOCKER FIRST"

# 1. Backend (factory pattern; --factory is required)
.venv/bin/uvicorn backend.app:create_app --factory --reload --port 8000
#    …or the preflight-aware launcher (runs RunPod checks when sandbox=runpod):
./start.sh

# 2. Frontend
cd frontend
export REPROLAB_BACKEND_URL=http://127.0.0.1:8000
npm run dev   # http://localhost:3000
```

`start.sh` runs `scripts/runpod_check.sh` when the sandbox is `runpod` (RunPod API auth + SSH key). As of 2026-05-30 it **also** checks that the local Docker daemon is up whenever the sandbox is not `local` (the previous preflight only checked RunPod and silently let a Docker-down run fail later). Bypass everything with `START_SKIP_PREFLIGHT=1 ./start.sh`.

## Cheapest local-dev configuration

- Root model: `--model gpt-5` (~$1/run via `OPENAI_API_KEY`) or `--model claude-oauth` ($0 on the Claude CLI subscription).
- Sub-agents: Claude OAuth ($0) — leave `ANTHROPIC_API_KEY` empty and `claude login` once.
- Sandbox: `--sandbox runpod` COMMUNITY (~$0.34/hr) **with Docker up**, or `--sandbox local` (no Docker, no RunPod — needs local deps/GPU), or `--sandbox docker` for CPU-only papers.

## Troubleshooting quick table

| Symptom | Likely step | Fix |
|---|---|---|
| `SandboxRuntimeError(backend_unavailable)` | build_environment / run_experiment | Start OrbStack/Docker; verify `docker info`. |
| Run "completes" as hollow `partial`, no metrics | evidence gate caught no real experiment | Check the lab blockers; usually Docker down or the SDK wedge below. |
| `[CRITICAL] SDK success-with-no-text (claude_agent_sdk)` | implement_baseline | Root/sub-agent auth: check `ANTHROPIC_API_KEY` is empty+`claude login`, or funded. Not a Docker issue. |
| RunPod auth / `ensure_runpod_available` error | run_experiment (runpod) | `REPROLAB_RUNPOD_API_KEY` + SSH key; run `scripts/runpod_check.sh`. |
| `401 invalid_api_key` at iter 0 | root model | Stale shell `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` shadows `.env`; prefix `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY`. |

See also: `system_overview.md` (architecture), `CLAUDE.md` (day-to-day + gotchas), `docs/runbooks/e2e-testing.md` (local E2E).
