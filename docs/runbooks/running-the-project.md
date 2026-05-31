# Running ReproLab — the full workflow, prerequisites, and sandbox matrix

> **Read this first if a run "completes" but produced no real experiment.** A common
> cause is a missing prerequisite the system does not loudly check — e.g. **the local
> Docker daemon being down** under `--sandbox docker`/`auto` (since `875995c`, `runpod`
> no longer needs a local daemon — `build_environment` short-circuits). This doc is the
> end-to-end "what actually has to be running, and why."
>
> **Just want to launch a run?** → [starting-a-run.md](starting-a-run.md) is the copy-paste
> recipe (readiness check → launch command → verify the workflow ran). This doc is the reference.

## TL;DR prerequisites by sandbox

| You are running with… | Local **Docker daemon** (OrbStack / Docker Desktop) | RunPod creds | Local NVIDIA GPU |
|---|:---:|:---:|:---:|
| `--sandbox local` | **not needed** (host subprocess) | no | yes (for GPU papers) |
| `--sandbox docker` | **REQUIRED** | no | optional |
| `--sandbox runpod` (the repo default) | **not needed** — `build_environment` short-circuits (`875995c`) | yes | no (runs on the pod) |
| `--sandbox auto` / unset / anything else | **REQUIRED** — falls back to `LocalDockerBackend` | no | optional |

**The non-obvious one:** `REPROLAB_DEFAULT_SANDBOX=runpod` is the repo default, so most people run on RunPod — and since `875995c` **`build_environment` short-circuits under `runpod`** (no local `docker build`), so a down Docker daemon no longer breaks RunPod runs. Docker is still required for `--sandbox docker` and `auto`/unknown (both use `LocalDockerBackend`).

## What each sandbox actually does

`backend/agents/rlm/primitives.py::_backend_for_sandbox_mode` maps the mode to a backend:

- **`local` → `LocalProcessBackend`.** `build_environment` short-circuits to a no-op (`primitives.py:1113`, returns `skipped: True`); `run_experiment` runs the code as a host subprocess with a per-run venv. No Docker, no RunPod. Needs the paper's deps + (for GPU papers) a local NVIDIA GPU.
- **`docker` → `LocalDockerBackend`.** `build_environment` builds a local image `reprolab/<project>:env-<digest>`; `run_experiment` runs it in a local container (network/memory/CPU bounded). Needs the Docker daemon up.
- **`runpod` → `RunpodBackend`.** Since `875995c`, `build_environment` **short-circuits to a no-op** (`primitives.py:1163`) — no local `docker build`. `run_experiment` boots a remote GPU pod (image = `REPROLAB_RUNPOD_IMAGE`, default `runpod/pytorch:…-devel-…`) and runs the code over SSH in a per-run venv. **No local Docker daemon required.** Needs `REPROLAB_RUNPOD_API_KEY` + `REPROLAB_RUNPOD_SSH_KEY_PATH`.
- **`auto` / unknown / `None` → `LocalDockerBackend`** (with a WARNING for unknown modes). Needs Docker up.

> **Resolved (`875995c`, was a rough edge flagged 2026-05-30):** `build_environment` used to do a
> wasted local `docker build` under `--sandbox runpod` (the pod runs its own image). It now
> short-circuits under `runpod` the same way it does under `local`, so RunPod runs no longer
> require a local Docker daemon. `82e9806` additionally normalizes a hallucinated `runpod/` FROM
> line for the (now docker/auto-only) build path.

## End-to-end run workflow (and where each step can fail on a prerequisite)

1. **Prereqs up** — Docker daemon (only for `--sandbox docker`/`auto`; not for `local` or `runpod`), backend `:8000`, frontend `:3000`, `.env` with an LLM key (and RunPod creds for `--sandbox runpod`).
2. **Ingest** — `ResolvingParser` (HTML > PDF > OCR). Fails on: unreachable arXiv / unparseable PDF.
3. **Understand** — `understand_section`, `extract_hyperparameters`. LLM sub-calls — fails on: bad/empty root-model credentials.
4. **Environment** — `detect_environment` → `build_environment`. **`build_environment` needs the local Docker daemon only for `--sandbox docker` and `auto`/unknown** (`local` and `runpod` short-circuit to a no-op). Fails on: Docker down (`backend_unavailable`, docker/auto only), malformed Dockerfile (`dockerfile_invalid`, BUG-NEW-042).
5. **Plan & Implement** — `plan_reproduction` → `implement_baseline` (Claude Sonnet via `claude-agent-sdk`). Fails on: sub-agent returning an empty completion (`SDK success-with-no-text`, the FM-001 wedge) — an **auth/SDK** problem, not Docker.
6. **Execute** — `run_experiment` on the chosen backend. Fails on: Docker down (docker/auto), RunPod auth/quota/OOM (runpod), missing deps/GPU (local).
7. **Score** — `verify_against_rubric`. 8. **Improve/iterate** — `propose_improvements`. 9. **Report** — `final_report.{json,md}` (subject to the evidence gate — a run with no successful `run_experiment` is downgraded to `failed`, see CLAUDE.md "Run-status enum / evidence gate").

**Diagnosing a hollow `partial` / `suspicious_partial`:** open the run in `/lab`, read the detail panel's blockers. `SDK success-with-no-text` = step 5 (auth/SDK), `backend_unavailable` / docker errors = step 4 or 6 (Docker), RunPod errors = step 6 (pod/creds).

## Startup sequence

```bash
# 0. Start your Docker engine (OrbStack or Docker Desktop) — only for --sandbox docker/auto.
#    (--sandbox local and runpod do NOT need a local daemon since 875995c.)
#    Verify: `docker info` must succeed.
docker info >/dev/null 2>&1 && echo "docker up" || echo "START ORBSTACK/DOCKER FIRST (docker/auto only)"

# 1. Backend (factory pattern; --factory is required)
.venv/bin/uvicorn backend.app:create_app --factory --reload --port 8000
#    …or the preflight-aware launcher (runs RunPod checks when sandbox=runpod):
./start.sh

# 2. Frontend
cd frontend
export REPROLAB_BACKEND_URL=http://127.0.0.1:8000
npm run dev   # http://localhost:3000
```

`start.sh` runs `scripts/runpod_check.sh` when the sandbox is `runpod` (RunPod API auth + SSH key). It also runs a `docker info` preflight — but, since the `875995c` runpod short-circuit, only when the default sandbox is `docker`/`auto` (i.e. not `local` and not `runpod`), because those are the only modes that still do a local `docker build`. Bypass everything with `START_SKIP_PREFLIGHT=1 ./start.sh`.

## Cheapest local-dev configuration

- Root model: `--model gpt-5` (~$1/run via `OPENAI_API_KEY`) or `--model claude-oauth` ($0 on the Claude CLI subscription).
- Sub-agents: Claude OAuth ($0) — leave `ANTHROPIC_API_KEY` empty and `claude login` once.
- Sandbox: `--sandbox runpod` COMMUNITY (~$0.34/hr — no local Docker needed since `875995c`), or `--sandbox local` (no Docker, no RunPod — needs local deps/GPU), or `--sandbox docker` for CPU-only papers (needs Docker up).

## Troubleshooting quick table

| Symptom | Likely step | Fix |
|---|---|---|
| `SandboxRuntimeError(backend_unavailable)` | build_environment / run_experiment | Start OrbStack/Docker; verify `docker info`. |
| Run "completes" as hollow `partial`, no metrics | evidence gate caught no real experiment | Check the lab blockers; usually Docker down or the SDK wedge below. |
| `[CRITICAL] SDK success-with-no-text (claude_agent_sdk)` | implement_baseline | Root/sub-agent auth: check `ANTHROPIC_API_KEY` is empty+`claude login`, or funded. Not a Docker issue. |
| RunPod auth / `ensure_runpod_available` error | run_experiment (runpod) | `REPROLAB_RUNPOD_API_KEY` + SSH key; run `scripts/runpod_check.sh`. |
| `401 invalid_api_key` at iter 0 | root model | Stale shell `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` shadows `.env`; prefix `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY`. |

See also: `system_overview.md` (architecture), `CLAUDE.md` (day-to-day + gotchas), `docs/runbooks/e2e-testing.md` (local E2E).
