# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **⚠ Architecture pivot in progress (2026-05).** This repo is being re-architected
> from the 14-stage `PipelineStage` state machine to an RLM-based orchestrator built
> on the `rlms` library (Recursive Language Models, arXiv 2512.24601). The canonical
> plan is **`docs/design/rlm-pivot-brief.md`** — read it first. The architecture
> described below is the *current, pre-pivot* code and is being replaced; where this
> file and the brief conflict, the brief is the direction and the code below is the
> present.

## Project: OpenResearch / ReproLab

An agent pipeline that reproduces research papers end-to-end: ingest paper → understand claims → build environment → implement + run a baseline → gate the result → explore improvements → emit a benchmark report comparing the reproduction to the paper's claims. See `system_overview.md` and `docs/design/rlm-pivot-brief.md` for the full "why" — read those before making non-trivial architectural changes.

## Common commands

### Backend (Python 3.14.2, FastAPI)

```bash
# Install
pip install -r backend/requirements.txt
pip install -r backend/requirements-dev.txt   # adds pytest + parallel runners

# Run the API (factory pattern — --factory is required)
.venv/bin/uvicorn backend.app:create_app --factory --reload --port 8000
# Or via the preflight-aware launcher (runs RunPod checks when sandbox=runpod):
./start.sh

# Tests
.venv/bin/python -m pytest tests/                       # all
.venv/bin/python -m pytest tests/ -n auto               # parallel (needs requirements-dev)
.venv/bin/python -m pytest tests/path/to/test_x.py::test_name   # single
.venv/bin/python -m pytest tests/ --reruns 2            # rerun flaky network tests
```

Pytest config lives in `pyproject.toml` (`testpaths = ["tests"]`, `pythonpath = ["src"]`). There is no separate lint/format step configured at the repo level.

### Frontend (Next.js 16, Node ≥20.19 <21 or ≥22.12)

```bash
cd frontend
npm ci
export REPROLAB_BACKEND_URL=http://127.0.0.1:8000   # required: server-side proxy target
npm run dev          # http://localhost:3000
npm run build        # production build
npm run lint         # eslint .
npm test             # vitest run
npx tsc --noEmit     # type-check only
```

E2E tests use Playwright (`frontend/e2e/`); run via `npx playwright test` from `frontend/`.

### CLI (non-UI runs)

```bash
python -m backend.cli reproduce paper.pdf --provider anthropic --sandbox docker
python -m backend.cli reproduce 2512.24601 --mode offline       # deterministic, no LLM
python -m backend.cli ingest paper.pdf                          # ingest only
python -m backend.cli eval <project_id> --paper-metrics '{...}' # score completed run
```

Useful flags: `--mode {offline,sdk,rlm,rdr}`, `--provider {anthropic,openai}`, `--verification-provider`, `--sandbox {auto,local,docker,runpod}`, `--execution-mode {efficient,max}`, `--n-paths N`, `--max-usd`, `--max-wall-clock`, `--model`, `--seed`. `--mode offline` is the right choice for fast deterministic testing without LLM cost. `--mode rlm` is the production-hardened RLM path (Phase 5): per-primitive deadlines, `max_usd` cost cap, corpus-leak redaction at every egress, atomic run-status writes. `--mode rdr` is the rubric-driven harness — a deterministic Python controller dispatches scoped Claude coding agents per rubric work-cluster on a PaperBench bundle (the positional arg is the bundle paper_id); see `docs/superpowers/specs/2026-05-22-rubric-driven-harness-design.md`. Set `REPROLAB_RLM_ROOT_MODEL` to `gpt-5`, `qwen3-coder`, `kimi-k2.5`, or `claude` (defaults to GPT-5 when `OPENAI_API_KEY` is set).

### Docker

```bash
cp .env.example .env   # set ANTHROPIC_API_KEY and/or OPENAI_API_KEY
docker compose up --build
```

## Architecture — the non-obvious parts

### One image, two processes
`docker/entrypoint.sh` (under `tini`) runs the FastAPI backend on internal `:8000` and the Next.js frontend on public `:$PORT`. The frontend reaches the backend **server-side only** through `/api/demo/*` proxy routes — there is no CORS layer because the browser never talks to the backend directly. When debugging UI-vs-API issues, check the Next.js proxy route under `frontend/src/app/api/demo/`, not CORS.

### File-backed run state, not a service
Each pipeline run is a **long-lived subprocess** spawned by the backend. Run state lives in `runs/<project_id>/`:
- `demo_status.json` — UI-facing status snapshot
- `pipeline_state.json` — checkpointed after every stage; resume-safe
- `final_report.{json,md}` — the computed benchmark output
- `*.jsonl` — agent event logs (SSE source)
- `code/` — the reproduced project
- Hermes audit chain artifacts

SQLite (`REPROLAB_DATABASE_URL`, defaults to `sqlite:///reprolab.db`) is the event/persistence store with CQRS projections. Pipeline state is **persisted atomically** at each stage transition.

### The 14-stage pipeline state machine
`PipelineStage` in `backend/agents/orchestrator.py` is the **source of truth** for stage order. Each stage has a corresponding agent under `backend/agents/`. Order: ingest → paper-understanding → artifact-discovery → environment-detective → reproduction-planner → Gate 1 → baseline-implementation → experiment-runner → Gate 2 → improvement-selection → improvement-paths (parallel) → Gate 3 → research-map → complete.

Two important behaviors **happen inside existing stages** rather than as new enum values — keep the stage count at 14:

- **Rubric verification + self-improvement** (`rubric_verifier_enabled`): a `rubric-verifier` agent scores against a PaperBench-style weighted rubric at Gate 2 (`baseline_verification`) and Gate 3 (`improved_verification`). The canonical rubric is resolved once per run (vendored bundle's `rubric.json` or LLM-generated) and stored on `PipelineState.rubric_spec` so the two checkpoints stay comparable. Below `rubric_target_score`, the orchestrator loops improvement-selection + Gate 3, capped by `rubric_max_improvement_iterations`. Fail-closed: a verifier error degrades to the heuristic rubric.
- **Environment build-and-repair** (`environment_build_validation_enabled`, docker-sandbox only): the reproduction Dockerfile is built at `ENVIRONMENT_BUILT` via `build_image()`; failures feed back to `environment-detective` in repair mode, capped by `environment_build_max_attempts`. Fail-soft — when the cap is spent, Gate 2 failure is allowed through and the run completes with an honest partial-reproduction verdict.

### Three verification gates
Structured pass/fail with dynamic confidence thresholds and a supervisor-verifier layer. Gates can halt a run with `blocked_requires_human` unless fail-soft modes (see above) are enabled.

### UI ↔ backend run lifecycle
1. Lab UI (`frontend/src/components/lab/lab-shell.tsx`) → `POST /api/demo` → backend `POST /runs` (or `/runs/upload`).
2. Backend spawns the pipeline subprocess, writes `demo_status.json`, returns initial state.
3. UI opens an **SSE** stream via `/api/demo/events` → backend `/runs/<id>/events`.
4. SSE frame types: `run_state` (full state + stage), `agent_log` (incremental log), `dashboard_event`. `stateMapForRun()` maps backend stage → graph node states.
5. Payload enrichment is timeout-capped on both the GET and SSE routes; the client's `coalesceRunState` keeps the last enriched frame so a timed-out, payload-less frame never regresses the graph. **Don't remove this guard** when refactoring SSE handling — it prevents UI flicker on transient timeouts.

A `localStorage` pointer auto-resumes an in-flight run when the user lands on a bare `/lab`.

### Where to look first
- HTTP layer: `backend/app.py`
- CLI / non-UI runs: `backend/cli.py`
- Pipeline state machine: `backend/agents/orchestrator.py`
- Run modes (sdk / offline): `backend/agents/pipeline.py`
- Subprocess spawn + SSE bridge: `backend/services/events/live_runs.py`
- Paper ingestion: `backend/services/ingestion/parser/resolving_parser.py` (`ResolvingParser` — HTML > PDF > OCR cascade; `ArxivFetcher` writes the HTML sibling)
- `backend/{agents,services}/` is named by function — read it directly.

## Sandboxes
`REPROLAB_DEFAULT_SANDBOX` selects the execution backend: `local`, `docker` (network/memory/CPU controlled), or `runpod` (remote GPU pods, requires `REPROLAB_RUNPOD_API_KEY` and `REPROLAB_RUNPOD_SSH_KEY_PATH`). `start.sh` runs `scripts/runpod_check.sh` as a preflight when sandbox is `runpod`; bypass with `START_SKIP_PREFLIGHT=1`. `START_FULL_SMOKE=1` boots a real pod for end-to-end verification — **this costs money** (cents-scale on RTX 4090).

## Demo gate
When `REPROLAB_DEMO_SECRET` is set, run-start endpoints require a matching `X-Demo-Secret` header (constant-time comparison via `hmac.compare_digest`). Empty/unset secret disables the gate — that's local dev behavior, not a bug.

## Maintaining this doc and `system_overview.md`
`system_overview.md` documents the "why" and "how it fits together"; this file documents the day-to-day. When you add a new pipeline stage, a new gate, a new sandbox, a new fail-soft/fail-closed mode, or change the SSE frame contract, update both. Don't document "what's where" — the code is named by function.

## Context-mode routing
This project inherits the context-mode MCP routing rules from `C:\Users\Armaan\Desktop\CLAUDE.md` (parent). In short: use `ctx_batch_execute` / `ctx_execute` / `ctx_execute_file` for any command or file read producing >20 lines, and `ctx_fetch_and_index` instead of `WebFetch` / `curl` / `wget`. The parent file has the full table of blocked vs. redirected tools.
