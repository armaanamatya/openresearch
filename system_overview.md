# System Overview — OpenResearch / ReproLab

> Orientation for new Claude Code / Codex sessions: the non-obvious "why" and
> "how it fits together." For *what's where*, read the code — it's named by
> function. Keep this current.

## Architecture pivot in progress (2026-05)

ReproLab is being re-architected. The **current** code is a 14-stage pipeline
state machine. The **target** is an RLM-based orchestrator built on the `rlms`
library (Recursive Language Models — arXiv 2512.24601, Zhang/Kraska/Khattab,
MIT CSAIL).

The canonical plan is **`docs/design/rlm-pivot-brief.md`** — read it before any
non-trivial change. This document describes the current code honestly and
sketches the target; where the two conflict, the brief is the direction.

RLM is the *paradigm the project is built on*, not a paper ReproLab reproduces.
ReproLab reproduces other papers; the `rlms` library is its substrate.

## Goal

Given a paper (arXiv link or uploaded PDF), ReproLab ingests it, understands the
claimed results, builds an environment, implements and runs a baseline, scores
the reproduction against a PaperBench-style rubric, explores improvements, and
emits a benchmark report comparing the reproduction to the paper's claims.

## How it fits together (infrastructure — unchanged by the pivot)

One Docker image, two processes (`docker/entrypoint.sh` under `tini`):

- **Backend** — FastAPI (`backend/app.py`), internal `:8000`. Each run is a
  **long-lived subprocess** it spawns and tracks.
- **Frontend** — Next.js 16 (`frontend/`), public `:$PORT`. The "lab" UI;
  reaches the backend **server-side only** via `/api/demo/*` proxy routes (no
  CORS — the browser never talks to the backend directly).

Run state is **file-backed**, not a service — `runs/<project_id>/` holds the
status snapshot, checkpointed state, `final_report.{json,md}`, `*.jsonl` event
logs, the reproduced `code/`, and audit artifacts. SQLite
(`REPROLAB_DATABASE_URL`) is the event/persistence store.

## Current architecture (pre-pivot — being replaced)

The pipeline is a **14-stage state machine** — `PipelineStage` in
`backend/agents/orchestrator.py` is the source of truth for stage order; the
orchestrator advances the stage and checkpoints `pipeline_state.json` after
each. Order: ingest → paper-understanding → artifact-discovery →
environment-detective → reproduction-planner → Gate 1 → baseline-implementation
→ experiment-runner → Gate 2 → improvement-selection → improvement-paths →
Gate 3 → research-map → complete. Three verification gates produce structured
pass/fail verdicts; a rubric-verifier scores the reproduction against a
PaperBench-style rubric at Gates 2 and 3.

This fixed-stage machine — its ordered stages, its gate control-flow — is what
the pivot replaces.

## Target architecture (RLM pivot)

The 14 stages collapse into a library of **primitives** (`understand_section`,
`build_environment`, `run_experiment`, `verify_against_rubric`, …). An **RLM
root model** — running the `rlms` library's recursive loop — receives the paper
offloaded as a REPL variable and writes Python that calls those primitives, and
recursive sub-calls (`llm_query` / `rlm_query`), in whatever order it judges
useful. There is no fixed stage order and no gate control-flow. The reproduction
is built up as REPL state and returned as the run's `answer`.

`rlms` is a dependency (`pip install rlms`); domain primitives are passed to it
via its `custom_tools` argument. See `docs/design/rlm-pivot-brief.md` for the
full design, the RLM-fidelity invariants, and the build order.

## Run lifecycle (UI ↔ backend)

1. Lab UI (`frontend/src/components/lab/lab-shell.tsx`) starts a run — arXiv
   link or PDF → `POST /api/demo` → backend `POST /runs` / `/runs/upload`.
2. Backend spawns the run subprocess, writes the status snapshot, returns state.
3. UI opens an **SSE** stream (`/api/demo/events` → backend `/runs/<id>/events`).
4. On completion the computed `final_report` replaces the placeholder benchmark.

The **current** SSE frames are stage-oriented (`run_state`, `agent_log`,
`dashboard_event`); the pivot replaces them with iteration / primitive-call
events (see the brief's event-schema section). **Which run is current is the
URL** — `/lab?projectId=<id>` — so a refresh or a shared link reopens it.

## Where to look

- **Backend** — `app.py` (HTTP), `cli.py` (CLI / non-UI runs),
  `agents/orchestrator.py` (current state machine), `agents/pipeline.py` (run
  modes), `services/events/live_runs.py` (subprocess spawn + SSE bridge).
- **Frontend** — `app/lab/page.tsx` → `components/lab/lab-shell.tsx` (lab UI),
  `app/api/demo/*` (backend proxy routes).
- **RLM substrate** — the `rlms` library (PyPI; reference implementation at
  `github.com/alexzhang13/rlm`) and the existing, currently dormant
  `backend/services/context/workspace/tools/rlm_query.py`.

## Docs

- `docs/design/rlm-pivot-brief.md` — the canonical pivot plan and spec.
- `learn.md` — post-mortems: bugs shipped + the guardrail for each.
- `docs/guides/setup-guide.md`, `docs/guides/deployment.md`, `README.md` — setup
  and deployment.

## Maintaining this doc

Orientation only — keep it at the "why / how it fits" altitude, never an
inventory of files. When a change makes a statement here wrong, fix it in the
same change. Once the pivot lands, delete the "current architecture" section and
fold the target into the present tense.
