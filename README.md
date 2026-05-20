# ReproLab

**Agent pipeline that reproduces ML papers end-to-end and scores the result.**

ReproLab takes a paper (PDF upload, arXiv ID, or DOI), reconstructs its
implementation environment inside a Docker or RunPod sandbox, reproduces the core
algorithm on the same dataset, explores improvements, and emits a benchmark
report — a computed PaperBench-style rubric plus a statistical comparison against
the paper's claimed results.

> **Architecture pivot in progress (2026-05).** ReproLab is being re-architected
> from a fixed 14-stage pipeline to an **RLM-based orchestrator** built on the
> `rlms` library (Recursive Language Models, arXiv 2512.24601). The canonical
> plan is [`docs/design/rlm-pivot-brief.md`](docs/design/rlm-pivot-brief.md) —
> read it first. The sections below describe the current (pre-pivot) code;
> setup, CLI, and deployment instructions remain valid through the pivot.

---

## Architecture

**Current (pre-pivot).** A 14-stage agent pipeline driven by a `PipelineStage`
state machine with three verification gates: ingest → paper-understanding →
artifact-discovery → environment-detective → reproduction-planner → Gate 1 →
baseline-implementation → experiment-runner → Gate 2 → improvement-selection →
improvement-paths → Gate 3 → research-map → complete.

**Target (RLM pivot).** The 14 stages become callable *primitives*. An RLM root
model — running the `rlms` library's recursive loop — drives them by writing
Python in a REPL, with the paper offloaded as a REPL variable rather than fed
into the model's context. No fixed stage order, no gate control-flow. See the
pivot brief for the full design and build order.

**Shared infrastructure (unchanged by the pivot):**

- **FastAPI backend** (`backend/`) — REST API + SSE event stream, run lifecycle,
  sandbox management
- **Next.js lab frontend** (`frontend/`) — live run UI and benchmark report view
- **Sandbox execution** — Docker (local, network/memory/CPU controlled) or
  RunPod GPU pods (remote, configurable GPU type/count/region)
- **File-backed run state** — `runs/<project_id>/`; SQLite event store with CQRS
  projections, persisted atomically

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.14 (3.11+ supported) |
| Node.js | >=20.19.0 and <21, OR >=22.12.0 (Node 21 is excluded by `engines` in `package.json`) |
| Docker | Engine 26+ / Desktop 4.30+ |
| RunPod account | Required only for `--sandbox runpod` |

API keys required (at minimum one of):

- `ANTHROPIC_API_KEY` — for the Anthropic/Claude provider
- `OPENAI_API_KEY` — for the OpenAI provider

---

## Quick Start

### Option A — Docker Compose (recommended)

```bash
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY and/or OPENAI_API_KEY
docker compose up --build
```

- Lab UI: http://localhost:3000/lab
- PaperBench UI: http://localhost:3000/paperbench
- Backend health: http://localhost:8000/health

### Option B — Local development

**Backend**

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
```

Required environment variables (copy from `.env.example` or set in shell):

```bash
# Minimum required
export ANTHROPIC_API_KEY=sk-ant-...    # or OPENAI_API_KEY for OpenAI provider

# Optional but strongly recommended
export REPROLAB_DEFAULT_SANDBOX=docker  # or runpod / local
export REPROLAB_RUNPOD_API_KEY=...      # required when sandbox=runpod
export REPROLAB_RUNPOD_SSH_KEY_PATH=~/.ssh/id_ed25519

# Database (SQLite default — no setup needed)
# export REPROLAB_DATABASE_URL=sqlite:///reprolab.db
```

Launch the backend:

```bash
.venv/bin/uvicorn backend.app:create_app --factory --reload --port 8000
# Or use the preflight-aware launcher (runs RunPod checks if sandbox=runpod):
./start.sh
```

**Frontend**

```bash
cd frontend
npm ci

# Point the frontend at your running backend
export REPROLAB_BACKEND_URL=http://127.0.0.1:8000

npm run dev        # starts on http://localhost:3000 (default Next.js port)
```

---

## Running a Reproduction

### Via the Lab UI

1. Open http://localhost:3000/lab
2. Upload a paper PDF (or paste an arXiv/DOI URL)
3. Choose provider (Anthropic / OpenAI), execution mode (efficient / max), and
   sandbox (docker / runpod / local)
4. Click **Start Run** and watch the run progress in real time
5. When the run reaches `complete`, open the final benchmark report

### Via the CLI

```bash
# Full SDK pipeline (uses LLM) — PDF, arXiv ID, or DOI all work
python -m backend.cli reproduce paper.pdf --provider anthropic --sandbox docker

# Key flags
#   --mode offline          deterministic, no LLM (for testing)
#   --mode sdk              LLM-powered (default)
#   --provider anthropic|openai
#   --verification-provider anthropic|openai   (separate model for verification)
#   --sandbox auto|local|docker|runpod
#   --execution-mode efficient|max
#   --n-paths 3             number of parallel improvement hypotheses
#   --max-usd 5.00          hard spend cap
#   --max-wall-clock 3600   wall-clock limit in seconds
#   --model claude-sonnet-4-6   override model
#   --seed 42               reproducible run

# Ingest only (no agent pipeline)
python -m backend.cli ingest paper.pdf
python -m backend.cli ingest 2512.24601          # arXiv ID

# Evaluate a completed run
python -m backend.cli eval <project_id> --paper-metrics '{"mean_reward": 500}'
```

---

## Where Results Land

Each run writes to `runs/<project_id>/`:

| File / Dir | Contents |
|---|---|
| `pipeline_state.json` | Full serialized run state including all checkpoints |
| `final_report.json` | Computed benchmark report (structured) |
| `final_report.md` | Human-readable benchmark report |
| `assumption_ledger.json` | Every agent assumption with citations |
| `agent_telemetry.jsonl` | Per-invocation timing and token counts |
| `cost_ledger.jsonl` | Per-agent USD spend and token ledger |
| `Dockerfile` | Generated environment Dockerfile |
| `code/` | Reproduced baseline implementation |
| `hermes/` | Verification audit chain checkpoints |
| `raw_paper.pdf` | Stored copy of the input paper |

The **final report** (`final_report.md`) contains: reproduction fidelity score,
paper-vs-reproduction metric delta table, computed PaperBench rubric
(weight-aware), statistical rigor assessment, improvement summaries, and a
research map of promising directions and negative results.

---

## Testing

**Backend**

```bash
# All tests
.venv/bin/python -m pytest tests/

# With dev dependencies (parallel execution, rerun flaky tests)
pip install -r backend/requirements-dev.txt
.venv/bin/python -m pytest tests/ -n auto
```

**Frontend**

```bash
cd frontend

# Type-check (no Node version constraints)
npx tsc --noEmit

# Unit tests (requires Node >=20.19 <21 or >=22.12)
npm test
```

---

## Project Layout

```
backend/          FastAPI app, agent pipeline, sandbox runtimes, evals
frontend/         Next.js lab UI: run dashboard, report viewer
tests/            Python test suite (unit, integration, e2e)
docs/             Pivot brief, design notes, setup + deployment guides
runs/             Per-run artifact directories (gitignored, mount as volume in prod)
third_party/      Vendored PaperBench bundles
docker/           Container entrypoint script
scripts/          RunPod preflight and utility scripts
```

---

See [`docs/design/rlm-pivot-brief.md`](docs/design/rlm-pivot-brief.md) for the
RLM pivot plan, and [`docs/guides/deployment.md`](docs/guides/deployment.md) for
production deployment.
