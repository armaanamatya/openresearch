# ReproLab

**Agent that reproduces ML papers end-to-end and scores the result.**

ReproLab takes a paper (PDF upload, arXiv ID, or DOI), offloads it into a
persistent Python REPL, and runs an RLM root model that writes code to navigate
and decompose the paper, detect and build the experiment environment, implement
and run a baseline, score against a PaperBench-style rubric, and explore
improvements — producing a `final_report.{json,md}` with a real rubric score.

---

## Architecture

ReproLab is built on the **Recursive Language Model** paradigm (arXiv 2512.24601,
Zhang/Kraska/Khattab, MIT CSAIL). The `rlms` library (`pip install rlms`) is the
engine; our code is the domain layer. The root model never receives the paper
text — it is offloaded as a REPL `context` variable and the model accesses it
programmatically via slices and recursive sub-calls. Domain primitives
(`understand_section`, `detect_environment`, `build_environment`,
`implement_baseline`, `run_experiment`, `verify_against_rubric`,
`propose_improvements`, and others) are exposed as REPL callables in
`backend/agents/rlm/primitives.py`. The root decides what to call and in what
order by writing Python — there is no fixed stage order.

The live run UI (`frontend/src/components/lab/rlm/`) shows the dynamic
exploration tree, a REPL-state panel, a live iteration view, a rubric score bar,
and a primitive-call history. See `docs/design/rlm-pivot-brief.md` for the full
architecture, and `frontend_integration.md` for the SSE event contract.

**Infrastructure (unchanged):**
- **FastAPI backend** (`backend/`) — REST API + SSE event stream, run lifecycle
- **Next.js lab frontend** (`frontend/`) — live run UI and benchmark report view
- **Sandbox execution** — Docker (local, network/memory/CPU controlled) or RunPod GPU pods
- **File-backed run state** — `runs/<project_id>/`; SQLite event store, persisted atomically

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.14 (3.11+ supported) |
| Node.js | >=20.19.0 and <21, OR >=22.12.0 |
| Docker | Engine 26+ / Desktop 4.30+ |
| RunPod account | Required only for `--sandbox runpod` |

API keys — at minimum one of `ANTHROPIC_API_KEY` (Anthropic/Claude) or `OPENAI_API_KEY` (OpenAI). The RLM path has two distinct auth surfaces:

- **Root model** (the `rlm` library orchestrator) talks raw HTTP and needs a real, **credited** API key in `os.environ`. Pick one of:
  - `--model claude` → needs `ANTHROPIC_API_KEY` with Anthropic API credits (≈ $2–3 per reproduction)
  - default / `--model gpt-5` → needs `OPENAI_API_KEY` with OpenAI credits (≈ $3–5 per reproduction) — paper-validated RLM root
  - `--model qwen3-coder-featherless` → needs `FEATHERLESS_API_KEY` (cheapest, ≈ $0.40/MTok); Qwen3-Coder via OpenRouter requires separate credentials
- **Sub-agents** (`implement_baseline` etc., via `claude-agent-sdk`) accept either an API key **or** OAuth via the local `claude` CLI subscription (no per-token billing). For local dev the cheapest setup is: **leave `ANTHROPIC_API_KEY` empty** in `.env`, run `claude login` once, and the SDK uses your subscription for every Sonnet sub-call.

**Pitfall:** if you set `ANTHROPIC_API_KEY` to a key whose **Anthropic API account has no credits**, the SDK tries that key first, gets a 400 *"credit balance too low"*, and **does not fall back to OAuth** — so your reproductions die at the first sub-call with `cost_usd=0.0`. The Anthropic *API* balance and the Claude Code *subscription* are billed separately; running `claude --print "ping"` proves the subscription works, but the API key still needs its own credits if you choose to set it. The safest default is **empty `ANTHROPIC_API_KEY` + working OAuth + a credited root model (OpenAI or Featherless)**.

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
4. Click **Start Run** and watch the dynamic exploration tree populate live
5. When the run finishes, open the final benchmark report

### Via the CLI

```bash
# Full pipeline — PDF, arXiv ID, or DOI all work
python -m backend.cli reproduce paper.pdf --provider anthropic --sandbox docker
python -m backend.cli reproduce 2512.24601   # arXiv ID

# Key flags
#   --provider anthropic|openai
#   --verification-provider anthropic|openai   (separate model for verification)
#   --sandbox auto|local|docker|runpod
#   --execution-mode efficient|max
#   --max-usd 5.00          hard spend cap
#   --max-wall-clock 3600   wall-clock limit in seconds
#   --model <name>          override root model
#   --seed 42               reproducible run

# Ingest only (no agent pipeline)
python -m backend.cli ingest paper.pdf
python -m backend.cli ingest 2512.24601      # arXiv ID
```

---

## Where Results Land

Each run writes to `runs/<project_id>/`:

| File / Dir | Contents |
|---|---|
| `final_report.json` | Computed benchmark report (structured) |
| `final_report.md` | Human-readable benchmark report |
| `dashboard_events.jsonl` | Append-only SSE event log |
| `cost_ledger.jsonl` | Per-primitive USD spend and token ledger |
| `experiment_runs.jsonl` | Every `run_experiment` result (logs, success, metrics) |
| `rlm_state/` | Per-iteration checkpoints (resume-safe) |
| `generated_rubric.json` | Auto-derived rubric (arXiv runs without a vendored bundle) |
| `code/` | Reproduced baseline implementation |
| `hermes/` | Verification audit chain checkpoints |
| `raw_paper.pdf` | Stored copy of the input paper |

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
backend/          FastAPI app, RLM orchestrator, primitives, sandbox runtimes
frontend/         Next.js lab UI: live run dashboard, report viewer
tests/            Python test suite (unit, integration, e2e)
docs/             Architecture brief, design notes, setup + deployment guides
runs/             Per-run artifact directories (gitignored, mount as volume in prod)
third_party/      Vendored PaperBench bundles
docker/           Container entrypoint script
scripts/          RunPod preflight and utility scripts
```

---

See [`docs/design/rlm-pivot-brief.md`](docs/design/rlm-pivot-brief.md) for the
full architecture reference, [`frontend_integration.md`](frontend_integration.md)
for the SSE event contract, and
[`docs/guides/deployment.md`](docs/guides/deployment.md) for production deployment.
