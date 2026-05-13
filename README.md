# ReproLab

**Autonomous research agent that reproduces ML papers, then improves them.**

ReproLab takes a paper (PDF upload, arXiv URL, or DOI), reconstructs its implementation environment inside a Docker sandbox, reproduces the core algorithm on the same dataset, launches multiple improvement agents, and independently verifies every result. The output is a reproducible experiment workspace: Docker image, codebase, metrics, plots, diffs, an assumption ledger, and a research map of promising directions and dead ends.

---

## Architecture

```
PDF / arXiv / DOI
        |
  [Ingestion Pipeline]    ---- PyMuPDF parser, arXiv/DOI fetchers
        |
  [Event-Sourced Store]   ---- SQLite event store, CQRS projections
        |
  [9-Stage Agent Pipeline]
        |
        +-- 1. Paper Understanding           -- extracts claims, methods, metrics
        +-- 2. Artifact Discovery             -- finds repos, datasets, configs
        +-- 3. Environment Detective          -- generates Dockerfile + deps
        +-- 4. Reproduction Planner           -- defines contract + eval plan
        +-- [Gate 1: Plan Verification]
        +-- 5. Baseline Implementation        -- adapts repo or generates from paper
        +-- 6. Experiment Runner              -- executes in Docker/RunPod sandbox
        +-- [Gate 2: Baseline Verification]
        +-- 7. Improvement Orchestrator       -- selects N improvement hypotheses
        +-- 8. Path Agents (parallel)         -- each explores one improvement
        +-- [Gate 3: Improvement Verification]
        +-- 9. Research Map                   -- summarizes directions + dead ends
        |
  [Outputs]
        +-- Docker image, reproduction code, metrics, plots
        +-- Assumption ledger, decision log, provenance manifest
        +-- Research map with promising paths + negative results
        +-- Verified baseline report + improvement branch diffs
```

## Partner Integrations

| Partner | Integration | Depth |
|---------|-------------|-------|
| **Claude Agent SDK** | Primary agent orchestration runtime — subagent spawning, tool execution, streaming, structured outputs | Core runtime (`backend/agents/runtime/claude_runtime.py`) |
| **OpenAI Agents SDK** | Full secondary runtime with transparent Claude-to-OpenAI fallback on quota exhaustion | Core runtime (`backend/agents/runtime/openai_runtime.py`) |
| **Nous Hermes** | Self-learning audit chain for independent verification — persists provider memory, quarantines failures, falls through Hermes -> Anthropic -> OpenAI -> Codex CLI | Audit layer (`backend/hermes_audit/`) |
| **OpenAI Codex CLI** | Last-resort audit fallback via `codex exec` using ChatGPT OAuth session | Audit fallback (`backend/hermes_audit/providers.py`) |
| **RunPod** | Remote GPU sandbox execution with pod-lifecycle guardrails (owned-pod allowlist + name-prefix safety) | Sandbox backend (`backend/services/runtime/`) |
| **Docker** | Primary sandbox: per-paper Dockerfile generation, container lifecycle, network/memory/CPU limits | Sandbox backend + Dockerfile generation |
| **ChromaDB** | Optional Layer 2 semantic search with vector embeddings + BM25 fallback | Context layer (`backend/services/context/semantic/`) |
| **Tavily** | AI-native web search for artifact discovery (falls back to DuckDuckGo) | Discovery tool |
| **DeepEval** | Reproduction + innovation evaluation framework | Eval suite (`backend/evals/`) |

## Research Foundation

ReproLab's context management layer is built on the **Recursive Language Model (RLM)** paper ([arXiv:2512.24601](https://arxiv.org/abs/2512.24601), Zhang/Kraska/Khattab, MIT CSAIL). Instead of stuffing context into prompts or embedding everything in a vector store, agents store context as **typed Python variables in per-agent workspaces** and programmatically explore, filter, and drill into them — ~2-3k tokens per query vs 95k+ for naive prompt stuffing.

### Three-Layer Context Strategy

| Layer | Role | Implementation |
|-------|------|----------------|
| **Layer 1 — RLM workspace** | Variables + programmatic exploration. Precise, structured queries. | Event-sourced workspace with `Cited[T]` invariant, 5 workspace tools (`LookupTool`, `SemanticSearchTool`, `ListVariablesTool`, `InspectVariableTool`, `RlmQueryTool`) |
| **Layer 2 — Semantic search** | Fuzzy similarity, discovery of things the agent didn't know to look for | ChromaDB vector embeddings (all-MiniLM-L6-v2) + BM25 fallback |
| **Layer 3 — Knowledge Graph** | Structural code/doc navigation via AST | `graph_query()` for cross-project structural exploration |

Every variable and tool result is wrapped in `Cited[T]` — you cannot construct one without evidence. This is the architectural keystone that enforces assumption transparency across the entire pipeline.

### Docker Sandboxes

Every paper gets its own isolated execution environment. The **Environment Detective** agent analyzes paper dependencies and generates a per-paper Dockerfile, and the **Experiment Runner** executes inside a sandboxed container with enforced resource limits:

- **Per-paper Dockerfile generation** — The Environment Detective parses dependency requirements from the paper, associated repos, and configs, then produces a reproducible Dockerfile
- **Container isolation** — Network, memory (default 4 GB), and CPU limits enforced via Docker runtime constraints
- **Dual sandbox backends** — `LocalDockerBackend` for local execution, `RunPodBackend` for remote GPU workloads (with owned-pod allowlist + name-prefix safety guardrails)
- **Event-sourced lifecycle** — `SandboxRequested → SandboxCreated → CommandExecuted → ArtifactsCollected → SandboxDestroyed`, fully auditable
- **Artifact extraction** — `copy_out` / `copy_in` for moving code, datasets, and results between host and sandbox
- **Local process fallback** — `LocalProcessBackend` for environments where Docker is unavailable (no isolation boundary, development only)

## Key Features

- **Provider resilience** — Typed failure classification (`QuotaExhausted`, `RateLimited`, `TransientError`), bidirectional Anthropic/OpenAI fallback, provider health cooldowns, cost ledger, budget enforcement (`--max-usd`, `--max-wall-clock`)
- **3 verification gates** — Structured pass/fail with dynamic confidence thresholds, supervisor verification, and verifier scoring
- **Assumption transparency** — Every agent decision is citation-backed; ambiguities are logged in a persistent assumption ledger
- **PaperBench head-to-head** — Weight-aware rubric scorer, seeded multi-attempt runner, score-vs-baseline comparison grid
- **Eval framework** — Reproduction scoring, innovation scoring, Bayesian A/B testing, Elo tournaments for comparing agent versions
- **Research workspace** — AST-backed knowledge graph (`graph_query()`), cross-project memory, Git worktree isolation for improvement paths
- **Lab UI** — Live progress strip, agent timeline with per-invocation cards, debug bundle export for triage
- **Dual execution modes** — Full LLM-powered SDK pipeline for any paper; deterministic offline mode for CI/testing

## Quick Start

### Docker (recommended)

```bash
cp .env.example .env
# Add your ANTHROPIC_API_KEY and/or OPENAI_API_KEY to .env

docker compose up --build
```

- Lab UI: http://localhost:3000/lab
- PaperBench: http://localhost:3000/paperbench
- API health: http://localhost:8000/health

### Local Development

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev,evals,semantic,websearch]"     # Windows
# .venv/bin/pip install -e ".[dev,evals,semantic,websearch]"       # Linux/macOS

cd frontend && npm ci && npm run dev &
uvicorn backend.app:create_app --factory --reload --port 8000
```

### CLI

```bash
# Ingest + reproduce a paper (SDK mode, uses LLM)
python -m backend.cli reproduce paper.pdf --provider anthropic

# Offline deterministic pipeline (no LLM, for testing)
python -m backend.cli reproduce paper.pdf --mode offline

# Ingest a paper from arXiv
python -m backend.cli ingest 2512.24601

# Evaluate a completed run
python -m backend.cli eval <project_id> --paper-metrics '{"mean_reward": 500}'

# PaperBench head-to-head
python -m backend.cli paperbench run --bundle ftrl --seed 42
```

## Project Structure

```
backend/
    agents/                   # 9 pipeline agents + orchestrator + resilience engine
        runtime/              # Provider-agnostic runtime (Claude SDK, OpenAI SDK)
        resilience/           # Failure classification, fallback, cost tracking
        prompts/              # Structured agent prompts
    services/                 # Domain services (ingestion, context, runtime, verification, ...)
    hermes_audit/             # Nous Hermes audit chain with self-learning provider memory
    evals/                    # Reproduction + innovation scoring, A/B testing, Elo
    persistence/              # SQLite repositories
    eventstore/               # Event-sourced store + subscriptions
    messaging/                # Command/event bus, idempotency, envelopes
frontend/
    src/app/                  # Next.js pages: landing, /lab, /paperbench
    src/components/           # Dashboard, lab timeline, paperbench client
    src/lib/                  # Event contracts, demo runners, state normalization
tests/                        # 66 test files — unit, integration, e2e
third_party/                  # Vendored PaperBench bundles
docker/                       # Entrypoint script for the full-stack image
```

## Codebase Stats

| Layer | Files | Lines |
|-------|-------|-------|
| Backend (Python) | 195 | ~29,000 |
| Frontend (TypeScript) | 52 | ~8,800 |
| Tests (Python) | 66 | ~12,700 |
| **Total** | **313** | **~50,500** |
