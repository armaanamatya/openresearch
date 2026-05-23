# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: OpenResearch / ReproLab

An agent that reproduces research papers end-to-end: ingest paper → offload it as a REPL variable → RLM root model writes Python to understand claims, build an environment, implement and run a baseline, score against a rubric, and explore improvements → emit `final_report.{json,md}`. See `system_overview.md` and `docs/design/rlm-pivot-brief.md` for the full "why" — read those before making non-trivial architectural changes.

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
python -m backend.cli reproduce 2512.24601                      # arXiv ID
python -m backend.cli ingest paper.pdf                          # ingest only
```

Useful flags: `--mode rlm` (the only supported mode, and the default), `--provider {anthropic,openai}`, `--verification-provider`, `--sandbox {auto,local,docker,runpod}`, `--execution-mode {efficient,max}`, `--max-usd`, `--max-wall-clock`, `--model`, `--seed`. Set `REPROLAB_RLM_ROOT_MODEL` to `gpt-5`, `qwen3-coder`, `kimi-k2.5`, or `claude` (defaults to GPT-5 when `OPENAI_API_KEY` is set).

### RLM auth — two surfaces, billed separately
The RLM path has **two distinct LLM auth surfaces** and they are NOT interchangeable:

1. **Root model** (the `rlm` library, `_completion_turn` in `rlm/core/rlm.py`) talks raw HTTP. It reads `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `FEATHERLESS_API_KEY` directly from `os.environ`. **There is no OAuth path for the root model.** Pick one model + provide that key with real billing credits:
   - `--model claude` → `ANTHROPIC_API_KEY` (Anthropic API credits)
   - default / `--model gpt-5` → `OPENAI_API_KEY` (OpenAI credits)
   - `--model qwen3-coder-featherless` → `FEATHERLESS_API_KEY` (cheapest)
2. **Sub-agents** (`implement_baseline` and other Sonnet calls, via `claude-agent-sdk`) accept either `ANTHROPIC_API_KEY` *or* OAuth via the local `claude` CLI subscription. The Claude Code subscription path is per-message-free; the API-key path is per-token-billed against your Anthropic API balance.

**The 2026-05-22 pitfall (see `docs/superpowers/specs/2026-05-22-rlm-debug-harden-handoff.md` §auth):** if you set `ANTHROPIC_API_KEY` to a key whose Anthropic *API account* has no credits, the SDK tries it first, hits 400 *"credit balance too low"*, and does **not** fall back to OAuth — every reproduction dies at the first Sonnet sub-call with `cost_usd=0.0`. Working `claude --print "ping"` proves only the *subscription* works; the *API key* needs its own credits. **Safest local dev**: leave `ANTHROPIC_API_KEY=` (empty) in `.env`, `claude login` once, and use OpenAI/Featherless for the root. Comment block in `.env` lines 14–18 is the canonical reference.

**Fixed 2026-05-23 — macOS Keychain OAuth detection.** Modern Claude Code on macOS stores OAuth credentials in the Keychain (`security find-generic-password -s "Claude Code-credentials"`), not in `~/.claude/.credentials.json`. Until 2026-05-23, `factory.py:has_provider_credentials` only checked the file path, so it returned False on every macOS dev machine with Claude Code logged in — the sub-agent runtime resolved as `unresolved` and `implement_baseline` died with a credential error. Both `validate_provider_credentials` and `has_provider_credentials` now route through `_has_claude_subscription_oauth()`, which probes the Keychain on `darwin` and the file on other platforms. **Cheapest local-dev cost model is now: OpenAI for the root model (~$1/run via `--model gpt-5`), OAuth subscription for Sonnet sub-agents ($0), RunPod COMMUNITY for the GPU sandbox (~$0.34/run). No Anthropic API balance needed.**

### Sandbox config gotcha
`REPROLAB_FORCE_SANDBOX` in `.env` (when set) **overrides per-run `--sandbox` flags** — useful for forcing all local runs to Docker, but it silently makes `--sandbox runpod` a no-op. `REPROLAB_RUNPOD_CLOUD_TYPE` choose `COMMUNITY` (≈ $0.34/hr on RTX 4090) vs `SECURE` (≈ $0.69/hr); the `.env` shipped with the repo defaults to `COMMUNITY` since 2026-05-22 (was `SECURE` before).

### Docker

```bash
cp .env.example .env   # set ANTHROPIC_API_KEY and/or OPENAI_API_KEY
docker compose up --build
```

## Architecture — the non-obvious parts

### One image, two processes
`docker/entrypoint.sh` (under `tini`) runs the FastAPI backend on internal `:8000` and the Next.js frontend on public `:$PORT`. The frontend reaches the backend **server-side only** through `/api/demo/*` proxy routes — there is no CORS layer because the browser never talks to the backend directly. When debugging UI-vs-API issues, check the Next.js proxy route under `frontend/src/app/api/demo/`, not CORS.

### File-backed run state, not a service
Each run is a **long-lived subprocess** spawned by the backend. Run state lives in `runs/<project_id>/`:
- `demo_status.json` — UI-facing status snapshot (atomic write)
- `rlm_state/` — per-iteration checkpoints; resume-safe
- `dashboard_events.jsonl` — append-only SSE event log
- `final_report.{json,md}` — the computed benchmark output
- `cost_ledger.jsonl` — per-primitive USD spend
- `experiment_runs.jsonl` — every `run_experiment` result (logs, success, metrics)
- `code/` — the reproduced project
- `generated_rubric.json` — auto-derived rubric (arXiv runs without a vendored bundle)
- Hermes audit chain artifacts

SQLite (`REPROLAB_DATABASE_URL`, defaults to `sqlite:///reprolab.db`) is the event/persistence store with CQRS projections. Iteration state is checkpointed atomically after each RLM loop.

### The RLM orchestrator
`backend/agents/rlm/run.py` is the run entry. It builds an `rlm.RLM(...)` from the `rlms` library (PyPI) and calls `.completion()` on a worker thread. The paper is offloaded as the REPL `context` variable — the root model sees only constant-size metadata about it (name, type, length), never the corpus itself (RLM Algorithm 1, arXiv 2512.24601).

The root model writes Python that calls **10 domain primitives** exposed in the REPL via `custom_tools`:

- `understand_section(text_slice)` — datasets, metrics, training recipe, hardware clues, ambiguities from a slice
- `extract_hyperparameters(text_slice)` — optimizer, learning rate, batch size, epochs
- `detect_environment(method_spec)` — EnvironmentSpec (Dockerfile, framework, packages)
- `build_environment(env_spec)` — build the Docker image, repairing the Dockerfile on failure
- `plan_reproduction(method_spec, env_spec)` — ReproductionContract (smoke-test plan, eval plan)
- `implement_baseline(plan)` — run the code-writing agent; returns the code directory path
- `run_experiment(code_path, env_id)` — execute the baseline in a Docker container; returns `{success, metrics, logs}`
- `verify_against_rubric(results, rubric)` — score results against a PaperBench-style rubric
- `propose_improvements(current_results, rubric_scores, k)` — paper-specific improvement hypotheses with free-form tags
- `record_candidate_outcome(candidate_id, outcome, parent_id)` — record the root's outcome decision for a candidate

Primitives are in `backend/agents/rlm/primitives.py`. The root also calls `llm_query` / `rlm_query` (library built-ins) to recursively navigate slices of `context`. Verification is the `verify_against_rubric` primitive — called when the root judges it useful; there are no fixed gate checkpoints. The run terminates via the library's `FINAL_VAR(<var>)` mechanism (no reserved `answer` variable), and produces `final_report.{json,md}`.

Time is bounded three ways: `rlm`'s `max_timeout` (between iterations), per-primitive deadlines via `RunContext`, and a process-level wall-clock watchdog that hard-exits a wedged run.

### UI ↔ backend run lifecycle
1. RLM lab UI (`frontend/src/components/lab/rlm/`) → `POST /api/demo` → backend `POST /runs` (or `/runs/upload` / `/runs/arxiv`).
2. Backend spawns the run subprocess, writes `demo_status.json`, returns initial state.
3. UI opens an **SSE** stream via `/api/demo/events` → backend `/runs/<id>/events`.
4. SSE event types (full schema in `frontend_integration.md`): `repl_iteration`, `primitive_call`, `sub_rlm_spawned`, `sub_rlm_complete`, `run_complete`, `candidate_proposed`, `candidate_outcome`, `rubric_score`.
5. All events route through `sse_bridge.sanitize_iteration` — the single egress chokepoint that strips REPL locals and bounds stdout/stderr to metadata prefixes. The paper corpus never reaches the stream.

A `localStorage` pointer auto-resumes an in-flight run when the user lands on a bare `/lab`.

### Where to look first
- HTTP layer: `backend/app.py`
- CLI / non-UI runs: `backend/cli.py`
- RLM run entry: `backend/agents/rlm/run.py`
- Domain primitives: `backend/agents/rlm/primitives.py`
- System prompt: `backend/agents/rlm/system_prompt.py`
- SSE bridge (egress chokepoint): `backend/agents/rlm/sse_bridge.py`
- Subprocess spawn + SSE bridge: `backend/services/events/live_runs.py`
- Paper ingestion: `backend/services/ingestion/parser/resolving_parser.py` (`ResolvingParser` — HTML > PDF > OCR cascade; `ArxivFetcher` writes the HTML sibling)
- `backend/{agents,services}/` is named by function — read it directly.

## Sandboxes
`REPROLAB_DEFAULT_SANDBOX` selects the execution backend: `local`, `docker` (network/memory/CPU controlled), or `runpod` (remote GPU pods, requires `REPROLAB_RUNPOD_API_KEY` and `REPROLAB_RUNPOD_SSH_KEY_PATH`). `start.sh` runs `scripts/runpod_check.sh` as a preflight when sandbox is `runpod`; bypass with `START_SKIP_PREFLIGHT=1`. `START_FULL_SMOKE=1` boots a real pod for end-to-end verification — **this costs money** (cents-scale on RTX 4090).

## Demo gate
When `REPROLAB_DEMO_SECRET` is set, run-start endpoints require a matching `X-Demo-Secret` header (constant-time comparison via `hmac.compare_digest`). Empty/unset secret disables the gate — that's local dev behavior, not a bug.

## Maintaining this doc and `system_overview.md`
`system_overview.md` documents the "why" and "how it fits together"; this file documents the day-to-day. When you add a new primitive, a new SSE event type, a new sandbox, or a new fail-soft/fail-closed mode, update both. Don't document "what's where" — the code is named by function.

## In-flight design docs and plans
Read whichever is relevant before non-trivial changes:
- `docs/design/rlm-pivot-brief.md` — canonical architecture reference for the RLM orchestrator.
- `docs/design/project-state-audit-2026-05-22.md` — read-only whole-repo audit captured 2026-05-22.
- `docs/superpowers/plans/2026-05-22-infrastructure-improvement-plan.md` — infra improvement catalog (7 candidates, phased) + detailed Phase 1 TDD plan for a `max_pod_seconds` pod-time budget cap that closes the runaway-RunPod-pod cost gap. Sandbox + resilience layer only.

## Context-mode routing
This project inherits the context-mode MCP routing rules from `C:\Users\Armaan\Desktop\CLAUDE.md` (parent). In short: use `ctx_batch_execute` / `ctx_execute` / `ctx_execute_file` for any command or file read producing >20 lines, and `ctx_fetch_and_index` instead of `WebFetch` / `curl` / `wget`. The parent file has the full table of blocked vs. redirected tools.
