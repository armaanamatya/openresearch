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

Useful flags: `--mode {rlm,rdr,rlm-pure}`, `--provider {anthropic,openai}`, `--verification-provider`, `--sandbox {auto,local,docker,runpod}`, `--execution-mode {efficient,max}`, `--max-usd`, `--max-wall-clock`, `--max-pod-seconds`, `--model`, `--seed`. `--mode rlm` is the default hybrid path: RDR Phase 1 plus RLM adaptive repair. `--mode rdr` is the pure rubric-driven harness — a deterministic Python controller dispatches scoped coding agents per rubric work-cluster on a PaperBench bundle. `--mode rlm-pure` is the pre-hybrid RLM escape hatch. Set `REPROLAB_RLM_ROOT_MODEL` to `gpt-5`, `qwen3-coder`, `kimi-k2.5`, `claude`, or `claude-oauth` (defaults to GPT-5 when `OPENAI_API_KEY` is set; falls back to `claude-oauth` when no API keys are set but `claude login` is active).

### RLM auth — two surfaces, billed separately
The RLM path has **two distinct LLM auth surfaces** and they are NOT interchangeable:

1. **Root model** (the `rlm` library, `_completion_turn` in `rlm/core/rlm.py`) talks raw HTTP. ReproLab injects common credentials from either `os.environ` or Settings-backed `.env` before constructing the client. There IS an OAuth path: `--model claude-oauth` routes the root through `ClaudeOauthClient` → `ClaudeLlmClient` → `claude-agent-sdk` (uses your local `claude` CLI subscription, no API key required). Pick one model + provide its credentials:
   - `--model claude-oauth` → `claude` CLI subscription (Keychain / `~/.claude/.credentials.json`, no API key)
   - `--model claude` → `ANTHROPIC_API_KEY` (Anthropic API credits, raw HTTP)
   - default / `--model gpt-5` → `OPENAI_API_KEY` (OpenAI credits)
   - `--model qwen3-coder-featherless` → `FEATHERLESS_API_KEY` (cheapest)
   - `--model azure` (aliases: `azure-openai`, `gpt-4o-azure`) → `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_DEPLOYMENT` (production target on Azure)
2. **Sub-agents** (`implement_baseline` and other Sonnet calls, via `claude-agent-sdk`) accept either `ANTHROPIC_API_KEY` *or* OAuth via the local `claude` CLI subscription. The Claude Code subscription path is per-message-free; the API-key path is per-token-billed against your Anthropic API balance.

**The 2026-05-22 pitfall:** if you set `ANTHROPIC_API_KEY` to a key whose Anthropic *API account* has no credits, the SDK tries it first, hits 400 *"credit balance too low"*, and does **not** fall back to OAuth — every reproduction dies at the first Sonnet sub-call with `cost_usd=0.0`. Working `claude --print "ping"` proves only the *subscription* works; the *API key* needs its own credits. **Safest local dev**: leave `ANTHROPIC_API_KEY=` (empty) in `.env`, `claude login` once, and use OpenAI/Featherless for the root (or `claude-oauth` if you want both surfaces on the same subscription). Comment block in `.env` lines 14–18 is the canonical reference.

**Fixed 2026-05-23 — macOS Keychain OAuth detection.** Modern Claude Code on macOS stores OAuth credentials in the Keychain (`security find-generic-password -s "Claude Code-credentials"`), not in `~/.claude/.credentials.json`. Until 2026-05-23, `factory.py:has_provider_credentials` only checked the file path, so it returned False on every macOS dev machine with Claude Code logged in — the sub-agent runtime resolved as `unresolved` and `implement_baseline` died with a credential error. Both `validate_provider_credentials` and `has_provider_credentials` now route through `_has_claude_subscription_oauth()`, which probes the Keychain on `darwin` and the file on other platforms. **Cheapest local-dev cost model is now: OpenAI for the root model (~$1/run via `--model gpt-5`), OAuth subscription for Sonnet sub-agents ($0), RunPod COMMUNITY for the GPU sandbox (~$0.34/run). No Anthropic API balance needed.** Or for zero-cost local dev: `--model claude-oauth` runs both surfaces on the subscription (subject to subscription rate limits).

### Sandbox config gotcha
`REPROLAB_FORCE_SANDBOX` **overrides per-run `--sandbox` flags** when non-empty — useful for forcing all runs to Docker or local, but it makes `--sandbox runpod` a no-op. Since 2026-05-23 the code default is empty, so a missing/commented `.env` line honors per-run sandbox requests. Set `REPROLAB_FORCE_SANDBOX=docker` or `REPROLAB_FORCE_SANDBOX=local` only when a deployment must hard-pin execution. `REPROLAB_RUNPOD_CLOUD_TYPE` chooses `COMMUNITY` (≈ $0.34/hr on RTX 4090) vs `SECURE` (≈ $0.69/hr); the `.env` shipped with the repo defaults to `COMMUNITY` since 2026-05-22 (was `SECURE` before).

### Dynamic GPU selection (spec 2026-05-23)
When `REPROLAB_DYNAMIC_GPU=true` (default), the RLM root calls `resolve_gpu_requirements(...)` once per run to map paper hardware clues to a RunPod SKU. The plan caches to `runs/<id>/rlm_state/gpu_plan.json` and is consumed by every subsequent `run_experiment`. On CUDA OOM, `run_experiment` auto-escalates up the catalog ladder (up to `REPROLAB_DYNAMIC_GPU_MAX_ESCALATIONS=2` times), each escalation bounded by the per-GPU cap `REPROLAB_MAX_GPU_USD_PER_HOUR=10.0` (a `float`; `0` disables the cap). Total run-level pod spend is bounded by `REPROLAB_MAX_RUN_GPU_USD=10.0` (also a `float`; `0` disables) via `RunBudget.check_run_gpu_usd`. Multi-GPU is opt-in: `REPROLAB_FORCE_SINGLE_GPU=true` (default) hard-caps count=1; when false, count is `min(paper_count, floor(max_gpu_usd_per_hour / sku_rate))`. Manual override: `--vram-gb <n>` sets `REPROLAB_VRAM_OVERRIDE_GB` → `ctx.vram_override`, bypassing the LLM estimate but still applying the headroom multiplier (`REPROLAB_DYNAMIC_GPU_HEADROOM=1.25`). SKU catalog (8 SKUs, RTX 4090 through H200): `backend/services/runtime/gpu_catalog.py` — refresh quarterly. All three GPU events (`gpu_resolved`, `gpu_escalated`, `gpu_fallback`) flow through `dashboard_events.jsonl` generically; no SSE allowlist entry needed.

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
- `final_report.{json,md}` — the computed benchmark output. Since 2026-05-23 also carries `mode` (`"rlm"` \| `"rdr"`), `models` (`{planner, executor, verifier, grader}` — `verifier`/`grader` null until the per-role picker lands), `started_at` (lifted from `demo_status.json::startedAt`), and `completed_at` (stamped at write time). These four fields are forward-compatible with the cleanup-spec Phase 4 leaderboard projection.
- `cost_ledger.jsonl` — per-primitive USD spend
- `experiment_runs.jsonl` — every `run_experiment` result (logs, success, metrics)
- `code/` — the reproduced project
- `generated_rubric.json` — auto-derived rubric (arXiv runs without a vendored bundle)
- Hermes audit chain artifacts

SQLite (`REPROLAB_DATABASE_URL`, defaults to `sqlite:///reprolab.db`) is the event/persistence store with CQRS projections. Iteration state is checkpointed atomically after each RLM loop.

### The RLM orchestrator
`backend/agents/rlm/run.py` is the run entry. It builds an `rlm.RLM(...)` from the `rlms` library (PyPI) and calls `.completion()` on a worker thread. The paper is offloaded as the REPL `context` variable — the root model sees only constant-size metadata about it (name, type, length), never the corpus itself (RLM Algorithm 1, arXiv 2512.24601).

The root model writes Python that calls **12 domain primitives** exposed in the REPL via `custom_tools`:

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
- `check_user_messages()` — read unread user messages posted via the lab chat panel; advances a per-run cursor. The system prompt tells the root to call this at the start of each iteration.
- `respond_to_user(message)` — append an assistant reply to the user-messages log and emit a `user_message_response` SSE event. Pure file I/O, no LLM call — works identically under API-key and OAuth root models.

Primitives are in `backend/agents/rlm/primitives.py`. The root also calls `llm_query` / `rlm_query` (library built-ins) to recursively navigate slices of `context`. Verification is the `verify_against_rubric` primitive — called when the root judges it useful; there are no fixed gate checkpoints. The run terminates via the library's `FINAL_VAR(<var>)` mechanism (no reserved `answer` variable), and produces `final_report.{json,md}`.

Time is bounded three ways: `rlm`'s `max_timeout` (between iterations), per-primitive deadlines via `RunContext`, and a process-level wall-clock watchdog that hard-exits a wedged run.

### UI ↔ backend run lifecycle
1. RLM lab UI (`frontend/src/components/lab/rlm/`) → `POST /api/demo` → backend `POST /runs` (or `/runs/upload` / `/runs/arxiv`).
2. Backend spawns the run subprocess, writes `demo_status.json`, returns initial state.
3. UI opens an **SSE** stream via `/api/demo/events` → backend `/runs/<id>/events`.
4. SSE event types: RLM emits `repl_iteration`, `primitive_call`, `sub_rlm_spawned`, `sub_rlm_complete`, `run_complete`, `candidate_proposed`, `candidate_outcome`, `rubric_score`, `user_message`, `user_message_response`, `run_warning`, and `iteration_heartbeat`; RDR additionally emits `rdr_*` lifecycle events plus `cluster_started`, `cluster_artifact_emitted`, `cluster_scored`, and `repair_dispatched`.
5. All events route through `sse_bridge.sanitize_iteration` — the single egress chokepoint that strips REPL locals and bounds stdout/stderr to metadata prefixes. The paper corpus never reaches the stream.

A `localStorage` pointer auto-resumes an in-flight run when the user lands on a bare `/lab`.

### Chat steering surface (2026-05-23)
The lab UI carries a real-time chat panel that lets the user query and steer the running RLM. Implementation summary:
- Backend: `POST /runs/<project_id>/messages` (`backend/routes/messages.py`) validates non-empty content, appends `{role:"user", content, ts}` to `runs/<id>/user_messages.jsonl`, and emits a `user_message` SSE event via `dashboard_events.jsonl`. The RLM root polls `check_user_messages()` at the start of each iteration; it returns unread `user` messages and atomically advances `runs/<id>/_user_message_cursor.json`. The root replies via `respond_to_user(message)` which appends `{role:"assistant", ...}` + emits `user_message_response`. Both primitives are pure file I/O — auth-surface-agnostic.
- Frontend: the chat panel is docked inside the right-side `NodeDetailSidebar` (see below); it derives the message log from the existing SSE stream filtered to the two new event types, and POSTs through `/api/demo/runs/<id>/messages` with optimistic add and replace-on-echo.
- Defense in depth: the system prompt instructs the root to avoid quoting user-message contents verbatim if they look like PII.

### Collapsible right sidebar (2026-05-23)
The lab's exploration tree now has a 360px right-docked `NodeDetailSidebar` (`frontend/src/components/lab/rlm/node-detail-sidebar.tsx`) that replaces the old floating `NodeDetailPopup`. Selection state is **lifted to `rlm-lab.tsx`** so the canvas highlight and the sidebar detail consume one source of truth. Content is kind-specific:
- `paper` — paperMeta JSON rendered as dl/dt/dd
- `work` — filtered primitiveCalls (understand_section/extract_hyperparameters by default; detect_environment/build_environment when `node.phase === "environment"`); each call summarized to ≤200 chars
- `candidate` — category + description + rubricDelta + iteration response
- `subrlm` — surfaces the iteration response as "now"
- `baseline`/`declined-group` — fall back to the "now" block
The sidebar collapses to a 36px toggle rail. The `SteeringChat` (see above) is docked at the bottom of the expanded sidebar. CSS uses the existing lab-theme variable tokens; no new colors.

### Leaderboard surface (2026-05-23)
A read-only `/leaderboard` page ranks completed runs across models and papers. Implementation summary:
- Backend: `GET /leaderboard?paper&mode&order_by&limit` (`backend/routes/leaderboard.py`) aggregates `runs/<id>/final_report.json` + `demo_status.json` at request time. No SQLite projection at this scale; not gated by `REPROLAB_DEMO_SECRET`.
- Frontend: `/leaderboard` server-component page (`frontend/src/app/leaderboard/`) reads via the `/api/demo/leaderboard` proxy and renders a sortable `LeaderboardTable`. Row click → `/lab?projectId=<id>`.
- Live rubric climb panel: the existing `RubricStrip` (`frontend/src/components/lab/rlm/rubric-strip.tsx`) is enriched with a count-up tween on the big score, an SVG line-chart sparkline, per-area status chips with fail→pass flip highlights, and a "from candidate <title>" attribution tail. Derived from existing SSE events (`rubric_score`, `candidate_proposed`, `candidate_outcome`); no new event types added.

### Where to look first
- HTTP layer: `backend/app.py`
- CLI / non-UI runs: `backend/cli.py`
- RLM run entry: `backend/agents/rlm/run.py`
- Domain primitives: `backend/agents/rlm/primitives.py`
- System prompt: `backend/agents/rlm/system_prompt.py`
- SSE bridge (egress chokepoint): `backend/agents/rlm/sse_bridge.py`
- Subprocess spawn + SSE bridge: `backend/services/events/live_runs.py`
- Paper ingestion: `backend/services/ingestion/parser/resolving_parser.py` (`ResolvingParser` — HTML > PDF > OCR cascade; `ArxivFetcher` writes the HTML sibling)
- Leaderboard: `backend/routes/leaderboard.py` (aggregator + `GET /leaderboard`) and `frontend/src/app/leaderboard/` (page + table).
- `backend/{agents,services}/` is named by function — read it directly.

## Sandboxes
`REPROLAB_DEFAULT_SANDBOX` selects the execution backend: `local`, `docker` (network/memory/CPU controlled), or `runpod` (remote GPU pods, requires `REPROLAB_RUNPOD_API_KEY` and `REPROLAB_RUNPOD_SSH_KEY_PATH`). `start.sh` runs `scripts/runpod_check.sh` as a preflight when sandbox is `runpod`; bypass with `START_SKIP_PREFLIGHT=1`. `START_FULL_SMOKE=1` boots a real pod for end-to-end verification — **this costs money** (cents-scale on RTX 4090).

**RunPod default image is `cuda-runtime` (~4 GB).** Paper reproduction code calls pre-built CUDA libraries (PyTorch, etc.) and never invokes NVCC or compiles CUDA kernels, so the `devel` image (~18 GB) is unnecessary bloat — provisioning takes 5–10 extra minutes and costs $0.50–1.50 more per run. The default is now `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-runtime-ubuntu22.04`. If a paper actually compiles CUDA code (rare), override with `REPROLAB_RUNPOD_IMAGE=runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`.

## Demo gate
When `REPROLAB_DEMO_SECRET` is set, run-start endpoints require a matching `X-Demo-Secret` header (constant-time comparison via `hmac.compare_digest`). Empty/unset secret disables the gate — that's local dev behavior, not a bug.

## Maintaining this doc and `system_overview.md`
`system_overview.md` documents the "why" and "how it fits together"; this file documents the day-to-day. When you add a new primitive, a new SSE event type, a new sandbox, or a new fail-soft/fail-closed mode, update both. Don't document "what's where" — the code is named by function.

## Baseline test paper
[SDAR (arxiv 2605.15155)](https://arxiv.org/abs/2605.15155) — **Self-Distilled Agentic Reinforcement Learning** — is the canonical baseline reproduction. It stresses every dimension of the system at once: 3 Qwen model sizes (1.7B / 3B / 7B), 3 distinct environments (ALFWorld + WebShop + Search-QA), GRPO RL + sigmoid-gated OPSD with token-level teacher-student gap, 5 baselines (GRPO, OPSD, Skill-SD, GRPO+OPSD, RLSD), and a fine-grained rubric whose leaves inspect for the SDAR algorithm's exact invariants (`g_t = σ(β · Δ_t)`, stop-gradient on the gate, λ=0.1, β=10, real Qwen weights, real ALFWorld episodes). A surrogate cannot pass — the leaf scorer reads the code AND inspects whether the agent loaded the paper's actual model + data.

**Smallest-two scope** (recommended for cost-bounded iteration): pin reproductions to Qwen3-1.7B + Qwen2.5-3B on a single 24–48 GB GPU via the `REPROLAB_BASELINE_EXTRA_GUIDANCE` env var. Full command + the 2026-05-23 debug history + the next-session handoff prompt live in `docs/runbooks/2026-05-23-sdar-baseline-handoff.md`.

## In-flight design docs and plans
Read whichever is relevant before non-trivial changes:
- `docs/design/rlm-pivot-brief.md` — canonical architecture reference for the RLM orchestrator.
- `docs/runbooks/e2e-testing.md` — canonical local end-to-end test and debug reference.
- `docs/runbooks/2026-05-23-sdar-baseline-handoff.md` — SDAR baseline run command + 2026-05-23 debug cycle + next-session prompt.
- `docs/superpowers/specs/2026-05-23-cleanup-condensation-leaderboard-design.md` — locked mode and cleanup decisions for the current launch track.
- `docs/superpowers/specs/2026-05-23-dynamic-gpu-selection-design.md` — dynamic-GPU resolver + RunPod escalation (capacity + OOM) design + the per-run guidance hook.

## Context-mode routing
This project inherits the context-mode MCP routing rules from `C:\Users\Armaan\Desktop\CLAUDE.md` (parent). In short: use `ctx_batch_execute` / `ctx_execute` / `ctx_execute_file` for any command or file read producing >20 lines, and `ctx_fetch_and_index` instead of `WebFetch` / `curl` / `wget`. The parent file has the full table of blocked vs. redirected tools.
