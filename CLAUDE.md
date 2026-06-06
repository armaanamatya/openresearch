<!-- doc-meta: status=current; last-verified=2026-06-06 -->
# CLAUDE.md

> **Doc status:** Current · source-of-truth tier 2 (day-to-day) · last verified
> 2026-06-06. Policy: [`docs/policies/documentation.md`](docs/policies/documentation.md).

Guidance for Claude Code (claude.ai/code) working in this repo. This documents the **day-to-day**; `system_overview.md` and `docs/design/rlm-pivot-brief.md` document the "why" — read those before non-trivial architectural changes.

## Project: OpenResearch / OpenResearch

An agent that reproduces research papers end-to-end: ingest paper → offload it as a REPL variable → RLM root model writes Python to understand claims, build an environment, implement and run a baseline, score against a rubric, and explore improvements → emit `final_report.{json,md}`.

## Common commands

### Backend (Python 3.14.2, FastAPI)

```bash
pip install -r backend/requirements.txt
pip install -r backend/requirements-dev.txt   # adds pytest + parallel runners

# Run the API (factory pattern — --factory is required)
.venv/bin/uvicorn backend.app:create_app --factory --reload --port 8000
./start.sh                                     # preflight-aware launcher (RunPod checks when sandbox=runpod)

# Tests
.venv/bin/python -m pytest tests/                       # all
.venv/bin/python -m pytest tests/ -n auto               # parallel (needs requirements-dev)
.venv/bin/python -m pytest tests/path/to/test_x.py::test_name   # single
.venv/bin/python -m pytest tests/ --reruns 2            # rerun flaky network tests
```

Pytest config in `pyproject.toml` (`testpaths=["tests"]`, `pythonpath=["src"]`). No repo-level lint/format step.

### Frontend (Next.js 16, Node ≥20.19 <21 or ≥22.12)

```bash
cd frontend && npm ci
export OPENRESEARCH_BACKEND_URL=http://127.0.0.1:8000   # required: server-side proxy target
npm run dev          # http://localhost:3000
npm run build
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

Flags: `--mode {rlm,rdr,rlm-pure}`, `--provider {anthropic,openai}`, `--verification-provider`, `--sandbox {auto,local,docker,runpod}`, `--execution-mode {efficient,max}`, `--max-usd`, `--max-wall-clock`, `--max-pod-seconds`, `--model`, `--seed`, `--vram-gb`.
- `--mode rlm` (default) — hybrid: RDR Phase 1 + RLM adaptive repair.
- `--mode rdr` — pure rubric-driven harness; a deterministic Python controller dispatches scoped coding agents per rubric work-cluster on a PaperBench bundle.
- `--mode rlm-pure` — pre-hybrid RLM escape hatch.

`OPENRESEARCH_RLM_ROOT_MODEL` ∈ `gpt-5`, `qwen3-coder`, `kimi-k2.5`, `claude`, `claude-oauth` (defaults to GPT-5 when `OPENAI_API_KEY` set; falls back to `claude-oauth` when no API keys but `claude login` active).

### RLM auth — two surfaces, billed separately
Two distinct LLM auth surfaces, NOT interchangeable:

1. **Root model** (`rlm` library, `_completion_turn` in `rlm/core/rlm.py`) talks raw HTTP. Pick one model + provide its credential:
   - `--model claude-oauth` → `claude` CLI subscription (macOS Keychain or `~/.claude/.credentials.json`, no API key)
   - `--model claude` → `ANTHROPIC_API_KEY` (Anthropic API credits)
   - default / `--model gpt-5` → `OPENAI_API_KEY`
   - `--model qwen3-coder-featherless` → `FEATHERLESS_API_KEY` (cheapest)
   - `--model azure` (aliases `azure-openai`, `gpt-4o-azure`) → `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_DEPLOYMENT`
2. **Sub-agents** (`implement_baseline` + other Sonnet calls, via `claude-agent-sdk`) accept either `ANTHROPIC_API_KEY` *or* OAuth via the local `claude` CLI subscription (subscription path is per-message-free; API-key path is per-token-billed).

**Gotchas (rules):**
- A no-credit `ANTHROPIC_API_KEY` does NOT fall back to OAuth — the SDK hits `400 credit balance too low` and every run dies at the first Sonnet sub-call with `cost_usd=0.0`. Working `claude --print "ping"` proves only the *subscription*, not the *API key*. Safest local dev: leave `ANTHROPIC_API_KEY=` empty, `claude login` once, use OpenAI/Featherless for the root (or `claude-oauth` for both surfaces on the subscription). See `.env` lines 14–18.
- **Shell wins over `.env`.** Credential injection in `run.py` reads `os.environ` first, falls back to Settings — so a stale shell export of `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `OPENRESEARCH_RUNPOD_API_KEY` silently shadows `.env`. Workaround: prefix the CLI with `env -u OPENAI_API_KEY` (or unset). Boot-time mismatch validator tracked as BUG-LR-014.
- **Cheapest local-dev cost model:** OpenAI root (~$1/run via `--model gpt-5`) + OAuth subscription for Sonnet sub-agents ($0) + RunPod COMMUNITY GPU (~$0.34/run); no Anthropic API balance needed. Zero-cost: `--model claude-oauth` for both surfaces (subject to subscription rate limits). macOS OAuth detection probes the Keychain via `_has_claude_subscription_oauth()`.

### Sandbox config
- `OPENRESEARCH_FORCE_SANDBOX` (default empty) **overrides per-run `--sandbox`** when non-empty — set to `docker`/`local` only to hard-pin a deployment; non-empty makes `--sandbox runpod` a no-op.
- `OPENRESEARCH_RUNPOD_CLOUD_TYPE` — `COMMUNITY` (~$0.34/hr RTX 4090, default) vs `SECURE` (~$0.69/hr).
- See the **Sandboxes** section below for the full backend matrix + the Docker-daemon prerequisite.

### Feature flags (default-off unless noted; full design in the cited spec; A/B ≥3 paired SDAR runs before flipping any default)
- **gpt-5-mini navigation route** — routes hot-volume `rlm_query`/`llm_query` *navigation* sub-calls off the bundled-CLI transport (orphaned-child wedge, FM-001) onto OpenAI gpt-5-mini via openai/httpx. Quality-critical grader (`verify_against_rubric`/`propose_improvements`) stays on Sonnet-OAuth: keep `OPENRESEARCH_ACCELERATOR_SCOPE=navigation`, do NOT use `all`. Opt-in: `OPENRESEARCH_ACCELERATOR=endpoint`, `OPENRESEARCH_ACCELERATOR_BASE_URL=https://api.openai.com/v1`, `OPENRESEARCH_ACCELERATOR_MODEL=gpt-5-mini`, `OPENRESEARCH_SUBRLM_OPENAI_TIMEOUT_S=120` (uses `OPENAI_API_KEY` automatically). Rollback `OPENRESEARCH_ACCELERATOR=off`. Code: `backend/agents/rlm/accelerator.py`.
- **Intra-run context map (PEEK-lite)** — `OPENRESEARCH_CONTEXT_MAP=on` enables a free deterministic orientation cache: write hook in `binding.py` unions structured outputs of `understand_section`/`extract_hyperparameters`/`detect_environment` into `runs/<id>/rlm_state/context_map.json` (union-per-field, ≤40 fields/≤8 values/≤8KB incremental ceiling, thread-safe, fail-soft); root reads via `read_context_map()` and (only when the flag is on) is told to consult it before re-deriving. Navigation aid only, never a report source (the evidence gate `OPENRESEARCH_EVIDENCE_GATE` remains the backstop). Off-state: write no-ops, read returns empty, prompt instruction omitted. Code: `backend/agents/rlm/context_map.py`; spec `2026-05-30-intra-run-context-map-design.md`.
- **Per-paper negative lessons (MUSE-lite)** — `OPENRESEARCH_NEGATIVE_LESSONS=1` enables cross-run per-paper failure memory: `run.py::_finalize` hook (`lesson_distiller.py::mine_lessons`) mines agent-correctable `failure_class` rows from `experiment_runs.jsonl` into `runs/_lessons/<arxiv_id>.json`; next run of the same `arxiv_id` injects active lessons into implementer guidance via `baseline_implementation.py::_negative_lessons_block`. Guardrails: recurrence-gated promotion (`occurrences>=2`, except `dockerfile_invalid` at 1), classifier-sourced `suggested_fix` (never agent prose), opportunity-aware retirement (staleness>=3, gated on phase reached). Capped ≤5/≤200 chars, advisory only. Needs `ctx.arxiv_id`. Spec `2026-05-30-...`.
- **Dynamic GPU selection** — `OPENRESEARCH_DYNAMIC_GPU=true` (**default ON**): root calls `resolve_gpu_requirements(...)` once/run to map paper hardware clues → RunPod SKU, cached to `runs/<id>/rlm_state/gpu_plan.json`, consumed by every `run_experiment`. On CUDA OOM, `run_experiment` escalates up the catalog ladder (≤`OPENRESEARCH_DYNAMIC_GPU_MAX_ESCALATIONS=2`). Caps: `OPENRESEARCH_MAX_GPU_USD_PER_HOUR=10.0` per-GPU + `OPENRESEARCH_MAX_RUN_GPU_USD=10.0` run-total (both float; `0` disables) via `RunBudget.check_run_gpu_usd`. Multi-GPU opt-in: `OPENRESEARCH_FORCE_SINGLE_GPU=false` (default true). `--vram-gb`→`OPENRESEARCH_VRAM_OVERRIDE_GB` bypasses the LLM estimate (still applies `OPENRESEARCH_DYNAMIC_GPU_HEADROOM=1.25`). SKU catalog `backend/services/runtime/gpu_catalog.py` (refresh quarterly). Events `gpu_resolved`/`gpu_escalated`/`gpu_fallback`. Spec `2026-05-23-dynamic-gpu-selection-design.md`.

### Docker
```bash
cp .env.example .env   # set ANTHROPIC_API_KEY and/or OPENAI_API_KEY
docker compose up --build
```

## Architecture — the non-obvious parts

### One image, two processes
`docker/entrypoint.sh` (under `tini`) runs FastAPI on internal `:8000` and Next.js on public `:$PORT`. Frontend reaches backend **server-side only** through `/api/demo/*` proxy routes — no CORS layer (browser never talks to backend directly). Debug UI-vs-API issues in `frontend/src/app/api/demo/`, not CORS.

### File-backed run state, not a service
Each run is a **long-lived subprocess**. State in `runs/<project_id>/`:
- `demo_status.json` — UI status snapshot (atomic write)
- `rlm_state/` — per-iteration checkpoints; resume-safe
- `dashboard_events.jsonl` — append-only SSE event log
- `final_report.{json,md}` — benchmark output; also carries `mode` (`rlm`|`rdr`), `models` (`{planner,executor,verifier,grader}` — verifier/grader null until the per-role picker lands), `started_at`, `completed_at`
- `cost_ledger.jsonl` — per-primitive USD spend
- `experiment_runs.jsonl` — every `run_experiment` result (logs, success, metrics)
- `code/` — the reproduced project
- `generated_rubric.json` — auto-derived rubric (arXiv runs without a vendored bundle)
- Hermes audit chain artifacts

SQLite (`OPENRESEARCH_DATABASE_URL`, default `sqlite:///openresearch.db`) is the event/persistence store (CQRS projections). Iteration state checkpointed atomically after each loop.

### The RLM orchestrator
`backend/agents/rlm/run.py` is the entry: builds `rlm.RLM(...)` (PyPI `rlms`) and calls `.completion()` on a worker thread. Paper is offloaded as the REPL `context` variable — the root sees only constant-size metadata (name, type, length), never the corpus (RLM Algorithm 1, arXiv 2512.24601).

Root writes Python calling **12 domain primitives** (`backend/agents/rlm/primitives.py`) via `custom_tools`:
`understand_section`, `extract_hyperparameters`, `detect_environment`, `build_environment`, `plan_reproduction`, `implement_baseline`, `run_experiment`, `verify_against_rubric`, `propose_improvements`, `record_candidate_outcome`, `check_user_messages` (root calls at start of each iteration), `respond_to_user` (pure file I/O + `user_message_response` SSE).

Root also calls `llm_query`/`rlm_query` (library built-ins) to navigate `context` slices. Verification is the `verify_against_rubric` primitive — called when the root judges useful; no fixed gate checkpoints. Run terminates via the library's `FINAL_VAR(<var>)` (no reserved `answer` variable). Time bounded three ways: `rlm`'s `max_timeout` (between iterations), per-primitive deadlines via `RunContext`, and a process-level wall-clock watchdog.

### UI ↔ backend run lifecycle
1. RLM lab UI (`frontend/src/components/lab/rlm/`) → `POST /api/demo` → backend `POST /runs` (or `/runs/upload` / `/runs/arxiv`).
2. Backend spawns the run subprocess, writes `demo_status.json`, returns initial state.
3. UI opens **SSE** via `/api/demo/events` → backend `/runs/<id>/events`.
4. SSE event types: RLM emits `repl_iteration`, `primitive_call`, `sub_rlm_spawned`, `sub_rlm_complete`, `run_complete`, `candidate_proposed`, `candidate_outcome`, `rubric_score`, `user_message`, `user_message_response`, `run_warning`, `iteration_heartbeat`; RDR adds `rdr_*` plus `cluster_started`, `cluster_artifact_emitted`, `cluster_scored`, `repair_dispatched`.
5. All events route through `sse_bridge.sanitize_iteration` — the single egress chokepoint that strips REPL locals and bounds stdout/stderr to metadata prefixes. The paper corpus never reaches the stream.

A `localStorage` pointer auto-resumes an in-flight run on a bare `/lab`.

### UI surfaces (chat steering, sidebar, leaderboard)
- **Chat steering:** real-time panel docked in the right sidebar. `POST /runs/<id>/messages` (`backend/routes/messages.py`) appends to `user_messages.jsonl` + emits `user_message`; root polls `check_user_messages()`, replies via `respond_to_user`. Both pure file I/O. System prompt instructs the root to avoid quoting PII-looking message contents verbatim.
- **Collapsible right sidebar:** 360px `NodeDetailSidebar` (`frontend/src/components/lab/rlm/node-detail-sidebar.tsx`); selection state lifted to `rlm-lab.tsx`; kind-specific content (paper/work/candidate/subrlm/baseline); `SteeringChat` docked at bottom; collapses to a 36px rail.
- **Leaderboard:** read-only `/leaderboard` ranks completed runs. `GET /leaderboard?paper&mode&order_by&limit` (`backend/routes/leaderboard.py`) aggregates `final_report.json` + `demo_status.json` at request time (no SQLite projection, not demo-gated). Frontend `frontend/src/app/leaderboard/`. Live rubric climb panel = enriched `RubricStrip` derived from existing SSE events.

### Where to look first
- HTTP layer: `backend/app.py` · CLI: `backend/cli.py` · RLM run entry: `backend/agents/rlm/run.py`
- Domain primitives: `backend/agents/rlm/primitives.py` · System prompt: `backend/agents/rlm/system_prompt.py`
- SSE bridge (egress chokepoint): `backend/agents/rlm/sse_bridge.py` · Subprocess spawn: `backend/services/events/live_runs.py`
- Paper ingestion: `backend/services/ingestion/parser/resolving_parser.py` (`ResolvingParser` — HTML > PDF > OCR cascade)
- Leaderboard: `backend/routes/leaderboard.py` + `frontend/src/app/leaderboard/`
- `backend/{agents,services}/` is named by function — read it directly.

## Sandboxes
`OPENRESEARCH_DEFAULT_SANDBOX` selects the backend: `local`, `docker` (network/memory/CPU controlled), or `runpod` (remote GPU pods; needs `OPENRESEARCH_RUNPOD_API_KEY` + `OPENRESEARCH_RUNPOD_SSH_KEY_PATH`). `start.sh` runs `scripts/runpod_check.sh` as a preflight when sandbox is `runpod` (bypass `START_SKIP_PREFLIGHT=1`). `START_FULL_SMOKE=1` boots a real pod — **costs money** (cents-scale on RTX 4090).

**Local Docker daemon is a prerequisite for every sandbox except `local` — including `runpod` (the repo default).** `build_environment` (`primitives.py:1090`) short-circuits to a no-op ONLY when `ctx.sandbox_mode == "local"`; for `docker`/`runpod`/`auto`/unknown it runs a real local `docker build` and raises `SandboxRuntimeError(backend_unavailable)` if no daemon is reachable. Under `runpod` the locally-built image is **never used on the pod** (the pod boots `OPENRESEARCH_RUNPOD_IMAGE` and runs over SSH in a per-run venv, `runpod_backend.py`) — so the local build is currently wasted work that still hard-requires Docker (**rough edge flagged 2026-05-30**; a future change could short-circuit `build_environment` under `runpod` too). Until then keep OrbStack/Docker up for RunPod runs, or use `--sandbox local`. Since 2026-05-30 `start.sh` preflight-checks `docker info` whenever the sandbox is not `local`. Full workflow + sandbox×prerequisite matrix + troubleshooting: `docs/runbooks/running-the-project.md`. A hollow `partial` with `SDK success-with-no-text` is an **auth/SDK** failure (credentials), NOT Docker — read the `/lab` detail-panel blockers to tell them apart.

**RunPod default image is `cuda-devel` (~18 GB)** — `config.runpod_image` defaults to `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`. The `-devel-` headers are required because common deps JIT-compile against CUDA at install/run time (bitsandbytes, flash-attn, deepspeed); an earlier switch to the lighter `-runtime-` image (~4 GB) silently broke a chained `pip install bitsandbytes && python train.py`, so it was reverted (commit `88c45b0`). Override with `OPENRESEARCH_RUNPOD_IMAGE=...-runtime-...` to get the lighter (~4 GB) image when a paper needs no CUDA JIT. **Two distinct knobs, intentionally different:** this pod image (what the pod actually boots) is `-devel-`, whereas the base of the *locally-generated* Dockerfile for `--sandbox runpod` (`environment_detective._RUNPOD_PYTORCH_BASE`) stays on the lighter `-runtime-` image — that local build is never used on the pod (see the prerequisite note above), so it is deliberately kept light.

### Local multi-GPU sandbox (docker-free, spec 2026-05-29)
For a host with local NVIDIA GPUs but no docker/RunPod (e.g. 8×A5000): the `local` sandbox runs experiments as host subprocesses with no image build. `SandboxMode.local` → `LocalProcessBackend`; `build_environment` is a no-op success under `local`; `SandboxConfig.gpu_device_ids` pins GPUs (`LocalProcessBackend`→`CUDA_VISIBLE_DEVICES`, `LocalDockerBackend`→`DeviceRequest(device_ids=...)`). `RunContext.gpu_device_ids` from `OPENRESEARCH_GPU_DEVICE_IDS`.
- Dynamic GPU scheduling: `backend/services/runtime/local_gpu_allocator.py` — nvidia-smi discovery, excludes GPUs another user holds (used-mem ≥ threshold OR any compute proc), `fcntl`-locked crash-safe leasing by GPU UUID, stale-lease reclaim by PID liveness.
- Batch scheduler: `.venv/bin/python scripts/batch_reproduce.py <paper>... --gpus-per-run 4|auto --model claude-oauth` — leases disjoint GPU sets, per-run venv (`runs/<id>/.venv`) with shared `HF_HOME`, auto-installs the per-run `requirements.txt`. `--gpus-per-run auto` gives one run all currently-free cards.
- **Setup note:** needs Python 3.12+ — `uv venv --python 3.12 .venv` (fresh clone ships no venv). Gotcha: `rlms==0.1.1` needs `pytest>=9.0.2`, conflicting with `requirements-dev.txt`'s `pytest>=8,<9`; install `backend/requirements.txt` alone (rlms pulls a working pytest 9).

## Stability & correctness invariants (rules — full incident histories in the cited specs/runbooks)
- **REPL safe-builtins** (`safe_builtins_patch.py`, imported atop `run.py`): restores `globals`/`locals` (pure namespace getters wrongly blocked by `rlm`'s `LocalREPL._SAFE_BUILTINS`); `safe_repl_traceback_patch.py` adds `traceback.format_exc()` to stderr. **DO NOT restore `eval`/`exec`/`compile`/`input`** — those are the genuine security boundary. Spec `2026-05-28-rlm-stability-remediation-design.md` (BUG-LR-011/012).
- **Forced-iteration policy** (`forced_iteration.py`, patches `LocalREPL._final_var` at import; per-run `ForcedIterationPolicy` via context manager around `rlm.completion`): when root calls `FINAL_VAR` but the latest `verify_against_rubric` score is `< target_score` **or `None`** AND `iteration_count < OPENRESEARCH_MIN_RUBRIC_ITERATIONS` (default 2), refuses `FINAL_VAR` and continues the loop (emits `run_warning` `code="forced_iteration"` naming score/target/iteration + next step). Wall-clock wins: bypassed when `ctx.remaining_s() <= 60`. `OPENRESEARCH_MIN_RUBRIC_ITERATIONS=0` disables. Paired with Lane G `rubric_guard.py` (`assert_metrics_schema` → `RubricGuardFailure` → next-iteration `repair_context`). Spec `2026-05-28-...` (BUG-LR-013).
- **Dockerfile shape guard** (`primitives.py::_validate_dockerfile_shape`): rejects any Dockerfile whose first non-blank/non-comment line isn't `FROM`/`ARG`/`# syntax=`. `implement_baseline` snapshots+restores a good Dockerfile if the sub-agent dumps prose (`code="dockerfile_shape_guard"`); `run_experiment` fail-fast with `failure_class="dockerfile_invalid"` (in `_RUN_EXPERIMENT_REPAIRABLE_FAILURES`). Tests `test_dockerfile_shape_guard.py` (BUG-NEW-042).
- **Run-status enum** (`live_runs.py:45`): `RunStatus` Literal is the single source of truth for `LiveRunState.status` and MUST include the terminal states `killed` (CLI signal handler) and `interrupted` (`run_liveness.sweep_orphaned_runs`) — else `_load_run` 500s on `/runs/latest` & `/runs/{id}` (the active-run guards already exclude both). Tests `test_live_runs_terminal_status.py` (BUG-NEW-045).
- **CLI signal handling** (`cli.py::_install_termination_handlers`): SIGTERM/SIGHUP atomically write `demo_status.json::status="killed"` (`killReason`) then `raise_signal(SIGINT)` for the graceful path. `_mark_demo_status_stopped/_failed` treat `killed` as terminal (won't overwrite). `_ACTIVE_PROJECT_ID` updated via `_set_active_project_id(...)` at every project-id-binding path. SIGKILL still unhandleable (`scripts/loops/kill_and_restart.sh` patches manually). Tests `test_termination_handler.py` (BUG-NEW-041).
- **claude-agent-sdk isolation:** every `ClaudeAgentOptions(...)` MUST pass `setting_sources=[]`, an explicit `mcp_servers` dict (`{}` if none), and a non-plan `permission_mode` — else the SDK loads the developer's `~/.claude/settings.json`, MCP servers, and plan-mode, contaminating the inner model. Three patched sites: `services/context/workspace/tools/rlm_query.py` (root completions), `agents/runtime/claude_runtime.py` (sub-agents), `hermes_audit/providers.py`. Defaults: root `bypassPermissions`, sub-agents inherit `agent.permission_mode` (also `bypassPermissions`, `agents/runtime/base.py:107`) (BUG-NEW-038).

## Demo gate
When `OPENRESEARCH_DEMO_SECRET` is set, run-start endpoints require a matching `X-Demo-Secret` header (`hmac.compare_digest`). Empty/unset disables the gate — that's local-dev behavior, not a bug.

## Universality — supporting "any ML paper"
Paper-agnostic by design. Per-paper customization surface:
- `--paper-hint <arxiv-id>` → `backend/agents/prompts/paper_hints.py::PAPER_HINTS` (optional; add an entry for algorithmic invariants the rubric should enforce — SDAR is the reference shape; missing entry = defaults).
- `--scope-spec` → operator-side narrow/expand of the hint's `default_scope` (models/datasets/seeds).
- `OPENRESEARCH_BASELINE_EXTRA_GUIDANCE` → free-text appended to the implementer prompt (CLI auto-merges `--paper-hint` guidance here, `cli.py:1032`).
- `paper_invariants.py` → deterministic invariants registry (model/env name canonicalization); falls back to no enforcement.
- `--vram-gb <n>` → manual VRAM override (still applies the headroom multiplier).

Out-of-the-box an arbitrary arXiv ID/PDF runs end-to-end: ingest → auto-generated rubric → 12 primitives → sandbox → `verify_against_rubric`. **Quality limit:** the auto-rubric is only as good as the LLM that writes it (no code fix changes this) — strict rubrics catch fakes but reject some good runs, loose rubrics give easy passes; for fidelity-critical papers add a `PAPER_HINTS` entry with regex-based `invariants`.

## Baseline test paper
[SDAR (arxiv 2605.15155)](https://arxiv.org/abs/2605.15155) — **Self-Distilled Agentic Reinforcement Learning** — the canonical baseline; stresses every dimension at once (3 Qwen sizes 1.7B/3B/7B, 3 environments ALFWorld+WebShop+Search-QA, GRPO RL + sigmoid-gated OPSD, 5 baselines, a fine-grained rubric whose leaves inspect the exact invariants `g_t=σ(β·Δ_t)`, stop-gradient on the gate, λ=0.1, β=10, real Qwen weights + real ALFWorld episodes). A surrogate cannot pass. **Smallest-two scope** (cost-bounded): pin to Qwen3-1.7B + Qwen2.5-3B on one 24–48 GB GPU via `OPENRESEARCH_BASELINE_EXTRA_GUIDANCE`. Full command + debug history + handoff prompt: `docs/runbooks/2026-05-23-sdar-baseline-handoff.md`.

## Reference docs (read whichever is relevant before non-trivial changes)
- `docs/design/rlm-pivot-brief.md` — canonical RLM-orchestrator architecture.
- `docs/runbooks/running-the-project.md` — run workflow + sandbox×prerequisite matrix + troubleshooting.
- `docs/runbooks/e2e-testing.md` — local end-to-end test & debug reference.
- `docs/runbooks/2026-05-23-sdar-baseline-handoff.md` — SDAR run command + debug cycle.
- `docs/superpowers/specs/2026-05-23-cleanup-condensation-leaderboard-design.md` — locked mode/cleanup decisions for the launch track.
- `docs/superpowers/specs/2026-05-23-dynamic-gpu-selection-design.md` — dynamic-GPU resolver + RunPod escalation.
- `docs/superpowers/specs/2026-05-28-rlm-stability-remediation-design.md` — five P0/P1 fixes (REPL safe-builtins, traceback surfacing, shell-env precedence BUG-LR-014, forced-iteration None-score, premature-exit detector).
- `docs/superpowers/specs/2026-05-28-subscription-cost-reduction-design.md` — sub-agent token-burn + retry-burst elimination.
- `docs/superpowers/specs/2026-05-30-intra-run-context-map-design.md` — context-map flag design.
- `docs/runbooks/2026-05-29-monitoring-loops.md` — the six loops for babysitting an SDAR sprint + the `BUG-NEW-NNN` doc-loop convention. Read before driving a retry sprint.

## Maintaining this doc
`system_overview.md` = the "why"; this file = the day-to-day. When you add a primitive, an SSE event type, a sandbox, or a fail-soft/fail-closed mode, update both. Don't document "what's where" — the code is named by function. Keep incident *narratives* in their spec/runbook/memory file; keep only the resulting *rule* here.

## Context-mode routing
Inherits context-mode MCP routing from the parent `CLAUDE.md`: use `ctx_batch_execute` / `ctx_execute` / `ctx_execute_file` for any command or file read producing >20 lines, and `ctx_fetch_and_index` instead of `WebFetch`/`curl`/`wget`. The parent file has the full blocked-vs-redirected table.
