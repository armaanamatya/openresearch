<!-- doc-meta: status=current; last-verified=2026-06-09 -->
# CLAUDE.md

> **Doc status:** Current · source-of-truth tier 2 (day-to-day) · last verified
> 2026-06-09. Policy: [`docs/policies/documentation.md`](docs/policies/documentation.md).

Guidance for Claude Code (claude.ai/code) working in this repo. This documents the **day-to-day**; `system_overview.md` and `docs/design/rlm-pivot-brief.md` document the "why" — read those before non-trivial architectural changes.

## Project: OpenResearch

An agent that reproduces research papers end-to-end: ingest paper → offload it as a REPL variable → RLM root model writes Python to understand claims, build an environment, implement and run a baseline, score against a rubric, and explore improvements → emit `final_report.{json,md}`.

## Common commands

### Backend (Python 3.14.x dev venv · 3.12 in the Docker image · floor 3.11, FastAPI)

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

Pytest config in `pyproject.toml` (`testpaths=["tests"]`, `pythonpath=["."]`).

# Dependency locking + lint (uv at /snap/bin/uv)
uv venv --python 3.12 && uv sync --frozen   # locked env matching Docker/CI (Python 3.12)
uvx ruff@0.15.16 check .                    # lint (E4/E7/E9/F defaults; config in pyproject.toml)
# pip + backend/requirements*.txt still works and the local dev venv may run newer Python (3.14).

### Frontend (Next.js 16, Node ≥20.19 <21 or ≥22.12)

```bash
cd frontend && npm ci
export OPENRESEARCH_BACKEND_URL=http://127.0.0.1:8000   # optional: proxy target (this IS the default; set only if the backend runs elsewhere)
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
   - `--model qwen3-coder` / `--model kimi-k2.5` → `OPENROUTER_API_KEY` (served via OpenRouter slugs in `models.py`)
   - `--model qwen3-coder-featherless` → `FEATHERLESS_API_KEY` (cheapest)
   - `--model azure` (aliases `azure-openai`, `gpt-4o-azure`) → `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_DEPLOYMENT`
2. **Sub-agents** (`implement_baseline` + other Sonnet calls, via `claude-agent-sdk`) accept either `ANTHROPIC_API_KEY` *or* OAuth via the local `claude` CLI subscription (subscription path is per-message-free; API-key path is per-token-billed).

**Gotchas (rules):**
- A no-credit `ANTHROPIC_API_KEY` does NOT fall back to OAuth — the SDK hits `400 credit balance too low` and every run dies at the first Sonnet sub-call with `cost_usd=0.0`. Working `claude --print "ping"` proves only the *subscription*, not the *API key*. Safest local dev: leave `ANTHROPIC_API_KEY=` empty, `claude login` once, use OpenAI/Featherless for the root (or `claude-oauth` for both surfaces on the subscription). See `.env` lines 14–18.
- **Shell wins over `.env`.** Credential injection in `run.py` reads `os.environ` first, falls back to Settings — so a stale shell export of `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `OPENRESEARCH_RUNPOD_API_KEY` silently shadows `.env`. Workaround: prefix the CLI with `env -u OPENAI_API_KEY` (or unset). A warn-only shell-shadows-.env validator runs on the CLI reproduce path (`cli.py::_warn_on_shell_env_override`, 6 suspect keys, advisory); the server boot path still has none (BUG-LR-014 partially remediated).
- **Cheapest local-dev cost model:** OpenAI root (~$1/run via `--model gpt-5`) + OAuth subscription for Sonnet sub-agents ($0) + RunPod COMMUNITY GPU (~$0.34/run); no Anthropic API balance needed. Zero-cost: `--model claude-oauth` for both surfaces (subject to subscription rate limits). macOS OAuth detection probes the Keychain via `_has_claude_subscription_oauth()`.

### Sandbox config
- `OPENRESEARCH_FORCE_SANDBOX` (default empty) **overrides per-run `--sandbox`** when non-empty — set to `docker`/`local` only to hard-pin a deployment; non-empty makes `--sandbox runpod` a no-op.
- `OPENRESEARCH_RUNPOD_CLOUD_TYPE` — `SECURE` (~$0.69/hr, **default** in `config.py`) vs `COMMUNITY` (~$0.34/hr RTX 4090; set this for the cheapest runs).
- See the **Sandboxes** section below for the full backend matrix + the Docker-daemon prerequisite.

### Feature flags (default-off unless noted; full design in the cited spec; A/B ≥3 paired SDAR runs before flipping any default)
- **gpt-5-mini navigation route** — routes hot-volume `rlm_query`/`llm_query` *navigation* sub-calls off the bundled-CLI transport (orphaned-child wedge, FM-001) onto OpenAI gpt-5-mini via openai/httpx. Quality-critical grader (`verify_against_rubric`/`propose_improvements`) stays on Sonnet-OAuth: keep `OPENRESEARCH_ACCELERATOR_SCOPE=navigation`, do NOT use `all`. Opt-in: `OPENRESEARCH_ACCELERATOR=endpoint`, `OPENRESEARCH_ACCELERATOR_BASE_URL=https://api.openai.com/v1`, `OPENRESEARCH_ACCELERATOR_MODEL=gpt-5-mini`, `OPENRESEARCH_SUBRLM_OPENAI_TIMEOUT_S=120` (honored by the client; unset → 300s default). Auth: `OPENRESEARCH_ACCELERATOR_API_KEY` is the endpoint credential (default `"local"`); for the `api.openai.com` host it falls back to `OPENAI_API_KEY` automatically when `OPENRESEARCH_ACCELERATOR_API_KEY` is unset (ACC-2). Rollback `OPENRESEARCH_ACCELERATOR=off`. Code: `backend/agents/rlm/accelerator.py`.
- **Intra-run context map (PEEK-lite)** — `OPENRESEARCH_CONTEXT_MAP=on` enables a free deterministic orientation cache: write hook in `binding.py` unions structured outputs of `understand_section`/`extract_hyperparameters`/`detect_environment` into `runs/<id>/rlm_state/context_map.json` (union-per-field, ≤40 fields/≤8 values/≤8KB incremental ceiling, thread-safe, fail-soft); root reads via `read_context_map()` and (only when the flag is on) is told to consult it before re-deriving. Navigation aid only, never a report source (the evidence-gate machinery — `OPENRESEARCH_METRIC_PROVENANCE`, default on, plus `OPENRESEARCH_METRICS_COMPLETENESS_CHECK` — remains the backstop). Off-state: write no-ops, read returns empty, prompt instruction omitted. Code: `backend/agents/rlm/context_map.py`; spec `2026-05-30-intra-run-context-map-design.md`.
- **Per-paper negative lessons (MUSE-lite)** — `OPENRESEARCH_NEGATIVE_LESSONS=1` enables cross-run per-paper failure memory: `run.py::_finalize` hook (`lesson_distiller.py::mine_lessons`) mines agent-correctable `failure_class` rows from `experiment_runs.jsonl` into `runs/_lessons/<arxiv_id>.json`; next run of the same `arxiv_id` injects active lessons into implementer guidance via `baseline_implementation.py::_negative_lessons_block`. Guardrails: recurrence-gated promotion (`occurrences>=2`, except `dockerfile_invalid` at 1), classifier-sourced `suggested_fix` (never agent prose), opportunity-aware retirement (staleness>=3, gated on phase reached). Capped ≤5/≤200 chars, advisory only. Needs `ctx.arxiv_id`. Spec `2026-05-30-...`.
- **Dynamic GPU selection** — `OPENRESEARCH_DYNAMIC_GPU=true` (**default ON**): root calls `resolve_gpu_requirements(...)` once/run to map paper hardware clues → RunPod SKU, cached to `runs/<id>/rlm_state/gpu_plan.json`, consumed by every `run_experiment`. On CUDA OOM, `run_experiment` escalates up the catalog ladder (≤`OPENRESEARCH_DYNAMIC_GPU_MAX_ESCALATIONS=2`). Caps: `OPENRESEARCH_MAX_GPU_USD_PER_HOUR=10.0` per-GPU + `OPENRESEARCH_MAX_RUN_GPU_USD=10.0` run-total (both float; `0` disables) via `RunBudget.check_run_gpu_usd`. Multi-GPU opt-in: `OPENRESEARCH_FORCE_SINGLE_GPU=false` (default true). `--vram-gb`→`OPENRESEARCH_VRAM_OVERRIDE_GB` bypasses the LLM estimate (still applies `OPENRESEARCH_DYNAMIC_GPU_HEADROOM=1.25`). SKU catalog `backend/services/runtime/gpu_catalog.py` (refresh quarterly). Events `gpu_resolved`/`gpu_escalated`/`gpu_fallback`. Spec `2026-05-23-dynamic-gpu-selection-design.md`.
- **BES competing candidates (both paths) + A/B harness (2026-06-11)** — master gate `REPROLAB_BES_ENABLED=1` + `REPROLAB_BES_CANDIDATES_PER_CLUSTER>=2` (+`REPROLAB_BES_SELECT_METRIC`) drives BES v1 on BOTH execution paths: RDR clusters (bundle papers; `rdr/controller._dispatch_competing_candidates`) and the RLM first `implement_baseline` (arXiv/PDF papers, which the hybrid bundle guard routes around RDR — `backend/agents/rlm/bes_rlm.py`): N isolated implementations (per-candidate angle appended to `REPROLAB_BASELINE_EXTRA_GUIDANCE`, Lane-A cache-busted), each snapshotted to `candidates/rlm_impl_<i>/` and statically graded by the leaf scorer (`degraded=False`, code-only — the SELECT signal; no GPU), winner restored into `code/`; repairs + re-entrant calls stay single-shot; pool persisted to `rlm_state/bes_candidates.json` + `candidate_proposed`/`candidate_outcome` SSE; wall-clock-guarded (`REPROLAB_BES_MIN_REMAINING_S`/`_CONTINUE_MIN_S`). **A/B harness:** every `final_report.json` carries an `experiment_arm` stamp ({arm: bes|control, ab_pair_id, bes flag/pool snapshot} — `REPROLAB_AB_ARM`/`REPROLAB_AB_PAIR_ID` override/pair); leaderboard rows + UI badge surface it; `scripts/ab_compare.py --paper <id>|--pair-id <id>` writes the deterministic paired report to `runs/_ab/<key>/ab_report.{md,json}`. **Adaptive gating:** `REPROLAB_BES_ADAPTIVE=1` (+`REPROLAB_BES_ADAPTIVE_SKIP_SCORE`, default 0.5) engages the RLM pool only on first-attempt / weak-history papers (the allcnn-ab pool discriminated weakly, 0.549 vs 0.557, because the seeded-best-attempt + champion rails already anchor quality when history exists); decision persisted to `rlm_state/bes_adaptive.json` + stamped into `experiment_arm`. Keep adaptive OFF on A/B arms (they need deterministic pool behaviour). Arm comparability: `REPROLAB_REUSE_RUBRIC=1` pins a pre-seeded `generated_rubric.json` (no per-run LLM rubric drift); `batch_reproduce --project-id-suffix <s>` gives each arm an independent full lineage (`register_project(project_id_override=...)` — a non-canonical `--project-id` now ingests instead of raising `UnknownProject`).

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

Root writes Python calling the **12 core domain primitives** (`backend/agents/rlm/primitives.py`) via `custom_tools`:
`understand_section`, `extract_hyperparameters`, `detect_environment`, `build_environment`, `plan_reproduction`, `implement_baseline`, `run_experiment`, `verify_against_rubric`, `propose_improvements`, `record_candidate_outcome`, `check_user_messages` (root calls at start of each iteration), `respond_to_user` (pure file I/O + `user_message_response` SSE).
The `PRIMITIVE_REGISTRY` also binds operational/optional helpers (so the bound `custom_tools` set is **17**, not 12): `heartbeat`, `recommend_next_tool`, `resolve_gpu_requirements` (dynamic-GPU), `codex_repair` (optional, default-off), and `read_context_map` (PEEK-lite; returns `{}` unless `OPENRESEARCH_CONTEXT_MAP` is on). Keep this count and `tests/rlm/test_registry.py`'s `EXPECTED` in sync when adding a primitive.

Root also calls `llm_query`/`rlm_query` (library built-ins) to navigate `context` slices. Verification is the `verify_against_rubric` primitive — called when the root judges useful; no fixed gate checkpoints. Run terminates via the library's `FINAL_VAR(<var>)` (no reserved `answer` variable). Time bounded three ways: `rlm`'s `max_timeout` (between iterations), per-primitive deadlines via `RunContext`, and a process-level wall-clock watchdog.

### UI ↔ backend run lifecycle
1. RLM lab UI (`frontend/src/components/lab/rlm/`) → `POST /api/demo` → backend `POST /runs` (or `/runs/upload` / `/runs/arxiv`).
2. Backend spawns the run subprocess, writes `demo_status.json`, returns initial state.
3. UI opens **SSE** via `/api/demo/events` → backend `/runs/<id>/events`.
4. SSE event types: RLM emits `repl_iteration`, `primitive_call`, `sub_rlm_spawned`, `sub_rlm_complete`, `run_complete`, `candidate_proposed`, `candidate_outcome`, `rubric_score`, `user_message`, `user_message_response`, `run_warning`, `iteration_heartbeat`; RDR adds `rdr_*` plus `cluster_started`, `cluster_artifact_emitted`, `cluster_scored`, `repair_dispatched`.
5. Iteration events route through `sse_bridge.sanitize_iteration` — the egress sanitizer that strips REPL locals and bounds stdout/stderr to metadata prefixes; corpus-derived fields pass through `redact_corpus`. (Terminal/control events such as `run_complete`/`run_fatal`/`run_interrupted` carry no corpus and are emitted directly, so `sanitize_iteration` is the per-iteration sanitizer, not a literal single function every event object passes through.) The paper corpus never reaches the stream.

A `localStorage` pointer auto-resumes an in-flight run on a bare `/lab`.

### UI surfaces (chat steering, sidebar, leaderboard)
- **Chat steering:** real-time panel docked in the right sidebar. `POST /runs/<id>/messages` (`backend/routes/messages.py`) appends to `user_messages.jsonl` + emits `user_message`; root polls `check_user_messages()`, replies via `respond_to_user`. Both pure file I/O. System prompt instructs the root to avoid quoting PII-looking message contents verbatim.
- **Collapsible right sidebar:** 360px `NodeDetailSidebar` (`frontend/src/components/lab/rlm/node-detail-sidebar.tsx`); selection state lifted to `rlm-lab.tsx`; kind-specific content (paper/work/candidate/subrlm/baseline); `SteeringChat` docked at bottom; collapses to a 36px rail.
- **Leaderboard + recent-runs panel:** read-only `/leaderboard` ranks runs; reachable from the left-sidebar nav and surfaced as a **recent-runs panel** atop the lab home (`frontend/src/components/lab/recent-runs-panel.tsx`, fed by `GET /leaderboard?order_by=finished_at&limit=`, then filtered to drop `interrupted` orphans + capped at 8). `GET /leaderboard?paper&mode&order_by&limit` (`backend/routes/leaderboard.py`) aggregates `final_report.json` + `demo_status.json` at request time (no SQLite projection, not demo-gated). Each project resolves its **best-scoring attempt** across top-level + `attempts/*` via `backend/services/runs/report_resolution.py` (`resolve_best_report`/`extract_scores` — normalizes nested `rubric.overall_score`/`compute_adjusted_score` + legacy flat top-level `rubric_score`; same extractor feeds run-detail `finalize_benchmark`); rows carry an honest `status` (stale `running`/`queued`→`completed` when a report exists) + `attempts` count, and `order_by=finished_at` is newest-first. The **recent-runs panel** rows carry Replay links (`?replay=<id>`) into the otherwise-orphaned replay surface (leaderboard rows link `/lab?projectId=`). Frontend `frontend/src/app/leaderboard/`. Live rubric climb panel = enriched `RubricStrip` derived from existing SSE events.

### Where to look first
- HTTP layer: `backend/app.py` · CLI: `backend/cli.py` · RLM run entry: `backend/agents/rlm/run.py`
- Domain primitives: `backend/agents/rlm/primitives.py` · System prompt: `backend/agents/rlm/system_prompt.py`
- SSE bridge (egress chokepoint): `backend/agents/rlm/sse_bridge.py` · Subprocess spawn: `backend/services/events/live_runs.py`
- Paper ingestion: `backend/services/ingestion/parser/resolving_parser.py` (`ResolvingParser` — HTML > PDF > OCR cascade)
- Leaderboard + recent-runs panel: `backend/routes/leaderboard.py` + `backend/services/runs/report_resolution.py` (best-attempt + score-schema resolution) + `frontend/src/app/leaderboard/` + `frontend/src/components/lab/recent-runs-panel.tsx`
- `backend/{agents,services}/` is named by function — read it directly.

## Sandboxes
`OPENRESEARCH_DEFAULT_SANDBOX` selects the backend: `local`, `docker` (network/memory/CPU controlled), or `runpod` (remote GPU pods; needs `OPENRESEARCH_RUNPOD_API_KEY` + `OPENRESEARCH_RUNPOD_SSH_KEY_PATH`). `start.sh` runs `scripts/runpod_check.sh` as a preflight when sandbox is `runpod` (bypass `START_SKIP_PREFLIGHT=1`). `START_FULL_SMOKE=1` boots a real pod — **costs money** (cents-scale on RTX 4090).

**Local Docker daemon is a prerequisite only for the `docker` and `auto` sandboxes.** `build_environment` (`primitives.py::build_environment`) short-circuits to a no-op for `local`, `runpod` (ported 2026-06-09 from the wedge-hardening line — the pod boots `OPENRESEARCH_RUNPOD_IMAGE` over SSH in a per-run venv, `runpod_backend.py`, so the old always-local build was wasted work that also hard-failed on unpullable base tags), and `azure` (image pre-baked in ACR); only `docker`/`auto`/unknown run a real local `docker build` and raise `SandboxRuntimeError(backend_unavailable)` without a daemon. `start.sh` preflight-warns on `docker info` for `docker`/`auto` sandboxes (non-fatal — a per-run `--sandbox` override changes the requirement). Hallucinated `runpod/` FROM tags are normalized to the configured image on the build paths — docker/auto (`_normalize_runpod_from_line`, unconditional after the shape guard); on the `runpod` sandbox the short-circuit itself is what makes a bad tag harmless. Full workflow + sandbox×prerequisite matrix + troubleshooting: `docs/runbooks/running-the-project.md`. A hollow `partial` with `SDK success-with-no-text` is an **auth/SDK** failure (credentials), NOT Docker — read the `/lab` detail-panel blockers to tell them apart.

**RunPod default image is `cuda-devel` (~18 GB)** — `config.runpod_image` defaults to `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`. The `-devel-` headers are required because common deps JIT-compile against CUDA at install/run time (bitsandbytes, flash-attn, deepspeed); an earlier switch to the lighter `-runtime-` image (~4 GB) silently broke a chained `pip install bitsandbytes && python train.py`, so it was reverted (commit `88c45b0`). Override with `OPENRESEARCH_RUNPOD_IMAGE=...-runtime-...` to get the lighter (~4 GB) image when a paper needs no CUDA JIT. **Two distinct knobs, intentionally different:** this pod image (what the pod actually boots) is `-devel-`, whereas the base of the *locally-generated* Dockerfile for `--sandbox runpod` (`environment_detective._RUNPOD_PYTORCH_BASE`) stays on the lighter `-runtime-` image — that local build is never used on the pod (see the prerequisite note above), so it is deliberately kept light.

### Local multi-GPU sandbox (docker-free, spec 2026-05-29)
For a host with local NVIDIA GPUs but no docker/RunPod (e.g. 8×A5000): the `local` sandbox runs experiments as host subprocesses with no image build. `SandboxMode.local` → `LocalProcessBackend`; `build_environment` is a no-op success under `local`; `SandboxConfig.gpu_device_ids` pins GPUs (`LocalProcessBackend`→`CUDA_VISIBLE_DEVICES`, `LocalDockerBackend`→`DeviceRequest(device_ids=...)`). `RunContext.gpu_device_ids` from `OPENRESEARCH_GPU_DEVICE_IDS`.
- Dynamic GPU scheduling: `backend/services/runtime/local_gpu_allocator.py` — nvidia-smi discovery, excludes GPUs another user holds (used-mem ≥ threshold OR any compute proc), `fcntl`-locked crash-safe leasing by GPU UUID, stale-lease reclaim by PID liveness.
- Batch scheduler: `.venv/bin/python scripts/batch_reproduce.py <paper>... --gpus-per-run 4|auto --model claude-oauth` — leases disjoint GPU sets, per-run venv (`runs/<id>/.venv`) with shared `HF_HOME`, auto-installs the per-run `requirements.txt`. `--gpus-per-run auto` gives one run all currently-free cards.
- **Setup note:** needs Python 3.12+ — `uv venv --python 3.12 .venv` (fresh clone ships no venv). (The old `requirements-dev.txt` pytest<9 pin that conflicted with `rlms==0.1.1` was fixed 2026-06-09 — installing both requirements files together resolves cleanly now.)
- **Distributed launch is always-on for multi-GPU** (BEH-1): when `run_experiment` sees >1 visible GPU it wraps training in an `accelerate`/FSDP `torchrun` launcher automatically (`primitives.py::_resolve_distributed_launch`) — there is no enable flag, only an opt-OUT `OPENRESEARCH_DISABLE_TORCHRUN_WRAP=1` for single-process debugging. So multi-GPU leasing implies sharded training, not just N independent cards.
- **Local torch core (env_pin, default ON, `local` only):** the harness owns a coherent cu121 torch/torchvision/torchaudio core — installs it from `OPENRESEARCH_LOCAL_TORCH_INDEX_URL` (default `…/whl/cu121`) **first** and strips the agent's conflicting re-pin from `requirements.txt` into `requirements.hardened.txt` (`primitives.py::_local_core_bootstrap_commands` → `env_pin.py`). `LocalProcessBackend` prepends the venv's bundled CUDA lib dirs — following the per-run venv's `_reprolab_base_inherit.pth` to the base `.venv` where the shared torch physically lives — to `LD_LIBRARY_PATH` so a system CUDA path can't shadow them. Fix for the 2026-06-07 All-Conv-Net collapse where the agent's `torch==2.2.0` downgraded the build → `libcupti.so.12` failed to dlopen → every experiment died at import (failure class `cuda_shlib_load`, repairable). Escape hatch: `OPENRESEARCH_DISABLE_ENV_PIN=1` (keep the agent's pins) or empty `OPENRESEARCH_LOCAL_TORCH_INDEX_URL` (raw `requirements.txt`). runpod/docker paths unaffected (the pin + `LD_LIBRARY_PATH` prepend are `local`-only). Spec/incident: `docs/archive/learn.md` (2026-06-07).

> Naming note: env vars are canonically `OPENRESEARCH_*` (what the shipped code reads). `config.py::_apply_legacy_env_aliases` bridges the legacy `REPROLAB_*` prefix to `OPENRESEARCH_*` **at import**, so an operator who exports either prefix before startup is fine — but a var set *after* import (e.g. a test monkeypatch) is not bridged, so use `OPENRESEARCH_*` in code and tests. The **cross-process cell-seam vars** (`OPENRESEARCH_CELL_*`) must stay byte-identical on both the runner-inject and `train_cell.py`-read ends (the child process does not import the bridge at all).

### One-GPU-per-cell execution + OOM remediation (spec 2026-05-31)
On a local/docker GPU backend, `run_experiment` routes the training matrix through `backend/agents/rlm/gpu_cell_runner.py::run_matrix` — **one subprocess per cell, pinned to one GPU** (`CUDA_VISIBLE_DEVICES=<one id>`), `min(free_gpus, cells)` in parallel, per-cell OOM shrink-retry (batch-scale 0.5→0.25 + grad-ckpt) — instead of the monolithic `commands.json`/`python train.py` path. This is the fix for the 2026-05-31 SDAR collapse where the agent's coordinator looped the whole matrix on `cuda:0` and OOM'd every cell. The route is **gated on the agent emitting `code/cells.json`** (the matrix manifest — the ONLY place the baseline axis lives; `ScopeSpec` is model×dataset×seed) **+ `code/train_cell.py`** (a single-cell trainer reading `OPENRESEARCH_CELL_PARAMS`/`OPENRESEARCH_CELL_OUTPUT_DIR` + argv `--cell-id`/`--output-dir`, honoring `OPENRESEARCH_CELL_BATCH_SCALE`/`OPENRESEARCH_CELL_GRAD_CHECKPOINT`, writing a FLAT per-cell `metrics.json`). Missing either manifest → **fail-soft to the legacy monolithic path** (non-SDAR papers unaffected). **Route retention (2026-06-11, default ON):** a repair pass that drops `cells.json` while keeping `train_cell.py` gets the manifest auto-restored from `rlm_state/last_cells.json` (+ `cells_manifest_restored` warning); cells-route failure classes (`cell_execution_error`/`cell_smoke_failed`), a missing trainer, or `REPROLAB_CELLS_ROUTE_RETENTION=0` degrade to an advisory `cells_manifest_dropped` warning — the monolithic regression is loud or impossible, never silent. Mutually exclusive with `_resolve_distributed_launch` (torchrun). The cell scheduler is shared with the Azure K8s path via `backend/agents/rlm/cell_scheduler.py`.

`describe_capacity(ctx) → GpuCapacity` (`backend/services/runtime/gpu_capacity.py`) is the backend-agnostic budget source (local real; runpod/brev real-ish; azure = AKS settings + `gpu_plan.json`, plan-aware). Three stages: **PREVENT** — `cell_matrix.capacity_gate` drops cells whose `est_vram_gb × headroom` exceeds the per-GPU budget → `scope.gaps`/`scope.models_skipped` (24 GB ⇒ smallest-two, never the 7B); `cell_matrix.dataset_url_preflight` HEAD-probes `cells.json::dataset_url` and drops a **confirmed** 404 → `scope.gaps` (fail-soft: a transient blip never drops a live env). **PLACEMENT** — `run_matrix`. **AGGREGATE** — `cell_matrix.aggregate_cell_metrics` synthesizes the canonical `per_model[model_key][env][baseline]` leaf shape the leaf scorer + ~8 postflight guards consume, persisted to `code/metrics.json` + `code/outputs/<run_id>/metrics.json` (treats `skipped`/`timeout` cells correctly for resume + reliability). **STOP** — when every run cell OOMs after shrink-exhaustion (`oom_shrink_exhausted`) or all cells are dropped (`capacity_exhausted`), `run_experiment` returns a terminal `stop_reason`; `run.py`'s tool wrapper calls `policy.note_terminal_failure(...)` so `forced_iteration` accepts the next `FINAL_VAR` — **stop + report, NO re-OOM loop** (do NOT reuse `silent_oom`, which correctly stays repairable for a *single* caught OOM). `final_report.json.stop_reason` carries the structured reason. `cell_matrix.py` is pure (stdlib-only) and unit-tested against a real on-disk `metrics.json` sample.

Copyable helpers the agent's generated code imports are auto-copied into `code/` by `run_with_sdk` (`gpu_cell_runner.py` + `cell_scheduler.py` + `sdar_env_base.py`, all zero-non-stdlib-dep — mirror of the `rubric_guard.py` emit pattern but a real file copy). `sdar_env_base.BaseEnv`/`AgenticEnv` are ABCs (`build_student_prompt`/`build_teacher_prompt`) so a `*Env` missing them fails at **construction** (cell start), not mid-grid; `preflight_ast._check_env_interface_contract` (recursive `rglob`, self-scoping) rejects a non-subclassing `*Env` *before* the grid runs — the fix for the 2026-05-31 ALFWorld `AttributeError` that zeroed 18 cells. Full design + per-component commit map: `docs/superpowers/specs/2026-05-31-oom-gpu-capacity-remediation-design.md` and `docs/runbooks/2026-05-31-oom-gpu-remediation-handoff.md`.

### Execution reliability — streaming + stall + finalize-on-timeout (spec 2026-06-08, `local`-scoped + flag-gated)
On the `local` sandbox, `run_experiment` STREAMS subprocess output to `runs/<id>/code/.exec_live.log` (+ a `.exec_heartbeat.json` sidecar; surfaced as periodic `experiment_progress` SSE) so a long GPU run is observable live (`tail -f code/.exec_live.log`) instead of buffering silent. A generous stall window (`OPENRESEARCH_EXPERIMENT_STALL_S`, default 3600s, `0` disables) kills only a GENUINE hang as `exec_stalled` — liveness = OR of {new output line, ckpt/`metrics.json` mtime bump, GPU-util (`OPENRESEARCH_EXPERIMENT_GPU_LIVENESS=1`), process-tree CPU-util}, distinct from the hard wall-clock budget cap (`exec_timeout`). A timeout/stall now FINALIZES completed work: the on-disk partial `metrics.json` is scored (failure class `partial_timeout`, repairable), never zeroed. All fail-soft; runpod/docker exec paths byte-for-byte unchanged. Full design: `docs/runbooks/2026-06-08-execution-reliability-redesign-handoff.md`.

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
- `docs/superpowers/specs/2026-05-31-oom-gpu-capacity-remediation-design.md` — one-GPU-per-cell `run_matrix` + capacity gate + OOM shrink-retry + STOP.
- `docs/superpowers/specs/2026-05-31-root-harness-hardening-design.md` — root-harness hardening invariants (Gap A/B, projection, blacklist).
- `docs/superpowers/specs/2026-06-03-azure-aks-gpu-backend-design.md` — Azure AKS GPU execution backend (Terraform/Helm IaC, in-Job OOM, blob I/O); `cloud-aware gpu_catalog`/`gpu_resolver` + `gpu_plan` threading.
- `docs/superpowers/specs/2026-06-07-rubric-scoring-harness-fairness-design.md` — scoring-fairness layer (anchored grader, provenance manifest, execution-smoke, theory-leaf exclusion).
- `docs/superpowers/specs/2026-05-28-rlm-stability-remediation-design.md` — five P0/P1 fixes (REPL safe-builtins, traceback surfacing, shell-env precedence BUG-LR-014, forced-iteration None-score, premature-exit detector).
- `docs/superpowers/specs/2026-05-28-subscription-cost-reduction-design.md` — sub-agent token-burn + retry-burst elimination.
- `docs/superpowers/specs/2026-05-30-intra-run-context-map-design.md` — context-map flag design.
- `docs/superpowers/specs/2026-05-30-rubric-scoring-fidelity-design.md` — scorer fidelity (Component A metrics-completeness gate + D events shipped; B/C pending).
- `docs/runbooks/2026-05-29-monitoring-loops.md` — the six loops for babysitting an SDAR sprint + the `BUG-NEW-NNN` doc-loop convention. Read before driving a retry sprint.

## Maintaining this doc
`system_overview.md` = the "why"; this file = the day-to-day. When you add a primitive, an SSE event type, a sandbox, or a fail-soft/fail-closed mode, update both. Don't document "what's where" — the code is named by function. Keep incident *narratives* in their spec/runbook/memory file; keep only the resulting *rule* here.

## Context-mode routing
Inherits context-mode MCP routing from the parent `CLAUDE.md`: use `ctx_batch_execute` / `ctx_execute` / `ctx_execute_file` for any command or file read producing >20 lines, and `ctx_fetch_and_index` instead of `WebFetch`/`curl`/`wget`. The parent file has the full blocked-vs-redirected table.
