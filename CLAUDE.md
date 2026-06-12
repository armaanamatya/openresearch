# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repo. This documents the **day-to-day**; `system_overview.md` and `docs/design/rlm-pivot-brief.md` document the "why" — read those before non-trivial architectural changes.

## Project: OpenResearch / ReproLab

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
export REPROLAB_BACKEND_URL=http://127.0.0.1:8000   # required: server-side proxy target
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

`REPROLAB_RLM_ROOT_MODEL` ∈ `gpt-5`, `qwen3-coder`, `kimi-k2.5`, `claude`, `claude-oauth` (defaults to GPT-5 when `OPENAI_API_KEY` set; falls back to `claude-oauth` when no API keys but `claude login` active).

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
- **Shell wins over `.env`.** Credential injection in `run.py` reads `os.environ` first, falls back to Settings — so a stale shell export of `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `REPROLAB_RUNPOD_API_KEY` silently shadows `.env`. Workaround: prefix the CLI with `env -u OPENAI_API_KEY` (or unset). Boot-time mismatch validator tracked as BUG-LR-014.
- **Cheapest local-dev cost model:** OpenAI root (~$1/run via `--model gpt-5`) + OAuth subscription for Sonnet sub-agents ($0) + RunPod COMMUNITY GPU (~$0.34/run); no Anthropic API balance needed. Zero-cost: `--model claude-oauth` for both surfaces (subject to subscription rate limits). macOS OAuth detection probes the Keychain via `_has_claude_subscription_oauth()`.

### Sandbox config
- `REPROLAB_FORCE_SANDBOX` (default empty) **overrides per-run `--sandbox`** when non-empty — set to `docker`/`local` only to hard-pin a deployment; non-empty makes `--sandbox runpod` a no-op.
- `REPROLAB_RUNPOD_CLOUD_TYPE` — `COMMUNITY` (~$0.34/hr RTX 4090, default) vs `SECURE` (~$0.69/hr).
- See the **Sandboxes** section below for the full backend matrix + the Docker-daemon prerequisite.

### Feature flags (default-off unless noted; full design in the cited spec; A/B ≥3 paired SDAR runs before flipping any default)
- **gpt-5-mini navigation route** — routes hot-volume `rlm_query`/`llm_query` *navigation* sub-calls off the bundled-CLI transport (orphaned-child wedge, FM-001) onto OpenAI gpt-5-mini via openai/httpx. Quality-critical grader (`verify_against_rubric`/`propose_improvements`) stays on Sonnet-OAuth: keep `REPROLAB_ACCELERATOR_SCOPE=navigation`, do NOT use `all`. Opt-in: `REPROLAB_ACCELERATOR=endpoint`, `REPROLAB_ACCELERATOR_BASE_URL=https://api.openai.com/v1`, `REPROLAB_ACCELERATOR_MODEL=gpt-5-mini`, `REPROLAB_SUBRLM_OPENAI_TIMEOUT_S=120` (uses `OPENAI_API_KEY` automatically). Rollback `REPROLAB_ACCELERATOR=off`. Code: `backend/agents/rlm/accelerator.py`.
- **Intra-run context map (PEEK-lite)** — `REPROLAB_CONTEXT_MAP=on` enables a free deterministic orientation cache: write hook in `binding.py` unions structured outputs of `understand_section`/`extract_hyperparameters`/`detect_environment` into `runs/<id>/rlm_state/context_map.json` (union-per-field, ≤40 fields/≤8 values/≤8KB incremental ceiling, thread-safe, fail-soft); root reads via `read_context_map()` and (only when the flag is on) is told to consult it before re-deriving. Navigation aid only, never a report source (the evidence gate `REPROLAB_EVIDENCE_GATE` remains the backstop). Off-state: write no-ops, read returns empty, prompt instruction omitted. Code: `backend/agents/rlm/context_map.py`; spec `2026-05-30-intra-run-context-map-design.md`.
- **Per-paper negative lessons (MUSE-lite)** — `REPROLAB_NEGATIVE_LESSONS=1` enables cross-run per-paper failure memory: `run.py::_finalize` hook (`lesson_distiller.py::mine_lessons`) mines agent-correctable `failure_class` rows from `experiment_runs.jsonl` into `runs/_lessons/<arxiv_id>.json`; next run of the same `arxiv_id` injects active lessons into implementer guidance via `baseline_implementation.py::_negative_lessons_block`. Guardrails: recurrence-gated promotion (`occurrences>=2`, except `dockerfile_invalid` at 1), classifier-sourced `suggested_fix` (never agent prose), opportunity-aware retirement (staleness>=3, gated on phase reached). Capped ≤5/≤200 chars, advisory only. Needs `ctx.arxiv_id`. Spec `2026-05-30-...`.
- **Dynamic GPU selection** — `REPROLAB_DYNAMIC_GPU=true` (**default ON**): root calls `resolve_gpu_requirements(...)` once/run to map paper hardware clues → RunPod SKU, cached to `runs/<id>/rlm_state/gpu_plan.json`, consumed by every `run_experiment`. On CUDA OOM, `run_experiment` escalates up the catalog ladder (≤`REPROLAB_DYNAMIC_GPU_MAX_ESCALATIONS=2`). Caps: `REPROLAB_MAX_GPU_USD_PER_HOUR=10.0` per-GPU + `REPROLAB_MAX_RUN_GPU_USD=10.0` run-total (both float; `0` disables) via `RunBudget.check_run_gpu_usd`. Multi-GPU opt-in: `REPROLAB_FORCE_SINGLE_GPU=false` (default true). `--vram-gb`→`REPROLAB_VRAM_OVERRIDE_GB` bypasses the LLM estimate (still applies `REPROLAB_DYNAMIC_GPU_HEADROOM=1.25`). SKU catalog `backend/services/runtime/gpu_catalog.py` (refresh quarterly). Events `gpu_resolved`/`gpu_escalated`/`gpu_fallback`. Spec `2026-05-23-dynamic-gpu-selection-design.md`.
- **BES competing candidates (both paths) + A/B harness (2026-06-11)** — master gate `REPROLAB_BES_ENABLED=1` + `REPROLAB_BES_CANDIDATES_PER_CLUSTER>=2` (+`REPROLAB_BES_SELECT_METRIC`) drives BES v1 on BOTH execution paths: RDR clusters (bundle papers; `rdr/controller._dispatch_competing_candidates`) and the RLM first `implement_baseline` (arXiv/PDF papers, which the hybrid bundle guard routes around RDR — `backend/agents/rlm/bes_rlm.py`): N isolated implementations (per-candidate angle appended to `REPROLAB_BASELINE_EXTRA_GUIDANCE`, Lane-A cache-busted), each snapshotted to `candidates/rlm_impl_<i>/` and statically graded by the leaf scorer (`degraded=False`, code-only — the SELECT signal; no GPU), winner restored into `code/`; repairs + re-entrant calls stay single-shot; pool persisted to `rlm_state/bes_candidates.json` + `candidate_proposed`/`candidate_outcome` SSE; wall-clock-guarded (`REPROLAB_BES_MIN_REMAINING_S`/`_CONTINUE_MIN_S`). **A/B harness:** every `final_report.json` carries an `experiment_arm` stamp ({arm: bes|control, ab_pair_id, bes flag/pool snapshot} — `REPROLAB_AB_ARM`/`REPROLAB_AB_PAIR_ID` override/pair); leaderboard rows + UI badge surface it; `scripts/ab_compare.py --paper <id>|--pair-id <id>` writes the deterministic paired report to `runs/_ab/<key>/ab_report.{md,json}`. **Adaptive gating:** `REPROLAB_BES_ADAPTIVE=1` (+`REPROLAB_BES_ADAPTIVE_SKIP_SCORE`, default 0.5) engages the RLM pool only on first-attempt / weak-history papers (the allcnn-ab pool discriminated weakly, 0.549 vs 0.557, because the seeded-best-attempt + champion rails already anchor quality when history exists); decision persisted to `rlm_state/bes_adaptive.json` + stamped into `experiment_arm`. Keep adaptive OFF on A/B arms (they need deterministic pool behaviour). **Leaf triage (2026-06-12, default ON):** `verify_against_rubric` deterministically classifies each weak leaf's justification against disk state into a cost-ordered repair plan (`render_artifact`/`provenance_gap`/`aggregation_gap` = NO retraining; `protocol_gap`/`result_quality` = one targeted re-run; grounded — a render directive is only issued when the data demonstrably exists) — attached to the verify result as `leaf_repair_plan`, persisted to `rlm_state/leaf_triage.json`, and injected into the next implementer prompt (hook 6.7). The automated form of the 06-11 operator steering that took Adam 0.0→0.716; zero LLM calls; `REPROLAB_LEAF_TRIAGE=0` disables. Code `backend/agents/rlm/leaf_triage.py`. Arm comparability: `REPROLAB_REUSE_RUBRIC=1` pins a pre-seeded `generated_rubric.json` (no per-run LLM rubric drift); `batch_reproduce --project-id-suffix <s>` gives each arm an independent full lineage (`register_project(project_id_override=...)` — a non-canonical `--project-id` now ingests instead of raising `UnknownProject`).

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

SQLite (`REPROLAB_DATABASE_URL`, default `sqlite:///reprolab.db`) is the event/persistence store (CQRS projections). Iteration state checkpointed atomically after each loop.

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
- **Leaderboard + recent-runs panel:** read-only `/leaderboard` ranks runs; reachable from the left-sidebar nav and surfaced as a **recent-runs panel** atop the lab home (`frontend/src/components/lab/recent-runs-panel.tsx`, fed by `GET /leaderboard?order_by=finished_at&limit=`, then filtered to drop `interrupted` orphans + capped at 8). `GET /leaderboard?paper&mode&order_by&limit` (`backend/routes/leaderboard.py`) aggregates `final_report.json` + `demo_status.json` at request time (no SQLite projection, not demo-gated). Each project resolves its **best-scoring attempt** across top-level + `attempts/*` via `backend/services/runs/report_resolution.py` (`resolve_best_report`/`extract_scores` — normalizes nested `rubric.overall_score`/`compute_adjusted_score` + legacy flat top-level `rubric_score`; same extractor feeds run-detail `finalize_benchmark`); rows carry an honest `status` (stale `running`/`queued`→`completed` when a report exists) + `attempts` count, and `order_by=finished_at` is newest-first. Per-row **Replay** links reach the otherwise-orphaned replay surface (`?replay=<id>`). Frontend `frontend/src/app/leaderboard/`. Live rubric climb panel = enriched `RubricStrip` derived from existing SSE events.

### Where to look first
- HTTP layer: `backend/app.py` · CLI: `backend/cli.py` · RLM run entry: `backend/agents/rlm/run.py`
- Domain primitives: `backend/agents/rlm/primitives.py` · System prompt: `backend/agents/rlm/system_prompt.py`
- SSE bridge (egress chokepoint): `backend/agents/rlm/sse_bridge.py` · Subprocess spawn: `backend/services/events/live_runs.py`
- Paper ingestion: `backend/services/ingestion/parser/resolving_parser.py` (`ResolvingParser` — HTML > PDF > OCR cascade)
- Leaderboard + recent-runs panel: `backend/routes/leaderboard.py` + `backend/services/runs/report_resolution.py` (best-attempt + score-schema resolution) + `frontend/src/app/leaderboard/` + `frontend/src/components/lab/recent-runs-panel.tsx`
- `backend/{agents,services}/` is named by function — read it directly.

## Sandboxes
`REPROLAB_DEFAULT_SANDBOX` selects the backend: `local`, `docker` (network/memory/CPU controlled), or `runpod` (remote GPU pods; needs `REPROLAB_RUNPOD_API_KEY` + `REPROLAB_RUNPOD_SSH_KEY_PATH`). `start.sh` runs `scripts/runpod_check.sh` as a preflight when sandbox is `runpod` (bypass `START_SKIP_PREFLIGHT=1`). `START_FULL_SMOKE=1` boots a real pod — **costs money** (cents-scale on RTX 4090).

**Local Docker daemon is a prerequisite for every sandbox except `local` — including `runpod` (the repo default).** `build_environment` (`primitives.py:1090`) short-circuits to a no-op ONLY when `ctx.sandbox_mode == "local"`; for `docker`/`runpod`/`auto`/unknown it runs a real local `docker build` and raises `SandboxRuntimeError(backend_unavailable)` if no daemon is reachable. Under `runpod` the locally-built image is **never used on the pod** (the pod boots `REPROLAB_RUNPOD_IMAGE` and runs over SSH in a per-run venv, `runpod_backend.py`) — so the local build is currently wasted work that still hard-requires Docker (**rough edge flagged 2026-05-30**; a future change could short-circuit `build_environment` under `runpod` too). Until then keep OrbStack/Docker up for RunPod runs, or use `--sandbox local`. Since 2026-05-30 `start.sh` preflight-checks `docker info` whenever the sandbox is not `local`. Full workflow + sandbox×prerequisite matrix + troubleshooting: `docs/runbooks/running-the-project.md`. A hollow `partial` with `SDK success-with-no-text` is an **auth/SDK** failure (credentials), NOT Docker — read the `/lab` detail-panel blockers to tell them apart.

**RunPod default image is `cuda-runtime` (~4 GB)** — reproduction code calls pre-built CUDA libs (PyTorch etc.), never compiles kernels, so `devel` (~18 GB) is unnecessary bloat. Default: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-runtime-ubuntu22.04`. Override with `REPROLAB_RUNPOD_IMAGE=...-devel-...` only if a paper actually compiles CUDA (rare).

The batch scheduler is `scripts/batch_reproduce.py` (run with the project venv: `.venv/bin/python scripts/batch_reproduce.py <paper>... --gpus-per-run 4|auto --model claude-oauth`). It leases disjoint GPU sets, creates a **per-run venv** (`runs/<id>/.venv`, isolated paper deps) with a **shared `HF_HOME`** weight cache, sets per-run torch/triton/xdg/mpl caches, launches `python -m backend.cli reproduce ... --project-id <id>`, monitors, releases leases (SIGINT-safe). `--gpus-per-run auto` gives one run all currently-free cards (heavy Qwen). For `local`, `build_environment` is a no-op and the per-run `requirements.txt` is auto-installed into the venv (mirrors the runpod bootstrap). **Local torch core (env_pin, default ON):** on the `local` sandbox the harness owns a coherent cu121 torch/torchvision/torchaudio core — it installs them from `REPROLAB_LOCAL_TORCH_INDEX_URL` (default `…/whl/cu121`) *first* and strips the agent's conflicting re-pin from `requirements.txt` into `requirements.hardened.txt` (`primitives.py::_local_core_bootstrap_commands` → `env_pin.py`). This is the fix for the 2026-06-07 All-Conv-Net collapse where the agent's `torch==2.2.0` downgraded the build → incoherent CUDA stack → `libcupti.so.12` failed to dlopen → every experiment died at import (failure class `cuda_shlib_load`, repairable). `LocalProcessBackend` prepends the venv's bundled CUDA lib dirs — following the per-run venv's `_reprolab_base_inherit.pth` to the base `.venv` where the shared torch physically lives — to `LD_LIBRARY_PATH` so a system CUDA path can't shadow them. **Escape hatch** for a paper that genuinely needs a non-cu121/older torch: `REPROLAB_DISABLE_ENV_PIN=1` (keep the agent's pins) or `REPROLAB_LOCAL_TORCH_INDEX_URL=` empty (disable the harness pin entirely, raw `requirements.txt`). runpod/docker paths are unaffected (the pin + the `LD_LIBRARY_PATH` prepend are `local`-only). **Setup note:** this host needs Python 3.12+ (repo `requires-python>=3.11`); provision with `uv venv --python 3.12 .venv` (a fresh clone ships no venv). Known repo gotcha: `rlms==0.1.1` depends on `pytest>=9.0.2`, which conflicts with `requirements-dev.txt`'s `pytest>=8,<9` — install `backend/requirements.txt` alone (rlms pulls a working pytest 9).

**RunPod default image is `cuda-runtime` (~4 GB).** Paper reproduction code calls pre-built CUDA libraries (PyTorch, etc.) and never invokes NVCC or compiles CUDA kernels, so the `devel` image (~18 GB) is unnecessary bloat — provisioning takes 5–10 extra minutes and costs $0.50–1.50 more per run. The default is now `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-runtime-ubuntu22.04`. If a paper actually compiles CUDA code (rare), override with `REPROLAB_RUNPOD_IMAGE=runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`.

### One-GPU-per-cell execution + OOM remediation (spec 2026-05-31)
On a local/docker GPU backend, `run_experiment` routes the training matrix through `backend/agents/rlm/gpu_cell_runner.py::run_matrix` — **one subprocess per cell, pinned to one GPU** (`CUDA_VISIBLE_DEVICES=<one id>`), `min(free_gpus, cells)` in parallel, per-cell OOM shrink-retry (batch-scale 0.5→0.25 + grad-ckpt) — instead of the monolithic `commands.json`/`python train.py` path. This is the fix for the 2026-05-31 SDAR collapse where the agent's coordinator looped the whole matrix on `cuda:0` and OOM'd every cell. The route is **gated on the agent emitting `code/cells.json`** (the matrix manifest — the ONLY place the baseline axis lives; `ScopeSpec` is model×dataset×seed) **+ `code/train_cell.py`** (a single-cell trainer reading `REPROLAB_CELL_PARAMS`/`REPROLAB_CELL_OUTPUT_DIR` + argv `--cell-id`/`--output-dir`, honoring `REPROLAB_CELL_BATCH_SCALE`/`REPROLAB_CELL_GRAD_CHECKPOINT`, writing a FLAT per-cell `metrics.json`). Missing either manifest → **fail-soft to the legacy monolithic path** (non-SDAR papers unaffected). **Route retention (2026-06-11, default ON):** a repair pass that drops `cells.json` while keeping `train_cell.py` gets the manifest auto-restored from `rlm_state/last_cells.json` (+ `cells_manifest_restored` warning); cells-route failure classes (`cell_execution_error`/`cell_smoke_failed`), a missing trainer, or `REPROLAB_CELLS_ROUTE_RETENTION=0` degrade to an advisory `cells_manifest_dropped` warning — the monolithic regression is loud or impossible, never silent. Mutually exclusive with `_resolve_distributed_launch` (torchrun).

`describe_capacity(ctx) → GpuCapacity` (`backend/services/runtime/gpu_capacity.py`) is the backend-agnostic budget source (local real; runpod/brev real-ish; azure = `NotImplementedError` stub). Three stages: **PREVENT** — `cell_matrix.capacity_gate` drops cells whose `est_vram_gb × headroom` exceeds the per-GPU budget → `scope.gaps`/`scope.models_skipped` (24 GB ⇒ smallest-two, never the 7B); `cell_matrix.dataset_url_preflight` HEAD-probes `cells.json::dataset_url` and drops a **confirmed** 404 → `scope.gaps` (fail-soft: a transient blip never drops a live env). **PLACEMENT** — `run_matrix`. **AGGREGATE** — `cell_matrix.aggregate_cell_metrics` synthesizes the canonical `per_model[model_key][env][baseline]` leaf shape the leaf scorer + ~8 postflight guards consume, persisted to `code/metrics.json` + `code/outputs/<run_id>/metrics.json`. **A ran cell never vanishes from the aggregate** (2026-06-09): `cell_matrix.normalize_cell_axes` derives missing `model_key`/`env`/`baseline` from agent-vocabulary synonyms (`dataset`/`variant`/... , falling back to the cell id) at manifest load, and aggregation derives as a second layer instead of silently skipping — the old skip turned a 14-cell paper-grade All-CNN run into `{status: failed, per_model: {}}`. Derivation emits a `cell_axes_derived` run_warning + `contract_warnings` on the result so the agent emits explicit axes next iteration. **STOP** — when every run cell OOMs after shrink-exhaustion (`oom_shrink_exhausted`) or all cells are dropped (`capacity_exhausted`), `run_experiment` returns a terminal `stop_reason`; `run.py`'s tool wrapper calls `policy.note_terminal_failure(...)` so `forced_iteration` accepts the next `FINAL_VAR` — **stop + report, NO re-OOM loop** (do NOT reuse `silent_oom`, which correctly stays repairable for a *single* caught OOM). `final_report.json.stop_reason` carries the structured reason. `cell_matrix.py` is pure (stdlib-only) and unit-tested against a real on-disk `metrics.json` sample.

Copyable helpers the agent's generated code imports are auto-copied into `code/` by `run_with_sdk` (`gpu_cell_runner.py` + `sdar_env_base.py`, both zero-non-stdlib-dep — mirror of the `rubric_guard.py` emit pattern but a real file copy). `sdar_env_base.BaseEnv` is an ABC (`build_student_prompt`/`build_teacher_prompt`) so a `*Env` missing them fails at **construction** (cell start), not mid-grid; `preflight_ast._check_env_interface_contract` (recursive `rglob`, self-scoping) rejects a non-subclassing `*Env` *before* the grid runs — the fix for the 2026-05-31 ALFWorld `AttributeError` that zeroed 18 cells. Full design + per-component commit map: `docs/superpowers/specs/2026-05-31-oom-gpu-capacity-remediation-design.md` and `docs/runbooks/2026-05-31-oom-gpu-remediation-handoff.md`.

### Execution reliability — streaming + stall + finalize-on-timeout (spec 2026-06-08, `local`-scoped + flag-gated)
On the `local` sandbox, `run_experiment` STREAMS subprocess output to `runs/<id>/code/.exec_live.log` (+ a `.exec_heartbeat.json` sidecar; surfaced as periodic `experiment_progress` SSE) so a long GPU run is observable live (`tail -f code/.exec_live.log`) instead of buffering silent. A generous stall window (`REPROLAB_EXPERIMENT_STALL_S`, default 3600s, `0` disables) kills only a GENUINE hang as `exec_stalled` — liveness = OR of {new output line, ckpt/`metrics.json` mtime bump, GPU-util (`REPROLAB_EXPERIMENT_GPU_LIVENESS=1`), process-tree CPU-util}, distinct from the hard wall-clock budget cap (`exec_timeout`). A timeout/stall now FINALIZES completed work: the on-disk partial `metrics.json` is scored (failure class `partial_timeout`, repairable), never zeroed. All fail-soft; runpod/docker exec paths byte-for-byte unchanged. Full design: `docs/runbooks/2026-06-08-execution-reliability-redesign-handoff.md`.

**Hard-stop salvage (2026-06-09):** the watchdog/SIGTERM finalizer (`run.py::_hard_stop_with_report`) salvages the run's earned evidence instead of shipping a scoreless `failed`: best-of-run rubric floor from dashboard events, verdict reconciled (≤`partial`), structured `stop_reason` (`wall_clock_watchdog`/`sigterm`), and `write_final_report_rlm`'s rubric_evaluation merge fills `overall_score`/`target_score`/`meets_target` when the report carries the unscored `None` defaults. Safe because attempt isolation now archives `rubric_evaluation.json`/`rubric_tree.json` + telemetry sidecars per attempt (`attempt_isolation._ARCHIVE_FILES`) — an existing eval file always belongs to the current attempt. Disk hygiene: `scripts/gc_runs.py` (dry-run by default) reclaims per-cell `datasets/` copies + weight files; `.preserved` runs are skipped.

## REPL sandbox safe-builtins patch (spec 2026-05-28)
`rlm`'s `LocalREPL._SAFE_BUILTINS` blocks `eval/exec/compile/input` (correctly — code-execution surface) AND `globals/locals` (incorrectly — pure namespace getters). The latter breaks any root-model script that does `globals().get("report_state", {...})` to persist state across iterations: it raises a bare `TypeError: 'NoneType' object is not callable` that the model cannot diagnose. `backend/agents/rlm/safe_builtins_patch.py` restores `globals`/`locals` at import time; `safe_repl_traceback_patch.py` extends `LocalREPL.execute_code` to include `traceback.format_exc()` in stderr so future REPL failures name the actual line. Both are imported at the top of `backend/agents/rlm/run.py` (mirror of `forced_iteration` pattern). DO NOT restore eval/exec/compile/input — those are the genuine security boundary. Full design + the 2026-05-28 SDAR death-spiral that motivated this: `docs/superpowers/specs/2026-05-28-rlm-stability-remediation-design.md` (BUG-LR-011/012).

## Forced-iteration policy (Lane H, spec 2026-05-24; extended 2026-05-28)
When the root model calls `FINAL_VAR` but the latest `verify_against_rubric` returned `overall_score < target_score` AND `iteration_count < REPROLAB_MIN_RUBRIC_ITERATIONS` (default 2), the orchestrator REFUSES the `FINAL_VAR` and forces the root-loop to continue. **Extended 2026-05-28:** the policy ALSO refuses when `latest_rubric_score is None` AND `iteration_count < REPROLAB_MIN_RUBRIC_ITERATIONS` — a root that has not even called `verify_against_rubric` has done strictly less work than one that scored zero, and should not be allowed to ship a `partial` report (BUG-LR-013). Mechanically: `backend/agents/rlm/forced_iteration.py` patches `rlm.environments.local_repl.LocalREPL._final_var` once at import time; a per-run `ForcedIterationPolicy` is pushed onto a thread-local stack via the `forced_iteration_policy` context manager around `rlm.completion`. Each refusal emits a `run_warning` event with `code="forced_iteration"` and a message naming the score/target/iteration numbers and the concrete next step (propose_improvements + implement_baseline + run_experiment). Wall-clock takes precedence: when `ctx.remaining_s() <= 60s` the policy bypasses and the partial report ships. `REPROLAB_MIN_RUBRIC_ITERATIONS=0` disables the policy entirely. Paired with Lane G (`backend/agents/rlm/rubric_guard.py` — the agent-written train.py calls `assert_metrics_schema(...)` at end-of-script; a missing key / artifact raises `RubricGuardFailure` whose JSON-shaped message becomes the next iteration's `repair_context`).

## Demo gate
When `REPROLAB_DEMO_SECRET` is set, run-start endpoints require a matching `X-Demo-Secret` header (`hmac.compare_digest`). Empty/unset disables the gate — that's local-dev behavior, not a bug.

## Universality — supporting "any ML paper"
Paper-agnostic by design. Per-paper customization surface:
- `--paper-hint <arxiv-id>` → `backend/agents/prompts/paper_hints.py::PAPER_HINTS` (optional; add an entry for algorithmic invariants the rubric should enforce — SDAR is the reference shape; missing entry = defaults).
- `--scope-spec` → operator-side narrow/expand of the hint's `default_scope` (models/datasets/seeds).
- `REPROLAB_BASELINE_EXTRA_GUIDANCE` → free-text appended to the implementer prompt (CLI auto-merges `--paper-hint` guidance here, `cli.py:1032`).
- `paper_invariants.py` → deterministic invariants registry (model/env name canonicalization); falls back to no enforcement.
- `--vram-gb <n>` → manual VRAM override (still applies the headroom multiplier).

Out-of-the-box an arbitrary arXiv ID/PDF runs end-to-end: ingest → auto-generated rubric → 12 primitives → sandbox → `verify_against_rubric`. **Quality limit:** the auto-rubric is only as good as the LLM that writes it (no code fix changes this) — strict rubrics catch fakes but reject some good runs, loose rubrics give easy passes; for fidelity-critical papers add a `PAPER_HINTS` entry with regex-based `invariants`.

## Baseline test paper
[SDAR (arxiv 2605.15155)](https://arxiv.org/abs/2605.15155) — **Self-Distilled Agentic Reinforcement Learning** — the canonical baseline; stresses every dimension at once (3 Qwen sizes 1.7B/3B/7B, 3 environments ALFWorld+WebShop+Search-QA, GRPO RL + sigmoid-gated OPSD, 5 baselines, a fine-grained rubric whose leaves inspect the exact invariants `g_t=σ(β·Δ_t)`, stop-gradient on the gate, λ=0.1, β=10, real Qwen weights + real ALFWorld episodes). A surrogate cannot pass. **Smallest-two scope** (cost-bounded): pin to Qwen3-1.7B + Qwen2.5-3B on one 24–48 GB GPU via `REPROLAB_BASELINE_EXTRA_GUIDANCE`. Full command + debug history + handoff prompt: `docs/runbooks/2026-05-23-sdar-baseline-handoff.md`.

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
