<!-- doc-meta: status=current; last-verified=2026-07-07 -->
# Handoff — SDAR-on-GCP (GKE, Foundry root+executor) live run + `/lab` picker feature

> **Date:** 2026-07-07 · **Status:** IN PROGRESS — run launched this session.
> Self-contained: a fresh zero-context Claude Code session should be able to **monitor and
> continue** the live GKE SDAR reproduction, and understand the whole `/lab` picker + Foundry
> setup, from this doc alone.

---

## 0. Status header (5-second read)

- The `/lab` provider/sandbox/Foundry picker feature is **DONE + committed locally** (commit
  `a279046f`, **NOT pushed**). It wires `root_provider` → root model, `subagent_auth` → executor
  env, adds Foundry + GCP/Azure sandbox options, and makes Foundry the pre-selected default.
  Verified: 296 backend tests, 57 frontend tests, tsc/eslint/ruff clean, visual screenshot
  confirmed. (Re-spot-checked this session: `tests/services/events/test_picker_overrides.py` +
  `tests/routes/test_auth_status.py` = 30/30 pass.)
- A GKE Job SDAR reproduction was **launched this session** on 4×A100 via `sandbox=gcp` with
  Foundry root (`opus-foundry`) + Foundry executor (`sonnet-foundry`) + skills, synced to the
  local `/lab` UI. **Live run: `project_id=prj_23f04429cd3beaf7`**, launched via the `/lab`
  arxiv endpoint (POST `/runs/arxiv`); watch it at
  **http://localhost:3000/lab?projectId=prj_23f04429cd3beaf7**. As of handoff: `running`,
  `opus-foundry` root driving (`understand_section`/`detect_environment` = ok), `skills_selected`
  fired, **0 temperature errors**, GKE A100 Job **not yet dispatched** (still in understand/plan;
  **$0 GPU** so far — the pod autoscales once the root reaches a training cell).

---

## 1. Model delegation policy (REQUIRED)

- Use **Opus** for orchestration, design, debugging, analysis, diff review, and every
  money-critical / cloud-critical decision (this run spends real GPU-hour + LLM-token budget).
- Use **Sonnet** for all non-critical, well-specified, mechanical tasks — drafting, boilerplate,
  test extension, recon reads.
- **Work in parallel and be efficient without sacrificing quality**: launch independent subagents
  concurrently, delegate non-critical work to Sonnet, and have Opus review every diff/decision
  before it lands.

---

## 2. Environment (VERIFIED this session)

- **Repo:** `/home/abheekp/openresearch` on branch `main`. Tip is the `/lab` picker feature commit
  `a279046f` (local only); the prior `deepinvent/main` tip is `fcba19ca`.
- **Foundry Anthropic endpoint LIVE:** base
  `https://appradhann-4738-resource.services.ai.azure.com/anthropic` ; `opus-foundry` →
  `claude-opus-4-8` and `sonnet-foundry` → `claude-sonnet-5` **both** returned live completions.
  Auth = `AZURE_FOUNDRY_API_KEY` (OAuth-free). `has_foundry_anthropic_credentials()` → `True`.
- `.env` has `AZURE_FOUNDRY_API_KEY` / `AZURE_FOUNDRY_ENDPOINT` /
  `AZURE_FOUNDRY_DEPLOYMENT` (= `grok-4.3`, the **OpenAI-compat** side — **not** used by the Claude
  `opus-foundry`/`sonnet-foundry` path, which talks to the `…/anthropic` endpoint form instead).
- **GCP:** project `deepinvent-ext-ut`. GKE cluster `openresearch-gpu` @ `us-central1-a` RUNNING.
  A100-80 scale-to-zero node pools `a100-80-1g` / `-2g` / `-4g` (labels
  `reprolab/sku=gcp_a100_80` / `_80x2` / `_80x4`; quota 4×A100-80GB total in `us-central1`).
  Bucket `gs://deepinvent-ext-ut-sdar-runs`. Base image
  `us-central1-docker.pkg.dev/deepinvent-ext-ut/reprolab/gke-cell-base:v1`. Kubernetes namespace
  for Job submission = **`reprolab`** (`backend/config.py` `gcp_namespace` default, matches the
  base-image path). `bash scripts/gcp_ready.sh` = the readiness preflight (read-only + a local
  `get-credentials` kubeconfig write; last run READY). `gke-gcloud-auth-plugin` lives at
  `~/.local/bin/gke-gcloud-auth-plugin` (must be on `PATH` — it isn't `apt`/`gcloud components`
  installable on this box; see the sibling handoff §6 for the no-sudo install method).
- All SDAR VMs (`sdar-2model-a` etc.) are **TERMINATED** — the VM path (§7 below) is **not** what
  this run uses.

---

## 3. What was launched — the GKE SDAR run

- **Path:** `sandbox=gcp` → a GKE Job on the `a100-80-4g` pool (4×A100-80), a **from-scratch**
  reproduction (no warm `/mnt/sdar-cache`), **not** the VM execute-mode path (§7).
- **Config:** `opus-foundry` root, `sonnet-foundry` executor/grader/verifier,
  `OPENRESEARCH_SKILLS=1` + `OPENRESEARCH_SKILL_SELECT=1`, guards on. **GPU cap $35**
  (`OPENRESEARCH_MAX_RUN_GPU_USD=35.0`, added to the autonomous run-spec this session); LLM spend
  uncapped on the autonomous UI path (the run-spec guards + budget govern).
- **Launch command (actual):** launched **through the `/lab` UI path** — a POST to the backend
  arxiv endpoint (the same one the UI "Begin" button hits), with the picker configs:
    ```bash
    curl -s -X POST http://127.0.0.1:8000/runs/arxiv -H 'Content-Type: application/json' -d '{
      "url":"https://arxiv.org/abs/2605.15155",
      "autonomous":true, "gpu_count":4,
      "root_provider":"foundry", "subagent_auth":"foundry", "sandbox":"gcp", "model":"opus"
    }'
    ```
    `autonomous:true` forces `opus-foundry` root + `sandbox=gcp` + the autonomous run-spec (roles
    = sonnet-foundry, skills, guards, $35 GPU cap); `gpu_count:4` → the `a100-80-4g` pool. Skills
    ride the run-spec env, not a flag. The equivalent **pure-CLI** form (shares `runs_root` with
    the running backend so the UI still shows it live):
    ```bash
    OPENRESEARCH_SKILLS=1 OPENRESEARCH_SKILL_SELECT=1 .venv/bin/python -m backend.cli reproduce \
      2605.15155 --sandbox gcp --gpu-count 4 --model opus-foundry \
      --models executor=sonnet-foundry,grader=sonnet-foundry,verifier=sonnet-foundry \
      --run-spec configs/autonomous_reproduction_run_spec.json --max-run-gpu-usd 35
    ```
- **project_id:** `prj_23f04429cd3beaf7` (the live run as of handoff). **NOTE:** because the
  launch went through the **`/runs/arxiv` endpoint**, the backend stages a fresh *timestamped
  upload* copy of the PDF per POST and hashes THAT path — so the id is NOT the bundled
  `papers/sdar.pdf` hash (`prj_77d3388db8e90d24`, which only applies to a pure-CLI
  `reproduce 2605.15155`). Each re-POST yields a new id; always read the real `project_id=…` from
  the run's `runner.stderr.log` / the POST response. Superseded attempts this session:
  `prj_192cf34aaa49f4e9` (failed — missing `google-cloud-storage`), `prj_13f7eef55bd0b55c`
  (stopped — pre-temperature-fix, rubric-less).
- **runs_root:** `/home/abheekp/openresearch/runs` (confirmed — neither the backend nor the run
  subprocess sets `OPENRESEARCH_RUNS_ROOT`, so both use the default). Must match the running
  backend's `runs_root`. Default for both CLI
  (`_REPRODUCE_DEFAULTS["runs_root"]`, `backend/cli.py:776`) and backend
  (`FileLiveRunService.__init__`, `backend/services/events/live_runs.py`) is
  `<repo_root>/runs`, i.e. `/home/abheekp/openresearch/runs`, unless `OPENRESEARCH_RUNS_ROOT` is
  set to something else in `.env`/the shell — confirm neither side has an override before
  assuming the default.

---

## 4. How it's synced + displayed in the `/lab` UI (VERIFIED mechanics)

- The CLI writes `runs/<project_id>/{demo_status.json, dashboard_events.jsonl,
  final_report.*}`. `project_id` is a deterministic hash
  (`backend/services/ingestion/intake/service.py:252-272`), printed to stderr
  (`backend/cli.py:1942`, `print(f"             project_id={project_id}", file=sys.stderr)`).
- The UI shows it live because backend + CLI share the **same** `runs_root`: `GET /runs`
  (`backend/app.py:724`) globs `runs_root/*/demo_status.json` for the recent-runs panel (comment:
  *"FileLiveRunService is already filesystem-backed, so CLI-created runs … are included by
  default"*), and `/lab?projectId=<id>` opens SSE via `/api/demo/events` → backend
  `/runs/{project_id}/events` (`backend/app.py:867`), which tails `dashboard_events.jsonl`
  (`FileLiveRunService.stream_events`, `backend/services/events/live_runs.py:879`).
- **REQUIRED invariant:** CLI and backend MUST share `runs_root`
  (`OPENRESEARCH_RUNS_ROOT`, or both default to `<repo>/runs`). Full-stack: `./start.sh` (backend
  `:8000` + frontend `:3000`) OR whatever dev servers are already running.
- **Monitoring commands** (fill in `<id>` = the project_id from §3):
  ```bash
  curl -s localhost:8000/runs/<id> | jq .status
  tail -f runs/<id>/dashboard_events.jsonl
  # UI:
  open http://localhost:3000/lab?projectId=<id>
  # GKE Job (namespace is "reprolab", per backend/config.py gcp_namespace default):
  kubectl get jobs,pods -n reprolab
  kubectl logs -n reprolab -l job-name=<job-name> --tail=100 -f
  ```

---

## 5. The `/lab` picker feature (context — commit `a279046f`, local, not pushed)

Files touched: `backend/agents/runtime/factory.py`, `backend/services/events/live_runs.py`,
`frontend/src/components/lab/upload-view.{tsx,css,test.tsx}`,
`frontend/src/lib/demo/demo-run-types.ts`, `tests/routes/test_auth_status.py`,
`tests/services/events/test_picker_overrides.py`.

- **`apply_picker_overrides(request)`** (`live_runs.py:579-612`) maps `root_provider` →
  `request.model` (which rides the `config` dict to the child's `resolve_root_model`):
  `root_provider=="foundry"` → `opus-foundry` if the current model string denotes the Opus tier
  (substring match on `"opus"`) else `sonnet-foundry`; `"anthropic_oauth"`→`claude-oauth`;
  `"openai_api"`→`gpt-5`; `"azure_openai"`→`azure-gpt-4o`; `"featherless"`→
  `qwen3-coder-featherless`; `"anthropic_api"` is a deliberate no-op (leaves `request.model`
  untouched — the existing opus/sonnet → `ANTHROPIC_API_KEY` path already handles it). **Identity**
  (no-op) when `root_provider` is `None` (byte-identical off-state) **or** a `run_spec` is pinned
  (an explicit run-spec wins over the picker). Called in the override chain *before*
  `apply_autonomous_profile_override`, which still runs last and still wins (forces
  `model="opus-foundry", sandbox="gcp"`) — the picker never beats autonomous mode.
- **`_subprocess_env`** (`live_runs.py:629-721`) threads `subagent_auth` as **env**, not
  `request.model` (it can't ride the root-model slot — it targets sub-agents):
  `"foundry"` → merges `{"executor": "sonnet-foundry"}` into any existing
  `OPENRESEARCH_ROLE_MODELS` (via `_merge_role_models`, which parses either JSON or the `k=v,k=v`
  CLI form so a `.env`/run-spec pin for other roles survives); `"anthropic_oauth"` →
  `OPENRESEARCH_LLM_AUTH_STRATEGY=oauth_only`; `"anthropic_api"` →
  `OPENRESEARCH_LLM_AUTH_STRATEGY=api_only`. Only applied when `subagent_auth` is explicitly set
  (unset ⇒ byte-identical to pre-wiring).
- **`aggregate_auth_status()`** (`factory.py:454-534`) adds a `"foundry"` entry to `providers` +
  `subagent_auth`, and makes `defaults.root_provider` / `defaults.subagent_auth` = `"foundry"`
  whenever `has_foundry_credentials()` is true (Foundry leads the priority chain ahead of
  anthropic_oauth/anthropic_api/openai_api/azure_openai/featherless).
- **Frontend** (`upload-view.tsx`): adds Foundry to the LLM-provider and sub-agent-auth radios plus
  GCP/Azure to `SANDBOX_OPTIONS` (`"GPU on GCP"` / `"GPU on Azure"`), each with a plain-language
  caption, a live selected-detail line, and a **"Recommended"** pill shown next to Foundry when it
  is available.

---

## 6. CLI ↔ `/lab` config equivalence (VERIFIED)

| `/lab` UI control | CLI equivalent |
|---|---|
| Sandbox = "GPU on GCP" | `--sandbox gcp` (`gke` is an accepted alias) |
| LLM provider = Foundry, Model = Opus | `--model opus-foundry` |
| LLM provider = Foundry, Model = Sonnet (default) | `--model sonnet-foundry` |
| Sub-agent auth = Foundry | `--models executor=sonnet-foundry` (env `OPENRESEARCH_ROLE_MODELS`) |
| GPU count = 4 | `--gpu-count 4` (env `OPENRESEARCH_GPU_COUNT`) |
| Skills toggle (on) | no CLI flag — env `OPENRESEARCH_SKILLS=1` + `OPENRESEARCH_SKILL_SELECT=1` |
| Budget | `--max-usd` (LLM token spend) + `--max-run-gpu-usd` (GPU spend cap) |

Note: SDAR (`2605.15155`) auto-resolves through the bundled-paper registry
(`backend/services/ingestion/paper_registry.py`) to the in-repo `papers/sdar.pdf` **and**
auto-applies `--paper-hint 2605.15155` when the flag is otherwise unset (comment in
`backend/cli.py`: *"SDAR's future-dated arXiv id does NOT fetch → was a degraded 469-char run"* —
this registry exists specifically so `reproduce sdar`/`reproduce 2605.15155` is fully
self-configuring offline).

---

## 7. The alternative VM path (for reference — "as we have been doing")

Canonical proven path = `scripts/sdar_phase1_foundry.sh` run **ON** VM `sdar-2model-a`
(`a2-ultragpu-4g` = 4×A100-80, `us-central1-a`, warm `/mnt/sdar-cache`), `--sandbox local`,
run-spec `configs/sdar_execute_run_spec.json` (`opus-foundry` root, all sub-roles
`sonnet-foundry`, `OPENRESEARCH_SKILLS`+`OPENRESEARCH_SKILL_SELECT=1`, execute mode, all guards on
— `OPENRESEARCH_LIFECYCLE_PRIMARY`, `_ENV_LIVENESS_GATE`, `_EVAL_PROVENANCE_GUARD`,
`_ZERO_METRICS_GUARD`, `_NO_LEARNING_SIGNAL_GATE`, `_EXTERNAL_VALIDATOR`, `_CELL_RESUME_AUTO`,
`OPENRESEARCH_MAX_RUN_GPU_USD=400`), cells `configs/sdar_execute_cells_phase1.json` (Search-QA
3B, `Qwen2.5-3B-Instruct`, authors' verl trainer). The script's exit trap always uploads
`runs/<pid>/` to `gs://deepinvent-ext-ut-sdar-runs` **before** `sudo shutdown -h now` — it always
self-stops, no debug escape hatch (unlike `sdar_gcp_run.sh`'s `NO_AUTOSTOP`).

**Use this as the fallback** if the from-scratch GKE run stalls on environment-build (no warm
cache means installing SDAR's conda envs / HF weights / Search index from scratch on first use —
this is the main risk of the GKE path vs. the VM's pre-staged disk). VM is currently
**TERMINATED** — start it first: `gcloud compute instances start sdar-2model-a --zone
us-central1-a --project deepinvent-ext-ut`.

---

## 8. Next actions / open items

- Monitor the GKE run to first training step (env-build → cell dispatch → `val/success_rate`
  metric appearing in the trainer log); watch the LLM token budget (`cost_ledger.jsonl`) and the
  GPU budget (`OPENRESEARCH_MAX_RUN_GPU_USD` / `_enforce_run_budget`) against the caps set in §3.
- If the environment build stalls (conda/HF/index staging from scratch on a fresh GKE node), fall
  back to the proven VM path — §7.
- **Fixes applied this session** (in the SDAR-launch commit): (1) `temperature`-deprecation for
  Claude Opus 4.8 / Sonnet 5 — `AnthropicMessagesClient` now probes-then-drops `temperature`
  (`backend/services/context/workspace/tools/anthropic_messages_client.py`); (2) installed
  `google-cloud-storage` into `.venv` (required by the `sandbox=gcp` preflight — the machine that
  *dispatches* a GKE run needs the GCS client); (3) added skills + raised the GPU cap to $35 in
  `configs/autonomous_reproduction_run_spec.json`. Full write-up: `issues.md` (2026-07-07).
- **Watch for:** the `run_experiment` cell dispatch → GKE node autoscale (cold-start up to
  ~1500s) → first `val/success_rate` in the trainer log. Env-build on a fresh GKE node (no warm
  cache) is the main risk — if it stalls, fall back to the VM path (§7).
- Two commits are **local, NOT pushed** to `deepinvent/main`: the `/lab` picker feature
  (`a279046f`) and the SDAR-launch fixes. Push only to `deepinvent`, only on operator OK.

---

## 9. Continuation prompt (paste to start the next session)

> You are continuing an OpenResearch session. Read this whole file first
> (`docs/runbooks/2026-07-07-sdar-gke-foundry-run-handoff.md`) — it is self-contained.
>
> **Model policy:** you are Opus — own orchestration, debugging, analysis, and every diff review.
> Delegate all non-critical, well-specified, mechanical work to **Sonnet** subagents. **Work in
> parallel and be efficient without sacrificing quality** (launch independent subagents
> concurrently; never let a wrong autonomous choice spend GPU money).
>
> **Task:** monitor + shepherd the live SDAR-on-GCP reproduction to a first real training step,
> then report. The run is a GKE Job on 4×A100 (`sandbox=gcp`) with `opus-foundry` root +
> `sonnet-foundry` executor/grader/verifier + skills, launched via the `/lab` UI, synced live to
> the local UI. As of this handoff it is `running` in the understand/plan phase (root driving OK,
> 0 errors, $0 GPU yet).
>
> 1. Confirm the stack is up: backend `curl -s localhost:8000/openapi.json` (200) and frontend
>    `localhost:3000`; if down, `./start.sh` (or the uvicorn one-liner in §4) with
>    `PATH=$HOME/.local/bin:$PATH` and `GOOGLE_APPLICATION_CREDENTIALS` set.
> 2. Find the live `project_id` (`GET /runs?limit=8`, newest `running`/`queued` with
>    `sandboxMode=gcp`; as of handoff `prj_23f04429cd3beaf7`). Watch it:
>    `tail -f runs/<id>/runner.stderr.log`, `runs/<id>/dashboard_events.jsonl`,
>    `runs/<id>/cost_ledger.jsonl`, and the UI at `localhost:3000/lab?projectId=<id>`.
> 3. Watch for the GKE cell dispatch → node autoscale → first `val/success_rate`. Track spend vs
>    the $35 GPU cap. `kubectl get jobs,pods -n reprolab` (PATH must include the auth plugin).
> 4. If the from-scratch env-build stalls on a fresh GKE node, fall back to the proven VM path
>    (§7): start `sdar-2model-a`, run `scripts/sdar_phase1_foundry.sh` on it.
> 5. Do NOT push either local commit without explicit operator OK. Update `issues.md` with any
>    new blocker (learn.md is archived — use `issues.md` + memory).
