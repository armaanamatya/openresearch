# SDAR-on-GCP Reproduction — Architecture & Methodology Walkthrough

> **Doc status:** reference walkthrough · last verified 2026-07-01 against
> `scripts/sdar_gcp_optimal_run.sh`, `scripts/gcp_sdar_preflight.sh`,
> `scripts/sdar_gcp_run.sh`, `scripts/sdar_gcp_assets.py`,
> `scripts/sdar_gcp_watch.sh`, and
> `docs/runbooks/2026-06-27-sdar-1model-e2e-handoff.md`.

How the harness reproduces a paper (the reference case is **SDAR**, arXiv
2605.15155 — *Self-Distilled Agentic RL*) end-to-end on a GCP A100 VM, step by
step.

---

## Table of contents

- [The two layers](#the-two-layers)
- [The single entry command](#the-single-entry-command)
- [Layer A — VM lifecycle orchestration](#layer-a--vm-lifecycle-orchestration)
  - [Step 1 — Secure an on-demand GPU VM (capacity polling)](#step-1--secure-an-on-demand-gpu-vm-capacity-polling)
  - [Step 2 — Arm the hard billing ceiling + warm storage](#step-2--arm-the-hard-billing-ceiling--warm-storage)
  - [Step 3 — Sync code](#step-3--sync-code)
  - [Step 4 — Prepare / warm assets to GREEN](#step-4--prepare--warm-assets-to-green)
  - [Step 5 — Launch the run, GREEN-gated + detached](#step-5--launch-the-run-green-gated--detached)
- [Layer B — the reproduction itself](#layer-b--the-reproduction-itself)
  - [The RLM root loop](#the-rlm-root-loop)
  - [The cells route (the SDAR-critical path)](#the-cells-route-the-sdar-critical-path)
  - [The fidelity guards (why SDAR scores honestly)](#the-fidelity-guards-why-sdar-scores-honestly)
- [Reproduction source: repo-first vs. from-scratch (the modes)](#reproduction-source-repo-first-vs-from-scratch-the-modes)
  - [Are we reproducing from scratch or from the repo?](#are-we-reproducing-from-scratch-or-from-the-repo)
  - [The three modes](#the-three-modes)
  - [How the repo is resolved and cloned](#how-the-repo-is-resolved-and-cloned)
  - [What crosses to the GPU layer, and the report stamp](#what-crosses-to-the-gpu-layer-and-the-report-stamp)
  - [The knobs](#the-knobs)
- [Dataset & environment provisioning (the 3 SDAR experiments)](#dataset--environment-provisioning-the-3-sdar-experiments)
  - [The core principle: fail-soft into an exclusion, never a fake 0](#the-core-principle-fail-soft-into-an-exclusion-never-a-fake-0)
  - [Two provisioning routes](#two-provisioning-routes)
  - [ALFWorld — one-time multi-GB game download](#alfworld--one-time-multi-gb-game-download)
  - [WebShop — the fragile one (best-effort)](#webshop--the-fragile-one-best-effort)
  - [Search-QA — datasets + the 132 GB dense index](#search-qa--datasets--the-132-gb-dense-index)
  - [Failure modes & how to stop them failing](#failure-modes--how-to-stop-them-failing)
  - [Concrete hardening checklist](#concrete-hardening-checklist)
- [Guardrails — the VM can never idle or burn](#guardrails--the-vm-can-never-idle-or-burn)
- [Step 6 — Watch → pull → log → stop](#step-6--watch--pull--log--stop)
- [The smoke = the first cell](#the-smoke--the-first-cell)
- [Cost, scope, expectation](#cost-scope-expectation)
- [Logging / debugging reference](#logging--debugging-reference)
- [The whole flow in one diagram](#the-whole-flow-in-one-diagram)

---

## The two layers

The system is **two layers stacked on top of each other**:

- **Layer A — VM lifecycle orchestration** (runs on your laptop): secure a GPU
  VM, warm it, gate it, launch the run detached, watch it, pull the report, kill
  the VM. This is what the `scripts/sdar_gcp_*` shell scripts do.

- **Layer B — the RLM reproduction harness** (runs *on* the VM): the actual
  `backend.cli reproduce` process — root model writes Python, calls the 12+
  primitives, builds SDAR's code, runs the training grid on the A100s, grades
  against the rubric, ships `final_report.json`.

The one command below drives Layer A end-to-end; Layer A eventually execs
Layer B (`sdar_gcp_run.sh` → `backend.cli reproduce 2605.15155`).

---

## The single entry command

From `docs/runbooks/2026-06-27-sdar-1model-e2e-handoff.md`:

```bash
setsid nohup env \
  OPENRESEARCH_GCP_ZONE=us-central1-c \
  OPENRESEARCH_GCP_GPU_MACHINE_TYPE=a2-ultragpu-4g \
  OPENRESEARCH_GCP_INSTANCE=sdar-1model \
  SDAR_VRAM_GB=80 \
  OPENRESEARCH_SDAR_USE_CACHE_DISK=1 \
  OPENRESEARCH_SDAR_CACHE_DISK=sdar-ultra \
  OPENRESEARCH_SDAR_CACHE_DISK_ZONE=us-central1-c \
  OPENRESEARCH_SDAR_USE_MI=1 \
  OPENRESEARCH_SDAR_ROOT=claude-oauth \
  OPENRESEARCH_SDAR_OUTER_WALL_S=144000 \
  OPENRESEARCH_GCP_MAX_RUN_DURATION=158400s \
  NO_AUTOSTOP=1 \
  bash scripts/sdar_gcp_optimal_run.sh > runs/_sdar_1model.log 2>&1 < /dev/null &

# watch the controller
tail -f runs/_sdar_1model.log
```

`setsid nohup … < /dev/null &` is deliberate: the controller must outlive your
SSH / terminal session because the whole thing runs 1–2 days. Everything
downstream is driven by env vars, so a fresh session re-runs it with one command
and no edits.

---

## Layer A — VM lifecycle orchestration

Driver: `scripts/sdar_gcp_optimal_run.sh`.

### Step 1 — Secure an on-demand GPU VM (capacity polling)

The scarce resource is A100 capacity, not money. The script pins **on-demand
`STANDARD`** (not spot — spot A100s get reclaimed within minutes right now) and
*polls*. Three branches:

- **VM already `RUNNING`** on the right type → use it as-is. It explicitly
  **refuses to stop a running VM** (stopping a live one is what once cascaded
  into a lost run + lost capacity).
- **VM `MISSING`** + `USE_MI=1` → `CREATE`-poll: repeatedly
  `gcloud compute instances create` from the machine image every `POLL_INTERVAL`
  (600 s), catching `STOCKOUT` / `EXHAUSTED` and sleeping until capacity returns
  (up to `MAX_POLLS`=288 → 48 h ceiling).
- **VM `TERMINATED`** → flip machine-type first, then flip scheduling to
  on-demand `STANDARD`, then **assert the flip took** (`FATAL` if it is still
  spot — a silent failure would preempt in minutes), then `start`-poll for
  capacity.

Polling is free while the VM is `TERMINATED` — you only pay once it is
`RUNNING`.

There is also a **cross-poller launch lock** (`mkdir runs/.cache/sdar_launch.lock`,
atomic): you can run one poller per zone to double the catch rate, but only the
first to win launches; the loser stops its just-started VM to release the slot.

### Step 2 — Arm the hard billing ceiling + warm storage

- **`arm_max_run_duration`**:
  `set-scheduling --max-run-duration=158400s --instance-termination-action=STOP`.
  This is **control-plane enforced** — it stops the VM at the ceiling even if the
  kernel / process is dead. The ultimate backstop against a runaway 4×A100.

- **Cache disk** (`OPENRESEARCH_SDAR_USE_CACHE_DISK=1`): attaches `sdar-ultra`
  (1 TB pd-ssd, us-central1-c) and mounts it at `/mnt/sdar-cache`, formatting
  ext4 on first attach. Holds HF weights, datasets, and the wiki-18 FAISS index
  so you do not re-download ~132 GB. **Zone-locked**: GCP forbids cross-zone
  attach, so the disk and VM must share a zone (hence us-central1-c everywhere);
  a mismatch warns and falls back to boot-disk cache, never aborts.

- **Machine image** (`OPENRESEARCH_SDAR_USE_MI=1`, `sdar-mi-20260620`): boots
  with OS + NVIDIA driver + venv already warm (multi-zonal), skipping ~30–60 min
  of cold re-warm. Refreshed after each good `prepare`.

### Step 3 — Sync code

`gcp_sdar_preflight.sh sync`: `rsync` of `backend scripts docs docker infra` +
`requirements.txt pyproject.toml CLAUDE.md` + `.env` into a temp stage, then
`gcloud compute scp` to the VM. Two hard-won details are baked in:

- `--ignore-missing-args` — optional root docs do not abort the sync.
- `shopt -s dotglob` — so `.env` (carrying the OAuth token) actually ships; its
  omission once caused a silent auth miss.

### Step 4 — Prepare / warm assets to GREEN

`gcp_sdar_preflight.sh prepare` → `scripts/sdar_gcp_assets.py`. This is the
**data half of the GPU-cost gate**. It runs on the cheap CPU machine type for
spot (no GPU billing during warm); for on-demand it stages on the GPU type
itself (GPU cannot `MIGRATE`, e2 cannot `TERMINATE`, no valid intermediate). It:

1. Installs system build deps (`cmake`, `ninja`, `build-essential`,
   `openjdk-17` for pyserini's JVM), ensures `uv`, and builds a **Python 3.12 run
   venv** (harness floor is 3.11; WebShop gets its *own* 3.10 venv because its
   2022 stack is frozen).
2. `pip install -r backend/requirements.txt`, then runs
   `sdar_gcp_assets.py --prepare --check`, which via
   `backend.services.runtime.asset_provisioning`:
   - warms HF model snapshots (`Qwen3-1.7B`, `Qwen2.5-3B-Instruct`,
     `Qwen2.5-7B-Instruct`) and datasets (`nq_open`, `hotpot_qa`) into `HF_HOME`
     on the cache disk;
   - provisions the three envs (`ALFWorld`, `Search-QA` **required**; **`WebShop`
     best-effort** — its fragile JVM / corpus can fail without blocking the run);
   - writes **`runs/.cache/sdar_gcp.env`** — the env file everything else sources
     (pins `OPENRESEARCH_FORCE_SANDBOX=local`, cache paths,
     `OPENRESEARCH_PRELOAD_ASSETS=0` so the run does not re-provision, and the
     WebShop interpreter path).
3. Installs the **VM-side systemd idle watchdog** (guardrail layer 2, below).
4. Writes `.warm_ok` to the cache disk and refreshes the machine image.

The check prints **`[GREEN]`** or **`[RED] N required checks failed`**
(return 1). Required checks include: python ≥ 3.11;
torch / transformers / accelerate / faiss / alfworld importable; ALFWorld data
present; a **GPU-free model-config resolve** (catches bad repo-ids or too-old
transformers *before* any GPU is leased); and — on `launch` —
`CUDA GPUs visible ≥ MIN_GPUS`.

### Step 5 — Launch the run, GREEN-gated + detached

`gcp_sdar_preflight.sh launch`:

1. `ensure_provisioning_model STANDARD` + `ensure_machine_type GPU` +
   `start_vm` — this is where the A100s actually attach and billing starts in
   earnest.
2. **The GPU-cost gate**: re-runs
   `sdar_gcp_assets.py --check --require-gpu --min-gpus N`. A `[RED]` **aborts
   before any GPU work** — a half-provisioned env can never burn A100 hours.
3. Builds **`run_spec.json`** (instead of a fragile 12-var env whitelist): it
   JSON-encodes all the `OPENRESEARCH_*` flags + the multi-line
   `baseline_extra_guidance` (so newlines survive SCP intact) and SCPs it. This
   is where **all the fidelity guards get turned on** — `EVIDENCE_GATE`,
   `ZERO_METRICS_GUARD`, `STUB_METRICS_GUARD`, `ARG_CONTRACTS`,
   `NO_LEARNING_SIGNAL_GATE`, `ENV_LIVENESS_GATE`, `PER_MODEL_STATUS_GATE`,
   repo-first, etc. (`EVAL_PROVENANCE_GUARD` is deliberately **0** for run 1 to
   avoid false-vetoes until the smoke confirms the agent writes
   `eval_provenance.json`.)
4. Refuses to double-launch (`pgrep` guard on
   `backend.cli reproduce 2605.15155`), requires the GREEN env file, then starts
   **fully detached** on the VM:

   ```bash
   setsid nohup bash scripts/sdar_gcp_run.sh --run-spec runs/.cache/run_spec.json \
     > runs/sdar_gcp_run.out 2>&1 < /dev/null &
   ```

---

## Layer B — the reproduction itself

`sdar_gcp_run.sh` sources `sdar_gcp.env`, lifts `CLAUDE_CODE_OAUTH_TOKEN` from
`.env` into the real environment (the SDK reads `os.environ`, not pydantic
Settings), then execs, wrapped in `timeout` as an outer wall-clock backstop:

```bash
timeout --signal=TERM --kill-after=180 "$OUTER_WALL_S" \
  env -u ANTHROPIC_API_KEY .venv/bin/python -m backend.cli reproduce 2605.15155 \
    --mode rlm --sandbox local --model claude-oauth \
    --models executor=sonnet,grader=sonnet,verifier=sonnet \
    --paper-hint 2605.15155 \
    --scope-spec '{"models": ["Qwen3-1.7B"]}' \
    --repo-url https://github.com/ZJU-REAL/SDAR \
    --gpu-mode max --gpu-parallelism multi --vram-gb 80 \
    --no-force-single-gpu --max-wall-clock 86400 \
    --project-id sdar_gcp_20260618
```

Note `--sandbox local`: on the VM, experiments run as **host subprocesses on the
local A100s**, no Docker.

Note `--repo-url ...` + the script-set `OPENRESEARCH_USE_AUTHOR_REPO=1`: this
SDAR run is **repo-first in `adapt` mode** — it clones the authors' repo and
adapts it into `code/`, *not* reproducing from scratch (which is the global
default). Full detail:
[Reproduction source: repo-first vs. from-scratch](#reproduction-source-repo-first-vs-from-scratch-the-modes).

### The RLM root loop

1. **Ingest + offload.** The paper is parsed and offloaded as the REPL `context`
   variable. The root model only ever sees constant-size metadata — never the
   corpus (RLM Algorithm 1). The orchestrator navigates the paper via
   `llm_query` / `rlm_query` slices.

2. **Repo-first seeding.** Because the SDAR scripts set
   `OPENRESEARCH_USE_AUTHOR_REPO=1`, the harness clones
   `github.com/ZJU-REAL/SDAR` into `runs/<id>/repo/` (pinned to a commit SHA,
   host-only — never uploaded to the GPU layer) and, in `adapt` mode, seeds
   `code/` from it. The agent reproduces *from the authors' real code* rather
   than from scratch. This is a distinct axis from everything else in this doc —
   see [Reproduction source: repo-first vs. from-scratch](#reproduction-source-repo-first-vs-from-scratch-the-modes)
   for the full picture (including the from-scratch default).

3. **The primitives.** The `claude-oauth` (Sonnet) root writes Python calling the
   domain primitives: `understand_section`, `extract_hyperparameters`,
   `detect_environment`, `build_environment` (a no-op under `local`),
   `plan_reproduction`, `implement_baseline`, `run_experiment`,
   `verify_against_rubric`, `propose_improvements`. Sub-agents (the executor that
   writes the actual training code) run on Sonnet via OAuth.

4. **`--paper-hint 2605.15155`** loads SDAR's invariants (`paper_hints.py`):
   β=5.0, sdar_coef=0.01, the gate-formula checks (`g_t=σ(β·Δ_t)`,
   stop-gradient), WebShop / Search marked in-process. The
   `baseline_extra_guidance` block tells the executor to load *real* Qwen
   weights, generate *real* rollouts, report *measured* success — never hardcode
   Table-1 numbers, never stub.

5. **Grade + finalize.** `verify_against_rubric` grades the evidence
   (grader = Sonnet, median-of-N samples). The run writes
   `final_report.{json,md}`, `dashboard_events.jsonl`, `experiment_runs.jsonl`,
   `cost_ledger.jsonl` into `runs/<project_id>/`.

### The cells route (the SDAR-critical path)

The agent emits `code/cells.json` (the model × env × baseline matrix manifest) +
`code/train_cell.py` (a single-cell trainer). `run_experiment` then routes
through `gpu_cell_runner.run_matrix`:

- **One subprocess per cell, pinned to one GPU** (`CUDA_VISIBLE_DEVICES=<one id>`),
  `min(free_gpus, cells)` in parallel.
- **Per-cell OOM shrink-retry** (batch 0.5 → 0.25 + grad-checkpoint).
- **Aggregate**: `aggregate_cell_metrics` synthesizes the canonical
  `per_model[model][env][baseline]` leaf shape into `code/metrics.json`.

For the 1-model run that is:

```
Qwen3-1.7B  ×  {ALFWorld, WebShop, Search-QA}  ×  {SDAR, GRPO}  =  6 cells
```

The 1.7B fits one 80 GB card, so up to 4 cells run concurrently.

### The fidelity guards (why SDAR scores honestly)

SDAR is the adversarial test case because a surrogate can fake it. The guards
(all shipped via `run_spec.json`) form a **deterministic evidence layer that
vetoes fabrication regardless of the LLM grade**:

| Guard | What it catches |
|---|---|
| `ZERO_METRICS_GUARD` | All-0.0 metrics that *claim* GPU training but have no `provenance.json` → `fabrication_suspected` (the v6 hallucination: real 8-GPU training, all-0.0 metrics) |
| `STUB_METRICS_GUARD` | Placeholder-only metric keys → re-implement |
| `ENV_LIVENESS_GATE` | A dead env (WebShop server down) becomes an honest `env_setup_failed` *exclusion*, not a counted 0 (`env_health.jsonl` `served > 0` is the pass signal) |
| `NO_LEARNING_SIGNAL_GATE` | Every curve flat → verdict forced `inconclusive` |
| `EVIDENCE_GATE` | A result-claiming leaf the grader credited but with no matching on-disk cell → vetoed to 0 |
| `EVAL_PROVENANCE_GUARD` *(off for run 1)* | `accuracy = reward × 100` (SDAR's specific eval-fabrication footgun) |

Plus the `cell_matrix.py` backstop that derives `baselines_vs_sdar`
deterministically, so the SDAR-vs-GRPO headline lift can never silently score 0.

---

## Reproduction source: repo-first vs. from-scratch (the modes)

A fundamental question the rest of this doc assumed away: **what does the agent
reproduce *from* — the paper text alone, or the authors' published code?** This is
GitHub-issue #62, the *repo-first reproduction* feature. It is a completely
separate axis from the fidelity guards and the dataset provisioning above.

Code: `backend/services/ingestion/repo/{resolver,provisioner,manifest}.py`,
wired in `run.py::_build_context`; the `inspect_repository` primitive; the
`final_report.reproduction` stamp in `report.py`.

### Are we reproducing from scratch or from the repo?

**It depends on the run, and the two answers are opposite:**

- **Globally, the default is FROM SCRATCH.** In `backend/config.py` the master
  flag is `use_author_repo: bool = False`. So a bare
  `python -m backend.cli reproduce <arxiv-id>` — and every run that does not
  explicitly opt in — **clones nothing**. The agent reads the paper (offloaded as
  `context`) and writes `code/` from nothing. `inspect_repository` returns
  `{"status": "disabled"}`, no `repo/` dir is created, no `reproduction` stamp is
  written — byte-for-byte identical to before the feature existed.

- **The SDAR GCP run explicitly flips it ON, in `adapt` mode.** Both
  `scripts/sdar_gcp_run.sh` and the `run_spec.json` built by
  `gcp_sdar_preflight.sh` set:

  ```bash
  OPENRESEARCH_USE_AUTHOR_REPO=1
  OPENRESEARCH_REPRODUCTION_MODE=adapt
  OPENRESEARCH_REPO_URL=https://github.com/ZJU-REAL/SDAR
  ```

  So for SDAR specifically, the harness **clones the authors' repo and adapts it
  into `code/`**, then runs *that*.

> **Important nuance — "adapt" is not "clone-and-run".** `adapt` seeds `code/`
> from the authors' repo, but the agent still has to adapt it to the harness
> contract: emit `code/cells.json` + `code/train_cell.py`, wire the metrics into
> the canonical `per_model` shape, satisfy the fidelity guards, and fit the
> scoped model/env matrix. The authors' repo is the *starting point*, not the
> answer; all the guards, the cells route, and the rubric grading still apply on
> top. The value is that the agent starts from real, working method code instead
> of reconstructing OPSD / the gate / GRPO from the paper text.

### The three modes

`RepoSpec.mode` (from `resolver.py`) is one of three:

| Mode | Trigger | What happens | `code/` written by |
|---|---|---|---|
| **`scratch`** | Flag off (default), or no repo resolved / blacklisted / clone failed | No clone; reproduce from the paper text alone | The agent, from nothing |
| **`adapt`** | Flag on + `OPENRESEARCH_REPRODUCTION_MODE=adapt` (default when on) | Clone → seed `code/` from the repo → agent **adapts** it to the harness contract | The agent, starting from the authors' code |
| **`reference`** | Flag on + `OPENRESEARCH_REPRODUCTION_MODE=reference` | Clone **read-only**; the agent consults `repo/` but writes `code/` clean-room | The agent, from scratch, *consulting* the repo |

The `reference` mode injects an explicit prompt note (`baseline_implementation.py`):

> *"The authors' reference implementation is available read-only at `repo/`.
> Consult it for exact details, but write your own `code/` from scratch."*

Use `adapt` when you want the fastest faithful reproduction (SDAR's default);
use `reference` when you want a genuine clean-room re-implementation that is
merely *informed* by the authors' code; use `scratch` (flag off) when there is no
trustworthy repo or you are specifically testing paper-text-only reproduction.

### How the repo is resolved and cloned

**Resolution** (`RepoResolver.resolve`, pure, no IO) picks the repo by priority:

1. **User-provided URL** (`--repo-url`, or `OPENRESEARCH_REPO_URL`) — wins.
2. **Highest-confidence discovered repository** — auto-extracted from the paper
   text during ingestion (so a paper that links its repo works with no flag).
3. **None** → `scratch`.

A URL on the blacklist is dropped (blocked = do not use) and the run falls back
to the next candidate, or to `scratch`. All github forms are normalized to
`https://github.com/owner/repo` (`github:owner/repo`, `git@github.com:...`, full
https with `/tree/...` all accepted).

**Cloning** (`RepoProvisioner.clone`) is deliberately conservative and
**fail-soft — a blocked clone never aborts a run**:

- Shallow `git clone --depth 1 --no-tags` into `runs/<id>/repo/`.
- Auth is hard-disabled (`GIT_TERMINAL_PROMPT=0`, `GIT_ASKPASS=true`,
  `credential.helper=`) so a private repo fails fast instead of hanging on a
  prompt.
- `GIT_LFS_SKIP_SMUDGE=1` unless `OPENRESEARCH_REPO_CLONE_LFS=1`.
- **Caps:** `OPENRESEARCH_REPO_CLONE_TIMEOUT_S` (300 s) and a post-clone size cap
  `OPENRESEARCH_REPO_CLONE_MAX_MB` (2048 MB — an oversize clone is discarded).
- Records the exact `commit_sha` via `git rev-parse HEAD`.
- On **any** failure (network/egress blocked, private/auth, 404, oversize,
  timeout) it returns `None`: the run flips the spec to `mode="scratch"`, emits a
  `repo_clone_failed` run-warning, and proceeds from scratch. **The clone is never
  a hard dependency.**

> **Infra precondition:** the orchestrator host (the VM, for the SDAR run) needs
> egress to `github.com`. A blocked clone is safe (falls back to scratch) but
> silently loses the repo-first benefit — worth confirming on a fresh VM.

### What crosses to the GPU layer, and the report stamp

- **`repo/` is host-only.** It is excluded from every cloud/GPU upload set — only
  the adapted `code/` ever crosses to a GPU backend. The pristine authors' repo
  stays on the orchestrator host as a reference.
- **`rlm_state/repo_spec.json`** is the deterministic, trusted source of truth
  (url, mode, `commit_sha`, `clone_succeeded`) that `implement_baseline` and the
  report writer read — not an agent-attested value.
- **`final_report.reproduction`** is stamped **only** when the flag is on **and**
  `repo_spec.json` carries a non-null url **and** the clone succeeded (so a
  green-looking report cannot forge a repo run):

  ```json
  "reproduction": {
    "mode": "adapt",
    "repo_url": "https://github.com/ZJU-REAL/SDAR",
    "commit_sha": "<pinned SHA>",
    "provider": "github",
    "execution": { "ran": true, "status": "success", "metrics_produced": true },
    "adaptation": { "...delta between repo/ and code/..." }
  }
  ```

  `execution.ran` is driven by `_has_experiment_evidence` (evidence-gated — it
  can't be faked), and `adaptation` is a real diff between `repo/` and `code/`.

- **SSE events:** `repo_resolved` (which repo + mode + why), `repo_cloned`
  (commit SHA, size, key files), and `repo_clone_failed` (a run-warning on
  fallback-to-scratch).
- **The 18th primitive `inspect_repository`** lets the root deep-read the cloned
  repo (path listing, grep, re-clone) during the run; it returns
  `{"status": "disabled"}` whenever the flag is off.

Note this is a *different* axis from the two-axis reproducibility verdict: repo-first
answers *"what source did we reproduce from"*; the `replication_verdict`
(`reproducibility.replication_verdict`, under `OPENRESEARCH_TWO_AXIS_VERDICT`)
answers *"did our result replicate the paper's claim"*.

### The knobs

| Env var | Default (global) | SDAR-run value | Meaning |
|---|---|---|---|
| `OPENRESEARCH_USE_AUTHOR_REPO` | `0` (off) | `1` | Master gate. Off ⇒ from-scratch, byte-identical to no feature. |
| `OPENRESEARCH_REPRODUCTION_MODE` | `adapt` | `adapt` | `adapt` (seed + adapt) vs `reference` (read-only clean-room). |
| `OPENRESEARCH_REPO_URL` | *(unset → auto-discover)* | `.../ZJU-REAL/SDAR` | Explicit repo; overrides paper-discovered repos. Also `--repo-url`. |
| `OPENRESEARCH_REPO_CLONE_TIMEOUT_S` | `300` | `300` | Clone timeout → fall back to scratch. |
| `OPENRESEARCH_REPO_CLONE_MAX_MB` | `2048` | `2048` | Post-clone size cap (`0` disables); oversize ⇒ scratch. |
| `OPENRESEARCH_REPO_CLONE_LFS` | `0` (off) | `0` | Off ⇒ `GIT_LFS_SKIP_SMUDGE=1` (skip large LFS blobs). |

---

## Dataset & environment provisioning (the 3 SDAR experiments)

SDAR trains across **three genuinely different agentic environments**, each with
its own data-acquisition shape. This is where most real-world run failures
originate, so it gets its own treatment.

| Experiment | What must be acquired | Rough size | Acquisition mechanism |
|---|---|---|---|
| **ALFWorld** | TextWorld game files + MaskRCNN detector | ~3–5 GB | `alfworld-download` console script → `ALFWORLD_DATA` |
| **WebShop** | Product corpus JSON + Lucene/BM25 index + the `web_agent_site` package | ~3–5 GB | in-process corpus (`WEBSHOP_DATA_DIR`) **or** a `web_agent_site.app` HTTP server |
| **Search-QA** | NQ + HotpotQA question sets + (optionally) the wiki-18 E5 FAISS index | ~hundreds MB + **~132 GB** for the dense index | HF `datasets` warm + optional `snapshot_download` of a prebuilt FAISS index |

The single owner of all of this is
**`backend/services/runtime/env_cache.py::EnvCacheManager`** (host-shared,
`fcntl`-locked, crash-safe, idempotent), with the pip/model/dataset half in
**`backend/services/runtime/asset_provisioning.py`**. On the VM these are driven
by `scripts/sdar_gcp_assets.py --prepare` during the `prepare` step.

### The core principle: fail-soft into an exclusion, never a fake 0

The single most important design rule (the *fairness principle*, 2026-06-01):

> **Never dock the rubric for an environment the harness could not stand up.**

So a provisioning failure does **not** raise and does **not** produce a counted
`0.0`. Instead:

- **ALFWorld / WebShop** — a setup that can't complete returns a **verified
  `env_setup_failed` `Exclusion`** (`EnvCacheManager._fail`). That flows through
  `exclusion.build_scope_block` into `metrics.json::scope`, and the leaf scorer
  **excludes** the env from *both* the numerator and denominator. A missing env
  is an honest gap, not a failure.
- **Search-QA** — **never excludes.** A cold or unavailable dense index degrades
  to `SEARCH_QA_RETRIEVER=bm25` (still real retrieval), so the environment always
  runs.

This is reinforced at *runtime* by the `ENV_LIVENESS_GATE` fidelity guard: even
if an env provisions but then serves **0 episodes** (e.g. a WebShop server that
comes up but answers nothing), `env_health.jsonl` records `served=0` and the
cell becomes an exclusion rather than a real 0 that pollutes the grade.

### Two provisioning routes

There are **two independent ways** the SDAR data can land on the VM. They write
to the *same* cache locations, so they are interchangeable and complementary.

1. **Harness self-provisioning (default).**
   `prepare` → `sdar_gcp_assets.py --prepare` → `asset_provisioning.ensure_assets`
   + `env_cache.provision_scope`. Downloads HF weights + datasets, provisions the
   three envs, writes `sdar_gcp.env`. This is the automatic path the launcher
   uses; it is idempotent (a warm cache disk short-circuits every download).

2. **Authors' setup (pre-staging / recovery).**
   `scripts/sdar_authors_repro.sh {base,alfworld,webshop,search}` reproduces the
   upstream `ZJU-REAL/SDAR` setup exactly: conda envs (`sdar` py3.12/vllm-0.11,
   `verl-webshop` py3.10/vllm-0.8.2, `retriever` py3.10/faiss-gpu), the ALFWorld
   download, WebShop's `setup.sh -d all` (gdown corpus + Lucene index), and the
   wiki-18 E5 index. These are **CPU-only downloads (no GPU needed)** — run them
   once onto the cache disk *before* the GPU run, or as a recovery step when the
   harness path can't stand an env up.

The handoff's advice — *"if the warm cache lacks the WebShop corpus or wiki-18
index, populate it once with `bash scripts/sdar_authors_repro.sh base alfworld
webshop search`"* — is exactly route 2 pre-staging route 1's cache.

### ALFWorld — one-time multi-GB game download

- **Mechanism:** `env_cache.ensure_alfworld` resolves the `alfworld-download`
  console script *by absolute path next to the interpreter* (it is often not on a
  child process's `PATH`), runs it with `ALFWORLD_DATA` pointed at
  `<cache>/alfworld`, and records a ready state under the `fcntl` lock. Idempotent
  and host-shared across every run/cell.
- **The critical correctness check:** a download that "succeeds" but writes **no
  game files** (`traj_data.json` / the `json_2.1.1` tree) is the classic silent
  killer — cells inherit an `ALFWORLD_DATA` pointing at an empty dir, every
  episode returns `info["unavailable"]`, and those `0.0` rewards get *counted*.
  `ensure_alfworld` therefore runs `_alfworld_has_games()` **both on a cache hit
  (re-verify) and after every fresh download**. No games → the cache record is
  cleared and it re-downloads; still no games → a verified `env_setup_failed`
  exclusion. The preflight check `[FAIL] ALFWorld data` (looks for `json_2.1.1`)
  is the GREEN-gate half of the same invariant.
- **Required env** — an ALFWorld exclusion is honest but costs a whole experiment;
  keep it staged.

### WebShop — the fragile one (best-effort)

WebShop is intentionally **best-effort** (`DEFAULT_BEST_EFFORT_ENVS`) because its
2022-frozen stack is the most brittle of the three:

- **Two backends** (`env_cache.acquire_webshop`):
  - **In-process (preferred, 2026-06-27):** activated when `WEBSHOP_DATA_DIR` is
    set. Reads the product corpus directly, uses BM25, **zero sockets, no
    Java/Lucene server**. Smoke = data files (`items_shuffle.json`,
    `items_ins_v2.json`) present **and** `web_agent_site` importable. This removed
    the old `:3000`-server bug that used to zero WebShop.
  - **Legacy HTTP server:** starts (or reuses) a ref-counted `web_agent_site.app`
    process on port 3000, health-probed; the last lease to release tears it down.
- **Dependency isolation:** `install_webshop_dedicated` builds WebShop its **own
  Python 3.10 venv** (`OPENRESEARCH_WEBSHOP_PYTHON`) so its ancient
  torch/transformers can't collide with the run venv's modern stack, and pins
  `werkzeug<2.1` (a known Flask-2.1 bit-rot fix — a fresh resolve otherwise grabs
  Werkzeug 3.x whose removed `url_quote` breaks the import).
- **Where the data comes from:** the authors' `setup.sh -d all` (route 2) `gdown`s
  the product JSON from Google Drive (~3–5 GB, **prone to rate-limits** — wrapped
  in a 5× retry) and builds a pyserini Lucene index (CPU + Java, ~10–30 min).
- **Failure is graceful:** any of these breaking → best-effort skip → verified
  exclusion. WebShop never blocks the multi-hour GPU run.

### Search-QA — datasets + the 132 GB dense index

- **Question data:** HF `datasets` — `nq_open` + `hotpot_qa` (distractor,
  validation). `warm_datasets` touches a bounded prefix to force cache
  materialization; this half is small and best-effort.
- **Retriever, two tiers** (`env_cache.ensure_search_qa_index`):
  - **Dense E5 (opt-in):** requires `OPENRESEARCH_SEARCH_QA_DENSE=1` **and** either
    a pre-staged `OPENRESEARCH_SEARCH_QA_INDEX_DIR` (a dir containing a
    `*.index`/`*.faiss` file) or an `OPENRESEARCH_SEARCH_QA_INDEX_REPO` to
    `snapshot_download`. The wiki-18 E5 index is **~60–70 GB compressed → ~132 GB
    uncompressed** — the single dominant cache artifact (route 2 assembles it from
    `part_aa`+`part_ab` → `e5_Flat.index` and decompresses `wiki-18.jsonl.gz`).
    The query **encoder must match the index** (`intfloat/e5-base-v2` by default,
    `OPENRESEARCH_SEARCH_QA_ENCODER` to override) or FAISS search errors.
  - **BM25 (always works):** absent the flag/repo/index, or on *any* dense build
    error, it degrades to `SEARCH_QA_RETRIEVER=bm25`.
- **Never excludes** — this is why Search-QA is the safe env: it always runs, just
  at possibly-lower retrieval quality.

### Failure modes & how to stop them failing

| Env | Failure mode | Root cause | Fix / mitigation |
|---|---|---|---|
| ALFWorld | Counted `0.0` on empty data | Partial/gameless download inherited by cells | `_alfworld_has_games` re-verify (cache hit **and** post-download) → re-download → exclusion; GREEN-gate `json_2.1.1` check |
| ALFWorld | `alfworld-download` not found | Console script not on child `PATH` | Resolved by abs path next to the interpreter (`_resolve_console_script`) |
| WebShop | Zeroed env / `:3000` bug | Old HTTP-server-only path | In-process backend (`WEBSHOP_DATA_DIR`), no server needed |
| WebShop | `ImportError: url_quote` | Werkzeug 3.x vs Flask 2.1.2 | `werkzeug<2.1` pin in the dedicated venv |
| WebShop | Version collisions | 2022 torch/transformers vs modern stack | Dedicated **Python 3.10** venv (`OPENRESEARCH_WEBSHOP_PYTHON`) |
| WebShop | gdown rate-limit / Lucene build slow | Google-Drive throttling + Java index build | Route-2 `setup.sh` wrapped in 5× retry; pre-stage the corpus on the cache disk |
| Search-QA | FAISS search error | Query encoder ≠ index encoder | Match `OPENRESEARCH_SEARCH_QA_ENCODER` to the index (e5-base-v2) |
| Search-QA | Silent BM25 (not E5) | `SEARCH_QA_INDEX_DIR` unset / no `.faiss` file | Point it at the cached wiki-18 index + set `OPENRESEARCH_SEARCH_QA_DENSE=1` |
| Any env | Env "up" but earns nothing | Server answers 0 episodes | `ENV_LIVENESS_GATE` → `env_health.jsonl served=0` → exclusion, not a fake 0 |
| All | 132 GB re-download every boot | Ephemeral boot disk | Persistent **cache disk** (`sdar-ultra`) + `.warm_ok` sentinel + machine image |

### Concrete hardening checklist

To make the three-experiment provisioning as failure-proof as possible on a fresh
run:

1. **Pre-stage everything onto the persistent cache disk once**, CPU-only, before
   leasing GPUs:
   ```bash
   # on the VM (CPU machine type — no GPU billing):
   bash scripts/sdar_authors_repro.sh base alfworld webshop search
   ```
   This lands the ALFWorld games, WebShop corpus+index, and the wiki-18 E5 index
   on `sdar-ultra`, then the harness `prepare` sees them and skips the downloads.

2. **Let the GREEN gate do its job.** `sdar_gcp_assets.py --check` verifies
   ALFWorld `json_2.1.1`, `web_agent_site` importability (in the *WebShop*
   interpreter, not the run venv), and a GPU-free model-config resolve — a `[RED]`
   aborts *before* any GPU hour is spent. Never bypass it
   (`OPENRESEARCH_SDAR_SKIP_LAUNCH_CHECK=1`) on a real run.

3. **Point Search-QA at the staged dense index explicitly** so it doesn't silently
   fall back to BM25:
   ```bash
   export OPENRESEARCH_SEARCH_QA_DENSE=1
   export OPENRESEARCH_SEARCH_QA_INDEX_DIR=/mnt/sdar-cache/.../searchR1   # dir with e5_Flat.index
   export OPENRESEARCH_SEARCH_QA_ENCODER=intfloat/e5-base-v2             # must match the index
   ```

4. **Accept WebShop as best-effort.** If it can't come up, let it exclude — the
   run is still valid on ALFWorld + Search-QA. Only invest in it (`setup.sh
   webshop`) when you specifically need the WebShop leaf.

5. **Trust the exclusion, distrust the 0.** In `dashboard_events.jsonl`, an
   `env_setup_failed` warning is *good* (honest gap); a counted `0.0` reward on an
   env whose data you never staged is the thing to catch. `grep run_warning` for
   `env_setup_failed` / `no_learning_signal` / `fabrication_suspected` after the
   smoke.

6. **Keep the cache disk warm.** The `.warm_ok` sentinel + machine image mean the
   ~132 GB wiki-18 index and the multi-GB game/corpus data are downloaded **once**,
   not per boot — the single biggest reliability *and* cost win for repeated runs.

---

## Guardrails — the VM can never idle or burn

Four independent layers, all armed by the provisioning command:

1. **GCP `max-run-duration` → STOP** (44 h) — control-plane, survives any
   process / kernel death.
2. **VM-side systemd idle watchdog** (`sdar-idle-watchdog.timer`, installed
   during `prepare`) — no `backend.cli reproduce` process *and* GPU idle →
   `sudo shutdown -h now`. Two-grace model: **300 s** when a run is known-dead
   (`.sdar_run_exited` sentinel present), **3600 s** otherwise. Honors
   `NO_AUTOSTOP=1`.
3. **Error / exit fast-shutdown** — `sdar_gcp_run.sh`'s EXIT trap + `self_stop()`:
   on *any* exit (crash / OOM / success), the VM stops in seconds. The boot disk
   persists; flip to a CPU machine type to debug without GPU charges.
   `FASTCRASH_STAY_UP=1` holds it up on a fast crash to preserve scarce A100
   capacity.
4. **The watcher** (below).

`NO_AUTOSTOP=1` means layers 2 & 3 stand down so the **watcher owns the stop** —
because the watcher pulls the report *before* shutting down (otherwise the report
strands on the VM disk).

**Manual controls** (from your laptop):

```bash
bash scripts/sdar_gcp_optimal_run.sh down      # stop the VM now (halts GPU billing; disk persists)
bash scripts/sdar_gcp_optimal_run.sh inspect   # pull the latest report from the VM
```

---

## Step 6 — Watch → pull → log → stop

`scripts/sdar_gcp_watch.sh`. The optimal runner spawns the watcher *inside
itself* (session-survivable). Each `INTERVAL` (120 s) it SSHes in, decodes
`dashboard_events.jsonl` + the experiment ledger into one status line per tick,
classifying the stage:

```
implementing/setup → harness-driving → training(gpu N%) → repairing(N) → FINALIZED score=<s> verdict=<v>
```

On a **terminal** event (`run_complete` / `run_fatal` / `run_interrupted`):

1. **`pull_report_and_log`** — SSH-tars `final_report.{json,md}`,
   `demo_status.json`, `cost_ledger.jsonl`, etc., SCPs them into your local
   `runs/<id>/`, and appends an honest outcome row to the coworker ledger via
   `sdar_runlog.py` (auto-derives status / score / verdict / cost / cells).
2. **`vm_stop`** — `gcloud compute instances stop` (with the
   `--discard-local-ssd=true` retry for a2-ultragpu), prints
   "VM stopped (billing halted)".

Exit codes: `0` terminal · `3` VM not running (preempted) · `4` MAX_TICKS
exhausted.

---

## The smoke = the first cell

There is no separate smoke phase — **the first training cell is the smoke**
(~$30–50, ~30–60 min). SSH in and watch three signals in `runs/<id>/`:

```bash
RUN=$(ls -dt runs/2605.15155* runs/prj_* 2>/dev/null | head -1)

tail -f "$RUN/code/.exec_live.log"       # 1) live training stdout (per-round logs)
tail -f "$RUN/dashboard_events.jsonl"    # 2) structured events (rubric_score, run_warning, …)
cat     "$RUN/code/.exec_heartbeat.json" # 3) liveness heartbeat (proves progress)
```

**Pass criteria** before letting the grid continue:

- Each env constructs and rolls **real** episodes (`env_health.jsonl`
  `served > 0`, not an `env_setup_failed` exclusion).
- **Reward is non-zero and moving** (mean reward climbs off 0).
- **No `eval_provenance` false-veto** in the run-warnings.
- Measured **per-round seconds** × 150 steps × cells projects within budget.

If healthy, the run **auto-continues** into the full 6-cell grid — no action
needed.

---

## Cost, scope, expectation

- **1-model run** (6 cells, 1.7B, 150 steps): ~$300–800 / ~1–2 days, pinned by
  the smoke.
- **Honest expectation: ~0.75–0.85 on the rubric, not 1.0.** The rubric grades
  the paper's *full breadth* (5 baselines, 3 gating modes, SkillBank, the 7B,
  exact hardware); a literal 1.0 ≈ reproducing the entire paper. The goal is
  every env genuinely earning reward + the SDAR-vs-GRPO lift demonstrated, with
  the untrained baselines and the 7B declared under `metrics.json['omitted']`
  (honest omission, not failure).
- **Scale to 2 models** (Qwen3-1.7B + Qwen2.5-3B, 12 cells): re-run with
  `OPENRESEARCH_SDAR_SCOPE_SPEC='{"models": ["Qwen3-1.7B", "Qwen2.5-3B-Instruct"]}'`.

---

## Logging / debugging reference

| What | Where |
|---|---|
| Controller (provision → watch) | `runs/_sdar_1model.log` (local) |
| Live training stdout | `runs/<id>/code/.exec_live.log` (VM) — `tail -f` |
| Structured events | `runs/<id>/dashboard_events.jsonl` (`rubric_score`, `run_warning`, `primitive_call`, `experiment_progress`) |
| Heartbeat / liveness | `runs/<id>/code/.exec_heartbeat.json` |
| Per-cell metrics | `runs/<id>/code/outputs/<run>/<cell>/metrics.json` |
| Aggregated metrics + lift | `runs/<id>/code/metrics.json` (`per_model`, `baselines_vs_sdar`) |
| Env health (did it earn?) | `runs/<id>/code/outputs/*/env_health.jsonl` (`served`, `unavailable`) |
| Provenance / held-out eval | `runs/<id>/code/**/provenance.json`, `eval_provenance.json` |
| Final report | `runs/<id>/final_report.{json,md}` (+ GCS if `OPENRESEARCH_SDAR_REPORT_GCS` set) |

**Run-warnings worth grepping** (`grep run_warning dashboard_events.jsonl`):
`env_setup_failed` (an env did not come up), `fabrication_suspected` (a guard
vetoed a fake result), `no_learning_signal` (flat curves → inconclusive),
`root_degenerate_refusal_loop` (root stuck — the lifecycle driver should
recover), `cells_manifest_dropped`.

---

## The whole flow in one diagram

```
[laptop] sdar_gcp_optimal_run.sh
   │
   ├─ poll on-demand A100 capacity (free while TERMINATED; refuse to stop a RUNNING VM)
   ├─ arm GCP max-run-duration=STOP (control-plane hard ceiling)
   ├─ attach warm cache disk (sdar-ultra) + boot machine image (sdar-mi-*)
   ├─ gcp_sdar_preflight.sh sync    → rsync + scp code (.env via dotglob)
   ├─ gcp_sdar_preflight.sh prepare → build 3.12 venv, warm HF/datasets,
   │                                   provision envs, [GREEN] gate,
   │                                   install VM-side idle watchdog
   └─ gcp_sdar_preflight.sh launch  → GPU-cost [GREEN] gate (--require-gpu)
                                       → build run_spec.json (fidelity guards ON)
                                       → detached: sdar_gcp_run.sh
        │
        [VM] sdar_gcp_run.sh → backend.cli reproduce 2605.15155 --mode rlm --sandbox local
             ├─ ingest + offload paper (root sees metadata, never the corpus)
             ├─ clone ZJU-REAL/SDAR into repo/ (host-only) → seed code/ (adapt)
             ├─ RLM root loop: primitives (understand → plan → implement → run → verify)
             ├─ cells.json grid on the A100s (1 GPU/cell, OOM shrink-retry)
             ├─ deterministic guards veto fakes (zero-metrics/stub/evidence/liveness)
             ├─ verify_against_rubric (Sonnet grader, median-of-N)
             └─ final_report.{json,md}
   │
   └─ sdar_gcp_watch.sh: on terminal → pull report to laptop + log ledger → STOP VM
                          (billing halts; boot disk persists → `inspect` to re-pull)
```

---

*Generated 2026-07-01. Source of truth is the scripts named at the top; when the
provisioning flow, guard set, or watcher behavior changes, update this doc and
the dated SDAR handoff runbook together.*
