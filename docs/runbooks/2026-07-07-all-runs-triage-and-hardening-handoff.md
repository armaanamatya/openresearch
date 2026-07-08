<!-- doc-meta: status=current; last-verified=2026-07-07 -->
# Handoff — Cross-run triage (6 runs / 3 papers) + evidence-grounded hardening plan (2026-07-07)

> Self-contained. A zero-context session can resume every task below from this doc alone.
> Supersedes/extends the two sibling 2026-07-07 handoffs: `2026-07-07-tool-rl-gke-reproduction-handoff.md`
> (single-paper, §5 deep-dive) and `2026-07-07-lab-provider-sandbox-foundry-wiring-handoff.md` (UI wiring).

## 0. Status header (5-second read)
Triaged all **6 run dirs** (3 papers: SDAR 2605.15155, UCPO 2605.00365, tool-RL 2604.02869) via 4 parallel Sonnet agents.
**Every failure is now root-caused.** The blocker for real GPU training is **ONE shared bug: §5 GKE code-staging**
(hits any `commands.json`-only gcp/gke run — confirmed on `prj_618` + `prj_c912`). Several "criticals" turned out
**already-fixed (flags default-OFF)** or **false alarms**. **Next:** implement §5 (Option B), turn the reliability
flags ON in the run-spec, then re-run a paper end-to-end on GPU. **Money: $0 GPU billing right now** (no A100 nodes up).

---

## 1. The cross-run inventory (the deliverable)

All 6 runs, all `gpu=$0` (none reached sustained GPU training), all on `sandbox=gcp` GKE, root `opus-foundry`,
executor/grader/verifier `sonnet-foundry` (Azure Foundry Anthropic endpoint).

| Run | Paper | Wall | Died at | Root cause | Fix status |
|---|---|---|---|---|---|
| `prj_192cf34aaa49f4e9` | SDAR 2605.15155 | 3s | gcp sandbox preflight | missing host dep `google-cloud-storage` | **OPEN** (small) |
| `prj_806692f8f0eb1be1` | UCPO 2605.00365 | 13s | `_build_context` | `REPRODUCTION_MODE=execute`+`USE_AUTHOR_REPO=1` but no repo URL → resolved `scratch` → fail-closed `assert_execute_mode_stamped` refused (**guard working as designed**) | **OPEN** (fail-fast validation) |
| `prj_13f7eef55bd0b55c` | SDAR 2605.15155 | 6.5m | `plan_reproduction` | Foundry **`temperature` deprecated 400** broke `generate_rubric_tree`+`plan_reproduction` | ✅ FIXED `7f776637` (this run predates it by ~19min; it *motivated* the fix) |
| `prj_23f04429cd3beaf7` | SDAR 2605.15155 | 70m | `run_experiment` ×3 preflight-block | **two stacked bugs**: preflight file-scoping **+ harvest-race** (impl reported "ok" at 140s; real code landed **12–14 min later**; preflight validated a stale mid-write snapshot) | ✅ FIXED via 2 flags (**default-OFF**) |
| `prj_618445173e9ae4f2` | UCPO 2605.00365 | 1h53m | `run_experiment` ×3 | attempt1 watchdog false-kill (✅ fixed `75baa542`) → attempt2 **§5 code-staging** (`can't cd to .../code`) → attempt3 external SIGTERM | **§5 OPEN** |
| `prj_c912f5df415f410c` | tool-RL 2604.02869 | ~1h | `run_experiment` | **§5 code-staging** (`can't cd to .../code`) + then **driver orphaned** ("run process disappeared — host suspend/SIGKILL/OOM") | **§5 OPEN** + durability OPEN |

**Faithfulness verdict (evidence-grounded, both auditors independent):** SDAR (`23f0`, `13f7`) and UCPO (`618`) impls
were **FAITHFUL** — real `AutoModelForCausalLM.from_pretrained` via model registry, real GRPO/UCPO math
(`sdar_loss.py`, `ucpo/trainer`), real 8-baseline / verl setups, `grep nn.Linear|nn.Embedding` = **0**. **No toy drift
in any run.** (Corroborates the already-corrected SDAR narrative; extends it to UCPO.)

### The "6 criticals" (from analysis workflow `wui39vtbz`) re-adjudicated against real artifacts
- **① harvest-race + `os._exit` orphans** → **REAL** (harvest-race confirmed as the decisive proximate cause of `23f0`; `prj_23f04429cd3beaf7` is cited by ID in `backend/agents/rlm/CLAUDE.md`). **FIXED** (`OPENRESEARCH_IMPL_ABANDON_GUARD` + `OPENRESEARCH_HARDEXIT_CLEANUP`) — **but default-OFF, so it did NOT protect any run yet.** Needs turning ON.
- **② preflight file-scoping** → **REAL** (guard's default scope is `train.py`+`exp_*.py`; missed `build_cells.py`/`sdar_loss.py`). **FIXED** (`OPENRESEARCH_PREFLIGHT_UNION_SCOPE`) — default-OFF. Needs ON.
- **③ no mid-training checkpoint/resume** → **REAL but LATENT** — no run reached sustained GPU training, so it has not bitten yet. Becomes P1 the moment §5 unblocks real training (a spot-preempt/OOM would restart from step 0).
- **④ `local` sandbox pip-into-backend-venv** → **N/A for the GCP path** (only relevant to `--sandbox local`). Deprioritize.
- **⑤ "deterministic" preflight check `_check_swallowed_backward_oom` gave 3 results on identical files** → **STRONG HYPOTHESIS: this is a facet of ① (harvest-race), not a separate bug.** The files were being **mutated mid-read** (the writer was still landing code 12–14 min after "ok"), so they were *not* identical at read-time. **Confirm before building any separate fix** (likely nothing to fix once ① is ON).
- **⑥ `plan_reproduction` silently substitutes a toy task (UCPO→"PPO on CartPole-v1")** → **FALSE ALARM.** The `ppo-cartpole-v1` / `mean_reward=475` signature is a **hardcoded cosmetic demo stub** in `backend/services/events/live_runs.py:1202/1255/1707`, stamped on **every** run's `demo_status.json`/`reprolab_manifest.json` regardless of paper, and never overwritten when a run fails pre-scoring. Confirmed in `13f7`, `618`, `23f0`. **Fix = remove/relabel the stub for non-demo runs, NOT a speculative toy-task guard (YAGNI).** (Caveat: a *hypothetical separate* run could have truly substituted; none in this set did. If you want certainty, jq the live-flagged plan text in the `wui39vtbz` workflow output to see whether it read the cosmetic manifest field or the real `plan_reproduction` output.)

### Real cross-cutting issues, prioritized (what actually needs doing)
- **P0 — §5 GKE code-staging.** `k8s_job_backend.exec()` runs `commands.json` via `sh -c` and **never downloads the GCS-uploaded code into the pod**; the smoke bootstrap also emits a host-absolute `cd <laptop path>` that doesn't exist in a remote pod. Blocks every non-cell gcp/gke run. **Fix = Option B** (below).
- **P1 — reliability flags default-OFF.** `PREFLIGHT_UNION_SCOPE`, `IMPL_ABANDON_GUARD`, `HARDEXIT_CLEANUP` are all real fixes that were **never active** on any triaged run. Enable them in the autonomous run-spec(s).
- **P1 — cost-ledger blind to Foundry spend.** Every run shows `$0` in `cost_ledger.jsonl`/`demo_status.json`, but `tokens_total.json` shows real spend (e.g. `618`: Sonnet-5 137K out + 19.7M cache-read; Opus-4.8). **This is why the dashboard always shows $0 — it is NOT proof of zero spend.** Price Foundry `claude-opus-4-8`/`claude-sonnet-5` tokens into the ledger + surface GKE node-up cost.
- **P1 — run/driver durability + true resume.** Drivers died from host-suspend/OOM (`c912`) and operator SIGTERM (`618`, `13f7`); `--resume` re-runs the pipeline (cheap via primitive cache) rather than pure-resuming. Options: run the *driver* on a durable host (small GCP VM / in-cluster), harden `--resume`, and add ③ mid-training checkpoint.
- **P2 — cosmetic `ppo-cartpole-v1` stub cleanup** (fixes the triage-trap + the phantom ⑥ signal).
- **P2 — bake `google-cloud-storage` into the gcp base image/venv** (`192c`).
- **P2 — execute-mode fail-fast validation** (`8066`): reject `REPRODUCTION_MODE=execute` at launch/UI when no repo URL resolves, instead of failing 13s in.
- **P3 — grader-output robustness.** `23f0`'s `verify_against_rubric` returned "0/24 leaves — LLM grader output unparseable on every batch." Latent; only bites a run that gets *past* preflight. Harden the grader-JSON parse/repair path.

---

## 2. Task inventory

### DONE this session (analysis only — no code written yet)
- **Triaged all 6 runs** (4 parallel Sonnet agents + Opus synthesis). Findings = §1. No files modified (read-only triage).
- **Verified money-safe:** only the `e2-small` default-pool node is up; **no A100 nodes, no `reprolab` pods** → $0 GPU. Run drivers all dead. Nothing to cancel.
- **Confirmed the §5 fix direction** — two independent analyses (the tool-RL handoff's Opus design + the `618` triage agent) converge on **Option B**.

### IN PROGRESS / NEXT — implement the fixes (nothing coded yet)
See §3 for the plan. P0 first (§5), then P1 flags + cost-ledger + durability.

### PLANNED (user's broader `/iterate` asks this session, not yet started)
- **`/gcp` skill** — a GCP/GKE ops-troubleshooting playbook skill (node pools, storage-rw scope, WI-disabled reality, SKU-label matching, stray-node cost check, `gke-gcloud-auth-plugin` no-sudo install, JSON-array `GPU_SKUS`). None exists today (`~/.claude/skills/` has: iterate, ml-code-quality, ml-experiment-rigor, scientific-data-viz, token). User asked to "create skills such as /gcp".
- **Self-improvement `learn.md`** — repo `learn.md` is **archived** (`docs/archive/learn.md`, frozen 2026-06-03; convention moved to per-bug memory + `docs/superpowers/specs/`). User asked for an active `learn.md`. **Open question (§8):** revive active `learn.md` for cross-cutting RULES vs. keep memory/specs only.
- **CLAUDE.md succinctness pass** — user: *"ensure u are updating claude.md and making it more succinct / optimized aswell so i dont repeat prompts"*. Capture standing prefs (Opus-designs/Sonnet-executes, push-only-to-deepinvent, GCP-node-SA-not-WI, cost-ledger-blindness, faithful-not-surrogate) so they stop being re-prompted.
- **Re-run a paper end-to-end on GPU + live-monitor** — the ultimate goal. Candidates in resume-readiness order: `618` (UCPO, fullest cache + cells.json/train_cell.py workaround already in place), `c912` (tool-RL), `23f0` (SDAR, needs the 2 flags ON).

---

## 3. The fix plan (specs a cold session can execute)

### P0 — §5 GKE code-staging (Option B). GKE-scoped, flag-gated default-OFF.
**Root cause (confirmed by code-read):** `backend/services/runtime/k8s_job_backend.py::exec` (~line 740) →
`_build_job_manifest` sets container `command = ["/bin/sh","-c", command]`, which **overrides** the base image's
GCS-downloading `ENTRYPOINT` and never sets `OPENRESEARCH_BLOB_CODE_PREFIX`; `create_sandbox` (~line 686) uploads code
to GCS but the pod never restages it. The cell-matrix path does it right:
`backend/agents/rlm/k8s_job_cell_runner.py` + `docker/gke-cell-base/gke_cell_entrypoint.py::main` (line 1010; requires
`train_cell.py` at ~1105-1127; `_bootstrap` downloads the GCS code prefix; uploads metrics/outputs back).
Routing gate: `backend/agents/rlm/primitives.py:408-467` takes the cell route only when `code/cells.json` (valid) +
`code/train_cell.py` exist; else falls back to the broken monolithic `exec`.

**Design (Option B):** When `sandbox ∈ {gcp,gke}` AND run_experiment would fall back to monolithic (valid
`commands.json`, missing/invalid `cells.json`+`train_cell.py`), **deterministically synthesize** (a) a single-cell
`code/cells.json` and (b) a `code/train_cell.py` shim that runs the monolithic command with
`OUTPUT_DIR`/`OPENRESEARCH_CELL_OUTPUT_DIR` pointed at the cell output and writes/copies `metrics.json` to the flat
cell-metrics path. Route to `k8s_job_cell_runner.run_matrix` → reuses the validated download+upload+GPU/OOM/preempt
machinery. **Second prong:** `preflight_smoke.py`/`execution_smoke.py` bootstrap must **not** emit a host-absolute
`cd <local code_dir>` when `sandbox_mode ∈ {gcp,azure}`.
- **Why B over A** (retrofit code-download into `exec`): A also needs the result/metrics GCS **upload** retrofitted (the exec path has neither) — B reuses both in one move, no command-path-rewrite fragility.
- **Flag:** `OPENRESEARCH_GKE_SYNTH_CELL` (or similar), default-OFF, byte-identical off; enable in the run-spec. GKE-scoped so `local`/`docker` monolithic stays byte-identical.
- **Delegate mechanical impl to Sonnet against this spec; Opus reviews the diff** (user rule).

### P1 — enable the already-landed reliability flags in the autonomous run-spec(s)
Set in `configs/tool_rl_2604_run_spec.json` (and the SDAR/UCPO run-specs): `OPENRESEARCH_PREFLIGHT_UNION_SCOPE=1`,
`OPENRESEARCH_IMPL_ABANDON_GUARD=1`, `OPENRESEARCH_HARDEXIT_CLEANUP=1`, plus the new `OPENRESEARCH_GKE_SYNTH_CELL=1`.
These are all default-OFF and did not protect any triaged run.

### P1 — cost-ledger Foundry pricing (money visibility)
`cost_ledger.jsonl` has no price for `claude-opus-4-8`/`claude-sonnet-5` routed via Foundry, so it reports `$0`.
Add per-token pricing for those models (source real spend from `tokens_total.json` shape) + a GKE node-up cost signal.
The user repeatedly asks "are we billing" — this gap is the reason the dashboard is untrustworthy on cost.

### P1 — durability (pick during design, see §8 open questions)
Driver died from host-suspend/OOM/SIGTERM. Candidate fixes: run the driver on a durable host (small always-on GCP VM
or in-cluster Job), harden `--resume` toward pure-resume, add ③ mid-training checkpoint (resume training from step N).

### P2/P3 — stub cleanup, gcs dep, execute-mode fail-fast, grader robustness (specs in §1).

---

## 4. Required context (coordinates — do NOT re-derive)

- **Repo/branch/remote:** `/home/abheekp/openresearch`, branch **`feat/gke-gpu-path-reproduction-reliability`**, tip **`75baa542`**, pushed to **`deepinvent`** = `git@github.com:Deepinvent/scientific_article_generator.git`. **Push ONLY here** (never origin/openresearch, never replix). Author `lolout1 <appradhann@gmail.com>` (local config), **no AI-attribution trailer**, no Conventional-Commit prefix. Push only when asked; only after changes are done.
- **Cluster:** `openresearch-gpu` @ `us-central1-a`, project `deepinvent-ext-ut`, namespace `reprolab`, GCS bucket `deepinvent-ext-ut-sdar-runs`, base image `us-central1-docker.pkg.dev/deepinvent-ext-ut/reprolab/gke-cell-base:v1`. `kubectl`/`gcloud` work from this host with `export PATH="$HOME/.local/bin:$PATH"` (gke-gcloud-auth-plugin lives there). Readiness: `bash scripts/gcp_ready.sh`.
- **Node pools (live, verified 2026-07-07):** `default-pool` (e2-small, read-only, always-on 1 node), `gpu-l4` (read-only, scale-to-zero), **`a100-80-rw`** (`a2-ultragpu-1g`, **read-WRITE**, label `reprolab/sku=gcp_a100_80`, scale-to-zero — the ONLY writable A100 pool), `a100-80-2g` + `a100-80-4g` (**read-ONLY** → cells train but can't upload results; `a100-80-4g` also stocked out in us-central1-a). Recreate 2g/4g with `--scopes storage-rw` (see wiring-handoff §4.3 for the exact command) if a run needs >1 GPU per cell.
- **Files + line refs touched/relevant:**
  - §5: `backend/services/runtime/k8s_job_backend.py::exec` ~740, `::_build_job_manifest`, `create_sandbox` ~686, `::_default_gpu_sku` ~758-769 (nodeSelector fallback, already fixed). `backend/agents/rlm/k8s_job_cell_runner.py::run_matrix`+`_build_job_manifest`. `docker/gke-cell-base/gke_cell_entrypoint.py::main:1010` (`_bootstrap`, requires `train_cell.py` ~1105-1127). Routing gate `backend/agents/rlm/primitives.py:408-467`. Smoke bootstrap: `preflight_smoke.py`/`execution_smoke.py`.
  - Harvest-race guard: `backend/agents/rlm/primitives.py` ~line 2884 (aclose-stall branch, `_ACLOSE_STALL_S=120`). Flag `OPENRESEARCH_IMPL_ABANDON_GUARD`.
  - Preflight scope: `backend/agents/rlm/pre_flight_validator.py::_iter_python_files` (default `train.py`+`exp_*.py`), consumed by `_check_real_model_loaded`/`_check_loss_terms_present`/`_check_variants`. Flag `OPENRESEARCH_PREFLIGHT_UNION_SCOPE`.
  - Cosmetic stub: `backend/services/events/live_runs.py:1202`, `:1255`, `:1707` (`paperbenchTaskId: "reprolab-demo/ppo-cartpole-v1"`, `targetMetric: mean_reward`).
  - Watchdog (fixed `75baa542`): `backend/agents/rlm/run_watchdog.py::_detect_active_primitive` (was checking `"phase"`, events use `"status"`).
  - Flag catalog: `backend/agents/rlm/CLAUDE.md` (section "Reproduction GPU-path + preflight reliability"; cites `prj_23f04429cd3beaf7` by ID for the harvest-race).
- **Run dirs + resume readiness** (all under `runs/`, all untracked in git):
  - `prj_618445173e9ae4f2` (UCPO): cache = understand/detect/plan/implement; `code/`+`repo/` (cloned `github.com/AnamikaLochab/UCPO`) persist the real impl; `cells.json`+`train_cell.py` workaround **already written**; `OPENRESEARCH_CELL_RESUME_AUTO=1`. **Most resume-ready** — retries `run_experiment` via the correct cell route.
  - `prj_c912f5df415f410c` (tool-RL 2604.02869): run-spec `configs/tool_rl_2604_run_spec.json`; cache = understand/detect/plan/implement; scratch mode (no author repo). Resume command in the tool-RL handoff §4.
  - `prj_23f04429cd3beaf7` (SDAR): run-spec `configs/autonomous_reproduction_run_spec.json` (`OPENRESEARCH_LIFECYCLE_PRIMARY=1`); cache = understand/detect/plan + 3× implement; needs `PREFLIGHT_UNION_SCOPE=1`+`IMPL_ABANDON_GUARD=1` to pass preflight. Repo `ZJU-REAL/SDAR`.
  - `prj_13f7eef55bd0b55c` (SDAR): cache = detect_environment only; fixed by `7f776637`.
  - `prj_192cf34aaa49f4e9` (SDAR) / `prj_806692f8f0eb1be1` (UCPO): throwaway early-fails, superseded within ~2min; nothing to resume.
- **Auth:** `AZURE_FOUNDRY_API_KEY` + `AZURE_FOUNDRY_ENDPOINT` in `.env` (root+executor via Foundry `…/anthropic` endpoint — base MUST end at `…/anthropic`, not `…/anthropic/v1`). `cli.py` does NOT load `.env` into `os.environ` → CLI launches need a `load_dotenv('.env')` wrapper (see tool-RL handoff §4 for the exact `nohup python -c` form).
- **`.env` gcp config** (local, gitignored): `OPENRESEARCH_DEFAULT_SANDBOX=gcp`, `OPENRESEARCH_GCP_PROJECT=deepinvent-ext-ut`, `OPENRESEARCH_GCP_GCS_BUCKET=deepinvent-ext-ut-sdar-runs`, `OPENRESEARCH_GCP_BASE_IMAGE=…/gke-cell-base:v1`, `OPENRESEARCH_GCP_GPU_SKUS=["gcp_a100_80","gcp_a100_80x2","gcp_a100_80x4"]` (**must be JSON-array**; `gpu_skus[0]=gcp_a100_80` targets the read-write pool).

---

## 5. Decisions + why (don't reverse by accident)
- **§5 Option B over A** — reuses code-download + result-upload + GPU/OOM/preempt in one move; A needs both retrofitted into the exec path. Two independent analyses agree.
- **⑥ toy-task guard is NOT worth building** — the CartPole signature is a cosmetic stub, not agent behavior; fix the stub (root-level) instead of a speculative guard (evidence-not-grade, YAGNI).
- **⑤ likely = ①** — treat the "nondeterministic preflight" as a symptom of the harvest-race (mid-write reads), confirm before spending a separate fix.
- **Opus designs/analyzes/orchestrates + reviews every diff; Sonnet executes** (user rule, restated this session). Trivial/mechanical → Sonnet.
- **Keep the cluster up** (scale-to-zero = $0 GPU idle) — we're about to re-run; tearing down adds re-provision latency for no savings.
- **Reliability fixes must be turned ON** — they shipped default-OFF (repo flag discipline) and therefore protected nothing; the run-spec is where they get enabled.

## 6. Prior instructions carried forward (user, this session — verbatim where it matters)
- *"delegate trivial stuff to sonnet save opus for all analysis / debugging / orchestration let sonnet execute"*.
- *"ensure u are updating claude.md and making it more succinct / optimized aswell so i dont repeat prompts"*.
- *"is gcp live and billing us ? if no run is going cancel it"* → answered: $0 GPU, no active run, nothing to cancel. Standing expectation: **watch cost, cancel idle GPU proactively.**
- *"we need to fix issues with all runs so prepare thorough handoff doc"* → this doc.
- From the tool-RL handoff (still in force): push ONLY to deepinvent; `$300` GPU cap / `--max-usd 450` total for the tool-RL run; *"fix any and all issues as they come up continuously"*; *"use gcp the way we have been doing it before"* (node-SA + read-write scope, **not** Workload Identity — the cluster has WI disabled + GCE_METADATA passthrough); *"opus should do the analysis and design the fix"*; commit at milestones, not per-fix.
- Broader `/iterate` ask: harden for future issues, consider UX + multi-paper generalization, create/use skills (`/gcp`), self-improve via `learn.md`, then re-run the paper end-to-end and monitor/fix live.

## 7. Gotchas discovered (don't rediscover the hard way)
- **`$0` in `cost_ledger.jsonl`/`demo_status.json` does NOT mean $0 spent** — real spend is in `tokens_total.json`; the Foundry-routed LLM cost is unpriced. Watch `kubectl get nodes` for stray A100s, not the ledger.
- **The `ppo-cartpole-v1` benchmark block in `demo_status.json`/`reprolab_manifest.json` is a cosmetic template default** — it is NOT evidence of toy-task substitution. It misled a fast triage pass this session before being caught.
- **A "faithful" impl can still fail preflight** because the guard reads the code **mid-write** (harvest race) — the impl reported "ok" ~140s in but the real code landed 12–14 min later. Don't conclude "surrogate" from a preflight block; check the *final* code snapshot mtimes.
- **`--resume` archives the prior attempt and re-runs the pipeline** (cheap via primitive cache, not a pure resume).
- **`grep` for the driver PID self-matches your own shell** (Codex-companion wraps commands); the `openresearch-gpu` cluster's node NAME contains "gpu" so a `grep gpu` on `kubectl get nodes` matches the e2-small default node — don't misread it as a GPU node. Check `machine-type`.
- **File mtimes on this WSL2 host are TZ-confusing** — trust event timestamps *inside* the JSON/JSONL artifacts, not `ls`/`find -printf`.
- **A concurrent Codex-companion session shares this checkout** — check `git status` before committing; scope `git add` to your files.

## 8. Open questions (user only)
1. **Harden-vs-run sequencing:** unblock (§5) + turn flags ON + run ASAP, hardening the rest in parallel while monitoring (recommended — the run is the fitness signal) — vs. fix the full P1/P2 list first, then run?
2. **Which paper to re-run first:** `618` (UCPO — most resume-ready), `c912` (tool-RL — the $300-capped target), or `23f0` (SDAR — the canonical baseline)?
3. **GPU footprint:** keep 1×A100-80 read-write smallest-slice (cheapest, matches executor steering) — vs. recreate `a100-80-2g`/`4g` as read-write for multi-GPU headroom (verl may want it)?
4. **Durability:** move the driver to a durable host (small GCP VM / in-cluster) so host-suspend/OOM/SIGTERM stops killing runs — or keep launching from the laptop with `nohup`?
5. **`learn.md`:** revive an active `learn.md` for cross-cutting reliability RULES (+ keep memory/specs for narratives) — vs. keep the memory+specs-only convention?
6. **Stopping point:** after one paper trains on GPU, continue the P1/P2 backlog, or stop at a proven single-paper GPU reproduction?

## 9. Next immediate action
Implement **§5 Option B** (delegate mechanical impl to Sonnet against §3; Opus reviews the diff), flag-gate default-OFF
(`OPENRESEARCH_GKE_SYNTH_CELL`), enable it + the 3 landed reliability flags in the run-spec, then **resume `prj_618`**
(most resume-ready) and watch `run_experiment` → cell dispatch on `a100-80-rw` → training
(`kubectl get jobs,pods,nodes -n reprolab`; real spend in `tokens_total.json`, not the ledger). Then commit + push to
deepinvent (single substantial commit).

---

## Durable-fact candidates for memory / CLAUDE.md (promote — a handoff gets archived)
- **The `ppo-cartpole-v1`/`mean_reward` block is a cosmetic stub** (`live_runs.py:1202/1255/1707`), NOT a toy-substitution → **corrects critical ⑥ and the `[[project_sdar_gke_robustness_run]]` "UCPO→CartPole" note.** Both SDAR and UCPO impls were faithful.
- **Cost ledger is blind to Foundry-routed LLM spend** — all runs show `$0`; real spend in `tokens_total.json`. → gotcha for memory + `backend/services/runtime/CLAUDE.md`.
- **The monolithic exec path (`k8s_job_backend.exec`) is not GKE-ready** (no code-staging; overrides the base-image ENTRYPOINT) → GKE training must go through the cell-matrix. → `backend/services/runtime/CLAUDE.md`.
- **The reliability fixes shipped default-OFF and protected no run** — enabling them in the run-spec is a required step, not a nicety. → flag discipline note.
- **Node-pool scope reality:** only `a100-80-rw` is read-write; `2g`/`4g` are read-only; `4g` stocked out in us-central1-a. → extends `[[gcp_a100_run_infra]]`.
- **UCPO = arXiv 2605.00365** ("Uniform-Correct Policy Optimization", Purdue; repo `AnamikaLochab/UCPO`; verl/GRPO RL, GPU-bound). New paper in the test set alongside SDAR (2605.15155) and tool-RL (2604.02869).
