# Handoff — tool-RL GKE reproduction + first-ever GPU bring-up (2026-07-07)

## Status (5-second read)
First real GKE GPU execution on cluster `openresearch-gpu` is **proven** (a container ran on an A100 node); reliability + guard fixes are **committed + pushed to deepinvent** (`feat/gke-gpu-path-reproduction-reliability`, `75baa542`). **Blocked on ONE remaining gap:** the GKE *monolithic exec path* doesn't stage code into the pod. **Next:** implement the code-staging fix (design below), then resume run `prj_c912f5df415f410c`.

---

## 1. What this session was doing
Robustness-testing the autonomous paper-reproduction harness by running **arXiv 2604.02869** ("Multi-Turn RL for Tool-Calling Agents with Iterative Reward Calibration", "tool-RL") end-to-end on `sandbox=gcp` GKE, with a **$300 GPU cap** (`OPENRESEARCH_MAX_RUN_GPU_USD=300`, `--max-usd 450` total). This was the **first time any run reached real GPU** on this cluster, which surfaced a chain of never-validated GKE-path gaps (fixed one by one) plus a cross-validated forensic that **overturned the prior SDAR post-mortem**.

---

## 2. Task inventory

### DONE (committed + pushed to deepinvent — branch `feat/gke-gpu-path-reproduction-reliability`, commit `75baa542`)
All flag-gated **default-OFF** (byte-identical off) unless noted. Tests added for each; `test_claude_md_fidelity` + ruff green.
1. **nodeSelector fallback** (`backend/services/runtime/k8s_job_backend.py::_default_gpu_sku`, ~line 758-769) — NOT flag-gated (straight bug fix). `exec()` emitted an empty `nodeSelector` when `gpu_sku` was None (the `OPENRESEARCH_LIFECYCLE_PRIMARY` path never calls `resolve_gpu_requirements` → no `GpuPlan`), so pods floated → autoscaler thrash. Now falls back to `gpu_skus[0]`.
2. **`OPENRESEARCH_PREFLIGHT_UNION_SCOPE`** (`backend/agents/rlm/pre_flight_validator.py`) — the guard scanned only `train.py`, false-blocking faithful multi-file impls. Widens to the training-file union, strips comments/docstrings before the loss-term search, grounds variable-arg `from_pretrained` (id must be a literal somewhere in the union). Fail-closed preserved (a real surrogate still blocks).
3. **`OPENRESEARCH_IMPL_ABANDON_GUARD`** (`backend/agents/rlm/primitives.py`, the SDK-aclose-stall branch ~line 2884, constant `_ACLOSE_STALL_S`=120s) — the give-up path harvested `code/` + reported `ok` while the writer kept mutating it. Now returns `failure_class="implement_timeout_abandoned"` / `outcome="repairable"`, never `ok`.
4. **`OPENRESEARCH_HARDEXIT_CLEANUP`** (new `backend/agents/rlm/process_cleanup.py::terminate_children_then_exit`) — the 4 `os._exit` sites (`run.py::_hard_stop_with_report`, `cli.py::_module_main`, `agents/rdr/controller.py::_ClusterWatchdog._fire`, `services/events/live_runs.py` f-string child script) orphaned executor subprocesses. Now route through a psutil-first SIGTERM→2s→SIGKILL→`os._exit` helper (fail-soft, bounded).
5. **`run_watchdog.py::_detect_active_primitive`** — checked `"phase"` but events use `"status"`, so the per-primitive idle override was dead → long GPU jobs killed at 25min. **This change was made by a concurrent Codex-companion session in this checkout, not by me** — included in the commit (verified correct).
6. **`configs/tool_rl_2604_run_spec.json`** — the tool-RL run profile (see §4).
7. **`backend/agents/rlm/CLAUDE.md`** — flag-catalog entries for the 3 new flags (section "Reproduction GPU-path + preflight reliability").

### DONE — cluster infra (one-time, NOT in git — live GCP/GKE state)
- **Created ServiceAccount `reprolab-sa`** in namespace `reprolab` (was missing → every cell pod hit `FailedCreate`; this is WHY no run ever reached GPU). Annotated `iam.gke.io/gcp-service-account=215090941756-compute@developer.gserviceaccount.com` (WI). **WI IAM binding is NOT set** (I lack `iam.serviceAccounts.setIamPolicy`; so does `abheek@deepinvent.ai`) **and is NOT needed** — the cluster has **Workload Identity disabled** (`workloadIdentityConfig.workloadPool` empty) and node pools use **GCE_METADATA passthrough**, so pods use the node SA directly (like the old `sdar-*` VMs did).
- **Created GPU node pool `a100-80-rw`** (`a2-ultragpu-1g`, 1×A100-80, label `reprolab/sku=gcp_a100_80`, autoscale 0→1) **with `--scopes storage-rw`** and **DELETED the old `a100-80-1g`** (which had `devstorage.read_only` → cells could read GCS but not upload results). NOTE: `a100-80-2g` + `a100-80-4g` still have **read-only** scope — recreate them read-write the same way if a run ever needs >1 GPU per cell. `a100-80-4g` (4-GPU) is also **stocked out** in us-central1-a (GCE out of resources); 1-GPU/2-GPU scale fine.

### IN PROGRESS / BLOCKED — the last GKE-path gap (analysis done, fix NOT implemented)
See §5 for full analysis + design. One-liner: the GKE **monolithic exec path** (`k8s_job_backend.exec`, used when code lacks `cells.json`+`train_cell.py`) never downloads the GCS-uploaded code into the pod, and the command `cd`s to the orchestrator's local path → `sh: cd: can't cd to /home/abheekp/…/code`. Design decision: **route GKE monolithic runs through the cell-matrix staging** (synthesize a single-cell manifest).

### PLANNED — redesign inventory (from analysis workflow `wui39vtbz`)
Cross-validated, prioritized problem list for a robust multi-paper harness (6 criticals). Full JSON at:
`/tmp/claude-1000/-home-abheekp-openresearch/fa06ed70-cf0b-4feb-9a39-bcb8616c3296/tasks/wui39vtbz.output` → `.result.critic.prioritized_problems` (via `jq`). Criticals: ① implement_baseline harvest-race + 4 `os._exit` orphans **(fixed, #4/#3 above)**; ② preflight file-scoping **(fixed, #2)**; ③ no mid-training checkpoint/resume (every OOM/stall/spot retry restarts from step 0); ④ `local` sandbox pip-installs into the backend's own venv; ⑤ a "deterministic" preflight check (`_check_swallowed_backward_oom`) gave 3 different results on identical files; ⑥ `plan_reproduction` can silently substitute a toy task (live: UCPO→"PPO on CartPole-v1"), nothing catches it pre-GPU. ③④⑤⑥ are NOT yet fixed.

---

## 3. The corrected SDAR narrative (do NOT re-derive; it's counterintuitive)
Earlier this session I (wrongly) reported SDAR "wrote a surrogate and the guard robustly refused a fake." **That was WRONG** (cross-validated by 2 auditors against live artifacts of run `prj_23f04429cd3beaf7`): SDAR's impl was **faithful** (real `AutoModelForCausalLM.from_pretrained` in `train_cell.py:260`, real GRPO in `sdar_loss.py`, real 8-baseline registry in `build_cells.py`; `grep nn.Linear|nn.Embedding` = 0). The guard **false-blocked** it via the file-scoping bug (fixed by #2). The "surrogate" was a misread of advisory boilerplate. Memory `project_sdar_gke_robustness_run.md` + `MEMORY.md` are corrected.

---

## 4. Required context (exact coordinates)
- **Repo:** `/home/abheekp/openresearch`. **Branch:** `feat/gke-gpu-path-reproduction-reliability` (off `main`). **Pushed commit:** `75baa542`. **Remote:** `deepinvent` = `git@github.com:Deepinvent/scientific_article_generator.git` — **push ONLY here**, never origin/openresearch, never replix. Author `lolout1` / `appradhann@gmail.com`, no AI-attribution trailer, no Conventional-Commit prefix.
- **Live run:** `prj_c912f5df415f410c`, paper **2604.02869**, `runs/prj_c912f5df415f410c/`. Status `interrupted`, `$0` GPU, driver dead. **Resumable** — 4 primitives cached in `rlm_state/primitive_cache.jsonl` (`understand_section`, `detect_environment`, `plan_reproduction`, `implement_baseline`). Watchers: `/tmp/watch_toolrl2.log`. Driver stdout: `/tmp/toolrl_2604_resume.log`.
- **Run-spec:** `configs/tool_rl_2604_run_spec.json` — scratch mode (`OPENRESEARCH_USE_AUTHOR_REPO=0`, no repo exists for this paper), all guards ON, `OPENRESEARCH_MAX_RUN_GPU_USD=300`, `OPENRESEARCH_GCP_GPU_SKUS=["gcp_a100_80","gcp_a100_80x2","gcp_a100_80x4"]` (NB: `gpu_skus[0]=gcp_a100_80` is what the nodeSelector fallback targets → the `a100-80-rw` pool), `models=executor=sonnet-foundry,grader=sonnet-foundry,verifier=sonnet-foundry,spec_validator=grok`, plus a `baseline_extra_guidance` steering the executor onto verl + tau-bench + smallest-slice-first. Root model `opus-foundry` (Claude Opus 4.8 via Azure Foundry), executor `sonnet-foundry`. Auth: `AZURE_FOUNDRY_API_KEY` in `.env` (root talks to `…/anthropic` Foundry endpoint).
- **EXACT resume command** (from a clean shell; `cli.py` does NOT load `.env` into `os.environ`, hence the `load_dotenv` wrapper):
  ```bash
  cd /home/abheekp/openresearch && export PATH="$HOME/.local/bin:$PATH"
  nohup .venv/bin/python -c "
  from dotenv import load_dotenv; load_dotenv('.env')
  import sys, runpy
  sys.argv=['backend.cli','reproduce','2604.02869','--project-id','prj_c912f5df415f410c','--resume','--mode','rlm','--sandbox','gcp','--model','opus-foundry','--provider','anthropic','--models','executor=sonnet-foundry,grader=sonnet-foundry,verifier=sonnet-foundry,spec_validator=grok','--run-spec','configs/tool_rl_2604_run_spec.json','--max-usd','450','--max-repair-iterations','3']
  runpy.run_module('backend.cli', run_name='__main__')
  " > /tmp/toolrl_2604_resume.log 2>&1 &
  ```
  NOTE: `--resume` archives the prior attempt (`attempts/…`) and re-runs the pipeline fresh; the primitive cache still makes early stages cheap. It reliably reproduced a **faithful** verl/MT-GRPO/IRC impl (markers: verl/grpo/tau_bench, no cartpole) — NOT the toy-task degradation.
- **Cluster:** `openresearch-gpu` @ `us-central1-a`, project `deepinvent-ext-ut`, namespace `reprolab`, GCS bucket `deepinvent-ext-ut-sdar-runs`, base image `us-central1-docker.pkg.dev/deepinvent-ext-ut/reprolab/gke-cell-base:v1`. `kubectl`/`gcloud` work from this host (gke-gcloud-auth-plugin at `~/.local/bin`; `export PATH="$HOME/.local/bin:$PATH"`). Readiness: `scripts/gcp_ready.sh`.
- **Redesign workflow output:** `/tmp/claude-1000/-home-abheekp-openresearch/fa06ed70-cf0b-4feb-9a39-bcb8616c3296/tasks/wui39vtbz.output` (jq `.result.critic` / `.result.analyses`).

---

## 5. THE NEXT FIX — GKE monolithic code-staging (Opus analysis + design)
**Symptom (live):** run_experiment dispatched a GPU cell; pod scheduled + container ran on the A100 node, then failed: `sh: 1: cd: can't cd to /home/abheekp/openresearch/runs/prj_c912f5df415f410c/code`. Job hit `BackoffLimitExceeded`; run_experiment returned repairable; the repair loop is **futile** (a harness bug, not a code bug) so I stopped it.

**Root cause (confirmed by reading the code):**
- GKE has TWO execution routes. (a) **Cell-matrix** — `backend/agents/rlm/k8s_job_cell_runner.py` + the pod entrypoint `docker/gke-cell-base/gke_cell_entrypoint.py::main` (line 1010): downloads the GCS code prefix to a pod tempdir (`_bootstrap`), **requires `train_cell.py`** (line 1105-1127), runs the shrink ladder, uploads outputs to GCS. This is the VALIDATED path. (b) **Monolithic exec** — `backend/services/runtime/k8s_job_backend.py::exec` (line 740): `create_sandbox` (line 686) uploads code to GCS, but `exec()` submits a Job that runs the command via raw `sh -c` (`OPENRESEARCH_EXEC_COMMAND`) and **NEVER downloads the code into the pod** and does NOT go through the entrypoint's bootstrap.
- Routing: `run_experiment` uses the cell-matrix ONLY when `code/cells.json` (valid) + `code/train_cell.py` exist (gates around `primitives.py:408-467`); else it falls back to the monolithic exec path. The tool-RL executor emitted `train.py`/`commands.json` (monolithic), **no `cells.json`+`train_cell.py`** → exec path → broken on GKE. On `local`/`docker` the monolithic path works because the pod/host shares the local `runs/…/code` path; on GKE the pod is remote so the local `cd` path doesn't exist.

**DESIGN DECISION (recommended — Option B): route GKE monolithic runs through the cell-matrix staging by synthesizing a single-cell manifest.** When `sandbox ∈ {gcp,gke}` AND run_experiment would fall back to monolithic (valid `commands.json`, missing/invalid `cells.json`+`train_cell.py`), the harness deterministically synthesizes (a) a single-cell `code/cells.json` and (b) a `code/train_cell.py` shim that runs the monolithic `commands.json` command with `OUTPUT_DIR`/`OPENRESEARCH_CELL_OUTPUT_DIR` pointed at the cell output and copies/writes the resulting `metrics.json` to the flat cell-metrics path. Then run_experiment routes to `k8s_job_cell_runner.run_matrix` → reuses the fully-validated GKE machinery (code **download** + output/metrics **GCS upload** + GPU/OOM/preemption). **GKE-scoped**: gate the synthesis on the k8s/gcp backend so `local`/`docker` monolithic stays byte-identical; flag-gate default-OFF (`OPENRESEARCH_GKE_SYNTH_CELL` or similar) per repo discipline, enable it in `configs/tool_rl_2604_run_spec.json`.
- **Why B over A:** Option A (retrofit code-download into `k8s_job_backend.exec`) also requires retrofitting the **result/metrics GCS upload** (the exec path has no output-upload back to the orchestrator either — the predicted "next gap"). B reuses download+upload+GPU in one move instead of reimplementing both in the exec path. It also has no command-path-rewrite fragility.
- **Implementation pointers:** study `k8s_job_cell_runner.run_matrix` + `_build_job_manifest` for the pod-path/env contract (`OPENRESEARCH_CELL_*`, `OPENRESEARCH_BLOB_CODE_PREFIX`/`_OUTPUT_PREFIX`, `OPENRESEARCH_CACHE_MOUNT`); `gke_cell_entrypoint.py::_bootstrap`/`main` for what the pod expects (`train_cell.py`, flat `metrics.json`); the routing gates at `primitives.py:408-467`; `cell_matrix.normalize_cell_axes`/`aggregate_cell_metrics` (runtime CLAUDE.md §"One-GPU-per-cell execution") for the metrics shape. Delegate the mechanical implementation to Sonnet against this spec; Opus reviews the diff.
- **After the fix:** enable the flag in the run-spec + **resume** (§4 command) → the cell should stage code into the pod, train, and upload results.

---

## 6. Decisions + why (don't reverse by accident)
- **Read-write node pool, not Workload Identity, for GCS.** WI IAM binding is perm-denied for everyone available AND unnecessary — the cluster has WI disabled + GCE_METADATA passthrough, exactly like the old `sdar-*` VMs (which had `devstorage.read_write` scope). The fix was a read-write-scoped node pool (`a100-80-rw`), matching "the way we've been doing it before."
- **Opus designs, Sonnet executes.** The user explicitly redirected the GKE-staging fix from Sonnet to Opus for the analysis+design ("opus should do the analysis and design the fix"). Money/model-boundary/design decisions stay with Opus; mechanical impl → Sonnet; Opus reviews every diff.
- **`gpu_skus[0]=gcp_a100_80`** in the run-spec so the nodeSelector fallback targets the read-write pool. Do NOT re-pin to `gcp_a100_80x4` (4-GPU pool is stocked out + read-only).

---

## 7. Prior instructions carried forward (user, this session — verbatim where it matters)
- "push ONLY to deepinvent" (`scientific_article_generator`); never origin/openresearch, never replix. Push only when asked. "push after changes are done" (don't push half-written/mid-agent state).
- "$300 cap using 4-8 a100s depending on what is required" → `OPENRESEARCH_MAX_RUN_GPU_USD=300`, `--max-usd 450` total backstop.
- "fix any and all issues as they come up continuously."
- "use opus to delegate execution to sonnet opus for panning [planning]"; "opus should do the analysis and design the fix."
- "use gcp the way we have been doing it before" (→ VM-style node-SA/read-write scope, not WI).
- Commit granularity: few substantial commits at milestones, not per-fix.

---

## 8. Gotchas discovered (don't rediscover the hard way)
- **`pkill -f "<pattern>"` self-matches your own shell command** (the pattern text is in your cmdline) → kills the shell mid-script (exit 144/1). Also: many processes on this host are `claude` (incl. the Claude Code CLI itself) — never blanket-kill by comm `claude`. Kill by explicit PID.
- **`os._exit` orphans + zombies:** a killed driver leaves the executor `claude_agent_sdk/_bundled/claude` subprocess orphaned (self-terminates soon) and a `Z` zombie. This is critical-finding #1 (now flag-fixed via `OPENRESEARCH_HARDEXIT_CLEANUP`, default-OFF).
- **Cost-ledger blindness:** thrashing GPU nodes bill GCP while `demo_status.json` shows `gpu=$0` (the ledger has no pricing for claude-opus-4-8/sonnet-5 and doesn't see node-idle time). Watch `kubectl get nodes` for stray A100s, not just the ledger.
- **`cli.py:150` reads `.env` only to *warn*, doesn't populate `os.environ`** → CLI launches from a fresh shell need the `load_dotenv('.env')` wrapper (§4).
- **File mtimes on this host are TZ-confusing** (`find -printf %TH:%TM` vs `date -r` vs `-newermt` disagree); trust `ps -o etime`/`elapsed` for "is it active."
- **A concurrent Codex-companion session edits this same checkout** (it made the `run_watchdog.py` change). Before committing, check `git status` for edits you didn't make; scope `git add` to your files.
- **`--resume` doesn't pure-cache-resume** — it archives the prior attempt and re-runs the pipeline (cheap via primitive cache, but expect fresh understand→implement).

---

## 9. NEXT IMMEDIATE ACTION
Implement the GKE monolithic code-staging fix per §5 (recommended Option B: synthesize single-cell manifest → route through `k8s_job_cell_runner`, GKE-scoped + flag-gated). Delegate mechanical impl to Sonnet against §5; Opus reviews the diff; enable the flag in `configs/tool_rl_2604_run_spec.json`; then run the §4 resume command and watch `run_experiment` → cell dispatch on `a100-80-rw` → training (`kubectl get jobs,pods,nodes -n reprolab`; `gpu` cost in `demo_status.json`). Then commit + push to deepinvent (single substantial commit).

## 10. Open questions (user only)
- **Option B vs A** for the code-staging fix (I recommend B). Confirm before large impl if unsure.
- After tool-RL trains: continue to the redesign backlog (criticals ③④⑤⑥, §2/`wui39vtbz`), or stop at a proven single-paper GPU run?

---

## Durable-fact candidates for memory / CLAUDE.md (promote these — a handoff gets archived)
- **GKE cluster reality:** `openresearch-gpu` has **Workload Identity DISABLED** + GCE_METADATA passthrough → cells use the **node SA** for GCS; node pools need `--scopes storage-rw` (the `sdar-*` VMs' model). WI IAM binding is perm-denied AND unneeded. Live: pool `a100-80-rw` (read-write) exists; `a100-80-2g`/`4g` are read-only; `a100-80-4g` (4-GPU) stocked out in us-central1-a. KSA `reprolab-sa` created (its absence was why no run ever reached GPU). → update memory `gcp_a100_run_infra.md`.
- **The monolithic exec path (`k8s_job_backend.exec`) is not GKE-ready** (no code-staging); GKE training must go through the cell-matrix (`k8s_job_cell_runner` + `gke_cell_entrypoint.py`). → candidate for `backend/services/runtime/CLAUDE.md`.
- Already in memory (verify current): corrected SDAR narrative + this session's fixes are in `project_sdar_gke_robustness_run.md` + `MEMORY.md`.
