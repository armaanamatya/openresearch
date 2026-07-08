---
name: gcp-gke-reproduction
description: Use when a reproduction run executes on the GKE GPU sandbox (sandbox=gcp) — how to make a run actually reach the GPU, return its metrics, and not silently burn money. Covers the cell-matrix code contract that gets code staged into the pod, the honest-metrics and OOM rules the agent controls, and the operator preflight (read-write node-SA scopes, the reprolab-sa KSA, JSON-array GPU SKUs, scale-to-zero pools, cost-visibility traps) that every past GKE failure traces back to.
category: cloud-compute
tags: [GCP, GKE, Kubernetes, cell-matrix, A100, GCS, node pool, scale-to-zero, sandbox gcp, cloud reproduction, node service account, workload identity]
---

# GCP / GKE reproduction — reach the GPU, return metrics, don't burn money

Every past GKE GPU run failed for one of a *small, known* set of reasons. This playbook is that set
plus the fixes. The cluster is deliberately run "the VM way" (node service-account + storage scope),
**not** Workload Identity — do not fight that.

## What the reproduction agent controls (do these in the code you emit)

**1. Emit a cell-matrix, never a monolithic `commands.json`.** On `sandbox=gcp`, the monolithic
`k8s_job_backend.exec` path **overrides the base image's GCS-download entrypoint and never stages the
uploaded code into the pod** → the Job dies with `sh: cd: can't cd to …/code` / `BackoffLimitExceeded`,
and repairing is futile (it's a harness limitation, not a code bug). The **only** path that stages code
*and* returns results is the cell-matrix: write `code/cells.json` (one entry per training cell) **and**
`code/train_cell.py` (the per-cell entry the pod runs). The pod entrypoint downloads the GCS code
prefix, `pip install`s `requirements.txt`, runs `train_cell.py`, and uploads `metrics.json` back.
(If you only produced `commands.json`, `OPENRESEARCH_GKE_SYNTH_CELL=1` synthesizes a single-cell
`cells.json`+`train_cell.py` wrapper for you — but writing the matrix yourself is better.)

**2. Write honest metrics — absence stays absence.** Each cell writes `metrics.json` +
`provenance.json`. If a cell produced no real number, **do not** emit a placeholder — a missing metric
must stay missing (the zero-metrics / eval-provenance guards veto a fabricated-looking value, and an
`accuracy = reward × 100` or `success ≡ reward` shortcut is a fidelity failure). Eval on a **held-out
disjoint** slice via `record_eval(..., held_out=True)`.

**3. Never `save_pretrained` per cell.** A grid of 15–30 GB checkpoints fills the pod disk. There is
**no mid-training checkpoint/resume** — an OOM or spot-preempt restarts that cell from step 0, so keep
cells short and let `OPENRESEARCH_CELL_RESUME_AUTO=1` re-run only the failed cell, not the whole grid.

**4. OOM → shrink, never mask.** On CUDA OOM reduce batch / rollout count, use 8-bit Adam or gradient
checkpointing. **Never** set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` — it hides the real
footprint and corrupts long runs. The pod entrypoint already runs an OOM shrink ladder.

## The load-bearing GKE realities (operator preflight — must all hold)
These are set once at the cluster/run-spec level; a run silently fails if any is wrong.

- **Workload Identity is DISABLED; pods auth to GCS via the NODE service account.** So GPU node pools
  **must be created with `--scopes storage-rw`** (`devstorage.read_write`). A read-only pool lets cells
  *download* code but silently **cannot upload results** → training completes and the metrics are lost.
  Only pool **`a100-80-rw`** is read-write; `a100-80-2g`/`a100-80-4g` are read-only (recreate before a
  >1-GPU cell). Do **not** chase a WI IAM binding — it's perm-denied and unnecessary.
- **The KSA `reprolab-sa` must exist in namespace `reprolab`** (`kubectl create serviceaccount
  reprolab-sa -n reprolab`). Its absence is why *no run ever reached GPU* — every pod hits `FailedCreate`.
- **Pools are scale-to-zero ($0 idle).** `kubectl get nodes` showing **0 GPU nodes is NORMAL** — GKE
  autoscales one in on the first Job. Cold-start is 10–12 min, so the pending window is **1500 s**, not
  the old 900 s that killed legitimate scale-ups.
- **`OPENRESEARCH_GCP_GPU_SKUS` must be JSON-array form** and name **exactly** the `reprolab/sku`
  labels your pools carry — e.g. `["gcp_a100_80","gcp_a100_80x2","gcp_a100_80x4"]`. A bare string throws
  a `pydantic-settings` parse error; a SKU with no matching pool label leaves the cell **Pending →
  capacity_exhausted**. **`gpu_skus[0]` is the nodeSelector fallback target — put the read-write pool
  (`gcp_a100_80` → `a100-80-rw`) first.** Never re-pin `[0]` to `gcp_a100_80x4` (stocked out + read-only).

## Failure → fix (the exhaustive list)
| Symptom | Cause | Fix |
|---|---|---|
| `sh: cd: can't cd to …/code` → `BackoffLimitExceeded` | monolithic `commands.json` took the non-staging exec path | emit `cells.json`+`train_cell.py`, or `OPENRESEARCH_GKE_SYNTH_CELL=1` |
| Cell pod `FailedCreate`, run never reaches GPU | `reprolab-sa` KSA missing in ns `reprolab` | `kubectl create serviceaccount reprolab-sa -n reprolab` |
| Trains, then metrics never appear in GCS | node pool has read-only scope | recreate pool with `--scopes storage-rw` |
| Cell `Pending` forever → `capacity_exhausted` | `reprolab/sku` label ≠ `GPU_SKUS`, or pool stocked out | match labels to `GPU_SKUS`; try a different count/pool |
| GPU nodes scale up/down, `$0` in ledger, GCP billed | empty `nodeSelector` (no `GpuPlan` on the LIFECYCLE_PRIMARY path) → pod floats | fixed: falls back to `gpu_skus[0]` — keep the RW pool first |
| `pydantic-settings` parse error at startup | `GPU_SKUS` a bare string, not a JSON array | use `["gcp_a100_80",…]` |
| `google-cloud-storage` import error 3 s in | host venv missing the SDK | `pip install google-cloud-storage` (bake into the gcp venv) |
| Run `orphaned`/`interrupted` at `$0` | driver died (WSL2 host suspend/OOM/SIGKILL) | launch from a durable host or `nohup`; resume via `rlm_state/primitive_cache.jsonl` |
| 6 h SIGTERM skips finalize/validator | wall-clock watchdog hard-stop | set `OPENRESEARCH_RUN_EXPERIMENT_TIMEOUT_S=1800` |

## CPU-stage → GPU-train (the correct expensive-work pattern)
Stage a clean cache disk on a **cheap CPU VM** (`scripts/sdar_cpu_stage.sh`), so the paid GPU only
trains. Idempotent + resumable (writes a `.warm_ok` sentinel). The six staging gotchas, each a real
debug cycle, are pre-solved in that script: accept **conda ToS** (miniconda ≥25); `build-essential`
for **g++** (flash-attn); `MAX_JOBS=4` (flash-attn nvcc OOM); `pip install tensordict` (+ verl deps for
Search-QA); loop the **wiki-18** ~70 GB E5 download (unauthenticated → rate-limited but resumable);
`default-jre-headless` for **WebShop** Lucene. Also `snapshot_download` the model weights into `HF_HOME`
so the GPU run doesn't re-download on paid time.

## Cost visibility — `$0` is not proof of $0
`cost_ledger.jsonl` / `demo_status.json` are **blind** to Foundry LLM spend (the pricing catalog has no
rate for `claude-opus-4-8` / `claude-sonnet-5`, so every Foundry call books `$0`) **and** to idle
GPU-node time (autoscaler thrash bills GCP with no ledger signal). Verify real spend two ways:
`kubectl get nodes` — any `a2-ultragpu-*` / `a2-highgpu-*` node means billing (check the **machine
type**, not the name: the cluster is *named* `openresearch-gpu` so `grep gpu` false-matches the
harmless `e2-small` default node) — **plus** the real token counts in `runs/<id>/tokens_total.json`.
`OPENRESEARCH_MAX_RUN_GPU_USD` is now enforced pre-Job; set it in the run-spec.

## Fixed coordinates & tooling
- Cluster `openresearch-gpu` @ `us-central1-a`, project `deepinvent-ext-ut`, namespace `reprolab`,
  KSA `reprolab-sa`. Bucket `gs://deepinvent-ext-ut-sdar-runs`. Base image (PINNED, never `:latest`)
  `us-central1-docker.pkg.dev/deepinvent-ext-ut/reprolab/gke-cell-base:v1`.
- Pools: `a100-80-rw` (1×A100-80, **read-write**, `reprolab/sku=gcp_a100_80`), `a100-80-2g`/`a100-80-4g`
  (read-only; `4g` stocked out in `us-central1-a`). Quota: A100-80=4, A100-40=8, A2_CPUS=48.
- `gke-gcloud-auth-plugin` — snap gcloud can't `components install` it and it's not in apt: install the
  official `.deb` no-sudo to `~/.local/bin`, then **`export PATH="$HOME/.local/bin:$PATH"`** before any
  `kubectl`/`gcloud`. `cli.py` does not load `.env` into `os.environ` — CLI launches need a
  `load_dotenv('.env')` wrapper.
- **`bash scripts/gcp_ready.sh`** = one-command readiness gate (repoints a foreign kube-context, lists
  pools + GPU capacity, checks bucket/base-image/ADC) — read-only, spends nothing.

## Launch checklist
1. `export PATH="$HOME/.local/bin:$PATH"`; `bash scripts/gcp_ready.sh` all-green.
2. `kubectl get sa reprolab-sa -n reprolab` exists; the `gpu_skus[0]` pool is read-write.
3. `GPU_SKUS` is a JSON array whose entries match real `reprolab/sku` labels; RW pool first.
4. Run-spec turns the reliability flags **ON** (they protect nothing until enabled): at minimum
   `GKE_SYNTH_CELL`, `CELL_RESUME_AUTO`, `PREFLIGHT_UNION_SCOPE`, `IMPL_ABANDON_GUARD`, `HARDEXIT_CLEANUP`.
5. Set `MAX_RUN_GPU_USD` + `--max-usd`; launch from a durable host / `nohup`.
6. During: `kubectl get jobs,pods,nodes -n reprolab`; 0 GPU nodes before first dispatch is normal;
   watch `kubectl get nodes` for stray A100s. After: confirm cells uploaded `metrics.json` to GCS and
   `grep -c 'nn.Linear\|nn.Embedding'` is 0 in the final `code/`.

## Sources (repo-native grounding — not vendored)
`learn.md`, the three `docs/runbooks/2026-07-07-*.md` handoffs, `backend/services/runtime/CLAUDE.md`,
`backend/agents/rlm/gke_cell_synth.py`, `scripts/{gcp_ready.sh,sdar_cpu_stage.sh}`.
