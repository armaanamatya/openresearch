# Spot GPU + preemption-resume + cost-gate — operator guide

> **Doc status:** Current · runbook · last verified 2026-06-17.
> Companion to the design spec
> [`docs/superpowers/specs/2026-06-17-multi-cloud-production-gpu-execution-design.md`](../superpowers/specs/2026-06-17-multi-cloud-production-gpu-execution-design.md)
> (Streams C + D). Covers the **shipped** knobs only; everything here is opt-in
> and default-off (an unset flag = the prior on-demand behavior, byte-identical).

The production path is a **durable CPU control plane** (the RLM orchestrator,
cents/hr, always-on) dispatching **one K8s Job per training cell** onto a
**spot GPU data plane** (8×A100, scale-to-zero, idle = \$0). This guide is how an
operator turns on spot, preemption-resume, and the per-run USD ceiling, plus the
admin gates that have lead time.

---

## 1. The three knob groups (and how they pair)

Spot resilience needs **three** layers turned on together. They are intentionally
separate so each can be reasoned about independently; turning on only one is safe
but incomplete.

| Layer | GCP knob | Azure knob | Effect |
|---|---|---|---|
| **IaC — provision spot nodes** | TF `use_spot = true` (`infra/gcp/modules/gpu_nodepool`) | Bicep `useSpot = true` (`infra/azure/bicep/modules/gpu-nodepool.bicep`) | Node pool is Spot/preemptible; ~60–91 % cheaper; reclaim with a ~15–30 s `TERMINATING`/SIGTERM window. GKE auto-taints `cloud.google.com/gke-spot`; AKS taints `kubernetes.azure.com/scalesetpriority=spot`. |
| **Runtime — tolerate + reschedule** | env `OPENRESEARCH_GCP_USE_SPOT=1` (config `gcp_use_spot`) | env `OPENRESEARCH_AZURE_USE_SPOT=1` (config `azure_use_spot`) | The cell-Job builder adds the matching **spot toleration** (else the Pod never schedules onto a spot node) and, when `*_job_backoff_limit` is 0, sets `backoffLimit = *_spot_backoff_limit` (default 3) so a **preempted cell Job reschedules** onto a fresh node. App failures (exit 40–44) still `FailJob` immediately. |
| **Entrypoint — flush on eviction** | *automatic* (no flag) | *automatic* (no flag) | On SIGTERM the cell entrypoint forwards the signal to `train_cell.py`, waits a bounded window, then flushes the **latest checkpoint + partial `metrics.json`** to object storage and writes a `preempted` status sentinel (exit 45). Grace window: `OPENRESEARCH_CELL_PREEMPT_GRACE_S` (default 20 s, clamped [1, 120]). |

> **Pair the IaC flag with the runtime flag.** Provisioning a spot node
> (`use_spot`) without `OPENRESEARCH_*_USE_SPOT=1` leaves the cell Pod without the
> spot toleration → it never schedules. Setting the runtime flag without
> provisioning spot nodes just adds an inert toleration. Always set both.

### Node SKU (sized for the 7B 8-GPU cell)
The GPU node-pool defaults to an **8×A100** machine so the SDAR 7B 8-GPU cell fits
one node and lighter cells pack onto the same node — **8×A100-80** (`a2-ultragpu-8g`)
on GCP (Stream F made this the default: the memory-correct size for the 3B/7B cells,
which don't fit a 40 GB card under full-Adam GRPO), **8×A100-40**
(`Standard_ND96asr_v4`) on Azure. Scale-to-zero means idle = \$0. Override the TF
`gpu_skus[].machine_type` / Bicep `vmSize` for a different size. Azure still defaults
to 40 GB/GPU — bump its `vmSize` to `Standard_ND96amsr_v4` for 80 GB/GPU parity.

---

## 2. Preemption-resume (don't redo the whole matrix)

When a spot **GPU node** is reclaimed mid-matrix, the orchestrator process
survives (it's on the CPU pool). The in-flight cell's Job reschedules onto a fresh
spot node (backoffLimit), and its last checkpoint was flushed to blob — so at most
one cell's partial work is redone.

When the **orchestrator pod itself** is rescheduled (control-plane preemption, or a
deliberate re-launch), turn on cross-pod resume so completed cells are **skipped
from object storage** instead of re-run:

```bash
OPENRESEARCH_RESUME_CELLS=1      # arm the resume skip (already set by --rerun-experiment)
OPENRESEARCH_STABLE_RUN_ID=1     # pin run_id = project_id (deterministic, paper+arm)
```

`OPENRESEARCH_STABLE_RUN_ID` makes a relaunched run reuse the **same**
`runs/<run_id>/cells/<cell_id>/` blob prefix; the K8s runner then reads each cell's
durable `status.json` and skips any whose prior outcome was `ok` (downloading its
blob `metrics.json` for the aggregate). The orchestrator Deployment/CronJob should
export **both** for unattended preemption-survival.

> **Trust model.** Cross-pod blob-resume keys on the stable `run_id` (same code
> prefix ⇒ unchanged cell definition); it does not re-check the per-cell
> fingerprint (the local-FS fast path still does). Only enable
> `OPENRESEARCH_STABLE_RUN_ID` for genuine resume of the *same* run, not across a
> code repair that reuses the project id.

---

## 3. Per-run cost ceiling (hard backstop)

The K8s cell runner refuses to dispatch a cell once the projected per-run GPU
spend would exceed the cap, and when **every** remaining cell is refused the run
stops with a terminal `budget_exhausted` `stop_reason` (no repair loop, no
re-burning the exceeded budget):

```bash
python -m backend.cli reproduce <paper> --sandbox gcp --model claude-oauth \
  --max-usd 400                 # → OPENRESEARCH_MAX_RUN_GPU_USD / RunBudget.max_run_gpu_usd
```

Bill against the **spot** rate (~⅓ on-demand) — the runner estimates
`gpu_count × gpu_usd_per_hour × est_seconds` per cell. The cluster's standing cost
is only the CPU system pool + control plane (cents/hr) because the GPU pool
scales to zero when idle.

---

## 4. Admin gates (operator lead-time — these are not code)

Provision these **before** the first run; several need a support ticket:

1. **GPU quota.** The required count per pool = `gpu_count × max_nodes` in the
   matching A100 family. The default GCP pool (`gcp_a100_80x8`, `gpu_count=8`)
   at `max_nodes=4` needs **32** `NVIDIA A100 80GB GPUs` quota; at `max_nodes=1`
   it needs 8. Request the matching family in the target region/zone.
   - GCP: `NVIDIA_A100_80GB_GPUS` (and `PREEMPTIBLE_NVIDIA_A100_80GB_GPUS` for
     spot) ≥ `gpu_count × max_nodes` in the zone (`nvidia-tesla-a100`/40 GB pools
     use the `NVIDIA_A100_GPUS` family instead).
   - Azure: the `Standard_ND96asr_v4` / `NDASv4` family vCPU quota (and the Spot quota for spot pools).

   **SKU ↔ pool invariant.** Terraform provisions one node pool labeled
   `reprolab/sku=<short_name>` per `gpu_skus` entry, and cells schedule via
   `nodeSelector {reprolab/sku: <short_name>}`. `config.gcp_gpu_skus` (env
   `OPENRESEARCH_GCP_GPU_SKUS`) must name **exactly** the `reprolab/sku` labels
   tfvars provisions — the SKU resolver only picks SKUs that have a matching
   pool. A mismatch (config SKU with no pool) leaves every cell Pending →
   `capacity_exhausted`. The shipped defaults already match (`gcp_a100_80x8`
   in both `config.py` and the TF default), and a guard test
   (`tests/config/test_gcp_sku_pool_invariant.py`) catches future drift.

   **Lean smallest-two recipe (low quota).** For a cheap validation run, swap
   the tfvars `gpu_skus` to a single 1-GPU pool and override config to match:
   ```
   # main.tfvars: a single gcp_a100_80 (a2-ultragpu-1g, gpu_count=1) pool
   #   (see the commented "lean smallest-two" block in main.tfvars.example)
   export OPENRESEARCH_GCP_GPU_SKUS='["gcp_a100_80"]'   # config matches the pool
   ```
   This needs only `gpu_count(1) × max_nodes` = ~8 (at `max_nodes=8`) — or as
   few as 1 (`max_nodes=1`) — `NVIDIA A100 80GB GPUs` of quota.
2. **Cluster RBAC.** Grant the operator `cluster-admin` (or the namespaced
   Job-management Role the chart binds) on the GKE/AKS cluster.
3. **Secret values, out-of-band.** The IaC creates secret **names** only. Set the
   values once into Secret Manager / Key Vault:
   `claude-code-oauth-token` (from `claude setup-token` — long-lived, safe in a
   secret store; never copy `~/.claude/.credentials.json`), `anthropic-api-key`,
   `azure-openai-api-key`.
4. **Secrets-Store CSI driver + provider** installed in the cluster (GCP/Azure
   provider DaemonSets) — the orchestrator SecretProviderClass mounts the secrets.
5. **Local bootstrap auth** for `terraform`/`helm`: `gcloud auth application-default
   login` (ADC) or `az login`.
6. **Cell image built + pushed** to Artifact Registry / ACR
   (`docker/gke-cell-base`, `docker/aks-cell-base`); the orchestrator image must
   carry the `claude` CLI + Node (it is the only place the OAuth token is used).

---

## 5. Launch (unattended, in-cluster)

```bash
# one-time per cloud: terraform apply / bicep deploy with the spot flag
#   GCP : -var 'use_spot=true'   (gpu_nodepool module)
#   Azure: --parameters useSpot=true   (gpu-nodepool module)

helm upgrade --install reprolab infra/<cloud>/helm \
  --set orchestrator.enabled=true \
  --set orchestrator.gcpProject=<project>        # GCP only
# the orchestrator Deployment runs:
#   python -m backend.cli reproduce <paper> --mode rlm --sandbox <cloud> \
#     --model claude-oauth --max-usd 400
# with OPENRESEARCH_GCP_USE_SPOT=1 (or _AZURE_),
#      OPENRESEARCH_RESUME_CELLS=1, OPENRESEARCH_STABLE_RUN_ID=1 exported.
```

Artifacts land in `gs://…/runs/<run_id>` (or `…blob…/runs/<run_id>`); the only
difference between clouds is `--sandbox gcp|azure` and which IaC provisioned the
substrate.

---

## 6. Teardown

- **Idle is already \$0** — the GPU node pool autoscales to zero between cells.
- To remove the GPU capacity entirely: `terraform destroy -target=module.gpu_nodepool`
  (GCP) or delete the AKS agent pool / `az stack group delete` (Azure).
- Finished cell Jobs auto-delete via `ttlSecondsAfterFinished` (default 3600 s).

---

## 7. Known follow-ups (non-blocking)

- The preemption `status.json` reports `attempts=0/retries=0` (cosmetic telemetry;
  the resume logic keys on `outcome=preempted`/`exit_code=45`, not the counters).
- `aks_cell_entrypoint._tracking_runner` / `gke_cell_entrypoint._tracking_runner`
  duplicate `_run_trainer_subprocess` to capture the child PID for SIGTERM
  forwarding (verified faithful). A future cleanup could hoist the child-capture
  into the real runner via a module-level registry.
- Intra-cell resume (a rescheduled cell resuming `train_cell.py` from its flushed
  checkpoint) requires the agent's `train_cell.py` to load the last checkpoint; the
  harness guarantees only **cell-level** skip of already-completed cells.
