# SDAR on Azure — first real GPU run (local-orchestrator profile)

> **Doc status:** design spec · 2026-06-14 · author: Opus (plan) → Sonnet (impl) → Codex (review).
> Pairs with `2026-06-03-azure-aks-gpu-backend-design.md` (the backend),
> `2026-06-13-azure-aionic-deploy-handoff.md` (the deploy), and
> `2026-05-23-sdar-baseline-handoff.md` (the paper).

## 1. Goal

Run the SDAR paper (arXiv 2605.15155 — *Self-Distilled Agentic Reinforcement
Learning*) end-to-end with the GPU training cells executing on the AIONIC Azure
AKS backend, at a deliberately cost-bounded scope, and produce the operator
handoff for the parts only an Azure admin can do.

This is the **first real end-to-end exercise** of the `--sandbox azure` path. The
backend is structurally complete but has never run a real Kubernetes Job (per the
06-13 handoff §5). The job here is to make the *minimal* path actually work and
hand the operator a deterministic runbook — not to build the autonomous
in-cluster orchestrator (that is Stream E, explicitly out of scope below).

## 2. Locked decisions (from brainstorming, 2026-06-14)

| Axis | Decision | Why |
|---|---|---|
| **Topology** | Local orchestrator + Azure GPU cells | RLM reasoning loop runs on WSL; only training cells dispatch as AKS Jobs. Avoids the missing orchestrator image, KeyVault, CSI, and in-cluster secrets entirely. |
| **Reasoning auth** | claude-oauth root + Sonnet-OAuth executor | The only profile validated to climb SDAR's fidelity rubric. OAuth can't run in AKS pods — which is fine, the reasoning loop is local. |
| **Compute** | 1× A100-80GB (`azure_a100_80` = `Standard_NC24ads_A100_v4`), scale-to-zero, maxNodes=1, capped | 80 GB is the floor for Qwen2.5-3B full-param GRPO RL (weights+grads+Adam fp32 ≈ 36 GB) plus the vLLM rollout KV-cache; 24 GB A10 would force LoRA and break fidelity. maxNodes=1 serializes cells for the lowest burst cost. |
| **Scope** | Smallest-two: Qwen3-1.7B + Qwen2.5-3B-Instruct, real HF weights, ~32-task eval slice/env, seed 42, GRPO baseline + SDAR proposed (ablations code-present) | Cost-bounded faithful subset. 7B honest-omitted (declared in `metrics.json["omitted"]`). Caps: `--max-wall-clock`/6h, `--max-usd 30`, `--max-run-gpu-usd 25`. Est. **$15–25 GPU**. |

## 3. Architecture / data flow

```
WSL (operator box)                         Azure (AIONIC, westus3)
─────────────────────                      ───────────────────────────────────
backend.cli reproduce 2605.15155           AKS sciart-aks  (scale-to-zero A100-80 pool)
  --sandbox azure --model claude-oauth        └─ ns reprolab, SA reprolab-sa (workload identity)
  --paper-hint 2605.15155                          │
  │                                                │  one GPU Job per cell
  ├─ root reasoning (claude-oauth)  ───────────────┤  nodeSelector reprolab/sku=azure_a100_80
  ├─ implement_baseline (Sonnet-OAuth)             │  resources nvidia.com/gpu=1
  └─ run_experiment ─► cells route                 ▼
        cells.json + train_cell.py        ┌──────────────────────────┐
        upload code  ───── Blob ─────────►│ reprolab-cell:<sha> (ACR) │
        download metrics ◄─ Blob ─────────│ aks_cell_entrypoint.py    │
                                          │  HF cache → emptyDir       │
                                          └──────────────────────────┘
                                          Storage sciartgenreprolab / reprolab-artifacts
```

- The root reasoning loop and all Sonnet sub-agents run **locally** on the
  operator's WSL box (claude-oauth subscription). The paper corpus never leaves
  the box except as the agent-written `code/` directory uploaded to Blob.
- `run_experiment` routes through the cells matrix
  (`code/cells.json` + `code/train_cell.py`). `AksJobBackend` +
  `k8s_job_cell_runner.run_matrix` submit **one Kubernetes Job per cell**, pinned
  to one A100 via `nvidia.com/gpu=1` and the `reprolab/sku=azure_a100_80`
  nodeSelector. Code in / `metrics.json` out via Blob (`reprolab-artifacts`).
- Cell Pods authenticate to Blob with the **workload managed identity**
  (`DefaultAzureCredential`), federated to `system:serviceaccount:reprolab:reprolab-sa`.
- No Azure Files, no KeyVault, no CSI driver, no in-cluster pod. **Blob-only.**

## 4. Required backend/Helm changes (the *only* code edits)

Two small, bounded edits make blob-only genuinely work. Both are also general
robustness fixes for any Azure run without a provisioned Files share.

### 4.1 Conditional cache volume — `emptyDir` fallback (P0)

**Problem:** `backend/agents/rlm/k8s_job_cell_runner.py:555-560` *unconditionally*
attaches a `persistentVolumeClaim` (`<ns>-files-pvc`) and mounts it at the cache
path. With no Files PVC provisioned, every cell Pod hangs `Pending` forever — which
would drag the Azure Files CSI mount and its **Storage Account Key Operator**
admin grant back onto the critical path, defeating the minimal profile.

**Fix:** make the cache volume conditional in the manifest builder:
- New setting `azure_files_cache_enabled: bool` (config.py), default **True**
  (preserves current behaviour for anyone who *has* provisioned Files).
- When `azure_files_cache_enabled` is False **or** `azure_files_share` is empty,
  the manifest emits an `emptyDir: {}` volume named `reprolab-cache` instead of
  the `persistentVolumeClaim`, still mounted at `cache_mount_path`. The cell
  entrypoint already reads `OPENRESEARCH_CACHE_MOUNT` for HF_HOME — an `emptyDir`
  is writable, so weights download to ephemeral pod disk and are re-fetched per
  cell (acceptable for the smallest-two scope; the persistent Files cache is a
  later speedup, not a correctness requirement).
- One canonical helper computes the volume spec so the PVC-vs-emptyDir decision
  lives in exactly one place. **Guard test** in `tests/` asserting: (a) PVC
  volume when enabled+share set, (b) `emptyDir` when disabled or share empty,
  (c) the mount path is identical in both. This is the invariant + guard-test
  shape, not a scattered patch.

### 4.2 Gate the Files PVC/StorageClass in Helm (P1)

`infra/azure/helm/templates/pvc-cache.yaml` and `storageclass-files.yaml` are
currently **always-on**, so a `helm install` for blob-only scaffolding would still
try to provision an azurefile PVC (and may hang/fail without the Key-Operator
grant). Gate both behind `.Values.storage.filesCache.enabled` (default **true**).
Setting it false installs the namespace + cell SA + NVIDIA device plugin + RBAC +
resource quota **without** the Files PVC. All other base templates stay always-on.

No other code changes. The CLI flags, `AksJobBackend`, the resolver, the catalog,
the paper YAML/hints/invariants are all already in place.

## 5. Deliverables — scripts + runbook

All scripts are `bash`, `set -euo pipefail`, idempotent, and read `.env`/`.env.azure`.

### 5.1 `scripts/azure_sdar_preflight.sh` (read-only validator)

Prints a green/red checklist and exits non-zero on any red. Checks, in order:
1. `az account show` succeeds and subscription id matches `OPENRESEARCH_AZURE_*`/arg.
2. `kubectl get nodes` works (kubelogin token present; cluster reachable).
3. GPU node pool exists and exposes the `reprolab/sku=azure_a100_80` label.
4. **GPU quota**: `az vm list-usage -l westus3` shows
   `Standard NCADS_A100_v4 Family vCPUs` limit ≥ 24 (the #1 blocker; was 0/0).
5. Operator AKS-RBAC: can `kubectl auth can-i create jobs -n reprolab`.
6. Namespace `reprolab` + SA `reprolab-sa` exist and SA has the workload-identity
   client-id annotation (i.e. the Helm bootstrap ran).
7. ACR reachable and `OPENRESEARCH_AZURE_BASE_IMAGE` tag exists (`az acr repository show-tags`).
8. Storage account + `reprolab-artifacts` container exist; workload MI has blob data access.
9. Python deps importable in `.venv`: `kubernetes`, `azure.identity`, `azure.storage.blob`.
10. `.env` Azure contract complete (every `OPENRESEARCH_AZURE_*` the path reads).
11. `claude --print ping` returns (the claude-oauth root surface is live).

### 5.2 `scripts/azure_sdar_bootstrap_cluster.sh` (one-time, idempotent)

`helm upgrade --install reprolab-aks infra/azure/helm` with
`orchestrator.enabled=false storage.filesCache.enabled=false`, and the namespace /
SA / workload-identity client-id / storage values sourced from `.env.azure` +
Bicep stack outputs. Creates namespace + cell SA + NVIDIA device plugin + RBAC +
resource quota. No orchestrator, no KeyVault, no Files. Requires the AKS-RBAC grant.

### 5.3 `scripts/azure_sdar_run.sh` (the launcher)

Sources `.env`/`.env.azure`, runs the preflight (abort on red), exports the run
env, then launches the CLI. The pinned invocation:

```bash
OPENRESEARCH_AZURE_GPU_SKUS='["azure_a100_80"]' \
OPENRESEARCH_AZURE_FILES_CACHE_ENABLED=false \
OPENRESEARCH_ACCELERATOR=off \
.venv/bin/python -m backend.cli reproduce 2605.15155 \
  --mode rlm --sandbox azure --model claude-oauth --paper-hint 2605.15155 \
  --scope-spec '{"models":["Qwen3-1.7B","Qwen2.5-3B-Instruct"],"seeds":[42]}' \
  --force-single-gpu \
  --max-wall-clock 21600 --max-usd 30 --max-run-gpu-usd 25 --max-gpu-usd-per-hour 4 \
  2>&1 | tee runs/sdar-azure-$(date +%s).log
```

Notes: `--max-pod-seconds` is RunPod-only (no-op on Azure — the Azure cap is
`--max-run-gpu-usd` + per-cell `activeDeadlineSeconds`). `--paper-hint` supplies the
algorithm-invariant guidance + the rubric invariants; `--scope-spec` narrows to
smallest-two + single seed (merges under the paper-hint default scope, operator
fields win). The slice-size guidance (~32 tasks/env, honest-omit 7B) is appended
to `OPENRESEARCH_BASELINE_EXTRA_GUIDANCE` by the script.

### 5.4 `scripts/azure_sdar_monitor.sh` (babysit loop)

Given a project id (or auto-detect latest), loops every ~30 s printing:
`kubectl get jobs,pods -n reprolab`, the current rubric score + iteration from
`runs/<id>/dashboard_events.jsonl`, the `cost_ledger.jsonl` USD total, the tail of
`code/.exec_live.log`, and any `gpu_escalated`/`capacity_exhausted`/`run_warning`
events. Recovery hint line: re-run with `--resume-cells` to retry only failed cells.

### 5.5 `docs/runbooks/2026-06-14-sdar-on-azure-run.md` (the handoff)

The operator playbook: admin-gated prerequisites (below), redeploy command,
build-cell-image, env wiring, bootstrap, preflight → run → monitor, a cost table,
teardown, and a troubleshooting matrix (Pending pods → quota/label; auth → RBAC;
blob 403 → workload MI role; partial verdict → SDK/auth not Docker).

## 6. Admin-gated handoff (operator's end — cannot be coded)

| Gate | Action | Blocking? |
|---|---|---|
| **GPU quota** | Support ticket: `Standard NCADS_A100_v4 Family vCPUs` ≥ 24, westus3 (was 0/0). | **Yes** — no GPU Job schedules without it. |
| **AKS RBAC** | Assign the operator objectId "Azure Kubernetes Service RBAC Cluster Admin" on `sciart-aks` (cluster has disableLocalAccounts + Entra RBAC → also needs `kubelogin`). | **Yes** — kubectl/helm/Job submission all need it. |
| **Redeploy** | Cluster was torn down 06-13. Re-run the proven `az stack group create` (deployKeyVault=false, deployStorageKeyOperatorRole=false). | **Yes** — nothing exists to run against. |
| Key Operator role | **DROPPED to optional** by §4.1 — only needed for the persistent Files weight-cache. | No |
| KeyVault / CSI / orchestrator image | Out of scope (Stream E). | No |

## 7. Testing / validation

- **Unit (offline, in CI):** the §4.1 guard test (PVC vs emptyDir vs mount-path);
  `bicep build` + `helm template` + `helm lint` of the gated chart; existing Azure
  unit tests still green (`tests/rlm/test_azure_executor_runtime.py` etc.). All
  socket-hermetic — no live Azure.
- **Live (operator, after §6 grants):** `azure_sdar_preflight.sh` all-green →
  one CPU-stub smoke (optional) → `azure_sdar_run.sh` → watch the rubric climb in
  `azure_sdar_monitor.sh`. Success criterion for *this* spec is **the run starts,
  a real GPU Job runs a cell, and `metrics.json` returns via Blob** — not a full
  rubric pass (that's the paper's own difficulty).

## 8. Out of scope (YAGNI)

In-cluster autonomous orchestrator, orchestrator Docker image, KeyVault secret
sync, Secrets-Store CSI driver, the persistent Files weight-cache, multi-GPU /
7B scope, CI auto-build of the cell image, AOAI all-Azure reasoning. Each is a
later, separately-specced increment.

## 9. Process

Opus owns this spec + reviews every diff. Sonnet executes §4–5 against it. Codex
reviews the result (shell safety, env-contract/flag accuracy vs the code, kubectl/az
correctness, the emptyDir guard). No real GPU run is triggered from here (quota/RBAC
gated + cluster down) — the deliverable is the validated, reviewed setup + handoff.
