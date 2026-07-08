# SDAR on Azure — operator run playbook

- **Date:** 2026-06-14
- **Status:** Ready for operator (pending §2 admin gates)
- **Pairs with:**
  - `docs/superpowers/specs/2026-06-14-sdar-on-azure-run-design.md` (source of truth)
  - `docs/runbooks/2026-06-13-azure-aionic-deploy-handoff.md` (IaC + redeploy)
  - `docs/runbooks/2026-05-23-sdar-baseline-handoff.md` (paper details + debug cycle)

---

## §1 What this does

Runs the SDAR paper (arXiv 2605.15155 — *Self-Distilled Agentic Reinforcement
Learning*) end-to-end using the `--sandbox azure` path:

- **Reasoning loop (local):** root model (`claude-oauth`) + Sonnet-OAuth executor
  run on the operator's WSL box. The paper corpus and RLM state never leave the box.
- **GPU training cells (AKS):** `run_experiment` uploads `code/` to Blob, submits one
  Kubernetes Job per cell to `sciart-aks` (ns `reprolab`, nodeSelector
  `reprolab/sku=azure_a100_80`), downloads `metrics.json` back via Blob. No Azure
  Files, no KeyVault, no in-cluster pod — **blob-only**.
- **Compute:** 1× A100-80GB (`Standard_NC24ads_A100_v4`, $3.67/hr), scale-to-zero,
  maxNodes=1 (cells serialize).
- **Scope:** smallest-two — Qwen3-1.7B + Qwen2.5-3B-Instruct, seed 42, ~32-task eval
  slice/env, GRPO baseline + SDAR proposed. 7B honest-omitted.
- **Cost caps:** `--max-wall-clock 21600` (6 h) / `--max-usd 30` / `--max-run-gpu-usd 25`
  / `--max-gpu-usd-per-hour 4`. Estimated total: **$15–25 GPU**.

---

## §2 Admin-gated prerequisites (your end)

These three gates cannot be scripted — a subscription admin must act first.

| Gate | Action | Blocking? |
|------|--------|-----------|
| **GPU quota** | Open a support ticket: family `Standard NCADS_A100_v4 Family vCPUs`, region `westus3`, limit ≥ 24 (was 0/0). Portal → Subscriptions → AIONIC Azure → Usage + quotas → filter NCADS_A100_v4 → Request increase. | **Yes** — no GPU Job schedules without it. |
| **AKS RBAC** | Assign `Azure Kubernetes Service RBAC Cluster Admin` on `sciart-aks` to the operator objectId. Cluster uses `disableLocalAccounts` + Entra RBAC, so this grant is the only path to `kubectl`/helm/Job submission. | **Yes** — bootstrap + run both require it. |
| **Redeploy** | Cluster was torn down 2026-06-13 to stop spend. Re-run the proven redeploy (see below). | **Yes** — nothing exists to run against. |
| Key-Operator role | Not needed for blob-only (§4.1 of the spec). Skip. | No |
| KeyVault / CSI / orchestrator image | Out of scope (Stream E). | No |

### Redeploy command (copy from 06-13 handoff §3)

```bash
az login --use-device-code
az account set --subscription <SUBSCRIPTION_ID>   # az account show --query id -o tsv

EGRESS=$(curl -s https://ipinfo.io/ip)

az stack group create \
  --name openresearch-l1 \
  --resource-group rg-sciartgen-external \
  --template-file infra/azure/bicep/infra.bicep \
  --parameters infra/azure/bicep/infra.bicepparam \
  --parameters kubernetesVersion=1.34 authorizedIpRanges="[\"${EGRESS}/32\"]" \
               deployKeyVault=false deployStorageKeyOperatorRole=false \
  --deny-settings-mode none \
  --action-on-unmanage detachAll \
  --yes

scripts/azure_wire_env.sh rg-sciartgen-external openresearch-l1 .env.azure
```

### AKS RBAC assignment (must be done by subscription admin)

```bash
SUB=$(az account show --query id -o tsv)
az role assignment create \
  --assignee "$(az ad signed-in-user show --query id -o tsv)" \
  --role "Azure Kubernetes Service RBAC Cluster Admin" \
  --scope /subscriptions/$SUB/resourceGroups/rg-sciartgen-external/providers/Microsoft.ContainerService/managedClusters/sciart-aks
```

---

## §3 One-time setup (operator, after §2 grants land)

Run these once per fresh deploy, in order.

```bash
# 1. Auth + cluster credentials
az login --use-device-code
az aks get-credentials -g rg-sciartgen-external -n sciart-aks
kubelogin convert-kubeconfig -l azurecli   # az aks install-cli if missing

# 2. Wire backend env from stack outputs → .env.azure (no secrets; gitignored)
scripts/azure_wire_env.sh rg-sciartgen-external openresearch-l1 .env.azure

# 3. Build + push the cell image, then pin it in .env.azure
scripts/azure_build_cell_image.sh sciartacr
# The script prints: OPENRESEARCH_AZURE_BASE_IMAGE=sciartacr.azurecr.io/reprolab-cell:<sha>
# Add that line to .env.azure:
echo 'OPENRESEARCH_AZURE_BASE_IMAGE=sciartacr.azurecr.io/reprolab-cell:<sha>' >> .env.azure

# 4. Scaffold the cluster (namespace + cell SA + NVIDIA device plugin + RBAC + quota)
#    orchestrator OFF, filesCache OFF (blob-only)
scripts/azure_sdar_bootstrap_cluster.sh
# Verify: kubectl get sa,ns,daemonset -A | grep -E 'reprolab|nvidia'
```

---

## §4 Run

```bash
# Terminal 1 — preflight must be GREEN before continuing
scripts/azure_sdar_preflight.sh          # reads .env.azure by default

# Terminal 1 — launch (preflight runs again inside the script)
scripts/azure_sdar_run.sh

# Terminal 2 — live monitor (open immediately after launch)
scripts/azure_sdar_monitor.sh            # auto-detects latest run; Ctrl-C to stop
```

The run log is tee'd to `runs/sdar-azure-<epoch>.log`.

Preflight checks (all must be green): az login, kubectl reachable, GPU quota ≥ 24,
RBAC can create jobs in `reprolab`, namespace + SA exist, SA has workload-identity
annotation, GPU pool labelled `reprolab/sku=azure_a100_80`, cell image in ACR,
storage account present, python deps importable, `claude --print ping` responds.

---

## §5 Cost table

| Resource | Rate | Notes |
|----------|------|-------|
| Standard_NC24ads_A100_v4 (A100-80GB) | $3.67/hr | Billed only while cells run; scale-to-zero idle = $0 |
| System node pool (Standard_D4s_v5) | ~$0.19/hr | Always-on while stack is deployed |
| ACR + storage | a few $/month | Negligible |
| **Per-run caps** | `--max-run-gpu-usd 25` + `--max-gpu-usd-per-hour 4` | Hard stops; run also capped at `--max-usd 30` total + 6 h wall-clock |
| **Estimated run total** | **$15–25 GPU** | Smallest-two scope, ~32 tasks/env |

GPU spend accumulates only while AKS Jobs are running. The `azure_sdar_monitor.sh`
script prints the running USD total from `cost_ledger.jsonl` every 30 s.

---

## §6 Teardown

Stops all spend. Deletes the 22 managed resources. **Keeps** the RG, AOAI, and tfstate.

```bash
az stack group delete --name openresearch-l1 -g rg-sciartgen-external \
  --action-on-unmanage deleteResources --yes
```

Do NOT delete the resource group (`rg-sciartgen-external`) — it holds
`sciartgen-azure-openai` (AOAI, eastus) and `sciartgentfstate`.

---

## §7 Troubleshooting matrix

| Symptom | Cause | Fix |
|---------|-------|-----|
| Cell Pods stuck `Pending` indefinitely | GPU quota = 0 (`Standard NCADS_A100_v4 Family vCPUs`) or GPU pool missing the `reprolab/sku=azure_a100_80` nodeSelector label | Open quota ticket (§2) or redeploy Bicep stack to re-create the labelled pool |
| `kubectl` / helm `forbidden` or 401 errors | AKS RBAC grant (`Azure Kubernetes Service RBAC Cluster Admin`) not yet assigned | Admin runs the `az role assignment create` from §2; operator re-runs `kubelogin convert-kubeconfig -l azurecli` |
| Blob 403 on cell artifact upload/download | Workload MI (`sciart-workload-mi`) lacks `Storage Blob Data Contributor` on `sciartgenreprolab` | Check `az role assignment list --assignee <clientId> --scope .../sciartgenreprolab` — add the role if missing |
| `partial` verdict with `SDK success-with-no-text` in the run log | claude-oauth / SDK auth failure (subscription session expired or not logged in) — **not** a Docker issue | Re-run `claude login`; confirm `claude --print "ping"` works; check `ANTHROPIC_API_KEY` is unset (a no-credit key silently beats OAuth) |
| Cell image pull error (`ErrImagePull` / `ImagePullBackOff`) | `OPENRESEARCH_AZURE_BASE_IMAGE` tag not present in ACR, or ACR attach not granted to the kubelet MI | Re-run `scripts/azure_build_cell_image.sh sciartacr` and update `.env.azure`; confirm `AcrPull` on `sciartacr` for `sciart-workload-mi` |
| `PVC not found` / Pods stuck on PVC bind | `azure_sdar_bootstrap_cluster.sh` was not run with `filesCache.enabled=false`, or the Helm chart was installed with defaults | Re-run bootstrap: `scripts/azure_sdar_bootstrap_cluster.sh` (sets `storage.filesCache.enabled=false`) |
| Run dies immediately with `credit balance too low` | `ANTHROPIC_API_KEY` is set in the shell and points to a no-credit account — shadows OAuth | `unset ANTHROPIC_API_KEY` then retry; or prefix the run command with `env -u ANTHROPIC_API_KEY` |
