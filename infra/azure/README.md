# ReproLab — Azure AKS GPU Backend: Bicep L1

**Client:** DeepInvent  
**Design lock:** 2026-06-03 · IaC sole-survivor update: 2026-06-12  
**Layer:** L1 (Azure infrastructure). See §3-layer split below.

---

## Why three layers — and why transient Jobs are NOT here

| Layer | What | Managed by | Lifecycle |
|-------|------|-----------|-----------|
| **L0 — Access bootstrap** (`infra/azure/bicep/`) | Subscription-scope RG creation + Contributor / User Access Administrator grants to the operator principal | **Bicep** (subscription admin, once) | One-time; stable |
| **L1 — Azure infra** (this directory) | Resource group, VNet/subnet, AKS cluster, GPU node pool, ACR, storage (Blob + Files), managed identity + role assignments | **Bicep** (`infra/azure/bicep/infra.bicep`) | Provisioned once; rarely changes |
| **L2 — In-cluster scaffold** (`infra/azure/helm/`) | Namespace, workload-identity ServiceAccount, Files PVC, orchestrator RBAC, NVIDIA device plugin, ResourceQuota | **Helm** (applied once per cluster) | Per-cluster; semi-static |
| **L3 — Runtime** | Per-cell Kubernetes **Jobs** | **Orchestrator code** (`backend/agents/rlm/k8s_job_cell_runner.py`) | Thousands; transient |

**L0 note:** Step 0 below requires subscription **Owner** because L0 creates the RGs and grants the operator Contributor + User Access Administrator on them. After L0, day-to-day operators need only RG-scoped Contributor — no subscription-level Owner for `az stack group create`. See `infra/azure/bicep/README.md` for the full L0 deploy.

Bicep deployment stacks store state server-side (in ARM). There is no tfstate file to manage. The boundary between L1/L2 and L3 is deliberate — only the durable substrate (cluster, pools, storage, identity) lives here.

---

## Module tree

```
infra/azure/
  bicep/                    Bicep L0 + L1 IaC (sole IaC)
    infra.bicep               L1 entry point (deployed via az stack group create)
    infra.bicepparam.example  L1 parameter template (copy → infra.bicepparam)
    main.bicep                L0 subscription bootstrap
    rg-grants.bicep           L0 helper: RG-scope role grants
    modules/
      network.bicep           VNet, AKS subnet, NSG + association
      aks.bicep               Managed-identity AKS; OIDC + workload identity
      gpu-nodepool.bicep      GPU node pool(s); autoscale min=0; GPU taint + labels
      acr.bicep               Standard ACR; admin disabled; AcrPull → kubelet identity
      storage.bicep           Blob artifacts bus; Azure Files cache; identity role grants
      identity.bicep          User-assigned MI; federated credential; Blob contributor
      monitoring.bicep        Log Analytics workspace; AKS/ACR/storage diagnostic settings
  helm/                     Helm L2 in-cluster scaffold
  envs/deepinvent/
    helm-values.yaml.example  Helm L2 values template (copy before use)
```

---

## Step-by-step: bootstrap → deploy → scaffold

### 0. Prerequisites

- `az login` as an account with **Owner** on the target subscription (needed for L0 role assignments — one-time only).
- Azure CLI ≥ 2.50 (`az --version`).
- Bicep CLI ≥ 0.29 — installed automatically by the Azure CLI, or manually:
  ```bash
  az bicep install && az bicep version
  ```
- GPU quota filed **before** apply:
  ```bash
  az quota create \
    --resource-name standardNCADSA100v4Family \
    --scope /subscriptions/<SUBSCRIPTION_ID>/providers/Microsoft.Compute/locations/<REGION> \
    --limit-object value=<24 * gpu_max_nodes>
  ```
  Quota approval can take hours to days on a fresh subscription. Start this immediately.

### 1. L0 — access bootstrap (one-time, subscription admin)

Run this once to create the resource groups and grant the operator the RG-scoped roles needed to deploy L1:

```bash
cp infra/azure/bicep/main.bicepparam.example infra/azure/bicep/main.bicepparam
# Fill in every <PLACEHOLDER> — see comments in the file

az deployment sub create \
  --location <AZURE_REGION> \
  --template-file infra/azure/bicep/main.bicep \
  --parameters infra/azure/bicep/main.bicepparam
```

The deployment is idempotent — re-running is safe. See `infra/azure/bicep/README.md` for the full L0 reference.

### 2. Configure L1 parameters

```bash
cp infra/azure/bicep/infra.bicepparam.example infra/azure/bicep/infra.bicepparam
# Fill in every <PLACEHOLDER> — see comments in the file
```

Key placeholders:

| Parameter | Notes |
|-----------|-------|
| `prefix` | ≤8 chars, lowercase alphanum |
| `location` | Must have A100 quota |
| `authorizedIpRanges` | All operator egress CIDRs |
| `kubernetesVersion` | Available in region |
| `storageAccountName` | Globally unique |
| `gpuSkus[].maxNodes` | Cost-driven; start at 1 |

### 3. Deploy L1 (create or update)

```bash
az stack group create \
  --name openresearch-infra \
  --resource-group <RESOURCE_GROUP_NAME> \
  --template-file infra/azure/bicep/infra.bicep \
  --parameters infra/azure/bicep/infra.bicepparam \
  --deny-settings-mode none \
  --action-on-unmanage detachAll
```

`--action-on-unmanage detachAll` detaches (rather than deletes) resources removed from the stack — the safe default for initial deployments. Flip to `deleteResources` once the stack is stable.

Re-deploying the same command is idempotent. Bicep deployment stacks manage state server-side; no local state file exists.

### 4. Read L1 outputs

```bash
az stack group show \
  --name openresearch-infra \
  --resource-group <RESOURCE_GROUP_NAME> \
  --query outputs
```

Or use `scripts/azure_wire_env.sh` to export the outputs directly into your shell / `.env`.

### 5. Populate Helm L2 values

```bash
cp infra/azure/envs/deepinvent/helm-values.yaml.example \
   infra/azure/envs/deepinvent/helm-values.yaml

# Populate from az stack group show --query outputs:
#   workloadIdentityClientId  → workloadIdentity.clientId
#   storageAccountNameOut     → storage.accountName
#   filesStorageAccountName   → storage.filesAccountName
#   filesShareNameOut         → storage.fileShareName
#   gpuSkus[*].maxNodes (param) → resourceQuota.maxGpu / maxGpuLimits / maxJobs
#
# Or run:  scripts/azure_wire_env.sh (writes the block to stdout for .env pasting)
```

### 6. Apply Helm L2

```bash
az aks get-credentials \
  --resource-group <RESOURCE_GROUP_NAME> \
  --name <CLUSTER_NAME>      # from az stack group show --query 'outputs.clusterName.value'

helm upgrade --install reprolab infra/azure/helm/ \
  -f infra/azure/envs/deepinvent/helm-values.yaml \
  --namespace reprolab --create-namespace
```

### 7. Build and push the base image

```bash
scripts/azure_build_cell_image.sh   # wraps az acr build -r <acr-name>
```

### 8. Smoke test (hello-GPU Job)

```bash
kubectl apply -f infra/azure/helm/smoke/hello-gpu-job.yaml
kubectl wait --for=condition=complete job/hello-gpu --timeout=600s -n reprolab
```

The GPU node pool should scale 0→1, run `nvidia-smi`, write a result to Blob, and scale back to 0.

---

## Key design decisions

- **Public API server + authorized IP ranges:** In Phase 1 the orchestrator runs locally (outside the VNet), so a private API server would require a VPN or bastion. The API server is **public but IP-restricted** to operator CIDRs only. Confirm DeepInvent's security policy permits this. Becomes private in Phase 2 when the control plane moves in-cluster.
- **Scale-to-zero GPU pool:** `minCount = 0`. Idle cost = $0. The Cluster Autoscaler scales 0→N as Jobs request `nvidia.com/gpu: 1`.
- **Zero static secrets:** No storage account keys in outputs or Kubernetes Secrets. All auth flows through managed identity + workload identity. The one documented exception: manual one-off operations may temporarily use `az storage account keys list`; rotate the key immediately after.
- **On-demand nodes only:** Spot nodes reduce cost but can be preempted mid-training. Deferred to Phase 2 with in-Job checkpoint/resume support.
- **Bicep deployment stacks:** State is managed server-side by ARM. No tfstate file, no bootstrap storage account for state, no `init` step.

---

## Workload Identity — federated credential subject format

The federated credential subject is **exactly**:

```
system:serviceaccount:<namespace>:<service-account-name>
```

with audience `api://AzureADTokenExchange`.

This is set in `bicep/modules/identity.bicep`. Both the `workloadIdentityNamespace` and `workloadIdentityServiceAccount` parameters must match the Helm L2 ServiceAccount exactly. A mismatch produces silent 401 errors in pod auth — hard to debug.

---

## Outputs surfaced to Helm L2

Retrieve with `az stack group show --name openresearch-infra --resource-group <RG> --query outputs` or `scripts/azure_wire_env.sh`.

| Bicep output | Purpose |
|-------------|---------|
| `clusterName` | `az aks get-credentials --name` |
| `oidcIssuerUrl` | Additional federated credentials |
| `nodeResourceGroup` | Quota / capacity checks |
| `kubeletIdentityClientId` | AcrPull + Files role confirmation |
| `workloadIdentityClientId` | Helm SA annotation (critical) |
| `acrLoginServer` | `OPENRESEARCH_AZURE_ACR_LOGIN_SERVER` |
| `storageAccountNameOut` | `OPENRESEARCH_AZURE_STORAGE_ACCOUNT` (Blob artifact bus) |
| `blobContainerNameOut` | `OPENRESEARCH_AZURE_BLOB_CONTAINER` |
| `filesShareNameOut` | `OPENRESEARCH_AZURE_FILES_SHARE` |
| `filesStorageAccountName` | Helm `storage.filesAccountName` — the account hosting the active Files share. Same as `storageAccountNameOut` when `filesPremium=false`; the dedicated Premium FileStorage account when `filesPremium=true`. Always pass this (not `storageAccountNameOut`) to the StorageClass. |
| `gpuNodepoolName` | `OPENRESEARCH_AZURE_NODE_POOL_NAME` (deprecated; prefer `gpuPools[0].name`) |
| `gpuPools` | Array of `{name, skuLabel}` parallel to `gpuSkus` input |
| `logAnalyticsWorkspaceId` | Log Analytics workspace (monitoring baseline) |
