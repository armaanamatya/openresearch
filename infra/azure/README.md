# ReproLab — Azure AKS GPU Backend: Terraform L1

**Client:** DeepInvent  
**Design lock:** 2026-06-03  
**Layer:** L1 (Azure infrastructure). See §3-layer split below.

---

## Why three layers — and why transient Jobs are NOT here

| Layer | What | Managed by | Lifecycle |
|-------|------|-----------|-----------|
| **L0 — Access bootstrap** (`infra/azure/bicep/`) | Subscription-scope RG creation + Contributor / User Access Administrator grants to the operator principal | **Bicep** (subscription admin, once) | One-time; stable |
| **L1 — Azure infra** (this directory) | Resource group, VNet/subnet, AKS cluster, GPU node pool, ACR, storage (Blob + Files), managed identity + role assignments, remote state | **Terraform** | Provisioned once; rarely changes |
| **L2 — In-cluster scaffold** (`infra/azure/helm/`) | Namespace, workload-identity ServiceAccount, Files PVC, orchestrator RBAC, NVIDIA device plugin, ResourceQuota | **Helm** (applied once per cluster) | Per-cluster; semi-static |
| **L3 — Runtime** | Per-cell Kubernetes **Jobs** | **Orchestrator code** (`backend/agents/rlm/k8s_job_cell_runner.py`) | Thousands; transient |

**L0 note:** Step 0 below currently requires subscription **Owner** because Terraform creates in-RG role assignments. Running the Bicep L0 bootstrap (`infra/azure/bicep/`) once lets the operator principal hold only RG-scoped Contributor + User Access Administrator — no subscription-level Owner needed for day-to-day `terraform apply`. See `infra/azure/bicep/README.md` for the full deploy + Terraform import handshake.

Terraform managing transient K8s Jobs is an anti-pattern: it causes state churn, drift, and `terraform apply` contention during active runs. The boundary between L1/L2 and L3 is deliberate. Only the durable substrate (cluster, pools, storage, identity) lives here.

---

## Module tree

```
infra/azure/
  bootstrap/            One-time: state RG + Standard LRS account + private tfstate container
  modules/
    network/            VNet, AKS subnet, NSG + association
    aks/                Managed-identity AKS; public API restricted by authorized IP ranges;
                        OIDC + workload identity ENABLED; small autoscaled CPU system pool;
                        Azure Files CSI enabled
    gpu_nodepool/       azurerm_kubernetes_cluster_node_pool; User mode;
                        Standard_NC24ads_A100_v4; on-demand; autoscale min=0 / max=N;
                        GPU taint + labels
    acr/                Standard ACR; admin disabled; AcrPull → kubelet identity
    storage/            Storage account; private Blob container (artifact bus);
                        Azure Files share (RWX cache); File Data SMB Contributor → kubelet
    identity/           User-assigned MI; federated credential (SA↔MI);
                        Storage Blob Data Contributor scoped to artifact container
  envs/deepinvent/      Client-specific placeholder files (*.example — copy before use)
  versions.tf           Provider pins
  providers.tf          azurerm + azuread provider blocks
  backend.tf            Remote state backend (partial config — see backend.hcl.example)
  variables.tf          All root variable declarations
  main.tf               Module wiring
  outputs.tf            All root outputs (consumed by Helm L2 values)
```

---

## Step-by-step: bootstrap → init → plan → apply

### 0. Prerequisites

- `az login` as an account with **Owner** on the target subscription (needed for role assignments)
- Terraform >= 1.6.0 (`terraform version`)
- GPU quota filed **before** apply:
  ```bash
  az quota create \
    --resource-name standardNCADSA100v4Family \
    --scope /subscriptions/<SUBSCRIPTION_ID>/providers/Microsoft.Compute/locations/<REGION> \
    --limit-object value=<24 * gpu_max_nodes>
  ```
  Quota approval can take hours to days on a fresh subscription. Start this immediately.

### 1. Bootstrap (one-time, local state)

```bash
cd infra/azure/bootstrap

# Create backend.tfvars for bootstrap (not committed):
cat > bootstrap.tfvars <<EOF
subscription_id            = "<SUBSCRIPTION_ID>"
tenant_id                  = "<TENANT_ID>"
region                     = "<REGION>"
state_storage_account_name = "<GLOBALLY_UNIQUE_STATE_SA_NAME>"
EOF

terraform init
terraform apply -var-file=bootstrap.tfvars
```

Bootstrap creates the RG + storage account + container that holds all subsequent `.tfstate` files. **Keep `bootstrap/terraform.tfstate` in a secrets manager — not git.**

### 2. Configure DeepInvent environment

```bash
cd infra/azure

cp envs/deepinvent/backend.hcl.example   envs/deepinvent/backend.hcl
cp envs/deepinvent/main.tfvars.example   envs/deepinvent/main.tfvars
```

Edit both files, filling every `<PLACEHOLDER>`. See comments in each file.

### 3. Init against remote state

```bash
terraform init -backend-config=envs/deepinvent/backend.hcl
```

### 4. Plan

```bash
terraform plan -var-file=envs/deepinvent/main.tfvars
```

Review the plan carefully — especially role assignments and the GPU node pool SKU.

### 5. Apply

```bash
terraform apply -var-file=envs/deepinvent/main.tfvars
```

### 6. Populate Helm L2 values

```bash
terraform output -json > /tmp/tf-outputs.json

cp envs/deepinvent/helm-values.yaml.example envs/deepinvent/helm-values.yaml
# Manually copy terraform output values into helm-values.yaml:
#   workload_identity_client_id  → workloadIdentity.clientId
#   storage_account_name         → storage.accountName
#   files_storage_account_name   → storage.filesAccountName
#                                   (same as storage_account_name when files_premium=false;
#                                    dedicated Premium account when files_premium=true)
#   files_share_name             → storage.fileShareName
#   gpu_max_nodes (from tfvars)  → resourceQuota.maxGpu / maxGpuLimits / maxJobs
#   operator_entra_group_object_id (from tfvars) → rbac.operatorGroupObjectId
```

### 7. Apply Helm L2

```bash
az aks get-credentials \
  --resource-group $(terraform output -raw resource_group_name) \
  --name $(terraform output -raw cluster_name)

helm upgrade --install reprolab infra/azure/helm/ \
  -f infra/azure/envs/deepinvent/helm-values.yaml \
  --namespace reprolab --create-namespace
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
- **Scale-to-zero GPU pool:** `min_count = 0`. Idle cost = $0. The Cluster Autoscaler scales 0→N as Jobs request `nvidia.com/gpu: 1`.
- **Zero static secrets:** No storage account keys in outputs, tfvars, or Kubernetes Secrets. All auth flows through managed identity + workload identity. The one documented exception: manual one-off operations may temporarily use `az storage account keys list`; rotate the key immediately after.
- **On-demand nodes only:** Spot nodes reduce cost but can be preempted mid-training. Deferred to Phase 2 with in-Job checkpoint/resume support.

---

## Workload Identity — federated credential subject format

The federated credential subject is **exactly**:

```
system:serviceaccount:<namespace>:<service-account-name>
```

with audience `api://AzureADTokenExchange`.

This is set in `modules/identity/main.tf`. Both the `namespace` and `service_account_name` variables must match the Helm L2 ServiceAccount exactly. A mismatch produces silent 401 errors in pod auth — hard to debug.

---

## Client-supplied placeholders (required before apply)

| Variable / file | Placeholder | Notes |
|----------------|-------------|-------|
| `main.tfvars` | `subscription_id` | DeepInvent subscription |
| `main.tfvars` | `tenant_id` | DeepInvent AAD tenant |
| `main.tfvars` | `region` | Must have A100 quota |
| `main.tfvars` | `prefix` | ≤8 chars, lowercase |
| `main.tfvars` | `authorized_ip_ranges` | All operator egress IPs |
| `main.tfvars` | `kubernetes_version` | Available in region |
| `main.tfvars` | `operator_entra_group_object_id` | AKS admin group |
| `main.tfvars` | `storage_account_name` | Globally unique |
| `main.tfvars` | `gpu_max_nodes` | Cost-driven; 1 to start |
| `backend.hcl` | `subscription_id`, `storage_account_name` | From bootstrap output |
| `helm-values.yaml` | `workloadIdentityClientId` | From `terraform output` |
| `helm-values.yaml` | `operatorGroupObjectId` | Same as tfvars |

---

## Validation note

`terraform fmt`, `terraform validate`, and `terraform plan` **could not be run during authoring** — no Terraform binary and no Azure credentials are present in the build environment. The operator MUST run these commands against a real Azure subscription before `terraform apply`. Required steps:

1. `terraform fmt -recursive` — normalise formatting
2. `terraform validate` — provider-level schema check
3. `terraform plan -var-file=envs/deepinvent/main.tfvars` — produces the full change-set for review

Report any validation errors back to the engineering team before proceeding to apply.

---

## Outputs surfaced to Helm L2

| Output | Purpose |
|--------|---------|
| `cluster_name` | `az aks get-credentials --name` |
| `oidc_issuer_url` | Additional federated credentials |
| `node_resource_group` | Quota / capacity checks |
| `kubelet_identity_client_id` | AcrPull + Files role confirmation |
| `workload_identity_client_id` | Helm SA annotation (critical) |
| `acr_login_server` | `REPROLAB_AZURE_ACR_LOGIN_SERVER` |
| `storage_account_name` | `REPROLAB_AZURE_STORAGE_ACCOUNT` (Blob artifact bus account) |
| `blob_container_name` | `REPROLAB_AZURE_BLOB_CONTAINER` |
| `files_share_name` | `REPROLAB_AZURE_FILES_SHARE` |
| `files_storage_account_name` | Helm `storage.filesAccountName` — the account hosting the active Files share. Same as `storage_account_name` when `files_premium=false`; the dedicated Premium FileStorage account when `files_premium=true`. Always pass this (not `storage_account_name`) to the StorageClass. |
| `gpu_nodepool_name` | `REPROLAB_AZURE_NODE_POOL_NAME` |
