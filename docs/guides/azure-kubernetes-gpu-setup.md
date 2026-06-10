# Azure Kubernetes GPU Setup

How to run paper reproductions on **Azure GPUs** by dispatching each training
cell as a **Kubernetes Job** on an AKS cluster with a scale-to-zero A100 node
pool.

This is the `--sandbox azure` execution backend. It is an alternative to the
default `runpod` and `local` sandboxes — nothing else about the app changes.

> **New to the repo?** Read the root [`README.md`](../../README.md) Quick Start
> first to get the app running locally. This guide only covers the Azure path.

---

## What you get

- **Scale-to-zero GPUs.** The A100 node pool sits at 0 nodes (≈ $0/idle). It
  scales up only while Jobs are running, then back down.
- **One Job per training cell**, each pinned to one GPU, run in parallel up to
  the node-pool cap.
- **Shared weight cache.** All cells mount the same Azure Files share for
  `HF_HOME` / pip cache, so model weights download once.
- **No static secrets.** All Azure auth flows through Workload Identity — there
  are no storage keys in the cluster, in `.env`, or in git.

---

## Mental model — three layers

The setup splits cleanly into three layers. You provision the first two **once**;
the third is created automatically every run.

| Layer | What it is | Managed by | How often |
|-------|-----------|-----------|-----------|
| **L1 — Infra** | Resource group, AKS cluster, GPU node pool, ACR, storage, identity | **Terraform** (`infra/azure/`) | Once per cluster |
| **L2 — In-cluster scaffold** | Namespace, ServiceAccount, RBAC, GPU device plugin, cache PVC, quota | **Helm** (`infra/azure/helm/`) | Once per cluster |
| **L3 — Runtime Jobs** | One Kubernetes Job per training cell | **The orchestrator** (`backend/agents/rlm/k8s_job_cell_runner.py`) | Thousands; transient |

**How a run flows:**

```
Your laptop / CI                         Azure
┌─────────────────────┐                  ┌──────────────────────────────────┐
│ backend.cli         │                  │ AKS cluster (reprolab namespace) │
│   --sandbox azure   │  submit K8s Job  │                                  │
│        run_matrix() ├──────────────────▶  Job: cell qwen3_1b-alfworld-42  │
│                     │                  │   └─ pulls code + base image     │
│ polls Job status    │◀─────────────────┤      from ACR + Blob             │
│ pulls metrics.json  │   read Blob      │   └─ trains on 1× A100           │
└─────────────────────┘                  │   └─ writes metrics → Blob       │
                                         │ GPU pool scales 0→N→0            │
                                         └──────────────────────────────────┘
```

The orchestrator runs **on your machine** (outside the cluster). It submits
Jobs, watches them, and reads artifacts back from Blob. The GPUs live only in
Azure.

---

## Prerequisites

### Tools (install locally)

| Tool | Why | Check |
|------|-----|-------|
| [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) (`az`) | Auth + cluster credentials | `az version` |
| [Terraform](https://developer.hashicorp.com/terraform/install) ≥ 1.6 | Provision L1 infra | `terraform version` |
| [`kubectl`](https://kubernetes.io/docs/tasks/tools/) | Inspect the cluster | `kubectl version --client` |
| [Helm](https://helm.sh/docs/intro/install/) ≥ 3 | Install L2 scaffold | `helm version` |
| Docker | Build the cell base image | `docker info` |
| Python venv with `backend/requirements.txt` | The orchestrator (Azure SDKs are already in there) | `python -c "import azure.identity, azure.storage.blob, kubernetes"` |

### Azure access

- A subscription where you (or your service principal) have **Owner** — required
  to create the role assignments Workload Identity depends on.
- **GPU quota for the A100 family** — this gates everything, so file it first
  (see Step 1). Approval can take hours to days on a fresh subscription.

---

## Step-by-step

> Run all `terraform` / `helm` commands from the repo root unless noted.
> Replace every `<PLACEHOLDER>`.

### 1. Request GPU quota (do this first)

A fresh subscription has **zero** A100 quota. Request it before anything else,
because approval is the long pole.

```bash
az login

az quota create \
  --resource-name standardNCADSA100v4Family \
  --scope /subscriptions/<SUBSCRIPTION_ID>/providers/Microsoft.Compute/locations/<REGION> \
  --limit-object value=24   # 24 vCPU = one Standard_NC24ads_A100_v4 node (1× A100 80GB)
```

Scale `value` to `24 × (max GPU nodes you want)`. You can continue with the next
steps while the request is pending — you just can't run GPU Jobs until it lands.

### 2. Bootstrap remote Terraform state (one-time)

This creates the storage account that holds all subsequent `.tfstate`.

```bash
cd infra/azure/bootstrap

cat > bootstrap.tfvars <<'EOF'
subscription_id            = "<SUBSCRIPTION_ID>"
tenant_id                  = "<TENANT_ID>"
region                     = "<REGION>"
state_storage_account_name = "<GLOBALLY_UNIQUE_NAME>"
EOF

terraform init
terraform apply -var-file=bootstrap.tfvars
cd ../../..
```

> Keep `bootstrap/terraform.tfstate` out of git — store it in a secrets manager.

### 3. Configure your environment files

The repo ships `*.example` templates. Copy and fill them in:

```bash
cd infra/azure
cp envs/deepinvent/backend.hcl.example  envs/deepinvent/backend.hcl
cp envs/deepinvent/main.tfvars.example  envs/deepinvent/main.tfvars
```

Edit both, replacing every `<PLACEHOLDER>` (each line is commented). The key
fields in `main.tfvars`:

| Field | Notes |
|-------|-------|
| `subscription_id`, `tenant_id` | Your Azure IDs |
| `region` | Must have A100 quota (Step 1) |
| `prefix` | ≤ 8 chars, lowercase — name prefix for all resources |
| `authorized_ip_ranges` | Operator egress CIDRs allowed to reach the K8s API |
| `operator_entra_group_object_id` | Entra group granted cluster admin + Job RBAC |
| `storage_account_name` | Globally unique |
| `gpu_max_nodes` | Cost cap — start at `1` |

### 4. Provision L1 infra with Terraform

```bash
terraform init -backend-config=envs/deepinvent/backend.hcl
terraform plan  -var-file=envs/deepinvent/main.tfvars      # review role assignments + SKU
terraform apply -var-file=envs/deepinvent/main.tfvars
cd ../..
```

This creates the AKS cluster, the scale-to-zero A100 pool, ACR, Blob + Files
storage, and the managed identity. Details: [`infra/azure/README.md`](../../infra/azure/README.md).

### 5. Build and push the cell base image

Every Job pod boots from this image (CUDA runtime + PyTorch + Azure SDK).
Build it, push it to your ACR, and **pin the tag** — never use `:latest`.

```bash
ACR=$(terraform -chdir=infra/azure output -raw acr_login_server)   # e.g. myacr.azurecr.io
TAG=$(git rev-parse --short HEAD)

docker build -t reprolab-aks-cell-base docker/aks-cell-base/
docker tag reprolab-aks-cell-base "${ACR}/reprolab/aks-cell-base:${TAG}"

az acr login --name "${ACR%%.*}"
docker push "${ACR}/reprolab/aks-cell-base:${TAG}"
```

Remember `${ACR}/reprolab/aks-cell-base:${TAG}` — it becomes
`REPROLAB_AZURE_BASE_IMAGE` in Step 8. Contract details:
[`docker/aks-cell-base/README.md`](../../docker/aks-cell-base/README.md).

### 6. Get cluster credentials

```bash
az aks get-credentials \
  --resource-group $(terraform -chdir=infra/azure output -raw resource_group_name) \
  --name          $(terraform -chdir=infra/azure output -raw cluster_name)

kubectl get nodes   # should list the CPU system pool (GPU pool is at 0)
```

### 7. Install the L2 in-cluster scaffold with Helm

```bash
cd infra/azure
MI_CLIENT_ID=$(terraform output -raw workload_identity_client_id)
STORAGE_ACCOUNT=$(terraform output -raw storage_account_name)
FILES_ACCOUNT=$(terraform output -raw files_storage_account_name)
FILES_SHARE=$(terraform output -raw files_share_name)
ACR_IMAGE=$(terraform output -raw acr_login_server)/reprolab/aks-cell-base:<TAG_FROM_STEP_5>
OPERATOR_GROUP_OID=<operator_entra_group_object_id>   # same value as main.tfvars

helm upgrade --install reprolab-aks ./helm \
  --set workloadIdentity.clientId="${MI_CLIENT_ID}" \
  --set storage.accountName="${STORAGE_ACCOUNT}" \
  --set storage.filesAccountName="${FILES_ACCOUNT}" \
  --set storage.fileShareName="${FILES_SHARE}" \
  --set image.aksCellBase="${ACR_IMAGE}" \
  --set rbac.operatorGroupObjectId="${OPERATOR_GROUP_OID}" \
  --set resourceQuota.maxGpu=1 --set resourceQuota.maxGpuLimits=1 --set resourceQuota.maxJobs=8
cd ../..
```

Match `resourceQuota.maxGpu` to `gpu_max_nodes`. Full values reference:
[`infra/azure/helm/README.md`](../../infra/azure/helm/README.md).

### 8. Point the orchestrator at Azure

Add these to your repo-root `.env` (host-side; all use the `REPROLAB_` prefix —
see the block in [`.env.example`](../../.env.example)):

```bash
REPROLAB_AZURE_RESOURCE_GROUP=<resource_group_name>
REPROLAB_AZURE_AKS_CLUSTER=<cluster_name>
REPROLAB_AZURE_STORAGE_ACCOUNT=<storage_account_name>
REPROLAB_AZURE_ACR_LOGIN_SERVER=<acr_login_server>
REPROLAB_AZURE_BASE_IMAGE=<ACR>/reprolab/aks-cell-base:<TAG>   # pinned tag from Step 5
# Defaults below already match the Helm chart — override only if you changed them:
REPROLAB_AZURE_NAMESPACE=reprolab
REPROLAB_AZURE_SERVICE_ACCOUNT=reprolab-sa
REPROLAB_AZURE_NODE_POOL_NAME=gpua100
REPROLAB_AZURE_BLOB_CONTAINER=reprolab-artifacts
REPROLAB_AZURE_FILES_SHARE=reprolab-cache
```

### 9. Smoke test, then run

Confirm a GPU Job schedules end-to-end before a real run:

```bash
kubectl apply -f infra/azure/helm/smoke/hello-gpu-job.yaml
kubectl wait --for=condition=complete job/hello-gpu --timeout=900s -n reprolab
```

The GPU pool should scale 0→1, run `nvidia-smi`, then scale back to 0. Now run a
reproduction:

```bash
python -m backend.cli reproduce 2605.15155 --sandbox azure --model claude-oauth
```

The orchestrator runs `ensure_azure_available()` at startup — if any
prerequisite is missing it fails fast with the exact fix (see Troubleshooting).

---

## Troubleshooting

`ensure_azure_available()` checks prerequisites in order and prints an actionable
message. The common ones:

| Symptom | Cause | Fix |
|---------|-------|-----|
| `requires 'azure-identity' / 'kubernetes'` | Deps missing in the venv | `pip install -r backend/requirements.txt` |
| `REPROLAB_AZURE_BASE_IMAGE ... is not set` | No pinned image | Set it to the Step 5 tag (never `:latest`) |
| `DefaultAzureCredential could not acquire a token` | Not logged in | `az login` (or set `AZURE_CLIENT_ID/SECRET/TENANT_ID`) |
| `could not load kubeconfig` | No cluster creds | Re-run Step 6 (`az aks get-credentials`) |
| Cell stuck `Pending` then `capacity_exhausted` | GPU cold-start from 0 (10–12 min) or quota = 0 | Confirm Step 1 quota approved; the default pending timeout is 1500s |
| Pod 401 / silent auth failure | SA name ≠ federated-credential subject | `REPROLAB_AZURE_SERVICE_ACCOUNT` must equal the Helm `reprolab-sa` and the Terraform `system:serviceaccount:reprolab:reprolab-sa` |

Inspect a run live:

```bash
kubectl get jobs -n reprolab
kubectl logs -n reprolab job/<job-name> --follow
```

---

## Cost and teardown

- **Idle cost is ~$0** — the GPU pool is scale-to-zero. The CPU system pool and
  storage incur small standing charges.
- Per-GPU budget tracking uses `REPROLAB_AZURE_GPU_USD_PER_HOUR` (default `3.67`,
  the A100 on-demand list price). Set your negotiated rate. The per-run cap is
  `REPROLAB_MAX_RUN_GPU_USD`.
- **Tear down everything:**

  ```bash
  helm uninstall reprolab-aks -n reprolab
  kubectl delete pvc reprolab-cache -n reprolab    # PVC is Retain — delete to release the share
  terraform -chdir=infra/azure destroy -var-file=envs/deepinvent/main.tfvars
  ```

---

## Reference docs

| Doc | Scope |
|-----|-------|
| [`infra/azure/README.md`](../../infra/azure/README.md) | L1 Terraform module tree + variables |
| [`infra/azure/helm/README.md`](../../infra/azure/helm/README.md) | L2 Helm values ↔ Terraform output mapping |
| [`docker/aks-cell-base/README.md`](../../docker/aks-cell-base/README.md) | Cell image env-var + exit-code contract |
| [`docs/superpowers/specs/2026-06-03-azure-aks-gpu-backend-design.md`](../superpowers/specs/2026-06-03-azure-aks-gpu-backend-design.md) | Full design rationale |
