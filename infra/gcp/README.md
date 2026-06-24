# ReproLab — GCP GKE GPU Backend: Terraform L1

**Client:** DeepInvent  
**Layer:** L1 (GCP infrastructure). GCP-native mirror of `infra/azure/`.

---

## Why layers — and why transient Jobs are NOT here

| Layer | What | Managed by | Lifecycle |
|-------|------|-----------|-----------|
| **L0 — Bootstrap** (`infra/gcp/bootstrap/`) | GCS remote-state bucket + optional keyless CI/deploy service account | **Terraform** (local state, once) | One-time; stable |
| **L1 — GCP infra** (this directory) | VPC + subnet (+ secondary ranges) + Cloud NAT, GKE cluster, GPU node pools, Artifact Registry, GCS bucket (+ optional Filestore), workload-identity GSA + IAM bindings, remote state | **Terraform** | Provisioned once; rarely changes |
| **L2 — In-cluster scaffold** (`infra/gcp/helm/`) | Namespace, workload-identity ServiceAccount, optional Filestore StorageClass + RWX PVC, orchestrator RBAC, ResourceQuota | **Helm** (applied once per cluster) | Per-cluster; semi-static |
| **L3 — Runtime** | Per-cell Kubernetes **Jobs** | **Orchestrator code** (`backend/services/runtime/gke_job_backend.py`) | Thousands; transient |

Terraform managing transient K8s Jobs is an anti-pattern (state churn, drift, apply contention during active runs). Only the durable substrate lives here.

---

## Azure → GCP primitive mapping

| Azure (`infra/azure/`) | GCP (`infra/gcp/`) |
|---|---|
| AKS (`azurerm_kubernetes_cluster`) | GKE (`google_container_cluster`, VPC-native, Workload Identity) |
| AKS node pool (`azurerm_kubernetes_cluster_node_pool`) | GKE node pool (`google_container_node_pool`, scale-to-zero, GPU driver auto-install) |
| ACR (`azurerm_container_registry`) | Artifact Registry (`google_artifact_registry_repository`, DOCKER) |
| Azure Blob container (`azurerm_storage_container`) | GCS bucket (`google_storage_bucket`, uniform access, public-access-prevention) |
| Azure Files share (`azurerm_storage_share`) | Filestore instance (`google_filestore_instance`, NFS RWX) — opt-in |
| User-assigned MI + federated credential | GSA + Workload Identity binding (`google_service_account` + `roles/iam.workloadIdentityUser`) |
| Entra group object ID | IAM members (`user:`/`group:`) bound in the Helm RoleBinding |

Device plugin: Azure ships an `nvidia-device-plugin.yaml` DaemonSet; **GKE manages the device plugin via the node-pool driver install** (`gpu_driver_installation_config`), so no DaemonSet is required.

---

## Module tree

```
infra/gcp/
  bootstrap/            One-time: GCS state bucket (versioned, uniform access) + optional keyless CI SA
  modules/
    network/            VPC (custom subnet), subnet + 2 secondary ranges (pods/services),
                        Cloud Router + NAT (private-node egress), internal firewall
    gke/                VPC-native GKE; Workload Identity; private nodes; IP-restricted
                        public control-plane endpoint; least-privilege node SA;
                        autoscaled CPU system pool (GKE_METADATA)
    gpu_nodepool/       google_container_node_pool; scale-to-zero (min=0); A2 machine;
                        guest_accelerator + DEFAULT driver auto-install; GPU taint + labels
    registry/           Artifact Registry DOCKER repo; artifactregistry.reader → node SA
    storage/            GCS bucket (private, versioned, lifecycle); optional Filestore RWX
    identity/           Workload-identity GSA; storage.objectAdmin on the bucket;
                        iam.workloadIdentityUser binding (KSA ↔ GSA)
  envs/deepinvent/      Client-specific placeholder files (*.example — copy before use)
  versions.tf           Provider pins (google + google-beta ~> 6.0)
  providers.tf          google + google-beta provider blocks
  backend.tf            GCS remote state (partial config — see backend.hcl.example)
  variables.tf          All root variable declarations
  main.tf               Module wiring
  outputs.tf            All root outputs (consumed by Helm L2 values)
  helm/                 L2 in-cluster scaffold chart
```

---

## Step-by-step: bootstrap → init → plan → apply → helm → image

### 0. Prerequisites

- `gcloud auth application-default login` (ADC for Terraform + the orchestrator).
- Terraform >= 1.6.0 (`terraform version`).
- A100 GPU quota filed **before** apply. Fresh projects ship with 0 A100 quota.
  ```bash
  # Request via the console: IAM & Admin → Quotas → filter by
  #   "NVIDIA A100 80GB GPUs" (for nvidia-a100-80gb) in your region,
  #   then "EDIT QUOTAS". Approval can take hours to days.
  ```
  Required quota = `gpu_count × max_nodes` per pool. Start with `max_nodes=1`.

### 1. Bootstrap (one-time, local state)

```bash
cd infra/gcp/bootstrap

cat > bootstrap.tfvars <<EOF
project_id        = "your-gcp-project"
region            = "us-central1"
state_bucket_name = "your-prefix-tfstate"
EOF

terraform init
terraform apply -var-file=bootstrap.tfvars
```

Bootstrap creates the GCS bucket that holds all subsequent state. **Keep `bootstrap/terraform.tfstate` in a secrets manager — not git.**

### 2. Configure the DeepInvent environment

```bash
cd infra/gcp
cp envs/deepinvent/backend.hcl.example  envs/deepinvent/backend.hcl
cp envs/deepinvent/main.tfvars.example  envs/deepinvent/main.tfvars
```

Edit both, filling every placeholder (`your-gcp-project`, `your-prefix`, operator IPs/emails). See comments in each file.

### 3. Init against remote state

```bash
terraform init -backend-config=envs/deepinvent/backend.hcl
```

### 4. Plan / 5. Apply

```bash
terraform plan  -var-file=envs/deepinvent/main.tfvars
terraform apply -var-file=envs/deepinvent/main.tfvars
```

Review the GPU node pool SKU + IAM bindings carefully.

### 6. Get cluster credentials

```bash
$(terraform output -raw gke_get_credentials_command)
# == gcloud container clusters get-credentials <cluster> --region <region> --project <project>
```

### 7. Install Helm L2

```bash
helm upgrade --install reprolab-gke ./infra/gcp/helm \
  --set workloadIdentity.gcpServiceAccount=$(terraform -chdir=infra/gcp output -raw workload_identity_gcp_service_account) \
  --set storage.bucket=$(terraform -chdir=infra/gcp output -raw gcs_bucket_name) \
  --set image.gkeCellBase=$(terraform -chdir=infra/gcp output -raw artifact_registry_url)/gke-cell-base:0.1.0 \
  --set rbac.operatorMembers='{user:your-operator@example.com}' \
  --namespace reprolab --create-namespace
# Add (only when filestore_enabled=true):
#   --set storage.filestoreShare=$(terraform -chdir=infra/gcp output -raw filestore_share) \
#   --set storage.filestoreIp=$(terraform -chdir=infra/gcp output -raw filestore_ip)
```

### 8. Build + push the gke-cell-base image to Artifact Registry

```bash
gcloud auth configure-docker $(terraform output -raw artifact_registry_url | cut -d/ -f1)
docker build -t $(terraform output -raw artifact_registry_url)/gke-cell-base:0.1.0 docker/gke-cell-base/
docker push  $(terraform output -raw artifact_registry_url)/gke-cell-base:0.1.0
```
(The `docker/gke-cell-base/` image is authored separately.)

### 9. Smoke test

```bash
kubectl apply -f infra/gcp/helm/smoke/cpu-stub-job.yaml   # no GPU quota needed
# once quota is granted:
kubectl apply -f infra/gcp/helm/smoke/hello-gpu-job.yaml
```

---

## Output → Helm / env mapping

| Output | Helm `--set` / env var | Purpose |
|--------|------------------------|---------|
| `gke_cluster_name` | `OPENRESEARCH_GCP_GKE_CLUSTER` | `gcloud container clusters get-credentials` |
| `gke_get_credentials_command` | — | Ready-to-run kubeconfig command |
| `workload_pool` | — | Additional WI bindings |
| `workload_identity_gcp_service_account` | `workloadIdentity.gcpServiceAccount` (KSA annotation) | **Critical** — keyless pod auth |
| `artifact_registry_url` | `image.gkeCellBase` (+ `/gke-cell-base:<tag>`), `OPENRESEARCH_GCP_ARTIFACT_REGISTRY` | Job image |
| `gcs_bucket_name` | `storage.bucket`, `OPENRESEARCH_GCP_GCS_BUCKET` | Artifact bus |
| `filestore_ip` / `filestore_share` | `storage.filestoreIp` / `storage.filestoreShare`, `OPENRESEARCH_GCP_FILESTORE_SHARE` | RWX cache (when enabled) |
| `gpu_pools` | smoke job nodeSelector | Map short_name → {sku_label, machine_type, gpu_count} |

These align with the `gcp_*` settings in `backend/config.py` (`gcp_project`, `gcp_region`, `gcp_gcs_bucket`, `gcp_gke_cluster`, `gcp_artifact_registry`, `gcp_filestore_share`, `gcp_namespace`, `gcp_service_account`, `gcp_base_image`).

---

## Key design decisions

- **Public control-plane endpoint + authorized networks:** in Phase 1 the orchestrator runs locally (outside the VPC), so the endpoint is public but IP-restricted to operator CIDRs. **Nodes are private** (no public IP; egress via Cloud NAT). Confirm DeepInvent's security policy permits this.
- **Scale-to-zero GPU pools:** `min_node_count = 0`. Idle cost = $0. The cluster autoscaler scales 0→N as Jobs request `nvidia.com/gpu`.
- **Zero static secrets:** NO service-account JSON key, NO GCS HMAC key. The orchestrator uses Application Default Credentials; Job pods use Workload Identity. GCS buckets use uniform bucket-level access + enforced public-access-prevention.
- **GKE-managed device plugin:** the GPU driver + device plugin are installed by GKE via the node-pool `gpu_driver_installation_config` — no DaemonSet.
- **On-demand nodes only:** spot deferred to a later phase with checkpoint/resume.

---

## Workload Identity — binding format

The IAM member granted `roles/iam.workloadIdentityUser` is **exactly**:

```
serviceAccount:<project_id>.svc.id.goog[<namespace>/<service-account-name>]
```

and the KSA carries the annotation `iam.gke.io/gcp-service-account: <gsa-email>`. Both `namespace` and `service_account_name` must match the Helm L2 ServiceAccount exactly — a mismatch produces silent 403s in pod auth.

---

## Validation note

`terraform fmt`, `terraform validate`, `terraform plan`, `helm lint`, and `helm template` were run during authoring **only where the binary was available**. The operator MUST run these against the real project before `terraform apply` / `helm install`:

1. `terraform fmt -recursive infra/gcp`
2. `terraform -chdir=infra/gcp validate`
3. `terraform -chdir=infra/gcp plan -var-file=envs/deepinvent/main.tfvars`
4. `helm lint infra/gcp/helm`

Report any errors back to the engineering team before proceeding.
