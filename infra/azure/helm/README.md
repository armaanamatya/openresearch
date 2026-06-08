# ReproLab AKS Helm chart (L2 in-cluster scaffold)

This chart installs the **static in-cluster scaffold** required by the ReproLab
Azure AKS GPU execution backend.  It is Layer 2 in the three-layer split
(L1 = Terraform, L2 = this chart, L3 = runtime Jobs emitted by
`k8s_job_cell_runner`).

> **Lint / apply note:** `helm lint`, `helm template`, and `kubectl apply` could
> NOT be run during chart authoring — no `helm` or `kubectl` binary is present in
> the dev environment.  The operator MUST run `helm lint` and `helm template
> --debug` locally before the first `helm install`, and verify the rendered YAML
> against the assertions in this document.

---

## Resources installed

| Template | Kind | Purpose |
|---|---|---|
| `namespace.yaml` | Namespace | Isolates all ReproLab workloads |
| `serviceaccount.yaml` | ServiceAccount | Annotated with MI client-id for Workload Identity |
| `storageclass-files.yaml` | StorageClass | Keyless Azure Files CSI (no static key) |
| `pvc-cache.yaml` | PersistentVolumeClaim | Shared RWX HF_HOME + pip cache across all Job pods |
| `role.yaml` | Role | Least-privilege Job CRUD + pod/log/events read |
| `rolebinding.yaml` | RoleBinding | Binds Role to the operator Entra group |
| `nvidia-device-plugin.yaml` | DaemonSet | Exposes GPU devices to Kubernetes on GPU nodes |
| `resourcequota.yaml` | ResourceQuota | Caps GPU requests/limits + Job count |

---

## Install sequence

**Prerequisites:**

1. `terraform apply` has completed successfully on `infra/azure/`.
2. `az aks get-credentials --resource-group <rg> --name <cluster>` has been run.
3. The AKS Workload Identity webhook is enabled (enabled by the Terraform `aks` module).
4. The Azure Files CSI driver is enabled (bundled with AKS >= 1.21, enabled in Terraform).

**Install:**

```bash
# 1. Collect Terraform outputs (run from infra/azure/)
cd infra/azure
MI_CLIENT_ID=$(terraform output -raw workload_identity_client_id)
STORAGE_ACCOUNT=$(terraform output -raw storage_account_name)
FILES_SHARE=$(terraform output -raw files_share_name)
ACR_IMAGE=$(terraform output -raw acr_login_server)/aks-cell-base:0.1.0
OPERATOR_GROUP_OID=<your-entra-operator-group-object-id>   # from main.tfvars, not a TF output

# 2. Install the chart
helm install reprolab-aks ./infra/azure/helm \
  --set workloadIdentity.clientId="${MI_CLIENT_ID}" \
  --set storage.accountName="${STORAGE_ACCOUNT}" \
  --set storage.fileShareName="${FILES_SHARE}" \
  --set image.aksCellBase="${ACR_IMAGE}" \
  --set rbac.operatorGroupObjectId="${OPERATOR_GROUP_OID}" \
  --set resourceQuota.maxGpu=8 \
  --set resourceQuota.maxGpuLimits=8 \
  --set resourceQuota.maxJobs=64

# 3. Verify
kubectl get namespace reprolab
kubectl get serviceaccount reprolab-sa -n reprolab -o yaml | grep "workload.identity"
kubectl get storageclass reprolab-azurefiles-wi
kubectl get pvc reprolab-cache -n reprolab
kubectl get role reprolab-orchestrator -n reprolab
kubectl get resourcequota reprolab-gpu-quota -n reprolab
kubectl get daemonset nvidia-device-plugin -n reprolab
```

**Upgrade:**

```bash
helm upgrade reprolab-aks ./infra/azure/helm --reuse-values \
  --set workloadIdentity.clientId="${MI_CLIENT_ID}" \
  ...
```

**Uninstall:**

```bash
helm uninstall reprolab-aks
# Note: the PVC and its underlying Files share are NOT deleted (reclaimPolicy: Retain).
# Delete manually if you want to release the quota:
kubectl delete pvc reprolab-cache -n reprolab
```

---

## Values ↔ Terraform output mapping

| values.yaml key | Source | Description |
|---|---|---|
| `workloadIdentity.clientId` | TF output `workload_identity_client_id` | User-assigned MI client ID for Workload Identity |
| `storage.accountName` | TF output `storage_account_name` | Storage account hosting Blob container + Files share |
| `storage.fileShareName` | TF output `files_share_name` | Azure Files share name for the RWX cache PVC |
| `image.aksCellBase` | TF output `acr_login_server` + image tag | Full ACR image ref for Job pods |
| `rbac.operatorGroupObjectId` | tfvar `operator_entra_group_object_id` (not a TF output) | Entra group OID bound to the orchestrator Role |
| `resourceQuota.maxGpu` | tfvar `gpu_max_nodes` (not a TF output) | Match to max node count for 1:1 GPU-per-Job |
| `resourceQuota.maxGpuLimits` | tfvar `gpu_max_nodes` (not a TF output) | Same as maxGpu for GPU workloads |
| `gpu.nodePoolLabelKey` | TF output `gpu_node_pool_label_key` (module constant: `"reprolab/node-type"`) | Node label key applied to GPU pool nodes |
| `gpu.nodePoolLabelValue` | TF output `gpu_node_pool_label_value` (module constant: `"gpu"`) | Node label value for the GPU pool |
| `gpu.taintKey` | TF output `gpu_taint_key` (module constant: `"nvidia.com/gpu"`) | Taint key on GPU nodes (value `present`, effect `NoSchedule`) |
| `serviceAccountName` | tfvar `workload_identity_service_account` (not a TF output) | Must match SA in federated credential subject |
| `namespace` | tfvar `workload_identity_namespace` (not a TF output) | Must match namespace in federated credential subject |

**Federated credential subject** (set in Terraform `identity` module) must exactly equal:

```
system:serviceaccount:<namespace>:<serviceAccountName>
```

---

## Workload Identity verification

After `helm install`, verify the identity chain end-to-end:

```bash
# 1. Confirm SA annotation
kubectl get sa reprolab-sa -n reprolab \
  -o jsonpath='{.metadata.annotations.azure\.workload\.identity/client-id}'
# Expected: <workload_identity_client_id value>

# 2. Run the CPU stub smoke job (no GPU quota needed)
kubectl apply -f infra/azure/helm/smoke/cpu-stub-job.yaml
kubectl logs -n reprolab job/cpu-stub-smoke -f
# Expected: "Blob write OK" + "Files PVC OK"

# 3. If GPU quota is granted, run the GPU smoke job
kubectl apply -f infra/azure/helm/smoke/hello-gpu-job.yaml
kubectl logs -n reprolab job/hello-gpu-smoke -f
# Expected: "nvidia-smi OK" + "blob write OK" + pool scales 0→1→0 after TTL
```

Fill in the `REPLACE_WITH_*` placeholders in each smoke YAML before applying.

---

## Smoke jobs

Both smoke manifests live in `smoke/` and are NOT installed by `helm install` —
they are one-shot gate tests applied manually with `kubectl apply`.

| File | Purpose | GPU quota needed? |
|---|---|---|
| `smoke/hello-gpu-job.yaml` | Phase-1a gate: GPU scale 0→1, nvidia-smi, Blob write | YES — use only after quota is granted |
| `smoke/cpu-stub-job.yaml` | CPU fallback: Blob bus + Files PVC + WI token validation | NO — use while quota is pending |

Neither smoke manifest contains any credentials or static secrets.
All Azure auth uses Workload Identity (projected OIDC token via
`azure.workload.identity/use: "true"` pod label + annotated ServiceAccount).

---

## Security / correctness assertions

- **No static secrets:** the StorageClass sets `storeAccountKey: "false"`.
  No Kubernetes Secret of any kind is created by this chart.
- **Workload Identity:** every pod that calls Azure APIs (Job pods, both smoke
  jobs) carries `azure.workload.identity/use: "true"` on its pod template.
- **Least-privilege RBAC:** the Role is namespaced (not ClusterRole) and grants
  only what `k8s_job_cell_runner` needs: `jobs` create/get/list/watch/delete;
  `pods` + `pods/log` + `events` get/list/watch.  No patch, update, or
  cluster-wide permissions.
- **GPU scheduling guard:** the NVIDIA device plugin DaemonSet has both a
  `nodeSelector` targeting the GPU pool label AND a `toleration` for the GPU
  taint — it will not land on CPU nodes.

---

## Cross-references

- Design doc: `docs/superpowers/specs/2026-06-03-azure-aks-gpu-backend-design.md` (§3 layer split, §6 L2 list, §7 auth)
- Terraform L1: `infra/azure/` (owners: W1-E)
- Runtime Jobs (L3): `backend/agents/rlm/k8s_job_cell_runner.py` (owners: W1-B)
- Runbook: `docs/runbooks/2026-06-03-azure-aks-gpu-backend-handoff.md`
