# ReproLab GKE Helm chart (L2 in-cluster scaffold)

This chart installs the **static in-cluster scaffold** required by the ReproLab
GCP GKE GPU execution backend.  It is Layer 2 in the layer split (L1 = Terraform,
L2 = this chart, L3 = runtime Jobs emitted by `gke_job_backend`).

> **Lint / apply note:** run `helm lint` and `helm template --debug` locally
> before the first `helm install`, and verify the rendered YAML against the
> assertions below.

---

## Resources installed

| Template | Kind | Purpose |
|---|---|---|
| `namespace.yaml` | Namespace | Isolates all ReproLab workloads |
| `serviceaccount.yaml` | ServiceAccount | Annotated `iam.gke.io/gcp-service-account` for Workload Identity |
| `storageclass-filestore.yaml` | StorageClass | Filestore CSI RWX (only when `storage.filestoreShare` set) |
| `pvc-cache.yaml` | PersistentVolume + PVC | Shared RWX HF_HOME + pip cache (only when `storage.filestoreShare` set) |
| `role.yaml` | Role | Least-privilege Job CRUD + pod/log/events read |
| `rolebinding.yaml` | RoleBinding | Binds Role to the operator IAM members (User/Group) |
| `resourcequota.yaml` | ResourceQuota | Caps GPU requests/limits + Job count |
| `nvidia-device-plugin.yaml` | DaemonSet | **Disabled by default** — GKE manages the device plugin; parity stub only |

---

## Install sequence

**Prerequisites:**

1. `terraform apply` has completed on `infra/gcp/`.
2. `gcloud container clusters get-credentials <cluster> --region <region> --project <project>` has been run.
3. Workload Identity is enabled (set by the Terraform `gke` module).
4. The Filestore CSI driver is enabled (default on recent GKE Standard clusters) — only needed when `storage.filestoreShare` is set.

**Install:**

```bash
cd infra/gcp
GSA_EMAIL=$(terraform output -raw workload_identity_gcp_service_account)
GCS_BUCKET=$(terraform output -raw gcs_bucket_name)
AR_IMAGE=$(terraform output -raw artifact_registry_url)/gke-cell-base:0.1.0

helm install reprolab-gke ./infra/gcp/helm \
  --set workloadIdentity.gcpServiceAccount="${GSA_EMAIL}" \
  --set storage.bucket="${GCS_BUCKET}" \
  --set image.gkeCellBase="${AR_IMAGE}" \
  --set rbac.operatorMembers='{user:your-operator@example.com}' \
  --set resourceQuota.maxGpu=8 \
  --set resourceQuota.maxGpuLimits=8 \
  --set resourceQuota.maxJobs=64
# When filestore_enabled=true, also pass:
#   --set storage.filestoreShare="$(terraform output -raw filestore_share)" \
#   --set storage.filestoreIp="$(terraform output -raw filestore_ip)"

# Verify
kubectl get namespace reprolab
kubectl get serviceaccount reprolab-sa -n reprolab -o yaml | grep "gcp-service-account"
kubectl get role reprolab-orchestrator -n reprolab
kubectl get resourcequota reprolab-gpu-quota -n reprolab
```

**Uninstall:**

```bash
helm uninstall reprolab-gke
# Note: the Filestore PVC/PV are NOT deleted (reclaimPolicy: Retain). The
# underlying Filestore instance is managed by Terraform, not Helm.
```

---

## Values ↔ Terraform output mapping

| values.yaml key | Source | Description |
|---|---|---|
| `workloadIdentity.gcpServiceAccount` | TF output `workload_identity_gcp_service_account` | GSA email annotated onto the KSA (keyless pod auth) |
| `storage.bucket` | TF output `gcs_bucket_name` | GCS artifact bus bucket |
| `storage.filestoreShare` | TF output `filestore_share` | Filestore share name (empty ⇒ no cache PVC) |
| `storage.filestoreIp` | TF output `filestore_ip` | Filestore instance IP (required when filestoreShare set) |
| `image.gkeCellBase` | TF output `artifact_registry_url` + `/gke-cell-base:<tag>` | Job pod image ref (PINNED tag) |
| `rbac.operatorMembers` | tfvar `operator_iam_members` (not a TF output) | `user:`/`group:` members bound to the orchestrator Role |
| `resourceQuota.maxGpu` | sum of `gpu_count × max_nodes` across `gpu_skus` | GPU request cap |
| `gpu.skuLabelKey` | TF output `gpu_sku_label_key` (constant `reprolab/sku`) | Job nodeSelector key |
| `gpu.taintKey` | TF output `gpu_taint_key` (constant `nvidia.com/gpu`) | GPU node taint key |
| `gpu.defaultSku` | TF output `gpu_pools` (first sku_label) | Default pool for smoke jobs |
| `serviceAccountName` | tfvar `workload_identity_service_account` | Must match WI binding subject |
| `namespace` | tfvar `workload_identity_namespace` | Must match WI binding subject |

**Workload Identity member** (set in Terraform `identity` module) must exactly equal:

```
serviceAccount:<project_id>.svc.id.goog[<namespace>/<serviceAccountName>]
```

with the KSA annotation `iam.gke.io/gcp-service-account: <gsa-email>`.

---

## Filestore static mount

When `storage.filestoreShare` is set, the chart renders a **static** PV + PVC
that mount the EXISTING Terraform-provisioned Filestore share (we want the single
shared cache, not a per-PVC dynamically-provisioned share). The PV's
`csi.volumeAttributes` carry `ip` (`storage.filestoreIp`) and `volume`
(`storage.filestoreShare`). Verify the rendered `volumeHandle` matches your
Filestore instance's expected handle before applying; adjust per the GKE
Filestore CSI docs if your cluster expects the fully-qualified instance path.

When `storage.filestoreShare` is empty (default, `filestore_enabled = false`),
NO StorageClass / PV / PVC is rendered and Jobs use an emptyDir / GCS-only cache.

---

## Smoke jobs

Both smoke manifests live in `smoke/` and are NOT installed by `helm install` —
they are one-shot gate tests applied manually with `kubectl apply`.

| File | Purpose | GPU quota needed? |
|---|---|---|
| `smoke/hello-gpu-job.yaml` | Phase-1a gate: GPU scale 0→1, nvidia-smi, GCS write | YES — after quota is granted |
| `smoke/cpu-stub-job.yaml` | CPU fallback: GCS bus + Filestore PVC + WI validation | NO — use while quota is pending |

Neither smoke manifest contains any credentials or static secrets. All GCP auth
uses Workload Identity (the KSA annotation + GKE metadata server) — there is NO
per-pod label on GKE (unlike Azure's `azure.workload.identity/use`).

---

## Security / correctness assertions

- **No static secrets:** no Secret of any kind is created by this chart; GCS is
  reached via Workload Identity (KSA → GSA) and ADC.
- **Workload Identity:** the KSA carries `iam.gke.io/gcp-service-account`; the
  Terraform identity module grants `roles/iam.workloadIdentityUser` on the GSA
  to `<project>.svc.id.goog[<namespace>/<ksa>]`.
- **Least-privilege RBAC:** the Role is namespaced (not ClusterRole) and grants
  only `jobs` create/get/list/watch/delete; `pods` + `pods/log` + `events`
  get/list/watch. No patch, update, or cluster-wide permissions.
- **GKE-managed GPU device plugin:** installed by the node-pool driver config; no
  hand-rolled DaemonSet (the parity stub is gated `devicePlugin.enabled`, default
  false — do NOT enable on a standard GKE cluster).

---

## Cross-references

- Terraform L1: `infra/gcp/`
- Runtime Jobs (L3): `backend/services/runtime/gke_job_backend.py`
- GCS helpers: `backend/services/runtime/gcs_blob.py`
- Settings: `backend/config.py` (`gcp_*` block)
