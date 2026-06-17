# ─── GCP project / location ──────────────────────────────────────────────────

variable "project_id" {
  description = "GCP project ID where all resources are created."
  type        = string
}

# ─── Naming & location ───────────────────────────────────────────────────────

variable "prefix" {
  description = "Short alphanumeric prefix prepended to every resource name (e.g. 'repro'). Keep ≤8 chars, lowercase, no leading digit (GCE resource names must start with a letter)."
  type        = string
}

variable "region" {
  description = "GCP region for all resources (e.g. 'us-central1'). Choose a region that has A100 GPU quota available (see gpu_skus for the accelerator/region quota note)."
  type        = string
  default     = "us-central1"
}

variable "labels" {
  description = "Map of labels applied to every resource (GCP analog of Azure tags). Keys/values must be lowercase, ≤63 chars, and match GCP label syntax."
  type        = map(string)
  default     = {}
}

# ─── Networking ──────────────────────────────────────────────────────────────
#
# VPC-native (alias-IP) GKE needs THREE ranges on the node subnet:
#   - the primary range  → node IPs
#   - a "pods" secondary range     → pod IPs
#   - a "services" secondary range → ClusterIP service IPs
# The secondary ranges are referenced by NAME in the GKE ip_allocation_policy.

variable "subnet_cidr" {
  description = "Primary CIDR of the GKE node subnet (node IPs), e.g. '10.0.0.0/22'."
  type        = string
  default     = "10.0.0.0/22"
}

variable "pods_secondary_cidr" {
  description = "Secondary CIDR for GKE pod IPs (VPC-native alias range), e.g. '10.4.0.0/14'. Size generously — one /14 covers ~256k pods."
  type        = string
  default     = "10.4.0.0/14"
}

variable "services_secondary_cidr" {
  description = "Secondary CIDR for GKE ClusterIP services (VPC-native alias range), e.g. '10.8.0.0/20'."
  type        = string
  default     = "10.8.0.0/20"
}

variable "authorized_ip_ranges" {
  description = "List of operator CIDR(s) allowed to reach the GKE public control-plane endpoint (e.g. ['203.0.113.0/32']). Required — the control-plane endpoint is public but IP-restricted via master_authorized_networks. Nodes themselves are private (no public IP)."
  type        = list(string)
}

# ─── GKE cluster ─────────────────────────────────────────────────────────────

variable "kubernetes_version" {
  description = "Minimum Kubernetes master version for the GKE cluster (e.g. '1.30'). Within the chosen release_channel GKE auto-upgrades; leave empty ('') to let the channel pick the default."
  type        = string
  default     = ""
}

variable "release_channel" {
  description = "GKE release channel: RAPID, REGULAR, STABLE, or UNSPECIFIED. REGULAR is the recommended default (auto-upgrade with a stability buffer)."
  type        = string
  default     = "REGULAR"

  validation {
    condition     = contains(["RAPID", "REGULAR", "STABLE", "UNSPECIFIED"], var.release_channel)
    error_message = "release_channel must be one of: RAPID, REGULAR, STABLE, UNSPECIFIED."
  }
}

variable "system_node_machine_type" {
  description = "Machine type for the system (CPU) node pool."
  type        = string
  default     = "e2-standard-4"
}

variable "system_node_min_count" {
  description = "Minimum nodes in the system pool."
  type        = number
  default     = 1
}

variable "system_node_max_count" {
  description = "Maximum nodes in the system pool."
  type        = number
  default     = 3
}

variable "operator_iam_members" {
  description = <<-EOT
    Google identities granted Job-management RBAC in the namespace via the Helm
    L2 RoleBinding (GCP analog of Azure's operator_entra_group_object_id).
    Each entry is a fully-qualified IAM member, e.g.:
      "user:alice@example.com"
      "group:reprolab-operators@example.com"
      "serviceAccount:ci@your-gcp-project.iam.gserviceaccount.com"
    The RoleBinding splits each on the prefix into a User / Group subject.
    Cluster access is via `gcloud container clusters get-credentials` (the GKE
    auth plugin maps the gcloud identity to a Kubernetes user).
  EOT
  type        = list(string)
  default     = []
}

# ─── GPU node pools ──────────────────────────────────────────────────────────
#
# gpu_skus is the primary control surface.  Each entry provisions one
# scale-to-zero GKE node pool, labeled reprolab/sku=<short_name>, so the
# orchestrator can target it by catalog short_name.
#
# Full A100 ladder (valid gpu_skus entries):
#   short_name      machine_type     accelerator_type    gpus
#   gcp_a100_40     a2-highgpu-1g     nvidia-tesla-a100   1
#   gcp_a100_40x2   a2-highgpu-2g     nvidia-tesla-a100   2
#   gcp_a100_40x4   a2-highgpu-4g     nvidia-tesla-a100   4
#   gcp_a100_40x8   a2-highgpu-8g     nvidia-tesla-a100   8
#   gcp_a100_80     a2-ultragpu-1g    nvidia-a100-80gb    1
#   gcp_a100_80x2   a2-ultragpu-2g    nvidia-a100-80gb    2
#   gcp_a100_80x4   a2-ultragpu-4g    nvidia-a100-80gb    4
#   gcp_a100_80x8   a2-ultragpu-8g    nvidia-a100-80gb    8
#
# QUOTA: Each entry consumes per-region A100 quota in the matching family.
#   nvidia-tesla-a100 (40 GB) → "NVIDIA A100 GPUs" quota in the region.
#   nvidia-a100-80gb          → "NVIDIA A100 80GB GPUs" quota in the region.
# The required GPU count per pool = gpu_count × max_nodes.  The default 8×A100-80
# pool (gpu_count=8, max_nodes=4) needs 32 "NVIDIA A100 80GB GPUs".
# START with max_nodes=1 (8 GPUs) until the quota request is granted — fresh
# projects ship with 0 A100 quota and approval can take hours to days.
#
# Each entry may set use_spot = true to back the pool with GKE Spot nodes
# (~60-91% cheaper; ~15-30 s reclaim notice).  Default false = on-demand.

variable "gpu_skus" {
  description = <<-EOT
    List of GPU SKU objects — one GKE scale-to-zero node pool is created per entry.
    Fields:
      short_name       — catalog identifier; written to the 'reprolab/sku' node
                         label and used by Job nodeSelector.  Must be unique.
      machine_type     — GCE A2 machine type backing the pool nodes
                         (e.g. 'a2-ultragpu-1g').
      accelerator_type — GCE GPU accelerator type
                         ('nvidia-a100-80gb' or 'nvidia-tesla-a100').
      gpu_count        — GPUs per node; written to the 'reprolab/gpu-count' label
                         and used to request nvidia.com/gpu resources.
      max_nodes        — maximum autoscaler node count for this pool. min is
                         always 0 (scale-to-zero).
      disk_size_gb     — boot disk size in GiB for nodes in this pool. Default
                         256 GiB is sufficient for the gke-cell-base image +
                         working dir; raise to 512 for SKUs that pull very large
                         images or have large local pip/HF cache overflow.
      use_spot         — when true, back this pool with GKE Spot nodes
                         (~60-91% cheaper; ~15-30 s reclaim notice; GKE adds the
                         cloud.google.com/gke-spot=true:NoSchedule taint, matched
                         by the runtime spot toleration). Default false =
                         on-demand.
    Default: a single 8×A100-80 pool (a2-ultragpu-8g) — ONE quota ask.  Add 40GB
    / other multi-GPU entries (each needing its own per-region quota) to enable
    the escalation ladder.  See the comment block above for the full catalog.
  EOT
  type = list(object({
    short_name       = string
    machine_type     = string
    accelerator_type = string
    gpu_count        = number
    max_nodes        = number
    disk_size_gb     = optional(number, 256)
    use_spot         = optional(bool, false)
  }))
  default = [
    {
      short_name       = "gcp_a100_80x8"
      machine_type     = "a2-ultragpu-8g"
      accelerator_type = "nvidia-a100-80gb"
      gpu_count        = 8
      max_nodes        = 4
      disk_size_gb     = 256
    }
  ]
}

# ─── Artifact Registry ───────────────────────────────────────────────────────

variable "artifact_registry_repo" {
  description = "Name of the Artifact Registry Docker repository that holds the gke-cell-base image. Created by the registry module; the repo URL is REGION-docker.pkg.dev/PROJECT/<repo>."
  type        = string
  default     = "reprolab"
}

# ─── Artifact + cache storage ────────────────────────────────────────────────

variable "gcs_bucket_name" {
  description = "Globally unique GCS bucket name (3-63 lowercase alphanum/hyphen). Hosts the artifact bus between the local orchestrator and GKE Jobs (runs/<id>/code/**, runs/<id>/cells/**)."
  type        = string
}

variable "filestore_share_name" {
  description = "Name of the Filestore file share mounted by Jobs as the RWX HuggingFace / pip cache."
  type        = string
  default     = "reprolab-cache"
}

variable "filestore_tier" {
  description = <<-EOT
    Filestore service tier for the cache instance.  BASIC_HDD (default) is the
    cheapest.  ZONAL or ENTERPRISE deliver substantially higher IOPS/throughput
    (the GCP analog of Azure files_premium) — use them for high-parallelism
    deployments (≥8 concurrent cells) where pip-install bootstrap contention on
    fresh nodes is the bottleneck.  Ignored when filestore_enabled = false.
  EOT
  type        = string
  default     = "BASIC_HDD"

  validation {
    condition     = contains(["BASIC_HDD", "BASIC_SSD", "ZONAL", "ENTERPRISE", "HIGH_SCALE_SSD"], var.filestore_tier)
    error_message = "filestore_tier must be one of: BASIC_HDD, BASIC_SSD, ZONAL, ENTERPRISE, HIGH_SCALE_SSD."
  }
}

variable "filestore_capacity_gb" {
  description = "Capacity of the Filestore share in GiB. Filestore BASIC_HDD enforces a 1024 GiB (1 TiB) minimum; ZONAL/ENTERPRISE have higher minimums. Ignored when filestore_enabled = false."
  type        = number
  default     = 1024
}

variable "filestore_enabled" {
  description = <<-EOT
    false (default): no Filestore instance is created — identical to Azure's
    files_premium being opt-in.  Jobs use an emptyDir / GCS-only cache (each
    fresh node re-downloads model weights).  Zero extra cost.

    true: provision a Filestore instance and a RWX (NFS) cache PVC so model
    weights download once per cluster lifetime, shared across all cells.
    Filestore is provisioned/GiB (charged for reserved capacity regardless of
    usage) — enable for high-parallelism runs, leave off otherwise.
  EOT
  type        = bool
  default     = false
}

# ─── Workload Identity ───────────────────────────────────────────────────────

variable "workload_identity_namespace" {
  description = "Kubernetes namespace where the Workload-Identity ServiceAccount lives (must match Helm L2 and gcp_namespace in backend/config.py)."
  type        = string
  default     = "reprolab"
}

variable "workload_identity_service_account" {
  description = "Name of the Kubernetes ServiceAccount annotated with the GCP service-account email (must match Helm L2 and gcp_service_account in backend/config.py)."
  type        = string
  default     = "reprolab-sa"
}

# ─── Remote state (reference only — used by backend.hcl, not provider) ───────

variable "state_bucket_name" {
  description = "Name of the GCS bucket that holds Terraform remote state (created by bootstrap). Informational — not used as a Terraform resource in this root. Set in backend.hcl."
  type        = string
  default     = ""
}

# ─── Secret Manager (Stream A — GCP orchestrator parity) ─────────────────────

variable "secret_manager_enabled" {
  description = <<-EOT
    false (default): no Secret Manager resources created — existing deployments
    unchanged.  true: provisions the three secret NAME resources
    ('claude-code-oauth-token', 'anthropic-api-key', 'azure-openai-api-key') and
    grants the orchestrator GSA secretmanager.secretAccessor on each.  Secret
    VALUES must be set out-of-band by the operator:
      gcloud secrets versions add <name> --data-file=-
  EOT
  type        = bool
  default     = false
}
