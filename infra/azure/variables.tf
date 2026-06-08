# ─── Azure tenant / subscription ────────────────────────────────────────────

variable "subscription_id" {
  description = "Azure subscription ID where all resources are created."
  type        = string
}

variable "tenant_id" {
  description = "Azure Active Directory tenant ID."
  type        = string
}

# ─── Naming & location ───────────────────────────────────────────────────────

variable "prefix" {
  description = "Short alphanumeric prefix prepended to every resource name (e.g. 'repro'). Keep ≤8 chars."
  type        = string
}

variable "region" {
  description = "Azure region for all resources (e.g. 'eastus'). Choose a region that has Standard_NC24ads_A100_v4 quota available."
  type        = string
}

variable "tags" {
  description = "Map of tags applied to every resource."
  type        = map(string)
  default     = {}
}

# ─── Resource group ──────────────────────────────────────────────────────────

variable "resource_group_name" {
  description = "Name of the main resource group. Created by root main.tf."
  type        = string
}

# ─── Networking ──────────────────────────────────────────────────────────────

variable "vnet_cidr" {
  description = "Address space for the VNet (e.g. '10.0.0.0/16')."
  type        = string
  default     = "10.0.0.0/16"
}

variable "aks_subnet_cidr" {
  description = "Subnet CIDR carved out of vnet_cidr for AKS nodes (e.g. '10.0.0.0/22')."
  type        = string
  default     = "10.0.0.0/22"
}

variable "authorized_ip_ranges" {
  description = "List of operator CIDR(s) allowed to reach the AKS public API server (e.g. ['203.0.113.0/32']). Required — the API server is public but IP-restricted."
  type        = list(string)
}

# ─── AKS cluster ─────────────────────────────────────────────────────────────

variable "kubernetes_version" {
  description = "Kubernetes version for the AKS cluster (e.g. '1.29'). Pin to a version available in the chosen region."
  type        = string
}

variable "system_node_sku" {
  description = "VM SKU for the system (CPU) node pool."
  type        = string
  default     = "Standard_D4s_v5"
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

variable "operator_entra_group_object_id" {
  description = "Object ID of the Entra (AAD) group whose members receive AKS admin RBAC. Used by Helm L2 RoleBinding."
  type        = string
}

# ─── GPU node pools ──────────────────────────────────────────────────────────
#
# gpu_skus is the primary control surface.  Each entry provisions one
# scale-to-zero AKS node pool, labeled reprolab/sku=<short_name>, so the
# orchestrator can target it by catalog short_name.
#
# Catalog short_names and the AKS pool-name suffix derived from them:
#   short_name        vm_size                        gpus  pool_suffix
#   azure_a10_24      Standard_NV36ads_A10_v5        1     a10
#   azure_a100_80     Standard_NC24ads_A100_v4       1     a10080
#   azure_a100_80x2   Standard_NC48ads_A100_v4       2     a100x2
#   azure_a100_80x4   Standard_NC96ads_A100_v4       4     a100x4
#
# Pool name formula: "<prefix><pool_suffix>" — must be ≤12 lowercase-alnum chars.
# With the default prefix "repro" (5 chars) the suffixes above produce:
#   repro + a10     = repoa10     (8 chars) ✓
#   repro + a10080  = reproa10080 (11 chars) ✓
#   repro + a100x2  = reproa100x2 (11 chars) ✓
#   repro + a100x4  = reproa100x4 (11 chars) ✓
#
# QUOTA: Each entry requires its own vCPU quota in the corresponding VM family.
#   azure_a10_24    → StandardNVADSA10v5Family   36 × max_nodes vCPUs
#   azure_a100_80   → StandardNCADSA100v4Family  24 × max_nodes vCPUs
#   azure_a100_80x2 → StandardNCADSA100v4Family  48 × max_nodes vCPUs
#   azure_a100_80x4 → StandardNCADSA100v4Family  96 × max_nodes vCPUs
# The default (single A100-80 pool, max_nodes=4) requires 96 A100 vCPUs.
# Start with max_nodes=1 (24 vCPUs) until quota is granted.

variable "gpu_skus" {
  description = <<-EOT
    List of GPU SKU objects — one AKS scale-to-zero node pool is created per entry.
    Fields:
      short_name    — catalog identifier; written to the 'reprolab/sku' node label
                      and used by Job nodeSelector.  Must be unique within the list.
      vm_size       — Azure VM SKU for the pool nodes.
      gpu_count     — GPUs per node; written to the 'nvidia.com/gpu' node label.
      pool_suffix   — short suffix (≤7 chars, lowercase-alnum) appended to <prefix>
                      to form the AKS pool name (≤12 chars total).
      max_nodes     — maximum autoscaler node count for this pool. min is always 0.
      os_disk_size_gb — OS disk size in GiB for nodes in this pool. Default 256 GiB
                        is sufficient for the aks-cell-base image + working dir;
                        raise to 512 for SKUs that pull very large Docker images
                        (e.g. the devel CUDA image) or have large pip/HF caches
                        that overflow the Azure Files PVC onto local disk.
    Default: a single A100-80 pool — ONE quota ask.  Add NC48/NC96/A10 entries
    (each needing separate quota) to enable the escalation ladder.
  EOT
  type = list(object({
    short_name      = string
    vm_size         = string
    gpu_count       = number
    pool_suffix     = string
    max_nodes       = number
    os_disk_size_gb = optional(number, 256)
  }))
  default = [
    {
      short_name      = "azure_a100_80"
      vm_size         = "Standard_NC24ads_A100_v4"
      gpu_count       = 1
      pool_suffix     = "a10080"
      max_nodes       = 4
      os_disk_size_gb = 256
    }
  ]
}

# ─── DEPRECATED — kept for back-compatibility ────────────────────────────────
# gpu_max_nodes was the pre-parameterization "max nodes for the single A100-80
# pool" knob.  It is now IGNORED — set max_nodes inside the gpu_skus entry
# instead.  This variable will be removed in a future release.

variable "gpu_max_nodes" {
  description = "DEPRECATED. Set max_nodes inside the gpu_skus list entry instead. This variable is no longer wired to any module and exists only to prevent plan-time errors for tfvars files that still contain it."
  type        = number
  default     = 4
}

# ─── Container registry ──────────────────────────────────────────────────────

variable "acr_sku" {
  description = "ACR SKU. Standard is sufficient; Premium adds geo-replication and Private Link."
  type        = string
  default     = "Standard"
}

# ─── Artifact + cache storage ────────────────────────────────────────────────

variable "storage_account_name" {
  description = "Globally unique storage account name (3-24 lowercase alphanum). Hosts the artifact Blob container and the Azure Files HF_HOME/pip-cache share."
  type        = string
}

variable "blob_container_name" {
  description = "Name of the private Blob container used as the artifact bus between the orchestrator and AKS Jobs."
  type        = string
  default     = "reprolab-artifacts"
}

variable "files_share_name" {
  description = "Name of the Azure Files share mounted by Jobs as the RWX HuggingFace / pip cache."
  type        = string
  default     = "reprolab-cache"
}

variable "files_share_quota_gb" {
  description = "Capacity quota of the Azure Files share in GiB."
  type        = number
  default     = 512
}

variable "files_premium" {
  description = <<-EOT
    false (default): Azure Files cache share lives on the existing Standard
    StorageV2 account — identical to pre-flag behaviour, zero extra resources.

    true: provision a dedicated FileStorage (Premium) account and create the
    cache share there.  Premium Files delivers ~100 000 IOPS vs ~1 000 IOPS
    on Standard, eliminating the pip-install bootstrap contention bottleneck
    when many cells start concurrently on fresh AKS nodes.

    Cost: Premium Files is provisioned/GiB (~$52/month for 512 GiB in eastus)
    vs consumption-based Standard (~$10/month).  Enable for high-parallelism
    runs (≥8 concurrent cells per cluster), leave off otherwise.
  EOT
  type    = bool
  default = false
}

variable "files_premium_storage_account_name" {
  description = <<-EOT
    Globally unique name for the dedicated Premium FileStorage storage account
    created when files_premium = true (3-24 lowercase alphanum, no hyphens).
    Ignored when files_premium = false.
    Must differ from storage_account_name.
    Convention: append "prem" to the base name, e.g. "reprolabsaprem".
  EOT
  type    = string
  default = ""
}

# ─── Workload identity ───────────────────────────────────────────────────────

variable "workload_identity_namespace" {
  description = "Kubernetes namespace where the workload-identity ServiceAccount lives (must match Helm L2)."
  type        = string
  default     = "reprolab"
}

variable "workload_identity_service_account" {
  description = "Name of the Kubernetes ServiceAccount annotated with the workload-identity client ID (must match Helm L2)."
  type        = string
  default     = "reprolab-sa"
}

# ─── Remote state (reference only — used by backend.hcl, not provider) ───────

variable "state_storage_account_name" {
  description = "Name of the storage account that holds Terraform remote state (created by bootstrap). Informational — not used as a Terraform resource in this root. Set in backend.hcl."
  type        = string
  default     = ""
}
