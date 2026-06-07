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

# ─── GPU node pool ───────────────────────────────────────────────────────────

variable "gpu_max_nodes" {
  description = "Maximum GPU node count (each Standard_NC24ads_A100_v4 = 1 × A100-80GB = 24 vCPUs). Required A100 quota = 24 × gpu_max_nodes NCADSA100v4-family vCPUs. Start with 1; scale after quota grant."
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
