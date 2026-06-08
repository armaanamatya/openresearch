variable "resource_group_name" {
  description = "Resource group where storage resources are created."
  type        = string
}

variable "region" {
  description = "Azure region."
  type        = string
}

variable "storage_account_name" {
  description = "Globally unique storage account name (3-24 lowercase alphanum). Hosts both the artifact Blob container and the Azure Files share."
  type        = string
}

variable "blob_container_name" {
  description = "Name of the private Blob container used as the artifact bus between the local orchestrator and AKS Jobs."
  type        = string
  default     = "reprolab-artifacts"
}

variable "files_share_name" {
  description = "Name of the Azure Files share mounted by Job pods as the RWX HuggingFace model cache and pip cache."
  type        = string
  default     = "reprolab-cache"
}

variable "files_share_quota_gb" {
  description = "Capacity quota for the Azure Files share in GiB. 512 GiB covers a full SDAR run with multiple Qwen model weights."
  type        = number
  default     = 512
}

variable "kubelet_object_id" {
  description = "Object ID of the AKS kubelet managed identity. Receives 'Storage File Data SMB Share Contributor' so Azure Files CSI can manage mount credentials without a storage account key Secret."
  type        = string
}

variable "aks_subnet_id" {
  description = "Resource ID of the AKS subnet. Added to the storage account network_rules allowlist so AKS Job pods and the CSI driver can reach the storage account via service endpoint. Required — locks the storage account to the AKS subnet."
  type        = string
}

variable "authorized_ip_ranges" {
  description = "List of operator CIDR(s) or single IPv4 addresses allowed to reach the storage account from outside the VNet (e.g. local dev machine, CI runner). Passed directly to azurerm_storage_account network_rules ip_rules. Use [] to allow only the AKS subnet."
  type        = list(string)
  default     = []
}

variable "files_premium" {
  description = <<-EOT
    When false (default): the Azure Files cache share is created on the existing
    Standard StorageV2 account — identical to the pre-flag behaviour, no extra
    resources are provisioned.

    When true: a dedicated FileStorage (Premium) storage account is provisioned
    alongside the Standard account and the cache share is created there instead.
    Premium Files delivers ~100 000 IOPS vs ~1 000 IOPS on Standard, eliminating
    the pip-install bootstrap bottleneck when many cells start concurrently.

    Azure constraint: Premium Files requires account_kind="FileStorage" +
    account_tier="Premium", which cannot coexist with the Blob service on the
    same general-purpose-v2 account — hence the dedicated account.

    Cost note: Premium Files is provisioned/GiB (charged for reserved capacity
    regardless of usage), not consumption-based.  A 512 GiB share in eastus is
    roughly $52/month vs $10/month for Standard.  Enable only when cell
    parallelism (≥8 concurrent cells) justifies the cost.
  EOT
  type    = bool
  default = false
}

variable "files_premium_storage_account_name" {
  description = <<-EOT
    Globally unique name for the dedicated Premium FileStorage account created
    when files_premium = true (3-24 lowercase alphanum, no hyphens).
    Ignored when files_premium = false.
    Must differ from storage_account_name (the Standard account).
    Convention: append "prem" to the base name, e.g. "reprolabsaprem".
  EOT
  type    = string
  default = ""
}

variable "tags" {
  description = "Map of tags applied to storage resources."
  type        = map(string)
  default     = {}
}
