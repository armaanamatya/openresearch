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

variable "tags" {
  description = "Map of tags applied to storage resources."
  type        = map(string)
  default     = {}
}
