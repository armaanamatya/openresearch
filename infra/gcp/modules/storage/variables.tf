variable "project_id" {
  description = "GCP project ID where storage resources are created."
  type        = string
}

variable "region" {
  description = "GCP region. The bucket is regional; the Filestore instance is zonal (placed in <region>-a)."
  type        = string
}

variable "bucket_name" {
  description = "Globally unique GCS bucket name (3-63 lowercase alphanum/hyphen). Hosts the artifact bus."
  type        = string
}

variable "filestore_enabled" {
  description = <<-EOT
    When false (default): no Filestore instance is created — Jobs use an
    emptyDir / GCS-only cache.  Zero extra resources, identical to Azure's
    files_premium being opt-in.

    When true: provision a Filestore instance + RWX (NFS) cache share so model
    weights download once per cluster lifetime.  Filestore is provisioned/GiB.
  EOT
  type        = bool
  default     = false
}

variable "filestore_share_name" {
  description = "Name of the Filestore file share mounted by Job pods as the RWX cache. Ignored when filestore_enabled = false."
  type        = string
  default     = "reprolab-cache"
}

variable "filestore_tier" {
  description = "Filestore tier (BASIC_HDD default; ZONAL/ENTERPRISE for higher IOPS, the GCP analog of Azure files_premium). Ignored when filestore_enabled = false."
  type        = string
  default     = "BASIC_HDD"
}

variable "filestore_capacity_gb" {
  description = "Capacity of the Filestore share in GiB. BASIC_HDD min is 1024. Ignored when filestore_enabled = false."
  type        = number
  default     = 1024
}

variable "network_self_link" {
  description = "Self-link of the VPC network the Filestore instance attaches to. Only used when filestore_enabled = true."
  type        = string
}

variable "labels" {
  description = "Map of labels applied to storage resources."
  type        = map(string)
  default     = {}
}
