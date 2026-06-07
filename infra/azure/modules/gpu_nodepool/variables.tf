variable "cluster_id" {
  description = "Full resource ID of the AKS cluster that owns this node pool."
  type        = string
}

variable "subnet_id" {
  description = "Resource ID of the subnet where GPU nodes are placed (same subnet as the system pool)."
  type        = string
}

variable "prefix" {
  description = "Resource name prefix. Pool name is derived as '<prefix><pool_suffix>' (see pool_suffix var)."
  type        = string
}

variable "pool_suffix" {
  description = <<-EOT
    Short suffix appended to <prefix> to form the AKS node pool name.
    AKS pool names must be ≤12 lowercase-alphanumeric chars.
    The root module derives this from the catalog short_name:
      azure_a10_24    → a10
      azure_a100_80   → a10080
      azure_a100_80x2 → a100x2
      azure_a100_80x4 → a100x4
    The caller is responsible for ensuring <prefix><pool_suffix> ≤12 chars.
  EOT
  type        = string
}

variable "vm_size" {
  description = "Azure VM SKU for this GPU node pool (e.g. 'Standard_NC24ads_A100_v4')."
  type        = string
}

variable "gpu_count" {
  description = "Number of GPUs per node. Written to the 'nvidia.com/gpu' node label so the orchestrator can request exact GPU counts via nodeSelector."
  type        = number
}

variable "sku_label" {
  description = "Catalog short_name written to the 'reprolab/sku' node label (e.g. 'azure_a100_80'). Used by Job nodeSelector to target this exact pool."
  type        = string
}

variable "gpu_max_nodes" {
  description = "Maximum number of GPU nodes in this pool (min is always 0 — scale-to-zero). Required vCPU quota ≈ vCPUs(vm_size) × gpu_max_nodes in the corresponding VM family."
  type        = number
  default     = 4
}

variable "tags" {
  description = "Map of tags applied to the node pool."
  type        = map(string)
  default     = {}
}
