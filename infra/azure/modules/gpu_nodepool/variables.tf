variable "cluster_id" {
  description = "Full resource ID of the AKS cluster that owns this node pool."
  type        = string
}

variable "subnet_id" {
  description = "Resource ID of the subnet where GPU nodes are placed (same subnet as the system pool)."
  type        = string
}

variable "prefix" {
  description = "Resource name prefix. Node pool name is derived as '<prefix>gpu'."
  type        = string
}

variable "gpu_max_nodes" {
  description = "Maximum number of GPU nodes. Required A100 vCPU quota = 24 × gpu_max_nodes (NCADSA100v4-family). Start with 1 and scale after quota grant."
  type        = number
  default     = 4
}

variable "tags" {
  description = "Map of tags applied to the node pool."
  type        = map(string)
  default     = {}
}
