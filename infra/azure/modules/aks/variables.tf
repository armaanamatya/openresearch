variable "resource_group_name" {
  description = "Resource group where the AKS cluster is created."
  type        = string
}

variable "region" {
  description = "Azure region."
  type        = string
}

variable "prefix" {
  description = "Resource name prefix."
  type        = string
}

variable "kubernetes_version" {
  description = "Kubernetes version (e.g. '1.29'). Must be available in the region."
  type        = string
}

variable "subnet_id" {
  description = "Resource ID of the subnet where AKS nodes are placed."
  type        = string
}

variable "authorized_ip_ranges" {
  description = "List of CIDR blocks permitted to reach the public API server. At minimum include the operator's egress IP."
  type        = list(string)
}

variable "system_node_sku" {
  description = "VM SKU for the system (CPU) node pool."
  type        = string
  default     = "Standard_D4s_v5"
}

variable "system_node_min" {
  description = "Minimum node count for the system pool."
  type        = number
  default     = 1
}

variable "system_node_max" {
  description = "Maximum node count for the system pool."
  type        = number
  default     = 3
}

variable "tags" {
  description = "Map of tags applied to the cluster."
  type        = map(string)
  default     = {}
}
