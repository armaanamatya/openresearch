variable "resource_group_name" {
  description = "Resource group where the container registry is created."
  type        = string
}

variable "region" {
  description = "Azure region."
  type        = string
}

variable "prefix" {
  description = "Resource name prefix. ACR name is derived as '<prefix>acr' (must be globally unique, lowercase alphanum only)."
  type        = string
}

variable "sku" {
  description = "ACR SKU. 'Standard' is sufficient for this workload; 'Premium' adds geo-replication and Private Link."
  type        = string
  default     = "Standard"

  validation {
    condition     = contains(["Basic", "Standard", "Premium"], var.sku)
    error_message = "ACR SKU must be one of: Basic, Standard, Premium."
  }
}

variable "kubelet_object_id" {
  description = "Object ID of the AKS kubelet managed identity. Receives the AcrPull built-in role so nodes can pull images without admin credentials."
  type        = string
}

variable "tags" {
  description = "Map of tags applied to the registry."
  type        = map(string)
  default     = {}
}
