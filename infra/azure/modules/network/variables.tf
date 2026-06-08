variable "resource_group_name" {
  description = "Name of the resource group where network resources are created."
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

variable "vnet_cidr" {
  description = "Address space for the VNet (CIDR notation, e.g. '10.0.0.0/16')."
  type        = string
}

variable "aks_subnet_cidr" {
  description = "Subnet CIDR for AKS nodes. Must be within vnet_cidr."
  type        = string
}

variable "tags" {
  description = "Map of tags applied to network resources."
  type        = map(string)
  default     = {}
}
