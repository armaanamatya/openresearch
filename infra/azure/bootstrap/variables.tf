variable "subscription_id" {
  description = "Azure subscription ID."
  type        = string
}

variable "tenant_id" {
  description = "Azure Active Directory tenant ID."
  type        = string
}

variable "region" {
  description = "Azure region for the bootstrap resources (e.g. 'eastus')."
  type        = string
}

variable "state_resource_group_name" {
  description = "Name of the resource group that holds Terraform remote state. Must not conflict with the main RG."
  type        = string
  default     = "rg-reprolab-tfstate"
}

variable "state_storage_account_name" {
  description = "Globally unique name for the storage account that holds Terraform remote state (3-24 lowercase alphanum)."
  type        = string
}

variable "state_container_name" {
  description = "Name of the blob container inside the state storage account."
  type        = string
  default     = "tfstate"
}

variable "tags" {
  description = "Map of tags applied to bootstrap resources."
  type        = map(string)
  default     = {}
}
