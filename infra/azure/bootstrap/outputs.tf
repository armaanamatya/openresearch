output "state_resource_group_name" {
  description = "Name of the resource group holding Terraform state."
  value       = azurerm_resource_group.tfstate.name
}

output "state_storage_account_name" {
  description = "Name of the storage account holding Terraform state. Use in backend.hcl."
  value       = azurerm_storage_account.tfstate.name
}

output "state_container_name" {
  description = "Name of the Blob container holding Terraform state files."
  value       = azurerm_storage_container.tfstate.name
}
