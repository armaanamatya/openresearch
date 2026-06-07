output "storage_account_name" {
  description = "Name of the storage account. Set as REPROLAB_AZURE_STORAGE_ACCOUNT."
  value       = azurerm_storage_account.main.name
}

output "storage_account_id" {
  description = "Full Azure resource ID of the storage account."
  value       = azurerm_storage_account.main.id
}

output "blob_container_name" {
  description = "Name of the private artifact Blob container. Set as REPROLAB_AZURE_BLOB_CONTAINER."
  value       = azurerm_storage_container.artifacts.name
}

output "blob_container_resource_id" {
  description = "Full Azure resource ID of the artifact Blob container. Passed to the identity module as the Storage Blob Data Contributor scope."
  value       = "${azurerm_storage_account.main.id}/blobServices/default/containers/${azurerm_storage_container.artifacts.name}"
}

output "files_share_name" {
  description = "Name of the Azure Files share. Set as REPROLAB_AZURE_FILES_SHARE."
  value       = azurerm_storage_share.cache.name
}

output "files_share_id" {
  description = "Full Azure resource ID of the Azure Files share."
  value       = azurerm_storage_share.cache.id
}

# NOTE: No storage account key is exported. See the zero-static-secrets note
# in modules/storage/main.tf for the documented exception policy.
