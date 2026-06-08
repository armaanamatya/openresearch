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
  description = "Name of the active Azure Files share (Standard or Premium). Set as REPROLAB_AZURE_FILES_SHARE."
  value       = var.files_share_name
}

output "files_share_id" {
  description = "Full Azure resource ID of the active Azure Files share."
  value       = local.files_active_share_id
}

output "files_storage_account_name" {
  description = <<-EOT
    Name of the storage account that hosts the active Files share.
    When files_premium = false (default): same as storage_account_name.
    When files_premium = true:            the dedicated FileStorage Premium account.
    Set as storage.accountName in Helm L2 so the StorageClass points at the
    correct account in both modes.
  EOT
  value = local.files_active_storage_account_name
}

# NOTE: No storage account key is exported. See the zero-static-secrets note
# in modules/storage/main.tf for the documented exception policy.
