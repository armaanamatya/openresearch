output "acr_id" {
  description = "Full Azure resource ID of the container registry."
  value       = azurerm_container_registry.main.id
}

output "acr_name" {
  description = "Name of the container registry."
  value       = azurerm_container_registry.main.name
}

output "login_server" {
  description = "ACR login server hostname (e.g. prefixacr.azurecr.io). Set as REPROLAB_AZURE_ACR_LOGIN_SERVER."
  value       = azurerm_container_registry.main.login_server
}
