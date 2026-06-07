output "mi_client_id" {
  description = "Client ID of the workload managed identity. Set as the 'azure.workload.identity/client-id' annotation on the Kubernetes ServiceAccount in Helm L2, and as REPROLAB_AZURE_WORKLOAD_CLIENT_ID in the orchestrator."
  value       = azurerm_user_assigned_identity.workload.client_id
}

output "mi_principal_id" {
  description = "Object/principal ID of the workload managed identity."
  value       = azurerm_user_assigned_identity.workload.principal_id
}

output "mi_resource_id" {
  description = "Full Azure resource ID of the workload managed identity."
  value       = azurerm_user_assigned_identity.workload.id
}

output "federated_subject" {
  description = "Exact federated credential subject string. Must match the Helm L2 ServiceAccount namespace and name."
  value       = "system:serviceaccount:${var.namespace}:${var.service_account_name}"
}
