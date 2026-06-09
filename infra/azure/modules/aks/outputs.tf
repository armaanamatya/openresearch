output "cluster_id" {
  description = "Full Azure resource ID of the AKS cluster."
  value       = azurerm_kubernetes_cluster.main.id
}

output "cluster_name" {
  description = "Name of the AKS cluster."
  value       = azurerm_kubernetes_cluster.main.name
}

output "oidc_issuer_url" {
  description = "OIDC issuer URL emitted by the cluster. Fed into the identity module to create federated credentials."
  value       = azurerm_kubernetes_cluster.main.oidc_issuer_url
}

output "node_resource_group" {
  description = "Auto-managed node resource group name (MC_*)."
  value       = azurerm_kubernetes_cluster.main.node_resource_group
}

output "kubelet_identity_client_id" {
  description = "Client ID of the kubelet managed identity. Used for AcrPull and Files role assignments."
  value       = azurerm_kubernetes_cluster.main.kubelet_identity[0].client_id
}

output "kubelet_identity_object_id" {
  description = "Object ID of the kubelet managed identity."
  value       = azurerm_kubernetes_cluster.main.kubelet_identity[0].object_id
}
