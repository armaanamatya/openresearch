output "nodepool_id" {
  description = "Full Azure resource ID of the GPU node pool."
  value       = azurerm_kubernetes_cluster_node_pool.gpu.id
}

output "nodepool_name" {
  description = "Name of the GPU node pool. Set as REPROLAB_AZURE_NODE_POOL_NAME in the orchestrator config."
  value       = azurerm_kubernetes_cluster_node_pool.gpu.name
}

output "node_label_key" {
  description = "Kubernetes label key applied to every GPU pool node. Use as nodeSelector key in Helm L2 and Job pod templates."
  value       = "reprolab/node-type"
}

output "node_label_value" {
  description = "Kubernetes label value applied to every GPU pool node (pairs with node_label_key)."
  value       = "gpu"
}

output "taint_key" {
  description = "Taint key on GPU nodes. Value is 'present', effect is NoSchedule. Helm L2 adds the matching toleration."
  value       = "nvidia.com/gpu"
}
