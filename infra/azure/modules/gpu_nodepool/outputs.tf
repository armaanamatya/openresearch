output "pool_name" {
  description = "Name of this GPU node pool (AKS resource name, ≤12 chars)."
  value       = azurerm_kubernetes_cluster_node_pool.gpu.name
}

output "sku_label" {
  description = "Value of the 'reprolab/sku' node label on this pool (catalog short_name, e.g. 'azure_a100_80'). Use as the nodeSelector value in Job pod templates."
  value       = var.sku_label
}

output "nodepool_id" {
  description = "Full Azure resource ID of this GPU node pool."
  value       = azurerm_kubernetes_cluster_node_pool.gpu.id
}

# ── Legacy outputs (kept for back-compat with any references in CI scripts) ──

output "nodepool_name" {
  description = "Alias for pool_name. Deprecated — prefer pool_name."
  value       = azurerm_kubernetes_cluster_node_pool.gpu.name
}

output "node_label_key" {
  description = "Primary node selector label key for this pool. Always 'reprolab/sku'."
  value       = "reprolab/sku"
}

output "node_label_value" {
  description = "Primary node selector label value for this pool (== sku_label)."
  value       = var.sku_label
}

output "taint_key" {
  description = "Taint key on GPU nodes. Value is 'present', effect is NoSchedule. Helm device-plugin DaemonSet tolerates with operator: Exists."
  value       = "nvidia.com/gpu"
}
