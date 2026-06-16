output "pool_name" {
  description = "Name of this GPU node pool (GKE resource name)."
  value       = google_container_node_pool.gpu.name
}

output "sku_label" {
  description = "Value of the 'reprolab/sku' node label on this pool (catalog short_name, e.g. 'gcp_a100_80'). Use as the nodeSelector value for the 'reprolab/sku' label key in Job pod templates."
  value       = var.short_name
}

output "machine_type" {
  description = "GCE machine type backing this pool."
  value       = var.machine_type
}

output "gpu_count" {
  description = "GPUs per node in this pool."
  value       = var.gpu_count
}

output "node_pool_id" {
  description = "Full resource ID of this GPU node pool."
  value       = google_container_node_pool.gpu.id
}
