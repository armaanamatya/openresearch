output "cluster_name" {
  description = "Name of the GKE cluster."
  value       = google_container_cluster.main.name
}

output "cluster_endpoint" {
  description = "Public (IP-restricted) control-plane endpoint of the cluster."
  value       = google_container_cluster.main.endpoint
  sensitive   = true
}

output "workload_pool" {
  description = "Workload Identity pool (<project_id>.svc.id.goog). Used by the identity module's IAM binding."
  value       = google_container_cluster.main.workload_identity_config[0].workload_pool
}

output "node_service_account_email" {
  description = "Email of the dedicated least-privilege node service account. Consumed by the GPU node pools (node identity) and granted Artifact Registry read in the registry module."
  value       = google_service_account.node.email
}

# system_node_pool_id is exported as a dependency handle so the GPU pools (and
# any caller that needs the cluster's node pools to exist) can sequence after it.
output "system_node_pool_id" {
  description = "Resource ID of the system node pool. Use as a depends-on handle."
  value       = google_container_node_pool.system.id
}
