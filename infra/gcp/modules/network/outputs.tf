output "network_self_link" {
  description = "Self-link of the VPC network."
  value       = google_compute_network.main.self_link
}

output "network_name" {
  description = "Name of the VPC network."
  value       = google_compute_network.main.name
}

output "subnet_self_link" {
  description = "Self-link of the GKE node subnet."
  value       = google_compute_subnetwork.main.self_link
}

output "subnet_name" {
  description = "Name of the GKE node subnet."
  value       = google_compute_subnetwork.main.name
}

output "pods_secondary_range_name" {
  description = "Name of the pods secondary IP range. Referenced by the GKE ip_allocation_policy."
  value       = "${var.prefix}-pods"
}

output "services_secondary_range_name" {
  description = "Name of the services secondary IP range. Referenced by the GKE ip_allocation_policy."
  value       = "${var.prefix}-services"
}
