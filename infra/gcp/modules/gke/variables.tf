variable "project_id" {
  description = "GCP project ID where the GKE cluster is created."
  type        = string
}

variable "region" {
  description = "GCP region (the cluster is regional)."
  type        = string
}

variable "prefix" {
  description = "Resource name prefix."
  type        = string
}

variable "kubernetes_version" {
  description = "Minimum master version. Empty string ('') = channel default."
  type        = string
  default     = ""
}

variable "release_channel" {
  description = "GKE release channel: RAPID, REGULAR, STABLE, or UNSPECIFIED."
  type        = string
  default     = "REGULAR"
}

variable "network_self_link" {
  description = "Self-link of the VPC network the cluster is placed in."
  type        = string
}

variable "subnet_self_link" {
  description = "Self-link of the node subnet (must carry the two secondary ranges below)."
  type        = string
}

variable "pods_secondary_range_name" {
  description = "Name of the pods secondary IP range on the subnet."
  type        = string
}

variable "services_secondary_range_name" {
  description = "Name of the services secondary IP range on the subnet."
  type        = string
}

variable "authorized_ip_ranges" {
  description = "List of CIDR blocks permitted to reach the public control-plane endpoint. At minimum include the operator's egress IP."
  type        = list(string)
}

variable "master_ipv4_cidr_block" {
  description = "RFC-1918 /28 reserved for the GKE control plane in a private cluster. Must not overlap the node subnet or secondary ranges."
  type        = string
  default     = "172.16.0.0/28"
}

variable "system_node_machine_type" {
  description = "Machine type for the system (CPU) node pool."
  type        = string
  default     = "e2-standard-4"
}

variable "system_node_min_count" {
  description = "Minimum node count for the system pool."
  type        = number
  default     = 1
}

variable "system_node_max_count" {
  description = "Maximum node count for the system pool."
  type        = number
  default     = 3
}

variable "labels" {
  description = "Map of labels applied to the cluster and system pool nodes."
  type        = map(string)
  default     = {}
}
