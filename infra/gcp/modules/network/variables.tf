variable "project_id" {
  description = "GCP project ID where network resources are created."
  type        = string
}

variable "region" {
  description = "GCP region."
  type        = string
}

variable "prefix" {
  description = "Resource name prefix."
  type        = string
}

variable "subnet_cidr" {
  description = "Primary CIDR of the GKE node subnet (node IPs)."
  type        = string
}

variable "pods_secondary_cidr" {
  description = "Secondary CIDR for GKE pod IPs (VPC-native alias range). Must not overlap subnet_cidr or services_secondary_cidr."
  type        = string
}

variable "services_secondary_cidr" {
  description = "Secondary CIDR for GKE ClusterIP services (VPC-native alias range). Must not overlap subnet_cidr or pods_secondary_cidr."
  type        = string
}
