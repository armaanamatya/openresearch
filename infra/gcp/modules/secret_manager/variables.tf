variable "project_id" {
  description = "GCP project ID where the Secret Manager secrets are created."
  type        = string
}

variable "region" {
  description = "GCP region for user-managed replication of each secret."
  type        = string
}

variable "labels" {
  description = "Map of labels applied to every Secret Manager resource."
  type        = map(string)
  default     = {}
}
