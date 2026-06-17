variable "project_id" {
  description = "GCP project ID where the Artifact Registry repository is created."
  type        = string
}

variable "region" {
  description = "GCP region (the repository is regional; must match the cluster region for low-latency pulls)."
  type        = string
}

variable "repository_id" {
  description = "Artifact Registry repository name (lowercase, hyphens allowed). The full URL is <region>-docker.pkg.dev/<project>/<repository_id>."
  type        = string
  default     = "reprolab"
}

variable "node_sa_email" {
  description = "Email of the GKE node service account. Receives roles/artifactregistry.reader on this repository so nodes can pull images without credentials."
  type        = string
}

variable "labels" {
  description = "Map of labels applied to the repository."
  type        = map(string)
  default     = {}
}
