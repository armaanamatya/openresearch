variable "project_id" {
  description = "GCP project ID for the bootstrap resources."
  type        = string
}

variable "region" {
  description = "GCP region for the state bucket (e.g. 'us-central1')."
  type        = string
  default     = "us-central1"
}

variable "state_bucket_name" {
  description = "Globally unique GCS bucket name that holds Terraform remote state (3-63 lowercase alphanum/hyphen)."
  type        = string
}

variable "create_ci_service_account" {
  description = "When true, create a CI/deploy service account with the minimal roles to run the root terraform apply. Keyless — impersonate it via Workload Identity Federation or serviceAccountTokenCreator; NO JSON key is created."
  type        = bool
  default     = false
}

variable "ci_service_account_id" {
  description = "Account ID (the part before @) for the CI/deploy service account. Ignored when create_ci_service_account = false."
  type        = string
  default     = "reprolab-ci"
}

variable "labels" {
  description = "Map of labels applied to bootstrap resources."
  type        = map(string)
  default     = {}
}
