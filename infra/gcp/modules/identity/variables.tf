variable "project_id" {
  description = "GCP project ID where the workload-identity GSA is created. Also forms the workload pool '<project_id>.svc.id.goog'."
  type        = string
}

variable "prefix" {
  description = "Resource name prefix. GSA account_id is derived as '<prefix>-workload'."
  type        = string
}

variable "namespace" {
  description = "Kubernetes namespace where the workload-identity ServiceAccount lives. Must match the namespace in Helm L2."
  type        = string
  default     = "reprolab"
}

variable "service_account_name" {
  description = "Name of the Kubernetes ServiceAccount annotated with this GSA's email. Must match the ServiceAccount in Helm L2."
  type        = string
  default     = "reprolab-sa"
}

variable "bucket_name" {
  description = "Name of the artifact bus GCS bucket. The GSA receives roles/storage.objectAdmin on this bucket (least-privilege — bucket-scoped, not project-wide)."
  type        = string
}

variable "labels" {
  description = "Map of labels (unused by IAM resources; accepted for interface symmetry with the other modules)."
  type        = map(string)
  default     = {}
}
