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

# ─── Orchestrator Secret Manager bindings (optional) ─────────────────────────
# When the secret_manager module is also deployed, pass its secret IDs here so
# the orchestrator GSA receives secretmanager.secretAccessor on each secret.
# Set secret_manager_module_enabled = false to skip the IAM bindings (e.g. during
# the initial bootstrap before the secret_manager module is applied).

variable "secret_manager_module_enabled" {
  description = "When true, create IAM bindings from the orchestrator GSA to the three Secret Manager secrets. Set false if the secret_manager module has not been applied yet."
  type        = bool
  default     = false
}

variable "claude_code_oauth_token_secret_id" {
  description = "Fully-qualified resource name of the 'claude-code-oauth-token' secret (output of the secret_manager module). Only used when secret_manager_module_enabled = true."
  type        = string
  default     = ""
}

variable "anthropic_api_key_secret_id" {
  description = "Fully-qualified resource name of the 'anthropic-api-key' secret (output of the secret_manager module). Only used when secret_manager_module_enabled = true."
  type        = string
  default     = ""
}

variable "azure_openai_api_key_secret_id" {
  description = "Fully-qualified resource name of the 'azure-openai-api-key' secret (output of the secret_manager module). Only used when secret_manager_module_enabled = true."
  type        = string
  default     = ""
}

variable "azure_foundry_api_key_secret_id" {
  description = "Fully-qualified resource name of the 'azure-foundry-api-key' secret (output of the secret_manager module). Optional OAuth-free grok credential; the IAM grant is created only when this is non-empty AND secret_manager_module_enabled = true."
  type        = string
  default     = ""
}

variable "openai_api_key_secret_id" {
  description = "Fully-qualified resource name of the 'openai-api-key' secret (output of the secret_manager module). Optional OAuth-free OpenAI sub-agent credential; the IAM grant is created only when this is non-empty AND secret_manager_module_enabled = true."
  type        = string
  default     = ""
}
