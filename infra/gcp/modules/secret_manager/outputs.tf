output "claude_code_oauth_token_secret_id" {
  description = "Fully-qualified Secret Manager resource name for 'claude-code-oauth-token'. Pass to the SecretProviderClass resourceName field."
  value       = google_secret_manager_secret.claude_code_oauth_token.name
}

output "anthropic_api_key_secret_id" {
  description = "Fully-qualified Secret Manager resource name for 'anthropic-api-key'. Pass to the SecretProviderClass resourceName field."
  value       = google_secret_manager_secret.anthropic_api_key.name
}

output "azure_openai_api_key_secret_id" {
  description = "Fully-qualified Secret Manager resource name for 'azure-openai-api-key'. Pass to the SecretProviderClass resourceName field."
  value       = google_secret_manager_secret.azure_openai_api_key.name
}

output "azure_foundry_api_key_secret_id" {
  description = "Fully-qualified Secret Manager resource name for 'azure-foundry-api-key'. Pass to the SecretProviderClass resourceName field."
  value       = google_secret_manager_secret.azure_foundry_api_key.name
}

output "openai_api_key_secret_id" {
  description = "Fully-qualified Secret Manager resource name for 'openai-api-key'. Pass to the SecretProviderClass resourceName field."
  value       = google_secret_manager_secret.openai_api_key.name
}

output "claude_code_oauth_token_name" {
  description = "Short secret ID 'claude-code-oauth-token'. Use with `gcloud secrets versions add`."
  value       = google_secret_manager_secret.claude_code_oauth_token.secret_id
}

output "anthropic_api_key_name" {
  description = "Short secret ID 'anthropic-api-key'. Use with `gcloud secrets versions add`."
  value       = google_secret_manager_secret.anthropic_api_key.secret_id
}

output "azure_openai_api_key_name" {
  description = "Short secret ID 'azure-openai-api-key'. Use with `gcloud secrets versions add`."
  value       = google_secret_manager_secret.azure_openai_api_key.secret_id
}

output "azure_foundry_api_key_name" {
  description = "Short secret ID 'azure-foundry-api-key'. Use with `gcloud secrets versions add`."
  value       = google_secret_manager_secret.azure_foundry_api_key.secret_id
}

output "openai_api_key_name" {
  description = "Short secret ID 'openai-api-key'. Use with `gcloud secrets versions add`."
  value       = google_secret_manager_secret.openai_api_key.secret_id
}
