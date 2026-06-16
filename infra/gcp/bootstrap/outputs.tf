output "state_bucket_name" {
  description = "Name of the GCS bucket holding Terraform state. Use as `bucket` in backend.hcl."
  value       = google_storage_bucket.tfstate.name
}

output "state_bucket_url" {
  description = "gs:// URL of the state bucket."
  value       = google_storage_bucket.tfstate.url
}

output "ci_service_account_email" {
  description = "Email of the CI/deploy service account (empty string when create_ci_service_account = false). Impersonate it (keyless) to run the root apply."
  value       = var.create_ci_service_account ? google_service_account.ci[0].email : ""
}
