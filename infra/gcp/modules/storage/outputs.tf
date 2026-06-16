output "bucket_name" {
  description = "Name of the artifact bus GCS bucket. Set as OPENRESEARCH_GCP_GCS_BUCKET and Helm storage.bucket."
  value       = google_storage_bucket.artifacts.name
}

output "bucket_url" {
  description = "gs:// URL of the artifact bucket."
  value       = google_storage_bucket.artifacts.url
}

output "filestore_ip" {
  description = "IP address of the Filestore instance (empty string when filestore_enabled = false). Used by the Helm Filestore CSI StorageClass."
  value       = var.filestore_enabled ? google_filestore_instance.cache[0].networks[0].ip_addresses[0] : ""
}

output "filestore_share" {
  description = "Name of the active Filestore file share (empty string when filestore_enabled = false). Set as OPENRESEARCH_GCP_FILESTORE_SHARE and Helm storage.filestoreShare."
  value       = var.filestore_enabled ? var.filestore_share_name : ""
}

# NOTE: No HMAC key / static credential is exported. The bucket is reached via
# Application Default Credentials (orchestrator) and Workload Identity (pods).
