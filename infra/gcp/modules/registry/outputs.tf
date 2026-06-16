output "repository_id" {
  description = "Name of the Artifact Registry repository."
  value       = google_artifact_registry_repository.main.repository_id
}

output "repository_url" {
  description = "Base Docker URL of the repository: <region>-docker.pkg.dev/<project>/<repo>. Append '/<image>:<tag>' for a full image ref. Set as OPENRESEARCH_GCP_ARTIFACT_REGISTRY."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.main.repository_id}"
}
