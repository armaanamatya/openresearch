output "gsa_email" {
  description = "Email of the workload-identity GSA. Set as the 'iam.gke.io/gcp-service-account' annotation on the Kubernetes ServiceAccount in Helm L2 (--set workloadIdentity.gcpServiceAccount=). The KSA in <namespace>/<service_account_name> impersonates it via Workload Identity."
  value       = google_service_account.workload.email
}

output "gsa_id" {
  description = "Fully-qualified resource name of the workload-identity GSA."
  value       = google_service_account.workload.name
}

output "workload_identity_member" {
  description = "Exact Workload Identity member string. Must match the Helm L2 ServiceAccount namespace and name."
  value       = "serviceAccount:${var.project_id}.svc.id.goog[${var.namespace}/${var.service_account_name}]"
}

output "orchestrator_gsa_email" {
  description = "Email of the orchestrator GSA (empty when secret_manager_module_enabled = false). Set as the 'iam.gke.io/gcp-service-account' annotation on the 'reprolab-orchestrator' Kubernetes ServiceAccount (--set orchestrator.gcpServiceAccount=). This GSA has secretmanager.secretAccessor on the three API-key secrets and storage.objectAdmin on the artifact bucket."
  value       = length(google_service_account.orchestrator) > 0 ? google_service_account.orchestrator[0].email : ""
}

output "orchestrator_gsa_id" {
  description = "Fully-qualified resource name of the orchestrator GSA (empty when secret_manager_module_enabled = false)."
  value       = length(google_service_account.orchestrator) > 0 ? google_service_account.orchestrator[0].name : ""
}
