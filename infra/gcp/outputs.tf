# ─── Cluster ─────────────────────────────────────────────────────────────────

output "gke_cluster_name" {
  description = "GKE cluster name. Pass to `gcloud container clusters get-credentials`. Set as OPENRESEARCH_GCP_GKE_CLUSTER."
  value       = module.gke.cluster_name
}

output "gke_cluster_endpoint" {
  description = "Public control-plane endpoint of the GKE cluster (IP-restricted to authorized_ip_ranges)."
  value       = module.gke.cluster_endpoint
  sensitive   = true
}

output "gke_get_credentials_command" {
  description = "Ready-to-run gcloud command that writes a kubeconfig entry for this cluster."
  value       = "gcloud container clusters get-credentials ${module.gke.cluster_name} --region ${var.region} --project ${var.project_id}"
}

output "workload_pool" {
  description = "Workload Identity pool of the cluster (<project_id>.svc.id.goog). Required when creating additional IAM bindings."
  value       = module.gke.workload_pool
}

# ─── Workload identity (GSA used by Job pods) ─────────────────────────────────

output "workload_identity_gcp_service_account" {
  description = "Email of the GCP IAM service account (GSA) bound to the Kubernetes ServiceAccount via Workload Identity. THE Helm `--set workloadIdentity.gcpServiceAccount=` source — annotated onto the KSA as iam.gke.io/gcp-service-account."
  value       = module.identity.gsa_email
}

output "orchestrator_gcp_service_account" {
  description = "Email of the orchestrator GSA. Set as --set orchestrator.gcpServiceAccount= in the Helm upgrade command. Annotated on the 'reprolab-orchestrator' KSA as iam.gke.io/gcp-service-account."
  value       = module.identity.orchestrator_gsa_email
}

# ─── Artifact Registry ───────────────────────────────────────────────────────

output "artifact_registry_url" {
  description = "Base Artifact Registry Docker URL (REGION-docker.pkg.dev/PROJECT/REPO). Append the image name + PINNED tag for image.gkeCellBase, e.g. <url>/gke-cell-base:0.1.0. Set as OPENRESEARCH_GCP_ARTIFACT_REGISTRY."
  value       = module.registry.repository_url
}

# ─── Storage ─────────────────────────────────────────────────────────────────

output "gcs_bucket_name" {
  description = "Name of the private GCS artifact bus bucket. Set as OPENRESEARCH_GCP_GCS_BUCKET and Helm storage.bucket."
  value       = module.storage.bucket_name
}

output "filestore_ip" {
  description = "IP address of the Filestore instance (when filestore_enabled = true; empty string otherwise). Used by the Helm Filestore CSI StorageClass."
  value       = module.storage.filestore_ip
}

output "filestore_share" {
  description = "Name of the active Filestore file share (when filestore_enabled = true; empty string otherwise). Set as OPENRESEARCH_GCP_FILESTORE_SHARE and Helm storage.filestoreShare."
  value       = module.storage.filestore_share
}

# ─── GPU node pools ───────────────────────────────────────────────────────────
#
# gpu_pools is the primary output for the orchestrator's k8s-runner.
# Shape:
#   {
#     "gcp_a100_80"   = { sku_label = "gcp_a100_80", machine_type = "a2-ultragpu-1g", gpu_count = 1 }
#     "gcp_a100_80x2" = { sku_label = "gcp_a100_80x2", machine_type = "a2-ultragpu-2g", gpu_count = 2 }
#     ...
#   }
#
# Orchestrator usage:
#   pool = gpu_pools[plan.short_name]
#   nodeSelector = { "reprolab/sku": pool.sku_label }

output "gpu_pools" {
  description = <<-EOT
    Map of provisioned GPU pools keyed by catalog short_name.
    Each entry: { sku_label = "<reprolab/sku value>", machine_type = "...", gpu_count = N }.
    The orchestrator's k8s job cell runner uses sku_label as the Job nodeSelector
    value for the 'reprolab/sku' label key.
  EOT
  value = {
    for short_name, mod in module.gpu_nodepool :
    short_name => {
      sku_label    = mod.sku_label
      machine_type = mod.machine_type
      gpu_count    = mod.gpu_count
    }
  }
}

# ─── Constants surfaced for Helm / smoke jobs ─────────────────────────────────

output "gpu_sku_label_key" {
  description = "Node selector label key for GPU pools. Always 'reprolab/sku'."
  value       = "reprolab/sku"
}

output "gpu_taint_key" {
  description = "Taint key on all GPU nodes (value 'present', effect NoSchedule). Job pods tolerate with operator: Exists."
  value       = "nvidia.com/gpu"
}
