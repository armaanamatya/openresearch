# ─── Cluster ─────────────────────────────────────────────────────────────────

output "cluster_name" {
  description = "AKS cluster name. Pass to `az aks get-credentials`."
  value       = module.aks.cluster_name
}

output "cluster_id" {
  description = "Full Azure resource ID of the AKS cluster."
  value       = module.aks.cluster_id
}

output "oidc_issuer_url" {
  description = "OIDC issuer URL emitted by the cluster. Required when creating additional federated credentials."
  value       = module.aks.oidc_issuer_url
}

output "node_resource_group" {
  description = "Name of the auto-managed node resource group (MC_*). Used for quota checks."
  value       = module.aks.node_resource_group
}

# ─── Kubelet identity ────────────────────────────────────────────────────────

output "kubelet_identity_client_id" {
  description = "Client ID of the AKS kubelet managed identity. Used in ACR and Files role assignments."
  value       = module.aks.kubelet_identity_client_id
}

output "kubelet_identity_object_id" {
  description = "Object ID of the AKS kubelet managed identity."
  value       = module.aks.kubelet_identity_object_id
}

# ─── Workload identity (MI used by Job pods) ─────────────────────────────────

output "workload_identity_client_id" {
  description = "Client ID of the user-assigned managed identity used by Job pods. Set as AZURE_CLIENT_ID env var or SA annotation in Helm L2."
  value       = module.identity.mi_client_id
}

output "workload_identity_principal_id" {
  description = "Object/principal ID of the workload MI."
  value       = module.identity.mi_principal_id
}

output "workload_identity_resource_id" {
  description = "Full Azure resource ID of the workload MI."
  value       = module.identity.mi_resource_id
}

# ─── Container registry ──────────────────────────────────────────────────────

output "acr_login_server" {
  description = "ACR login server hostname (e.g. prefix.azurecr.io). Set as OPENRESEARCH_AZURE_ACR_LOGIN_SERVER."
  value       = module.acr.login_server
}

output "acr_id" {
  description = "Full Azure resource ID of the container registry."
  value       = module.acr.acr_id
}

# ─── Storage ─────────────────────────────────────────────────────────────────

output "storage_account_name" {
  description = "Name of the storage account hosting Blob artifacts and Azure Files cache. Set as OPENRESEARCH_AZURE_STORAGE_ACCOUNT."
  value       = module.storage.storage_account_name
}

output "blob_container_name" {
  description = "Name of the private Blob container (artifact bus). Set as OPENRESEARCH_AZURE_BLOB_CONTAINER."
  value       = module.storage.blob_container_name
}

output "files_share_name" {
  description = "Name of the Azure Files share (RWX HF_HOME / pip cache). Set as OPENRESEARCH_AZURE_FILES_SHARE."
  value       = module.storage.files_share_name
}

output "files_storage_account_name" {
  description = <<-EOT
    Name of the storage account hosting the active Azure Files share.
    When files_premium = false (default): same as storage_account_name.
    When files_premium = true:            the dedicated Premium FileStorage account.
    Pass as storage.accountName to Helm L2 so the StorageClass points at the
    correct account in both modes.  Replaces the previous hard-coded assumption
    that Blob and Files always share one account.
  EOT
  value = module.storage.files_storage_account_name
}

# ─── Network ─────────────────────────────────────────────────────────────────

output "vnet_id" {
  description = "Resource ID of the VNet."
  value       = module.network.vnet_id
}

output "aks_subnet_id" {
  description = "Resource ID of the AKS subnet."
  value       = module.network.aks_subnet_id
}

# ─── GPU node pools ───────────────────────────────────────────────────────────
#
# gpu_pools is the primary output for the orchestrator's k8s-runner.
# Shape:
#   {
#     "azure_a100_80"   = { name = "reproa10080", sku_label = "azure_a100_80" }
#     "azure_a100_80x2" = { name = "reproa100x2", sku_label = "azure_a100_80x2" }
#     ...
#   }
#
# Orchestrator usage:
#   pool = gpu_pools[plan.short_name]
#   nodeSelector = { "reprolab/sku": pool.sku_label }
#
# Helm usage (smoke jobs):
#   terraform output -json gpu_pools | jq '."azure_a100_80".name'

output "gpu_pools" {
  description = <<-EOT
    Map of provisioned GPU pools keyed by catalog short_name.
    Each entry: { name = "<aks pool name>", sku_label = "<reprolab/sku value>" }.
    The orchestrator's k8s_job_cell_runner uses sku_label as the Job nodeSelector
    value for the 'reprolab/sku' label key.
  EOT
  value = {
    for short_name, mod in module.gpu_nodepool :
    short_name => {
      name      = mod.pool_name
      sku_label = mod.sku_label
    }
  }
}

# ── Legacy single-pool outputs (deprecated — use gpu_pools map above) ─────────
# Kept for CI scripts / runbooks that reference the old output names.
# These resolve to the FIRST entry in gpu_skus (by short_name sort order).
# Will be removed once all callers are migrated to gpu_pools.

output "gpu_nodepool_name" {
  description = "DEPRECATED. Name of the first GPU node pool. Use gpu_pools[<short_name>].name instead."
  value       = values(module.gpu_nodepool)[0].pool_name
}

output "gpu_node_pool_label_key" {
  description = "DEPRECATED. Node selector label key for GPU pools. Always 'reprolab/sku'."
  value       = "reprolab/sku"
}

output "gpu_node_pool_label_value" {
  description = "DEPRECATED. SKU label of the first GPU pool. Use gpu_pools[<short_name>].sku_label instead."
  value       = values(module.gpu_nodepool)[0].sku_label
}

output "gpu_taint_key" {
  description = "Taint key on all GPU nodes (value 'present', effect NoSchedule). Helm device-plugin tolerates with operator: Exists."
  value       = "nvidia.com/gpu"
}
