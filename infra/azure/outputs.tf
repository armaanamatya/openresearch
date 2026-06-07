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
  description = "ACR login server hostname (e.g. prefix.azurecr.io). Set as REPROLAB_AZURE_ACR_LOGIN_SERVER."
  value       = module.acr.login_server
}

output "acr_id" {
  description = "Full Azure resource ID of the container registry."
  value       = module.acr.acr_id
}

# ─── Storage ─────────────────────────────────────────────────────────────────

output "storage_account_name" {
  description = "Name of the storage account hosting Blob artifacts and Azure Files cache. Set as REPROLAB_AZURE_STORAGE_ACCOUNT."
  value       = module.storage.storage_account_name
}

output "blob_container_name" {
  description = "Name of the private Blob container (artifact bus). Set as REPROLAB_AZURE_BLOB_CONTAINER."
  value       = module.storage.blob_container_name
}

output "files_share_name" {
  description = "Name of the Azure Files share (RWX HF_HOME / pip cache). Set as REPROLAB_AZURE_FILES_SHARE."
  value       = module.storage.files_share_name
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

# ─── GPU node pool ───────────────────────────────────────────────────────────

output "gpu_nodepool_name" {
  description = "Name of the GPU node pool. Set as REPROLAB_AZURE_NODE_POOL_NAME."
  value       = module.gpu_nodepool.nodepool_name
}

output "gpu_node_pool_label_key" {
  description = "Kubernetes label key applied to GPU pool nodes. Use as nodeSelector key in Helm L2 (gpu.nodePoolLabelKey)."
  value       = module.gpu_nodepool.node_label_key
}

output "gpu_node_pool_label_value" {
  description = "Kubernetes label value applied to GPU pool nodes (gpu.nodePoolLabelValue in Helm L2)."
  value       = module.gpu_nodepool.node_label_value
}

output "gpu_taint_key" {
  description = "Taint key on GPU nodes (value 'present', effect NoSchedule). Use in Helm L2 tolerations (gpu.taintKey)."
  value       = module.gpu_nodepool.taint_key
}
