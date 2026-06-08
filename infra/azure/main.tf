# ─── Root resource group ─────────────────────────────────────────────────────

resource "azurerm_resource_group" "main" {
  name     = var.resource_group_name
  location = var.region
  tags     = var.tags
}

# ─── Network layer ───────────────────────────────────────────────────────────

module "network" {
  source = "./modules/network"

  resource_group_name = azurerm_resource_group.main.name
  region              = var.region
  prefix              = var.prefix
  vnet_cidr           = var.vnet_cidr
  aks_subnet_cidr     = var.aks_subnet_cidr
  tags                = var.tags
}

# ─── AKS cluster ─────────────────────────────────────────────────────────────

module "aks" {
  source = "./modules/aks"

  resource_group_name  = azurerm_resource_group.main.name
  region               = var.region
  prefix               = var.prefix
  kubernetes_version   = var.kubernetes_version
  subnet_id            = module.network.aks_subnet_id
  authorized_ip_ranges = var.authorized_ip_ranges
  system_node_sku      = var.system_node_sku
  system_node_min      = var.system_node_min_count
  system_node_max      = var.system_node_max_count
  tags                 = var.tags
}

# ─── GPU node pools — one per gpu_skus entry ─────────────────────────────────
#
# for_each key = short_name (e.g. "azure_a100_80").
# AKS pool name = "<prefix><pool_suffix>" — guaranteed ≤12 lowercase-alnum chars
# by the pool_suffix values in gpu_skus (see variables.tf for the mapping table).
#
# Label contract (consumed by the k8s-runner orchestrator):
#   reprolab/sku   = <short_name>   — Job nodeSelector key
#   nvidia.com/gpu = <gpu_count>    — GPU count per node
#
# Taint on every GPU node: nvidia.com/gpu=present:NoSchedule
#   → device-plugin DaemonSet tolerates with operator: Exists (all pools)
#   → non-GPU workloads cannot land here
#
# locals trick: convert the list to a map keyed by short_name so for_each
# can reference each entry by its catalog identifier.

locals {
  gpu_skus_map = { for sku in var.gpu_skus : sku.short_name => sku }
}

module "gpu_nodepool" {
  source   = "./modules/gpu_nodepool"
  for_each = local.gpu_skus_map

  cluster_id      = module.aks.cluster_id
  subnet_id       = module.network.aks_subnet_id
  prefix          = var.prefix
  pool_suffix     = each.value.pool_suffix
  vm_size         = each.value.vm_size
  gpu_count       = each.value.gpu_count
  sku_label       = each.key
  gpu_max_nodes   = each.value.max_nodes
  os_disk_size_gb = each.value.os_disk_size_gb
  tags            = var.tags
}

# ─── Container registry ──────────────────────────────────────────────────────

module "acr" {
  source = "./modules/acr"

  resource_group_name    = azurerm_resource_group.main.name
  region                 = var.region
  prefix                 = var.prefix
  sku                    = var.acr_sku
  kubelet_object_id      = module.aks.kubelet_identity_object_id
  tags                   = var.tags
}

# ─── Storage (artifact Blob + Files cache) ───────────────────────────────────

module "storage" {
  source = "./modules/storage"

  resource_group_name    = azurerm_resource_group.main.name
  region                 = var.region
  storage_account_name   = var.storage_account_name
  blob_container_name    = var.blob_container_name
  files_share_name       = var.files_share_name
  files_share_quota_gb   = var.files_share_quota_gb
  kubelet_object_id      = module.aks.kubelet_identity_object_id
  # P1: restrict storage account network access to AKS subnet + operator IPs.
  aks_subnet_id          = module.network.aks_subnet_id
  authorized_ip_ranges   = var.authorized_ip_ranges
  tags                   = var.tags
}

# ─── Workload identity (user-assigned MI + federated credential) ──────────────

module "identity" {
  source = "./modules/identity"

  resource_group_name      = azurerm_resource_group.main.name
  region                   = var.region
  prefix                   = var.prefix
  oidc_issuer_url          = module.aks.oidc_issuer_url
  namespace                = var.workload_identity_namespace
  service_account_name     = var.workload_identity_service_account
  artifact_container_id    = module.storage.blob_container_resource_id
  tags                     = var.tags
}
