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

# ─── GPU node pool ───────────────────────────────────────────────────────────

module "gpu_nodepool" {
  source = "./modules/gpu_nodepool"

  cluster_id    = module.aks.cluster_id
  subnet_id     = module.network.aks_subnet_id
  prefix        = var.prefix
  gpu_max_nodes = var.gpu_max_nodes
  tags          = var.tags
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
