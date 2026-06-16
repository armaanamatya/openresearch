# ─── Network layer ───────────────────────────────────────────────────────────

module "network" {
  source = "./modules/network"

  project_id              = var.project_id
  region                  = var.region
  prefix                  = var.prefix
  subnet_cidr             = var.subnet_cidr
  pods_secondary_cidr     = var.pods_secondary_cidr
  services_secondary_cidr = var.services_secondary_cidr
}

# ─── GKE cluster ─────────────────────────────────────────────────────────────

module "gke" {
  source = "./modules/gke"

  project_id                    = var.project_id
  region                        = var.region
  prefix                        = var.prefix
  kubernetes_version            = var.kubernetes_version
  release_channel               = var.release_channel
  network_self_link             = module.network.network_self_link
  subnet_self_link              = module.network.subnet_self_link
  pods_secondary_range_name     = module.network.pods_secondary_range_name
  services_secondary_range_name = module.network.services_secondary_range_name
  authorized_ip_ranges          = var.authorized_ip_ranges
  system_node_machine_type      = var.system_node_machine_type
  system_node_min_count         = var.system_node_min_count
  system_node_max_count         = var.system_node_max_count
  labels                        = var.labels
}

# ─── GPU node pools — one per gpu_skus entry ─────────────────────────────────
#
# for_each key = short_name (e.g. "gcp_a100_80").
#
# Label contract (consumed by the k8s-runner orchestrator):
#   reprolab/sku       = <short_name>   — Job nodeSelector key
#   reprolab/gpu-count = <gpu_count>    — GPU count per node
#
# Taint on every GPU node: nvidia.com/gpu=present:NoSchedule
#   → the orchestrator's Job pods tolerate it with operator: Exists
#   → non-GPU workloads cannot land here
#
# GKE auto-installs the NVIDIA driver + manages the device plugin via the
# node-pool gpu_driver_installation_config — there is NO hand-rolled
# device-plugin DaemonSet on GKE (unlike the Azure mirror).
#
# locals trick: convert the list to a map keyed by short_name so for_each
# can reference each entry by its catalog identifier.

locals {
  gpu_skus_map = { for sku in var.gpu_skus : sku.short_name => sku }
}

module "gpu_nodepool" {
  source   = "./modules/gpu_nodepool"
  for_each = local.gpu_skus_map

  project_id       = var.project_id
  cluster_name     = module.gke.cluster_name
  location         = var.region
  prefix           = var.prefix
  short_name       = each.key
  machine_type     = each.value.machine_type
  accelerator_type = each.value.accelerator_type
  gpu_count        = each.value.gpu_count
  max_nodes        = each.value.max_nodes
  disk_size_gb     = each.value.disk_size_gb
  service_account  = module.gke.node_service_account_email
  labels           = var.labels
}

# ─── Artifact Registry ───────────────────────────────────────────────────────

module "registry" {
  source = "./modules/registry"

  project_id    = var.project_id
  region        = var.region
  repository_id = var.artifact_registry_repo
  node_sa_email = module.gke.node_service_account_email
  labels        = var.labels
}

# ─── Storage (artifact GCS bucket + optional Filestore RWX cache) ────────────

module "storage" {
  source = "./modules/storage"

  project_id            = var.project_id
  region                = var.region
  bucket_name           = var.gcs_bucket_name
  filestore_enabled     = var.filestore_enabled
  filestore_share_name  = var.filestore_share_name
  filestore_tier        = var.filestore_tier
  filestore_capacity_gb = var.filestore_capacity_gb
  network_self_link     = module.network.network_self_link
  labels                = var.labels
}

# ─── Workload identity (GSA + IAM binding) ───────────────────────────────────

module "identity" {
  source = "./modules/identity"

  project_id           = var.project_id
  prefix               = var.prefix
  namespace            = var.workload_identity_namespace
  service_account_name = var.workload_identity_service_account
  bucket_name          = module.storage.bucket_name
  labels               = var.labels
}
