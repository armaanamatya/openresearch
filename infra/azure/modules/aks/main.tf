resource "azurerm_kubernetes_cluster" "main" {
  name                = "${var.prefix}-aks"
  resource_group_name = var.resource_group_name
  location            = var.region
  dns_prefix          = "${var.prefix}-aks"
  kubernetes_version  = var.kubernetes_version

  # ── API server access ────────────────────────────────────────────────────────
  # Public endpoint restricted to the operator's egress CIDRs.
  # The API server is intentionally public in Phase 1 because the orchestrator
  # runs locally (outside the VNet). Becomes private in Phase 2 when the control
  # plane moves in-cluster. Confirm this posture with DeepInvent's security team.
  api_server_access_profile {
    authorized_ip_ranges = var.authorized_ip_ranges
  }

  # ── Managed identity ─────────────────────────────────────────────────────────
  identity {
    type = "SystemAssigned"
  }

  # ── OIDC + Workload Identity ──────────────────────────────────────────────────
  # Required for federated credentials (no static Kubernetes Secrets for pod auth).
  oidc_issuer_enabled       = true
  workload_identity_enabled = true

  # ── System (CPU) node pool ───────────────────────────────────────────────────
  default_node_pool {
    name                = "system"
    vm_size             = var.system_node_sku
    vnet_subnet_id      = var.subnet_id
    enable_auto_scaling = true
    min_count           = var.system_node_min
    max_count           = var.system_node_max
    os_disk_size_gb     = 128
    type                = "VirtualMachineScaleSets"

    node_labels = {
      "reprolab/node-type" = "system"
    }
  }

  # ── Azure Files CSI driver ───────────────────────────────────────────────────
  # Enables the Azure Files StorageClass used for the RWX HF_HOME/pip-cache PVC.
  storage_profile {
    file_driver_enabled = true
    blob_driver_enabled = false
    disk_driver_enabled = false
  }

  # ── Networking ───────────────────────────────────────────────────────────────
  network_profile {
    network_plugin    = "azure"
    network_policy    = "azure"
    load_balancer_sku = "standard"
    outbound_type     = "loadBalancer"
  }

  # ── Azure RBAC ───────────────────────────────────────────────────────────────
  azure_active_directory_role_based_access_control {
    managed                = true
    azure_rbac_enabled     = true
  }

  tags = var.tags
}
