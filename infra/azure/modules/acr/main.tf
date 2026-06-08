# ─── Container Registry ───────────────────────────────────────────────────────

resource "azurerm_container_registry" "main" {
  # ACR names: globally unique, 5-50 lowercase alphanum. Strip hyphens from prefix.
  name                = "${replace(var.prefix, "-", "")}acr"
  resource_group_name = var.resource_group_name
  location            = var.region
  sku                 = var.sku

  # Admin account disabled — nodes authenticate via the kubelet MI (AcrPull below).
  admin_enabled = false

  tags = var.tags
}

# ─── AcrPull → kubelet identity ──────────────────────────────────────────────
# This lets AKS nodes pull images from the registry without any Docker credentials
# or Kubernetes image pull secrets. Zero static secrets.

data "azurerm_subscription" "current" {}

resource "azurerm_role_assignment" "acr_pull" {
  scope                = azurerm_container_registry.main.id
  role_definition_name = "AcrPull"
  principal_id         = var.kubelet_object_id
}
