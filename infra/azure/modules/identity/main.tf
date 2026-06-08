# ─── User-assigned managed identity ─────────────────────────────────────────
# Used by AKS Job pods via Workload Identity. Job pods exchange a projected
# ServiceAccount token for a short-lived Azure AD token — no secrets in-cluster.

resource "azurerm_user_assigned_identity" "workload" {
  name                = "${var.prefix}-workload-mi"
  resource_group_name = var.resource_group_name
  location            = var.region
  tags                = var.tags
}

# ─── Federated identity credential ───────────────────────────────────────────
# Links this MI to a specific Kubernetes ServiceAccount (SA) in a specific
# namespace via the cluster's OIDC issuer.
#
# Subject format is EXACTLY:
#   system:serviceaccount:<namespace>:<service-account-name>
# Audience is EXACTLY:
#   api://AzureADTokenExchange
#
# Both values must match what the AKS Workload Identity webhook injects into
# pod ServiceAccount tokens. Changing either silently breaks pod authentication.

resource "azurerm_federated_identity_credential" "workload" {
  name                = "${var.prefix}-fed-cred"
  resource_group_name = var.resource_group_name
  parent_id           = azurerm_user_assigned_identity.workload.id

  # OIDC token issuer — the AKS cluster.
  issuer = var.oidc_issuer_url

  # The federated subject must match the SA projected token exactly.
  subject = "system:serviceaccount:${var.namespace}:${var.service_account_name}"

  # Standard audience for Azure AD Workload Identity.
  audience = ["api://AzureADTokenExchange"]
}

# ─── Storage Blob Data Contributor → artifact container ───────────────────────
# Grants this MI read + write access to the artifact Blob container only
# (not the whole storage account). Job pods call DefaultAzureCredential to
# upload metrics/logs and download code — no SAS tokens, no account keys.

resource "azurerm_role_assignment" "blob_contributor" {
  scope                = var.artifact_container_id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_user_assigned_identity.workload.principal_id
}
