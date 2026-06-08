# ─── Active-share routing locals ─────────────────────────────────────────────
#
# These locals abstract away whether we are in Standard (default) or Premium
# mode.  Every reference below that needs "which account / share is active"
# reads from here — no conditional logic is scattered across resources.
#
#   files_active_share_id          → scope for the kubelet role assignment
#   files_active_storage_account_name → name to expose in outputs + Helm

locals {
  # Share resource ID used for the kubelet role-assignment scope.
  # In Standard mode: the single azurerm_storage_share.cache resource.
  # In Premium mode:  the dedicated azurerm_storage_share.cache_premium resource.
  files_active_share_id = var.files_premium ? (
    azurerm_storage_share.cache_premium[0].id
  ) : (
    azurerm_storage_share.cache[0].id
  )

  # Storage account name that the active Files share lives on.
  # Helm's StorageClass storageAccount parameter must reference this name.
  files_active_storage_account_name = var.files_premium ? (
    azurerm_storage_account.files_premium[0].name
  ) : (
    azurerm_storage_account.main.name
  )
}

# ─── Storage account ─────────────────────────────────────────────────────────

resource "azurerm_storage_account" "main" {
  name                            = var.storage_account_name
  resource_group_name             = var.resource_group_name
  location                        = var.region
  account_tier                    = "Standard"
  account_replication_type        = "LRS"
  account_kind                    = "StorageV2"
  min_tls_version                 = "TLS1_2"
  allow_nested_items_to_be_public = false

  # P1 security: disable shared key auth — force token-only (workload identity /
  # DefaultAzureCredential).  No code path uses a storage account key; this
  # closes the SAS-token lateral-movement vector entirely.
  shared_access_key_enabled = false

  # P1 security: lock the storage account to the AKS subnet (via service
  # endpoint) plus any operator IPs listed in var.authorized_ip_ranges.
  # All other traffic (including public internet) is denied at the Azure
  # Storage firewall layer — defence-in-depth on top of no-shared-key.
  public_network_access_enabled = false

  network_rules {
    default_action             = "Deny"
    virtual_network_subnet_ids = [var.aks_subnet_id]
    ip_rules                   = var.authorized_ip_ranges
    bypass                     = ["AzureServices"]
  }

  # IMPORTANT — zero static secrets policy:
  # No storage account key is exported in outputs or used by the orchestrator.
  # - Orchestrator-to-Blob auth: DefaultAzureCredential (az login) → user token,
  #   no key required.
  # - AKS Job pods: workload-identity MI (Storage Blob Data Contributor) — see
  #   the identity module.
  # - Azure Files CSI: kubelet MI (Storage File Data SMB Share Contributor below)
  #   with storeAccountKey=false in the StorageClass (Helm L2).
  #
  # If a one-off manual operation absolutely requires a key (e.g. local
  # azcopy migration), generate it on-demand with `az storage account keys list`
  # and rotate it immediately after. Do NOT store it in .tfvars or tfstate
  # outputs. This is the ONLY documented exception to the no-static-secrets rule.

  blob_properties {
    versioning_enabled = true
  }

  tags = var.tags
}

# ─── Artifact Blob container (private) ───────────────────────────────────────
# Bus between the local orchestrator and AKS Job pods.
# Layout: runs/<run_id>/code/** (uploaded by orchestrator)
#         runs/<run_id>/cells/<cell_id>/{metrics.json,status.json,logs/**}

resource "azurerm_storage_container" "artifacts" {
  name                  = var.blob_container_name
  storage_account_name  = azurerm_storage_account.main.name
  container_access_type = "private"
}

# ─── Azure Files share — Standard tier (default, files_premium = false) ───────
# Mounted by every Job pod at /mnt/reprolab-cache.
# Shared across cells so model weights download once per cluster lifetime.
#
# count = 0 when files_premium = true (the Premium share below is used instead).
# count = 1 when files_premium = false (default) — no change to the plan.

resource "azurerm_storage_share" "cache" {
  count                = var.files_premium ? 0 : 1
  name                 = var.files_share_name
  storage_account_name = azurerm_storage_account.main.name
  quota                = var.files_share_quota_gb
}

# ─── Dedicated Premium FileStorage account (files_premium = true only) ────────
# Azure Files Premium cannot live on a general-purpose-v2 (StorageV2) account.
# It requires account_kind="FileStorage" + account_tier="Premium".
# This account is created only when files_premium = true; when false it does not
# exist and the Standard account above is the sole storage account.
#
# Security flags are kept identical to the Standard account:
#   shared_access_key_enabled=false  — token-only access (no SAS key lateral move)
#   public_network_access_enabled=false — storage firewall, subnet-locked
#   network_rules same AKS subnet + authorized IPs + AzureServices bypass
#
# NOTE: FileStorage accounts do not support blob_properties.versioning_enabled.
# NOTE: FileStorage accounts do not support allow_nested_items_to_be_public.

resource "azurerm_storage_account" "files_premium" {
  count                    = var.files_premium ? 1 : 0
  name                     = var.files_premium_storage_account_name
  resource_group_name      = var.resource_group_name
  location                 = var.region
  account_kind             = "FileStorage"
  account_tier             = "Premium"
  account_replication_type = "LRS"
  min_tls_version          = "TLS1_2"

  # Match the Standard account's zero-key-auth policy.
  shared_access_key_enabled = false

  # Lock to AKS subnet + operator IPs, identical to the Standard account.
  public_network_access_enabled = false

  network_rules {
    default_action             = "Deny"
    virtual_network_subnet_ids = [var.aks_subnet_id]
    ip_rules                   = var.authorized_ip_ranges
    bypass                     = ["AzureServices"]
  }

  tags = var.tags
}

# ─── Azure Files share — Premium tier (files_premium = true only) ─────────────
# Premium shares are provisioned-size; Azure enforces a minimum of 100 GiB.
# quota = max(100, files_share_quota_gb) ensures the minimum is always met.

resource "azurerm_storage_share" "cache_premium" {
  count                = var.files_premium ? 1 : 0
  name                 = var.files_share_name
  storage_account_name = azurerm_storage_account.files_premium[0].name
  quota                = max(100, var.files_share_quota_gb)
}

# ─── Storage File Data SMB Share Contributor → kubelet identity ─────────────
# Allows the Azure Files CSI driver (running as kubelet) to manage the share
# without storing a storage account key in a Kubernetes Secret.
# The Helm L2 StorageClass sets storeAccountKey: "false" to rely on this role.
#
# P1 security (least-privilege): scope is narrowed to the ACTIVE Files share
# resource ID (resolved via local.files_active_share_id) rather than the full
# storage account.  The local always points to whichever share is live —
# Standard or Premium — so the role assignment follows the active path without
# granting broader permissions.

resource "azurerm_role_assignment" "files_smb_kubelet" {
  scope                = local.files_active_share_id
  role_definition_name = "Storage File Data SMB Share Contributor"
  principal_id         = var.kubelet_object_id
}
