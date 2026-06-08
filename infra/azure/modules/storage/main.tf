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

# ─── Azure Files share (RWX HF_HOME + pip cache) ─────────────────────────────
# Mounted by every Job pod at /mnt/reprolab-cache.
# Shared across cells so model weights download once per cluster lifetime.

resource "azurerm_storage_share" "cache" {
  name                 = var.files_share_name
  storage_account_name = azurerm_storage_account.main.name
  quota                = var.files_share_quota_gb
}

# ─── Storage File Data SMB Share Contributor → kubelet identity ─────────────
# Allows the Azure Files CSI driver (running as kubelet) to manage the share
# without storing a storage account key in a Kubernetes Secret.
# The Helm L2 StorageClass sets storeAccountKey: "false" to rely on this role.
#
# P1 security (least-privilege): scope is narrowed to the specific Files SHARE
# resource ID instead of the whole storage account. This prevents the kubelet
# identity from accessing any other share or Blob container in the account.
# The Blob Data Contributor assignment (identity module) is already
# container-scoped — no change needed there.

resource "azurerm_role_assignment" "files_smb_kubelet" {
  scope                = azurerm_storage_share.cache.id
  role_definition_name = "Storage File Data SMB Share Contributor"
  principal_id         = var.kubelet_object_id
}
