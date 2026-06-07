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

resource "azurerm_role_assignment" "files_smb_kubelet" {
  scope                = azurerm_storage_account.main.id
  role_definition_name = "Storage File Data SMB Share Contributor"
  principal_id         = var.kubelet_object_id
}
