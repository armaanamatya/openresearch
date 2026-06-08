terraform {
  required_version = ">= 1.6.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.110"
    }
  }

  # Bootstrap uses local state intentionally — it creates the remote-state bucket.
  # Commit the resulting terraform.tfstate to a secrets manager or secure storage,
  # NOT to git.
}

provider "azurerm" {
  features {}
  subscription_id = var.subscription_id
  tenant_id       = var.tenant_id
}

# ─── Resource group for remote state ─────────────────────────────────────────

resource "azurerm_resource_group" "tfstate" {
  name     = var.state_resource_group_name
  location = var.region
  tags     = var.tags
}

# ─── Storage account — Standard LRS is sufficient for state blobs ─────────────

resource "azurerm_storage_account" "tfstate" {
  name                            = var.state_storage_account_name
  resource_group_name             = azurerm_resource_group.tfstate.name
  location                        = azurerm_resource_group.tfstate.location
  account_tier                    = "Standard"
  account_replication_type        = "LRS"
  account_kind                    = "StorageV2"
  min_tls_version                 = "TLS1_2"
  allow_nested_items_to_be_public = false

  # P1 security: disable shared key auth — Terraform backend authenticates via
  # service principal / az login (DefaultAzureCredential / OIDC), not account keys.
  # This prevents SAS-token generation against the tfstate bucket.
  shared_access_key_enabled = false

  # NOTE on network_rules: the bootstrap root runs BEFORE the AKS VNet and subnet
  # exist (those are created by the main root).  We cannot reference the AKS subnet
  # here without a circular dependency.  Network restriction of the tfstate bucket
  # must be applied as a post-bootstrap step:
  #   az storage account update \
  #     --name <state_storage_account_name> \
  #     --resource-group <state_resource_group_name> \
  #     --default-action Deny \
  #     --public-network-access Disabled \
  #     --add networkRuleSet.ipRules action=Allow ipAddressOrRange=<operator_ip>
  # Track this as a post-apply runbook step in docs/runbooks/running-the-project.md.
  # The main storage account (modules/storage/main.tf) applies network_rules inline
  # because the AKS subnet ID is available at plan time.

  blob_properties {
    versioning_enabled = true
  }

  tags = var.tags
}

# ─── Private blob container for .tfstate files ────────────────────────────────

resource "azurerm_storage_container" "tfstate" {
  name                  = var.state_container_name
  storage_account_name  = azurerm_storage_account.tfstate.name
  container_access_type = "private"
}
