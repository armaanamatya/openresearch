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
