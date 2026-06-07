# ─── Virtual Network ──────────────────────────────────────────────────────────

resource "azurerm_virtual_network" "main" {
  name                = "${var.prefix}-vnet"
  resource_group_name = var.resource_group_name
  location            = var.region
  address_space       = [var.vnet_cidr]
  tags                = var.tags
}

# ─── AKS subnet ──────────────────────────────────────────────────────────────

resource "azurerm_subnet" "aks" {
  name                 = "${var.prefix}-aks-subnet"
  resource_group_name  = var.resource_group_name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = [var.aks_subnet_cidr]

  # Service endpoints let pods reach Azure Storage without a NAT hop.
  service_endpoints = ["Microsoft.Storage"]
}

# ─── Network Security Group ──────────────────────────────────────────────────

resource "azurerm_network_security_group" "aks" {
  name                = "${var.prefix}-aks-nsg"
  resource_group_name = var.resource_group_name
  location            = var.region
  tags                = var.tags

  # Allow intra-cluster traffic (AKS CNI overlay manages pod-to-pod rules internally).
  security_rule {
    name                       = "AllowIntraCluster"
    priority                   = 100
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "*"
    source_port_range          = "*"
    destination_port_range     = "*"
    source_address_prefix      = var.aks_subnet_cidr
    destination_address_prefix = var.aks_subnet_cidr
  }

  # Deny all other inbound by default (Azure NSG default-deny covers this,
  # but explicit rule makes the intent reviewable).
  security_rule {
    name                       = "DenyAllOtherInbound"
    priority                   = 4000
    direction                  = "Inbound"
    access                     = "Deny"
    protocol                   = "*"
    source_port_range          = "*"
    destination_port_range     = "*"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }
}

# ─── Associate NSG → AKS subnet ──────────────────────────────────────────────

resource "azurerm_subnet_network_security_group_association" "aks" {
  subnet_id                 = azurerm_subnet.aks.id
  network_security_group_id = azurerm_network_security_group.aks.id
}
