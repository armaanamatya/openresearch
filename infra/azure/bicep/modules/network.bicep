// ─── Network module — VNet, AKS subnet, NSG + association ────────────────────
//
// Ports infra/azure/modules/network/ (main.tf + variables.tf + outputs.tf).
// Every setting the Terraform module sets is replicated faithfully here.

targetScope = 'resourceGroup'

// ─── Parameters ───────────────────────────────────────────────────────────────

@description('Short alphanumeric prefix prepended to every resource name.')
param prefix string

@description('Azure region for all network resources.')
param location string = resourceGroup().location

@description('Address space for the VNet (e.g. "10.0.0.0/16").')
param vnetCidr string = '10.0.0.0/16'

@description('Subnet CIDR carved out of vnetCidr for AKS nodes (e.g. "10.0.0.0/22").')
param aksSubnetCidr string = '10.0.0.0/22'

@description('Map of tags applied to network resources.')
param tags object = {}

// ─── Virtual Network ──────────────────────────────────────────────────────────
// Matches: azurerm_virtual_network.main — name = "${var.prefix}-vnet"

resource vnet 'Microsoft.Network/virtualNetworks@2023-11-01' = {
  name:     '${prefix}-vnet'
  location: location
  tags:     tags
  properties: {
    addressSpace: {
      addressPrefixes: [vnetCidr]
    }
  }
}

// ─── AKS subnet ──────────────────────────────────────────────────────────────
// Matches: azurerm_subnet.aks — name = "${var.prefix}-aks-subnet"
// service_endpoints = ["Microsoft.Storage"] — lets pods reach Storage without NAT.

resource aksSubnet 'Microsoft.Network/virtualNetworks/subnets@2023-11-01' = {
  parent: vnet
  name:   '${prefix}-aks-subnet'
  properties: {
    addressPrefix: aksSubnetCidr
    serviceEndpoints: [
      { service: 'Microsoft.Storage' }
    ]
    // NSG is associated below via a separate resource to match TF ordering.
    networkSecurityGroup: {
      id: nsg.id
    }
  }
}

// ─── Network Security Group ──────────────────────────────────────────────────
// Matches: azurerm_network_security_group.aks — name = "${var.prefix}-aks-nsg"
// Two explicit rules: AllowIntraCluster (priority 100) + DenyAllOtherInbound (priority 4000).

resource nsg 'Microsoft.Network/networkSecurityGroups@2023-11-01' = {
  name:     '${prefix}-aks-nsg'
  location: location
  tags:     tags
  properties: {
    securityRules: [
      // Allow intra-cluster traffic (AKS CNI overlay manages pod-to-pod rules internally).
      {
        name: 'AllowIntraCluster'
        properties: {
          priority:                 100
          direction:                'Inbound'
          access:                   'Allow'
          protocol:                 '*'
          sourcePortRange:          '*'
          destinationPortRange:     '*'
          sourceAddressPrefix:      aksSubnetCidr
          destinationAddressPrefix: aksSubnetCidr
        }
      }
      // Deny all other inbound by default (Azure NSG default-deny covers this,
      // but explicit rule makes the intent reviewable).
      {
        name: 'DenyAllOtherInbound'
        properties: {
          priority:                 4000
          direction:                'Inbound'
          access:                   'Deny'
          protocol:                 '*'
          sourcePortRange:          '*'
          destinationPortRange:     '*'
          sourceAddressPrefix:      '*'
          destinationAddressPrefix: '*'
        }
      }
    ]
  }
}

// NOTE: The NSG association is expressed inline in the subnet resource above
// (networkSecurityGroup.id reference).  This is ARM/Bicep canonical: the
// azurerm_subnet_network_security_group_association TF resource has no separate
// ARM type — the association is a property of the subnet.

// ─── Outputs ──────────────────────────────────────────────────────────────────

@description('Resource ID of the virtual network.')
output vnetId string = vnet.id

@description('Name of the virtual network.')
output vnetName string = vnet.name

@description('Resource ID of the AKS subnet.')
output aksSubnetId string = aksSubnet.id

@description('Name of the AKS subnet.')
output aksSubnetName string = aksSubnet.name

@description('Resource ID of the network security group.')
output nsgId string = nsg.id
