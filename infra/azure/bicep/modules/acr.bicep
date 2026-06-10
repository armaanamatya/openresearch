// ─── ACR module — Standard ACR, admin disabled, AcrPull to kubelet identity ───
//
// Ports infra/azure/modules/acr/ (main.tf + variables.tf + outputs.tf).
// Every setting the Terraform module sets is replicated faithfully here.
//
// ACR name convention: "${replace(var.prefix, "-", "")}acr" (hyphens stripped,
// globally unique, lowercase alphanum only) — exact match to TF.

targetScope = 'resourceGroup'

// ─── Parameters ───────────────────────────────────────────────────────────────

@description('Resource name prefix. ACR name is "<prefix without hyphens>acr". Min 2 chars (so ACR name ≥5 chars, meeting the ACR minimum).')
@minLength(2)
@maxLength(8)
param prefix string

@description('Azure region for the container registry.')
param location string = resourceGroup().location

@description('ACR SKU. Standard is sufficient; Premium adds geo-replication and Private Link.')
@allowed(['Basic', 'Standard', 'Premium'])
param sku string = 'Standard'

@description('Object ID of the AKS kubelet managed identity. Receives AcrPull so nodes can pull images.')
param kubeletObjectId string

@description('Map of tags applied to the registry.')
param tags object = {}

// ─── Role definition IDs (stable across all Azure environments) ───────────────
// AcrPull built-in role ID.
var acrPullRoleId = '7f951dda-4ed3-4680-a7ca-43fe172d538d'

// ─── Container Registry ───────────────────────────────────────────────────────
// Matches: azurerm_container_registry.main
//   name = "${replace(var.prefix, "-", "")}acr"
//   sku = var.sku
//   admin_enabled = false

resource acr 'Microsoft.ContainerRegistry/registries@2023-11-01-preview' = {
  // Strip hyphens from prefix to match TF: replace(var.prefix, "-", "")
  name:     '${replace(prefix, '-', '')}acr'
  location: location
  tags:     tags
  sku: {
    name: sku
  }
  properties: {
    // Admin account disabled — nodes authenticate via kubelet MI (AcrPull below).
    adminUserEnabled: false
  }
}

// ─── AcrPull → kubelet identity ──────────────────────────────────────────────
// Matches: azurerm_role_assignment.acr_pull
//   scope = azurerm_container_registry.main.id
//   role_definition_name = "AcrPull"
//   principal_id = var.kubelet_object_id
//
// Name is guid()-seeded per RULES (scope + principalId + roleDefinitionId).

resource acrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: acr
  name:  guid(acr.id, kubeletObjectId, acrPullRoleId)
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', acrPullRoleId)
    principalId:      kubeletObjectId
    principalType:    'ServicePrincipal'
    description:      'AKS kubelet identity — AcrPull on registry'
  }
}

// ─── Outputs ──────────────────────────────────────────────────────────────────

@description('Full Azure resource ID of the container registry.')
output acrId string = acr.id

@description('Name of the container registry.')
output acrName string = acr.name

@description('ACR login server hostname (e.g. prefixacr.azurecr.io).')
output loginServer string = acr.properties.loginServer
