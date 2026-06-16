// ─── RG-scoped role grants module ────────────────────────────────────────────
//
// Called twice by main.bicep (once per resource group) with
// scope: resourceGroup(...).  Grants Contributor + User Access Administrator
// to the operator principal on the target RG.
//
// Role assignment names are produced by guid() seeded with
//   scope (RG resource ID) + principalId + roleDefinitionId
// making every assignment name deterministic and idempotent across redeploys.

targetScope = 'resourceGroup'

// ─── Parameters ───────────────────────────────────────────────────────────────

@description('Object ID of the principal to grant roles to.')
param principalId string

@description('Type of the principal: User, Group, or ServicePrincipal.')
@allowed(['User', 'Group', 'ServicePrincipal'])
param principalType string

@description('Resource ID of the resource group being granted (used as guid seed).')
param rgId string

@description('Built-in Contributor role definition ID.')
param contributorRoleId string

@description('Built-in role definition ID for the role-administration grant (User Access Administrator for the human operator; RBAC Administrator for the pipeline principal).')
param secondRoleId string

@description('Optional ABAC condition (version 2.0) restricting which roles the principal may assign/delete. Empty = unconditional.')
param secondRoleCondition string = ''

// ─── Role assignments ─────────────────────────────────────────────────────────

// Contributor — lets Terraform create and manage every resource in the RG.
resource contributorAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  // guid() is deterministic: same inputs → same GUID → idempotent ARM PUT.
  name: guid(rgId, principalId, contributorRoleId)
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', contributorRoleId)
    principalId:      principalId
    principalType:    principalType
    description:      'Terraform L1 operator — Contributor on RG'
  }
}

// Role-administration grant — lets L1 deployments create in-RG role assignments
// (AcrPull on kubelet identity, Storage Blob/File Data roles on MI) without
// the principal holding subscription-level Owner.  When secondRoleCondition is
// set, the assignment is ABAC-constrained to specific assignable roles.
resource secondRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(rgId, principalId, secondRoleId)
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', secondRoleId)
    principalId:      principalId
    principalType:    principalType
    description:      'L1 operator — role-administration grant on RG (scoped role assignment rights)'
    condition:         empty(secondRoleCondition) ? null : secondRoleCondition
    conditionVersion:  empty(secondRoleCondition) ? null : '2.0'
  }
}
