// ─── L0 Access Bootstrap — subscription scope ────────────────────────────────
//
// Run ONCE by a subscription Owner before Terraform / Bicep L1.  Creates the
// main and (optionally) tfstate resource groups, then grants the operator
// principal Contributor + User Access Administrator on each RG.  After this
// deployment the operator principal can run L1 without subscription-level Owner.
//
// Pass deployPrincipalId (the OIDC service-principal object ID produced by
// admin-bootstrap.sh) to grant the same RG-scoped roles to GitHub Actions — no
// standing human credentials are needed after that.
//
// Deploy:
//   az deployment sub create \
//     --location <region> \
//     --template-file main.bicep \
//     --parameters main.bicepparam
//
// Re-deploying is safe: resource-group creates are idempotent and role-assignment
// names are seeded with guid() so they are stable across runs.

targetScope = 'subscription'

// ─── Parameters ───────────────────────────────────────────────────────────────

@description('Azure region for both resource groups (e.g. "eastus").')
param location string

@description('Name of the main resource group managed by Bicep L1.')
param mainRgName string = 'rg-reprolab'

@description('Tags applied to both resource groups.')
param tags object = {
  project: 'reprolab'
  managedBy: 'bicep-l0'
}

@description('Create the tfstate resource group as well as the main RG.')
param createTfstateRg bool = true

@description('Name of the Terraform remote-state resource group (bootstrap).')
param tfstateRgName string = 'rg-reprolab-tfstate'

// Optional: empty skips the operator grants entirely (e.g. when only the OIDC
// deploy principal needs access, or the admin granted the operator manually).
@description('Object ID of the principal (user, group, or service principal) to grant Contributor + User Access Administrator on each RG. Leave empty to skip.')
param principalId string = ''

@description('Type of the principal: User, Group, or ServicePrincipal.')
@allowed(['User', 'Group', 'ServicePrincipal'])
param principalType string = 'Group'

// Optional: object ID of the GitHub Actions OIDC service principal.  When
// non-empty, receives the same Contributor + User Access Administrator grants
// as the operator principal on each RG, enabling keyless GitHub Actions deploys.
// Leave empty ('') to skip — no assignments are created (condition guard below).
@description('Object ID of the GitHub Actions OIDC service principal (leave empty to skip).')
param deployPrincipalId string = ''

// ─── Built-in role definition IDs (subscription-scope constants) ──────────────
// These IDs are stable across all Azure environments.

var contributorRoleId = 'b24988ac-6180-42a0-ab88-20f7382dd24c'
var uaaRoleId         = '18d7d88d-d35e-4fb5-a5c3-7773c20a72d9'
// RBAC Administrator — role-assignment rights only (no denyAssignments, unlike
// UAA).  Granted to the PIPELINE principal, ABAC-constrained to exactly the
// roles L1 assigns.
var rbacAdminRoleId   = 'f58310d9-a9f6-439a-9e8d-f62e7b41a168'

// The only roles L1 creates assignments for: AcrPull, Storage Blob Data
// Contributor, Storage File Data SMB Share Contributor, Storage Account Key
// Operator Service Role.  The pipeline can assign/delete these and nothing else.
var l1AssignableRoleIds = '7f951dda-4ed3-4680-a7ca-43fe172d538d, ba92f5b4-2d11-453d-a403-e96b0029c9fe, 0c867c2a-1d8c-454a-a3db-ab2ea1bdc8bb, 81a9662b-bebf-436f-a333-f67b29880f12'
var deployRoleCondition = '((!(ActionMatches{\'Microsoft.Authorization/roleAssignments/write\'})) OR (@Request[Microsoft.Authorization/roleAssignments:RoleDefinitionId] ForAnyOfAnyValues:GuidEquals {${l1AssignableRoleIds}})) AND ((!(ActionMatches{\'Microsoft.Authorization/roleAssignments/delete\'})) OR (@Resource[Microsoft.Authorization/roleAssignments:RoleDefinitionId] ForAnyOfAnyValues:GuidEquals {${l1AssignableRoleIds}}))'

// ─── Main resource group ──────────────────────────────────────────────────────

resource mainRg 'Microsoft.Resources/resourceGroups@2022-09-01' = {
  name:     mainRgName
  location: location
  tags:     tags
}

// ─── Tfstate resource group (conditional) ────────────────────────────────────

resource tfstateRg 'Microsoft.Resources/resourceGroups@2022-09-01' = if (createTfstateRg) {
  name:     tfstateRgName
  location: location
  tags:     tags
}

// ─── Role grants — main RG ────────────────────────────────────────────────────
// Bicep cannot create role assignments at RG scope from a subscription-scope
// deployment directly — a nested module with scope: resourceGroup(...) is
// required.  This is the correct pattern per ARM/Bicep docs.

module mainRgGrants 'rg-grants.bicep' = if (!empty(principalId)) {
  name:  'mainRgGrants'
  scope: mainRg
  params: {
    principalId:   principalId
    principalType: principalType
    rgId:          mainRg.id
    contributorRoleId: contributorRoleId
    secondRoleId:      uaaRoleId
  }
}

// ─── Role grants — tfstate RG (conditional) ───────────────────────────────────

module tfstateRgGrants 'rg-grants.bicep' = if (createTfstateRg && !empty(principalId)) {
  name:  'tfstateRgGrants'
  scope: tfstateRg
  params: {
    principalId:   principalId
    principalType: principalType
    rgId:          tfstateRg.id
    contributorRoleId: contributorRoleId
    secondRoleId:      uaaRoleId
  }
}

// ─── Role grants — OIDC deploy principal (main RG only) ───────────────────────
// Only created when deployPrincipalId is non-empty.  Differences from the
// operator grants, both deliberate least-privilege choices:
//   * RBAC Administrator instead of User Access Administrator, ABAC-constrained
//     to exactly the four roles L1 assigns — the pipeline cannot grant itself
//     (or anyone) Owner/Contributor.
//   * NO tfstate-RG grants: Bicep deploys keep no state; only the human
//     operator running legacy Terraform needs that RG.

module mainRgDeployGrants 'rg-grants.bicep' = if (!empty(deployPrincipalId)) {
  name:  'mainRgDeployGrants'
  scope: mainRg
  params: {
    principalId:   deployPrincipalId
    principalType: 'ServicePrincipal'
    rgId:          mainRg.id
    contributorRoleId:   contributorRoleId
    secondRoleId:        rbacAdminRoleId
    secondRoleCondition: deployRoleCondition
  }
}

// ─── Outputs ──────────────────────────────────────────────────────────────────

output mainRgName string = mainRg.name
output mainRgId   string = mainRg.id

output tfstateRgName string = createTfstateRg ? tfstateRg.name : ''
output tfstateRgId   string = createTfstateRg ? tfstateRg.id   : ''
