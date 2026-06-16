// ─── Pipeline identity — user-assigned managed identity + GitHub federation ───
//
// The Contributor-only path: when the admin has already created the resource
// group and granted the operator plain Contributor (no Entra rights, no User
// Access Administrator), the operator can still create this identity themselves —
// a user-assigned managed identity and its federated credentials are ordinary
// ARM resource writes, unlike an Entra app registration.
//
// Deploy (as the operator, RG scope — Contributor is sufficient):
//   az deployment group create \
//     --resource-group <rg> \
//     --template-file pipeline-identity.bicep \
//     --parameters githubOrg=<org> githubRepo=<repo>
//
// What the ADMIN must still do afterwards (one portal visit, two role
// assignments on the RG to this identity's principalId — see README):
//   1. Contributor                            — lets the pipeline deploy L1
//   2. Role Based Access Control Administrator — lets L1 create its in-RG
//      role assignments (AcrPull, Storage Blob/File data roles); grant with a
//      condition restricting assignable roles for least privilege.  Use
//      User Access Administrator instead only if deployment-stack deny-settings
//      are wanted (denyAssignments/write is not in RBAC Administrator).
//
// GitHub Actions then authenticates with azure/login@v2 using this identity's
// clientId — managed identities support GitHub OIDC federation directly; no
// app registration, no secret.

targetScope = 'resourceGroup'

// ─── Parameters ───────────────────────────────────────────────────────────────

@description('Name of the user-assigned managed identity.')
param identityName string = 'id-openresearch-deployer'

@description('Azure region; defaults to the resource group location.')
param location string = resourceGroup().location

@description('GitHub org or user owning the repo (e.g. "Deepinvent").')
param githubOrg string

@description('GitHub repository name (e.g. "scientific_article_generator").')
param githubRepo string

@description('GitHub environment name protected by required reviewers (deploy gate).')
param githubEnvironment string = 'azure'

// ─── User-assigned managed identity ───────────────────────────────────────────

resource uami 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: identityName
  location: location
}

// ─── Federated credential ─────────────────────────────────────────────────────
// ONE subject only: the reviewer-protected GitHub environment.
//   environment:<env>  — workflow_dispatch deploys, approval-gated in GitHub
// Deliberately NO `pull_request` subject: that claim binds neither actor nor
// branch nor workflow content, so any PR able to run a modified workflow could
// exchange it for this identity's Contributor access. PR jobs compile and lint
// Bicep without Azure credentials; the what-if preview runs inside the
// approval-gated deploy job instead.

resource ficDeploy 'Microsoft.ManagedIdentity/userAssignedIdentities/federatedIdentityCredentials@2023-01-31' = {
  parent: uami
  name: 'github-deploy-${githubEnvironment}'
  properties: {
    issuer: 'https://token.actions.githubusercontent.com'
    subject: 'repo:${githubOrg}/${githubRepo}:environment:${githubEnvironment}'
    audiences: ['api://AzureADTokenExchange']
  }
}

// ─── Outputs ──────────────────────────────────────────────────────────────────
// clientId  → GitHub repo variable AZURE_CLIENT_ID
// principalId → what the admin grants the two roles to
output clientId string = uami.properties.clientId
output principalId string = uami.properties.principalId
output tenantId string = uami.properties.tenantId
