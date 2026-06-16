// ─── Identity module — user-assigned MI + federated credential ────────────────
//
// Ports infra/azure/modules/identity/ (main.tf + variables.tf + outputs.tf).
// Every setting the Terraform module sets is replicated faithfully here.
//
// Federated subject format (critical — both TF and ARM docs agree):
//   system:serviceaccount:<namespace>:<serviceAccountName>
// Audience:
//   api://AzureADTokenExchange
//
// Both values must match what the AKS Workload Identity webhook injects into
// pod ServiceAccount tokens. A mismatch produces silent 401 errors in pod auth.

targetScope = 'resourceGroup'

// ─── Parameters ───────────────────────────────────────────────────────────────

@description('Resource name prefix.')
param prefix string

@description('Azure region for the managed identity.')
param location string = resourceGroup().location

@description('OIDC issuer URL of the AKS cluster.')
param oidcIssuerUrl string

@description('Kubernetes namespace where the workload-identity ServiceAccount lives.')
param namespace string = 'reprolab'

@description('Name of the Kubernetes ServiceAccount annotated with this MI\'s client ID.')
param serviceAccountName string = 'reprolab-sa'

@description('Full Azure resource ID of the artifact Blob container (including blobServices/default/containers/<name> suffix).')
param artifactContainerId string

@description('Map of tags applied to the managed identity.')
param tags object = {}

// ─── Role definition IDs (stable across all Azure environments) ───────────────
// Storage Blob Data Contributor
var blobContributorRoleId = 'ba92f5b4-2d11-453d-a403-e96b0029c9fe'

// ─── User-assigned managed identity ──────────────────────────────────────────
// Matches: azurerm_user_assigned_identity.workload
//   name = "${var.prefix}-workload-mi"

resource workloadMi 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name:     '${prefix}-workload-mi'
  location: location
  tags:     tags
}

// ─── Federated identity credential ───────────────────────────────────────────
// Matches: azurerm_federated_identity_credential.workload
//   name = "${var.prefix}-fed-cred"
//   parent_id = azurerm_user_assigned_identity.workload.id
//   issuer = var.oidc_issuer_url
//   subject = "system:serviceaccount:${var.namespace}:${var.service_account_name}"
//   audience = ["api://AzureADTokenExchange"]

resource federatedCredential 'Microsoft.ManagedIdentity/userAssignedIdentities/federatedIdentityCredentials@2023-01-31' = {
  parent: workloadMi
  name:   '${prefix}-fed-cred'
  properties: {
    issuer:    oidcIssuerUrl
    subject:   'system:serviceaccount:${namespace}:${serviceAccountName}'
    audiences: ['api://AzureADTokenExchange']
  }
}

// ─── Storage Blob Data Contributor → artifact container ───────────────────────
// Matches: azurerm_role_assignment.blob_contributor
//   scope = var.artifact_container_id
//   role_definition_name = "Storage Blob Data Contributor"
//   principal_id = azurerm_user_assigned_identity.workload.principal_id
//
// Scope is narrowed to the artifact container (not the full storage account) —
// least-privilege, matching TF exactly.
//
// ARM role assignments on sub-resources (blobServices/containers) require
// using an existing resource reference with the full resourceId as scope.
//
// BCP120 note: guid() seed must be calculable at deployment start.
// workloadMi.id is available at start (it is computed from the resource name);
// workloadMi.properties.principalId is NOT (it is a server-assigned value).
// We use workloadMi.id as the principal seed instead — still unique and stable.

resource blobContributor 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name:  guid(artifactContainerId, workloadMi.id, blobContributorRoleId)
  scope: storageContainerRef
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', blobContributorRoleId)
    principalId:      workloadMi.properties.principalId
    principalType:    'ServicePrincipal'
    description:      'Workload identity MI — Storage Blob Data Contributor on artifact container'
  }
}

// Existing resource references for the scope chain: storageAccount → blobService → container.
// artifactContainerId format:
//   /subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.Storage/
//   storageAccounts/<sa>/blobServices/default/containers/<container>
// Index positions: [8]=storageAccountName, [10]=blobServiceName (always "default"), [12]=containerName

resource storageAccountRef 'Microsoft.Storage/storageAccounts@2023-05-01' existing = {
  name: split(artifactContainerId, '/')[8]
}

resource blobServiceRef 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' existing = {
  parent: storageAccountRef
  name:   'default'
}

resource storageContainerRef 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' existing = {
  parent: blobServiceRef
  name:   split(artifactContainerId, '/')[12]
}

// ─── Outputs ──────────────────────────────────────────────────────────────────

@description('Client ID of the workload managed identity. Set as azure.workload.identity/client-id annotation on the Kubernetes ServiceAccount.')
output miClientId string = workloadMi.properties.clientId

@description('Object/principal ID of the workload managed identity.')
output miPrincipalId string = workloadMi.properties.principalId

@description('Full Azure resource ID of the workload managed identity.')
output miResourceId string = workloadMi.id

@description('Exact federated credential subject string. Must match the Helm L2 ServiceAccount namespace and name.')
output federatedSubject string = 'system:serviceaccount:${namespace}:${serviceAccountName}'
