// ─── Storage module — artifact Blob + Azure Files cache ──────────────────────
//
// Ports infra/azure/modules/storage/ (main.tf + variables.tf + outputs.tf).
// Every setting the Terraform module sets is replicated faithfully here.
//
// Two modes controlled by filesPremium:
//   false (default): Standard StorageV2 account hosts both Blob container and
//                    Files share — identical to pre-flag behaviour.
//   true:            A dedicated FileStorage (Premium) account is provisioned;
//                    the cache share is created there instead.  The Standard
//                    account still hosts the artifact Blob container.
//
// Zero static-secrets policy (matches TF):
//   allowSharedKeyAccess = false (token-only, no SAS key lateral move)
//   publicNetworkAccess = 'Disabled' (storage firewall, subnet-locked)

targetScope = 'resourceGroup'

// ─── Parameters ───────────────────────────────────────────────────────────────

@description('Globally unique storage account name (3-24 lowercase alphanum).')
param storageAccountName string

@description('Azure region for storage resources.')
param location string = resourceGroup().location

@description('Name of the private Blob container used as the artifact bus.')
param blobContainerName string = 'reprolab-artifacts'

@description('Name of the Azure Files share mounted by Job pods.')
param filesShareName string = 'reprolab-cache'

@description('Capacity quota for the Azure Files share in GiB.')
param filesShareQuotaGb int = 512

@description('Object ID of the AKS kubelet managed identity. Receives Storage File Data SMB Share Contributor.')
param kubeletObjectId string

@description('Resource ID of the AKS subnet. Added to storage account network rules allowlist.')
param aksSubnetId string

@description('List of operator CIDR(s) or single IPv4 addresses allowed outside the VNet.')
param authorizedIpRanges array = []

@description('When true, provision a dedicated FileStorage (Premium) account for the cache share.')
param filesPremium bool = false

@description('Globally unique name for the Premium FileStorage account (filesPremium = true only).')
param filesPremiumStorageAccountName string = ''

@description('Map of tags applied to storage resources.')
param tags object = {}

// ─── Role definition IDs (stable across all Azure environments) ───────────────
// Storage File Data SMB Share Contributor
var filesSmbContributorRoleId = '0c867c2a-1d8c-454a-a3db-ab2ea1bdc8bb'
// Storage Account Key Operator Service Role — the azurefile CSI driver mounts
// SMB with the account key fetched via listKeys (storeAccountKey=false only
// stops it persisting the key in a Secret); the SMB data role above does NOT
// include listKeys, so the kubelet needs this on the files-hosting account.
var keyOperatorRoleId = '81a9662b-bebf-436f-a333-f67b29880f12'

// ─── Standard StorageV2 account ───────────────────────────────────────────────
// Matches: azurerm_storage_account.main
//   account_tier = "Standard", account_replication_type = "LRS"
//   account_kind = "StorageV2", min_tls_version = "TLS1_2"
//   allow_nested_items_to_be_public = false
//   shared_access_key_enabled = false (token-only access)
//   public_network_access_enabled = false (storage firewall)
//   network_rules { default_action = "Deny", virtual_network_subnet_ids, ip_rules, bypass = ["AzureServices"] }
//   blob_properties { versioning_enabled = true }

resource storageAccount 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name:     storageAccountName
  location: location
  tags:     tags
  kind:     'StorageV2'
  sku: {
    name: 'Standard_LRS'
  }
  properties: {
    minimumTlsVersion:       'TLS1_2'
    // allow_nested_items_to_be_public = false
    allowBlobPublicAccess:   false
    // PARITY BREAK (see MIGRATION.md): TF sets shared_access_key_enabled=false,
    // but the azurefile CSI driver can only mount SMB with key auth — so shared
    // key stays enabled while this account hosts the Files share
    // (filesPremium=false) and is disabled when the share lives on the
    // dedicated Premium account.
    allowSharedKeyAccess:    !filesPremium
    // PARITY BREAK (see MIGRATION.md): TF disables public network access AND
    // configures firewall rules — contradictory (Disabled = private endpoints
    // only, and none exist, making storage unreachable). Enabled + Deny default
    // + subnet/IP allow-list is the intended "firewall, subnet-locked" semantic.
    publicNetworkAccess:     'Enabled'
    networkAcls: {
      defaultAction:            'Deny'
      virtualNetworkRules: [
        {
          id:     aksSubnetId
          action: 'Allow'
        }
      ]
      ipRules: [for cidr in authorizedIpRanges: {
        value:  cidr
        action: 'Allow'
      }]
      bypass: 'AzureServices'
    }
  }
}

// ─── Blob service — versioning enabled ───────────────────────────────────────
// Matches: blob_properties { versioning_enabled = true }

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  parent: storageAccount
  name:   'default'
  properties: {
    isVersioningEnabled: true
  }
}

// ─── Artifact Blob container (private) ───────────────────────────────────────
// Matches: azurerm_storage_container.artifacts
//   name = var.blob_container_name, container_access_type = "private"

resource artifactContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name:   blobContainerName
  properties: {
    publicAccess: 'None'
  }
}

// ─── Azure Files share — Standard tier (filesPremium = false) ─────────────────
// Matches: azurerm_storage_share.cache (count = filesPremium ? 0 : 1)
//   name = var.files_share_name, quota = var.files_share_quota_gb

resource filesShareStandard 'Microsoft.Storage/storageAccounts/fileServices/shares@2023-05-01' = if (!filesPremium) {
  name:   '${storageAccountName}/default/${filesShareName}'
  properties: {
    shareQuota: filesShareQuotaGb
  }
  dependsOn: [storageAccount]
}

// ─── Dedicated Premium FileStorage account (filesPremium = true only) ─────────
// Matches: azurerm_storage_account.files_premium (count = filesPremium ? 1 : 0)
//   account_kind = "FileStorage", account_tier = "Premium",
//   account_replication_type = "LRS", min_tls_version = "TLS1_2"
//   shared_access_key_enabled = false, public_network_access_enabled = false
//   network_rules { same as Standard account }
//
// NOTE: FileStorage accounts do NOT support blob versioning or
//       allow_nested_items_to_be_public — those properties are omitted here.

resource filesPremiumAccount 'Microsoft.Storage/storageAccounts@2023-05-01' = if (filesPremium) {
  name:     filesPremiumStorageAccountName
  location: location
  tags:     tags
  kind:     'FileStorage'
  sku: {
    name: 'Premium_LRS'
  }
  properties: {
    minimumTlsVersion:    'TLS1_2'
    // PARITY BREAKS (see MIGRATION.md): this account hosts the active Files
    // share, so shared key must stay enabled for the CSI SMB mount; public
    // network access Enabled with Deny-default firewall (Disabled would
    // require private endpoints that do not exist).
    allowSharedKeyAccess: true
    publicNetworkAccess:  'Enabled'
    networkAcls: {
      defaultAction:            'Deny'
      virtualNetworkRules: [
        {
          id:     aksSubnetId
          action: 'Allow'
        }
      ]
      ipRules: [for cidr in authorizedIpRanges: {
        value:  cidr
        action: 'Allow'
      }]
      bypass: 'AzureServices'
    }
  }
}

// ─── Azure Files share — Premium tier (filesPremium = true only) ──────────────
// Matches: azurerm_storage_share.cache_premium (count = filesPremium ? 1 : 0)
//   quota = max(100, var.files_share_quota_gb)  — Premium minimum is 100 GiB.

resource filesSharePremium 'Microsoft.Storage/storageAccounts/fileServices/shares@2023-05-01' = if (filesPremium) {
  name:   '${filesPremiumStorageAccountName}/default/${filesShareName}'
  properties: {
    // max(100, filesShareQuotaGb) — enforces Premium minimum of 100 GiB.
    shareQuota: filesShareQuotaGb > 100 ? filesShareQuotaGb : 100
  }
  dependsOn: [filesPremiumAccount]
}

// ─── Storage File Data SMB Share Contributor → kubelet identity ───────────────
// Matches: azurerm_role_assignment.files_smb_kubelet
//   scope = local.files_active_share_id  (narrowed to the active share)
//   role_definition_name = "Storage File Data SMB Share Contributor"
//   principal_id = var.kubelet_object_id
//
// P1 security (least-privilege): scope is narrowed to the active share resource
// (Standard when filesPremium=false, Premium share when filesPremium=true).
//
// Bicep conditional: two separate role assignment resources gated on filesPremium.
// ARM will deploy exactly one.

resource filesSmbKubeletStandard 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!filesPremium) {
  // Scope inline on the Standard Files share.
  scope: filesShareStandard
  name:  guid(filesShareStandard.id, kubeletObjectId, filesSmbContributorRoleId)
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', filesSmbContributorRoleId)
    principalId:      kubeletObjectId
    principalType:    'ServicePrincipal'
    description:      'AKS kubelet identity — Storage File Data SMB Share Contributor (Standard)'
  }
}

resource filesSmbKubeletPremium 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (filesPremium) {
  scope: filesSharePremium
  name:  guid(filesSharePremium.id, kubeletObjectId, filesSmbContributorRoleId)
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', filesSmbContributorRoleId)
    principalId:      kubeletObjectId
    principalType:    'ServicePrincipal'
    description:      'AKS kubelet identity — Storage File Data SMB Share Contributor (Premium)'
  }
}

// ─── Storage Account Key Operator → kubelet identity (files-hosting account) ──
// NOT in TF (see MIGRATION.md): without listKeys the CSI driver cannot fetch
// the key it mounts with, and the first PVC attach fails at runtime. Scoped to
// the whole account (listKeys is an account-level action — no narrower scope
// exists). Exactly one of the two deploys, matching the SMB-role pattern above.

resource keyOperatorKubeletStandard 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!filesPremium) {
  scope: storageAccount
  name:  guid(storageAccount.id, kubeletObjectId, keyOperatorRoleId)
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', keyOperatorRoleId)
    principalId:      kubeletObjectId
    principalType:    'ServicePrincipal'
    description:      'AKS kubelet identity — Storage Account Key Operator for CSI SMB mount (Standard)'
  }
}

resource keyOperatorKubeletPremium 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (filesPremium) {
  scope: filesPremiumAccount
  name:  guid(filesPremiumStorageAccountName, kubeletObjectId, keyOperatorRoleId)
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', keyOperatorRoleId)
    principalId:      kubeletObjectId
    principalType:    'ServicePrincipal'
    description:      'AKS kubelet identity — Storage Account Key Operator for CSI SMB mount (Premium)'
  }
}

// ─── Outputs ──────────────────────────────────────────────────────────────────

@description('Name of the storage account.')
output storageAccountName string = storageAccount.name

@description('Full Azure resource ID of the storage account.')
output storageAccountId string = storageAccount.id

@description('Name of the private artifact Blob container.')
output blobContainerName string = artifactContainer.name

@description('Full Azure resource ID of the artifact Blob container (including blobServices/default/containers/<name> suffix).')
output blobContainerResourceId string = '${storageAccount.id}/blobServices/default/containers/${artifactContainer.name}'

@description('Name of the active Azure Files share (Standard or Premium).')
output filesShareName string = filesShareName

@description('Name of the storage account hosting the active Files share. When filesPremium=false: same as storageAccountName. When filesPremium=true: the dedicated Premium FileStorage account.')
output filesStorageAccountName string = filesPremium ? filesPremiumStorageAccountName : storageAccountName
