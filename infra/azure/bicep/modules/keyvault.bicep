// ─── Key Vault module — hardened secrets store for orchestrator credentials ──
//
// Creates a Key Vault that holds the API-key secrets the orchestrator needs:
//   • azure-openai-api-key   (AZURE_OPENAI_API_KEY / OPENRESEARCH_AZURE_OPENAI_API_KEY)
//   • anthropic-api-key      (ANTHROPIC_API_KEY)
//   • claude-code-oauth-token (CLAUDE_CODE_OAUTH_TOKEN — Stream B, long-lived headless
//                              OAuth token minted by `claude setup-token`; enables
//                              unattended --model claude-oauth runs inside AKS at $0/token)
//   • azure-foundry-api-key  (AZURE_FOUNDRY_API_KEY — grok root/sub-agent; opt-in,
//                              enables a fully OAuth-free run with model=azure-foundry)
//   • openai-api-key         (OPENAI_API_KEY — OpenAI sub-agent; opt-in)
//
// Secret VALUES are NEVER stored in this file or any params file.  They must be
// injected out-of-band by an operator with Key Vault Administrator/Officer rights:
//
//   az keyvault secret set \
//     --vault-name <keyVaultName> \
//     --name azure-openai-api-key \
//     --value "$(op read op://…)"      # value sourced from a gitignored secret store
//
//   az keyvault secret set \
//     --vault-name <keyVaultName> \
//     --name anthropic-api-key \
//     --value "$(op read op://…)"
//
//   # OAuth-free providers (opt-in; see orchestrator-deployment.yaml
//   # .Values.orchestrator.azureFoundry.apiKey.enabled / .openaiApiKey.enabled):
//   az keyvault secret set \
//     --vault-name <keyVaultName> \
//     --name azure-foundry-api-key \
//     --value "$(op read op://…)"
//   az keyvault secret set \
//     --vault-name <keyVaultName> \
//     --name openai-api-key \
//     --value "$(op read op://…)"
//
//   # Stream B — long-lived OAuth token (opt-in; see orchestrator-deployment.yaml
//   # .Values.orchestrator.claudeOauthToken.enabled):
//   az keyvault secret set \
//     --vault-name <keyVaultName> \
//     --name claude-code-oauth-token \
//     --value "$(claude setup-token)"
//
// Security posture:
//   • RBAC-only (enableRbacAuthorization = true) — no legacy access policies.
//   • Soft-delete + purge-protection enabled — accidental deletions are recoverable.
//   • Public endpoint enabled with a Deny-default firewall; only the AKS subnet and
//     operator egress IPs may reach it (mirrors the storage module's network pattern).
//   • The orchestrator UAMI receives "Key Vault Secrets User" (read-only) on this vault.
//     An operator must separately hold "Key Vault Administrator" or "Key Vault Officer"
//     to write secrets — workload access is read-only by design.

targetScope = 'resourceGroup'

// ─── Parameters ───────────────────────────────────────────────────────────────

@description('Key Vault name (3-24 chars, globally unique, lowercase alphanum + hyphens).')
param name string

@description('Azure region for the Key Vault.')
param location string = resourceGroup().location

@description('Map of tags applied to the Key Vault.')
param tags object = {}

@description('List of operator CIDR(s) or single IPv4 addresses allowed to reach the vault outside the VNet. Mirror of the authorizedIpRanges param passed to storage.bicep.')
param authorizedIpRanges array = []

@description('Resource ID of the AKS subnet. Added to the vault\'s network allow-list so in-cluster workloads can reach the vault without leaving the VNet.')
param aksSubnetId string

@description('Principal ID of the orchestrator user-assigned managed identity. Receives the "Key Vault Secrets User" role (read-only) on this vault.')
param secretsUserPrincipalId string

// ─── Role definition IDs (stable across all Azure environments) ───────────────
// Key Vault Secrets User — read secret values; no write, no list metadata.
var kvSecretsUserRoleId = '4633458b-17de-408a-b874-0445c86b69e6'

// ─── Key Vault ────────────────────────────────────────────────────────────────
// enableSoftDelete: true  — enforced by Azure since 2020; explicit for clarity.
// enablePurgeProtection: true — prevents permanent deletion during the
//   soft-delete retention period; required for production credential stores.
// enableRbacAuthorization: true — turns off legacy access-policy plane; all
//   permission is via Azure RBAC role assignments on this resource.
// publicNetworkAccess: 'Enabled' + networkAcls.defaultAction: 'Deny' — the
//   "Disabled" value would require private endpoints (none exist in this design);
//   Enabled + Deny-default + subnet/IP allow-list gives the same isolation with
//   no private-endpoint cost.

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name:     name
  location: location
  tags:     tags
  properties: {
    sku: {
      family: 'A'
      name:   'standard'
    }
    tenantId:                  tenant().tenantId
    enableRbacAuthorization:   true
    enableSoftDelete:          true
    softDeleteRetentionInDays: 90
    enablePurgeProtection:     true
    publicNetworkAccess:       'Enabled'
    networkAcls: {
      defaultAction:            'Deny'
      bypass:                   'AzureServices'
      virtualNetworkRules: [
        {
          id:                               aksSubnetId
          ignoreMissingVnetServiceEndpoint: false
        }
      ]
      // Strip /32 — Key Vault (like Storage) rejects /31 and /32 IP network rules;
      // a single host must be a bare IPv4 (only /0–/30 ranges are accepted).
      ipRules: [for cidr in authorizedIpRanges: {
        value: replace(cidr, '/32', '')
      }]
    }
  }
}

// ─── Key Vault Secrets User → orchestrator identity ───────────────────────────
// Scope: the vault (read-secret-value is a vault-scoped action).
// PrincipalType: ServicePrincipal — user-assigned MI principal IDs are always
//   service principals from an RBAC perspective.
//
// guid() seed: use the vault resource ID + principalId + role ID so the
// assignment name is stable across re-deployments (idempotent).

resource kvSecretsUserAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: keyVault
  name:  guid(keyVault.id, secretsUserPrincipalId, kvSecretsUserRoleId)
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', kvSecretsUserRoleId)
    principalId:      secretsUserPrincipalId
    principalType:    'ServicePrincipal'
    description:      'Orchestrator UAMI — Key Vault Secrets User (read API keys from vault)'
  }
}

// ─── Outputs ──────────────────────────────────────────────────────────────────
//
// Expected secret NAMES (values are NEVER set by Bicep — set out-of-band):
//   azure-openai-api-key   →  env AZURE_OPENAI_API_KEY / OPENRESEARCH_AZURE_OPENAI_API_KEY
//   anthropic-api-key      →  env ANTHROPIC_API_KEY
//   claude-code-oauth-token →  env CLAUDE_CODE_OAUTH_TOKEN (Stream B; opt-in headless OAuth)
//   azure-foundry-api-key  →  env AZURE_FOUNDRY_API_KEY (grok root/sub-agent; opt-in OAuth-free)
//   openai-api-key         →  env OPENAI_API_KEY (OpenAI sub-agent; opt-in OAuth-free)

@description('Name of the Key Vault. Pass to SecretProviderClass.parameters.keyvaultName and to `az keyvault secret set`.')
output keyVaultName string = keyVault.name

@description('HTTPS URI of the Key Vault (e.g. https://<name>.vault.azure.net/). Used in SecretProviderClass.parameters.keyvaultName is preferred; this is available for diagnostic use.')
output keyVaultUri string = keyVault.properties.vaultUri
