# Terraform → Bicep Migration: L1 Infrastructure

## Status

**`bicep build` and `bicep lint` pass with zero errors.**
One warning persists (see below).

> **NOT YET PROVEN LIVE.** The Bicep files have been compiled and linted successfully, but no Azure deployment has been executed against a real subscription. Terraform remains authoritative until a live parity deployment succeeds and outputs are verified against the Terraform equivalents.

---

## Deployment command

```bash
# One-time: create the deployment stack (lifecycle-managed, deny-delete on all resources)
az stack group create \
  --name openresearch-infra \
  --resource-group <RESOURCE_GROUP_NAME> \
  --template-file infra/azure/bicep/infra.bicep \
  --parameters infra/azure/bicep/infra.bicepparam \
  --deny-settings-mode denyWriteAndDelete \
  --action-on-unmanage detachAll

# Update in-place (idempotent re-deploy)
az stack group create \
  --name openresearch-infra \
  --resource-group <RESOURCE_GROUP_NAME> \
  --template-file infra/azure/bicep/infra.bicep \
  --parameters infra/azure/bicep/infra.bicepparam \
  --deny-settings-mode denyWriteAndDelete \
  --action-on-unmanage detachAll

# Show outputs
az stack group show \
  --name openresearch-infra \
  --resource-group <RESOURCE_GROUP_NAME> \
  --query outputs
```

The `--action-on-unmanage detachAll` flag detaches (rather than deletes) resources removed from the stack — matching Terraform's default `prevent_destroy` posture during initial parity testing. Flip to `deleteResources` once the stack is stable.

Prerequisites before running:
1. L0 (`main.bicep`) must have already run — the resource group must exist.
2. `az account set --subscription <SUBSCRIPTION_ID>`
3. Copy `infra.bicepparam.example` → `infra.bicepparam` and fill all `<PLACEHOLDER>` values.
4. GPU quota must be filed and approved before the agentPool arm portion succeeds.

---

## TF resource → Bicep resource parity table

| Terraform resource | Bicep resource type | Bicep module | Notes |
|---|---|---|---|
| `azurerm_virtual_network.main` | `Microsoft.Network/virtualNetworks` | `modules/network.bicep` | Exact parity |
| `azurerm_subnet.aks` | `Microsoft.Network/virtualNetworks/subnets` | `modules/network.bicep` | `serviceEndpoints: [Microsoft.Storage]` preserved |
| `azurerm_network_security_group.aks` | `Microsoft.Network/networkSecurityGroups` | `modules/network.bicep` | Both rules (AllowIntraCluster p100, DenyAllOtherInbound p4000) preserved |
| `azurerm_subnet_network_security_group_association.aks` | inline `networkSecurityGroup.id` on subnet | `modules/network.bicep` | No separate ARM type; expressed as subnet property |
| `azurerm_kubernetes_cluster.main` | `Microsoft.ContainerService/managedClusters` | `modules/aks.bicep` | All settings preserved (see detail below) |
| `azurerm_kubernetes_cluster_node_pool.gpu` (× N via for_each) | `Microsoft.ContainerService/managedClusters/agentPools` | `modules/gpu-nodepool.bicep` | Loop via Bicep `[for sku in gpuSkus:]` |
| `azurerm_container_registry.main` | `Microsoft.ContainerRegistry/registries` | `modules/acr.bicep` | admin disabled, Standard SKU |
| `azurerm_role_assignment.acr_pull` | `Microsoft.Authorization/roleAssignments` (scope: ACR) | `modules/acr.bicep` | AcrPull → kubelet identity, guid() seeded |
| `azurerm_storage_account.main` | `Microsoft.Storage/storageAccounts` (Standard_LRS, StorageV2) | `modules/storage.bicep` | All security flags preserved |
| `azurerm_storage_container.artifacts` | `Microsoft.Storage/storageAccounts/blobServices/containers` | `modules/storage.bicep` | privateAccess=None |
| `azurerm_storage_share.cache` | `Microsoft.Storage/storageAccounts/fileServices/shares` | `modules/storage.bicep` | Conditional on `!filesPremium` |
| `azurerm_storage_account.files_premium` | `Microsoft.Storage/storageAccounts` (Premium_LRS, FileStorage) | `modules/storage.bicep` | Conditional on `filesPremium=true` |
| `azurerm_storage_share.cache_premium` | `Microsoft.Storage/storageAccounts/fileServices/shares` | `modules/storage.bicep` | `max(100, quota)` preserved |
| `azurerm_role_assignment.files_smb_kubelet` | `Microsoft.Authorization/roleAssignments` (scope: share) | `modules/storage.bicep` | Split into two conditional resources (Standard/Premium) |
| `azurerm_user_assigned_identity.workload` | `Microsoft.ManagedIdentity/userAssignedIdentities` | `modules/identity.bicep` | Exact parity |
| `azurerm_federated_identity_credential.workload` | `Microsoft.ManagedIdentity/userAssignedIdentities/federatedIdentityCredentials` | `modules/identity.bicep` | subject/audience exact match |
| `azurerm_role_assignment.blob_contributor` | `Microsoft.Authorization/roleAssignments` (scope: container) | `modules/identity.bicep` | Storage Blob Data Contributor, container-scoped |
| *(no TF equivalent)* | `Microsoft.OperationalInsights/workspaces` | `modules/monitoring.bicep` | NEW — security baseline |
| *(no TF equivalent)* | `Microsoft.Insights/diagnosticSettings` × 3 | `modules/monitoring.bicep` | NEW — AKS + ACR + storage account |

### AKS module — settings cross-reference

| TF setting | Bicep property | Status |
|---|---|---|
| `identity { type = "SystemAssigned" }` | `identity.type: 'SystemAssigned'` | ✓ parity |
| `dns_prefix = "${prefix}-aks"` | `properties.dnsPrefix: '${prefix}-aks'` | ✓ parity |
| `kubernetes_version` | `properties.kubernetesVersion` | ✓ parity |
| `api_server_access_profile.authorized_ip_ranges` | `properties.apiServerAccessProfile.authorizedIPRanges` | ✓ parity |
| `oidc_issuer_enabled = true` | `properties.oidcIssuerProfile.enabled: true` | ✓ parity |
| `workload_identity_enabled = true` | `properties.securityProfile.workloadIdentity.enabled: true` | ✓ parity |
| `default_node_pool.name = "system"` | `agentPoolProfiles[0].name: 'system'` | ✓ parity |
| `default_node_pool.vm_size` | `agentPoolProfiles[0].vmSize` | ✓ parity |
| `default_node_pool.vnet_subnet_id` | `agentPoolProfiles[0].vnetSubnetID` | ✓ parity |
| `default_node_pool.enable_auto_scaling = true` | `agentPoolProfiles[0].enableAutoScaling: true` | ✓ parity |
| `default_node_pool.min_count` | `agentPoolProfiles[0].minCount` | ✓ parity |
| `default_node_pool.max_count` | `agentPoolProfiles[0].maxCount` | ✓ parity |
| `default_node_pool.os_disk_size_gb = 128` | `agentPoolProfiles[0].osDiskSizeGB: 128` | ✓ parity |
| `default_node_pool.type = "VirtualMachineScaleSets"` | `agentPoolProfiles[0].type: 'VirtualMachineScaleSets'` | ✓ parity |
| `default_node_pool.node_labels` | `agentPoolProfiles[0].nodeLabels` | ✓ parity |
| `storage_profile.file_driver_enabled = true` | `storageProfile.fileCSIDriver.enabled: true` | ✓ parity |
| `storage_profile.blob_driver_enabled = false` | `storageProfile.blobCSIDriver.enabled: false` | ✓ parity |
| `storage_profile.disk_driver_enabled = false` | `storageProfile.diskCSIDriver.enabled: false` | ✓ parity |
| `network_profile.network_plugin = "azure"` | `networkProfile.networkPlugin: 'azure'` | ✓ parity |
| `network_profile.network_policy = "azure"` | `networkProfile.networkPolicy: 'azure'` | ✓ parity |
| `network_profile.load_balancer_sku = "standard"` | `networkProfile.loadBalancerSku: 'standard'` | ✓ parity |
| `network_profile.outbound_type = "loadBalancer"` | `networkProfile.outboundType: 'loadBalancer'` | ✓ parity |
| `azure_active_directory_role_based_access_control.managed = true` | `aadProfile.managed: true` | ✓ parity |
| `azure_active_directory_role_based_access_control.azure_rbac_enabled = true` | `aadProfile.enableAzureRBAC: true` | ✓ parity |
| `local_account_disabled = true` | `properties.disableLocalAccounts: true` | ✓ parity (TF sets this — not a hardening delta) |

---

## Hardening deltas vs Terraform

These are additions beyond Terraform parity, or intentional omissions documented here.

### Added

| Delta | Module | Rationale |
|---|---|---|
| **Log Analytics workspace** (`${prefix}-law`, PerGB2018, 30-day retention) | `monitoring.bicep` | Security-baseline requirement. Centralised control-plane logs for AKS, registry events for ACR, and storage transaction/capacity metrics. No feature flag — always on (chosen requirement per spec). |
| **AKS diagnostic settings** (kube-apiserver, kube-controller-manager, kube-scheduler, kube-audit, kube-audit-admin, guard, cluster-autoscaler + AllMetrics) | `monitoring.bicep` | Enables Defender for Cloud / Sentinel ingestion. Not in TF. |
| **ACR diagnostic settings** (ContainerRegistryRepositoryEvents, ContainerRegistryLoginEvents + AllMetrics) | `monitoring.bicep` | Registry pull/push audit trail. Not in TF. |
| **Storage account diagnostic settings** (Transaction + Capacity metrics) | `monitoring.bicep` | Storage consumption and anomaly baseline. Not in TF. |

### Parity BREAKS — TF settings deliberately not ported (latent TF bugs)

The Terraform was never deployed live (its live gates stayed blocked-on-tooling);
these three defects would have surfaced on first deployment and are fixed here
rather than ported. Codex security review 2026-06-10 flagged all three.

| TF setting | Bicep behavior | Why the TF version is broken |
|---|---|---|
| `public_network_access_enabled = false` + `network_rules` (both storage accounts) | `publicNetworkAccess: 'Enabled'` + `defaultAction: 'Deny'` + subnet/IP allow-list | `Disabled` permits traffic ONLY via private endpoints — none exist in either IaC — so AKS pods, the Files mount, and the local orchestrator could never reach storage. The network rules were dead code. Enabled+Deny is the semantic the TF comments describe. |
| `shared_access_key_enabled = false` on the files-hosting account | `allowSharedKeyAccess: true` on whichever account hosts the active Files share (`!filesPremium` → the Standard account; `filesPremium` → the Premium account, with the Standard/blob account then staying hardened at `false`) | The azurefile CSI driver mounts SMB with the account key (`storeAccountKey=false` only stops persisting it in a Secret). Key auth disabled = first PVC attach fails. Identity-based SMB (Entra Kerberos) is a future hardening, not a flag-flip. |
| (absent) | **Storage Account Key Operator Service Role → kubelet** on the files-hosting account | The SMB data-plane role TF grants does not include `listKeys`, which the CSI driver needs to fetch the mount key. Without it the mount fails even with key auth enabled. |

### Not added (considered and rejected)

| Item | Reason |
|---|---|
| Private AKS API server (`enablePrivateCluster`) | TF explicitly documents public-API-with-authorized-IP as Phase 1 posture. Becomes private in Phase 2. Adding it here would break the orchestrator's local `kubectl` access without a VPN. |
| Microsoft Defender for Containers | Cost-driven; not in TF. Listed as a future hardening opportunity. |
| Blob soft-delete | Not in TF. Listed as future hardening opportunity. |
| Private endpoints + Private DNS for storage | The correct end-state for `publicNetworkAccess: 'Disabled'`, but requires operator private connectivity (VPN/bastion) the Phase-1 external-team model doesn't have. Pairs with the Phase-2 private API server. |

---

## Known Bicep limitations vs Terraform (with workarounds chosen)

| TF behaviour | Bicep equivalent | Workaround / limitation |
|---|---|---|
| `for_each` over `gpu_skus` map produces a keyed map of module outputs | Bicep module loop produces an **ordered array** of outputs, parallel to the input array | `gpuPools` output is an array `[{ name, skuLabel }, ...]` indexed by position. Callers reconstruct the `shortName → pool` map by zipping with the `gpuSkus` input array (same index). |
| `values(module.gpu_nodepool)[0].pool_name` (first entry by sort order) | `gpuNodepools[0].outputs.poolName` (first entry by declaration order) | Semantically the same when there is one pool (the default). If multiple pools are declared, the Bicep `[0]` picks the first in the parameter array, not the lexicographic sort TF uses. The deprecated `gpu_nodepool_name` output is documented accordingly. |
| `azurerm_subnet_network_security_group_association` is a standalone TF resource | ARM subnet has `networkSecurityGroup` as an inline property | No semantic difference. The inline approach avoids a potential circular dependency and is the correct ARM/Bicep pattern. |
| `max(100, var.files_share_quota_gb)` inline Terraform expression | Bicep ternary: `filesShareQuotaGb > 100 ? filesShareQuotaGb : 100` | Semantically identical. |
| TF `azurerm_role_assignment` with a single `files_smb_kubelet` conditional local | Bicep requires two separate conditional role assignment resources (one for Standard, one for Premium) | ARM deploys exactly one based on the condition. Functionally identical. |
| Role assignment `guid()` seed: TF uses `scope + principalId + roleDefId` | Bicep uses same seed pattern; for identity's blob contributor, `workloadMi.id` replaces `workloadMi.properties.principalId` in the guid seed (BCP120: principalId is not calculable at deploy start) | The GUID differs from the TF-created assignment name. A live parity deployment will create a new role assignment alongside the TF one if both exist in the same RG; after Terraform is decommissioned there is no conflict. |
| AgentPool `tags` in Terraform is propagated via the cluster-level tag blob | `Microsoft.ContainerService/managedClusters/agentPools` has no top-level `tags` property in ARM | Tags are accepted as a parameter but not applied to the agentPool resource (no ARM property exists). Cluster-level tags are set on the AKS module. |
| Storage account `shared_access_key_enabled = false` + `public_network_access_enabled = false` | `allowSharedKeyAccess: false` + `publicNetworkAccess: 'Disabled'` | Exact parity (different property names, same semantics). |
| Storage blob `versioning_enabled = true` | `blobService.properties.isVersioningEnabled: true` | Exact parity. Expressed as a child blobService resource as required by the ARM API. |
| `blob_properties` is a property of `azurerm_storage_account` in TF | In ARM/Bicep blob properties are on the child `blobServices/default` resource | No semantic difference; structural difference in the ARM resource model. |

---

## Remaining warning (verbatim)

```
/home/abheekp/openresearch/infra/azure/bicep/modules/acr.bicep(43,13) : Warning BCP334:
The provided value can have a length as small as 3 and may be too short to assign to a
target with a configured minimum length of 5.
```

**Root cause:** The ACR name is `${replace(prefix, '-', '')}acr`. Bicep's static analyser sees that `replace()` on a `@minLength(2)` `@maxLength(8)` input could theoretically produce a 2-char string, making the total 5 chars — exactly the ACR minimum. Bicep cannot statically prove that `replace()` preserves length (it only removes hyphens), so it conservatively warns. The warning is informational and will not prevent deployment. In practice `prefix` is required to be `≥2` chars of non-hyphen content (e.g. `repro`), producing a name well above the minimum.

---

## Outputs mapping (TF → Bicep)

| `terraform output` name | Bicep output name | Notes |
|---|---|---|
| `cluster_name` | `clusterName` | |
| `cluster_id` | `clusterId` | |
| `oidc_issuer_url` | `oidcIssuerUrl` | |
| `node_resource_group` | `nodeResourceGroup` | |
| `kubelet_identity_client_id` | `kubeletIdentityClientId` | |
| `workload_identity_client_id` | `workloadIdentityClientId` | |
| `acr_login_server` | `acrLoginServer` | |
| `storage_account_name` | `storageAccountNameOut` | |
| `blob_container_name` | `blobContainerNameOut` | |
| `files_share_name` | `filesShareNameOut` | |
| `files_storage_account_name` | `filesStorageAccountName` | |
| `gpu_nodepool_name` | `gpuNodepoolName` | DEPRECATED; use `gpuPools[0].name` |
| `gpu_node_pool_label_key` | `gpuNodePoolLabelKey` | DEPRECATED constant `'reprolab/sku'` |
| `gpu_node_pool_label_value` | `gpuNodePoolLabelValue` | DEPRECATED; use `gpuPools[0].skuLabel` |
| `gpu_taint_key` | `gpuTaintKey` | Constant `'nvidia.com/gpu'` |
| `gpu_pools` | `gpuPools` (array) | Array parallel to `gpuSkus` input; index matches declaration order |
| `vnet_id` | `vnetId` | |
| `aks_subnet_id` | `aksSubnetId` | |
| *(new)* | `logAnalyticsWorkspaceId` | Log Analytics workspace ID (monitoring.bicep) |
