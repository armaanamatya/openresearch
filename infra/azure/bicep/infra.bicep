// ─── L1 Infrastructure — RG-scoped root ──────────────────────────────────────
//
// Ports infra/azure/main.tf (module wiring) and mirrors its parameter flow.
// Outputs cover everything infra/azure/outputs.tf exposes.
//
// Deploy with a deployment stack for lifecycle management (deny-delete on all
// managed resources, detach-on-unmanage for graceful decommission):
//
//   az stack group create \
//     --name openresearch-infra \
//     --resource-group <rg> \
//     --template-file infra.bicep \
//     --parameters infra.bicepparam \
//     --deny-settings-mode denyWriteAndDelete \
//     --action-on-unmanage detachAll
//
// The resource group itself is NOT created here — it was created by L0 (main.bicep).
// This deployment targets an existing RG (targetScope = 'resourceGroup').

targetScope = 'resourceGroup'

// ─── Parameters ───────────────────────────────────────────────────────────────
// Mirror of infra/azure/variables.tf.  Deprecated vars (gpu_max_nodes,
// state_storage_account_name) are not forwarded — they have no wired consumers.

@description('Short alphanumeric prefix prepended to every resource name (e.g. "repro"). Keep ≤8 chars.')
param prefix string

@description('Azure region for all resources. Must have Standard_NC24ads_A100_v4 quota available.')
param location string = resourceGroup().location

@description('Map of tags applied to every resource.')
param tags object = {}

// ─── Networking ───────────────────────────────────────────────────────────────

@description('Address space for the VNet (e.g. "10.0.0.0/16").')
param vnetCidr string = '10.0.0.0/16'

@description('Subnet CIDR carved out of vnetCidr for AKS nodes (e.g. "10.0.0.0/22").')
param aksSubnetCidr string = '10.0.0.0/22'

@description('List of operator CIDR(s) allowed to reach the AKS public API server. Required.')
param authorizedIpRanges array

// ─── AKS cluster ──────────────────────────────────────────────────────────────

@description('Kubernetes version for the AKS cluster (e.g. "1.29"). Pin to a version available in the region.')
param kubernetesVersion string

@description('VM SKU for the system (CPU) node pool.')
param systemNodeSku string = 'Standard_D4s_v5'

@description('Minimum nodes in the system pool.')
param systemNodeMin int = 1

@description('Maximum nodes in the system pool.')
param systemNodeMax int = 3

// ─── GPU node pools ────────────────────────────────────────────────────────────
//
// One AKS scale-to-zero node pool is created per entry.
// Default: a single A100-80 pool (one vCPU quota ask).
// Fields:
//   shortName      — catalog identifier; written to the 'reprolab/sku' node label.
//   vmSize         — Azure VM SKU.
//   gpuCount       — GPUs per node; written to the 'nvidia.com/gpu' node label.
//   poolSuffix     — short suffix (≤7 chars, lowercase-alnum) appended to <prefix>
//                    to form the AKS pool name (≤12 chars total).
//   maxNodes       — maximum autoscaler node count (min is always 0).
//   osDiskSizeGb   — OS disk size in GiB.

type gpuSkuObject = {
  shortName:    string
  vmSize:       string
  gpuCount:     int
  poolSuffix:   string
  maxNodes:     int
  osDiskSizeGb: int
}

@description('List of GPU SKU objects — one AKS scale-to-zero node pool per entry.')
param gpuSkus gpuSkuObject[] = [
  {
    shortName:    'azure_a100_80'
    vmSize:       'Standard_NC24ads_A100_v4'
    gpuCount:     1
    poolSuffix:   'a10080'
    maxNodes:     4
    osDiskSizeGb: 256
  }
]

// ─── Container registry ───────────────────────────────────────────────────────

@description('ACR SKU. Standard is sufficient; Premium adds geo-replication and Private Link.')
@allowed(['Basic', 'Standard', 'Premium'])
param acrSku string = 'Standard'

// ─── Storage ──────────────────────────────────────────────────────────────────

@description('Globally unique storage account name (3-24 lowercase alphanum).')
param storageAccountName string

@description('Name of the private Blob container used as the artifact bus.')
param blobContainerName string = 'reprolab-artifacts'

@description('Name of the Azure Files share mounted by Jobs as the RWX HuggingFace / pip cache.')
param filesShareName string = 'reprolab-cache'

@description('Capacity quota of the Azure Files share in GiB.')
param filesShareQuotaGb int = 512

@description('When true, provision a dedicated Premium FileStorage account for the cache share.')
param filesPremium bool = false

@description('Name of the dedicated Premium FileStorage account (filesPremium = true only).')
param filesPremiumStorageAccountName string = ''

// ─── Workload identity ────────────────────────────────────────────────────────

@description('Kubernetes namespace where the workload-identity ServiceAccount lives.')
param workloadIdentityNamespace string = 'reprolab'

@description('Name of the Kubernetes ServiceAccount annotated with the workload-identity client ID.')
param workloadIdentityServiceAccount string = 'reprolab-sa'

// ─── Module wiring ────────────────────────────────────────────────────────────
// Mirrors infra/azure/main.tf module invocations, in the same order.

// ── Network layer ─────────────────────────────────────────────────────────────

module network 'modules/network.bicep' = {
  name: 'network'
  params: {
    prefix:        prefix
    location:      location
    vnetCidr:      vnetCidr
    aksSubnetCidr: aksSubnetCidr
    tags:          tags
  }
}

// ── AKS cluster ───────────────────────────────────────────────────────────────

module aks 'modules/aks.bicep' = {
  name: 'aks'
  params: {
    prefix:             prefix
    location:           location
    kubernetesVersion:  kubernetesVersion
    subnetId:           network.outputs.aksSubnetId
    authorizedIpRanges: authorizedIpRanges
    systemNodeSku:      systemNodeSku
    systemNodeMin:      systemNodeMin
    systemNodeMax:      systemNodeMax
    tags:               tags
  }
}

// ── GPU node pools — one per gpuSkus entry ────────────────────────────────────
//
// for-each equivalent in Bicep: a loop over the gpuSkus array.
// Each iteration deploys one gpu-nodepool module for the corresponding SKU.
//
// NOTE: Bicep module loops produce an array of module outputs, not a keyed map.
// The root outputs below adapt the array to match the TF gpu_pools map shape.

module gpuNodepools 'modules/gpu-nodepool.bicep' = [for sku in gpuSkus: {
  name: 'gpu-nodepool-${sku.shortName}'
  params: {
    clusterId:    aks.outputs.clusterId
    subnetId:     network.outputs.aksSubnetId
    prefix:       prefix
    poolSuffix:   sku.poolSuffix
    vmSize:       sku.vmSize
    gpuCount:     sku.gpuCount
    skuLabel:     sku.shortName
    gpuMaxNodes:  sku.maxNodes
    osDiskSizeGb: sku.osDiskSizeGb
    tags:         tags
  }
}]

// ── Container registry ────────────────────────────────────────────────────────

module acr 'modules/acr.bicep' = {
  name: 'acr'
  params: {
    prefix:          prefix
    location:        location
    sku:             acrSku
    kubeletObjectId: aks.outputs.kubeletIdentityObjectId
    tags:            tags
  }
}

// ── Storage ───────────────────────────────────────────────────────────────────

module storage 'modules/storage.bicep' = {
  name: 'storage'
  params: {
    storageAccountName:             storageAccountName
    location:                       location
    blobContainerName:              blobContainerName
    filesShareName:                 filesShareName
    filesShareQuotaGb:              filesShareQuotaGb
    kubeletObjectId:                aks.outputs.kubeletIdentityObjectId
    aksSubnetId:                    network.outputs.aksSubnetId
    authorizedIpRanges:             authorizedIpRanges
    filesPremium:                   filesPremium
    filesPremiumStorageAccountName: filesPremiumStorageAccountName
    tags:                           tags
  }
}

// ── Workload identity ─────────────────────────────────────────────────────────

module identity 'modules/identity.bicep' = {
  name: 'identity'
  params: {
    prefix:              prefix
    location:            location
    oidcIssuerUrl:       aks.outputs.oidcIssuerUrl
    namespace:           workloadIdentityNamespace
    serviceAccountName:  workloadIdentityServiceAccount
    artifactContainerId: storage.outputs.blobContainerResourceId
    tags:                tags
  }
}

// ── Monitoring (security baseline — always on, not in TF) ─────────────────────

module monitoring 'modules/monitoring.bicep' = {
  name: 'monitoring'
  params: {
    prefix:           prefix
    location:         location
    aksClusterId:     aks.outputs.clusterId
    acrId:            acr.outputs.acrId
    storageAccountId: storage.outputs.storageAccountId
    tags:             tags
  }
}

// ─── Outputs ──────────────────────────────────────────────────────────────────
// Covers everything infra/azure/outputs.tf exposes.

// ── Cluster ───────────────────────────────────────────────────────────────────

@description('AKS cluster name. Pass to "az aks get-credentials".')
output clusterName string = aks.outputs.clusterName

@description('Full Azure resource ID of the AKS cluster.')
output clusterId string = aks.outputs.clusterId

@description('OIDC issuer URL emitted by the cluster.')
output oidcIssuerUrl string = aks.outputs.oidcIssuerUrl

@description('Name of the auto-managed node resource group (MC_*). Used for quota checks.')
output nodeResourceGroup string = aks.outputs.nodeResourceGroup

// ── Kubelet identity ──────────────────────────────────────────────────────────

@description('Client ID of the AKS kubelet managed identity.')
output kubeletIdentityClientId string = aks.outputs.kubeletIdentityClientId

@description('Object ID of the AKS kubelet managed identity (role-assignment target).')
output kubeletIdentityObjectId string = aks.outputs.kubeletIdentityObjectId

// ── Workload identity ─────────────────────────────────────────────────────────

@description('Client ID of the user-assigned managed identity used by Job pods.')
output workloadIdentityClientId string = identity.outputs.miClientId

@description('Object/principal ID of the workload managed identity.')
output workloadIdentityPrincipalId string = identity.outputs.miPrincipalId

@description('Full Azure resource ID of the workload managed identity.')
output workloadIdentityResourceId string = identity.outputs.miResourceId

// ── Container registry ────────────────────────────────────────────────────────

@description('ACR login server hostname (e.g. prefixacr.azurecr.io).')
output acrLoginServer string = acr.outputs.loginServer

@description('Full Azure resource ID of the container registry.')
output acrId string = acr.outputs.acrId

// ── Storage ───────────────────────────────────────────────────────────────────

@description('Name of the storage account hosting Blob artifacts and Azure Files cache.')
output storageAccountNameOut string = storage.outputs.storageAccountName

@description('Name of the private Blob container (artifact bus).')
output blobContainerNameOut string = storage.outputs.blobContainerName

@description('Name of the Azure Files share (RWX HF_HOME / pip cache).')
output filesShareNameOut string = storage.outputs.filesShareName

@description('Name of the storage account hosting the active Azure Files share. When filesPremium=false: same as storageAccountName. When filesPremium=true: the dedicated Premium FileStorage account.')
output filesStorageAccountName string = storage.outputs.filesStorageAccountName

// ── Network ───────────────────────────────────────────────────────────────────

@description('Resource ID of the VNet.')
output vnetId string = network.outputs.vnetId

@description('Resource ID of the AKS subnet.')
output aksSubnetId string = network.outputs.aksSubnetId

// ── GPU node pools ────────────────────────────────────────────────────────────
//
// gpu_pools equivalent: array of { name, skuLabel } objects (one per gpuSkus entry).
// The orchestrator indexes by shortName; use the index-keyed array pattern or
// reconstruct the map client-side from the parallel gpuSkus param array.
//
// gpu_nodepool_name (legacy): first pool name, matching TF's values(module.gpu_nodepool)[0].pool_name.

@description('Array of provisioned GPU pools. Each entry: { name: "<aks pool name>", skuLabel: "<reprolab/sku value>" }. Parallel to the gpuSkus input array (index 0 = gpuSkus[0], etc.).')
output gpuPools array = [for (sku, i) in gpuSkus: {
  name:     gpuNodepools[i].outputs.poolName
  skuLabel: gpuNodepools[i].outputs.skuLabelOut
}]

@description('DEPRECATED. Name of the first GPU node pool. Use gpuPools[0].name instead.')
output gpuNodepoolName string = gpuNodepools[0].outputs.poolName

@description('DEPRECATED. Node selector label key for GPU pools. Always "reprolab/sku".')
output gpuNodePoolLabelKey string = 'reprolab/sku'

@description('DEPRECATED. SKU label of the first GPU pool. Use gpuPools[0].skuLabel instead.')
output gpuNodePoolLabelValue string = gpuNodepools[0].outputs.skuLabelOut

@description('Taint key on all GPU nodes (value "present", effect NoSchedule).')
output gpuTaintKey string = 'nvidia.com/gpu'

// ── Monitoring ────────────────────────────────────────────────────────────────

@description('Resource ID of the Log Analytics workspace (new — not in TF).')
output logAnalyticsWorkspaceId string = monitoring.outputs.workspaceId
