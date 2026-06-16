// ─── GPU node pool module — User-mode agent pool, scale-to-zero ───────────────
//
// Ports infra/azure/modules/gpu_nodepool/ (main.tf + variables.tf + outputs.tf).
// Every setting the Terraform module sets is replicated faithfully here.
//
// One instance of this module is created per GPU SKU entry (loop in infra.bicep).
// Scale-to-zero: minCount = 0. When no GPU Jobs are pending the pool drains to
// zero nodes. Azure Cluster Autoscaler scales 0→N as Jobs request nvidia.com/gpu.
//
// Label contract (consumed by the orchestrator's k8s_job_cell_runner):
//   reprolab/sku               = <catalog short_name>   e.g. "azure_a100_80"
//   nvidia.com/gpu             = <gpu_count>            e.g. "1"
//   reprolab/node-type         = "gpu"
//   kubernetes.azure.com/scalesetpriority = "regular"
//
// Taint: nvidia.com/gpu=present:NoSchedule (same key on every pool).

targetScope = 'resourceGroup'

// ─── Parameters ───────────────────────────────────────────────────────────────

@description('Full resource ID of the AKS cluster that owns this node pool.')
param clusterId string

@description('Resource ID of the subnet where GPU nodes are placed.')
param subnetId string

@description('Resource name prefix. Pool name is derived as "<prefix><poolSuffix>".')
param prefix string

@description('Short suffix appended to <prefix> to form the AKS node pool name (≤12 lowercase-alnum chars total).')
param poolSuffix string

@description('Azure VM SKU for this GPU node pool (e.g. "Standard_NC24ads_A100_v4").')
param vmSize string = 'Standard_NC24ads_A100_v4'

@description('Number of GPUs per node. Written to the "nvidia.com/gpu" node label.')
param gpuCount int

@description('Catalog short_name written to the "reprolab/sku" node label (e.g. "azure_a100_80").')
param skuLabel string

@description('Maximum number of GPU nodes in this pool (min is always 0 — scale-to-zero).')
param gpuMaxNodes int = 4

@description('OS disk size in GiB for nodes in this GPU pool.')
param osDiskSizeGb int = 256

// NOTE: AgentPool resources (Microsoft.ContainerService/managedClusters/agentPools)
// do not support a top-level tags property in the ARM API. The Terraform provider
// propagates tags via the cluster-level tags blob; in Bicep the param is accepted
// but intentionally not applied (no ARM property exists to set it on a child pool).
// MIGRATION.md documents this as a parity note.
#disable-next-line no-unused-params
param tags object = {}

// ─── GPU node pool ────────────────────────────────────────────────────────────
// Matches: azurerm_kubernetes_cluster_node_pool.gpu
//   name = "${var.prefix}${var.pool_suffix}"
//   mode = "User", enable_auto_scaling = true, min_count = 0, max_count = var.gpu_max_nodes
//   priority = "Regular", eviction_policy = null (Regular nodes have no eviction policy)
//   os_disk_size_gb = var.os_disk_size_gb
//   node_taints = ["nvidia.com/gpu=present:NoSchedule"]
//   node_labels = { reprolab/sku, reprolab/node-type, nvidia.com/gpu, kubernetes.azure.com/scalesetpriority }

resource gpuNodePool 'Microsoft.ContainerService/managedClusters/agentPools@2024-02-01' = {
  // clusterId is a full resource ID; extract just the cluster resource name via split.
  // The parent reference uses the symbolic resource name approach with existing.
  name: '${split(clusterId, '/')[8]}/${prefix}${poolSuffix}'
  properties: {
    vmSize:            vmSize
    vnetSubnetID:      subnetId
    mode:              'User'
    // Scale-to-zero: min = 0
    enableAutoScaling: true
    minCount:          0
    maxCount:          gpuMaxNodes
    // On-demand only (priority = "Regular"). Spot is deferred to Phase 2.
    // eviction_policy is only applicable for Spot pools; omitted for Regular.
    scaleSetPriority:  'Regular'
    osDiskSizeGB:      osDiskSizeGb
    type:              'VirtualMachineScaleSets'
    // GPU taint — shared across ALL GPU pools.
    // device-plugin DaemonSet tolerates with operator: Exists.
    nodeTaints: [
      'nvidia.com/gpu=present:NoSchedule'
    ]
    nodeLabels: {
      'reprolab/sku':                          skuLabel
      'reprolab/node-type':                    'gpu'
      'nvidia.com/gpu':                        string(gpuCount)
      'kubernetes.azure.com/scalesetpriority': 'regular'
    }
  }
}

// ─── Outputs ──────────────────────────────────────────────────────────────────

@description('Name of this GPU node pool (AKS resource name, ≤12 chars).')
output poolName string = last(split(gpuNodePool.name, '/'))

@description('Value of the "reprolab/sku" node label on this pool (catalog short_name).')
output skuLabelOut string = skuLabel

@description('Primary node selector label key for this pool. Always "reprolab/sku".')
output nodeLabelKey string = 'reprolab/sku'

@description('Taint key on GPU nodes. Value is "present", effect is NoSchedule.')
output taintKey string = 'nvidia.com/gpu'
