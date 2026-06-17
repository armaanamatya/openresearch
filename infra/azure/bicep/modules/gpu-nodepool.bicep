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
// (No kubernetes.azure.com/* label — that prefix is AKS-reserved.)
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

@description('Azure VM SKU for this GPU node pool. Default Standard_ND96asr_v4 = 8×A100-40GB. Sized so the 7B 8-GPU SDAR cell fits one node; scale-to-zero ⇒ idle=$0; override the param for a different size (e.g. Standard_NC24ads_A100_v4 for 1×A100-80GB).')
param vmSize string = 'Standard_ND96asr_v4'

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

@description('''
Enable Azure Spot VMSS for this GPU node pool (default false = Regular/on-demand).

When true:  scaleSetPriority = Spot, spotMaxPrice = -1 (pay market rate, never
            exceeds on-demand), evictionPolicy = Delete (node disk discarded on
            eviction; run state lives in Blob/Files).
When false: scaleSetPriority = Regular — rendered exactly as before this param
            was added; no spotMaxPrice or evictionPolicy properties emitted.

Pair with the RUNTIME spot flag (config azure_use_spot / env
OPENRESEARCH_AZURE_USE_SPOT=1): it adds the spot toleration to cell Pods and a
>0 backoffLimit so a preempted cell reschedules onto a fresh node. The cell
entrypoint already flushes its checkpoint + partial metrics to Blob on the
preemption SIGTERM (grace window OPENRESEARCH_CELL_PREEMPT_GRACE_S).
''')
param useSpot bool = false

// ─── Spot-only properties (conditionally merged into pool properties) ─────────
// useSpot = false (default): spotProps is empty → the union below is a no-op →
//   the rendered ARM template is byte-identical to the pre-useSpot output.
// useSpot = true: spotProps carries the two Spot-only fields that the ARM API
//   requires for Spot VMSS pools; they MUST NOT be present on Regular pools.
//   spotMaxPrice = -1 means "pay the current Spot price up to the on-demand cap."
//   evictionPolicy = Delete discards the node OS disk on eviction; ephemeral run
//   state must already be in Blob/Azure Files (the reprolab default).
var spotProps = useSpot ? {
  scaleSetPriority: 'Spot'
  spotMaxPrice:     json('-1')
  evictionPolicy:   'Delete'
} : {
  scaleSetPriority: 'Regular'
}

// ─── GPU node pool ────────────────────────────────────────────────────────────
// Matches: azurerm_kubernetes_cluster_node_pool.gpu
//   name = "${var.prefix}${var.pool_suffix}"
//   mode = "User", enable_auto_scaling = true, min_count = 0, max_count = var.gpu_max_nodes
//   priority = "Regular", eviction_policy = null (Regular nodes have no eviction policy)
//   os_disk_size_gb = var.os_disk_size_gb
//   node_taints = ["nvidia.com/gpu=present:NoSchedule"]
//   node_labels = { reprolab/sku, reprolab/node-type, nvidia.com/gpu }

resource gpuNodePool 'Microsoft.ContainerService/managedClusters/agentPools@2024-02-01' = {
  // clusterId is a full resource ID; extract just the cluster resource name via split.
  // The parent reference uses the symbolic resource name approach with existing.
  name: '${split(clusterId, '/')[8]}/${prefix}${poolSuffix}'
  properties: union({
    vmSize:            vmSize
    vnetSubnetID:      subnetId
    mode:              'User'
    // Scale-to-zero: min = 0
    enableAutoScaling: true
    minCount:          0
    maxCount:          gpuMaxNodes
    osDiskSizeGB:      osDiskSizeGb
    type:              'VirtualMachineScaleSets'
    // GPU taint — shared across ALL GPU pools.
    // device-plugin DaemonSet tolerates with operator: Exists.
    nodeTaints: [
      'nvidia.com/gpu=present:NoSchedule'
    ]
    // NOTE: do NOT set any 'kubernetes.azure.com/*' node label here — that prefix
    // is reserved by AKS and the API rejects the pool create
    // (InvalidNodeLabelKey). AKS sets scalesetpriority automatically (and only
    // for Spot pools); a Regular pool simply has no such label.
    nodeLabels: {
      'reprolab/sku':       skuLabel
      'reprolab/node-type': 'gpu'
      'nvidia.com/gpu':     string(gpuCount)
    }
  }, spotProps)
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
