# GPU node pool — parameterized SKU, scale-to-zero
#
# One instance of this module is created per entry in var.gpu_skus (root).
# Scale-to-zero: min_count = 0. When no GPU Jobs are pending the pool drains
# to zero nodes and idle cost is $0. Azure Cluster Autoscaler scales 0→N as
# Jobs request nvidia.com/gpu resources.
#
# Label contract (consumed by the orchestrator's k8s_job_cell_runner):
#   reprolab/sku    = <catalog short_name>   e.g. "azure_a100_80"
#   nvidia.com/gpu  = <gpu_count>            e.g. "1"
#   reprolab/node-type = "gpu"               (unchanged — for the device plugin)
#
# Job nodeSelector pattern:
#   nodeSelector:
#     reprolab/sku:   azure_a100_80
#     nvidia.com/gpu: "1"
#
# Taint (NoSchedule):
#   nvidia.com/gpu=present:NoSchedule        (same key on every pool)
#   Helm device-plugin DaemonSet tolerates this with operator: Exists.

resource "azurerm_kubernetes_cluster_node_pool" "gpu" {
  # AKS node pool names: lowercase, ≤12 chars, alphanumeric only.
  # Name is constructed as "<prefix><pool_suffix>".  The root derives pool_suffix
  # from the catalog short_name (see variable description for the mapping).
  name = "${var.prefix}${var.pool_suffix}"

  kubernetes_cluster_id = var.cluster_id
  vnet_subnet_id        = var.subnet_id
  vm_size               = var.vm_size
  mode                  = "User"

  # ── Scale-to-zero ─────────────────────────────────────────────────────────
  enable_auto_scaling = true
  min_count           = 0
  max_count           = var.gpu_max_nodes

  # ── On-demand only ─────────────────────────────────────────────────────────
  # Spot nodes reduce cost but can be evicted mid-training; deferred to Phase 2
  # with in-Job checkpoint/resume support.
  priority        = "Regular"
  eviction_policy = null

  os_disk_size_gb = 256

  # ── GPU taint ─────────────────────────────────────────────────────────────
  # Shared taint key across ALL GPU pools — the device-plugin DaemonSet
  # tolerates it with operator: Exists (matches any value).
  # Prevents non-GPU workloads from scheduling on any GPU pool node.
  node_taints = ["nvidia.com/gpu=present:NoSchedule"]

  # ── Node labels ───────────────────────────────────────────────────────────
  # reprolab/sku is the PRIMARY selector key for Job placement.
  # The orchestrator resolves plan.short_name → nodeSelector { reprolab/sku: <sku> }.
  node_labels = {
    "reprolab/sku"                           = var.sku_label
    "reprolab/node-type"                     = "gpu"
    "nvidia.com/gpu"                         = tostring(var.gpu_count)
    "kubernetes.azure.com/scalesetpriority"  = "regular"
  }

  tags = var.tags
}
