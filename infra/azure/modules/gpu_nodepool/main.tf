# GPU node pool — Standard_NC24ads_A100_v4 (1 × A100-80GB per node, 24 vCPUs)
#
# Scale-to-zero: min_count = 0. When no GPU Jobs are pending the pool drains
# to zero nodes and idle cost is $0. Azure Cluster Autoscaler scales 0→N as
# Jobs request nvidia.com/gpu resources.
#
# Each pod requests exactly 1 × nvidia.com/gpu; the node taint
# (nvidia.com/gpu=present:NoSchedule) prevents non-GPU workloads from landing
# here. Helm L2 adds the matching toleration to Job pod templates.

resource "azurerm_kubernetes_cluster_node_pool" "gpu" {
  # AKS node pool names: lowercase, ≤12 chars, alphanumeric only
  name = "${var.prefix}gpu"

  kubernetes_cluster_id = var.cluster_id
  vnet_subnet_id        = var.subnet_id
  vm_size               = "Standard_NC24ads_A100_v4"
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
  # Prevents non-GPU workloads from scheduling here.
  node_taints = ["nvidia.com/gpu=present:NoSchedule"]

  # ── Node labels ───────────────────────────────────────────────────────────
  node_labels = {
    "reprolab/node-type"          = "gpu"
    "reprolab/gpu-sku"            = "a100-80gb"
    "accelerator"                 = "nvidia-a100"
    "kubernetes.azure.com/scalesetpriority" = "regular"
  }

  tags = var.tags
}
