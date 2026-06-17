variable "project_id" {
  description = "GCP project ID that owns this node pool."
  type        = string
}

variable "cluster_name" {
  description = "Name of the GKE cluster that owns this node pool."
  type        = string
}

variable "location" {
  description = "Location of the cluster (region for a regional cluster)."
  type        = string
}

variable "prefix" {
  description = "Resource name prefix. Pool name is derived as '<prefix>-<short_name-with-hyphens>'."
  type        = string
}

variable "short_name" {
  description = "Catalog short_name written to the 'reprolab/sku' node label (e.g. 'gcp_a100_80'). Used by Job nodeSelector to target this exact pool. Underscores are sanitized to hyphens in the pool name."
  type        = string
}

variable "machine_type" {
  # Sized so the 7B 8-GPU SDAR cell fits one node; scale-to-zero ⇒ idle=$0;
  # override the var for a different size (e.g. a2-highgpu-1g for single-GPU).
  # Default: a2-highgpu-8g = 8×A100-40GB (catalog short_name gcp_a100_40x8).
  # a2-ultragpu-8g (8×A100-80GB) is available for override when 80 GB/GPU needed.
  description = "GCE A2 machine type for this GPU node pool (e.g. 'a2-highgpu-8g'). Default 8×A100-40 GB fits the 7B SDAR cell on one node."
  type        = string
  default     = "a2-highgpu-8g"
}

variable "accelerator_type" {
  description = "GCE GPU accelerator type ('nvidia-a100-80gb' or 'nvidia-tesla-a100')."
  type        = string
}

variable "gpu_count" {
  description = "Number of GPUs per node. Written to the 'reprolab/gpu-count' node label and used to request nvidia.com/gpu resources. Must match the GPU count implied by machine_type (e.g. a2-ultragpu-1g = 1)."
  type        = number
}

variable "max_nodes" {
  description = "Maximum number of GPU nodes in this pool (min is always 0 — scale-to-zero). Required per-region GPU quota ≈ gpu_count × max_nodes in the matching A100 family."
  type        = number
  default     = 4
}

variable "disk_size_gb" {
  description = "Boot disk size in GiB for nodes in this GPU pool. Default 256 GiB is sufficient for the gke-cell-base image + working dir; raise to 512 for SKUs that pull large images or have large local pip/HF cache overflow. Sourced from gpu_skus[].disk_size_gb."
  type        = number
  default     = 256
}

variable "service_account" {
  description = "Email of the node service account (the dedicated least-privilege GSA from the gke module)."
  type        = string
}

variable "labels" {
  description = "Map of labels applied to the node pool nodes."
  type        = map(string)
  default     = {}
}

variable "use_spot" {
  description = <<-EOT
    Enable GKE Spot nodes for this pool (default false = on-demand).

    Spot nodes are reclaimed by GCP with a ~15-30 s TERMINATING window.
    GKE automatically taints Spot nodes with:
      cloud.google.com/gke-spot=true:NoSchedule
    Cell Job pods opt in via the RUNTIME spot flag (config gcp_use_spot / env
    OPENRESEARCH_GCP_USE_SPOT=1), which adds the matching toleration and a >0
    backoffLimit so a preempted cell reschedules.  The existing nvidia.com/gpu
    taint is unaffected.  The cell entrypoint flushes its checkpoint to GCS on
    the preemption SIGTERM (grace window OPENRESEARCH_CELL_PREEMPT_GRACE_S).
  EOT
  type    = bool
  default = false
}
