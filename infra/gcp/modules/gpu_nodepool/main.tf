# GPU node pool — parameterized SKU, scale-to-zero
#
# One instance of this module is created per entry in var.gpu_skus (root).
# Scale-to-zero: min_node_count = 0.  When no GPU Jobs are pending the pool
# drains to zero nodes and idle cost is $0.  The GKE cluster autoscaler scales
# 0→N as Jobs request nvidia.com/gpu resources.
#
# Label contract (consumed by the orchestrator's k8s job cell runner):
#   reprolab/sku       = <catalog short_name>   e.g. "gcp_a100_80"
#   reprolab/gpu-count = <gpu_count>            e.g. "1"
#   reprolab/node-type = "gpu"
#
# Job nodeSelector pattern:
#   nodeSelector:
#     reprolab/sku: gcp_a100_80
#
# Taint (NoSchedule):
#   nvidia.com/gpu=present:NoSchedule
#   The orchestrator's Job pods tolerate this with operator: Exists.
#
# Driver + device plugin:
#   gpu_driver_installation_config.gpu_driver_version = "DEFAULT" tells GKE to
#   auto-install the matching NVIDIA driver AND run the managed device plugin —
#   no hand-rolled DaemonSet is required (this is the key GKE divergence from the
#   Azure mirror, which ships an nvidia-device-plugin.yaml).

resource "google_container_node_pool" "gpu" {
  provider = google-beta

  # GKE node-pool names: lowercase, ≤40 chars, RFC-1035.  The catalog short_name
  # uses underscores (invalid in a pool name), so we sanitize to hyphens.
  name     = "${var.prefix}-${replace(var.short_name, "_", "-")}"
  project  = var.project_id
  location = var.location
  cluster  = var.cluster_name

  # ── Scale-to-zero ───────────────────────────────────────────────────────────
  autoscaling {
    min_node_count = 0
    max_node_count = var.max_nodes
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    machine_type    = var.machine_type
    disk_size_gb    = var.disk_size_gb
    service_account = var.service_account
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]

    # ── On-demand only ────────────────────────────────────────────────────────
    # Spot/preemptible nodes reduce cost but can be reclaimed mid-training;
    # deferred to a later phase with in-Job checkpoint/resume support.
    spot        = false
    preemptible = false

    # ── GPU accelerator + auto driver install ─────────────────────────────────
    guest_accelerator {
      type  = var.accelerator_type
      count = var.gpu_count

      gpu_driver_installation_config {
        gpu_driver_version = "DEFAULT"
      }
    }

    # ── Node labels ───────────────────────────────────────────────────────────
    # reprolab/sku is the PRIMARY selector key for Job placement.  The
    # orchestrator resolves plan.short_name → nodeSelector { reprolab/sku: <sku> }.
    labels = merge(var.labels, {
      "reprolab/sku"       = var.short_name
      "reprolab/node-type" = "gpu"
      "reprolab/gpu-count" = tostring(var.gpu_count)
    })

    # ── GPU taint ─────────────────────────────────────────────────────────────
    # Shared taint key across ALL GPU pools — Job pods tolerate it with
    # operator: Exists (matches any value).  Prevents non-GPU workloads from
    # scheduling on any GPU pool node.
    taint {
      key    = "nvidia.com/gpu"
      value  = "present"
      effect = "NO_SCHEDULE"
    }

    # Workload Identity metadata server (so WI-annotated Job pods can mint GSA
    # tokens to reach GCS).
    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
  }
}
