# ─── Artifact bus GCS bucket ──────────────────────────────────────────────────
#
# Private bucket — the bus between the local orchestrator and GKE Job pods.
# Layout: runs/<run_id>/code/**            (uploaded by orchestrator)
#         runs/<run_id>/cells/<cell_id>/{metrics.json,status.json,logs/**}
#
# Security posture (matches the zero-public-access policy of the Azure mirror):
#   - uniform_bucket_level_access = true  → NO per-object ACLs; IAM only.
#   - public_access_prevention   = enforced → allUsers / allAuthenticatedUsers
#     can NEVER be granted, even by a later accidental IAM edit.
#   - NO storage account key / HMAC key is ever created or exported.  The
#     orchestrator authenticates with Application Default Credentials and Job
#     pods with Workload Identity (objectAdmin granted in the identity module).

resource "google_storage_bucket" "artifacts" {
  name     = var.bucket_name
  project  = var.project_id
  location = var.region

  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  # Versioning protects against accidental overwrite of run artifacts.
  versioning {
    enabled = true
  }

  # Lifecycle: expire noncurrent (overwritten) versions after 30 days so the
  # versioned bucket does not grow unbounded.
  lifecycle_rule {
    condition {
      num_newer_versions = 3
      with_state         = "ARCHIVED"
    }
    action {
      type = "Delete"
    }
  }

  labels = var.labels
}

# ─── Filestore RWX cache (conditional — filestore_enabled = true only) ─────────
#
# The GCP analog of the Azure Files Premium share: an NFS file share mounted by
# every Job pod as the shared HF_HOME / pip cache so model weights download once
# per cluster lifetime.  Created only when filestore_enabled = true; when false
# no Filestore exists and Jobs fall back to an emptyDir / GCS-only cache.
#
# BASIC_HDD enforces a 1 TiB (1024 GiB) minimum capacity.  ZONAL / ENTERPRISE
# tiers give higher IOPS at higher cost + higher minimum capacity.

resource "google_filestore_instance" "cache" {
  count = var.filestore_enabled ? 1 : 0

  name     = "${var.bucket_name}-cache"
  project  = var.project_id
  location = "${var.region}-a"
  tier     = var.filestore_tier

  file_shares {
    name        = var.filestore_share_name
    capacity_gb = var.filestore_capacity_gb
  }

  networks {
    network = var.network_self_link
    modes   = ["MODE_IPV4"]
  }

  labels = var.labels
}
