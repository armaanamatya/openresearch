# ─── Node service account (least-privilege) ───────────────────────────────────
#
# GKE's default node identity is the project's default Compute Engine service
# account, which is over-privileged.  We create a dedicated GSA for the nodes
# with only the roles a kubelet needs: logging, monitoring, metric-writing, and
# (granted in the registry module) Artifact Registry read.  This is the node
# identity — distinct from the WORKLOAD-identity GSA used by Job pods.

resource "google_service_account" "node" {
  project      = var.project_id
  account_id   = "${var.prefix}-gke-node"
  display_name = "${var.prefix} GKE node service account"
}

resource "google_project_iam_member" "node_log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.node.email}"
}

resource "google_project_iam_member" "node_metric_writer" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.node.email}"
}

resource "google_project_iam_member" "node_monitoring_viewer" {
  project = var.project_id
  role    = "roles/monitoring.viewer"
  member  = "serviceAccount:${google_service_account.node.email}"
}

# ─── GKE cluster ──────────────────────────────────────────────────────────────
#
# VPC-native, Workload-Identity-enabled, private nodes, IP-restricted public
# control-plane endpoint.  The default node pool is removed immediately — every
# real pool (system + GPU) is a separately-managed google_container_node_pool.

resource "google_container_cluster" "main" {
  provider = google-beta

  name     = "${var.prefix}-gke"
  project  = var.project_id
  location = var.region

  network    = var.network_self_link
  subnetwork = var.subnet_self_link

  # ── Remove the default node pool ────────────────────────────────────────────
  # We manage every pool explicitly (system + GPU) so we can pin machine types,
  # autoscaling, taints, and the node SA per pool.
  remove_default_node_pool = true
  initial_node_count       = 1

  # ── Release channel + version ───────────────────────────────────────────────
  release_channel {
    channel = var.release_channel
  }
  # Pin the master version only when explicitly supplied; otherwise let the
  # release channel pick the default (empty string = "channel default").
  min_master_version = var.kubernetes_version != "" ? var.kubernetes_version : null

  # ── VPC-native (alias IP) ───────────────────────────────────────────────────
  networking_mode = "VPC_NATIVE"
  ip_allocation_policy {
    cluster_secondary_range_name  = var.pods_secondary_range_name
    services_secondary_range_name = var.services_secondary_range_name
  }

  # ── Workload Identity ───────────────────────────────────────────────────────
  # Pods bound to an annotated KSA exchange a projected token for GSA creds.
  # No node-level key material is ever mounted into Job pods.
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # ── Private cluster ─────────────────────────────────────────────────────────
  # Nodes have no public IP (egress via Cloud NAT).  The control-plane endpoint
  # is public but IP-restricted (master_authorized_networks below) — the GCP
  # analog of the Azure "public API server + authorized IP ranges" posture, kept
  # because the orchestrator runs locally outside the VPC in Phase 1.
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = var.master_ipv4_cidr_block
  }

  master_authorized_networks_config {
    dynamic "cidr_blocks" {
      for_each = var.authorized_ip_ranges
      content {
        cidr_block   = cidr_blocks.value
        display_name = "operator-${cidr_blocks.key}"
      }
    }
  }

  # ── Hardening ───────────────────────────────────────────────────────────────
  # Disable the legacy client-cert/basic-auth attestation surface; auth is
  # exclusively via the GKE gcloud auth plugin (IAM-mapped identities).
  master_auth {
    client_certificate_config {
      issue_client_certificate = false
    }
  }

  # Shielded nodes (secure boot + integrity monitoring) on every node pool.
  enable_shielded_nodes = true

  # ── Observability ───────────────────────────────────────────────────────────
  # Ship system + workload logs to Cloud Logging and system metrics to Cloud
  # Monitoring, with Google Managed Prometheus collection enabled.
  logging_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS"]
  }
  monitoring_config {
    enable_components = ["SYSTEM_COMPONENTS"]
    managed_prometheus {
      enabled = true
    }
  }

  resource_labels = var.labels

  # GKE requires a deletion-protection acknowledgement; default true keeps the
  # cluster from being destroyed by an accidental `terraform destroy`.
  deletion_protection = true

  lifecycle {
    # The default node-pool block churns after remove_default_node_pool; ignore
    # node_config drift on the cluster object itself.
    ignore_changes = [node_config]
  }
}

# ─── System (CPU) node pool ───────────────────────────────────────────────────

resource "google_container_node_pool" "system" {
  provider = google-beta

  name     = "system"
  project  = var.project_id
  location = var.region
  cluster  = google_container_cluster.main.name

  autoscaling {
    min_node_count = var.system_node_min_count
    max_node_count = var.system_node_max_count
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    machine_type    = var.system_node_machine_type
    disk_size_gb    = 128
    service_account = google_service_account.node.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]

    labels = merge(var.labels, {
      "reprolab/node-type" = "system"
    })

    # Workload Identity metadata server on system nodes (required so any
    # WI-annotated pod scheduled here can mint GSA tokens).
    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
  }
}
