# ─── VPC network (custom subnet mode) ─────────────────────────────────────────

resource "google_compute_network" "main" {
  name                    = "${var.prefix}-vpc"
  project                 = var.project_id
  auto_create_subnetworks = false
  routing_mode            = "REGIONAL"
}

# ─── GKE node subnet (VPC-native, two secondary ranges) ───────────────────────
#
# The primary range carries node IPs; the two secondary ranges carry pod IPs
# and ClusterIP service IPs respectively.  GKE's ip_allocation_policy references
# these secondary ranges BY NAME (see modules/gke).

resource "google_compute_subnetwork" "main" {
  name          = "${var.prefix}-subnet"
  project       = var.project_id
  region        = var.region
  network       = google_compute_network.main.id
  ip_cidr_range = var.subnet_cidr

  # Required for private nodes to reach Google APIs without a public IP.
  private_ip_google_access = true

  secondary_ip_range {
    range_name    = "${var.prefix}-pods"
    ip_cidr_range = var.pods_secondary_cidr
  }

  secondary_ip_range {
    range_name    = "${var.prefix}-services"
    ip_cidr_range = var.services_secondary_cidr
  }
}

# ─── Cloud Router + NAT — egress for private nodes ────────────────────────────
#
# Private GKE nodes have no external IP.  Cloud NAT provides outbound internet
# (pip install, HuggingFace weight downloads, base-image pulls) without exposing
# any node to inbound traffic.

resource "google_compute_router" "main" {
  name    = "${var.prefix}-router"
  project = var.project_id
  region  = var.region
  network = google_compute_network.main.id
}

resource "google_compute_router_nat" "main" {
  name                               = "${var.prefix}-nat"
  project                            = var.project_id
  region                             = var.region
  router                             = google_compute_router.main.name
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}

# ─── Firewall — allow intra-VPC traffic ───────────────────────────────────────
#
# GKE manages pod-to-pod rules internally; this rule permits node↔node and
# node↔control-plane traffic on the primary + secondary ranges.  There is NO
# inbound-from-internet rule — the implicit default-deny covers external ingress.

resource "google_compute_firewall" "allow_internal" {
  name    = "${var.prefix}-allow-internal"
  project = var.project_id
  network = google_compute_network.main.name

  allow {
    protocol = "tcp"
  }
  allow {
    protocol = "udp"
  }
  allow {
    protocol = "icmp"
  }

  direction = "INGRESS"
  source_ranges = [
    var.subnet_cidr,
    var.pods_secondary_cidr,
    var.services_secondary_cidr,
  ]
}
