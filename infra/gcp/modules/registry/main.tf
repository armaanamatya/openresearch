# ─── Artifact Registry (Docker) ───────────────────────────────────────────────
#
# Hosts the gke-cell-base image pulled by GKE Job pods.  The repository URL is
#   <region>-docker.pkg.dev/<project_id>/<repository_id>
# Image refs append "/<image-name>:<PINNED-tag>".

resource "google_artifact_registry_repository" "main" {
  project       = var.project_id
  location      = var.region
  repository_id = var.repository_id
  format        = "DOCKER"
  description   = "ReproLab gke-cell-base images for the GKE GPU execution backend."

  labels = var.labels
}

# ─── artifactregistry.reader → GKE node service account ───────────────────────
#
# Lets GKE nodes pull images from this repository using the node's GSA — no
# Docker credentials or imagePullSecrets.  Scoped to THIS repository only
# (least privilege), not project-wide.

resource "google_artifact_registry_repository_iam_member" "node_reader" {
  project    = var.project_id
  location   = google_artifact_registry_repository.main.location
  repository = google_artifact_registry_repository.main.name
  role       = "roles/artifactregistry.reader"
  member     = "serviceAccount:${var.node_sa_email}"
}
