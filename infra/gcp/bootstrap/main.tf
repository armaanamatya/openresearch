terraform {
  required_version = ">= 1.6.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }

  # Bootstrap uses local state intentionally — it creates the remote-state bucket.
  # Keep the resulting terraform.tfstate in a secrets manager or secure storage,
  # NOT in git.
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# ─── GCS bucket for Terraform remote state ────────────────────────────────────
#
# Versioned + uniform access + public-access-prevention, mirroring the root
# bucket's security posture.  The `prefix` in the GCS backend config (backend.hcl)
# separates the deepinvent state object under this bucket.

resource "google_storage_bucket" "tfstate" {
  name     = var.state_bucket_name
  project  = var.project_id
  location = var.region

  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = true
  }

  # Keep a bounded version history of state objects.
  lifecycle_rule {
    condition {
      num_newer_versions = 10
      with_state         = "ARCHIVED"
    }
    action {
      type = "Delete"
    }
  }

  labels = var.labels
}

# ─── Optional CI/deploy service account ───────────────────────────────────────
#
# Created only when create_ci_service_account = true.  Granted the minimal roles
# to run the root `terraform apply`:
#   - roles/container.admin       (create/manage the GKE cluster + node pools)
#   - roles/compute.networkAdmin  (VPC, subnet, router, NAT, firewall)
#   - roles/storage.admin         (the artifact bucket + state bucket access)
#   - roles/iam.serviceAccountAdmin (create the node + workload GSAs)
#   - roles/resourcemanager.projectIamAdmin (grant the project-level node roles)
#   - roles/artifactregistry.admin (create the Docker repository)
#   - roles/file.editor           (create the Filestore instance)
# Keyless: NO service-account JSON key is created — grant a human or CI
# workload-identity-federation principal `roles/iam.serviceAccountTokenCreator`
# on this SA to impersonate it instead.

resource "google_service_account" "ci" {
  count        = var.create_ci_service_account ? 1 : 0
  project      = var.project_id
  account_id   = var.ci_service_account_id
  display_name = "ReproLab GKE CI/deploy service account"
}

locals {
  ci_roles = var.create_ci_service_account ? toset([
    "roles/container.admin",
    "roles/compute.networkAdmin",
    "roles/storage.admin",
    "roles/iam.serviceAccountAdmin",
    "roles/resourcemanager.projectIamAdmin",
    "roles/artifactregistry.admin",
    "roles/file.editor",
  ]) : toset([])
}

resource "google_project_iam_member" "ci" {
  for_each = local.ci_roles
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.ci[0].email}"
}
