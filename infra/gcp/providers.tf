# Both the GA `google` provider and the `google-beta` provider are configured
# from the same project + region.  google-beta is required by a handful of GKE
# attributes (e.g. node-pool GPU driver auto-install) that are still beta in the
# GA provider; modules that need it set `provider = google-beta` explicitly.

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}
