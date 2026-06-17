# ─── Workload-identity GSA ────────────────────────────────────────────────────
# The GCP service account that GKE Job pods impersonate via Workload Identity.
# Job pods exchange a projected Kubernetes ServiceAccount token for a short-lived
# GSA access token — NO service-account JSON key is ever created, downloaded, or
# mounted in-cluster.  This is the keyless GCP analog of the Azure user-assigned
# managed identity + federated credential.

resource "google_service_account" "workload" {
  project      = var.project_id
  account_id   = "${var.prefix}-workload"
  display_name = "${var.prefix} GKE workload-identity service account"
}

# ─── storage.objectAdmin → artifact bucket ────────────────────────────────────
# Grants the GSA read + write on the artifact bus bucket ONLY (not project-wide
# storage).  Job pods call ADC (DefaultCredentials) to upload metrics/logs and
# download code — no HMAC keys, no signed URLs with static secrets.

resource "google_storage_bucket_iam_member" "workload_object_admin" {
  bucket = var.bucket_name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.workload.email}"
}

# ─── Workload Identity binding (KSA ↔ GSA) ────────────────────────────────────
# Allows the Kubernetes ServiceAccount
#   <namespace>/<service_account_name>
# in the cluster's workload pool to impersonate this GSA.
#
# The member string is EXACTLY:
#   serviceAccount:<project_id>.svc.id.goog[<namespace>/<ksa-name>]
# and MUST match the KSA created by Helm L2 (namespace + name) and the
# iam.gke.io/gcp-service-account annotation on that KSA.  A mismatch silently
# breaks pod authentication (the GCP analog of the Azure federated-subject trap).

resource "google_service_account_iam_member" "workload_identity_user" {
  service_account_id = google_service_account.workload.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[${var.namespace}/${var.service_account_name}]"
}
