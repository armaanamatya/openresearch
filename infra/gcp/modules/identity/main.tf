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

# ─── Orchestrator GSA ─────────────────────────────────────────────────────────
# A SEPARATE identity from the training-cell GSA (workload above) so Secret
# Manager access is auditable independently.  Mirrors the Azure orchestrator
# user-assigned MI + Key Vault Secrets User role assignment.
#
# The orchestrator runs the RLM root loop (CPU-only) and:
#   • Reads API keys from Secret Manager (secretmanager.secretAccessor)
#   • Reads and writes run artifacts to the GCS bucket (storage.objectAdmin)
#   • Is impersonated by the 'reprolab-orchestrator' KSA in the cluster
#     (via Workload Identity, same keyless mechanism as the training GSA)

resource "google_service_account" "orchestrator" {
  count        = var.secret_manager_module_enabled ? 1 : 0
  project      = var.project_id
  account_id   = "${var.prefix}-orchestrator"
  display_name = "${var.prefix} GKE orchestrator service account"
}

# ─── secretmanager.secretAccessor → three orchestrator secrets ─────────────────
# Read-only access to secret VALUES — no list, no write, no create.
# Scoped to individual secrets (least-privilege; not project-wide accessor).

resource "google_secret_manager_secret_iam_member" "orchestrator_oauth_token" {
  count     = var.secret_manager_module_enabled ? 1 : 0
  project   = var.project_id
  secret_id = var.claude_code_oauth_token_secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.orchestrator[0].email}"
}

resource "google_secret_manager_secret_iam_member" "orchestrator_anthropic_key" {
  count     = var.secret_manager_module_enabled ? 1 : 0
  project   = var.project_id
  secret_id = var.anthropic_api_key_secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.orchestrator[0].email}"
}

resource "google_secret_manager_secret_iam_member" "orchestrator_azure_openai_key" {
  count     = var.secret_manager_module_enabled ? 1 : 0
  project   = var.project_id
  secret_id = var.azure_openai_api_key_secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.orchestrator[0].email}"
}

# ─── storage.objectAdmin → artifact bucket (orchestrator) ─────────────────────
# The orchestrator reads code from the bucket (sends it into GKE cell Jobs) and
# reads back metrics/logs produced by those Jobs.

resource "google_storage_bucket_iam_member" "orchestrator_object_admin" {
  count  = var.secret_manager_module_enabled ? 1 : 0
  bucket = var.bucket_name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.orchestrator[0].email}"
}

# ─── Workload Identity binding (reprolab-orchestrator KSA ↔ orchestrator GSA) ─

resource "google_service_account_iam_member" "orchestrator_workload_identity_user" {
  count              = var.secret_manager_module_enabled ? 1 : 0
  service_account_id = google_service_account.orchestrator[0].name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[${var.namespace}/reprolab-orchestrator]"
}
