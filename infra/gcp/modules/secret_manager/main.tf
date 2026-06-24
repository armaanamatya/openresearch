# ─── Secret Manager module — GCP analog of Azure keyvault.bicep ───────────────
#
# Creates three Secret Manager secret NAMES that the orchestrator pod reads at
# runtime via the Secrets Store CSI driver (provider: gcp).  Secret VALUES are
# NEVER stored here — the operator adds versions out-of-band:
#
#   gcloud secrets versions add claude-code-oauth-token \
#     --data-file=<(echo -n "$TOKEN")
#   gcloud secrets versions add anthropic-api-key \
#     --data-file=<(echo -n "$ANTHROPIC_API_KEY")
#   gcloud secrets versions add azure-openai-api-key \
#     --data-file=<(echo -n "$AZURE_OPENAI_API_KEY")
#
# Secret names match the Azure Key Vault names exactly so the operational
# contract is identical across clouds (see spec §Operational contract).
#
# The three secrets exist in the same GCP project as the GKE cluster so the
# orchestrator GSA can reach them via Workload Identity without cross-project
# Secret Manager permissions.

resource "google_secret_manager_secret" "claude_code_oauth_token" {
  project   = var.project_id
  secret_id = "claude-code-oauth-token"

  replication {
    user_managed {
      replicas {
        location = var.region
      }
    }
  }

  labels = var.labels
}

resource "google_secret_manager_secret" "anthropic_api_key" {
  project   = var.project_id
  secret_id = "anthropic-api-key"

  replication {
    user_managed {
      replicas {
        location = var.region
      }
    }
  }

  labels = var.labels
}

resource "google_secret_manager_secret" "azure_openai_api_key" {
  project   = var.project_id
  secret_id = "azure-openai-api-key"

  replication {
    user_managed {
      replicas {
        location = var.region
      }
    }
  }

  labels = var.labels
}
