# Remote state is stored in the GCS bucket provisioned by the bootstrap module.
# Populate the partial backend configuration from envs/deepinvent/backend.hcl.example
# before running terraform init:
#
#   terraform init -backend-config=envs/deepinvent/backend.hcl
#
terraform {
  backend "gcs" {}
}
