variable "resource_group_name" {
  description = "Resource group where the managed identity is created."
  type        = string
}

variable "region" {
  description = "Azure region."
  type        = string
}

variable "prefix" {
  description = "Resource name prefix."
  type        = string
}

variable "oidc_issuer_url" {
  description = "OIDC issuer URL of the AKS cluster (azurerm_kubernetes_cluster.oidc_issuer_url). Required to create the federated credential."
  type        = string
}

variable "namespace" {
  description = "Kubernetes namespace where the workload-identity ServiceAccount lives. Must match the namespace in Helm L2."
  type        = string
  default     = "reprolab"
}

variable "service_account_name" {
  description = "Name of the Kubernetes ServiceAccount annotated with this MI's client ID. Must match the ServiceAccount in Helm L2."
  type        = string
  default     = "reprolab-sa"
}

variable "artifact_container_id" {
  description = "Full Azure resource ID of the artifact Blob container (including the '/blobServices/default/containers/<name>' suffix). Used as the scope for the Storage Blob Data Contributor assignment."
  type        = string
}

variable "tags" {
  description = "Map of tags applied to the managed identity."
  type        = map(string)
  default     = {}
}
