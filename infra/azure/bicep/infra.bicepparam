// ──────────────────────────────────────────────────────────────────────────────
// L1 infrastructure parameters — AIONIC environment
//
// SAFE TO COMMIT: contains only resource names, CIDRs, and region — no secrets.
// Read by .github/workflows/infra-deploy.yml (gated deploy job).
//
// Usage:
//   az stack group create \
//     --name openresearch-l1 \
//     --resource-group rg-sciartgen-external \
//     --template-file infra.bicep \
//     --parameters infra.bicepparam \
//     --deny-settings-mode none \
//     --action-on-unmanage detachAll
//
// ──────────────────────────────────────────────────────────────────────────────

using 'infra.bicep'

// Short alphanumeric prefix prepended to every resource name (≤8 chars, lowercase).
param prefix = 'sciart'

// Azure region for all resources.
// westus3 has Standard_NC24ads_A100_v4 quota available.
param location = 'westus3'

// Tags applied to every resource.
param tags = {
  project:     'reprolab'
  environment: 'production'
  client:      'aionic'
  managedBy:   'bicep-l1'
}

// ─── Networking ───────────────────────────────────────────────────────────────

// VNet address space.
param vnetCidr = '10.0.0.0/16'

// AKS subnet CIDR. Must be within vnetCidr.
param aksSubnetCidr = '10.0.0.0/22'

// Operator egress CIDR(s) permitted to reach the AKS public API server.
// fill: az rest --method get --url https://ipinfo.io/ip  → then use <ip>/32
param authorizedIpRanges = ['<OPERATOR_EGRESS_CIDR>']

// ─── AKS cluster ──────────────────────────────────────────────────────────────

// pin: az aks get-versions --location westus3 --query 'values[].patchVersions' -o table
// 1.34 is GA in westus3 (1.31–1.35 available 2026-06); one minor behind latest for stability.
param kubernetesVersion = '1.34'

// VM SKU for the system (CPU) node pool.
param systemNodeSku = 'Standard_D4s_v5'

// System pool autoscaler bounds.
param systemNodeMin = 1
param systemNodeMax = 3

// ─── GPU node pools ───────────────────────────────────────────────────────────
//
// Single A100-80 pool. Start with maxNodes=1 until quota is approved.
// Pool name formula: "<prefix><poolSuffix>" ≤12 lowercase-alnum chars.
// sciart (6) + a10080 (6) = sciarta10080 (12 chars) ✓

param gpuSkus = [
  {
    shortName:    'azure_a100_80'
    vmSize:       'Standard_NC24ads_A100_v4'
    gpuCount:     1
    poolSuffix:   'a10080'
    maxNodes:     1     // Start at 1; increase after quota approval.
    osDiskSizeGb: 256
  }
  // Uncomment to add additional pools (each requires separate quota):
  // {
  //   shortName:    'azure_a100_80x2'
  //   vmSize:       'Standard_NC48ads_A100_v4'
  //   gpuCount:     2
  //   poolSuffix:   'a100x2'
  //   maxNodes:     2
  //   osDiskSizeGb: 256
  // }
]

// ─── Container registry ───────────────────────────────────────────────────────

// Standard is sufficient for a single-region production deployment.
param acrSku = 'Standard'

// ─── Storage ──────────────────────────────────────────────────────────────────

// Globally unique storage account name (3-24 lowercase alphanum, no hyphens).
// NOTE: must be globally unique across all Azure subscriptions.
// If 'sciartgenreprolab' is taken, operator may append digits (e.g. sciartgenreprolab2).
param storageAccountName = 'sciartgenreprolab'

// Blob container name (artifact bus).
param blobContainerName = 'reprolab-artifacts'

// Azure Files share name (RWX HF_HOME / pip-cache mount).
param filesShareName = 'reprolab-cache'

// Files share quota in GiB. 512 GiB covers a full SDAR run.
param filesShareQuotaGb = 512

// false = Standard FileStorage (~$10/month for 512 GiB).
// true  = Premium FileStorage (~$52/month, ~100k IOPS) — set for ≥8 concurrent cells.
param filesPremium = false

// Required when filesPremium = true. Leave empty for Standard (the default).
param filesPremiumStorageAccountName = ''

// ─── Workload identity ────────────────────────────────────────────────────────

// Must match Helm L2 values.yaml → workloadIdentity.namespace.
param workloadIdentityNamespace = 'reprolab'

// Must match Helm L2 values.yaml → workloadIdentity.serviceAccountName.
param workloadIdentityServiceAccount = 'reprolab-sa'
