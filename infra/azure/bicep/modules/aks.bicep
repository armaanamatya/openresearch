// ─── AKS module — managed-identity AKS cluster ───────────────────────────────
//
// Ports infra/azure/modules/aks/ (main.tf + variables.tf + outputs.tf).
// Every setting the Terraform module sets is replicated faithfully here.
//
// TF sets both local_account_disabled = true AND azure_rbac_enabled = true.
// Both are included below (not hardening deltas — already in TF).

targetScope = 'resourceGroup'

// ─── Parameters ───────────────────────────────────────────────────────────────

@description('Short alphanumeric prefix prepended to every resource name.')
param prefix string

@description('Azure region for the AKS cluster.')
param location string = resourceGroup().location

@description('Kubernetes version (e.g. "1.29"). Must be available in the region.')
param kubernetesVersion string

@description('Resource ID of the subnet where AKS nodes are placed.')
param subnetId string

@description('List of CIDR blocks permitted to reach the public API server.')
param authorizedIpRanges array

@description('VM SKU for the system (CPU) node pool.')
param systemNodeSku string = 'Standard_D4s_v5'

@description('Minimum node count for the system pool.')
param systemNodeMin int = 1

@description('Maximum node count for the system pool.')
param systemNodeMax int = 3

@description('Map of tags applied to the cluster.')
param tags object = {}

@description('Kubernetes service CIDR (in-cluster ClusterIP range). MUST be disjoint from vnetCidr/aksSubnetCidr — AKS rejects an overlap with ServiceCidrOverlapExistingSubnetsCidr. The 10.2.0.0/16 default is disjoint from the 10.0.0.0/16 VNet.')
param serviceCidr string = '10.2.0.0/16'

@description('Cluster DNS service IP. MUST lie within serviceCidr.')
param dnsServiceIp string = '10.2.0.10'

// ─── AKS Cluster ──────────────────────────────────────────────────────────────
// Matches: azurerm_kubernetes_cluster.main — name = "${var.prefix}-aks"

resource aks 'Microsoft.ContainerService/managedClusters@2024-02-01' = {
  name:     '${prefix}-aks'
  location: location
  tags:     tags

  // ── Managed identity ─────────────────────────────────────────────────────
  // Matches: identity { type = "SystemAssigned" }
  identity: {
    type: 'SystemAssigned'
  }

  properties: {
    dnsPrefix:         '${prefix}-aks'
    kubernetesVersion: kubernetesVersion

    // ── API server access ──────────────────────────────────────────────────
    // Matches: api_server_access_profile { authorized_ip_ranges = var.authorized_ip_ranges }
    // Public endpoint restricted to operator egress CIDRs.
    apiServerAccessProfile: {
      authorizedIPRanges: authorizedIpRanges
    }

    // ── OIDC + Workload Identity ───────────────────────────────────────────
    // Matches: oidc_issuer_enabled = true, workload_identity_enabled = true
    oidcIssuerProfile: {
      enabled: true
    }
    securityProfile: {
      workloadIdentity: {
        enabled: true
      }
    }

    // ── System (CPU) node pool ─────────────────────────────────────────────
    // Matches: default_node_pool {
    //   name = "system", vm_size = var.system_node_sku,
    //   vnet_subnet_id = var.subnet_id,
    //   enable_auto_scaling = true, min_count = var.system_node_min,
    //   max_count = var.system_node_max, os_disk_size_gb = 128,
    //   type = "VirtualMachineScaleSets",
    //   node_labels = { "reprolab/node-type" = "system" }
    // }
    agentPoolProfiles: [
      {
        name:               'system'
        vmSize:             systemNodeSku
        vnetSubnetID:       subnetId
        enableAutoScaling:  true
        minCount:           systemNodeMin
        maxCount:           systemNodeMax
        osDiskSizeGB:       128
        type:               'VirtualMachineScaleSets'
        mode:               'System'
        nodeLabels: {
          'reprolab/node-type': 'system'
        }
      }
    ]

    // ── Azure Files CSI driver ─────────────────────────────────────────────
    // Matches: storage_profile {
    //   file_driver_enabled = true, blob_driver_enabled = false, disk_driver_enabled = false
    // }
    storageProfile: {
      fileCSIDriver:  { enabled: true  }
      blobCSIDriver:  { enabled: false }
      diskCSIDriver:  { enabled: false }
    }

    // ── Networking ─────────────────────────────────────────────────────────
    // Matches: network_profile {
    //   network_plugin = "azure", network_policy = "azure",
    //   load_balancer_sku = "standard", outbound_type = "loadBalancer"
    // }
    networkProfile: {
      networkPlugin:    'azure'
      networkPolicy:    'azure'
      loadBalancerSku:  'standard'
      outboundType:     'loadBalancer'
      serviceCidr:      serviceCidr
      dnsServiceIp:     dnsServiceIp
    }

    // ── Azure RBAC ──────────────────────────────────────────────────────────
    // Matches: azure_active_directory_role_based_access_control {
    //   managed = true, azure_rbac_enabled = true
    // }
    aadProfile: {
      managed:         true
      enableAzureRBAC: true
    }

    // ── Disable local accounts (CIS AKS 5.1.1) ────────────────────────────
    // Matches: local_account_disabled = true
    // TF sets this explicitly — this is parity, not a hardening delta.
    disableLocalAccounts: true
  }
}

// ─── Outputs ──────────────────────────────────────────────────────────────────

@description('Full Azure resource ID of the AKS cluster.')
output clusterId string = aks.id

@description('Name of the AKS cluster.')
output clusterName string = aks.name

@description('OIDC issuer URL emitted by the cluster.')
output oidcIssuerUrl string = aks.properties.oidcIssuerProfile.issuerURL

@description('Auto-managed node resource group name (MC_*).')
output nodeResourceGroup string = aks.properties.nodeResourceGroup

@description('Client ID of the kubelet managed identity.')
output kubeletIdentityClientId string = aks.properties.identityProfile.kubeletidentity.clientId

@description('Object ID of the kubelet managed identity.')
output kubeletIdentityObjectId string = aks.properties.identityProfile.kubeletidentity.objectId
