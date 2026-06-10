// ─── Monitoring module — Log Analytics workspace + diagnostic settings ─────────
//
// NEW — security-baseline requirement, not in Terraform. Wire into infra.bicep
// behind no flag (always on, chosen requirement).
//
// Provisions:
//   - Log Analytics workspace (sku: PerGB2018, 30-day retention)
//   - diagnosticSettings for AKS (control-plane logs + metrics)
//   - diagnosticSettings for ACR (container registry events + metrics)
//   - diagnosticSettings for the storage account (blob + file service metrics)

targetScope = 'resourceGroup'

// ─── Parameters ───────────────────────────────────────────────────────────────

@description('Short alphanumeric prefix prepended to every resource name.')
param prefix string

@description('Azure region for the Log Analytics workspace.')
param location string = resourceGroup().location

@description('Retention in days for Log Analytics workspace data.')
param retentionDays int = 30

@description('Full resource ID of the AKS cluster to monitor.')
param aksClusterId string

@description('Full resource ID of the ACR registry to monitor.')
param acrId string

@description('Full resource ID of the storage account to monitor (Standard account).')
param storageAccountId string

@description('Map of tags applied to the workspace.')
param tags object = {}

// ─── Log Analytics workspace ──────────────────────────────────────────────────

resource logAnalyticsWorkspace 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name:     '${prefix}-law'
  location: location
  tags:     tags
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: retentionDays
  }
}

// Existing resource references — required for Bicep scope expressions.

resource aksClusterRef 'Microsoft.ContainerService/managedClusters@2024-02-01' existing = {
  name: last(split(aksClusterId, '/'))
}

resource acrRef 'Microsoft.ContainerRegistry/registries@2023-11-01-preview' existing = {
  name: last(split(acrId, '/'))
}

resource storageAccountRef 'Microsoft.Storage/storageAccounts@2023-05-01' existing = {
  name: last(split(storageAccountId, '/'))
}

// ─── AKS diagnostic settings ──────────────────────────────────────────────────
// Control-plane logs (kube-apiserver, kube-controller-manager, etc.) + metrics.

resource aksDiagnostics 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = {
  name:  '${prefix}-aks-diag'
  scope: aksClusterRef
  properties: {
    workspaceId: logAnalyticsWorkspace.id
    logs: [
      {
        category: 'kube-apiserver'
        enabled:  true
      }
      {
        category: 'kube-controller-manager'
        enabled:  true
      }
      {
        category: 'kube-scheduler'
        enabled:  true
      }
      {
        category: 'kube-audit'
        enabled:  true
      }
      {
        category: 'kube-audit-admin'
        enabled:  true
      }
      {
        category: 'guard'
        enabled:  true
      }
      {
        category: 'cluster-autoscaler'
        enabled:  true
      }
    ]
    metrics: [
      {
        category: 'AllMetrics'
        enabled:  true
      }
    ]
  }
}

// ─── ACR diagnostic settings ──────────────────────────────────────────────────
// Container registry events (push, pull, login) + metrics.

resource acrDiagnostics 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = {
  name:  '${prefix}-acr-diag'
  scope: acrRef
  properties: {
    workspaceId: logAnalyticsWorkspace.id
    logs: [
      {
        category: 'ContainerRegistryRepositoryEvents'
        enabled:  true
      }
      {
        category: 'ContainerRegistryLoginEvents'
        enabled:  true
      }
    ]
    metrics: [
      {
        category: 'AllMetrics'
        enabled:  true
      }
    ]
  }
}

// ─── Storage account diagnostic settings (metrics) ────────────────────────────
// Storage account-level transaction + capacity metrics.

resource storageAccountDiagnostics 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = {
  name:  '${prefix}-sa-diag'
  scope: storageAccountRef
  properties: {
    workspaceId: logAnalyticsWorkspace.id
    metrics: [
      {
        category: 'Transaction'
        enabled:  true
      }
      {
        category: 'Capacity'
        enabled:  true
      }
    ]
  }
}

// ─── Outputs ──────────────────────────────────────────────────────────────────

@description('Resource ID of the Log Analytics workspace.')
output workspaceId string = logAnalyticsWorkspace.id

@description('Name of the Log Analytics workspace.')
output workspaceName string = logAnalyticsWorkspace.name
