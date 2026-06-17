# Bicep is now the sole IaC

This file formerly tracked Terraform→Bicep parity during the migration period. The Terraform tree was removed on 2026-06-12 (see git history at the commit prior to removal for the full `*.tf` tree). Bicep deployment stacks (`az stack group create`) are now authoritative for all L1 infrastructure.

Parity/hardening deltas that still matter are summarized below as historical rationale for current Bicep behavior.

---

## Hardening deltas vs Terraform

These are additions beyond Terraform parity, or intentional omissions, fixed in the Bicep implementation.

### Added

| Delta | Module | Rationale |
|---|---|---|
| **Log Analytics workspace** (`${prefix}-law`, PerGB2018, 30-day retention) | `monitoring.bicep` | Security-baseline requirement. Centralised control-plane logs for AKS, registry events for ACR, and storage transaction/capacity metrics. No feature flag — always on. |
| **AKS diagnostic settings** (kube-apiserver, kube-controller-manager, kube-scheduler, kube-audit, kube-audit-admin, guard, cluster-autoscaler + AllMetrics) | `monitoring.bicep` | Enables Defender for Cloud / Sentinel ingestion. Not in the original Terraform. |
| **ACR diagnostic settings** (ContainerRegistryRepositoryEvents, ContainerRegistryLoginEvents + AllMetrics) | `monitoring.bicep` | Registry pull/push audit trail. Not in the original Terraform. |
| **Storage account diagnostic settings** (Transaction + Capacity metrics) | `monitoring.bicep` | Storage consumption and anomaly baseline. Not in the original Terraform. |

### Parity BREAKS — TF settings deliberately not ported (latent TF bugs)

The Terraform was never deployed live; these three defects would have surfaced on first deployment and are fixed in Bicep rather than ported. Codex security review 2026-06-10 flagged all three.

| TF setting | Bicep behavior | Why the TF version is broken |
|---|---|---|
| `public_network_access_enabled = false` + `network_rules` (both storage accounts) | `publicNetworkAccess: 'Enabled'` + `defaultAction: 'Deny'` + subnet/IP allow-list | `Disabled` permits traffic ONLY via private endpoints — none exist in either IaC — so AKS pods, the Files mount, and the local orchestrator could never reach storage. The network rules were dead code. Enabled+Deny is the semantic the TF comments describe. |
| `shared_access_key_enabled = false` on the files-hosting account | `allowSharedKeyAccess: true` on whichever account hosts the active Files share | The azurefile CSI driver mounts SMB with the account key. Key auth disabled = first PVC attach fails. Identity-based SMB (Entra Kerberos) is a future hardening, not a flag-flip. |
| (absent) | **Storage Account Key Operator Service Role → kubelet** on the files-hosting account | The SMB data-plane role TF grants does not include `listKeys`, which the CSI driver needs to fetch the mount key. Without it the mount fails even with key auth enabled. |

### Known Bicep warning (informational, non-blocking)

```
/home/abheekp/openresearch/infra/azure/bicep/modules/acr.bicep(43,13) : Warning BCP334:
The provided value can have a length as small as 3 and may be too short to assign to a
target with a configured minimum length of 5.
```

Root cause: the ACR name is `${replace(prefix, '-', '')}acr`. Bicep's static analyser sees that `replace()` on a `@minLength(2)` `@maxLength(8)` input could theoretically produce a 2-char string. In practice `prefix` is required to be ≥2 chars of non-hyphen content (e.g. `repro`), producing a name well above the minimum. The warning does not prevent deployment.
