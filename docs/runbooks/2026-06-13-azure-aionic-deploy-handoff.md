# Azure AIONIC L1 deploy — handoff & redeploy runbook

- **Date:** 2026-06-13
- **Pairs with:** `docs/runbooks/2026-06-03-azure-aks-gpu-backend-handoff.md` (the standup design) and `infra/azure/STREAM-E-NOTES.md` (Key Vault + orchestrator).
- **Status:** L1 stack was deployed to the live AIONIC subscription (state `succeeded`, 22 resources) **then torn down** to stop spend. The IaC is now deploy-proven. The execution path (Jobs/GPU) is **not** yet tested — it is admin-gated (§4).

This runbook records the real first deploy: what stood up, the three IaC bugs fixed to get there, the permissions wall, and the exact commands to redeploy and to hand off the privileged actions.

---

## 1. Target environment (live)

| Field | Value |
|-------|-------|
| Subscription | `AIONIC Azure` — ID via `az account show --query id -o tsv` |
| Tenant | AIONIC — ID via `az account show --query tenantId -o tsv` |
| Resource group | `rg-sciartgen-external` (westus3) — **shared**; also holds `sciartgen-azure-openai` (AOAI, eastus) + `sciartgentfstate`. Never delete the RG. |
| Stack name | `openresearch-l1` |
| Operator (deploying user) | objectId via `az ad signed-in-user show --query id -o tsv` |

Deployed resources (22): AKS `sciart-aks` + GPU pool `sciarta10080` (A100-80, scale-to-zero), ACR `sciartacr`, storage `sciartgenreprolab` (+ blob `reprolab-artifacts`, files `reprolab-cache`), VNet/subnet/NSG `sciart-*`, Log Analytics `sciart-law` + diagnostics, workload identities `sciart-workload-mi` + `sciart-orch-workload-mi` (+ federated creds), and the 4 role assignments the operator condition permits.

---

## 2. The three first-deploy IaC bugs (fixed in this branch)

Each was a latent defect in never-before-deployed Bicep; each failed a real `az stack group create` until fixed.

| # | Symptom (ARM error) | Root cause | Fix |
|---|---------------------|-----------|-----|
| 1 | `ServiceCidrOverlapExistingSubnetsCidr` | `modules/aks.bicep` set `networkProfile` but no `serviceCidr`, so AKS defaulted to `10.0.0.0/16` — overlapping the VNet/subnet. | Added explicit disjoint `serviceCidr = 10.2.0.0/16` + `dnsServiceIp = 10.2.0.10` (parameterized, defaulted). |
| 2 | `Invalid node label key kubernetes.azure.com/scalesetpriority ... prefix is reserved` | `modules/gpu-nodepool.bicep` manually set a `kubernetes.azure.com/*` node label; that prefix is AKS-reserved. | Removed the label (AKS manages scalesetpriority itself, and only for Spot pools). |
| 3 | `networkAcls.ipRule[*].value ... invalid` | Storage (and Key Vault) reject `/31` + `/32` in IP network rules; a single host must be a bare IPv4. The operator CIDR arrived as `x.x.x.x/32`. | One shared `storageIpRules` var (and the same in `keyvault.bicep`) strips `/32` via `replace(cidr,'/32','')`. AKS `authorizedIPRanges`, conversely, *requires* CIDR — so the `/32` stays for AKS. |

Also added in this branch (Stream E + deploy enablement):
- `modules/network.bicep`: added the `Microsoft.KeyVault` service endpoint on the AKS subnet (the KV firewall `virtualNetworkRules` need it).
- Two gating params so a constrained operator can still deploy the Blob-capable stack (§4):
  - `infra.bicep` → `deployKeyVault bool = true` (set `false` until `Microsoft.KeyVault` is registered).
  - `infra.bicep`/`storage.bicep` → `deployStorageKeyOperatorRole bool = true` (set `false` when the operator can't assign role `81a9662b`).

---

## 3. Redeploy command (validated)

`Microsoft.KeyVault` was unregistered and the operator lacks the Key Operator grant, so the proven deploy gates both off. With full admin grants (§4 done), drop the two `deploy*=false` overrides.

```bash
az login --use-device-code
az account set --subscription <SUBSCRIPTION_ID>   # see: az account show --query id -o tsv

# operator egress /32 locks the public AKS API server:
EGRESS=$(curl -s https://ipinfo.io/ip)

az stack group create \
  --name openresearch-l1 \
  --resource-group rg-sciartgen-external \
  --template-file infra/azure/bicep/infra.bicep \
  --parameters infra/azure/bicep/infra.bicepparam \
  --parameters kubernetesVersion=1.34 authorizedIpRanges="[\"${EGRESS}/32\"]" \
               deployKeyVault=false deployStorageKeyOperatorRole=false \
  --deny-settings-mode none \
  --action-on-unmanage detachAll \
  --yes

# wire backend env from the stack outputs (no secrets; .env.azure is gitignored):
scripts/azure_wire_env.sh rg-sciartgen-external openresearch-l1 .env.azure
```

Prerequisites that are NOT operator-gated: `az`/`kubectl`/`helm`/`jq` installed; `azure-identity`+`azure-storage-blob`+`kubernetes` in the venv (`pip install -r backend/requirements.txt`); providers `Microsoft.ContainerService/ContainerRegistry/Storage/Network/ManagedIdentity/OperationalInsights` registered (they already are on this sub).

---

## 4. Admin hand-off — privileged actions the operator cannot do

The operator has RG **Contributor** + a **constrained RBAC-Administrator** whose ABAC condition only permits assigning AcrPull (`7f951dda`), Storage Blob Data Contributor (`ba92f5b4`), and Storage File SMB Contributor (`0c867c2a`) — to ServicePrincipals. So the operator can build infra but cannot use the cluster or assign other roles. These require an AIONIC subscription admin / User Access Administrator:

```bash
SUB=$(az account show --query id -o tsv)
RG=rg-sciartgen-external

# (A) Kubernetes API access for the operator (kubectl + Job submission)
az role assignment create \
  --assignee "$(az ad signed-in-user show --query id -o tsv)" \
  --role "Azure Kubernetes Service RBAC Cluster Admin" \
  --scope /subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.ContainerService/managedClusters/sciart-aks

# (B) Register the Key Vault provider (subscription-scoped) → unblocks Stream E KV
az provider register -n Microsoft.KeyVault --subscription $SUB

# (C) Files CSI mount — Key Operator on the kubelet identity (objectId f48f064c-... when deployed)
#     kubelet objectId comes from: stack output kubeletIdentityObjectId
az role assignment create \
  --assignee <kubeletIdentityObjectId> \
  --role "Storage Account Key Operator Service Role" \
  --scope /subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.Storage/storageAccounts/sciartgenreprolab

# (D) GPU quota (support ticket; was 0/0): family "Standard NCADS_A100_v4 Family vCPUs", >=24, westus3
#     Portal → Subscriptions → AIONIC Azure → Usage + quotas → filter NCADS_A100_v4 → Request increase.
```

Equivalent simpler alternative to (A)+(C): grant the operator **User Access Administrator** on the RG and they self-serve.

Operator-side, once (A) lands: install kubelogin — `az aks install-cli` — then `kubelogin convert-kubeconfig -l azurecli` to use the cached `az` token non-interactively.

---

## 5. Tested vs NOT tested

**Proven:** IaC compiles + deploys to real Azure (22 resources after the 3 fixes); stack outputs → `.env.azure` wiring; AKS + GPU pool + ACR + storage + identities + monitoring all created.

**NOT tested (all blocked on §4 / quota):** no Kubernetes Job ever ran; CPU-stub smoke (Blob bus / Job watch / artifact path) never ran; no GPU job (quota 0); cell image never built/pushed; no end-to-end `--sandbox azure` reproduce; Stream E (KV + orchestrator) never deployed.

Next validation after the admin grants: build the cell image (`scripts/azure_build_cell_image.sh sciartacr`), run the Blob-only CPU-stub (STREAM-E-NOTES §4), then one SDAR cell, then the smallest-two matrix.

Next: run SDAR → `docs/runbooks/2026-06-14-sdar-on-azure-run.md`

---

## 6. Cost + teardown

Scale-to-zero GPU pool = $0 idle; standing cost is the system node + control plane (~$0.10–0.20/hr) + ACR/storage (a few $/month). To stop all spend (deletes the 22 managed resources, **keeps the RG + AOAI + tfstate**):

```bash
az stack group delete --name openresearch-l1 -g rg-sciartgen-external \
  --action-on-unmanage deleteResources --yes
```

This is what was run on 2026-06-13 after the successful deploy.
