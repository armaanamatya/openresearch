# Stream E — Key Vault hardening + in-AKS orchestrator: operator notes

> **Doc status:** Current · authored 2026-06-13.
> Design authority: `docs/superpowers/specs/2026-06-12-azure-bicep-canonical-aionic-compute-design.md` (D4/D5/D8).

---

## 1. What each file does

| File | Role |
|------|------|
| `infra/azure/bicep/modules/keyvault.bicep` | Creates the hardened Key Vault (RBAC-only, soft-delete, purge-protection, Deny-default firewall). Grants the orchestrator UAMI "Key Vault Secrets User" (read-only). Secret **values** are never set here. |
| `infra/azure/bicep/infra.bicep` (edited) | Added `orchestratorIdentity` module (dedicated UAMI + federated credential for `reprolab-orchestrator` SA), `keyvault` module (optional, default on), and three new outputs (`orchestratorIdentityClientId`, `keyVaultNameOut`, `keyVaultUriOut`). All new params have defaults so the existing `infra.bicepparam` is unchanged. |
| `infra/azure/helm/templates/orchestrator-serviceaccount.yaml` | Kubernetes ServiceAccount `reprolab-orchestrator` annotated with the orchestrator UAMI client ID. Gated on `.Values.orchestrator.enabled`. |
| `infra/azure/helm/templates/orchestrator-secretproviderclass.yaml` | Secrets Store CSI SecretProviderClass that pulls `azure-openai-api-key` and `anthropic-api-key` from Key Vault and syncs them to k8s Secret `reprolab-orchestrator-secrets`. Gated on `.Values.orchestrator.enabled`. |
| `infra/azure/helm/templates/orchestrator-deployment.yaml` | Long-running Deployment running `python -m backend.cli reproduce <paper> --mode rlm --model azure --sandbox azure` on the CPU system pool. Gated on `orchestrator.enabled` AND `orchestrator.deployment.enabled`. |
| `infra/azure/helm/templates/orchestrator-cronjob.yaml` | Scheduled CronJob (default 02:00 UTC daily) running the same command. Gated on `orchestrator.enabled` AND `orchestrator.cronjob.enabled`. |
| `infra/azure/helm/values.yaml` (edited) | Added fully-commented `orchestrator:` block (all defaults disabled so existing installs are unaffected). |
| `infra/azure/helm/smoke/cpu-stub-job.yaml` (updated) | CPU-only smoke Job that writes a canned `metrics.json` to `smoke/cpu-stub/metrics.json` in Blob via Workload Identity + mounts the Files PVC. Reuses the existing `reprolab-sa` ServiceAccount. |

---

## 2. Out-of-band commands to set the two Key Vault secrets

These commands set **secret values** that MUST NOT appear in any committed file.
Source the values from a gitignored secret store (1Password, Azure Key Vault import, env file, etc.).

```bash
KV_NAME=$(az stack group show \
  -g rg-sciartgen-external \
  -n openresearch-l1 \
  --query outputs.keyVaultNameOut.value -o tsv)

# Azure OpenAI API key (for --model azure root LLM)
az keyvault secret set \
  --vault-name "${KV_NAME}" \
  --name azure-openai-api-key \
  --value "$(op read 'op://Private/AIONIC-AOAI/credential')"
  # ^ substitute your actual secret-retrieval command; never hardcode the value

# Anthropic API key (for Sonnet sub-agents)
az keyvault secret set \
  --vault-name "${KV_NAME}" \
  --name anthropic-api-key \
  --value "$(op read 'op://Private/Anthropic-API/credential')"
```

**Important:** the operator identity must hold `Key Vault Administrator` or `Key Vault Secrets Officer`
on the vault to write secrets. The orchestrator UAMI only holds `Key Vault Secrets User` (read-only) —
it cannot self-provision its own secrets.

---

## 3. How to enable the orchestrator

### 3a. Prerequisites

1. Bicep L1 deployed:
   ```bash
   az stack group create \
     --name openresearch-l1 \
     --resource-group rg-sciartgen-external \
     --template-file infra/azure/bicep/infra.bicep \
     --parameters infra/azure/bicep/infra.bicepparam \
     --parameters authorizedIpRanges='["<OPERATOR_IP>/32"]' \
     --deny-settings-mode none \
     --action-on-unmanage detachAll
   ```
   > Key Vault deploys only when `Microsoft.KeyVault` is registered on the
   > subscription (`az provider register -n Microsoft.KeyVault`). See the deploy
   > handoff runbook `docs/runbooks/2026-06-13-azure-aionic-deploy-handoff.md`
   > for the full prerequisite + admin-grant list.

2. Collect outputs:
   ```bash
   ORCH_CLIENT_ID=$(az stack group show \
     -g rg-sciartgen-external -n openresearch-l1 \
     --query outputs.orchestratorIdentityClientId.value -o tsv)
   KV_NAME=$(az stack group show \
     -g rg-sciartgen-external -n openresearch-l1 \
     --query outputs.keyVaultNameOut.value -o tsv)
   TENANT_ID=$(az account show --query tenantId -o tsv)
   ```

3. Set the two Key Vault secrets (see section 2 above).

4. Install the Secrets Store CSI driver + Azure Key Vault provider (cluster-wide, once):
   ```bash
   helm install csi-secrets-store secrets-store-csi-driver/secrets-store-csi-driver \
     -n kube-system --set syncSecret.enabled=true
   helm install azure-csi-provider azure-keyvault-secrets-provider/secrets-store-csi-driver-provider-azure \
     -n kube-system
   ```

5. Build and push the orchestrator image to ACR:
   ```bash
   az acr build \
     --registry sciartacr \
     --image reprolab-orchestrator:$(git rev-parse --short HEAD) \
     --file docker/Dockerfile .
   ```

### 3b. Enable the Deployment (always-on run)

```bash
helm upgrade reprolab-aks infra/azure/helm \
  --set orchestrator.enabled=true \
  --set orchestrator.deployment.enabled=true \
  --set orchestrator.image=sciartacr.azurecr.io/reprolab-orchestrator:<sha> \
  --set orchestrator.paper=2605.15155 \
  --set orchestrator.keyVaultName="${KV_NAME}" \
  --set orchestrator.identityClientId="${ORCH_CLIENT_ID}" \
  --set orchestrator.tenantId="${TENANT_ID}" \
  --set orchestrator.env.resourceGroup=rg-sciartgen-external \
  --set orchestrator.env.region=westus3 \
  --set orchestrator.env.aksCluster=sciart-aks \
  --set orchestrator.env.acrLoginServer=sciartacr.azurecr.io \
  --set orchestrator.env.storageAccount=sciartgenreprolab \
  --set orchestrator.env.blobContainer=reprolab-artifacts \
  --set orchestrator.env.filesShare=reprolab-cache \
  --set orchestrator.env.nodePoolName=sciarta10080 \
  --set orchestrator.env.baseImage=sciartacr.azurecr.io/aks-cell-base:<sha>
```

### 3c. Enable the CronJob (scheduled/nightly batch)

Same flags as above, but replace `orchestrator.deployment.enabled=true` with
`orchestrator.cronjob.enabled=true` (or add both to run both).
Override the schedule with `--set orchestrator.cronjob.schedule="0 3 * * *"` as needed.

---

## 4. CPU stub smoke test

After `helm install`/`helm upgrade` (before enabling the orchestrator):

```bash
# Fill these two values from Bicep outputs:
STORAGE_ACCOUNT=sciartgenreprolab
BLOB_CONTAINER=reprolab-artifacts

# Edit the REPLACE_WITH_ placeholders in the Job:
sed \
  -e "s/REPLACE_WITH_STORAGE_ACCOUNT/${STORAGE_ACCOUNT}/" \
  -e "s/REPLACE_WITH_BLOB_CONTAINER/${BLOB_CONTAINER}/" \
  infra/azure/helm/smoke/cpu-stub-job.yaml | kubectl apply -f - -n reprolab

# Watch:
kubectl get jobs -n reprolab -w
kubectl logs -n reprolab job/cpu-stub-smoke -f

# Verify the blob was written:
az storage blob show \
  --account-name "${STORAGE_ACCOUNT}" \
  --container-name "${BLOB_CONTAINER}" \
  --name smoke/cpu-stub/metrics.json \
  --auth-mode login
```

**Success criteria:** Job status = `Complete`; logs contain `Blob write OK` and `Files PVC OK`;
the blob `smoke/cpu-stub/metrics.json` appears in Azure Storage Explorer or via `az storage blob show`.

> **Caveat (Files mount):** the `Files PVC OK` half requires the kubelet to hold the
> Storage Account Key Operator role (`81a9662b`). When the stack was deployed with
> `deployStorageKeyOperatorRole=false` (operator lacked that grant — see the handoff
> runbook), the azurefile CSI mount fails and the Job cannot reach `Complete`. For a
> **Blob-only** validation in that state, delete the Files `volume` + `volumeMount`
> from `cpu-stub-job.yaml` and assert only `Blob write OK`. Re-add Files once the
> Key Operator role is granted and the stack is redeployed with the default
> `deployStorageKeyOperatorRole=true`.

---

## 5. Known gaps and assumptions

### `--model claude-oauth` is NOT usable in an unattended pod

The Claude OAuth credential lives in `~/.claude/.credentials.json` on the operator's local machine
(or macOS Keychain). There is no supported way to mount it into a Kubernetes pod. Therefore:

- **Default model for both Deployment and CronJob is `azure`** (AOAI root via `AZURE_OPENAI_API_KEY`
  pulled from Key Vault). This is the correct production posture: 100% client-owned credentials.
- `--model claude-oauth` works for local orchestrator runs (`python -m backend.cli reproduce ...`)
  but will fail in an unattended pod with an auth error.
- If an operator temporarily wants Anthropic-API root model in a pod, they would set
  `--model claude` and ensure `ANTHROPIC_API_KEY` (already in Key Vault / the synced Secret) is
  funded with credits (see CLAUDE.md "Gotchas" — a no-credit key dies at the first Sonnet sub-call).

### Secrets Store CSI driver is a cluster-level prerequisite

The `SecretProviderClass` and the CSI volume in both the Deployment and CronJob require the
`secrets-store-csi-driver` DaemonSet and the Azure provider to be installed cluster-wide.
This chart does NOT install them (they are kube-system infra, not namespace-scoped).
See section 3a step 4 for the install commands.

### Key Vault network rules require the AKS subnet service endpoint

The `keyvault.bicep` module adds the AKS subnet to the vault's `virtualNetworkRules`.
Azure requires that the subnet has the `Microsoft.KeyVault` service endpoint enabled.
The `network.bicep` module must have `serviceEndpoints: [{service: 'Microsoft.KeyVault'}]`
on the AKS subnet for this rule to be effective. Verify with:
```bash
az network vnet subnet show \
  -g rg-sciartgen-external \
  --vnet-name sciart-vnet \
  --name sciart-aks-subnet \
  --query serviceEndpoints
```
If the endpoint is absent, the orchestrator pod cannot reach the vault from inside the cluster.
Add it to `modules/network.bicep` under the AKS subnet `serviceEndpoints` array if missing.

### Orchestrator image must be pre-built

The Deployment and CronJob reference `.Values.orchestrator.image` which defaults to `""`.
A non-empty PINNED tag (never `:latest`) must be supplied at `helm upgrade` time.
The image must have the full repo installed (`pip install -r backend/requirements.txt`) and
the `azure-identity` + `azure-storage-blob` + `kubernetes` packages available.

### Tenant ID is not surfaced as a Bicep output

The Bicep `tenant()` function is used internally in `keyvault.bicep` but is not output because
it is always the subscription's tenant, which the operator already knows. Use
`az account show --query tenantId -o tsv` to retrieve it for the Helm `--set orchestrator.tenantId=`.
