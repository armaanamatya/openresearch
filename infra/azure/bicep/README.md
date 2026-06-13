# ReproLab — Azure Access Bootstrap: Bicep L0

**Layer:** L0 (subscription-scope one-time access bootstrap). See §Layer split below.

---

## Why L0 exists

Bicep L1 (`infra/azure/bicep/infra.bicep`) creates role assignments for the kubelet managed identity (AcrPull, Storage Blob/File Data Contributor). This requires the operator running `az stack group create` to hold **User Access Administrator** (or higher) on the resource groups — which today means subscription **Owner**.

L0 removes that ongoing requirement. A subscription admin runs this Bicep deployment **once**. It:

1. Creates the main resource group and optionally a tfstate-legacy resource group.
2. Grants the operator principal two built-in roles scoped to **each** created RG:
   - **Contributor** (`b24988ac-6180-42a0-ab88-20f7382dd24c`) — lets the operator create and manage all resources in the RG.
   - **User Access Administrator** (`18d7d88d-d35e-4fb5-a5c3-7773c20a72d9`) — lets the operator create in-RG role assignments without subscription-level Owner.

After L0, day-to-day operators need no subscription-level permissions at all.

---

## Layer split

| Layer | What | Managed by | Run by |
|-------|------|-----------|--------|
| **L0 — Access bootstrap** (this directory) | RG creation + operator role grants | **Bicep** | Subscription admin, once |
| **L1 — Azure infra** (`infra/azure/`) | AKS, ACR, storage, networking, workload identity | **Bicep** (`infra.bicep`) | Operator (Contributor + UAA on RG) |
| **L2 — In-cluster scaffold** (`infra/azure/helm/`) | Namespace, ServiceAccount, RBAC, PVC, NVIDIA plugin | **Helm** | Operator, per cluster |
| **L3 — Runtime** | Per-cell Kubernetes Jobs | **Orchestrator** (`k8s_job_cell_runner.py`) | Automated |

---

## Files

```
infra/azure/bicep/
  main.bicep                Subscription-scope deployment entry point (L0)
  rg-grants.bicep           RG-scope module; called once per RG for role grants
  main.bicepparam.example   Placeholder parameter file (copy before use)
  infra.bicep               L1 deployment entry point
  infra.bicepparam.example  L1 parameter template (copy → infra.bicepparam, then commit)
  bootstrap/
    admin-bootstrap.sh      One-time OIDC wiring for GitHub Actions deploys
    pipeline-identity.bicep Managed-identity alternative for Contributor-only operators
  modules/
    acr.bicep  aks.bicep  gpu-nodepool.bicep  identity.bicep
    monitoring.bicep  network.bicep  storage.bicep
```

---

## Adoption path: existing RG, Contributor-only operator (no admin script)

Use this path when the admin has **already** created the resource group in the portal and granted the operator plain **Contributor** on it — no User Access Administrator, no Entra rights, and no appetite for running a script. **This is the current AIONIC reality:** RG `rg-sciartgen-external`, region `westus3`, operator holds Contributor scoped to that RG and nothing else.

The app-registration bootstrap below won't work here — creating an app registration needs Entra permissions the operator doesn't have. A **user-assigned managed identity** is the substitute: it supports the same GitHub OIDC federation, but it is an ordinary ARM resource, so Contributor is enough to create it.

### Step 1 — operator creates the pipeline identity (you, today)

```bash
az deployment group create \
  --resource-group rg-sciartgen-external \
  --template-file infra/azure/bicep/bootstrap/pipeline-identity.bicep \
  --parameters githubOrg=<ORG> githubRepo=<REPO>

# Note the outputs: clientId (GitHub variable), principalId (admin grant target)
```

### Step 2 — admin grants two roles (one portal visit, ~2 minutes)

Hand the admin the `principalId` output and these instructions — it is the same **Access control (IAM) → Add role assignment** screen they used to grant the operator Contributor:

| Role | Why |
|---|---|
| **Contributor** | Lets the pipeline deploy L1 resources into the RG |
| **Role Based Access Control Administrator** | Lets L1 create its in-RG role assignments (AcrPull → kubelet, Storage Blob/File data roles → workload identity). Recommend adding a condition restricting assignable roles to exactly those three. |

Substitute **User Access Administrator** for RBAC Administrator only if deployment-stack deny-settings are wanted (`denyAssignments/write` is not in RBAC Administrator; see the deny-settings note in `infra-deploy.yml`).

### Step 3 — set GitHub repo variables and the environment gate

Same as the app-registration path: set `AZURE_CLIENT_ID` (the identity's clientId output), `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`, `AZURE_MAIN_RG_NAME` as repo **variables**, and create the `azure` environment with required reviewers. No secret exists in either path.

---

## L0 usage — full admin bootstrap

### 0. Prerequisites

- `az login` as an account with **Owner** on the target subscription (this is the one-time admin step that L0 removes for all subsequent work).
- Azure CLI ≥ 2.50 (`az --version`).
- Bicep CLI ≥ 0.29 — installed automatically by the Azure CLI, or manually:
  ```bash
  az bicep install
  az bicep version
  ```

### 1. Configure parameters

```bash
cd infra/azure/bicep

cp main.bicepparam.example main.bicepparam
# Fill in every <PLACEHOLDER> — see comments in the file
```

Required placeholders:

| Parameter | Placeholder | Notes |
|-----------|-------------|-------|
| `location` | `<AZURE_REGION>` | Must match the region you will use for L1 |
| `principalId` | `<OPERATOR_ENTRA_GROUP_OBJECT_ID>` | The operator group/user/SP object ID |
| `principalType` | `'Group'` | Change to `'User'` or `'ServicePrincipal'` if not a group |

Optional:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `mainRgName` | `rg-reprolab` | Must match the RG you pass to `az stack group create` for L1 |
| `createTfstateRg` | `false` | Legacy only; set `true` only if a separate tfstate RG is needed for a legacy workflow |
| `tfstateRgName` | `rg-reprolab-tfstate` | Only relevant when `createTfstateRg = true` |

### 2. Deploy

```bash
az deployment sub create \
  --location <AZURE_REGION> \
  --template-file infra/azure/bicep/main.bicep \
  --parameters infra/azure/bicep/main.bicepparam
```

The deployment is **idempotent** — re-running it is safe. Role assignment names are seeded with `guid(rgId, principalId, roleDefinitionId)`, so ARM treats them as the same resource on every run.

### 3. Verify

```bash
# Check role assignments on the main RG
az role assignment list \
  --resource-group rg-reprolab \
  --assignee <OPERATOR_ENTRA_GROUP_OBJECT_ID> \
  --output table
```

Each command should show two rows: Contributor and User Access Administrator.

---

## Role-assignment design

| Scope | Role | Built-in ID | Purpose |
|-------|------|-------------|---------|
| `mainRgName` RG | Contributor | `b24988ac-6180-42a0-ab88-20f7382dd24c` | Full resource CRUD in the main RG |
| `mainRgName` RG | User Access Administrator | `18d7d88d-d35e-4fb5-a5c3-7773c20a72d9` | Create in-RG role assignments (AcrPull, Blob/File Data roles) |

Assignment names use `guid(rgId, principalId, roleDefinitionId)` — deterministic and idempotent.

Bicep cannot create RG-scoped role assignments directly from a `targetScope = 'subscription'` file. `rg-grants.bicep` is a helper module invoked with `scope: resourceGroup(...)` — the standard ARM/Bicep pattern for this case.

---

## Validation note

`bicep build` **was run during authoring** against the Bicep CLI and completed with no errors (one known BCP334 warning in `modules/acr.bicep` — see `MIGRATION.md`). Re-run before any structural edits:

```bash
bicep build  infra/azure/bicep/main.bicep
bicep build  infra/azure/bicep/infra.bicep
bicep lint   infra/azure/bicep/main.bicep
```

---

## One-time admin bootstrap (OIDC path)

`infra/azure/bicep/bootstrap/admin-bootstrap.sh` is the **single reviewed script** the subscription Owner runs once to wire up the GitHub Actions OIDC deploy path. No secrets are ever created; OIDC replaces them entirely.

### What the script does (step-by-step)

| Step | Action |
|------|--------|
| 1 | `az account set` — pins the CLI session to `$SUBSCRIPTION_ID`. |
| 2 | Creates (or finds by display name) an Entra **app registration** and its **service principal** (`$APP_NAME`, default `openresearch-deployer`). Idempotent: re-running skips creation if the app already exists. |
| 3 | Adds ONE **federated credential** to the app (idempotent upsert by name): `github-deploy-<ENVIRONMENT>` for `repo:<ORG>/<REPO>:environment:<ENVIRONMENT>`. There is deliberately no `pull_request` credential — that subject binds neither actor nor workflow content, so PR jobs run with no Azure access at all. |
| 4 | Runs `az deployment sub what-if` and **pauses** for the admin to review the change-set before any mutation. |
| 5 | On confirmation (`yes`), runs `az deployment sub create` passing `deployPrincipalId=<SP object id>` (and the operator group if `$OPERATOR_GROUP_OBJECT_ID` is set). |
| 6 | Prints the three GitHub repo variables to set: `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`. |

### What the admin should review in the what-if

The what-if output will show resource groups and role assignments. Check that:

- Exactly the expected resource groups appear as **Create** or **No Change**.
- The deploy principal (SP object ID) gets **Contributor** and **RBAC Administrator** (ABAC-constrained to the four roles L1 assigns) on the **main RG only** — no subscription-scope resources appear.
- The operator group grants (if `$OPERATOR_GROUP_OBJECT_ID` is set) match what was already deployed.

### Exact invocation

```bash
export SUBSCRIPTION_ID="<SUBSCRIPTION_GUID>"
export TENANT_ID="<TENANT_GUID>"
export LOCATION="eastus"              # must match the L1 region
export GITHUB_ORG="openresearch-ai"
export GITHUB_REPO="openresearch"
export MAIN_RG_NAME="rg-reprolab"

# Optional — include if you also want the operator group to receive grants:
export OPERATOR_GROUP_OBJECT_ID="<GROUP_OBJECT_ID>"

bash infra/azure/bicep/bootstrap/admin-bootstrap.sh
```

`APP_NAME` and `GITHUB_ENVIRONMENT` can be overridden (defaults: `openresearch-deployer` and `azure`).

---

## GitHub Actions OIDC deploys

After the bootstrap script succeeds, all deploys run through `.github/workflows/infra-deploy.yml` with no stored secrets.

### The three repo variables

Set these as repository **variables** (not secrets) under Settings → Secrets and variables → Actions → Variables:

| Variable | Value | Where to find it |
|----------|-------|-----------------|
| `AZURE_CLIENT_ID` | App registration client ID (or the managed identity's clientId — see the adoption path above) | Printed by `admin-bootstrap.sh` step 6, or: `az ad app list --display-name openresearch-deployer --query '[0].appId' -o tsv` |
| `AZURE_TENANT_ID` | Entra tenant GUID | Printed by step 6, same as `$TENANT_ID` |
| `AZURE_SUBSCRIPTION_ID` | Azure subscription GUID | Printed by step 6, same as `$SUBSCRIPTION_ID` |
| `AZURE_MAIN_RG_NAME` | Resource group name for L1 deployment (e.g. `rg-sciartgen-external`) | From the admin / L0 outputs |

Optional variables used by the workflow (safe defaults shown):

| Variable | Default | Purpose |
|----------|---------|---------|
| `AZURE_DENY_SETTINGS_MODE` | `none` | Deployment-stack deny-settings; needs `Microsoft.Resources/deploymentStacks/manageDenySetting/action` (Azure Deployment Stack Owner role) — neither Contributor, RBAC Administrator, nor UAA includes it (see workflow comment) |

### The `azure` environment protection rule — the approval gate

Create a GitHub environment named `azure` (Settings → Environments → New environment). Add at least one **required reviewer**. The `deploy` job declares `environment: azure`; GitHub will pause and request approval before OIDC credentials are exchanged and any `az` command runs.

Reviewers see the commit SHA, the triggering actor, and (in the logs) the preceding what-if output. After approval the job exchanges the short-lived OIDC token, deploys L1 as a deployment stack, and exits — the token is never stored. (L0 never runs in the pipeline: it needs subscription-scope rights the pipeline identity deliberately lacks; it is the admin's one-time step.)

### PR behavior — validation only, zero Azure access

Every PR that touches `infra/azure/bicep/**` triggers the `validate` job: `bicep build` + `bicep lint` on every entry point, with **no Azure login and no `id-token` permission**. The `pull_request` OIDC claim binds neither actor nor workflow content, so granting it federation would let any PR that modifies the workflow exchange tokens for the deploy identity — instead, the ARM `what-if` change-set is shown inside the approval-gated deploy job, where the environment reviewers see it before the stack is applied.

### Why no secrets exist

OIDC works by issuing a short-lived JWT from GitHub's OIDC provider, which Azure's federated credential configuration trusts. The JWT subject must match the single registered subject (`environment:azure`). There is no client secret, no certificate, and nothing to rotate — revoking the app registration or deleting the federated credential is sufficient to cut access.

---

## Cross-references

- Bicep L1: `infra/azure/bicep/infra.bicep` + `modules/` — deployed by the `deploy` job.
- Helm L2: `infra/azure/helm/` — depends on L1 outputs.
- Design doc: `docs/superpowers/specs/2026-06-03-azure-aks-gpu-backend-design.md`
- Runbook: `docs/runbooks/2026-06-03-azure-aks-gpu-backend-handoff.md`
