# Azure Bicep-canonical, AIONIC-wired, all-Azure compute — design

> **Doc status:** Draft · spec · authored 2026-06-12 · supersedes the Terraform L1
> path in [`2026-06-03-azure-aks-gpu-backend-design.md`](2026-06-03-azure-aks-gpu-backend-design.md)
> for IaC tooling. Pairs with runbook `docs/runbooks/2026-06-03-azure-aks-gpu-backend-handoff.md`.

## 1. Context & current state

The client (AIONIC / DeepInvent) wants OpenResearch to run **end-to-end on their
Azure**, off the operator's local 8×A5000 box, with A100-80GB GPUs and support for
unattended/scheduled runs. Driver (confirmed): A100-80 memory + off-my-hardware /
client requirement + unattended.

**What already exists in the subscription** (portal, 2026-06-12):
- Subscription **AIONIC Azure** `51008c59-ebf4-4699-8f2c-896724144d42`, tenant AIONIC.
- Resource group **`rg-sciartgen-external`** in **`westus3`**; operator holds
  **Contributor-only** on it (no Owner, no User Access Administrator, no Entra rights).
- Storage account `sciartgentfstate` (a Terraform-state account — becomes vestigial
  under pure Bicep), Azure OpenAI **`sciartgen-azure-openai`**.
- **No** AKS cluster, ACR, GPU pool, workload identity, or Key Vault yet.

**What the repo already has** (built, `bicep build`+`lint` clean as of 2026-06-12,
**not yet proven live**):
- Bicep L0 (`main.bicep` — subscription-scope RG + role grants), L1 (`infra.bicep` +
  `modules/` — VNet, AKS, GPU node pool(s), ACR, storage, workload identity, monitoring),
  and `bootstrap/pipeline-identity.bicep` (the Contributor-only OIDC adoption path).
- A parallel **Terraform** L1 (`infra/azure/*.tf`) — the legacy authoritative path.
- Fully-implemented backend Azure execution: `aks_job_backend.py`,
  `k8s_job_cell_runner.py`, `gpu_catalog.py` Azure SKUs, `config.py` `OPENRESEARCH_AZURE_*`.
  Scheduling is label-based (`nodeSelector: {reprolab/sku: <short_name>}`) and matches
  the Bicep `gpuPools[].skuLabel`. The backend is **IaC-tool-agnostic** — it consumes
  env vars only; removing Terraform does not touch runtime code.
- GitHub OIDC deploy workflow `infra-deploy.yml` (Bicep-only, `az stack group create`).

**The gap:** (a) two parallel IaC systems (drift risk); (b) the Bicep params/docs lead
with the wrong path for AIONIC (L0/Terraform-import, which needs Owner) instead of the
Contributor-only adoption path; (c) the "wire infra outputs → backend env" and "build &
pin the ACR cell image" steps are manual/undocumented; (d) AKS provisions *training*
compute only — nothing defines **where the reasoning loop runs unattended, or with what
non-personal credential**; (e) a node-pool-name inconsistency (`gpua100` vs `gpunodes`
vs Bicep `<prefix>a10080`) — cosmetic (bookkeeping, not the selector) but a smell.

## 2. Target architecture — all-Azure, client-owned

```
  ┌── rg-sciartgen-external (westus3, sub 51008c59…) ───────────────────────┐
  │                                                                          │
  │  [Key Vault]  AOAI key · Anthropic API key   ← workload identity (get)   │
  │       ▲                                                                  │
  │  [CPU orchestrator host]  reasoning loop (no GPU)                        │
  │     root model → sciartgen-azure-openai (--model azure)                  │
  │     sub-agents → Sonnet via ANTHROPIC_API_KEY                            │
  │       │ k8s API (in-VNet → no public-IP allowlist friction)             │
  │       ▼                                                                  │
  │  [AKS]  CPU system pool + A100-80 GPU pool (scale-to-zero 0→N)           │
  │     per-cell K8s Job · ACR cell image · Blob artifact bus · Files cache  │
  │  [ACR]  reprolab-cell:<sha>   [Log Analytics]  diagnostics              │
  └──────────────────────────────────────────────────────────────────────────┘
   IaC: Bicep only (deployment stacks, server-side state — no tfstate account)
```

Two auth surfaces, both **metered, both Key-Vault-sourced, zero personal creds**
(the accepted cost of the no-personal-creds posture): AOAI root (per-token) + Anthropic
API Sonnet sub-agents (per-token, must be a **funded** key — a no-credit key dies at the
first sub-call, see CLAUDE.md). Training GPU billed only while a Job holds a node
(scale-to-zero idle = $0).

## 3. Locked decisions (from 2026-06-12 Q&A)

| # | Decision | Rationale |
|---|---|---|
| D1 | **Bicep is the sole IaC; delete Terraform.** | One source of truth, no drift. Recoverable via git. Backend is IaC-agnostic. |
| D2 | **Training = AKS Jobs** (`--sandbox azure`). | A100-80 + 2–4 concurrent + unattended; per-cell isolation + scale-to-zero earn their keep. |
| D3 | **AIONIC entry = Contributor-only adoption path** (`pipeline-identity.bicep` + admin 2-role grant), NOT L0. | RG exists; operator has Contributor-only. L0 needs Owner. |
| D4 | **Reasoning = AOAI root + Anthropic-API Sonnet, keys in Key Vault.** | 100% client-owned, no personal creds; reuses the existing AOAI resource. |
| D5 | **Orchestrator host runs in Azure** (Phase 2). Recommended: **in-AKS Deployment + CronJob** (in-VNet ⇒ no API allowlist friction; scheduled = CronJob; reuses workload identity). Fallbacks: Container Apps (VNet-integrated) or a CPU VM. **To confirm at Phase 2.** | Unattended needs an always-on home with a stable path to the API server + Key Vault. |
| D6 | **pure-Bicep ⇒ drop tfstate machinery** (account/RG). Deployment stacks hold state server-side. | `sciartgentfstate` + `rg-reprolab-tfstate` become dead weight. |
| D7 | **Harden the AOAI root onto ONE shared transport** (retries=6 + `@with_429_backoff` + api-version `2024-10-21`) — register the backend's `AzureOpenAILlmClient` into rlm so the root loop and primitives share it. | rlm's vendored `AzureOpenAIClient` has SDK-default 2 retries, no 429 backoff, stale api-version `2024-02-01`; AOAI throttles aggressively → root surfaces `RateLimitError` too soon. (audit 2026-06-12) |
| D8 | **Opt-in AOAI sub-agent runtime** (new `AgentRuntime`), **default Sonnet**. | Quality-critical code-writing executor stays on validated Sonnet; AOAI capability exists behind a flag for all-Azure/cost experiments. gpt-4o code-writing is `paper_validated=False` for this harness. |
| D9 | **Fan-out stays sequential best-of-N** (no parallelization now). | Already fail-soft + idempotent + wall-clock-guarded; parallelizing needs RDR-style per-candidate dirs everywhere + the global env-guidance mutation removed + concurrency tests — deferred. |

Robustness basis: audit `[[2026-06-12 AOAI + fan-out audit]]` (in-conversation). Key code anchors:
rlm vendored Azure client `rlm/clients/azure_openai.py`; backend hardened client
`backend/services/context/workspace/tools/azure_openai_client.py`; root construction
`run.py:~1916`; accelerator nav-gating `run.py:~1906`; agent-runtime Protocol `backend/agents/runtime/base.py:159`.

## 4. Phased implementation plan

Verify gate on every IaC step: `bicep build` + `bicep lint` on all entry points
(local standalone `bicep` 0.44.1, as CI does — no Azure creds required). Backend
changes gate on the existing Azure tests under `tests/`.

### Phase 1 — Bicep-canonical, AIONIC-wired, training path seamless

1. **Retire Terraform (D1, D6).** Delete `infra/azure/{main,outputs,providers,variables,versions}.tf`,
   `backend.tf`, `bootstrap/*.tf`, `modules/*/` (TF modules), `envs/deepinvent/*.tf*`.
   Drop tfstate from L0 (`createTfstateRg` default → `false`; keep the param for back-compat,
   stop documenting it). Strip TF lines from `.gitignore`.
2. **Rewrite the docs to Bicep-only.** `infra/azure/README.md` (L1 = `az stack group create`),
   `bicep/README.md` (drop the Terraform-import handshake; lead with the Contributor-only
   adoption path for AIONIC), fold `MIGRATION.md` into a short "Bicep is authoritative" note,
   update the handoff runbook + the CLAUDE.md Azure-backend line ("Terraform/Helm IaC" → "Bicep/Helm IaC").
3. **Commit a real `infra.bicepparam`** (no secrets): `prefix='sciart'`, `location='westus3'`,
   `storageAccountName='sciartgenreprolab'` (≤24 lc-alnum, globally unique — verify free),
   A100-80 pool `maxNodes:1`, `acrSku:'Standard'`; only `authorizedIpRanges` + `kubernetesVersion`
   left as clearly-marked placeholders with the exact `az` fill commands.
4. **Seam glue:** `scripts/azure_wire_env.sh` (`az stack group show … --query outputs` →
   writes `.env.azure` with the `OPENRESEARCH_AZURE_*` mapping — one command, not 5 manual copies);
   `scripts/azure_build_cell_image.sh` (`az acr build` `docker/aks-cell-base/` →
   `<acr>/reprolab-cell:<git-sha>`, prints the pinned `OPENRESEARCH_AZURE_BASE_IMAGE`).
5. **Unify the node-pool name (E).** One source of truth for `azure_node_pool_name`
   (align `config.py` + `k8s_job_cell_runner.py` defaults; document it as bookkeeping-only,
   the `reprolab/sku` label is the real selector). Fix `.env.example` Azure block to map
   1:1 to Bicep outputs with AIONIC defaults (`OPENRESEARCH_AZURE_RESOURCE_GROUP=rg-sciartgen-external`,
   `OPENRESEARCH_AZURE_REGION=westus3`, etc.).
6. **Repo-variable doc** for `infra-deploy.yml`: `AZURE_SUBSCRIPTION_ID=51008c59…`,
   `AZURE_MAIN_RG_NAME=rg-sciartgen-external`, `AZURE_TENANT_ID`, `AZURE_CLIENT_ID` (from
   pipeline-identity output).

### Phase 1.5 — AOAI root robustness (definite; backend-only, IaC-independent, per D7)

6a. **Unify the AOAI root onto the hardened transport.** The root iteration loop must use
    the same hardened Azure client as the primitives (retries=6, `@with_429_backoff`,
    api-version `2024-10-21`, `temperature=0`/`max_tokens`). Two routes — pick the cleaner at
    implementation: (i) thread `AZURE_OPENAI_API_VERSION` + a retry/backoff config through
    `models.py::_inject_azure_kwargs` into rlm `backend_kwargs`; or (ii, canonical) register a
    backend-owned Azure client into rlm `get_client` so root + primitives share one transport.
    Files: `backend/agents/rlm/models.py`, `backend/agents/rlm/run.py`,
    `backend/services/context/workspace/tools/azure_openai_client.py`. Eliminate the api-version
    drift between the two clients.
6b. **Un-gate the Azure accelerator for navigation.** Build an `azure_openai` rlm sub-backend in
    the `other_backends` override (`run.py:~1906`) so `rlm_query`/`llm_query` can offload to AOAI
    under the default `scope=navigation` (today Azure is excluded → no-op). Keep `=endpoint`
    rejecting AOAI URLs (bearer+`/v1` shape) with a clear error.
6c. **Tests:** rlm-root AOAI transport (retries, api-version, a real mocked completion); the
    accelerator-navigation Azure path; api-version-consistency guard across both clients.

### Phase 2 — All-Azure orchestrator (Key Vault + AOAI root + in-AKS reasoning)

7. **Key Vault module** (`modules/keyvault.bicep`): vault + two secrets placeholders
   (AOAI key, Anthropic key); RBAC `Key Vault Secrets User` → the orchestrator identity.
   Secrets injected out-of-band (never in git/params).
8. **Orchestrator deployment (D5 — confirm host first):** image build (reuse the repo Docker
   image) + an in-AKS `Deployment` (interactive) / `CronJob` (scheduled) Helm template,
   annotated with workload identity, env from Key Vault via CSI Secret Store or
   `azure-keyvault` init, `--model azure --sandbox azure`.
8a. **Opt-in AOAI agent-runtime (D8).** Implement an `AzureOpenAiAgentRuntime` satisfying the
    `AgentRuntime` Protocol (`backend/agents/runtime/base.py:159` — `provider_name` +
    `async run_agent(*, agent, user_input) -> AsyncIterator[StreamEvent]`). Cheapest route: an
    Azure-client swap on the existing `OpenAiAgentRuntime` (follow the
    `configure_openai_agents_sdk_for_endpoint` pattern, `factory.py:340-366`) rather than a
    from-scratch runtime. Plumb `ProviderName` (`base.py:11`) + `make_runtime` (`factory.py:437`).
    **Flag-gated, default Sonnet.** The executor + BES candidates inherit it through the same
    factory. Tool-call-translation tests required; mark the path experimental (gpt-4o code
    quality unvalidated).
9. **Backend auth wiring:** verify the `--model azure` (AOAI root) path end-to-end against
   `sciartgen-azure-openai`; confirm Sonnet sub-agents read `ANTHROPIC_API_KEY` from the
   injected env; document the two-surface wiring for this topology in CLAUDE.md.
10. **Quickstart runbook:** quota → pipeline-identity → admin grant → `az stack group create`
    → `azure_build_cell_image.sh` → Helm L2 → `azure_wire_env.sh` → smoke GPU Job →
    orchestrator deploy → first `--sandbox azure` run.

## 5. Risks & long poles

- **A100-80 quota in westus3** (`StandardNCADSA100v4Family`) — hours-to-days approval;
  **file immediately**, independent of all code work. Without it the cluster deploys but
  the GPU pool never scales 0→1.
- **Funded Anthropic API key** required for Sonnet sub-agents (no-credit key = instant
  death at first sub-call). Both reasoning surfaces are per-token metered in this posture.
- **westus3 A100 capacity** can be regionally constrained even with quota; eastus2 /
  southcentralus are fallbacks (changes `location` + `authorizedIpRanges` only).
- **kubernetesVersion** must be pinned to one available in westus3 (`az aks get-versions`);
  can't be resolved offline — left as a documented placeholder.
- **gpt-4o code-writing quality unvalidated** for this harness (`azure-gpt-4o` `paper_validated=False`).
  The opt-in AOAI sub-agent runtime (D8) must stay default-off; flipping the executor to gpt-4o
  risks reproduction-fidelity regressions vs Sonnet. Validate on the SDAR baseline before any default flip.
- **AOAI throttling** (429s on standard/PTU deployments) is the main root-loop failure mode; D7's
  retry+backoff unification is what makes unattended AOAI runs survivable.

## 6. Open items

- **D5 orchestrator host** — confirm in-AKS vs Container Apps vs VM at Phase 2 start.
- Whether to keep the deprecated L0 `main.bicep` tfstate params at all, or remove the
  conditional entirely (leaning: keep the param, default `false`, undocumented — minimal churn).
