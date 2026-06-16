# Azure AKS GPU backend — standup runbook & handoff

- **Date:** 2026-06-03
- **Pairs with:** `docs/superpowers/specs/2026-06-03-azure-aks-gpu-backend-design.md` (the *why* and the decision table). This doc is the *how* — stand it up, run it, debug it.
- **Status:** Design locked, implementation not started. This runbook is written ahead of the code so the **quota request (the critical-path blocker) can start immediately**.

---

## 0. Prerequisites

**Tools (local box):** `az` CLI (logged in: `az login`), `terraform` ≥ 1.6, `kubectl`, `helm` ≥ 3.12, the repo venv (`.venv/bin/python`).

**Access (from DeepInvent):**
- Subscription ID + a role that can create RG/AKS/ACR/storage + **assign roles** (Owner or Contributor+User Access Administrator — workload identity needs role assignments).
- Confirmation that a **public AKS API server with authorized-IP-ranges is permitted** (see spec §9).
- A **per-run / monthly cost ceiling** (sets `max_run_gpu_usd` and node-pool `max`).

---

## 1. ⚠️ FIRST: GPU quota (do this before any code — it gates everything)

A fresh subscription has **0** A100 quota. Lead time is hours→days.

```bash
SUB=<subscription-id>; REGION=eastus2          # pick region by A100-v4 availability, not latency
az account set --subscription "$SUB"

# What A100-v4 quota do we have right now? (expect 0 on a fresh sub)
az vm list-usage --location "$REGION" -o table | grep -i "NCADS\|A100"
```

**Request the increase** for family **`Standard NCADSA100v4 Family vCPUs`** (each `NC24ads_A100_v4` node = 24 vCPUs, so request ≥ 24 × desired max nodes):
- **Portal (reliable):** Subscriptions → *Usage + quotas* → filter `NCADSA100v4` → *Request increase*.
- **CLI (Microsoft.Quota):**
  ```bash
  SCOPE="/subscriptions/$SUB/providers/Microsoft.Compute/locations/$REGION"
  az quota show --resource-name standardNCADSA100v4Family --scope "$SCOPE" -o table
  # then az quota update … (GPU bumps often still route to a support ticket)
  ```
- If region has no stock, try `eastus`, `westus2`, `westus3`, `southcentralus`, `westeurope`.

**Record the ticket # in this file when filed.** Until granted, every real-GPU gate stays blocked (the wiring can still be validated on CPU — see §6).

---

## 2. Phase 0 — remote-state bootstrap (chicken-and-egg)

The Terraform state backend storage account must exist before `terraform init` can use it.

```bash
cd infra/azure/bootstrap
terraform init && terraform apply      # creates RG + storage account + "tfstate" container
# note the outputs (storage account name, container) → infra/azure/backend.tf
```

---

## 3. Phase 1a — provision infra (L1) + scaffold (L2), then smoke

```bash
cd infra/azure
terraform init -backend-config=envs/deepinvent/backend.hcl
terraform plan  -var-file=envs/deepinvent/main.tfvars
terraform apply -var-file=envs/deepinvent/main.tfvars     # AKS + GPU pool(min=0) + ACR + storage + identity

az aks get-credentials --resource-group <rg> --name <aks-name>
kubectl get nodes                                          # system pool only; GPU pool shows 0 nodes

# L2 scaffold
helm upgrade --install reprolab-aks infra/azure/helm \
  -n reprolab --create-namespace -f envs/deepinvent/helm-values.yaml
kubectl -n reprolab get sa,pvc,resourcequota
```

**Build & push the base image:**
```bash
az acr build -r <acr-name> -t aks-cell-base:latest docker/aks-cell-base/
```

**GATE — hello-GPU smoke** (proves: pool scales 0→1, GPU visible, Blob writable via workload identity, pool scales back to 0):
```bash
kubectl -n reprolab apply -f infra/azure/helm/smoke/hello-gpu-job.yaml   # nvidia-smi → write marker to Blob
kubectl -n reprolab wait --for=condition=complete job/hello-gpu --timeout=900s
az storage blob list --account-name <sa> --container-name <c> --prefix smoke/ -o table  # marker present
# ~10 min after completion: kubectl get nodes shows GPU pool back to 0
```
Do not proceed until this passes.

---

## 4. Phase 1b — one SDAR cell end-to-end

`.env` (or shell) needs the `OPENRESEARCH_AZURE_*` block (resource group, region, storage account, blob container, files share, ACR login server, AKS cluster, namespace, SA, node pool, per-GPU VRAM=80, max nodes, base image). See spec §4b.

```bash
# NOTE the shell-vs-.env precedence pitfall (CLAUDE.md): unset stale keys so .env wins.
env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY \
  .venv/bin/python -m backend.cli reproduce 2605.15155 \
    --mode rlm --model claude-oauth --sandbox azure \
    --max-usd 20 --max-wall-clock 50400 \
    --project-id sdar_azure_smoke_$(date +%s)
```
(Smallest-two scope via `OPENRESEARCH_BASELINE_EXTRA_GUIDANCE` — see `docs/runbooks/2026-05-23-sdar-baseline-handoff.md`. Walltime 50400s = 14h per operator preference.)

**GATE:** one cell's `metrics.json` lands in `runs/<id>/code/outputs/<id>/<cell>/metrics.json` and `aggregate_cell_metrics` folds it.

## 5. Phase 1c — full smallest-two matrix + regression guard

Run the smallest-two matrix on `--sandbox azure` to a scored `final_report.json` under budget.

**GATE (overall success-check):** scored Azure run **AND** `--sandbox local` + `--sandbox runpod` smoke runs still pass unchanged.

---

## 6. Validating wiring WITHOUT quota (fallback while the ticket is pending)

Prove L1/L2/1b plumbing on CPU: point the cell Job at a CPU node pool with a **fake-GPU stub** entrypoint that writes a canned `metrics.json` to Blob. This exercises Blob bus + Files cache + Job watch + artifact download + `aggregate_cell_metrics` — everything except the real GPU. The real-GPU gate stays **open and reported as blocked-on-quota**, never green.

---

## 7. Troubleshooting

| Symptom | Likely cause → fix |
|---------|--------------------|
| Job stuck `Pending`, event `insufficient nvidia.com/gpu` | Autoscaler scaling up (wait ~3–5 min) **or** quota not granted / no stock. `kubectl -n reprolab describe pod <p>`. Past the orchestrator's pending-timeout → surfaces as `capacity_exhausted`. |
| `ImagePullBackOff` | AcrPull not bound to kubelet identity, or wrong ACR login server in settings. `az aks check-acr`. |
| Job runs but can't read/write Blob | Workload identity broken: SA missing `azure.workload.identity/client-id` annotation; federated-credential **subject mismatch** (must be `system:serviceaccount:<ns>:<sa>`); MI lacks `Storage Blob Data Contributor` on the container. |
| Files PVC won't mount | Azure Files CSI driver not enabled on the cluster, or storage class isn't RWX. |
| GPU pod scheduled on CPU node / vice-versa | Missing taint `nvidia.com/gpu=true:NoSchedule` on the pool, or the Job lacks the matching toleration + `nodeSelector`. |
| Node never scales to 0 | Lingering pod on the GPU node (DaemonSet is fine); check cluster-autoscaler scale-down delay (default ~10 min). |
| `--sandbox azure` falls back to Docker with a warning | `azure` not added to `SandboxMode` / `_backend_for_sandbox_mode` (spec §4b not done). |

---

## 8. Cost notes

- `NC24ads_A100_v4` (1× A100 80GB) ≈ **$3.6–3.9/hr** on-demand pay-as-you-go (US; confirm DeepInvent's negotiated rate). **Scale-to-zero ⇒ you pay only during runs + the scale-down delay.**
- System CPU pool runs 24/7 — keep it tiny (e.g. `Standard_D2s_v5`, ~$0.10/hr).
- ACR (Standard) + Blob + a small Files share: a few $/month.
- The orchestrator's `max_run_gpu_usd` cap is the per-run guardrail; the node-pool `max` is the parallelism (and cost) ceiling.

---

## 9. Next-session resume prompt

> Implementing the Azure AKS GPU backend per `docs/superpowers/specs/2026-06-03-azure-aks-gpu-backend-design.md`. Design is locked (decision table §1). Start by confirming the change-map (§4) against the live tree, then implement in phase order (§11): Phase 1a Terraform/Helm (`infra/azure/`) + base image, gated on the hello-GPU smoke; then Phase 1b the `k8s_job_cell_runner.run_matrix` drop-in (identical signature/return to `gpu_cell_runner.run_matrix`, §5) + the four wiring edits (`SandboxMode`, `cli.py:1605` choices, `config.py` Literals+`azure_*` block, `_backend_for_sandbox_mode`, `_execute_cell_matrix` runner-select, `_describe_azure` fill, `build_environment` no-op). Keep `local`/`runpod` byte-for-byte unchanged — that's the acceptance bar. ⚠️ Check GPU quota status first (§1); if still pending, validate wiring on the CPU stub (§6) and leave the real-GPU gate open. Commit as `lolout1`, no co-author trailer.

---

## Cross-references

- Spec: `docs/superpowers/specs/2026-06-03-azure-aks-gpu-backend-design.md`
- Cell-matrix model: `docs/superpowers/specs/2026-05-31-oom-gpu-capacity-remediation-design.md`
- SDAR smallest-two scope + run command: `docs/runbooks/2026-05-23-sdar-baseline-handoff.md`
- Shell-vs-`.env` precedence + auth pitfalls: `CLAUDE.md` (RLM auth section).
