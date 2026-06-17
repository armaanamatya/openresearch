# SDAR-on-Azure — LIVE end-to-end test handoff (post-hardening)

> **Purpose:** the single doc to actually *run the first real GPU test* of the
> `--sandbox azure` path with the SDAR paper, with all underlying context. Reflects
> the **2026-06-14 hardening pass** (a P0 + 7 fixes, offline-validated) — supersedes
> the pre-hardening command notes in `2026-06-14-sdar-on-azure-run.md` where they differ.
> **Date:** 2026-06-14 · **Branch:** `feat/azure-bicep-canonical-aoai-hardening` · HEAD `5a530617`
> **Pairs with:** `…-session-handoff.md` (dev continuation), `…-sdar-on-azure-run.md`
> (operator command ref), `2026-06-13-azure-aionic-deploy-handoff.md` (IaC), and
> `2026-05-23-sdar-baseline-handoff.md` (the paper).

---

## 0. TL;DR

- **What:** first real GPU run of `--sandbox azure` — SDAR (arXiv 2605.15155),
  smallest-two (Qwen3-1.7B + Qwen2.5-3B-Instruct, seed 42), GRPO baseline + SDAR
  proposed, ~32-task slice/env, 1× A100-80GB, blob-only, capped ~$15–25.
- **Topology:** local reasoning loop (claude-oauth root + Sonnet-OAuth executor) on
  WSL; only the GPU **training cells** become one-Job-per-cell AKS Jobs.
- **Code state:** hardened + offline-validated (§7). The cell-route now actually
  fires for azure (it did **not** before — see §6 P0).
- **Live state (verified 2026-06-14, §1):** **RED — cannot run yet.** The whole
  Azure stack is torn down; 8/11 preflight checks fail. The local half is ready
  (`claude --print ping` → `pong`). Clearing the gates (§3) is required first.
- **Success for THIS test** = a real GPU Job runs a cell and `metrics.json` returns
  via Blob, end-to-end. **Not** a rubric pass (that's the paper's own difficulty).

---

## 1. Current live state (captured 2026-06-14, `scripts/azure_sdar_preflight.sh`)

```
[OK]   tool: az / kubectl / jq
[WARN] kubelogin missing  → az aks install-cli
[OK]   az logged in (sub 51008c59-…)
[FAIL] kubectl cannot reach cluster        → cluster torn down; redeploy + get-credentials
[FAIL] GPU quota too low: NCADS_A100_v4 = 0 (< 24)   → quota ticket (admin)
[FAIL] RBAC: cannot create jobs in reprolab          → AKS RBAC grant (admin)
[FAIL] namespace reprolab missing / SA reprolab-sa missing  → bootstrap (after redeploy)
[FAIL] no node pool labelled reprolab/sku=azure_a100_80     → redeploy
[FAIL] OPENRESEARCH_AZURE_BASE_IMAGE unset                  → build+pin the cell image
[FAIL] storage account sciartgenreprolab not found         → redeploy (stack-managed)
[OK]   python deps: kubernetes, azure-identity, azure-storage-blob
[OK]   claude-oauth root surface responds
== 8 fail(s), 1 warn(s) == RED
```

**Reading:** everything Azure (cluster, GPU pool, storage/blob bus, namespace) went
away with the `openresearch-l1` stack teardown on 06-13. The local reasoning surface
is fully ready. So the gating work is **operator/admin infra**, not code.

---

## 2. What happens end-to-end (the underlying flow)

```
WSL (operator box, local)                       Azure (AIONIC, westus3)
────────────────────────                        ─────────────────────────────────
backend.cli reproduce 2605.15155                AKS sciart-aks (scale-to-zero A100-80)
  --sandbox azure --model claude-oauth            ns reprolab · SA reprolab-sa (workload id)
  1. ingest SDAR → corpus offloaded as REPL var
  2. root reasoning (claude-oauth)
  3. resolve_gpu_requirements → rlm_state/gpu_plan.json (azure_a100_80, gpu=1)
  4. build_environment → NO-OP for azure (image pre-baked in ACR)
  5. implement_baseline (Sonnet-OAuth) → code/ incl.
       cells.json (the model×env×baseline matrix) + train_cell.py (single-cell trainer)
  6. run_experiment → CELLS ROUTE (the P0 fix, §6) ───────────► one GPU Job per cell:
        ├─ upload code/ ── Blob (reprolab-artifacts) ─────────►  reprolab-cell:<sha> (ACR)
        ├─ submit K8s Job (manifest in §8 Appendix)              aks_cell_entrypoint.py:
        │   nodeSelector reprolab/sku=azure_a100_80                · pull code from Blob
        │   nvidia.com/gpu=1 · emptyDir cache (blob-only)          · HF cache → emptyDir
        ├─ watch Job to terminal                                  · run train_cell.py
        └─ download metrics.json ◄── Blob ◄──────────────────────  · upload metrics+logs+status
  7. aggregate_cell_metrics → per_model[mk][env][baseline] leaf → code/metrics.json
  8. verify_against_rubric → rubric score
  9. final_report.{json,md} + cost_ledger.jsonl + dashboard_events.jsonl
```

**Where to watch it (the underlying content of a live run):**
`kubectl get jobs,pods -n reprolab` · `runs/<id>/code/metrics.json` ·
`runs/<id>/dashboard_events.jsonl` · `runs/<id>/cost_ledger.jsonl` · the tee'd
`runs/sdar-azure-<epoch>.log` · `scripts/azure_sdar_monitor.sh` ties these together.

---

## 3. Prerequisites — clear these before the test can run

The launcher (`azure_sdar_run.sh`) refuses to start while preflight is RED, so **all
of these must be green first.**

| # | Gate | Who | How |
|---|------|-----|-----|
| 1 | **Redeploy the stack** (restores cluster, GPU pool, ACR, storage/blob, identities) | operator (has RG Contributor) | §4 redeploy command |
| 2 | **AKS RBAC** `Azure Kubernetes Service RBAC Cluster Admin` on `sciart-aks` | **subscription admin** (operator cannot self-assign) | §4 RBAC command |
| 3 | **GPU quota** `Standard NCADS_A100_v4 Family vCPUs` ≥ 24 in westus3 (now 0/0) | **admin** (support ticket — has lead time) | Portal → Subscriptions → AIONIC → Usage + quotas → filter NCADS_A100_v4 → Request increase |
| 4 | **kubelogin** installed | operator | `az aks install-cli` |
| 5 | **Local blob data-plane RBAC** — *new, easy to miss* | operator/admin | §4 blob-RBAC command |
| 6 | **Cell image built + pinned** in `.env.azure` | operator | `scripts/azure_build_cell_image.sh sciartacr` |

**#5 is a hardening-surfaced requirement (§6).** The local orchestrator uploads `code/`
and downloads `metrics.json` to Blob **as your `az login` identity** (DefaultAzureCredential
→ AzureCliCredential), *not* as the cell workload MI. Without `Storage Blob Data
Contributor` on the container for **your** identity, the local upload 403s **before any
Job is submitted**, and the preflight now **hard-fails** on it (was a soft warn). The
cell Pods are a *separate* identity (`sciart-workload-mi`) that *also* needs the role.

---

## 4. Gate-clearing commands

```bash
# --- 1. Redeploy (operator) — restores all 22 resources incl. storage + GPU pool ---
az login --use-device-code
az account set --subscription 51008c59-ebf4-4699-8f2c-896724144d42
EGRESS=$(curl -s https://ipinfo.io/ip)
az stack group create \
  --name openresearch-l1 --resource-group rg-sciartgen-external \
  --template-file infra/azure/bicep/infra.bicep \
  --parameters infra/azure/bicep/infra.bicepparam \
  --parameters kubernetesVersion=1.34 authorizedIpRanges="[\"${EGRESS}/32\"]" \
               deployKeyVault=false deployStorageKeyOperatorRole=false \
  --deny-settings-mode none --action-on-unmanage detachAll --yes

# --- 2. AKS RBAC (subscription ADMIN runs this; operator objectId 5709e8a6-…) ---
SUB=51008c59-ebf4-4699-8f2c-896724144d42
az role assignment create \
  --assignee 5709e8a6-5c55-4628-a828-23529030815c \
  --role "Azure Kubernetes Service RBAC Cluster Admin" \
  --scope /subscriptions/$SUB/resourceGroups/rg-sciartgen-external/providers/Microsoft.ContainerService/managedClusters/sciart-aks

# --- 5. Local blob data-plane RBAC (your az-login identity; storage id after redeploy) ---
STG=$(az storage account show -n sciartgenreprolab -g rg-sciartgen-external --query id -o tsv)
az role assignment create \
  --assignee "$(az ad signed-in-user show --query id -o tsv)" \
  --role "Storage Blob Data Contributor" --scope "$STG"
# (The cell workload MI sciart-workload-mi needs the SAME role on $STG — the Bicep
#  identity module grants it; verify with: az role assignment list --scope "$STG" -o table)

# --- 3. GPU quota: support ticket (admin) — NCADS_A100_v4 ≥ 24, westus3. No CLI. ---
# --- 4. kubelogin ---
az aks install-cli
```

---

## 5. One-time setup → run → monitor (after §3 is green)

```bash
# setup (once per fresh deploy, in order)
az login --use-device-code
az aks get-credentials -g rg-sciartgen-external -n sciart-aks
kubelogin convert-kubeconfig -l azurecli
scripts/azure_wire_env.sh rg-sciartgen-external openresearch-l1 .env.azure
scripts/azure_build_cell_image.sh sciartacr      # prints OPENRESEARCH_AZURE_BASE_IMAGE=…
echo 'OPENRESEARCH_AZURE_BASE_IMAGE=sciartacr.azurecr.io/reprolab-cell:<sha>' >> .env.azure
scripts/azure_sdar_bootstrap_cluster.sh          # ns + cell SA + NVIDIA plugin + quota (orchestrator+Files OFF)

# run
scripts/azure_sdar_preflight.sh                  # MUST be GREEN
scripts/azure_sdar_run.sh                         # capped launcher (preflight re-runs inside; aborts if RED)
scripts/azure_sdar_monitor.sh                     # 2nd terminal; auto-detects latest run

# retry only failed cells (cheap; reuses ok cells):
scripts/azure_sdar_run.sh .env.azure --resume-cells
```

The launcher pins: `--sandbox azure --model claude-oauth --paper-hint 2605.15155
--scope-spec '{"models":["Qwen3-1.7B","Qwen2.5-3B-Instruct"],"seeds":[42]}'
--force-single-gpu --max-wall-clock 21600 --max-usd 30 --max-run-gpu-usd 25
--max-gpu-usd-per-hour 4`, with `OPENRESEARCH_AZURE_GPU_SKUS='["azure_a100_80"]'` and
`OPENRESEARCH_AZURE_FILES_CACHE_ENABLED=false`, and `unset`s `ANTHROPIC_API_KEY`/`OPENAI_API_KEY`.

**Recommended add for a robust first run:** append `--vram-gb 50` to pin the A100-80
selection deterministically (see §6 resolver note).

---

## 6. Hardening-pass changes that affect THIS test (read before running)

The 2026-06-14 audit found the path would **not have worked as committed**. What
changed (full lesson: `docs/archive/learn.md` 2026-06-14):

- **P0 — the cell route now fires for azure.** The dispatch gate in
  `run_experiment` enumerated only `("local","docker")`, so `--sandbox azure` would
  have **silently run the monolithic legacy path** (one Job of `commands.json`),
  never one-Job-per-cell `train_cell.py`. A sibling empty-`env_id` guard exempted only
  `local` and would have blocked azure even after the tuple fix (azure's no-op
  `build_environment` yields an empty `env_id`). Both fixed + guard-tested.
- **Resolver fallback → pass `--vram-gb 50`.** If the root returns a `None`/low-
  confidence VRAM estimate, the azure resolver fallback now picks the **cheapest
  *provisioned* SKU** (was hard-coding the unprovisioned `azure_a10_24` → the Job
  would target a non-existent pool and hang Pending → `capacity_exhausted`). Passing
  `--vram-gb 50` bypasses the estimate entirely and pins A100-80 — do this for the
  first run to remove the variable.
- **Local blob RBAC is now a hard preflight gate** (§3 #5) — don't be surprised by it.
- **`train_cell.py` argv parity.** The AKS entrypoint now passes `--cell-id`/`--output-dir`
  to `train_cell.py` (matching the local runner), so a trainer that reads argparse
  flags works on both backends.
- **Helm fails closed** on an empty `workloadIdentity.clientId`; the bootstrap script
  supplies it, so this only bites a hand-run `helm install`.
- **Config/cleanup:** OOM-scale config keys now actually apply; pending-timeout
  fallback 900→1500 (AKS cold-start from zero takes 10–12 min); dead env var + a no-op
  alias shim removed; the dead orchestrator Role/RoleBinding is gated off on blob-only.

---

## 7. Why we trust the path (offline validation already done — no live GPU)

The orchestration was proven end-to-end against a **simulated cluster** (the K8s API
+ Blob bus mocked via the runner's `_k8s_clients_override` / `_blob_*` seams). The
*only* thing not exercised is the real GPU training inside the cell — i.e. the paper's
own difficulty, not the harness.

- **2959 backend tests** + the full azure suite green; **ruff clean**.
- **Capstone** (`tests/agents/rlm/test_azure_cell_integration.py::TestBlobOnlyManifestDispatch`):
  the real `run_matrix` dispatch on the blob-only profile submits a manifest with an
  **emptyDir** cache (not a Files PVC), `reprolab/sku=azure_a100_80`, `nvidia.com/gpu=1`,
  the pinned ACR image, and the canonical env contract — reaching `create_namespaced_job`.
- **Integration** (`TestHappyPathEndToEnd`/`TestMixedOutcome`/`TestCapacityExhaustion`):
  `run_matrix → metrics.json on disk → aggregate_cell_metrics` produces the canonical
  `per_model[mk][env][baseline]` leaf shape across success / partial / capacity-exhausted.
- **IaC:** `helm lint` passes, fail-closed `clientId` proven, `bicep build` OK.

To re-run the offline proof yourself (no Azure needed):
`.venv/bin/python -m pytest tests/agents/rlm/test_azure_cell_integration.py
tests/agents/rlm/test_run_experiment_cell_route.py -q`.

---

## 8. Appendix — the actual GPU Job manifest a SDAR cell submits

Generated from the **production** `_build_job_manifest` on the blob-only profile for
cell `qwen2_5_3b__sdar__alfworld__s42` (the real artifact `create_namespaced_job`
receives; trimmed):

```jsonc
{ "apiVersion": "batch/v1", "kind": "Job",
  "metadata": { "name": "reprolab-cell-prj-sdar-demo-qwen2-5-3b--sdar--alfworld--s42",
                "namespace": "reprolab", "labels": {"app": "reprolab-cell"} },
  "spec": {
    "backoffLimit": 0, "activeDeadlineSeconds": 21600, "ttlSecondsAfterFinished": 3600,
    "podFailurePolicy": { "rules": [{ "action": "FailJob",
        "onExitCodes": {"containerName": "cell", "operator": "In",
                        "values": [40,41,42,43,44]} }] },   // terminal wrapper exits
    "template": {
      "metadata": { "labels": { "azure.workload.identity/use": "true",   // → Blob auth
                                "app": "reprolab-cell", "cell-id": "qwen2_5_3b__sdar__alfworld__s42" } },
      "spec": {
        "serviceAccountName": "reprolab-sa", "restartPolicy": "Never",
        "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
        "nodeSelector": { "reprolab/sku": "azure_a100_80" },             // the A100-80 pool
        "volumes": [{ "name": "reprolab-cache", "emptyDir": {} }],        // blob-only (NOT a PVC)
        "containers": [{ "name": "cell",
          "image": "sciartacr.azurecr.io/reprolab-cell:abc1234",
          "resources": { "requests": {"nvidia.com/gpu": "1"}, "limits": {"nvidia.com/gpu": "1"} },
          "volumeMounts": [{ "name": "reprolab-cache", "mountPath": "/mnt/reprolab-cache" }],
          "env": [  // the runner→entrypoint contract (no dead CELL_OUTPUT_DIR, no dup names)
            "OPENRESEARCH_CELL_ID", "OPENRESEARCH_CELL_PARAMS", "OPENRESEARCH_CELL_MAX_OOM_RETRIES",
            "OPENRESEARCH_AZURE_STORAGE_ACCOUNT", "OPENRESEARCH_AZURE_BLOB_CONTAINER",
            "OPENRESEARCH_BLOB_CODE_PREFIX", "OPENRESEARCH_BLOB_OUTPUT_PREFIX",
            "OPENRESEARCH_CACHE_MOUNT", "OPENRESEARCH_CELL_OOM_BATCH_SCALE_STEP1/_FLOOR",
            "OPENRESEARCH_BOOTSTRAP_PIP_TIMEOUT_S", "OPENRESEARCH_CELL_FINGERPRINT/_NOW_ISO" ] }] } } } }
```

The matrix (`cells.json`) the agent emits for smallest-two is the cartesian product
`{qwen3_1_7b, qwen2_5_3b} × {alfworld, search_qa, webshop} × {grpo, sdar} × seed 42`
(7B honest-omitted), one cell = one such Job.

---

## 9. Success criteria + iterate

**This test passes** when, after gates clear:
1. `kubectl get jobs -n reprolab` shows at least one **Completed** cell Job.
2. `runs/<id>/code/metrics.json` has a real `per_model[…][env][baseline]` leaf (a number).
3. `runs/<id>/final_report.json` is written (`mode: rlm`).

That proves the `--sandbox azure` path runs a real GPU cell and returns metrics via
Blob — the milestone. A low rubric score is expected on a first pass (SDAR's fidelity
is the paper's difficulty); iterate with `--resume-cells` and operator steering, not
by changing the harness.

---

## 10. Triage (symptom → cause → fix)

| Symptom | Cause → fix |
|---|---|
| Preflight: blob data-plane FAIL as current identity | your az-login id lacks `Storage Blob Data Contributor` → §4 #5 |
| `403 AuthorizationPermissionMismatch` on local code upload | same as above (the local upload is YOU, not the workload MI) |
| Pods stuck `Pending` | GPU quota 0 (§3 #3) **or** pool missing `reprolab/sku=azure_a100_80` (redeploy) **or** the resolver picked an unprovisioned SKU → pass `--vram-gb 50` (§6) |
| kubectl/helm `forbidden`/401 | AKS RBAC not granted (§3 #2) → admin assigns, re-run `kubelogin convert-kubeconfig -l azurecli` |
| `ErrImagePull` | `OPENRESEARCH_AZURE_BASE_IMAGE` tag absent / AcrPull missing → rebuild via `azure_build_cell_image.sh`, confirm AcrPull on `sciart-workload-mi` |
| Ran but only ONE monolithic Job (no per-cell Jobs) | the agent did not emit BOTH `cells.json` + `train_cell.py` → the route fell back to legacy; check `code/`. (The azure route itself is now wired — P0 §6.) |
| `partial` + `SDK success-with-no-text` | claude-oauth/SDK auth, **not** Docker → `claude login`; `unset ANTHROPIC_API_KEY` (a no-credit key silently beats OAuth) |
| `credit balance too low` at first Sonnet call | stale `ANTHROPIC_API_KEY` shadowing OAuth → `unset` it (the run script already does) |

---

## 11. Cost + teardown

A100-80 (`Standard_NC24ads_A100_v4`) $3.67/hr, billed only while cells run (scale-to-zero
idle = $0). System pool ~$0.19/hr while deployed. Per-run caps: `--max-run-gpu-usd 25`
+ `--max-gpu-usd-per-hour 4` + `--max-usd 30` total + 6 h wall-clock. **Est. $15–25 GPU.**

Teardown (stops all spend; keeps RG + AOAI + tfstate):
```bash
az stack group delete --name openresearch-l1 -g rg-sciartgen-external \
  --action-on-unmanage deleteResources --yes
```
Do **not** delete `rg-sciartgen-external` (holds AOAI + tfstate).

---

## 12. Process constraints (for whoever drives this)

Opus plans + reviews every diff; Sonnet executes; verify diffs not summaries. Commit
infrequently at milestones; no Conventional-Commits prefix; **no AI/Co-Authored-By
trailer**; author = `lolout1 / appradhann@gmail.com`. Push only to `origin`
(openresearch) on explicit request; never `replix`. Secrets → Key Vault, never
git/`.env`-in-git (`.env.azure` = resource names only, gitignored). The hardening-pass
changes are **in-tree, uncommitted** as of HEAD `5a530617`.
