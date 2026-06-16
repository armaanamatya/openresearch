# Azure AKS GPU execution backend — design

- **Date:** 2026-06-03
- **Status:** Design locked (grilled), implementation not started
- **Driver:** Client deliverable for **DeepInvent** (their Azure tenant). Goals, in priority order: **scale** (run more cells in parallel than the local 8×A5000 box or a single RunPod pod), **cost predictability**, and **production polish** (a clean, reviewable, transferable IaC artifact).
- **One-line:** Fill the Azure-shaped hole the codebase already left (`SandboxMode`, `_backend_for_sandbox_mode`, `gpu_capacity._describe_azure`) with a **Job-native AKS GPU backend** selected by `--sandbox azure`, driven from the existing local orchestrator, leaving `local`/`runpod`/`docker` byte-for-byte untouched.

Read alongside `CLAUDE.md` (§ Sandboxes, § One-GPU-per-cell execution) and `docs/superpowers/specs/2026-05-31-oom-gpu-capacity-remediation-design.md` (the cell-matrix model this extends). Operational standup steps live in the sibling runbook `docs/runbooks/2026-06-03-azure-aks-gpu-backend-handoff.md`.

---

## 1. Decisions (locked via grill, 2026-06-03)

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | Driver | DeepInvent client deliverable — scale + cost + prod polish | Not institutional credits; this is a contracted artifact. |
| 2 | Execution model | **Job-native cells**, dynamically selectable per run | Multi-hour training must not be tied to a streamed connection. `local`/`runpod` stay selectable and unchanged. |
| 3 | Scope (Phase 1) | **Defer app hosting.** Cluster + Job backend only; orchestrator stays on the local box | Fastest path to a working end-to-end Azure run; sidesteps file-backed `runs/` + SQLite on a shared volume. |
| 4 | Data plane | **Azure Blob** as the code/artifact bus + **Azure Files (RWX)** as `HF_HOME` *and* pip cache | Decoupled, scalable, survives orchestrator disconnects; weights download once across the whole matrix. |
| 5 | GPU pool | **One scale-to-zero node pool, on-demand**, A100-80GB-class (`Standard_NC24ads_A100_v4`) | One quota ask, one pool, no resolver changes; 80 GB fits 1.7B/3B/7B. Idle cost = 0. |
| 6 | IaC | **Terraform**, remote state in Azure Storage, modular; **3-layer split** | Standard, portable, reviewable client artifact. |
| 7 | Image | **Pre-baked base in ACR + runtime `pip install`**, cached on Files | Mirrors the runpod-bootstrap / local-venv pattern; `build_environment` under `azure` ≈ no-op. |
| 8 | Auth | **Workload Identity** (AAD federation), zero static secrets in-cluster | Correct production handoff posture; first thing a reviewer checks. |
| 9 | OOM retry | **In-Job self-retry** (shrink ladder inside the Job); `backoffLimit` for infra | Most faithful to current `run_matrix` semantics; fewest orchestrator↔cluster round-trips. |

**Explicit non-goals / deferred to Phase 2:** moving the FastAPI/Next.js control plane onto AKS; a multi-SKU node-pool *ladder* with dynamic per-cell SKU selection (the Azure analogue of `gpu_catalog.py`); Spot GPU nodes; a private API server.

---

## 2. Architecture — end to end

The orchestrator (RLM root + primitives) keeps running **locally**, exactly as today. The only thing that changes is *where the training matrix executes* when `--sandbox azure`.

```
LOCAL BOX (orchestrator, unchanged)                 AZURE (DeepInvent tenant)
─────────────────────────────────                  ────────────────────────────────────────
  cli reproduce … --sandbox azure
      │
      ▼
  run_experiment → _execute_cell_matrix
      │  (capacity_gate, dataset_preflight — unchanged)
      │
      ├─ upload code/ (+cells.json, train_cell.py) ─────────►  Blob:  runs/<id>/code/
      │
      ▼
  k8s_job_cell_runner.run_matrix(...)   ── K8s API ──►  AKS  ──► for each cell: submit Job
      │   (drop-in for gpu_cell_runner.run_matrix)         │       (nvidia.com/gpu: 1)
      │                                                     ▼
      │                                            GPU node pool autoscales 0→N
      │                                                     │  Job pod:
      │                                                     │   1. pull code from Blob
      │   watches Job status + tails logs ◄── K8s API ──    │   2. mount Files PVC (HF_HOME+pip cache)
      │   (emits existing SSE events)                       │   3. pip install deltas
      │                                                     │   4. run train_cell.py  ← reused verbatim
      │                                                     │   5. on CUDA OOM: self-re-exec, batch 0.5→0.25
      │   download metrics.json  ◄──────── Blob ───────────┤   6. push metrics.json/logs → Blob
      ▼                                                     ▼
  runs/<id>/code/outputs/<id>/<cell>/metrics.json      Job completes → TTL cleanup → node scales to 0
      │
      ▼
  cell_matrix.aggregate_cell_metrics(...)  ← UNCHANGED
      │
      ▼
  leaf scorer → final_report.{json,md}     ← UNCHANGED
```

The expensive, must-be-robust part (training) runs **decoupled** from any streamed connection: if the local orchestrator's laptop sleeps, the Jobs keep running and the orchestrator re-attaches by watching Job status on next poll.

---

## 3. The 3-layer split (why ephemeral Jobs are NOT in the IaC)

| Layer | What | Managed by | Lifecycle |
|-------|------|-----------|-----------|
| **L1 — Azure infra** | Resource group, VNet/subnet, AKS, GPU node pool, ACR, storage (Blob + Files), managed identity + role assignments, remote state | **Terraform** | Provisioned once, rarely changes |
| **L2 — in-cluster scaffold** | Namespace, ServiceAccount (workload-identity-annotated), Files PVC, orchestrator RBAC, NVIDIA device plugin, ResourceQuota | **Helm/kustomize** (applied once) | Per-cluster, semi-static |
| **L3 — runtime** | The per-cell K8s **Jobs** | **orchestrator code** (`k8s_job_cell_runner`, Python K8s client) | Thousands, transient |

Terraform managing transient Jobs is an anti-pattern (state churn, drift). The boundary is deliberate.

---

## 4. Component & file change-map

*All paths/lines verified against the tree on 2026-06-03; re-confirm at implement-time.*

### 4a. New files

| File | Role |
|------|------|
| `backend/agents/rlm/k8s_job_cell_runner.py` | **`run_matrix(...)` drop-in** — submits one K8s Job per cell, watches status, pulls per-cell `metrics.json` from Blob, returns the identical `{cell_id → result_dict}` shape with the same status vocabulary (`ok` / `oom_failed` / other). |
| `backend/services/runtime/aks_job_backend.py` | `AksJobBackend(RuntimeBackend)` — satisfies the generic 5-method interface (used by non-matrix calls: env build no-op, smoke tests) via short K8s Jobs / the K8s API. |
| `backend/services/runtime/azure_blob.py` | Thin Blob helper (upload code prefix, download artifacts) using `DefaultAzureCredential`. |
| `infra/azure/` (Terraform L1) | Modules + envs (see §6). |
| `infra/azure/helm/` (L2) | Namespace, SA, PVC, RBAC, device-plugin values. |
| `docker/aks-cell-base/` | Base image Dockerfile (CUDA+PyTorch+vLLM+transformers) → ACR. |
| Job entrypoint wrapper (shipped in base image) | Blob-pull → mount cache → pip install → run `train_cell.py` → **in-Job OOM shrink loop** → Blob-push. |

### 4b. Modified files (surgical)

| File:line | Change |
|-----------|--------|
| `backend/agents/execution.py:32` | Add `azure = "azure"` to `SandboxMode`. Add `ensure_sandbox_mode_available` / `resolve_sandbox_mode` handling (mirror runpod). |
| `backend/cli.py:1605` | Add `"azure"` to `--sandbox` choices (it is **not** there today). |
| `backend/config.py:130,137` | Widen `default_sandbox` / `force_sandbox` Literals to include `"azure"`. |
| `backend/config.py:~186` | Add `azure_*` settings block (mirror the `runpod_*` block): resource group, region, storage account, blob container, files share, ACR login server, AKS cluster, namespace, service account, node pool name, per-GPU VRAM (80), max nodes, base image, boot/pending timeout. |
| `backend/agents/rlm/primitives.py:1926` (`_backend_for_sandbox_mode`) | Add `azure` branch → `AksJobBackend(...)`. |
| `backend/agents/rlm/primitives.py:3663` (`_execute_cell_matrix`) | Select the runner module by `ctx.sandbox_mode`: `azure` → `k8s_job_cell_runner.run_matrix(...)`, else `gpu_cell_runner.run_matrix(...)`. **Identical args** (the `gpus` param is ignored by the K8s runner; `per_cell_timeout_s`→`activeDeadlineSeconds`, `max_oom_retries`→in-Job shrink count). |
| `backend/services/runtime/gpu_capacity.py:176` (`_describe_azure`) | Replace `NotImplementedError` with a settings-driven descriptor: `GpuCapacity("azure", num_gpus=<pool max>, per_gpu_vram_gb=80.0, free_gpu_ids=tuple(str(i)…), can_escalate=False)`. Note `_backend_kind` (line 107) already routes `"azure"` here. |
| `build_environment` primitive | Add an `azure` short-circuit (no-op success), mirroring the existing `local` short-circuit (image is pre-baked). |

### 4c. Untouched — the "don't break anything" guarantee

`local_process.py`, `local_docker.py`, `runpod_backend.py`, `gpu_cell_runner.py`, `cell_matrix.py` (`capacity_gate`, `dataset_url_preflight`, `aggregate_cell_metrics`), `train_cell.py`, all SSE event types, `sse_bridge.py`, forced-iteration + rubric-guard machinery. A `--sandbox local` or `--sandbox runpod` run takes a code path with **zero** new branches on its critical line.

---

## 5. The cell-runner contract (the seam that makes this clean)

`gpu_cell_runner.run_matrix` signature (verified, `gpu_cell_runner.py:300`):

```python
def run_matrix(cells, cell_script, *, output_root,
               gpus=None, max_parallel=None, max_oom_retries=2,
               per_cell_timeout_s=None) -> dict[str, dict]:  # {cell_id → CellResult.to_dict}
```

`k8s_job_cell_runner.run_matrix` implements the **same signature + same return contract**:

- **`gpus`** → ignored (the K8s scheduler places). `max_parallel` → orchestrator-side concurrency cap + namespace `ResourceQuota`.
- **`per_cell_timeout_s`** → Job `activeDeadlineSeconds`.
- **`max_oom_retries`** → passed to the Job as `OPENRESEARCH_CELL_MAX_OOM_RETRIES`; the **in-Job wrapper** runs the shrink ladder (decision #9). The orchestrator does **not** resubmit on OOM.
- **Status mapping:** Job Succeeded → `"ok"`; wrapper exits `oom_shrink_exhausted` → `"oom_failed"`; Job failed/timeout/**stuck-Pending past timeout** → other (→ `_execute_cell_matrix` already counts `n_err`).
- Per-cell `metrics.json` is pulled from Blob into `output_root/<cell_id>/metrics.json` so `aggregate_cell_metrics` reads it exactly as in the local path.

Because the return shape and status vocabulary match, **every downstream consumer is unchanged**.

---

## 6. Terraform (L1) module tree

```
infra/azure/
  bootstrap/            # one-time: RG + storage account + container for remote state
  modules/
    network/            # VNet, subnet, NSG
    aks/                # cluster, OIDC issuer + workload identity ENABLED, system (CPU) pool
    gpu_nodepool/       # NC24ads_A100_v4, autoscale min=0 max=N, on-demand, taint nvidia.com/gpu
    acr/                # registry + AcrPull role assignment to kubelet identity
    storage/            # blob container (artifact bus) + files share (RWX cache)
    identity/           # user-assigned MI, federated credential (SA↔MI), Storage Blob Data Contributor
  envs/
    deepinvent/         # *.tfvars: region, sizes, max nodes, names
  backend.tf            # remote state → bootstrap storage account
```

**L2 (Helm/kustomize)** applies: namespace, `ServiceAccount` annotated `azure.workload.identity/client-id`, the Files-CSI `PersistentVolumeClaim` (RWX), `Role`/`RoleBinding` for the orchestrator (create/get/list/watch/delete Jobs + read pod logs), the NVIDIA device plugin DaemonSet (tolerating the GPU taint), and a `ResourceQuota`.

---

## 7. Auth / identity flow (workload identity)

- **Jobs → Blob:** Job pod runs as the annotated `ServiceAccount`; `DefaultAzureCredential` picks up the federated token; the user-assigned MI holds `Storage Blob Data Contributor` on the artifact container. **No secrets in-cluster.**
- **Nodes → ACR:** AcrPull granted to the AKS kubelet identity.
- **Local orchestrator → cluster:** `az aks get-credentials` (operator's `az login`) for the kubeconfig; the K8s client uses it to submit/watch Jobs.
- **Local orchestrator → Blob:** `DefaultAzureCredential` (operator's `az login`) to upload code / download artifacts.

---

## 8. Budget + robustness mapping

| Existing mechanism | Azure realization |
|--------------------|-------------------|
| `max_run_gpu_usd`, `max_pod_seconds` (`RunBudget`) | Orchestrator tracks Σ(Job wall-clock × pool $/hr), enforces caps before submitting each cell. |
| per-cell wall-clock | Job `activeDeadlineSeconds`. |
| completed-cell cleanup | Job `ttlSecondsAfterFinished`. |
| infra failure (node death) | Job `backoffLimit` (low, e.g. 2) reschedules. |
| OOM (`oom_shrink_exhausted`) | In-Job shrink ladder → terminal `oom_failed` (no re-loop). |
| capacity exhaustion | **Stuck-Pending Job past a timeout** (pool can't scale — quota/stock) → orchestrator fails the cell → existing `capacity_exhausted` `stop_reason`. |
| live UI | Orchestrator tails Job logs via K8s API → emits existing `repl_iteration` / `primitive_call` / `run_warning` SSE events. **No new event types.** |

---

## 9. Consequences (decided by earlier choices, flagged not re-asked)

- **AKS API server = public with authorized-IP-ranges** locked to the operator IP — forced by "orchestrator is local" (a private API server would need VPN/bastion). Becomes private in Phase 2 when the app moves in-cluster. ⚠️ **Confirm DeepInvent's security policy permits a public API server.**
- **Mode name = `azure`** (not `k8s`): the deliverable is Azure-specific (AKS+ACR+Blob+Files+workload-identity). Backend class `AksJobBackend`; runner `k8s_job_cell_runner`.

---

## 10. ⚠️ Open operational risks (these, not the code, are the critical path)

1. **GPU quota is the blocker.** A fresh subscription has **0** `Standard NCADSA100v4 Family` vCPUs in any region; raising it is a support-ticket round-trip (hours→days). **Kick this off first, in parallel with implementation.** (Commands in the runbook.)
2. **Region + A100-80GB stock** varies — pick region by quota+availability, not latency.
3. **Per-run / monthly cost ceiling** — need a number from DeepInvent to set `max_run_gpu_usd` and the node-pool `max`.
4. **Public-API-server policy** — see §9.

---

## 11. Phased plan with stop conditions

> Success-check discipline (this design came in via `/iterate`): each phase has a concrete pass/fail gate. Stop at the gate; do not proceed on "should work."

- **Phase 0 — bootstrap (parallel):** (a) file the A100 quota request; (b) `az login`, create the remote-state storage account. **Gate:** `terraform init` succeeds against remote state; quota request filed (ticket #).
- **Phase 1a — Terraform L1 + Helm L2.** **Gate:** `terraform apply` produces a cluster; a hand-rolled "hello-GPU" Job (`nvidia-smi` → write a file to Blob) **scales the pool 0→1, runs, writes to Blob, and the pool scales back to 0.**
- **Phase 1b — `k8s_job_cell_runner` + routing + `_describe_azure` + settings.** **Gate:** `--sandbox azure` runs **one** SDAR cell end-to-end: code from Blob, weights cached on Files, `metrics.json` lands in local `runs/<id>/…` and `aggregate_cell_metrics` consumes it.
- **Phase 1c — full matrix + budget + in-Job OOM retry + Pending-timeout.** **Gate (overall success-check):** the **smallest-two SDAR matrix** completes on Azure, is scored into `final_report.json`, and stays under budget — **AND** a `--sandbox local` and a `--sandbox runpod` smoke run still pass unchanged (the regression guard for "don't break anything").

**Cap:** if quota is not granted within the engagement window, Phase 1a/1b can be validated on a CPU node + a fake-GPU stub Job to prove the wiring, but the real-GPU gate stays open and is reported as blocked-on-quota — not as success.

---

## 12. Cross-references

- `CLAUDE.md` — § Sandboxes, § One-GPU-per-cell execution + OOM remediation, § Dynamic GPU selection.
- `docs/superpowers/specs/2026-05-31-oom-gpu-capacity-remediation-design.md` — the cell-matrix model (`describe_capacity`, `run_matrix`, `aggregate_cell_metrics`) this extends; `_describe_azure`'s docstring points here.
- `docs/runbooks/2026-06-03-azure-aks-gpu-backend-handoff.md` — operational standup + the next-session resume prompt.

---

## 13. Recon — change-map → real paths @ HEAD (2026-06-07, post BES+harden merge)

The §4b line numbers were verified 2026-06-03 and **drifted** after the BES + harden + consolidation merges. Re-confirmed by symbol at `feat/azure-aks-gpu` (HEAD `8cddb31`), and the design was IMPLEMENTED against these:

| Seam (design §4b) | Real location @ HEAD | Notes |
|---|---|---|
| `SandboxMode` enum | `backend/agents/execution.py:32` (members docker/local/runpod/**brev**/auto) | `brev` already exists — azure is purely additive. |
| `ensure_sandbox_mode_available` | `backend/agents/execution.py:~270` | azure branch mirrors the runpod branch. |
| `--sandbox` choices | `backend/cli.py:1921` (was 1605) | was `("auto","local","docker","runpod")`. |
| `default_sandbox` / `force_sandbox` Literals | `backend/config.py:143` / `:150` | widened to include `"azure"`. |
| `runpod_*` settings block | `backend/config.py:165-199` | `azure_*` block added after it (16 fields). |
| `_backend_for_sandbox_mode` | `backend/agents/rlm/primitives.py:2206` (was 1926) | azure → `AksJobBackend(run_budget=...)`. |
| `_execute_cell_matrix` | `backend/agents/rlm/primitives.py:4094` (was 3663); `run_matrix` call @4202, import @4111 | runner-select wraps the call; non-azure call preserved byte-for-byte. |
| `_describe_azure` / `_backend_kind` | `backend/services/runtime/gpu_capacity.py:176` / `:104` | `_backend_kind` already routed "azure" here; stub replaced. |
| `gpu_cell_runner.run_matrix` (the contract) | `backend/agents/rlm/gpu_cell_runner.py:413` (was 300) | **Signature drifted**: the real one has 11 params — `cells, cell_script, *, output_root, gpus, max_parallel, max_oom_retries, per_cell_timeout_s, overall_timeout_s, gpus_per_cell, fingerprints, force_cells, now_iso`. `k8s_job_cell_runner.run_matrix` mirrors ALL of them. |
| `CellResult.to_dict` (return shape) | `backend/agents/rlm/gpu_cell_runner.py:70` | returns `{"status","metrics","gpu","retries","error"}`; cell_id is the MAP KEY. Confirmed by the W3 integration test: NO adapter needed into `aggregate_cell_metrics`. |
| `RuntimeBackend` ABC | `backend/services/runtime/interface.py:115` | 5 abstract async methods + 2 optional hooks (probe_alive/soft_recover). |

**Implementation status (2026-06-07):** all offline-buildable components landed on `feat/azure-aks-gpu` with a full mocked test suite (azure_blob, k8s_job_cell_runner, aks_job_backend, in-Job entrypoint, wiring, gpu_capacity, config, + cross-module integration). Live gates (terraform apply, ACR build, hello-GPU, live SDAR cell, `local`/`runpod` regression *smoke runs*) remain **blocked-on-tooling/quota** (no az/terraform/kubectl/helm/docker-daemon/Azure-creds in the build env) and are reported as such, never green. One design refinement: `azure_gpu_usd_per_hour` defaults to the real NC24ads_A100_v4 list price (3.67) so the run-USD cost cap is active by default (a 0 default would silently disable it).

---

## 14. Dynamic refinement (2026-06-07) — re-opened decision #5 (single static pool)

Decision #5 (one fixed `NC24ads_A100_v4` pool, "no resolver changes") and the §4c "gpu_cell_runner untouched" bar were deliberately re-opened (grilled) so the Azure backend is **dynamic** and **reuses existing logic** instead of forking a static parallel path. Locked + implemented:

- **Cloud-aware catalog/resolver (D1).** `gpu_catalog.GpuSku` gains `provider` (default `runpod` → existing rows byte-identical) + `gpu_count`; Azure NC-series rows use **provider-distinct** short_names (`azure_a10_24`/`azure_a100_80`/`azure_a100_80x2`/`azure_a100_80x4`) so `GPU_PRICING` attributes azure cost correctly. `find_ladder(..., provider=)` + `gpu_resolver.resolve(..., provider=)` are cloud-parameterized (runpod default unchanged). `resolve_gpu_requirements` selects the cloud from `ctx.sandbox_mode`. `GpuPlan`/`GpuPriceEntry` gained `ONDEMAND`.
- **Plan-threaded backend.** `_backend_for_sandbox_mode` now passes `gpu_plan` to `AksJobBackend` (was dropped); `_execute_cell_matrix` loads `rlm_state/gpu_plan.json` and binds it via `bind_run_context(gpu_plan=)`. The k8s Job manifest targets the plan's pool (`nodeSelector reprolab/sku=<short_name>`, `nvidia.com/gpu=<gpu_count>`). `_describe_azure` is plan-driven (settings fallback, fail-soft).
- **SKU escalation.** In-Job batch-shrink first; on `oom_failed` the k8s runner escalates up `plan.ladder_remaining` ∩ `azure_gpu_skus` (bounded by `dynamic_gpu_max_escalations`, emits the existing `gpu_escalated`); empty ladder / unprovisioned SKU / cap → graceful-degrade to terminal `oom_failed` (no loop).
- **Parameterized pools (re-opened #5).** Terraform replaces the single pool with `var.gpu_skus` (list) → one scale-to-zero pool each via `for_each`; **default = single A100-80 = one quota ask**. Adding NC48/NC96/A10 pools (each its own quota) enables the escalation ladder with no code change.
- **De-fork (D2).** Shared scheduler logic (resume-skip, manifest write, status vocab, wave/deadline math, `CellResult`) extracted to pure `backend/agents/rlm/cell_scheduler.py`; both `gpu_cell_runner` and `k8s_job_cell_runner` call it. Bar met: **behavior-identical, full suite 4281 passed / 0 regress** (local/runpod unchanged).

Built Codex-planned (Codex hit its usage limit mid-plan; the authored brief served as the plan) + Sonnet-fanned (catalog/resolver/config/capacity/infra/runner/primitives) + Opus-integrated. Live gates remain blocked-on-quota/tooling as in §13. ⚠ A concurrent rubric-scoring-fairness workstream shares the working tree (`env_pin`/`provenance`/`execution_smoke`/`leaf_scorer`) — left uncommitted/untouched.
