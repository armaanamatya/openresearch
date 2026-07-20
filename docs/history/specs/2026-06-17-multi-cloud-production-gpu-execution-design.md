# Multi-cloud production GPU execution — Azure + GCP, unattended / spot / cost-bounded

> **Status:** design (2026-06-17). Branch `feat/azure-bicep-canonical-aoai-hardening` (current).
> **Builds on (does not replace):** `2026-06-03-azure-aks-gpu-backend-design.md`,
> `2026-06-12-azure-bicep-canonical-aionic-compute-design.md`,
> `2026-06-14-sdar-on-azure-run-design.md`, `2026-06-16-gcp-gke-execution-backend-design.md`,
> `2026-05-31-oom-gpu-capacity-remediation-design.md`.
> **Supersedes one prior decision:** the Stream-E note that `claude-oauth` "cannot run in a pod"
> (true for the rotating `~/.claude/.credentials.json`, false for a `claude setup-token`
> long-lived `CLAUDE_CODE_OAUTH_TOKEN` — see Stream B).
> **Architecture decision (this doc):** productionize the **already-~90%-built K8s cluster path**
> (AKS + GKE) into a robust, unattended, spot-cheap, cost-bounded, **symmetric** multi-cloud
> solution. **Do NOT add a single-VM IaC path** — the manual 8×A100 SSH run we just cancelled
> was a stopgap; see *Alternatives considered*.

## Goal

Run any ML-paper reproduction (SDAR is the acceptance proof) on **either Azure or GCP** GPU
infrastructure, **unattended** (no operator laptop in the loop), on **spot/preemptible** GPUs
(the dominant cost lever), with **preemption-safe resume**, a **hard USD ceiling**, and an
**identical operator contract** across both clouds. Reuse the existing IaC + the cloud-agnostic
L3 cell runner; close exactly the gaps that block "robust + unattended + cheap." Every change is
**additive and flag-gated** — existing sandboxes (`local`/`docker`/`runpod`) and the current
on-demand cluster path are byte-for-byte unchanged when the new flags are off.

## What already exists (reuse — do NOT rebuild)

| Layer | Azure | GCP | State |
|---|---|---|---|
| L0 bootstrap (state/identity) | `infra/azure/bicep/main.bicep` | `infra/gcp/bootstrap/` | complete |
| L1 IaC (net/cluster/pool/registry/storage/identity) | `bicep/infra.bicep` + 8 modules | `infra/gcp/` TF + 6 modules | complete (on-demand) |
| L1 secret store | `bicep/modules/keyvault.bicep` (+monitoring) | **none** | Azure-only ⟵ Gap (Stream A) |
| L2 Helm (ns/SA/RBAC/quota/device-plugin) | `infra/azure/helm/` (13 tmpl) | `infra/gcp/helm/` (8 tmpl) | ~85 % shared |
| L2 in-cluster orchestrator (Stream E) | `helm/templates/orchestrator-{deployment,cronjob,serviceaccount,secretproviderclass}.yaml` | **none** | Azure-only ⟵ Gap (Stream A) |
| L3 cell runner (cloud-agnostic) | `backend/agents/rlm/k8s_job_cell_runner.py` | same file | **unified, complete** |
| L3 exec backend | `aks_job_backend.py` (thin) + `k8s_job_backend.py` base | `gke_job_backend.py` (thin) | complete; 5 GCP gates fixed in `1bffb376` |
| L3 capacity | `gpu_capacity._describe_azure` | `_describe_gcp` | complete |
| Cell image | (ACR, pre-baked) | `docker/gke-cell-base/` (+ OOM ladder entrypoint) | complete |

**Net:** the code path is done and mock-tested on both clouds; the data-plane infra is done but
**on-demand only**; the durable control plane + secret store exists **only on Azure**; and
**nothing has run end-to-end on real GPUs**. The work below is four narrow streams, not a rebuild.

## Alternatives considered & rejected

1. **Single-VM-spot fleet (productionize the manual SSH run as Terraform `compute_instance`).**
   Tempting (best-tested `--sandbox local` path; idle cost ≈ 0). Rejected as the *primary*: it is a
   **second divergent execution shape** (new Terraform module + startup-script + preemption
   supervisor + its own auth), it is **not the "scalable/dynamic" path the user asked for**, and it
   re-introduces exactly the bootstrap/auth/resume gaps we just suffered. `--sandbox local` on an
   operator-provisioned VM remains a documented dev/one-off escape hatch — **not** promoted to IaC.
2. **One "fat" 8-GPU Job running the whole matrix via the local cell scheduler in-pod.** Simplest
   (reuses the proven local path verbatim). Kept as the **degenerate fallback** (Stream C note), but
   not the default — it forfeits cross-node cell scaling and couples one paper to one held node.
3. **Move the RLM reasoning loop into a containerized control pod (full in-cluster orchestrator
   image).** This *is* the direction (Stream A) — Azure already did it (Stream E) and we mirror it on
   GCP, reusing the existing `backend.cli` entrypoint image rather than authoring a new one.

## Hardware sizing — are 4–8 A100s enough, or are H100s needed?

**8×A100-40 GB is sufficient for the full SDAR paper (incl. Qwen2.5-7B); H100s are not required.**
- 7B full-FT GRPO with ZeRO-3 / FSDP `full_shard` across 8×40 GB (320 GB aggregate) + gradient
  checkpointing + rollout generation fits with headroom (sharded weights+optim ≈ 10 GB/GPU, leaving
  room for activations + KV cache). The 1.7B/3B cells are 1-GPU each.
- **4×A100-40 (160 GB) is risky** for 7B GRPO once rollout/KV is included — do not scope the heavy
  cell to 4. 8 is the right call.
- **H100-80** would be faster (FP8, more memory headroom) but is a cost/quota step-up with no
  correctness need here. Keep A100 as the default ladder; allow H100 as an opt-in SKU.
- **Node-SKU correction (important).** The current Bicep GPU pool defaults to
  `Standard_NC24ads_A100_v4` = **1 A100 per node**. A per-cell Job that needs 8 GPUs (the 7B,
  `shard_degree=8`) **cannot be scheduled** on 1-GPU nodes as a single-node Job. The GPU pool for
  SDAR must be an **8×A100 node SKU** so the 7B cell is one 8-GPU Job and the light cells pack
  8-to-a-node: **GCP `a2-highgpu-8g` (8×A100-40, what we provisioned manually) or `a2-ultragpu-8g`
  (8×A100-80); Azure `ND96asr_v4` (8×A100-40) or `ND96amsr_A100_v4` (8×A100-80)**. This is a config
  change to `gpu_skus`/the node-pool module, not new code.
- **Open validation risk (flagged, not assumed):** multi-GPU-per-cell **distributed launch inside a
  single K8s Job** (torchrun/accelerate across the pod's 8 GPUs) is exercised today only on the
  `local` path (`primitives._resolve_distributed_launch`). The cell entrypoint
  (`docker/gke-cell-base/gke_cell_entrypoint.py`) must `torchrun`-wrap a cell whose `gpu_count>1`.
  Stream D's live run is the first real test of this; treat it as the highest-risk item.

## Architecture — durable control plane + spot data plane (symmetric per cloud)

```
            ┌──────────────── cloud (Azure RG / GCP project) ────────────────┐
 secret ──► │  [secret store]  KeyVault / Secret Manager                      │
 (op sets   │     claude-code-oauth-token · anthropic-api-key · aoai-key      │
  out-of-   │            │ (Workload Identity, keyless)                       │
  band)     │            ▼                                                    │
            │  [control plane]  orchestrator Deployment/CronJob  (CPU pool)   │
            │     python -m backend.cli reproduce <paper> --sandbox {az,gcp}  │
            │     --model claude-oauth      (unattended, $0/token)            │
            │            │ dispatches one K8s Job per cell                     │
            │            ▼                                                    │
            │  [data plane]  SPOT GPU node pool (8×A100, scale-to-zero)       │
            │     cell Job ──► train_cell.py ──(SIGTERM→ckpt flush)──► blob   │
            │            ▲ resume: deterministic run_id + cell-skip           │
            │  [object store]  GCS / Azure Blob  (code in · metrics/ckpt out) │
            └─────────────────────────────────────────────────────────────────┘
```

Control plane (CPU, always-on, cents/hr) holds the LLM loop + the single Claude token; data plane
(spot GPU, idle = $0) holds only training. **The OAuth-orphaning failure class is structurally
absent** — one token, one process, sourced from the secret store, never a rotating creds file
copied between machines.

## Invariants to preserve (regression guard)

- All existing runtime/config tests stay green (the 189 Azure baseline + the GCP suite
  `tests/agents/rlm/test_gcp_sandbox_hardening.py` + `tests/services/runtime/*`).
- **No behaviour change for any existing sandbox or for the on-demand cluster path** when the new
  flags are unset. Spot, in-cluster-GCP-orchestrator, the long-lived-token path, and resume are all
  **opt-in**.
- Azure Stream E manifests (`orchestrator-*.yaml`, `keyvault.bicep`) are **not edited** except to add
  the new `claude-code-oauth-token` secret name (additive) — the canonical AOAI branch must keep
  working.
- The L3 cell runner stays cloud-agnostic and module-patchable (`azure_blob.*` / `gcs_blob.*`).
- Secret **values** never enter IaC/git — only secret **names**; operator sets values out-of-band.

## Stream A — GCP in-cluster orchestrator parity (the durable control plane)

Mirror Azure Stream E on GCP so the orchestrator runs unattended in-cluster on both clouds.

### NEW `infra/gcp/modules/secret_manager/` (TF module, mirror of `keyvault.bicep`)
- `google_secret_manager_secret` resources for `claude-code-oauth-token`, `anthropic-api-key`,
  `azure-openai-api-key` (names only; **no versions/values** in TF — operator runs
  `gcloud secrets versions add` out-of-band). `outputs.tf`: secret ids.

### CHANGED `infra/gcp/modules/identity/` — add an **orchestrator GSA**
- Second GSA `<prefix>-orchestrator`, `roles/secretmanager.secretAccessor` on the three secrets,
  `roles/storage.objectAdmin` on the bucket, KSA↔GSA binding for `reprolab-orchestrator`. Mirror of
  Azure's orchestrator-MI. New outputs `orchestrator_gsa_email`.

### NEW `infra/gcp/helm/templates/orchestrator-{serviceaccount,deployment,cronjob}.yaml`
- Port the four Azure orchestrator templates. GCP secret injection uses the **Secret Manager CSI
  driver** (`secrets-store.csi.x-k8s.io`, `provider: gcp`) → projected files/env, OR
  `gcsfuse`-free env-from-CSI — mirror Azure's `orchestrator-secretproviderclass.yaml` shape with a
  GCP SecretProviderClass. Deployment runs
  `python -m backend.cli reproduce <paper> --mode rlm --sandbox gcp --model claude-oauth` on the
  **system (CPU) pool**, gated on `.Values.orchestrator.enabled` (default false).
- `infra/gcp/helm/values.yaml`: add the `orchestrator:` block (enabled, image, paper args, model,
  secret names, schedule) — copy the Azure block verbatim where cloud-neutral.

### CHANGED `backend/config.py`
- Add `gcp_*` parity for the orchestrator/secret knobs Azure already has (secret names, csi mount
  path). No new behaviour — settings only.

**Acceptance:** `helm template infra/gcp/helm --set orchestrator.enabled=true` renders a valid
Deployment + SecretProviderClass + orchestrator SA; `terraform plan` shows the secret-manager module
+ orchestrator GSA; existing GCP tests unchanged.

## Stream B — Long-lived Claude OAuth token in-cluster ($0/token, unattended)

The root-level auth fix learned this session. `claude setup-token` mints a **stable, non-rotating**
`CLAUDE_CODE_OAUTH_TOKEN`; unlike `~/.claude/.credentials.json` it is safe to store in a secret
store and inject into an unattended pod, and both the root transport and the claude-agent-sdk
sub-agents read it (they shell out to the `claude` CLI, which honors the env var). This makes
`--model claude-oauth` viable in-cluster — **subscription-billed, $0 per token** — the cheapest
unattended root.

### CHANGED both secret stores (additive)
- Azure `bicep/modules/keyvault.bicep`: add secret **name** `claude-code-oauth-token` to the managed
  list (value set out-of-band). GCP: covered by Stream A's secret-manager module.

### CHANGED both orchestrator manifests
- Inject `CLAUDE_CODE_OAUTH_TOKEN` from the secret store as an env var on the orchestrator
  Deployment/CronJob (Azure `orchestrator-deployment.yaml` + GCP new template). The cell Jobs do
  **not** receive it (cells never call the LLM — confirmed by the GCP code-path audit).

### CHANGED `backend/config.py` / `backend/cli.py` (tiny)
- Recognize `CLAUDE_CODE_OAUTH_TOKEN` as a first-class unattended-root credential so the
  shell-vs-`.env` precedence validator (`cli.py::_warn_on_shell_env_override`) and the model resolver
  treat `--model claude-oauth` + this env var as a valid headless combination (today the assumption
  is a local creds file).

### Cell image (`docker/gke-cell-base/Dockerfile` + ACR equivalent)
- Bundle the `claude` CLI + Node **only if** the orchestrator image is the cell image; if the
  orchestrator runs from a dedicated `backend.cli` image, ensure **that** image carries the `claude`
  CLI + Node (it is the only place the token is used). Pin the CLI version to match
  `claude_agent_sdk._cli_version` to avoid drift.

**Acceptance:** in a control pod with the secret mounted,
`CLAUDE_CODE_OAUTH_TOKEN=<tok> claude --print --output-format json` returns `is_error:false`; a smoke
`reproduce` reaches rubric generation with **zero** `401`/`cost_usd=0.0` first-Sonnet-call deaths.

## Stream C — Spot/preemptible GPU pools + preemption-safe resume (the cost + robustness lever)

### CHANGED node-pool IaC (flag-gated)
- Azure `bicep/modules/gpu-nodepool.bicep`: add `scaleSetPriority: 'Spot'` + `spotMaxPrice: -1`
  (pay up to on-demand) + `evictionPolicy: 'Delete'`, gated on a `useSpot` param (default false).
- GCP `infra/gcp/modules/gpu_nodepool/main.tf`: add `spot = var.use_spot` (default false) (preferred
  over legacy `preemptible`; no 24h cap, supports graceful 15-30s `TERMINATING`).
- **Set the node SKU to an 8×A100 machine** (see Hardware sizing): default `gpu_skus` →
  `a2-highgpu-8g` (GCP) / `ND96asr_v4` (Azure) so the 7B 8-GPU cell fits one node; light cells pack.

### CHANGED cell entrypoint — in-Job checkpoint flush on preemption
- `docker/gke-cell-base/gke_cell_entrypoint.py` (+ Azure cell entrypoint): install a **SIGTERM
  handler** that, on the cloud's preemption signal, flushes the latest `train_cell.py` checkpoint +
  partial `metrics.json` to the object store before the grace window expires. Reuse the existing exit
  vocabulary; add `preempted` status. The agent's `train_cell.py` already checkpoints every 50–100
  steps (SDAR guidance); the handler just guarantees the last one lands in blob.

### CHANGED resume protocol — deterministic run_id + cell-skip (closes the "redo everything" gap)
- **Root cause (verified):** a fresh launch gets a **random `run_id`** + attempt-archive, so a
  preempted run that restarts re-does every cell from scratch. Fix:
  - `k8s_job_cell_runner.run_matrix` + the cell scheduler: before launching a cell Job, **check the
    object store** for `runs/<run_id>/cells/<cell_id>/status.json == ok` and **skip** it (gated by
    `OPENRESEARCH_RESUME_CELLS=1`, which already exists but is not honored on the K8s path).
  - The orchestrator Deployment/CronJob passes a **stable `--project-id`/run_id** (derived from the
    paper + arm, not random) so a rescheduled pod re-attaches to the same object-store prefix and
    skips completed cells. K8s `restartPolicy: OnFailure` + Job `backoffLimit` reschedules the
    preempted cell Job onto a new spot node.
- `aggregate_cell_metrics` already treats `skipped` cells correctly (per CLAUDE.md) — verify it
  counts blob-resumed cells.

**Acceptance:** kill a running spot node mid-matrix; the orchestrator reschedules only the in-flight
cell, completed cells are skipped from blob, the final aggregate is identical to an
un-preempted run. Cost of a preemption is bounded to one cell's redo, not the whole matrix.

## Stream D — Cost guardrails + live validation (the actual proof)

### Cost ceiling enforced on the K8s path
- Thread `RunBudget.check_run_gpu_usd` (exists for local/runpod) through `k8s_job_cell_runner`: before
  dispatching a cell Job, estimate `gpu_count × gpu_usd_per_hour × est_seconds` and **refuse** once
  the run-total `max_run_gpu_usd` (default from `--max-usd`) would be exceeded → terminal
  `capacity_exhausted`/`budget_exhausted` stop_reason (reuses the existing STOP plumbing, no re-OOM
  loop). Spot price is ~⅓ on-demand; bill against the spot rate.
- Cluster teardown guardrail: Job `ttlSecondsAfterFinished` (set) + an **operator teardown command**
  (`az stack group delete` / `terraform destroy` of the GPU pool) documented; scale-to-zero already
  makes idle GPU cost $0, so the standing cost is only the CPU system pool + control plane (cents/hr).

### Live validation (the gate that has never been crossed)
- Run the existing smoke jobs (`helm/smoke/{hello-gpu,cpu-stub}-job.yaml`) on each cloud, then the
  **real SDAR reproduction** end-to-end on one cloud (GCP first — we already have a provisioned
  GCP project, quota history, and a working `claude setup-token`), then repeat on Azure.
- Document the **admin gates** (operator-side, have lead time): GPU quota raise (8×A100 family),
  AKS/GKE RBAC cluster-admin grant, secret values set out-of-band, ADC/`az login` for the local
  bootstrap, image build+push to ACR/Artifact Registry. These are runbook items, not code.

**Acceptance:** `final_report.json` for SDAR (all three Qwen sizes, core comparison present) lands in
the object store from an **unattended** in-cluster run on **both** clouds, under the USD ceiling, with
at least one survived spot preemption.

## Stream E (optional, lower priority) — Helm chart unification

The two Helm charts are ~85 % identical. Dedupe **additively**: a shared base chart (namespace, RBAC,
resource-quota, device-plugin, orchestrator) + thin per-cloud overlays for the 15 % that diverges
(storage-class CSI driver, SA identity annotation form, secret-provider). **Guard:** Azure manifests
must render byte-identical before/after (`helm template` diff = ∅); do this only after Streams A–D are
green, and never let it block the SDAR run.

## Cost model ($400 cap context)

| Item | On-demand | **Spot** | Notes |
|---|---|---|---|
| 8×A100-40 node (`a2-highgpu-8g` / `ND96asr_v4`) | ~$10–16/hr | **~$3–6/hr** | idle = $0 (scale-to-zero) |
| CPU system pool + control plane | ~$0.15–0.40/hr | same | always-on; the only standing cost |
| Root LLM (`--model claude-oauth`) | — | **$0/token** | subscription; the cost win |
| Object store + egress | cents | cents | |

A full SDAR run (generous wall-clock, 8×A100 spot, claude-oauth root) lands comfortably inside $400;
the spot rate + `$0/token` root + scale-to-zero are what make it cheap. `max_run_gpu_usd` is the hard
backstop.

## Operational contract (identical across clouds)

- **Same secret names** both clouds: `claude-code-oauth-token`, `anthropic-api-key`,
  `azure-openai-api-key`.
- **Same env knobs:** `OPENRESEARCH_RESUME_CELLS=1`, `--max-usd`, `--max-wall-clock`, `--paper-hint`,
  `OPENRESEARCH_BASELINE_EXTRA_GUIDANCE` (the 3-Qwen-size SDAR directive).
- **Same launch:** `helm upgrade --install ... --set orchestrator.enabled=true` → unattended run;
  artifacts in `gs://…/runs/<run_id>` or `…blob…/runs/<run_id>`.
- **Only difference:** `--sandbox gcp` vs `--sandbox azure` and which IaC tool provisioned the
  substrate. Everything downstream is shared code.

## Tests (new)
- `tests/services/runtime/test_k8s_resume_skip.py` — `run_matrix` skips a cell whose
  `status.json==ok` exists in a fake object store when `OPENRESEARCH_RESUME_CELLS=1`; re-runs it when 0.
- `tests/services/runtime/test_k8s_budget_gate.py` — cell dispatch refused once `max_run_gpu_usd`
  would be exceeded (spot rate), terminal stop_reason emitted, no re-loop.
- `tests/runtime/test_cell_entrypoint_preempt.py` — SIGTERM handler flushes ckpt+partial metrics +
  `preempted` status (pure-function level, no GPU).
- `tests/config/test_claude_oauth_token_headless.py` — `--model claude-oauth` + `CLAUDE_CODE_OAUTH_TOKEN`
  resolves as a valid unattended root; shell-override validator does not warn on it.
- Helm render tests: GCP `orchestrator.enabled=true` renders Deployment+SecretProviderClass+SA;
  spot param renders `scaleSetPriority/Spot` (Azure) and `spot=true` (GCP); Azure unchanged-render diff.

## Acceptance gates (in order)
1. **A/B unit + render** — all new tests green; existing 189 Azure + GCP hardening suites unchanged;
   `helm template` Azure diff = ∅; `terraform plan` / `bicep build` clean.
2. **Auth smoke** — control pod with secret → `claude --print` non-401; smoke `reproduce` reaches
   rubric gen with no `cost_usd=0.0` first-call death.
3. **GPU smoke** — `hello-gpu-job` runs `nvidia-smi` + blob write on a **spot** 8×A100 node, scales 0→1→0.
4. **Preemption-resume** — killed spot node → only the in-flight cell reruns; aggregate identical.
5. **SDAR end-to-end (the proof)** — unattended in-cluster SDAR (3 Qwen sizes) → `final_report.json`
   in blob, under `--max-usd`, on **GCP then Azure**.

## Open decisions for the operator (flagged, not silently chosen)
- **D1 — 8×A100-80 vs 8×A100-40 node SKU.** 40 GB (a2-highgpu-8g / ND96asr) is proven-sufficient and
  cheaper; 80 GB (a2-ultragpu-8g / ND96amsr) buys headroom + easier quota story on some
  subscriptions. Recommend **40 GB**; switch via one `gpu_skus` value.
- **D2 — cloud to validate first.** Recommend **GCP** (existing project, quota history, working
  `setup-token`), then Azure.
- **D3 — Stream E (Helm dedup) now or later.** Recommend **later** — ship the SDAR proof first.
