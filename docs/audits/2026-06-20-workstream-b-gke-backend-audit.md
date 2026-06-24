# Workstream B — GKE/GCP execution backend audit (2026-06-20)

> READ-ONLY audit. No production code changed. Every claim cites `file:line`.
> Auditor scope: is `--sandbox gcp` (the GKE Job backend) a stub, half-wired, or
> nearly-complete-but-unproven? Compared against the two design specs and against
> the Azure/AKS reference path.

## 1. Executive summary — VERDICT

**GKE is NEARLY-COMPLETE and hermetically unit-tested, but never live-validated on
real GPUs.** It is emphatically NOT a stub and NOT half-wired. The entire L3 code
layer promised by the 2026-06-16 spec is implemented and reachable end-to-end:
the CLI flag, the `SandboxMode` enum, the backend-factory dispatch, the
availability gate, the `build_environment` no-op, the GPU catalog/resolver/capacity
branches, the GCS object store, and the cell-matrix arm are all present and
mirror Azure structurally. **Surprisingly, the bulk of the *2026-06-17* "production"
spec is ALSO already landed** — GCP Secret Manager IaC, in-cluster orchestrator
Helm templates, spot/preemptible node pools, preemption-safe resume, a K8s-path
USD/pod-second budget gate, and the long-lived `CLAUDE_CODE_OAUTH_TOKEN` headless
root are all implemented and tested. **289 GCP/GKE-specific tests pass** (see §8).

Two caveats keep this from being "done": (a) the **289 green tests cover only the
Python code path — NOT the IaC**; `helm template` / `terraform plan` rendering and,
above all, a **real end-to-end SDAR run on GPUs, are unverified by this audit and
(per both specs) have never been crossed**; (b) **one genuine code gap remains**:
the GKE cell entrypoint does not `torchrun`-wrap a multi-GPU (`gpu_count>1`) cell,
which the 06-17 spec itself flags as the highest-risk open item. Net: this is a
**validation gate, not a build gate** — the code is ready to run; nobody has run it.

### Naming correction the design author almost certainly has wrong
The selectable value is **`gcp`, NOT `gke`.** There is no `SandboxMode.gke`.
`--sandbox gcp` → `SandboxMode.gcp` → instantiates the class named `GkeJobBackend`.
"GKE" is the *backend implementation* name; "gcp" is the *operator-facing token*.
Any design/runbook that tells an operator to pass `--sandbox gke` is wrong —
it would fall through to a `LocalDockerBackend` warning fallback
(`backend/agents/rlm/primitives.py:2749-2757`).

---

## 2. Spec-vs-reality table (2026-06-16 GCP/GKE spec)

| Spec promise (§ of 2026-06-16) | Status | Evidence (file:line) |
|---|---|---|
| `gcs_blob.py` — GCS store mirroring `azure_blob.py` | **IMPLEMENTED** | `backend/services/runtime/gcs_blob.py` (354 lines; azure_blob.py is 356) |
| `k8s_job_backend.py` — shared base `_KubernetesJobBackend` + `CloudSpec` | **IMPLEMENTED** | `k8s_job_backend.py:444` (`class _KubernetesJobBackend`), `:402` (`class CloudSpec`) |
| `AzureBlobStore` + `GcsStore` adapters in base | **IMPLEMENTED** | `k8s_job_backend.py:280` (`AzureBlobStore`), `:339` (`GcsStore`) |
| `aks_job_backend.py` slimmed to Azure `CloudSpec` adapter | **IMPLEMENTED** | `aks_job_backend.py:202` (`_AZURE_CLOUD`), `:219` (`AksJobBackend`) |
| `gke_job_backend.py` — GCP adapter (mirror of slim aks) | **IMPLEMENTED** | `gke_job_backend.py:159` (`_GCP_CLOUD`), `:176` (`GkeJobBackend`) |
| `ensure_gcp_available()` — SDK + settings + ADC + kubeconfig preflight | **IMPLEMENTED** | `gke_job_backend.py:62-156` |
| `SandboxMode.gcp = "gcp"` enum | **IMPLEMENTED** | `backend/agents/execution.py:39` |
| `ensure_sandbox_mode_available` gcp branch | **IMPLEMENTED** | `backend/agents/execution.py:294-296` (`elif resolved is SandboxMode.gcp`) |
| `config.py` — `gcp_*` block + `default_sandbox`/`force_sandbox` literals | **IMPLEMENTED** | `config.py:234` + `:241` literals; **38** `gcp_` field refs; `:416-461` core block |
| `__init__.py` import + `__all__` for `GkeJobBackend`/`ensure_gcp_available` | **IMPLEMENTED** | `backend/services/runtime/__init__.py:43`, `:54` |
| `gpu_capacity.py` — `_describe_gcp` + `gcp` dispatch + `startswith("gcp_")` plan disambig | **IMPLEMENTED** | `gpu_capacity.py:101-102` dispatch, `:253` `_describe_gcp`, `:290` prefix disambig |
| `primitives.py` factory `_backend_for_sandbox_mode` gcp arm | **IMPLEMENTED** | `backend/agents/rlm/primitives.py:2742-2747` |
| `resolve_gpu_requirements` gcp provider arm | **IMPLEMENTED** | `primitives.py:1251` (`_provider = "gcp"`); resolver `gpu_resolver.py:117-119` |
| `cli.py` — `gcp` in `--sandbox` choices | **IMPLEMENTED** | `backend/cli.py:2208` (`choices=("auto","local","docker","runpod","azure","gcp")`) |
| `build_environment` no-op for gcp (Artifact Registry pre-baked) | **IMPLEMENTED** | `primitives.py:1405-1411` (`gcp sandbox: image is pre-baked in Artifact Registry … no-op`) |
| Cell-matrix `k8s_job_cell_runner` gcp arm in `_execute_cell_matrix` | **IMPLEMENTED** | `primitives.py:5612` (`_sb_key_ecm in ("azure","gcp")`); cell-route gate `:6070` |
| `gpu_catalog` GCP A2 rows (A100 40/80, 1–8×) | **IMPLEMENTED** | `gpu_catalog.py:93-108` (8 `provider="gcp"` GpuSku rows) |
| New tests (gcs_blob, gke_job_backend, gpu_capacity_gcp, config/factory) | **IMPLEMENTED** | see §8 (all present + passing) |

**Conclusion:** the 2026-06-16 spec is **100% implemented.** Its own acceptance gate
(§ "Acceptance gate" 1–4) appears met for the unit layer.

### Spec-vs-reality (2026-06-17 multi-cloud production spec — the harder bar)

| Stream / promise | Status | Evidence (file:line) |
|---|---|---|
| Stream A — GCP Secret Manager TF module | **IMPLEMENTED** | `infra/gcp/modules/secret_manager/{main,outputs,variables}.tf` |
| Stream A — orchestrator GSA in identity module | **IMPLEMENTED (verify scopes)** | `infra/gcp/modules/identity/main.tf` (present; per-role grants not line-audited here) |
| Stream A — orchestrator Helm templates (SA/Deployment/CronJob/SecretProviderClass) | **IMPLEMENTED** | `infra/gcp/helm/templates/orchestrator-{serviceaccount,deployment,cronjob,secretproviderclass}.yaml` |
| Stream B — long-lived `CLAUDE_CODE_OAUTH_TOKEN` headless root | **IMPLEMENTED + TESTED** | `config.py:562-566` (`claude_code_oauth_token` field); `orchestrator-deployment.yaml:165-169` env injection; `tests/config/test_claude_oauth_token_headless.py` (5 passed) |
| Stream C — spot/preemptible node pool IaC | **IMPLEMENTED** | `infra/gcp/modules/gpu_nodepool/main.tf:54-64` (`spot = var.use_spot`); runtime `config.py:486-488` `gcp_use_spot`/`gcp_spot_backoff_limit` |
| Stream C — 8×A100 node SKU default (a2-highgpu-8g) | **IMPLEMENTED** | `infra/gcp/modules/gpu_nodepool/variables.tf:29-33` (`default = "a2-highgpu-8g"`) |
| Stream C — preemption SIGTERM ckpt-flush + `preempted` status | **IMPLEMENTED** | `docker/gke-cell-base/gke_cell_entrypoint.py:67` (`EXIT_PREEMPTED=45`), `:76` (`_OUTCOME_PREEMPTED`), `:293` (`build_preempt_sentinel`) |
| Stream C — deterministic-run_id cell-skip resume (`OPENRESEARCH_RESUME_CELLS`) | **IMPLEMENTED** | `k8s_job_cell_runner.py:24` (resume doc), `:1139-1143` (blob `status.json` reconcile) |
| Stream D — `RunBudget` USD/pod-second gate on K8s path | **IMPLEMENTED** | `k8s_job_cell_runner.py:520-552` (`budget_exhausted:` terminal-stop contract) |
| Stream D — multi-GPU-per-cell `torchrun` wrap in cell entrypoint | **ABSENT (the real gap)** | `gke_cell_entrypoint.py:527-534` launches `train_cell.py` directly via `sys.executable`; **no `torchrun`/`nproc_per_node`/`WORLD_SIZE`** anywhere in the file |
| Stream D — live SDAR end-to-end on real GPUs | **NOT DONE (never crossed)** | 06-17 spec §"Live validation": "the gate that has never been crossed"; no GCP `final_report.json` artifact found |
| Stream E — Helm chart unification | **NOT DONE (spec marks optional/later)** | per spec §Stream E, deferred |
| IaC render proof (`helm template` / `terraform plan`) | **UNVERIFIED BY THIS AUDIT** | tools not run (read-only audit; pytest does not render Helm/TF) |

---

## 3. AKS vs GKE method-coverage diff

**There is no per-method diff to make — by design.** Both `AksJobBackend`
(`aks_job_backend.py:219`) and `GkeJobBackend` (`gke_job_backend.py:176`) subclass
the **same** base `_KubernetesJobBackend` (`k8s_job_backend.py:444`) and **override
zero RuntimeBackend methods.** Each is a 1-line `__init__` passing a `CloudSpec`:
- `AksJobBackend.__init__` → `super().__init__(_AZURE_CLOUD, **kw)` (`aks_job_backend.py:246-247`)
- `GkeJobBackend.__init__` → `super().__init__(_GCP_CLOUD, **kw)` (`gke_job_backend.py:203-204`)

| RuntimeBackend method | AKS impl | GKE impl | Where it lives |
|---|---|---|---|
| `create_sandbox` (upload project → object store, return logical Sandbox) | inherited | inherited | `k8s_job_backend.py:620` |
| `exec` (submit K8s Job, poll, capture logs → ExecResult) | inherited | inherited | `k8s_job_backend.py:674` |
| `destroy` (delete tracked Jobs; preserve artifacts) | inherited | inherited | `k8s_job_backend.py:794` |
| `_upload_project_sync` (build → object-store prefix) | inherited | inherited | `k8s_job_backend.py:815` |
| Job manifest build (GPU toleration, nodeSelector, resources) | inherited via `_build_job_manifest` | inherited | `k8s_job_backend.py:165` |
| Availability preflight | `ensure_azure_available` | `ensure_gcp_available` | `aks_job_backend.py:113` / `gke_job_backend.py:62` |

**The only per-cloud deltas are the `CloudSpec` fields** (`k8s_job_backend.py:429-436`):
| CloudSpec field | Azure (`_AZURE_CLOUD`) | GCP (`_GCP_CLOUD`) |
|---|---|---|
| `provider` | `"azure"` | `"gcp"` |
| `settings_prefix` | `"azure"` | `"gcp"` |
| `sandbox_prefix` / `sandbox_label` | `"aks"` | `"gke"` |
| `pod_template_extra_labels` | `{"azure.workload.identity/use":"true"}` | `{}` (GKE binds WI cluster-side via KSA annotation) |
| `base_image_setting` | `"azure_base_image"` | `"gcp_base_image"` |
| `make_object_store` | `_make_azure_store` → `AzureBlobStore` | `_make_gcs_store` → `GcsStore` |
| `ensure_available` | `ensure_azure_available` | `ensure_gcp_available` |

Refs: `aks_job_backend.py:202-211`, `gke_job_backend.py:159-168`.
**The architecture decision in the 06-16 spec ("shared K8s base + cloud adapters,
NOT a duplicated GkeJobBackend") was followed exactly.**

---

## 4. Sandbox-selection reachability trace — IS `--sandbox gcp` WIRED? YES.

End-to-end, fully reachable (value is **`gcp`**, not `gke`):

1. **CLI flag** `--sandbox` accepts `gcp` — `backend/cli.py:2207-2208`
   (`choices=("auto","local","docker","runpod","azure","gcp")`).
2. **Enum** `SandboxMode.gcp = "gcp"` — `backend/agents/execution.py:39`.
3. **Force override** `OPENRESEARCH_FORCE_SANDBOX` returns `SandboxMode(force)` —
   honors `gcp` like any mode (`execution.py:253-255`); literals include `gcp`
   (`config.py:234`, `:241`).
4. **Availability gate** `ensure_sandbox_mode_available` → `ensure_gcp_available()`
   — `execution.py:294-296`.
5. **Backend factory** `_backend_for_sandbox_mode` (the `*Backend` dispatch):
   `if mode is SandboxMode.gcp:` → `ensure_gcp_available()` →
   `return GkeJobBackend(run_budget=..., gpu_plan=...)` —
   `backend/agents/rlm/primitives.py:2742-2747`.
6. **Cell-matrix path** routes `gcp` through the shared K8s cell runner —
   `primitives.py:5612` (`_sb_key_ecm in ("azure","gcp")`) + cell-route admission
   `primitives.py:6070`.

**Not orphaned.** The unknown-mode fallback to `LocalDockerBackend`
(`primitives.py:2749-2757`) is what a stray `--sandbox gke` would hit — `gcp` does
not. (`auto`/`brev`/`simulate` are the modes still unwired for the RLM path.)

---

## 5. GPU-planning GCP branch coverage — FULL

- **Catalog SKUs (`gpu_catalog.py:93-108`)** — 8 GCP A2 rows, all `provider="gcp"`,
  with $/hr:
  - `gcp_a100_40` (a2-highgpu-1g, 40GB, $2.93) … `gcp_a100_40x8` (a2-highgpu-8g, 40GB, $23.44)
  - `gcp_a100_80` (a2-ultragpu-1g, 80GB, $3.93) … `gcp_a100_80x8` (a2-ultragpu-8g, 80GB, $31.44)
  - **No L4 / no H100 rows** (catalog is A100-only for GCP; 06-17 §D1 leaves H100 an
    opt-in to add). If workstream B wants L4/H100 on GCP, those rows do not yet exist.
- **Resolver branch (`gpu_resolver.py:117-119`)** — `if provider in ("azure","gcp")`
  routes to `_resolve_provisioned_cloud`; GCP fallback `_GCP_FALLBACK_SHORT_NAME =
  "gcp_a100_40"` (`gpu_resolver.py:40`), cloud_type `("ONDEMAND",)` (`:39`).
- **Capacity descriptor (`gpu_capacity.py:253-309`)** — `_describe_gcp` exists,
  mirrors `_describe_azure` (`:178`), reads `gcp_*` settings, `num_gpus =
  gcp_max_nodes × gcp_gpus_per_node` (`:277`), `can_escalate=False` (GKE scales
  horizontally via node-pool), and **disambiguates GCP vs Azure plans by
  `short_name.startswith("gcp_")`** (`:290`) — NOT by `cloud_type` (both are
  `ONDEMAND`). Dispatch at `gpu_capacity.py:101-102`; `gcp` in the cloud-kind tuple
  at `:109`.

The Azure reference (`_describe_azure`) and the GCP analog (`_describe_gcp`) are
both present and parallel — exactly what the spec asked for.

---

## 6. GAP LIST (ranked, concrete)

1. **[HIGH — real code gap] Multi-GPU-per-cell `torchrun` wrap missing in the GKE
   cell entrypoint.** `docker/gke-cell-base/gke_cell_entrypoint.py:527-534` launches
   `train_cell.py` directly (`[sys.executable, train_cell_path, --cell-id, --output-dir]`)
   with no `torchrun`/`accelerate`/`nproc_per_node`. A cell with `gpu_count>1` (the
   SDAR 7B 8-GPU cell) will run single-process and under-utilize / fail.
   **Template to mirror:** the runner-side `_resolve_distributed_launch`
   (`backend/agents/rlm/primitives.py:3193`, which wraps `torchrun`/`accelerate`/
   `deepspeed` for the local multi-GPU path). The fix belongs in
   `gke_cell_entrypoint.py` (the entrypoint, not `train_cell.py` — distributed
   launch is a *runner* concern on every other path). The 06-17 spec §"Hardware
   sizing / Open validation risk" names this the highest-risk item explicitly.

2. **[HIGH — process gap, not code] Zero live GPU validation.** No GCP/GKE
   `final_report.json` exists; both specs state the end-to-end gate "has never been
   crossed" (06-17 §"Live validation"). All 289 passing tests are hermetic
   (mock batch_api/core_api/blob_client). **Action:** run the smoke jobs
   (`infra/gcp/helm/smoke/{hello-gpu,cpu-stub}-job.yaml`) then a real SDAR run.
   Template: the Azure SDAR run design (`docs/superpowers/specs/2026-06-14-sdar-on-azure-run-design.md`).

3. **[MEDIUM — verification gap] IaC render unproven by this audit.** `helm template`
   and `terraform plan` were not run (read-only; tooling may be absent). The
   06-17 acceptance gate lists these as distinct gates. **Action:** `helm template
   infra/gcp/helm --set orchestrator.enabled=true` and `terraform plan` in
   `infra/gcp/` — confirm they render before trusting the IaC.

4. **[LOW — missing operator tool] No GKE-cluster preflight script.** Azure has
   `scripts/azure_sdar_preflight.sh` and `scripts/azure_sdar_bootstrap_cluster.sh`;
   the GCP scripts (`scripts/gcp_sdar_preflight.sh`, `cancel_gcp_sdar_run.sh`,
   `sdar_gcp_assets.py`) target the **single-VM SSH path** (the explicitly-rejected
   alternative — `gcp_sdar_preflight.sh:33` describes a `gcloud compute instances`,
   not a GKE cluster). There is **no GKE analog of `scripts/runpod_check.sh`**
   (kubeconfig/`get-credentials`/nodepool/RBAC preflight). **Template:**
   `scripts/runpod_check.sh` (429 lines) + `scripts/azure_sdar_preflight.sh`.

5. **[LOW — catalog completeness] No GCP L4 or H100 SKUs.** `gpu_catalog.py`
   GCP rows are A100-only (40/80 GB). If the design wants cheaper L4 inference
   nodes or H100 training, add `provider="gcp"` rows mirroring `gpu_catalog.py:93-108`.

6. **[INFO — naming] Operator-facing token is `gcp`, the class is `GkeJobBackend`.**
   Any doc/runbook that instructs `--sandbox gke` is wrong and silently falls back
   to local docker. Standardize on `--sandbox gcp` in all design copy.

---

## 7. Open questions for DECIDE points

- **Autopilot vs Standard GKE:** the IaC provisions a **Standard** cluster with an
  explicit GPU node pool + nvidia device-plugin (`infra/gcp/modules/gke/`,
  `infra/gcp/modules/gpu_nodepool/`, `infra/gcp/helm/templates/nvidia-device-plugin.yaml`).
  This is the right call for pinned GPU SKUs + spot; Autopilot would remove node-pool
  control. **Already decided in IaC → Standard.** No code change needed unless WB
  wants Autopilot.
- **Target SKUs / regions:** default region `us-central1` (`config.py:417`); default
  GPU node SKU `a2-highgpu-8g` = 8×A100-40GB (`gpu_nodepool/variables.tf:33`), with
  `a2-ultragpu-8g` (80GB) as the documented override. 06-17 §D1 recommends 40GB.
  **Decided in IaC defaults.** Catalog supports 1–8× A100 at both memory tiers.
- **IaC vs manual provisioning:** **Full Terraform + Helm already in repo** —
  `infra/gcp/` (root TF + 7 modules: gke, gpu_nodepool, network, identity, registry,
  storage, secret_manager), `infra/gcp/bootstrap/` (state/identity), and an 11-template
  Helm chart incl. the in-cluster orchestrator. Plus the cell image at
  `docker/gke-cell-base/`. This is NOT a manual stopgap — it mirrors the Azure Bicep
  posture. The 06-17 spec explicitly **rejected** a single-VM IaC path; the SSH-VM
  scripts in `scripts/` are the documented dev escape hatch only.
- **Spot default:** spot is **opt-in, default OFF** on both IaC (`gpu_nodepool/main.tf:64`
  `spot = var.use_spot`) and runtime (`config.py:486-488` `gcp_use_spot=false`).
  WB decides whether to flip the default for cost.

---

## 8. Test evidence (hermetic)

`tests/services/runtime/{test_gke_job_backend,test_gcs_blob,test_gpu_capacity_gcp,
test_gpu_resolver_gcp,test_backend_factory_gcp,test_gke_cell_runner,
test_gke_cell_entrypoint,test_spot_preemptible_sku,test_cell_entrypoint_preempt}.py`
+ `tests/config/test_gcp_orchestrator_settings.py` +
`tests/agents/rlm/test_gcp_sandbox_hardening.py` → **289 passed in 7.65s.**
`tests/config/test_claude_oauth_token_headless.py` → **5 passed.**
Also present: `tests/config/test_gcp_sku_pool_invariant.py`. All inject fake
`batch_api`/`core_api`/`blob_client` — **no real cluster or GPU is exercised.**
