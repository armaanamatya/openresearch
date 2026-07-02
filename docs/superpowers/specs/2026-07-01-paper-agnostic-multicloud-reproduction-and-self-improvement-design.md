# Paper-Agnostic Multi-Cloud Reproduction + Self-Improvement — Design

> **Doc status:** Design spec · **v2 (2026-07-01)** · brainstormed + approved
> section-by-section, then **adversarially reviewed by Codex and reworked** (all 16
> findings resolved — see §16). Supersedes the ad-hoc SDAR-on-GCP bash as the
> *architecture of record* for the reproduction compute+provisioning layer (the bash
> remains the live fallback until the Phase-1 A/B proves parity — §10). Reconciles
> with `2026-06-16-gcp-gke-execution-backend-design.md`, the 2026-06-17 multi-cloud
> IaC spec (§3.1), and the 2026-06-20 GKE-backend audit.

## 1. Context & goal

OpenResearch reproduces a research paper end-to-end: ingest → RLM root writes
Python → build environment → implement + run a baseline → score against a rubric →
emit `final_report.{json,md}`. The reference product experience is **alphaxiv's
"Autoresearch"** (`alphaxiv_autoresearc.png`): *"turns this paper into a runnable
project: an agent sets up the project, writes experiments, and launches them on
cloud GPUs,"* works **with or without a GitHub link**, and lets the user inspect
runs / steer. The long-horizon goal is to feed a **patent-generation application**;
the near-horizon goal is faithful, hands-off reproduction of **any** paper.

**The problem this spec solves:** the compute + provisioning layer that actually
works today is a **bespoke, single-paper harness**, not a paper-agnostic backend.
The SDAR reproduction rides a stack of shell scripts
(`scripts/sdar_gcp_optimal_run.sh` → `gcp_sdar_preflight.sh` → `sdar_gcp_run.sh` +
`sdar_gcp_assets.py`) that hardcode the paper (`2605.15155`), the models (Qwen),
the three environments and their entire provisioning logic
(`backend/services/runtime/env_cache.py`), the zone, the cache disk, and the
machine image. Making this work "seamlessly and interchangeably across any paper on
GCP or Azure, handling setup / dataset / experiment execution" — and learning from
its own failures so it stops repeating them — is the work.

## 2. Locked decisions (the design contract)

| # | Decision | Choice |
|---|---|---|
| 1 | Paper scope | **Truly any paper, best-effort** — always produce the best honest partial reproduction, gated by (2). |
| 2 | Cost control | **Cheap deterministic triage before any GPU lease** → proceed / auto-down-scope / plan-only. |
| 3 | Compute model | **Both single-VM and managed-Job execution, behind one interface** (single-VM = Phase 1; cluster = north-star). |
| 4 | Multi-cloud | **One `--cloud gcp\|azure` flag; GCP primary, Azure parity-secondary** (Azure VM experimental — §5.5/§11). |
| 5 | Provisioning | **Generic resolver + optional per-paper/per-env adapter registry** (SDAR's 3 envs = the first adapters). |
| 6 | Self-improvement trust | **Advisory memory only** (never auto-mutates run mechanics); **evidence-not-grade** red line (a multi-predicate `EvidenceVector`, §7). |
| 7 | SI memory scope | **Global infra/provisioning knowledge + per-paper method specifics.** |
| 8 | Deliverable | **One north-star doc + Phase-1 detailed enough to implement** (this doc). |
| 9 | Architecture | **Approach 2 (unified backend abstraction) as Phase 1; Approach 3 (cloud-native control plane) as north-star.** |
| 10 | Compute tiering | **CPU tier (cheap) for dataset download + env setup where feasible; lease/attach the GPU tier only for training** — minimize the GPU billing window. (GCP on-demand A100 is `stage_on_gpu` until the warm-disk handoff is built — §5.5.) |
| 11 | Quality bar | **Robust · optimal · dynamic · modular · scalable** — each pinned to a concrete mechanism in §4.1. |

## 3. Current state (grounded)

Two **disjoint execution worlds** exist today:

- **The architected backend — managed K8s Jobs — is already ~95% cloud-unified.**
  `RuntimeBackend` (ABC, `backend/services/runtime/interface.py`) →
  `_KubernetesJobBackend` (`k8s_job_backend.py`) parameterized by a `CloudSpec`
  descriptor; `GkeJobBackend` and `AksJobBackend` are ~5%-thin adapters (object
  store + workload-identity label + settings prefix). Code + metrics move via a
  GCS/Blob bus; nodes scale from zero via the cluster autoscaler. IaC exists
  (`infra/gcp/` Terraform, `infra/azure/bicep/` + Helm). So "interchangeable
  GCP/Azure" is largely **solved for the Job path.**

- **The actually-exercised SDAR production path is a single VM + `--sandbox local`,
  orchestrated by bash — with no backend class.** `gcloud compute instances create`
  from a machine image, warm cache disk, GREEN gate, then `backend.cli reproduce`
  runs **on the VM** with `--sandbox local` (host subprocesses on the A100s). All
  lifecycle intelligence (capacity-polling, control-plane billing ceiling, idle
  watchdog, watcher-pull-before-stop) lives in scripts, not Python — and it SCPs a
  raw `.env` (OAuth token) to the VM (`gcp_sdar_preflight.sh:342`).

Other current-state facts the design builds on:

- **Provisioning is SDAR-hand-written:** `env_cache.py` (ALFWorld/WebShop/Search-QA,
  `fcntl`-locked, crash-safe, idempotent) + `asset_provisioning.py`. The generic
  path is agent-prompt guidance (`baseline_implementation.py::_dataset_setup_block`,
  `dataset_recipes.py`) that breaks for custom-hosted / gated / oversized data,
  non-PyPI wheels, and CUDA-compilation deps. No credential-injection seam.
- **Env inference** is `environment_detective.py::run_offline` with a hardcoded
  `_FRAMEWORK_COMPATIBILITY` CUDA/Python matrix.
- **The reproduction harness (Layer B) is genuinely paper-agnostic** — SDAR is NOT
  special-cased in code; it is a `PAPER_HINTS` dict entry
  (`backend/agents/prompts/paper_hints.py`). The 12+ primitives, the cells route
  (`gpu_cell_runner.py`/`cell_scheduler.py`/`cell_matrix.py`), the fidelity guards,
  and auto-rubric generation (`rubric_gen.py`) are already generic.
- **Self-improvement substrate exists but is per-paper and default-OFF:**
  `lesson_distiller.py` (→ `runs/_lessons/<arxiv_id>.json`), `recipe_library.py`
  (multi-predicate evidence-gated recipes — evidence gate + ledger row + validator +
  deterministic target, `recipe_library.py:424`), `failure_capsule.py`,
  `failure_classifier.py` (returns only `(failure_class, suggested_fix)` today).
- **A unifying "what should this reproduce" object exists** (default-OFF): the
  round-2 **`SemanticReproductionContract`** (`backend/agents/rlm/semantic_contract.py`,
  `OPENRESEARCH_REPRO_CONTRACT`) with structured `resource_identities` +
  `capability_profile` + `metric_contracts` — distinct from the legacy *textual*
  `ReproductionContract`/`MetricContract` in `schemas.py`.

### 3.1 Reconciliation with the 2026-06-17 multi-cloud IaC decision — a scoped transitional reversal

The 2026-06-17 multi-cloud IaC spec **explicitly rejected single-VM as a primary
path** ("Do NOT add a single-VM IaC path — the manual 8×A100 SSH run we just
cancelled was a stopgap") on three grounds: (1) a *divergent execution shape*
(Terraform + startup-script + preemption supervisor + own auth), (2) *not the
scalable/dynamic path*, (3) *re-introduces the bootstrap/auth/resume/supervision
gaps we just suffered* (OAuth orphaning + secret handling). This spec re-introduces
single-VM — framed honestly as a **scoped transitional reversal, not a clean win**:

- **Transitional.** Single-VM is the Phase-1 proving ground; the managed cluster
  path is the scalable primary and the north-star (§4, §11), with **explicit exit
  criteria** (below).
- **(1) shape → narrowed by decision 3, not erased.** Single-VM is a *runtime
  backend* (`VmComputeProvider`) behind the same `ComputeProvider` interface, not
  new Terraform IaC. But this does not erase the *auth/bootstrap/supervision*
  divergence — that is handled explicitly, not waved away.
- **(3) auth/secret handling → a hard precondition, not "already solved."** The
  OAuth-orphaning fix (long-lived `CLAUDE_CODE_OAUTH_TOKEN` + the lifecycle driver)
  makes single-VM *run* today (`sdar_merged_full_2g`), but the current scripts still
  **SCP a raw `.env`** to the VM — unacceptable for a productized path. **Live VM
  use is gated on `CredentialBroker` / cloud-secret injection (§6.2): no raw `.env`
  on the wire, no secret-shaped value in a staged bundle or log.**

**Exit criteria back to the cluster path** (when single-VM is retired as the
default): the north-star in-cluster control plane supersedes it once the 2026-06-20
GKE-backend audit gaps close (multi-GPU torchrun-wrap, zero-live-GPU-validation,
IaC-render-unproven) *and* the cluster CPU-tier (§5.5) is wired. Until then
single-VM is the *transitional* execution shape; cluster is the *destination*.

## 4. North-star architecture (all five pillars)

Read through the **Layer A / Layer B** split: Layer A (VM lifecycle) is today
*bash*; Layer B (`run.py`) is real Python. The refactor lifts Layer A into a cloud-
and execution-agnostic Python controller on the unified backend interface, inserts
three new seams (triage, provisioning, self-improvement) at its boundaries, and
leaves Layer B mostly intact.

```
┌─ PRODUCT / CONTROL PLANE ─────────────────────────────────── north-star (Approach 3)
│  "Autoresearch" surface · in-cluster orchestrator service · SSE stream · leaderboard
│  Phase 1: CLI-driven (backend.cli reproduce), but library-structured for a service
└─────────────────────────────────────────────────────────────────────────────────────
                                   │  paper + optional experiment prompt + budget
                                   ▼
┌─ LAYER A · ReproductionRun  (lifecycle controller) ─────────── Phase 1 KEYSTONE (NEW)
│  cloud- & execution-agnostic state machine; lifts the SDAR bash into testable Python
│  RESOLVE → TRIAGE → PROVISION_CPU → PROVISION_ASSETS → GREEN_GATE →
│    ACQUIRE_GPU → RUN(Layer B) → WATCH(+sync) → COLLECT → RELEASE_GPU → FINALIZE → TEARDOWN
└─────────────────────────────────────────────────────────────────────────────────────
       │ compute (tiered)           │ assets / env               │ evidence
       ▼                            ▼                            ▼
┌─ EXECUTION BACKENDS ──────┐ ┌─ PROVISIONING ──────────┐ ┌─ SELF-IMPROVEMENT ─────────┐
│ ComputeProvider (NEW)     │ │ FeasibilityTriage (NEW) │ │ ExperienceMemory (advisory)│
│  provision_cpu·acquire_gpu│ │ AssetResolver (generic) │ │  held-out gate over an     │
│  ·release_gpu·collect·… + │ │ EnvironmentAdapter reg. │ │  EvidenceVector (predicate)│
│  ComputeLease{cpu,disk,gpu}│ │  SDAR envs = 1st adapters│ │  FailureAttribution schema │
│  └ VmComputeProvider (P1) │ │ CredentialBroker (NEW)  │ │  wraps lesson/recipe (✓);  │
│    ClusterProvider→north★ │ │ AssetCache (generalized)│ │  global-infra store = new  │
│ RuntimeBackend (ABC, ✓)   │ │ reads SemanticContract  │ │  never mutates mechanics(P1)│
│  CloudProfile{k8s✓ | vm}  │ │ (fallback rubric+claim) │ │  extends failure_classifier │
└───────────────────────────┘ └─────────────────────────┘ └────────────────────────────┘
                                   │
                                   ▼
┌─ LAYER B · RLM reproduction harness  (mostly ✓, consumed) ──────────────────────────
│  root loop · 12+ primitives · fidelity guards · rubric gen · cells route · report
└─────────────────────────────────────────────────────────────────────────────────────
        ✓ = exists today   ·   NEW = built in Phase 1   ·   ★ = north-star
```

**Phase-1 (this spec) vs north-star boundary:**

- **In (Phase 1):** `ReproductionRun` controller · `ComputeProvider` (typed tier
  ops) + **`VmComputeProvider` only** (GCP first) · one `--cloud` flag ·
  `FeasibilityTriage` + `estimate_scope_cost` + `RunBudget` (the gates) ·
  `AssetResolver` + `EnvironmentAdapter` registry + `CredentialBroker` + generalized
  `AssetCache` · advisory `ExperienceMemory` (global-infra + per-paper, `EvidenceVector`
  held-out gate) · **SDAR re-homed onto the abstraction, parity proven by
  characterization tests.**
- **Out (→ north-star):** in-cluster control-plane service · `ClusterComputeProvider`
  (the Job path stays as-is) · **Azure VM provider** (experimental until validated —
  Azure default = cluster) · staged harness self-edit · product-UI beyond today's
  SSE/steering/leaderboard.

### 4.1 Design principles (robust · optimal · dynamic · modular · scalable)

Requirements, each pinned to a concrete mechanism (not adjectives):

- **Robust** — every seam fail-soft (provisioning failure → `Exclusion`; execution
  failure → repairable class); the never-idle/never-burn invariant with **graceful vs
  emergency teardown + periodic off-VM sync + `recover()`** (§5.4); the fabrication
  guards as the evidence backstop; the zero-regression strangler-fig migration proven
  by **characterization tests** (§9/§10); new feature-flags default-OFF ⇒ byte-identical.
- **Optimal (cost)** — CPU tier for download/setup where feasible, GPU leased only for
  the training window via the explicit `acquire_gpu`/`release_gpu` bracket (§5.2/§5.5);
  warm `AssetCache` reuse across runs; `FeasibilityTriage` DOWN_SCOPE to fit budget;
  parallel cells. (GCP on-demand A100 = `stage_on_gpu` until the warm-disk handoff lands.)
- **Dynamic** — `FeasibilityTriage` auto-scopes per paper+budget; dynamic-GPU SKU
  selection (`OPENRESEARCH_DYNAMIC_GPU`) + OOM escalation; cloud + tiering strategy
  chosen at runtime from the `CloudProfile`; adapters resolved by `applies()`; advisory
  memory adapts guidance per run.
- **Modular** — five independently-testable interface seams (`ComputeProvider`,
  `RuntimeBackend`, `EnvironmentAdapter`, `AssetResolver`, `ExperienceMemory`); cloud
  differences localized to a **neutral `CloudProfile`** (not the K8s `CloudSpec`); each
  seam = one responsibility + a fake/guard test.
- **Scalable** — CPU provisioning pools independently of scarce GPU; the cluster/Job
  path fans many concurrent papers over autoscaled nodes with a shared object-store
  cache; the control-plane north-star removes the laptop; the state machine is
  checkpointed/resume-safe so runs survive restarts.

## 5. Phase 1 — unified execution interface + `ReproductionRun`

### 5.1 Two interfaces, not one

The two execution models run the reproduction at different *topologies*: single-VM
hosts Layer B **on the GPU box** (`--sandbox local`); managed-Job runs Layer B on
the orchestrator host and dispatches each **cell** as a Job via `RuntimeBackend`.
What differs is "where the whole run is hosted and how its lifecycle + cost are
managed," not "how a command execs." Forcing the VM path into the per-exec
`RuntimeBackend` would make it a degenerate one-giant-exec and discard the lifecycle
richness.

**Decision:** Phase 1 adds one new interface at the Layer-A boundary —
`ComputeProvider` — and leaves `RuntimeBackend` (per-cell exec) untouched.

### 5.2 `ComputeProvider` (typed tier operations) + a neutral `CloudProfile`

The interface must be able to *enforce* the CPU→GPU billing split (§5.5), so it
exposes explicit tier operations and a typed lease — not a single coarse
`provision()`:

```python
class ComputeProvider(ABC):                          # Layer-A seam (NEW)
    def preflight(self, plan)  -> CapacityReport      # quota/capacity — NO lease, NO billing
    def provision_cpu(self, plan) -> ComputeLease     # cheap CPU tier (+cache disk); NO gpu, ~$0
    def stage(self, lease, bundle, run_spec)          # ship code+spec to the tier
    def acquire_gpu(self, lease) -> ComputeLease      # ONLY after GREEN+triage+budget; ARMS ceiling; billing starts
    def launch(self, lease, run_spec) -> RunHandle    # start Layer B on the GPU tier
    def watch(self, handle) -> Iterator[Status]       # stream status + PERIODIC off-VM artifact sync
    def collect(self, handle) -> ReportBundle         # pull final_report + artifacts
    def release_gpu(self, lease)                      # drop the GPU tier the instant COLLECT ends
    def teardown(self, lease, *, reason)              # release everything; cache disk persists
    def recover(self, lease_ref) -> ReportBundle|None # re-collect from a stopped-but-uncollected VM

# ComputeLease = {cpu: Handle, disk: Handle|None, gpu: Handle|None}   # typed tier state
VmComputeProvider(cloud_profile)     # the Phase-1 lift (GCP first)
# ClusterComputeProvider → north-star; the Job path already works and is NOT wrapped in Phase 1
```

Cloud differences live in a **neutral `CloudProfile`**, NOT the Kubernetes-specific
`CloudSpec` (which carries pod/object-store/base-image fields and belongs to
`k8s_job_backend.py`):

```python
CloudProfile = {                     # neutral, per-cloud — the ComputeProvider reads this
    k8s: K8sCloudSpec | None,         # the EXISTING Kubernetes descriptor, unchanged
    vm:  VmSpec | None,               # VM ops (below), used only by VmComputeProvider
}
VmSpec = (create_cmd, capacity_signatures, cost_ceiling, cache_disk_ops,
          machine_image_ops, ssh_transport, idle_watchdog, tiering_strategy)
#         gcloud … | az …            STOCKOUT|AllocFail   STOP|autoshutdown+budget   …
```

**Honest asymmetry (localized in `VmSpec`):** GCP's `max-run-duration=STOP` is a
control-plane hard ceiling surviving kernel death; Azure has no exact per-VM
equivalent, so Azure's ceiling is assembled from auto-shutdown + budget-action + the
in-VM idle watchdog + wall-clock — which is why **Azure VM is experimental/off in
Phase 1 and Azure defaults to the cluster path** (§5.5, §11).

### 5.3 The `ReproductionRun` state machine (the bash, lifted)

One checkpointed, resume-safe, fail-soft state machine. `ACQUIRE_GPU` is reachable
**only after** triage + budget pass (§6.1) and GREEN passes — enforcing "no GPU
before the gates."

| State | Does | Replaces (SDAR bash) | Fail-soft |
|---|---|---|---|
| `RESOLVE` | build `RunPlan` (paper, repo spec, scope, budget, **required_assets**); load advisory memory | env-var assembly | — |
| `TRIAGE` | `FeasibilityTriage` → PROCEED / DOWN_SCOPE / PLAN_ONLY (no GPU, needs creds to probe) | *(none — new)* | PLAN_ONLY ⇒ plan report, no lease |
| `PROVISION_CPU` | `provision_cpu`: cheap CPU tier + cache disk | preflight + cache-disk attach | wait for capacity; idempotent |
| `PROVISION_ASSETS` | `AssetResolver` + adapters download data / warm models / stand up envs — **CPU only** | `prepare` / `sdar_gcp_assets.py` | missing asset ⇒ exclusion, not fail |
| `GREEN_GATE` | deterministic readiness (data present, import ok, GPU-free config resolve) — still CPU | the `[GREEN]/[RED]` gate | RED ⇒ down-scope or abort **before any GPU lease** |
| `ACQUIRE_GPU` | `acquire_gpu` (gated on TRIAGE+budget+GREEN); **arm cost ceiling**; strategy per `VmSpec.tiering_strategy` | machine-type flip / capacity-poll + max-run-duration | capacity-poll; never stop a RUNNING VM; **GPU billing starts HERE** |
| `RUN` | `launch` Layer B (VM=`local` \| cluster=Jobs); inject memory + guards | detached `sdar_gcp_run.sh` | — |
| `WATCH` | stream status; **periodic off-VM artifact sync**; enforce idle/stall/wall-clock/cost guards | `sdar_gcp_watch.sh` + watchdog | terminal or guard-stop |
| `COLLECT` | pull `final_report` + artifacts | `pull_report_and_log` | periodic-sync + `recover()` backstop |
| `RELEASE_GPU` | `release_gpu` the instant COLLECT ends (keep CPU/disk if needed) | *(new — tighten the window)* | idempotent |
| `FINALIZE` | persist report; feed `ExperienceMemory` (evidence-gated) | ledger log | admission gated on evidence |
| `TEARDOWN` | release everything; halt billing; disk persists | `vm_stop` | idempotent; safe on crash; `recover()` on re-entry |

**Strategy-parameterized (honesty note):** `PROVISION_CPU` / `PROVISION_ASSETS` /
`GREEN_GATE` run **GPU-free only under a strategy that provides a real CPU tier**
(`machine_type_flip`, `cpu_warm_disk_then_gpu_attach`). Under `stage_on_gpu` (the GCP
on-demand A100 default today, §5.5) there is no separate cheap tier — these states
execute on the GPU VM and `ACQUIRE_GPU` folds into `PROVISION_CPU`, so a small
GPU-warm cost applies until the Phase-1d warm-disk handoff lands.

### 5.4 The "never idle / never burn" invariant — graceful vs emergency

The bash guardrail layers become properties the controller owns, split honestly into
two teardown regimes (the current watcher only pulls on terminal-finalized states,
exits if the VM is already non-running, and can leave a VM running on budget
exhaustion — `sdar_gcp_watch.sh:282,353` — so "never strands" must be *engineered*,
not asserted):

- **Graceful teardown** (normal terminal): `COLLECT` → `RELEASE_GPU` → `TEARDOWN`.
- **Emergency shutdown** (budget-kill / watchdog / preemption / kernel death): the
  control-plane cost ceiling (GCP `max-run-duration=STOP`) and the in-VM idle
  watchdog stop the VM *unconditionally*. Because that path can bypass `COLLECT`,
  the report is protected by **periodic off-VM artifact sync during `WATCH`** (the
  latest `final_report`/metrics/ledger are already off-VM) plus a **`recover()`
  re-collect** for a stopped-but-uncollected VM (the boot disk persists). So an
  emergency stop degrades to "recover the last synced state," never "strand."
- **Budget** — a `RunBudget` (`max_usd` / `max_wall_clock_seconds` / `max_run_gpu_usd`
  in `budget.py`, + an optional new `max_gpu_hours`) fixed at `RESOLVE` is checked at
  `TRIAGE` (feasibility), `GREEN_GATE` (pre-GPU abort), and `WATCH` (live) — extending
  `OPENRESEARCH_MAX_RUN_GPU_USD`. On live-budget breach, emergency shutdown fires after
  a sync.

### 5.5 CPU-provisioning / GPU-only-when-needed tiering

Dataset download and environment setup are CPU-bound and should not burn GPU hours
*where the cloud allows a cheap CPU tier*. So compute is provisioned in **two tiers**
and the GPU tier is leased only for `[ACQUIRE_GPU, RELEASE_GPU]`. How the tiers
connect is `VmSpec.tiering_strategy`, chosen dynamically — and, critically, its
availability differs by cloud/SKU:

| Strategy | Mechanism | Status / when |
|---|---|---|
| **`stage_on_gpu`** | provision on the GPU type, warm there | **GCP on-demand A100 default for Phase 1.** The maintenance-policy coupling *blocks* a CPU stage (GPU forces `onHostMaintenance=TERMINATE`, e2 needs `MIGRATE`, no valid intermediate); the current script literally stages on the GPU VM (`gcp_sdar_preflight.sh:386`, `sdar_gcp_optimal_run.sh:274`). Accepts a small GPU-warm cost — honest, and what runs today. |
| **`machine_type_flip`** | one VM: stop → retype CPU→GPU → start | GCP **spot** (the repo already flips e2↔a2 for prep vs. launch); **Azure** cross-family SKU resize (no maintenance coupling). Reuses the boot disk. |
| **`cpu_warm_disk_then_gpu_attach`** | a cheap CPU VM warms the persistent `AssetCache` disk, detaches; the GPU VM attaches it | **The target real CPU-tier for GCP on-demand A100 — a NEW two-VM handoff to build + validate (Phase 1d), NOT yet the default.** Same-zone-locked (a zonal disk attaches only in its own zone), so the CPU warmer, the disk, and the A100 must co-locate in a zone that *has* on-demand A100 capacity — the strategy and the `ACQUIRE_GPU` capacity-poll share a zone. (Azure managed disks are same-*region*, cross-AZ OK.) |

- **The cluster/Job model needs work in BOTH runtime paths for a CPU tier, not one:**
  the shared object-store cache (GCS/Blob) is already reusable, so the CPU tier can
  warm it **out-of-band**; but a true in-cluster no-GPU provisioning Job requires
  `gpu_count=0` (skip the `nvidia.com/gpu` request/toleration/nodeSelector) in **both**
  `k8s_job_backend.py::_build_job_manifest` (floors at 1 today, `:205`) **and**
  `k8s_job_cell_runner.py` (also always adds GPU selection/toleration/requests, `:705`),
  plus CPU node-pool selection. This is north-star work; Phase 1 warms out-of-band.
- The GPU cost ceiling (§5.4) is armed at `ACQUIRE_GPU`, and `RELEASE_GPU` drops the
  GPU tier the instant `COLLECT` finishes — so the GPU billing window is the minimum
  the tiering strategy allows.

## 6. Phase 1 — provisioning + the triage gate

### 6.1 `FeasibilityTriage` (deterministic cost gate)

Runs at `TRIAGE`, before any GPU lease, off three deterministic probes (no LLM in
the decision). Its input is `RunPlan.required_assets`, extracted (at `RESOLVE`/`TRIAGE`)
from the **`SemanticReproductionContract`** (`semantic_contract.py`, gated by
`OPENRESEARCH_REPRO_CONTRACT`) — its structured `resource_identities` (typed
`kind`/`identifier`) and `capability_profile` (datasets / frameworks /
external_services) — **falling back to the rubric + claim-map when the flag is off**
(triage must degrade gracefully, never assume the contract is present). Triage and
the generic resolver read the *same* `required_assets`.

| Axis | Deterministic probe (no download, no GPU) | Outcomes |
|---|---|---|
| **Data reachability** | HF Hub metadata API · HTTP/S3/GDrive HEAD · registry lookup — **uses `CredentialBroker`** (a *gated* resource returns 401/403, not 404, only with creds) | reachable / gated(needs-cred) / missing / too-large-vs-disk |
| **Compute feasibility** | scope (models×datasets×seeds×steps) × per-cell cost model vs `RunBudget` | within / down-scope-to-fit / infeasible |
| **Env stand-ability** | adapter registry `applies()` · generic-resolvable heuristic | adapter / generic / needs-adapter |

> **New component (not reuse):** the compute-feasibility axis needs an
> `estimate_scope_cost(scope, sku) → (gpu_hours, usd)` that `gpu_catalog` /
> `gpu_capacity` do **not** provide today — they estimate per-cell *VRAM*, not
> wall-clock. Phase 1b adds it: a deterministic `est_train_seconds` per cell (heuristic
> on model-size × steps, optionally refined by a bounded one-time LLM estimate) × scope
> expansion (|models|×|datasets|×|seeds|) × SKU `$/hr`. It need only be *conservative*
> (under-scope, never over-lease) because the live `RunBudget` check at `WATCH` (§5.4)
> is the true backstop.

Combined decision:

- **PROCEED** — all green, or the only gaps are fail-soft-excludable.
- **DOWN_SCOPE** — auto-trim `ScopeSpec` to the feasible subset (smallest model that
  fits budget, reachable datasets, fewer seeds) and proceed. The automatic answer to a
  512-GPU paper: run the biggest slice the budget allows, declare the rest omitted.
- **PLAN_ONLY** — a *blocking* gap (gated data with no credential, compute ≫ budget
  even minimal, env needs an unwritten adapter): emit an honest **"reproduction plan
  + why-we-didn't-run + what-would-unblock-it"** report, **never lease a GPU.**

Property: `TRIAGE` (pre-lease, probe-based) and `GREEN_GATE` (post-provision,
disk-based) are the same feasibility question at two fidelities.

### 6.2 The provisioning seam — generic Tier 1, optional Tier 2

Strict resolution order per required asset/env:

> **registered `EnvironmentAdapter` (if `applies`) → generic `AssetResolver` → fail-soft `Exclusion`.**

**Tier 1 — `AssetResolver` (generic, default, zero per-paper code):**

- Data/models: HF snapshot/datasets · URL/S3/GDrive fetch with retries · torchvision &
  standard registries (extends `dataset_recipes.py`) — all into the shared cache.
- Env/deps: generalizes `environment_detective` so the CUDA/Python matrix is
  data-driven (not the hardcoded `_FRAMEWORK_COMPATIBILITY`); carries the `env_pin`
  cu121-core fix; infers apt/system libs from deps.
- **`CredentialBroker` (a scoped dependency of resolve + triage + stage + provision, not
  provision-only):** brokers configured secrets (HF_TOKEN for gated HF, the paper's own
  S3/Kaggle creds) to whichever stage needs them — triage's reachability probes need
  them to tell *gated* from *missing*. **No raw `.env` on the wire, and no secret-shaped
  value in any staged bundle or log** (redaction tests enforce this — closing the
  current `.env`-SCP gap). A needed-but-absent credential → a `gated` exclusion, never a
  hang.
- **`AssetCache`:** `EnvCacheManager` generalized — host-shared, `fcntl`-locked,
  crash-safe, idempotent, keyed by asset identity. The warm `sdar-ultra` disk made
  paper-agnostic. VM path = attached cache disk; cluster path = shared object store.
- Fail-soft: any unresolved asset → a verified `Exclusion` (fairness principle), never
  a fake-0.

**Tier 2 — `EnvironmentAdapter` registry (optional, for what Tier 1 can't stand up):**

```python
class EnvironmentAdapter(ABC):
    key: str
    def applies(self, plan) -> bool                 # does this adapter fire for this paper/env?
    def provision(self, ctx) -> ProvisionResult     # stand up env + data (may use its own venv)
    def smoke(self, ctx)     -> SmokeResult         # cheap liveness (the served>0 signal)
    def health(self, ctx)    -> HealthReport         # runtime env_health.jsonl
```

SDAR's three envs become the first three adapters — `AlfworldAdapter` /
`WebShopAdapter` / `SearchQaAdapter` are today's `env_cache.py` logic refactored
behind this interface, **behavior-preserving (proven by characterization tests, §9)**.
Every hard-won special-case (the dedicated py3.10 venv, `werkzeug<2.1`, the in-process
WebShop backend, the 132 GB FAISS index, `alfworld-download`-by-abspath,
`_alfworld_has_games` re-verify) becomes an adapter internal.

**Concurrency note:** the adapters run inside `EnvCacheManager`'s single host-level
`fcntl` lock (`env_cache.py:409`) — provisioning is **serialized** across envs by
design (WebShop is ref-counted; ALFWorld/Search-QA downloads are expensive/huge), so
`EnvironmentAdapter.provision()` must not assume it runs concurrently with a sibling.

**Runtime coupling:** provisioning stays wired to the existing `ENV_LIVENESS_GATE` /
`env_health.jsonl served>0` machinery — an env that provisions but serves zero becomes
an exclusion, not a fake-0.

## 7. Phase 1 — self-improvement (`ExperienceMemory`, advisory)

Grounded in **Self-Harness (arXiv:2606.09498)** and the surrounding literature
(Reflexion 2303.11366, Voyager 2305.16291, ExpeL 2308.10144, ADAS 2408.08435,
AgentDebug 2509.25370, memory-drift 2605.12978). `ExperienceMemory` **wraps** the
existing per-paper `_lessons` (`lesson_distiller`) + gated `recipes`
(`recipe_library`) — it does **not** replace them; the only genuinely new store is the
**global-infra** one.

**Two stores by scope (decision 7):**

- **Global infra store** (`runs/_memory/infra/`, NEW) — cross-paper, paper-invariant
  fixes keyed by `FailureAttribution.signature`, not arxiv_id (the `werkzeug` pin, the
  cu121 core, gdown retries, alfworld-verify).
- **Per-paper store** (existing `_lessons` + recipes, wrapped) — method-specific
  scope/recipes.

**The safety boundary needs a real schema, not an assertion.** Today
`failure_classifier` returns only `(failure_class, suggested_fix)`. Phase 1 adds a
`FailureAttribution{signature, root_cause, scope ∈ {infra, method}, confidence,
evidence_refs}` (root-cause attribution per AgentDebug — first-decisive-error, not
substring). `scope` is the routing key: `infra` → global store, `method` → per-paper —
**enforced by a test that a method-scoped lesson can never enter global memory.**

**Lifecycle:**

| Stage | Mechanism | Extends | Red-line discipline |
|---|---|---|---|
| **Capture** | `FailureAttribution` (root-cause) for failures; successes → recipe | `failure_classifier`, `failure_capsule`, `recipe_library` | admission reads the **`EvidenceVector`**, never a scalar grade |
| **Store** | weakness-mining: cluster rows into signatures; promote on recurrence≥2 + dedup + caps + staleness | `lesson_distiller` | mined from deterministic rows |
| **Retrieve** | top-k by relevance (env/framework fingerprint ‖ arxiv_id); bounded ≤5 / ≤200c | existing caps | never inject-all (ExpeL bloat anti-pattern) |
| **Apply** | injected as agent/provisioning **hints only** | `_negative_lessons_block` + recipe injection | **advisory — never auto-mutates run mechanics in Phase 1** |

**Keystone — the non-regressive held-out gate over an `EvidenceVector`, not a scalar.**
The trust signal is the SAME multi-predicate evidence layer the harness already
enforces (`recipe_library.py:424`): an `EvidenceVector` of deterministic predicates —
`{provenance_present, metrics_not_all_zero_or_constant, external_validator_verdict
(veto), deterministic meets_target, ledger_success_row}` — **never a scalar grade.** A
candidate lesson/recipe is promoted to **active** only if, replayed against a fixed
held-out replay set, **every** held-out predicate is non-regressed (the validator
verdict is an absolute veto) and ≥1 improves, with no held-in predicate regressing;
rejects are logged, never applied.

The gate needs an **executable artifact contract** (specified before
`held_out_gate.py`): `ReplayCase{id, resources (allowed, CPU/cheap-tier only),
apply(lesson)→env, expected_predicates}`, `CandidateLesson{attribution, patch,
admission_state}`, a fixed held-in/held-out split, and the promotion store. Phase-1
pragmatics: the replay set starts tiny (a couple of cached runs + a golden toy repro);
infra lessons gate on the CPU/cheap tier (no GPU); **until replay evidence exists a
lesson stays candidate/advisory-low-confidence, never promoted** (fail-soft).

**North-star tier (opt-in, NOT Phase-1 default):** staged **harness self-edit** — the
harness proposes minimal edits to its own prompt/primitive config (Self-Harness
proposal), gated by the *same* `EvidenceVector` held-out replay. Off by default because
it exceeds "advisory."

**Anti-patterns → guardrails (from the survey):** lesson-poisoning → `EvidenceVector`
admission + validator veto (never a scalar); memory bloat/drift → caps + dedup +
staleness + top-k; overfit/reward-hacking → held-out non-regression + the `scope` axis
+ deterministic hard-to-hack predicates.

## 8. Error-handling philosophy — two regimes

- **Provisioning/infra failure** (env won't stand up, dataset gated) → **fail-soft to a
  verified `Exclusion`** (fairness principle), never a fake-0, never a hard abort →
  feeds global-infra memory.
- **Reproduction/execution failure** (cell errors, OOM, stall) → **repairable failure
  class** → bounded evidence-fingerprint-keyed repair loop (existing forced-iteration +
  `REPAIR_MAX_ITERATIONS`) → honest terminal if unrepairable → feeds per-paper +
  capsule memory.
- **The fabrication guards remain the backstop** (zero-metrics / stub / eval-provenance
  / evidence-gate / no-learning-signal): SI can only *propose* a better evidence state,
  never *assert* one.
- **New feature-flags default-OFF ⇒ byte-identical** (flipped per-flag after the A/B +
  grader-σ gates). **This does NOT cover the refactors** (adapter extraction, `AssetCache`
  generalization): those change imports/tests even behind flags, so they are proven
  behavior-preserving by **characterization tests** (§9/§10), not by a byte-identical claim.

## 9. Testing strategy

Porting the bash into Python without regression, and refactoring `env_cache` behind
adapters, are the primary risks — tests are first-class:

- **Characterization tests of the OLD SDAR path FIRST** — capture current behavior of
  `env_cache`/provisioning + the bash lifecycle *before* refactoring, so the adapter
  refactor and the `VmComputeProvider` lift are proven behavior-preserving (this is what
  "zero regression" means operationally, not a byte-identical assertion).
- **Hermetic unit/guard tests** for every pure component: `FeasibilityTriage` (probe
  fixtures → decisions), `estimate_scope_cost` (scope → conservative cost), the `VmSpec`
  command builders (assert exact `gcloud`/`az` argv), the `ReproductionRun` state machine
  (transitions incl. `ACQUIRE_GPU`-gated-on-triage, `recover()`, every fail-soft branch),
  `AssetResolver`/adapters (fixtures → exclusions), `CredentialBroker` (**no secret in
  any staged bundle or log** — redaction test), and `ExperienceMemory` (**a
  fabricated-`EvidenceVector` lesson is REJECTED; a held-out-predicate-regressing lesson
  is REJECTED; a `method`-scoped lesson never enters global memory**).
- **`FakeVmComputeProvider`** — simulates stockout → provision_cpu → acquire_gpu →
  watch-to-terminal → collect → release_gpu → teardown → recover, so the *entire*
  lifecycle + guardrails (incl. emergency-shutdown + `recover()`) are tested with **zero
  cloud spend** (socket-hermetic; `pytest-socket` blocks non-loopback).
- **SDAR golden-command characterization test** — assert `VmComputeProvider(gcp)` emits
  the *same* effective gcloud lifecycle as the current bash before it touches a real A100.
- **Live GPU smoke stays operator-gated** (never CI), same discipline as `gke_check.sh`.

## 10. SDAR migration — strangler-fig, zero-regression

The live SDAR run must not break.

1. **Characterize first:** capture the current `env_cache`/provisioning + bash-lifecycle
   behavior as characterization tests (§9) *before* touching them.
2. **SDAR's 3 envs move behind the `EnvironmentAdapter` interface** — proven
   behavior-preserving by those tests (not merely "re-point" the tests).
3. **`VmComputeProvider` built alongside the bash, behind a flag** (`--via-controller` /
   `OPENRESEARCH_UNIFIED_RUN=1`); unset = today's bash, byte-identical.
4. **Prove parity** via the golden-command test + one operator-gated paired A/B (bash vs
   controller, SDAR 1-model) — the ≥3-paired-run discipline the repo mandates.
5. **Flip default → thin/retire the bash** only after parity holds. Bash stays the
   fallback until then.

## 11. Phased roadmap

Ordering enforces the gates-before-GPU and asset-list-before-resolver dependencies.

- **Phase 1a — provisioning-seam refactor (lowest risk, no external deps):**
  characterization tests → `EnvironmentAdapter` + SDAR's 3 adapters + generalized
  `AssetCache`. Scoped to the refactor; does **not** need the generic asset list.
- **Phase 1b — the gates:** `RunPlan.required_assets` extraction (`SemanticReproductionContract`,
  fallback rubric/claim-map) + `FeasibilityTriage` + `estimate_scope_cost` + `RunBudget`
  (extended `max_gpu_hours`). Deterministic, no compute.
- **Phase 1c — unified execution:** `ComputeProvider` (typed tier ops) + neutral
  `CloudProfile` + `VmComputeProvider(gcp)` with **`stage_on_gpu` default** +
  `ReproductionRun` state machine (`ACQUIRE_GPU` gated on 1b) + `FakeVmComputeProvider`
  tests + golden-command parity + periodic-sync/`recover()`. GCP only.
- **Phase 1d — provisioning + credentials + real CPU-tier:** `CredentialBroker` (scoped
  to resolve/triage/stage/provision; no raw `.env`) + generic `AssetResolver` (consumes
  1b's asset list) + build & **validate** the `cpu_warm_disk_then_gpu_attach` two-VM
  handoff (then it may become the GCP on-demand default).
- **Phase 1e — advisory `ExperienceMemory`:** `FailureAttribution` schema + global-infra
  store (wraps existing memory) + the `EvidenceVector` held-out gate + `ReplayCase`/
  `CandidateLesson` contracts.
- **Phase 1f — SDAR cutover:** A/B parity → flip default → thin the bash.
- **North-star (Phase 2+):** in-cluster control-plane service (closes the 2026-06-20
  GKE-backend audit gaps — multi-GPU torchrun-wrap, zero-live-GPU-validation,
  IaC-render-unproven — first) · `ClusterComputeProvider` + the cluster CPU-tier
  (`gpu_count=0` in both k8s paths) · **validated Azure VM** (live deallocate +
  budget-action latency test) · staged harness self-edit · alphaxiv-style product surface.

## 12. Non-goals (Phase 1)

- No in-cluster control-plane service (CLI-driven, but library-structured).
- **No `ClusterComputeProvider`** — the managed-Job path stays as-is; only `VmComputeProvider`
  ships in Phase 1.
- **No production Azure VM** — Azure VM is experimental/default-off until a live deallocate
  + budget-action latency test; Azure default = the managed cluster path.
- No staged harness self-edit (advisory memory only).
- No product-UI work beyond today's SSE/steering/leaderboard.

## 13. Component → file map

| Component | New/extends | Where |
|---|---|---|
| `ComputeProvider` (typed tier ops) + `VmComputeProvider` | NEW | `backend/services/runtime/compute_provider.py` (peer to `interface.py`); `ClusterComputeProvider` = north-star |
| `CloudProfile{k8s, vm}` + `VmSpec` | NEW (neutral) | `backend/services/runtime/cloud_profile.py` — wraps the existing `K8sCloudSpec`; does **not** extend `k8s_job_backend.py`'s `CloudSpec` |
| `ReproductionRun` state machine | NEW | `backend/agents/rlm/reproduction_run.py` (wraps existing `run.py` Layer B) |
| `RunBudget` | extends | `backend/agents/resilience/budget.py` — `max_usd`/`max_wall_clock_seconds`/`max_run_gpu_usd`/`check_run_gpu_usd` exist; add optional `max_gpu_hours` |
| `RunPlan.required_assets` extraction | NEW | reads `SemanticReproductionContract.resource_identities`/`capability_profile`; fallback rubric+claim-map |
| `FeasibilityTriage` + `estimate_scope_cost`/`est_train_seconds` | NEW | `backend/services/runtime/feasibility_triage.py` (+ the wall-clock cost model `gpu_catalog` lacks) |
| `AssetResolver` / `AssetCache` / `CredentialBroker` | NEW/generalizes | `backend/services/runtime/asset_resolver.py` (generalizes `env_cache.py`/`asset_provisioning.py`/`dataset_recipes.py`); `CredentialBroker` scoped to resolve/triage/stage/provision |
| `EnvironmentAdapter` + SDAR adapters | NEW | `backend/services/runtime/env_adapters/` (refactors `env_cache.py`) |
| `ExperienceMemory` (wraps existing) + global-infra store | NEW/orchestrates | `backend/agents/rlm/experience_memory.py` over `lesson_distiller`/`recipe_library`/`failure_capsule`; only the global-infra store is a new surface |
| `FailureAttribution{signature,root_cause,scope,confidence,evidence_refs}` | extends | `backend/agents/rlm/failure_classifier.py` (today returns only `(class, fix)`) |
| `EvidenceVector` held-out gate + `ReplayCase`/`CandidateLesson` | NEW | `backend/agents/rlm/held_out_gate.py` (predicate-level, validator-veto) |
| Cluster CPU-only manifests (`gpu_count=0`) | extends (north-star) | **both** `k8s_job_backend.py::_build_job_manifest` **and** `k8s_job_cell_runner.py` |

## 14. References

- **alphaxiv Autoresearch** — product reference (`alphaxiv_autoresearc.png`).
- **Self-Harness**, arXiv:2606.09498 — the self-improving-harness loop + non-regressive
  held-out gate.
- Reflexion 2303.11366 · Voyager 2305.16291 · ExpeL 2308.10144 · ADAS 2408.08435 ·
  AgentDebug 2509.25370 (root-cause attribution) · memory-drift 2605.12978 (bloat/drift).
- `backend/agents/rlm/recipe_library.py` (`:424`) — the multi-predicate evidence red line
  the `EvidenceVector` gate must match (evidence gate + ledger row + validator + target).
- `docs/superpowers/specs/2026-06-16-gcp-gke-execution-backend-design.md` — the Job
  backend + `CloudSpec` (K8s-specific) this keeps separate from `VmSpec`.
- The 2026-06-17 multi-cloud IaC spec (reconciled in §3.1) + the 2026-06-20 GKE-backend
  audit (the gap list the north-star must close).
- `docs/superpowers/specs/2026-06-20-grounded-self-improvement-and-harness-reliability-redesign-design.md`
  — the evidence-not-grade red line + fabrication guards this rides on.
- `docs/runbooks/2026-07-01-sdar-gcp-reproduction-walkthrough.md` — the bash being lifted.

## 15. Open questions / risks

1. **Bash-port fidelity** — the golden-command test + A/B must prove parity; the
   capacity-poll, the emergency-shutdown/`recover()` path, and the periodic-sync cadence
   are the highest-risk to port.
2. **GCP on-demand A100 CPU-tier** — the real cheap-CPU tier for on-demand A100 depends on
   the un-built `cpu_warm_disk_then_gpu_attach` two-VM handoff (Phase 1d); until it is
   built + validated, on-demand A100 pays a small GPU-warm cost (`stage_on_gpu`).
3. **Azure VM cost-safety** — Azure's ceiling (auto-shutdown+budget) is weaker than GCP's
   control-plane STOP; Azure VM stays experimental/off and Azure defaults to the cluster
   path until a live deallocate + budget-action latency test passes.
4. **Triage compute estimate (new work)** — `estimate_scope_cost` does not exist today
   (`gpu_catalog` estimates VRAM, not runtime); it need only be conservative because the
   live `WATCH` budget check is the true backstop.
5. **Contract dependency** — `FeasibilityTriage` degrades to rubric+claim-map when
   `OPENRESEARCH_REPRO_CONTRACT` is off; it must never assume the contract is present.
6. **Held-out replay corpus** — the SI gate is only as good as the replay set; Phase 1e
   ships a minimal set and it grows over runs; until then lessons stay
   advisory-low-confidence.

## 16. Review resolution (Codex adversarial review, 2026-07-01)

Verdict at v1 was **needs rework**; all 16 findings resolved in v2.

| # | Finding (Codex) | Resolution |
|---|---|---|
| BLOCKER | Phase graph leases GPU before triage/budget exist (§11 vs §2/§5.3) | Reordered §11: 1a refactor → **1b gates** → 1c execution; `ACQUIRE_GPU` gated on 1b (§5.3) |
| BLOCKER | `ComputeProvider` can't express the CPU→GPU lease split | Split into typed tier ops (`provision_cpu`/`acquire_gpu`/`release_gpu`/`collect`/`recover`) + `ComputeLease{cpu,disk,gpu}` (§5.2) |
| BLOCKER | GCP on-demand default contradicts the cited code | Default = **`stage_on_gpu`**; `cpu_warm_disk_then_gpu_attach` is a Phase-1d build-and-validate (§5.5) |
| BLOCKER | Collect-before-teardown overclaimed | Graceful vs emergency teardown + **periodic off-VM sync + `recover()`** (§5.4) |
| BLOCKER | §3.1 single-VM reconciliation not airtight | Reframed as a **scoped transitional reversal**: CredentialBroker-before-live-VM precondition + explicit exit criteria (§3.1) |
| MAJOR | `CloudSpec.vm` crosses the K8s boundary | Neutral **`CloudProfile{k8s, vm}`** module; K8s `CloudSpec` untouched (§5.2/§13) |
| MAJOR | Cluster CPU-tier gap understated (one path) | Requires `gpu_count=0` in **both** `k8s_job_backend` **and** `k8s_job_cell_runner` (§5.5/§13) — north-star |
| MAJOR | Generic provisioning sequenced before the asset list | `required_assets` extraction moved to **1b**; 1a scoped to the adapter refactor (§6.1/§11) |
| MAJOR | `CredentialBroker` too narrow | Scoped to **resolve+triage+stage+provision**; no raw `.env`; redaction tests (§6.1/§6.2) |
| MAJOR | SI gate weakens the evidence red line (scalar) | Gate compares an **`EvidenceVector`** (predicate-level + validator veto), never a scalar (§7) |
| MAJOR | infra-vs-method boundary unspecified | Concrete **`FailureAttribution`** schema + a test that method lessons never go global (§7) |
| MAJOR | Held-out gate lacks an artifact contract | Specified **`ReplayCase`/`CandidateLesson`** + split + promotion store (§7) |
| MAJOR | Azure VM cost-unsafe | Azure VM **experimental/off**; Azure default = cluster (§5.5/§11/§12) |
| MINOR | `ClusterComputeProvider` YAGNI for Phase 1 | **Cut** from Phase 1 → north-star; only `VmComputeProvider` ships (§5.2/§12) |
| MINOR | "byte-identical" too broad for refactors | Scoped to new flags; refactors need **characterization tests** (§8/§9/§10) |
| MINOR | `ExperienceMemory` duplicates existing memory | **Wraps** existing `_lessons`/recipes; only the global-infra store is new (§7/§13) |
