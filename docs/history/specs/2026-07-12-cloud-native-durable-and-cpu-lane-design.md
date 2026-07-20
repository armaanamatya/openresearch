# Cloud-Native Durable Reproduction + CPU Cloud Lane — Design

- **Date:** 2026-07-12
- **Branch:** `feat/gke-gpu-path-reproduction-reliability` (continues the WS3 durable-controller work already in the working tree)
- **Status:** Design approved (brainstorming); **Revision 2** incorporates a Codex robustness review (§12).
  Next: writing-plans → implementation under subagent guardrails.
- **Predecessor specs:** `docs/history/specs/2026-07-10-durable-cloud-native-orchestration-ws3-design.md`,
  `docs/history/specs/2026-07-01-paper-agnostic-multicloud-reproduction-and-self-improvement-design.md`
- **Handoff context:** `docs/runbooks/2026-07-11-unified-platform-phase1-2-implementation-handoff.md`

---

## 1. Problem statement

Two gaps block "reproduction runs fully on cloud, no laptop, for any paper":

1. **The durable controller is a stub wired into the live path.** `live_runs.py`'s
   `_start_python_run` already calls `_submit_durable_controller(...)` when
   `_should_use_durable_controller(request)` is true (flag on + `sandbox == "gcp"`), but the
   submit body is `raise NotImplementedError(...)`. Flipping `OPENRESEARCH_DURABLE_CONTROLLER`
   on today would crash every gcp run at start. The WS3 **primitives** exist and are unit-tested
   (`BlobLease` CAS lease in `blob_lease.py`, generation-fenced job naming + deadline helpers in
   `k8s_job_cell_runner.py`, `run_controller.py` command/exit/lease helpers) — what is missing is
   the orchestration that consumes them safely.

2. **CPU-class papers have no cloud path.** The GKE job spec in `k8s_job_backend.py`
   unconditionally adds the `nvidia.com/gpu` toleration, a `reprolab/sku` GPU nodeSelector, and
   `requests/limits nvidia.com/gpu` with a `max(1, gpu_count)` clamp (`k8s_job_backend.py:314`); the
   in-pod entrypoint coerces `OPENRESEARCH_CELL_GPU_COUNT=0 → 1`. A CPU-only reproduction (e.g. Adam,
   arXiv 1412.6980 — proven at `best_runs/adam` 0.831 but only on `sandbox=local`) cannot be
   scheduled on cloud; the executor runs training in-process on the orchestrator host, the exact
   SDK-`aclose`-stall that sank three prior Adam GKE attempts.

**Goal:** make the durable controller real, safe, and ON-by-default for `sandbox=gcp`; add a
dedicated CPU Job lane so CPU-class papers run on cloud by default — both degrading to today's
exact behavior on any failure, and both metric-neutral when they succeed.

## 2. The reconciling principle — operational vs score-affecting defaults

`CLAUDE.md` requires a default-flip to clear **≥3 paired A/B runs + the grader-σ gate**. That rule
governs **score-affecting** defaults. Neither change here touches *what is measured* — they change
*where compute runs*. They are **operational** defaults. The honest safety bar:

> **Every new remote path MUST degrade to today's exact local behavior on any failure, and MUST be
> measurement-identical when it succeeds.** Worst case = today's behavior + a loud warning.

Consequences:
- Default-ON is acceptable **without** the grader-σ gate, because the paths are metric-neutral and
  fail-soft.
- The **reliability** trust — dropping the fallback "training wheels" — is what the operator Pod-kill
  drill certifies. Until it passes, the fallback stays and we do not claim "proven".
- **Metric neutrality is drill-certified, not asserted.** A unit test proves the CPU cloud cell
  issues the identical cell command; the *drill* proves a CPU cloud Adam run reproduces the
  `best_runs/adam` baseline within tolerance (§9). Environments can differ (image/torch/numpy/CUDA),
  so command-identity is necessary, not sufficient.

## 3. Component map (each fake-testable, socket-hermetic)

### 3.0 Fence model — stable fence epoch vs CAS generation (correctness foundation)
`BlobLease.renew()` advances the GCS object generation every heartbeat; `reap_older_generations`
deletes Jobs whose generation `< token.generation`. Using the CAS generation as the Job fence would
make a controller **reap its own still-running Jobs after its first renewal.** So the lease payload
carries a **stable `fence_epoch`** distinct from the CAS generation:
- `fence_epoch` is bumped **only on a real takeover** (acquire by a *different* owner, or after TTL
  expiry) and is **preserved across same-owner reacquire/renew**.
- `LeaseToken` exposes both `generation` (CAS version, for `if_generation_match`) and `fence_epoch`
  (stable, for Job naming/reaping).
- All fenced Job names embed `fence_epoch`; `reap_older_generations` compares against `fence_epoch`,
  never the CAS generation.
This is a required change to `blob_lease.py` (payload + `LeaseToken` + reaper key) with its own
hermetic tests (renew preserves `fence_epoch`; cross-owner takeover bumps it).

### 3.1 `controller_launch.py` (new, pure)
`build_controller_job_manifest(*, paper, project_id, fence_epoch, image, cpu_pool_label, namespace,
env, service_account, backoff_limit) -> dict`. Pure dict builder (no cloud calls) for a fenced K8s
**Job** whose container runs the controller entrypoint (§3.6).
- **Job, not Deployment:** a campaign is finite. `backoffLimit = N` (default 3) gives restart-on-crash.
- **Stable identity across restarts:** `owner_id = project_id`; a restarted Pod reacquires the same
  lease (preserving `fence_epoch`).
- **CPU placement:** reuses the CPU-job shape (§3.5) — CPU nodeSelector, **no** `nvidia.com/gpu`,
  **no** GPU taint toleration.
- **Fenced name:** `fence_epoch`-embedded so the reaper can parse `(job_name, fence_epoch)`.

### 3.2 `_submit_durable_controller` — real body (`live_runs.py`)
Replaces the `NotImplementedError`. All I/O injected via a `_ControllerCluster` seam
(`acquire_lease`, `is_current`, `list_jobs`, `delete_job`, `delete_job_confirmed`, `submit_job`,
`wait_ready`, `now`). **Ordering is takeover-safe: submit and confirm the successor BEFORE reaping
the predecessor.**
```
token = lease.acquire(project_id, owner_id=project_id, now_epoch=now())
if token is None:
    return await self.get_run(project_id)                # another driver owns it — idempotent adopt
if not lease.is_current(token):                          # lost between acquire and use → adopt, do NOT fallback
    return await self.get_run(project_id)
manifest = controller_launch.build_controller_job_manifest(..., fence_epoch=token.fence_epoch, ...)
submit_job(manifest)                                     # pre-submit failures fall through to fallback (§3.3)
ready = wait_ready(job_name, timeout_s=READY_TIMEOUT_S)
if not ready:
    delete_job_confirmed(job_name)                       # confirm gone by UID; if unconfirmed → raise _ControllerStuck (fail loud, NO local fallback)
    raise _ControllerNotReady(...)                       # caught by fallback ONLY because no remote work is live
if lease.is_current(token):
    reap_older_generations(project_id, token, ...)       # reap predecessor's older-fence_epoch Jobs AFTER successor is ready
record handle {controller_job, fence_epoch, submitted_epoch} into demo_status.json (CAS on fence_epoch)
return LiveRunState(... pid=None, controller=<handle> ...)
```
`is_current` gates guard against two API replicas both acquiring; the loser adopts current state and
never submits or writes a handle.

### 3.3 Graceful fallback (`_start_python_run`) — pre-submit only
```
if self._should_use_durable_controller(request):
    try:
        return await self._submit_durable_controller(...)
    except _ControllerStuck:
        raise                                            # remote work may be live → fail loud, never split-brain
    except Exception as exc:                             # pre-submit / not-ready-with-confirmed-delete only
        logger.warning("durable_controller_fallback: %s", exc)
        emit run_warning code="durable_controller_fallback"
        # fall through to the existing subprocess.Popen path
```
**Invariant: local fallback is only reachable when no remote controller Job is live** — either the
failure was pre-submit, or the submitted Job's deletion was confirmed. If a submitted Job cannot be
confirmed deleted, we fail loud (`_ControllerStuck`) rather than risk a local+remote split-brain
writing the same GCS run dir.

### 3.4 `cpu_class.py` (new, pure)
`requires_gpu(cell: dict, *, trusted_cpu: bool) -> bool` — deterministic per-cell classifier with
**hard GPU signals overriding a soft CPU declaration** (an `accelerator` field is usually
agent-authored and must not be able to downgrade a real GPU paper):
1. **Hard GPU signals win:** `est_vram_gb > 0`, a GPU framework (`image_key`/`framework` in the
   known-GPU set, e.g. `verl`), or a distributed marker ⇒ **GPU** (even if `accelerator=="cpu"`;
   emit a `cpu_gpu_conflict` warning).
2. Else a **trusted** CPU declaration (`trusted_cpu=True` — from an operator/run-spec source, not
   agent prose) or `accelerator=="cpu"` with no hard GPU signal ⇒ **CPU**.
3. **Conservative default: unknown ⇒ GPU.**
A run is CPU-class only when *every* runnable cell is CPU-class; mixed matrices keep the GPU path.

### 3.5 CPU Job branch (`k8s_job_backend.py` + `k8s_job_cell_runner.py`)
An explicit `accelerator="cpu"` lane parameter (**not** a `gpu_count==0` sentinel — that collides
with the `max(1, …)` clamp and the entrypoint's `0→1` coercion) branches **before all GPU clamps**:
- **omit** the `nvidia.com/gpu` toleration, the `reprolab/sku` nodeSelector, the
  `requests/limits nvidia.com/gpu` block, GPU billing, and GPU-count env semantics
  (`OPENRESEARCH_CELL_GPU_COUNT` is not emitted on the CPU lane);
- **add** a CPU `nodeSelector = {cpu_pool_label}` and explicit CPU/memory `requests`;
- share everything else (image, code-staging via `_command_needs_staged_code`, env, fenced name).
- **GPU manifests are byte-identical** (golden-compare test) — the CPU branch is additive + gated.

### 3.6 Controller entrypoint (new subcommand / wrapper)
`build_controller_command` today runs plain `python -m backend.cli campaign … --resume`, which cannot
renew the lease or thread the fence into launched Jobs. Add a thin controller wrapper entrypoint that:
acquires/renews the lease on the heartbeat cadence, exports the **stable `fence_epoch`** into the
child `RunContext` (threaded into `k8s_job_cell_runner.bind_run_context` so training Jobs it launches
carry `fence_epoch`), and drives the campaign. `build_controller_command` points at this wrapper.

### 3.7 Controller handle in live-run state (`live_runs.py`)
`LiveRunState` today has no controller field and idempotency keys on a live `pid`. Add a typed
`controller` handle field; `_load_run`, the stop/liveness paths, and SSE serialization treat
`status=="running"` + a present `controller` handle as **active** (a durable run legitimately has
`pid=None`), so a durable run is never restarted or archived out from under itself.

## 4. Data flow

**Start a gcp run (durable, default-ON):** `POST /runs` → `_start_python_run` →
`_should_use_durable_controller` (gcp + not opted-out) → `_submit_durable_controller` (§3.2, submit
→ ready → reap → record handle) → any *pre-submit* failure falls back to Popen (§3.3). The controller
Pod renews the lease every 60 s (`LEASE_TTL_S = 180`), preserving `fence_epoch`; training Jobs it
launches carry `fence_epoch` and are reaped by any successor that takes over.

**A CPU-class paper (Adam) on gcp:** `run_experiment` → cells → `cpu_class.requires_gpu`? GPU cells
take the existing path; CPU cells submit to a CPU Job. Because `run_matrix` reports infra failures as
per-cell `STATUS_ERROR` **dicts** (not exceptions), the lane detects an **all-cells infra failure**
and reruns the same cells locally with a `cpu_cloud_fallback` warning (§5).

## 5. Error handling & invariants

- **Fail-soft to local, detected correctly.** Durable: pre-submit only (§3.3). CPU lane: an
  all-cells infra failure is detected from the returned per-cell status dicts (K8s init/upload/
  manifest/submit/Pending), then the same cells rerun locally — a `try/except` alone would miss these.
- **Never split-brain.** Local fallback is unreachable once a remote Job is live and unconfirmed-deleted.
- **Metric neutrality is layered:** unit test = identical cell command; drill = CPU cloud Adam
  matches `best_runs/adam` within tolerance. Command-identity is necessary, not sufficient — the
  image digest / torch / numpy / CUDA-visibility parity is what the drill certifies.
- **Lease correctness is inherited where safe, extended where needed.** `acquire`/`renew`/`is_current`
  are consumed as-is; `fence_epoch` (§3.0) is the one required extension.
- **No silent caps.** The absent autonomous resubmit sweeper (§7) is logged, not implied away.

## 6. Flags & default semantics

- `durable_controller_enabled()` stays **env-only, default-false** (unchanged global helper). The
  default-ON decision lives at the launch boundary: a new sandbox-aware
  `durable_controller_default_for_sandbox(sandbox, request)` (used only in
  `_should_use_durable_controller`) returns **True for `sandbox=="gcp"` unless explicitly opted out**
  via `OPENRESEARCH_DURABLE_CONTROLLER=0`. This keeps cell-fencing helpers that read the global flag
  unaffected.
- `OPENRESEARCH_CPU_CLOUD_CELLS` (new) — **default-ON for `sandbox=gcp`**; `0` keeps CPU cells local.
- `OPENRESEARCH_CONTROLLER_READY_TIMEOUT_S` (180) · `OPENRESEARCH_CPU_POOL_LABEL`
  (`reprolab/pool=cpu`) · `OPENRESEARCH_CONTROLLER_BACKOFF_LIMIT` (3).
- All default-ON behavior is gated behind `sandbox == "gcp"` **and** the fail-soft fallback, so a
  non-gcp or misconfigured environment is byte-identical to today.

## 7. Scope — v1 vs follow-on

**In v1:** the stable fence epoch (§3.0); real controller submit + lease/reap/readiness/handle with
takeover-safe ordering; controller entrypoint (§3.6) + handle state (§3.7); graceful pre-submit
fallback; controller Job builder; CPU-class classifier; CPU Job lane + infra-failure fallback; both
default-ON-for-gcp flags; hermetic tests; operator runbook.

**Explicit follow-on (logged, not implied):**
- **Autonomous dead-controller resubmit sweeper** — v1 takeover is via K8s Job `backoffLimit` restart
  reacquiring the lease + the operator drill. (Approved scope cut.)
- **Azure/AKS parity** for both lanes — GCS-only lease per the WS3 non-goal.

## 8. Operator boundary (an agent cannot do these)

Ships code + hermetic tests; these live steps are the operator's (exact commands in the companion
runbook):
1. **Provision one CPU node pool** (serves both the controller and CPU cells), scale-to-zero:
   ```
   gcloud container node-pools create cpu-pool \
     --cluster openresearch-gpu --region us-central1 \
     --machine-type e2-standard-8 --num-nodes 0 --enable-autoscaling \
     --min-nodes 0 --max-nodes 3 --node-labels reprolab/pool=cpu
   ```
2. **Controller KSA RBAC** — GCS access + `batch/jobs` create/list/delete (Workload Identity binding
   on `reprolab-sa`, per the 2026-07-07 bring-up).
3. **The Pod-kill durability drill** — kill the controller Pod mid-run on live hardware; confirm a
   successor reacquires the lease (preserving `fence_epoch`), reaps the predecessor's older-fence Jobs,
   and resumes the same lineage with **no local fallback and no old-generation evidence loss**. Also
   confirm a CPU cloud Adam run matches `best_runs/adam` within tolerance. Passing this certifies
   "proven" and authorizes dropping the fallback.

## 9. Testing (all hermetic, socket-hermetic per repo policy)

Fakes: `BlobLease` double (with `fence_epoch`), the `_ControllerCluster` seam, an injected clock.
- **Fence model (§3.0):** renew preserves `fence_epoch` while advancing CAS generation; cross-owner
  takeover bumps `fence_epoch`; reaper keys on `fence_epoch`, never CAS generation (regression for
  "controller reaps its own Jobs after a heartbeat").
- **Durable submit:** acquire→submit→ready→reap→record handle; lease `None` or lost-`is_current` →
  idempotent adopt (no submit, no fallback); **two concurrent same-owner submits** → exactly one
  submits, the other adopts; **submit raises pre-ready** → Popen fallback + warning; **readiness
  timeout with confirmed delete** → fallback; **readiness timeout with unconfirmable delete** →
  `_ControllerStuck` (fail loud, no fallback).
- **Controller handle idempotency (§3.7):** a running durable run with `pid=None` + live handle is
  not restarted/archived.
- **Fence threading (§3.6):** the controller entrypoint exports `fence_epoch` into the child
  `RunContext`; launched cell Jobs carry it.
- **CPU classifier (§3.4):** hard GPU signal overrides `accelerator="cpu"` (+warning); trusted CPU
  declaration → CPU; unknown → GPU; mixed matrix → GPU run.
- **CPU Job manifest (§3.5):** no `nvidia.com/gpu`, CPU nodeSelector + CPU/mem requests present, no
  `OPENRESEARCH_CELL_GPU_COUNT`; **GPU manifest byte-identical** (golden compare).
- **CPU lane fallback (§5):** all-cells `STATUS_ERROR` dict → same cells rerun locally +
  `cpu_cloud_fallback` warning (NOT aggregated as an experiment failure).
- **Metric neutrality:** CPU cloud cell command == local cell command (unit); Adam-tolerance is the
  drill's job (§8).
- **Off-state:** both flags `0` ⇒ byte-identical (durable → Popen; CPU cells → local).
- **Tripwires after any touched file:** `tests/agents/rlm/test_single_verdict_authority_guard.py`
  and `tests/rlm/test_registry.py` (still **19** primitives — no new primitive).

## 10. File-level owner plan (for the implementation phase)

Disjoint owners per the clobber-safe rule; hot files get one owner per phase, lead reviews + commits.

| File | Change | Kind |
|---|---|---|
| `backend/services/runtime/blob_lease.py` | add stable `fence_epoch` (§3.0) | hot file (1 owner) |
| `backend/agents/rlm/controller_launch.py` | new pure Job builder | new module |
| `backend/agents/rlm/cpu_class.py` | new pure classifier | new module |
| controller entrypoint (§3.6) | new wrapper subcommand + `run_controller.build_controller_command` retarget | new + small |
| `backend/services/events/live_runs.py` | real `_submit_durable_controller` + fallback + `controller` handle state + `durable_controller_default_for_sandbox` | hot file (1 owner) |
| `backend/services/runtime/k8s_job_backend.py` | `accelerator="cpu"` manifest branch | hot file (1 owner) |
| `backend/agents/rlm/k8s_job_cell_runner.py` | CPU routing + fence threading in `run_matrix` | hot file (1 owner) |
| `backend/agents/rlm/feature_flags.py` | new flag readers | small |
| `docs/runbooks/2026-07-12-cpu-lane-and-durable-drill-operator-checklist.md` | operator commands + drill | new doc |
| `tests/...` | OFF+ON pairs per §9 | new tests |

Subagent guardrails (verbatim in every implementer prompt): forbidden git state commands; write only
an explicit allowlist; never edit/delete an existing test; new capability = `env_truthy` flag,
default-safe, with a hermetic OFF+ON test pair; return a structured summary; the lead verifies the
git footprint and re-runs tests before trusting it.

## 11. Sequencing (correctness dependencies)

1. **`blob_lease.py` fence epoch first** — everything fences on it.
2. **Controller entrypoint + `controller_launch.py`** — need the fence to build/thread.
3. **`_submit_durable_controller` + handle state** — consume 1–2.
4. **`cpu_class.py` + CPU Job branch** — independent of 1–3 except they share the CPU pool label;
   can proceed in parallel with a disjoint owner.
5. **Flag defaults + fallbacks** — last, once both lanes are green off-state.

## 12. Changelog — Codex review incorporated (Revision 2)

A Codex design review (grounded in `blob_lease.py`, `run_controller.py`, `live_runs.py`,
`k8s_job_backend.py`, `k8s_job_cell_runner.py`) surfaced 11 findings; all accepted:
- **Fence ≠ CAS generation (blocker):** `renew()` advances generation → a controller would reap its
  own Jobs → introduced the stable `fence_epoch` (§3.0), reaper keys on it (§3.2, §9).
- **Missing `is_current` gates / concurrent same-owner submit (blocker):** added `is_current` gates
  before submit/reap/record + CAS handle write; loser adopts, never fallbacks (§3.2, §9).
- **Reap/delete ordering & split-brain (blocker):** submit→ready→reap; local fallback pre-submit
  only; unconfirmable delete → fail loud (§3.2, §3.3).
- **Fence not threaded into campaign execution (major):** added the controller entrypoint (§3.6).
- **No controller handle in `LiveRunState` (major):** added handle state + active-run semantics (§3.7).
- **Default-ON flag boundary (major):** global helper stays default-false; a sandbox-aware launch
  boundary owns the default (§6).
- **CPU fail-soft can't use exceptions (major):** detect all-cells `STATUS_ERROR` dicts → local
  rerun (§5).
- **`gpu_count==0` collides with clamps (major):** use an explicit `accelerator="cpu"` lane (§3.5).
- **Metric neutrality overclaimed (major):** command-identity unit test + drill-certified Adam
  tolerance (§2, §5, §8).
- **`accelerator="cpu"` could downgrade GPU work (major):** hard GPU signals override the soft CPU
  declaration (§3.4).
- **Test gaps (major):** §9 expanded to cover every new failure mode.
