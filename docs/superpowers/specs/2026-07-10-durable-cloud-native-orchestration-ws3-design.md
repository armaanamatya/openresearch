# Durable Cloud-Native Orchestration (WS3) — laptop-as-launcher-only, SaaS-grade

- **Date:** 2026-07-10
- **Status:** Design (grounded in a full read-only recon of the GCP orchestration internals; supersedes the Track-D §3.1 controller sketch, which failed adversarial review on three counts).
- **Track:** D / WS3. Prereq: none (independent of WS1 eval-integrity). Enables: WS4 (CPU-class durable lane runs on this same backbone).
- **Operator intent (verbatim):** "everything works purely on cloud/vm … the laptop will not always be here … production grade 24/7 robust … integrated into a SaaS … 100% purely vm … build this for scale, robust, modular."

## 1. Problem

Today the reproduction **driver runs in-process on the operator's laptop** — `live_runs.py:1019` `Popen`s `python -c "cmd_reproduce"`, authenticated by the operator's *personal* ADC + kubeconfig. A laptop sleep / disconnect / Claude-Code teardown kills the run mid-training (forensics: `prj_adam_gcp_1/2` SIGTERM, `prj_adam_gcp_3`/`prj_c912` orphaned). This is the single biggest barrier to "unattended on cloud" and it is a **plumbing** problem, not a modeling one.

The Track-D §3.1 sketch ("in-cluster controller + lease") is directionally right but **failed adversarial review** — its lease primitive doesn't exist, a dead controller isn't resurrected, and it split-brains the GPU. This spec closes all three with the actual seams the recon found.

## 2. The load-bearing insight: most of this is already built, dormant

| Capability | Status today | Recon anchor |
|---|---|---|
| **GCS compare-and-swap substrate** | `google-cloud-storage 3.12` supports `if_generation_match` + `blob.generation` + delete-with-precondition — **100% unused** | `gcs_blob.py:277` (unconditional `upload_from_string`) |
| **In-cluster controller skeleton** | `orchestrator-{deployment,cronjob}.yaml` + KSA + Workload-Identity + RBAC — exists, `enabled:false`, never load-tested, wired to `reproduce` not `campaign` | `infra/gcp/helm/templates/`, `values.yaml:192` |
| **Reaper permission** | RBAC `Role reprolab-orchestrator` already grants `create/get/list/watch/**delete**` on `batch/jobs` — **unused** (no code ever calls `list_namespaced_job`) | `helm/templates/role.yaml:19-21` |
| **A resumable driver unit** | `ReproductionCampaign.run()` is already disk-ledger-resumable (reads `campaign.json`, dispatches on state) | `reproduction_campaign.py:379-396` |
| **Reschedule-aware resume** | `_maybe_auto_arm_cell_resume` + `STABLE_RUN_ID`/`CELL_RESUME_AUTO` were *built for the rescheduled-pod case* (docstring says so), default-OFF | `run.py:2751-2789`, `primitives.py:7106-7111` |
| **Deterministic cell Job names** | cell path names are already `(cell_id, run_id)`-deterministic (adoptable) | `k8s_job_cell_runner.py:499-507` |

So WS3 = **(a) 4 genuinely-new pieces** + **(b) turn on + harden the dormant machinery** — the same discipline as WS1/WS2.

The 4 genuinely-new pieces:
1. A **GCS generation-CAS lease** (single-writer, TTL/heartbeat).
2. **Fencing** — stamp the lease generation into cell Job names + result-blob paths + persisted absolute-epoch deadlines; **adopt-by-name on 409** instead of erroring.
3. **GCS-mirrored campaign ledger** so a rescheduled pod (fresh local disk) resumes instead of re-running.
4. The **API/SaaS path reaching `campaign`** (today `live_runs.py` only Popens single-shot `reproduce`).

## 3. Goals / Non-goals

**Goals**
- G1. A `sandbox=gcp` run **survives the launcher dying** — the driver executes as a controller in-cluster (or on a controller VM); the laptop is a stateless launcher that may disconnect immediately after submit.
- G2. **Exactly one writer** per run at any instant (no split-brain), enforced by a real CAS lease — a superseded driver can neither write the evidence bus nor keep GPU Jobs alive.
- G3. **A dead controller is resurrected**, not silently lost — restart adopts in-flight Jobs by name and resumes from the last checkpoint; no duplicate GPU submit, no A100 cost-leak.
- G4. **Flag-gated, default-OFF, byte-identical when off**; the durable path is opt-in via `OPENRESEARCH_DURABLE_CONTROLLER` until the durability drill passes, then default-ON for `sandbox=gcp`.

**Non-goals**
- Azure (deferred, per operator). Design the lease + fencing **cloud-agnostically** (a `BlobLease` interface) so Azure is a later adapter, but ship only the GCS impl.
- Changing the verdict/eval (WS1 owns that).
- CPU-class routing (WS4 — but WS3 is its prerequisite; keep the controller GPU-count-agnostic).

## 4. Design

### 4.1 `BlobLease` — the single-writer generation-CAS lease (new)

A new pure module `backend/services/runtime/blob_lease.py` on top of a small `gcs_blob` extension:

- **Extend `gcs_blob.upload_bytes`** to accept `if_generation_match: int | None` and **return the written `blob.generation`**; add `read_bytes_with_generation() -> (data, generation)`. (No new dependency — the SDK already supports it; `gcs_blob.py:277` is the only line to touch, plus the `GcsStore` passthrough at `k8s_job_backend.py:363-382`.)
- `BlobLease.acquire(run_id) -> LeaseToken` — read `runs/<id>/rlm_state/owner.lease` (generation `g`); write it back with `if_generation_match=g` (or `=0` to create). On `PreconditionFailed`, someone else won → return `None` (caller must not drive). The returned `LeaseToken.generation` is the **fence token**.
- `renew(token)` — heartbeat: re-write with `if_generation_match=token.generation`, advancing the generation; a stale holder's next renew fails-closed → it exits without writing.
- `is_current(token)` — cheap check a driver calls **before every state/evidence write and before every Job submit**: reads the lease generation and confirms it still equals the token's chain. A driver that finds a newer generation **exits without writing** (closes the split-brain-on-driver hazard).
- TTL: a lease older than `LEASE_TTL_S` (heartbeat interval × 3) with no renewal is acquirable by a successor.

**This is the primitive Track-D §3.1 assumed existed.** It's ~120 lines + a hermetic test using a fake generation-tracking blob double.

### 4.2 Fencing the *work*, not just the driver (closes split-brain-on-GPU)

The lease alone fences the driver; the recon shows GPU Jobs + result blobs are keyed by `run_id` only, so a suspended-not-dead driver's Job races a successor's. Fix by threading the **lease generation** into the work:

- **Cell Job name** (`k8s_job_cell_runner._job_name`, already deterministic at `:499`): fold the lease generation → `reprolab-cell-<run>-<cell>-g<gen>` (collision-safe under the 63-char cap via a short hash when needed). A superseded owner's Jobs carry an older `g<gen>` and cannot collide with the successor's.
- **Result-blob path** (`run_matrix` `output_blob_prefix`, `:1643`): `runs/<id>/gen-<gen>/cells/<cell>/…` so two generations' writes never overwrite the same `metrics.json`. The orchestrator reads the **current generation's** prefix.
- **Adopt-by-name on 409** (`k8s_job_cell_runner.py:1323-1332`): replace the bare `except Exception` with an `ApiException.status==409` branch → `read_namespaced_job_status(name)` → attach to `_watch_job` instead of `STATUS_ERROR`. A restart with the same `(run_id, cell_id, gen)` re-attaches to the running Job rather than duplicating it.
- **Persisted absolute-epoch deadlines**: `_watch_job`'s `time.monotonic()+timeout` (`:879`/`k8s_job_backend.py:960`) is replaced by an absolute epoch persisted to the run bucket at submit, re-read on restart — so an adopted run inherits its *remaining* budget, not a fresh full one (closes the "double the GPU wall-clock" leak).
- **Write-ahead submit-intent**: before `create_namespaced_job`, persist `(run_id, cell_id, gen, job_name)` to `runs/<id>/gen-<gen>/intents/` (CAS-fenced). A controller that crashes after submit but before recording finds the intent + the deterministically-named Job on restart and adopts it.

### 4.3 The durable controller — resurrect the dormant skeleton (closes "dead controller not resurrected")

- **Wire the existing `orchestrator-deployment.yaml`** (`enabled:false` today) to run `python -m backend.cli campaign <paper> --project-id <stable> --resume …` (NOT `reproduce`). `ReproductionCampaign.run()` is already ledger-resumable — the correct `main()`.
- **Self-heal:** `restartPolicy: OnFailure` + `backoffLimit > 0` on the controller pod; because `run()` re-reads the ledger and dispatches on state, a restarted controller **resumes**, not restarts. A `Deployment` (not bare `Job`) keeps the pod respawned across node preemption.
- **Distinguish exits:** `run()` returning mid-campaign is a *checkpoint/PAUSED* exit (CLAUDE.md exit-code-2), not terminal. The controller wrapper maps: exit-0-terminal → complete; exit-2-paused/checkpoint → the ledger holds `paused`, a `ScheduleWakeup`/CronJob re-enters; non-zero-crash → `OnFailure` respawn + lease-guarded resume.
- **GCS-mirror the ledger** (closes the recon's "rescheduled pod re-runs everything"): the campaign ledger + `rlm_state/` are checkpoint-mirrored to `runs/<id>/` in the bucket; on a fresh pod, the controller hydrates from GCS before dispatching. **Promote `_try_reconcile_status` (GCS `status.json`) to the PRIMARY resume-skip source** ahead of the local `cell_manifest.json` (`cell_scheduler.should_skip_cell:159`) — so a pod with empty local disk still skips already-`ok` cells.
- **Reaper:** on lease acquisition, the controller uses the **already-granted** `list/watch/delete batch/jobs` RBAC to enumerate `label_selector=run_id=<id>` Jobs of an **older generation** and delete them (closes the A100 cost-leak — orphaned Jobs no longer bill until `activeDeadlineSeconds`). This is the first code in the repo to call `list_namespaced_job`; the permission exists.
- **Identity:** the controller runs under the `reprolab-orchestrator` KSA (Workload-Identity → the orchestrator GSA: GCS objectAdmin + the RBAC Role). Creds from Workload-Identity, never baked in. A missing precondition (KSA unbound, bucket unwritable, egress blocked) **fails loud at controller-submit**, not mid-run.

### 4.4 Launcher + SaaS path

- The launcher (CLI/UI) submits the controller Deployment, records its handle + the run bucket in `demo_status.json`, acquires the lease **only** to hand off (writes the controller as intended owner), then releases and may disconnect. `--resume`/reconnect reads authoritative state from the bucket and may drive **only** if it can acquire the lease.
- **SaaS reachability:** `live_runs.py` (today `Popen reproduce`, `:1019`) gains a `sandbox=gcp` branch that submits the controller instead — so the product's upload→live-reproduction path gets durability, not just the CLI. Off-flag → the existing `Popen reproduce` path, byte-identical.
- **Fallback (no cluster):** when no controller host is available, the local driver runs under `setsid` (not `nohup` — dies on Claude-Code teardown) with `OPENRESEARCH_HARDEXIT_CLEANUP` on + periodic GCS-mirrored checkpoints, so a relaunch resumes. This is the degraded path, not the target.

## 5. Interfaces (new/changed)

```python
# gcs_blob.py — extend (no new dep)
def upload_bytes(..., if_generation_match: int | None = None) -> int   # returns new generation
def read_bytes_with_generation(...) -> tuple[bytes, int]

# blob_lease.py (new) — cloud-agnostic interface, GCS impl only
class BlobLease:
    def acquire(self, run_id: str) -> LeaseToken | None
    def renew(self, token: LeaseToken) -> LeaseToken | None   # None ⇒ superseded, stop
    def is_current(self, token: LeaseToken) -> bool
    def reap_older_generations(self, run_id: str, token: LeaseToken) -> int  # deletes stale Jobs

# k8s_job_cell_runner.py — changed
def _job_name(cell_id, run_id="", gen: int | None = None) -> str          # +generation fence
# 409 → adopt_by_name(name) → _watch_job   (was: STATUS_ERROR)
```

Everything gated on `OPENRESEARCH_DURABLE_CONTROLLER`; off ⇒ byte-identical (uuid names, monotonic deadlines, local-disk resume, `Popen reproduce`).

## 6. Testing & acceptance

- **Durability drill (headline):** launch a `sandbox=gcp` run, **kill the controller pod** (not just the launcher) mid-training → a successor adopts the in-flight cell Job by name, resumes from the GCS-mirrored ledger, and finalizes a real metric. Today the run dies; success = it finishes.
- **Split-brain drill:** hold a stale lease token, attempt a state/evidence write and a Job submit → both refused (`is_current` false); a superseded generation's blob writes land under `gen-<old>/` and never overwrite the current generation.
- **Reaper drill:** an older-generation Job is deleted on lease acquisition; no A100 pod bills past the takeover.
- **Adopt-not-duplicate:** resubmit a cell whose Job is still live → 409 → adopted (one Job), not a duplicate; adopted run inherits remaining budget, not a fresh one.
- **CAS unit tests** (hermetic, fake generation-tracking blob): two concurrent `acquire` — exactly one wins; a stale `renew` fails-closed.
- **Off-state byte-identical:** `OPENRESEARCH_DURABLE_CONTROLLER=0` reproduces today's names/deadlines/resume/Popen path exactly.

## 7. Rollout

1. `gcs_blob` CAS extension + `blob_lease.py` + hermetic tests (no cluster).
2. Fencing (job-name gen, blob-path gen, adopt-on-409, persisted-epoch, write-ahead intent) behind the flag.
3. Controller wiring (turn on `orchestrator-deployment` → `campaign --project-id --resume`; GCS-mirror ledger; reaper; GCS-status as primary resume source).
4. Durability drill on a real 1×A100 cheap paper → then default-ON for `sandbox=gcp`, and the `live_runs.py` SaaS branch.

## 8. Risks

- **R1 — controller pod itself preempted:** `Deployment` + `OnFailure` respawns it; the lease + adopt-by-name + GCS ledger make respawn a resume. The failure the §3.1 sketch ignored is the drill's explicit target.
- **R2 — GCS-mirror latency / partial checkpoint:** checkpoints are CAS-fenced and atomic (tmp+precondition-replace); a torn write is superseded by the next generation, never adopted.
- **R3 — the dormant skeleton has never run:** treat it as scaffolding, not a validated base; the durability drill is the gate, not its mere existence.
- **R4 — cost visibility on the GPU path** stays partial; but the controller now *owns* node request/teardown + reaps orphans, making node lifetime observable — the prerequisite for a later cost-accounting spec.
