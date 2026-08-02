<!-- doc-meta: status=current; last-verified=2026-07-23 -->
# Tier-3: end-to-end scheduler, validated by an ADAM scheduler A/B — design

**Date:** 2026-07-22 · **Author:** operator + Claude · **Scope:** Tier-3 (approved) — base
pipeline green → scheduler applied → real-cloud ADAM A/B + operator gate.

> **STATUS 2026-07-23 — sub-projects A + B DONE; C blocked on one precisely-characterized gap.**
> A (clean GCP-VM run path) and B (scheduler APPLIES freeze/promote/revive from verified receipts,
> LOCAL, hermetic, default-OFF) are complete and green. **C (the billed ADAM A/B) has been attempted
> LIVE on a GCP A100 twice and both fail-closed at the same gate — correctly:**
> - `adam_treat2_grok_20260723` (executor=grok) → EXHAUSTED / `SchedulerRuntimeError`, 0 receipts.
> - `adam_treat3_gpt_20260723` (executor=`gpt-chat-latest`, Azure Foundry reasoning model — confirmed
>   the true executor via `tokens_total.json`) → **identical** EXHAUSTED / `SchedulerRuntimeError`.
>
> **Root cause (empirical, not a config typo):** the branch reproduction never emits the 5-component
> checkpoint (`checkpoints/step_N/{model,optimizer,lr_scheduler,rng,data_order}`), so `build_raw_receipt`
> fails CLOSED (evidence-not-grade red line — working as designed). The model reuses the SDAR workspace
> scaffold, writes a monolithic `train.py` + template `metrics.json`, and GPU stays 0% (no real training).
> **Decisive finding: swapping grok → a reasoning model did NOT fix it** — guidance-in-prompt alone won't
> produce the cell-matrix checkpoint contract. **Model choice is not the lever.** The fix is HARNESS-FORCED
> emission: a pre-scaffolded `train_cell.py` stub wired to `cell_checkpoint.write_checkpoint(...)` at the
> rung steps, or `OPENRESEARCH_GKE_SYNTH_CELL`. Secondary robustness gap: one branch's missing checkpoint
> raises through the cohort loop and aborts the WHOLE campaign (never runs branches 2-4 / attempts 2-3) —
> consider per-branch fail-isolation. Full trail: `docs/progress/2026-07-22-tier3-adam-progress.md`
> (2026-07-23 07:0x entry); memory `tier3-phaseab-scheduler-applies-2026-07-22`.

## Relationship to prior work

Builds directly on **`2026-07-22-restore-gcp-vm-path-surgical-degke-design.md`** (the
GCP-without-GKE single-VM path). That spec is the foundation of sub-project A here. This
spec **supersedes its GKE posture**: the operator has escalated "GKE parked" → **"GKE is
never used; docs must always say so"** (see memory `no-gke-directive`).

## Goals / exit criteria

The ADAM paper (arXiv 1412.6980, Kingma & Ba) reproduces end-to-end on the **GCP single-VM**
path, run as a **fixed-budget A/B**:

- **Control:** multi-branch campaign, scheduler authority **OFF**.
- **Treatment:** identical campaign + budget, scheduler authority **ON** (branch / freeze /
  promote / revive applied from verified receipts).
- **Headline deliverable:** a scorecard comparing **best rubric score** (primary) + GPU$ +
  wall-clock, plus the ON-arm branch tree (what froze / promoted / revived).

**Definition of done, per sub-project:**
- **A:** one clean ADAM (or small paper) reproduction on one GCP VM with
  `evidence_gate_passed: true` and a `success=True` experiment row.
- **B:** a **local** multi-branch run where one branch freezes, another promotes, and a
  frozen one revives — all from verified receipts, not fixtures; `SCHEDULER_AUTHORITATIVE`
  default-OFF and byte-identical when off.
- **C:** the ADAM fixed-budget A/B scorecard on real GCP single-VM.

## Non-goals

- **No GKE / no Kubernetes for the GCP path.** Azure/AKS + AWS/EKS keep the shared K8s
  runner (removing it would break Azure, ~40 tests, ~100 config keys — see prior spec §Decision).
- **No autonomous default-ON flip** of scheduler authority. CLAUDE.md gates that behind ≥3
  paired A/B + grader-σ + operator sign-off; the verdict stays evidence-gated. This spec
  delivers everything *up to* the flip; the flip is the operator's.
- No net-new receipt transport — reuse the existing transport-agnostic receipt schema.

## GKE posture (operator directive — escalated)

- GKE is **not a supported execution path**. Add a **fail-closed guard** so GKE cannot be
  selected via `--sandbox gcp/gke` on the reproduction path (redirect to the VM/campaign
  path with a clear error), backed by a test. Gated so **Azure/AKS is unaffected**.
- Every cloud-posture doc (root `CLAUDE.md`, `backend/services/runtime/CLAUDE.md`,
  `backend/agents/rlm/CLAUDE.md`, `docs/*`, runbooks) must state **"GKE is not used"** —
  not "parked."
- **Reversible default:** keep the inert GKE modules (`gke_job_backend.py`,
  `gke_cell_synth.py`, `infra/gcp/`) in the tree, guarded + doc-hardened, rather than a
  risky full code rip. (Operator may later request full deletion.)

## Architecture — the load-bearing insight

The GCP single-VM path (`VmComputeProvider`) runs the reproduction **inside the VM as
`--sandbox local`** (prior spec §Architecture). The receipt-gated authority runtime is
**transport-agnostic**: `scheduler_evidence.py:188` — *"A cell runner must supply its
already-materialized metric, checkpoint, evidence bundle."* And a **local** cell runner
already exists: `backend/agents/rlm/gpu_cell_runner.py` (produces `cell_manifest.json`),
distinct from `k8s_job_cell_runner.py`.

**Therefore receipts are sourced from the local cell path — no K8s, no GKE.** Sub-project B
is "construct the controller + emit receipts from `gpu_cell_runner` at rungs + let ASHA
apply," not "invent a VM receipt transport."

A `BranchRungReceipt` (`scheduler_evidence.py:80-128`) binds: `metric_value`+`metric_sha256`,
`checkpoint_path`+`checkpoint_sha256` (5-field resumable checkpoint: model / optimizer /
lr_scheduler / rng / data_order), `evidence_bundle_sha256`, code/dataset/run_spec
fingerprints, `ladder_sha256`, `termination_cause` (only `training_diverged` → true-kill).
The **checkpoint** requirement is exactly sub-project A's resume work — A is a hard
prerequisite for B.

## Sub-project A — base pipeline lands a clean run on GCP single-VM

Foundation is the prior spec (restore VM path via `campaign --campaign-driver unified
--sandbox local --billing-sandbox gcp`). Additional work here:

1. **Cell completion:** ensure the local cell run flips `status: running → completed`
   (`train.py` flips only at end-of-run) and the watcher records a `success=True`
   experiment row + `evidence_gate_passed: true`. (Root cause of the recent `failed` runs
   was cells never completing, not grading.)
2. **Checkpoint + resume:** materialize the 5-field resumable checkpoint at rung boundaries
   and resume from it (stable run-id; `OPENRESEARCH_CELL_RESUME_AUTO`,
   `OPENRESEARCH_STABLE_RUN_ID`) so a stopped branch continues rather than restarting. This
   is the freeze/revive substrate **and** the fix for the ~9.6h-wasted-on-restart defect.
3. **Arm existing default-OFF salvage flags** where they apply to the local/VM path
   (`OPENRESEARCH_CELL_ERROR_SALVAGE`).
4. **GKE fail-closed guard + doc rewrite** (see GKE posture).

**Exit:** ADAM reproduces once, clean, on one GCP VM; hermetic tests green.

## Sub-project B — wire the scheduler to *apply* (local, then behind the flag)

1. **Construct `SchedulerAuthorityController`** in the campaign decide seam
   (`campaign_composition.py`, alongside the existing shadow `_maybe_attach_asha_advisory` /
   `_maybe_apply_asha_authority`).
2. **Emit receipts** from `gpu_cell_runner` at each rung: materialize metric + checkpoint +
   evidence bundle, then call `controller.record_cell_receipt(...)` with the controller-attested
   `attest` callback.
3. **Let ASHA apply** promote / freeze / kill / revive across a multi-branch attempt cohort.
   Branches run **serially on one VM**, checkpoint-frozen between rungs; freeze = checkpoint
   + stop; revive = restore + resume (`scheduler_runtime.revive_branch`).
4. **Flag discipline:** `OPENRESEARCH_SCHEDULER_TREE` (shadow) and
   `OPENRESEARCH_SCHEDULER_AUTHORITATIVE` (apply) stay **default-OFF**; OFF is byte-identical.
   Authority governs **scheduling only** — never the deterministic verdict.

**Exit:** a purely **local** multi-branch run (tiny model, zero cloud cost) demonstrating a
freeze, a promote, and a revive from verified receipts.

## Sub-project C — ADAM fixed-budget A/B on real GCP single-VM

- **Branches** = distinct hyperparameter/approach configs the campaign explores for ADAM
  (ADAM's own experiments — logistic regression / MLP on MNIST, small CNN on CIFAR-10 — are
  minutes-scale, keeping GPU$ modest).
- **Fixed budget:** identical `--max-gpu-usd` / `--max-gpu-hours` / `--max-llm-usd` and seed
  for both arms; only `OPENRESEARCH_SCHEDULER_AUTHORITATIVE` differs (OFF vs ON).
- **Scorecard:** best rubric score (primary), GPU$, wall-clock, and the ON-arm branch tree.
  Expectation: at equal budget, ON reaches a **higher best score** by reallocating GPU from
  frozen losers to promoted/revived winners.
- **Operator A/B gate (deferred to sign-off):** the ≥3-paired-A/B + grader-σ manifest
  (`asha_authority_gate.py`) — I set it up; the default-ON flip stays the operator's. The
  single ADAM paired A/B is the **headline**; the ≥3-paired gate is the follow-on.

**Exit:** committed scorecard comparing control vs treatment on ADAM.

## Cost & safety

- **Local-first:** B is validated entirely locally (no cloud). Only A's final check and C
  spend GPU$.
- **Every cloud spend is gated** behind a preflight cost estimate + operator checkpoint
  before provisioning; tight `--max-gpu-usd/-hours` caps; `--instance-termination-action=STOP`
  + `max_run_duration_s`; post-run `gcloud compute instances list` to catch stray VMs.
- **Cost visibility:** verify real spend via `tokens_total.json` + `gcloud`, never the
  ledger alone (ledger is blind to Foundry-routed LLM spend and idle GPU time).

## Test strategy

- Every new seam gets **hermetic** unit tests (fakes, no live cloud), following the
  `test_vm_compute_provider` golden-argv pattern and the existing scheduler unit tests.
- Flags default-OFF and **byte-identical when off** (guarded by the flag-registry tests).
- The GKE fail-closed guard gets an explicit test asserting Azure/AKS is unaffected.

## Open questions to settle in the implementation plan

1. Does the **unified/local** reproduction path already invoke the cell-matrix
   (`cell_matrix.py` / `gpu_cell_runner`) and emit `cell_manifest.json`, or must the
   campaign explicitly enable cells? (Determines how much of B's receipt emission is wiring
   vs. new.)
2. Exact **rung ladder** for ADAM (optimizer-step boundaries) — from the paper's training
   schedule; may warrant a light deep-research pass on ADAM's experimental protocol.
3. Branch-state reconciliation across the current branch (`integrate/degke-runpod-on-trunk`)
   vs. the in-flight cleanup on `remove-runpod-railway-cleanup` (prior spec §What-changes-a).

## Key references (file:line)

- Prior spec: `docs/superpowers/specs/2026-07-22-restore-gcp-vm-path-surgical-degke-design.md`
- VM path: `backend/services/runtime/vm_compute_provider.py`; `compute_provider.py:102-163`
- Scheduler: `backend/agents/rlm/{asha_scheduler,scheduler_runtime,scheduler_authority_controller,scheduler_evidence}.py`
- Local cell runner: `backend/agents/rlm/gpu_cell_runner.py`; `cell_matrix.py`
- Decide seam: `backend/agents/rlm/campaign_composition.py`
- Receipt schema: `backend/agents/rlm/scheduler_evidence.py:80-210`
- Operator gate: `backend/agents/rlm/asha_authority_gate.py`
- Diagnosis of recent failures: this session (cutout_val1 / resnetgcp — cells never complete + no resume)
