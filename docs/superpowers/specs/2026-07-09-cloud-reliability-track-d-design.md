# Cloud Reliability (Track D) — Durable Off-Laptop Orchestration + Promote the Built Reliability Layer to Default

- **Date:** 2026-07-09
- **Status:** Design (approved in brainstorming; Codex adversarial review pending)
- **Track:** D (cloud reliability). Runs in parallel with Track A (eval integrity,
  `2026-07-09-eval-integrity-track-a-design.md`).
- **Scope:** GCP-first. Azure parity and in-cloud CPU-triage are explicitly deferred (§10).

## 1. Problem

"Reproduce most papers unattended on GCP/Azure" is today a **reliability/plumbing** problem,
not a modeling one. Forensics over 11 real run dirs: **only 1 reproduced, and it was local
CPU.** Every GCP/GKE run failed or went partial, and the causes are dominated by non-modeling
failures:

1. **The driver isn't durable.** The orchestrator runs on the operator's laptop; ~half the
   GCP deaths are `SIGTERM`/host-suspend/orphan before finalize
   (`prj_adam_gcp_1/2` `demo_status` "run terminated by SIGTERM"; `prj_adam_gcp_3`/
   `prj_c912…` "orphaned_stale_run"). A run needing hours of GPU cannot survive a driver that
   dies with the laptop. This directly contradicts "unattended."
2. **The default remote path is broken for *uncellified* projects.** A **cellified**
   `sandbox=gcp` project already uses the working cell route (`primitives.py:7170`, default-ON).
   But a monolithic **commands-only (uncellified)** project falls back to `k8s_job_backend.exec`,
   which **never downloads the GCS-uploaded code into the pod** (`gke_cell_synth.py:5` docstring).
   The cell-synth path that would cellify it (so it uses the working route) is default-OFF —
   so the *uncellified* GCP experience is the broken one.
3. **~17 reliability/honesty guards ship default-OFF.** The correct GCP config is ~15 env
   vars exported by a bespoke `scripts/adam_gcp_e2e.sh`, never the autonomous/UI path.
4. **No mid-training checkpoint/resume by default.** A preempt/OOM/`SIGTERM` mid-cell restarts
   that cell from step 0 on scarce, contended A100s; resume also collides with leftover K8s
   Jobs (409 Conflict).
5. **The SDK aclose-stall abandons repair.** `implement_baseline` gives up after 120s of no
   activity and reports `ok` on half-written code (`prj_adam_gcp_1` worker report), so the GCP
   repair loop never converges the way the local run self-heals.

### 1.1 The load-bearing insight: nearly every fix is already built and default-OFF

The reliability layer is implemented and unit-tested; it is simply **not the default**, and no
durable host runs it. This track's work is **(a) one genuinely-new piece — durable off-laptop
orchestration — and (b) a disciplined default-ON promotion** of the built flags:

| Failure | Built fix (flag, all default-OFF) | Module |
|---|---|---|
| Broken default remote path | `OPENRESEARCH_GKE_SYNTH_CELL` | `gke_cell_synth.py` |
| Execute-mode framework repos | `OPENRESEARCH_EXECUTE_SYNTH`, `OPENRESEARCH_FRAMEWORK_IMAGES` | `execute_cell_synth.py`, `k8s_job_cell_runner.py` |
| No cell resume | `OPENRESEARCH_CELL_RESUME_AUTO` | `primitives.py` |
| SDK aclose-stall abandons repair | `OPENRESEARCH_IMPL_ABANDON_GUARD` | `primitives.py` |
| Guard false-blocks faithful code | `OPENRESEARCH_PREFLIGHT_UNION_SCOPE` | `pre_flight_validator.py` |
| Orphaned token-burning children on hard exit | `OPENRESEARCH_HARDEXIT_CLEANUP` | `process_cleanup.py` |
| All-models-errored fake-green | `OPENRESEARCH_PER_MODEL_STATUS_GATE` | `primitives.py` |
| Zero/constant fabricated metrics | `OPENRESEARCH_ZERO_METRICS_GUARD` | `zero_metrics_detection.py` |
| Train-reward-as-held-out | `OPENRESEARCH_EVAL_PROVENANCE_GUARD` | `eval_provenance.py` |
| Dead env scored as real 0 | `OPENRESEARCH_ENV_LIVENESS_GATE` | `env_liveness.py` |
| Flat/no-learning run scored as success | `OPENRESEARCH_NO_LEARNING_SIGNAL_GATE` | `no_learning_signal.py` |
| Credited result-leaf lacking on-disk evidence | `OPENRESEARCH_LEAF_EVIDENCE_GATE` (leaf gate, default-OFF — the *verdict-level* evidence gate in `report.py:1551` is already default-ON) | `evidence_gate.py` |

## 2. Goals / Non-goals

**Goals**
- G1. **Durable orchestration** — a run survives the operator's laptop dying/disconnecting.
- G2. **The default GCP path works out of the box** — cell-synth is the default; the broken
  monolithic exec is retired/guarded.
- G3. **A curated subset of the built guards is default-ON**, promoted through the repo's
  A/B + σ-gate discipline, so the autonomous path gets the honesty/reliability backstops.
- G4. **Checkpoint/resume by default** — a preempted cell resumes from its last checkpoint;
  resume never collides with a leftover Job.

**Non-goals**
- Azure as a live target (Bicep exists, has never run) — deferred (§10).
- In-cloud CPU-tier triage (deprioritized in brainstorming Q5) — CPU-class papers rely on
  `sandbox=local` routing; §9 R2 covers the residual risk.
- Any change to the verdict/eval (Track A owns that). Track D coordinates the guard flips with
  Track A's `VerdictAuthority`.

## 3. Design

### 3.1 Durable off-laptop orchestration (the one genuinely-new piece)

Today `cli.py` / `lifecycle_driver.run_lifecycle_primary` / `reproduction_campaign` run
**in-process on the launching host**. Make the driver survivable:

- **Primary: in-cluster controller.** For `sandbox=gcp`, the run driver executes as a small
  **controller Job/pod in the same GKE cluster** (not on the laptop). The launcher (CLI/UI)
  submits the controller Job, records its handle in `demo_status.json`, and may disconnect;
  the controller owns the reproduction lifecycle and writes run-state to the run bucket
  (GCS-backed `runs/<id>/`), so `--resume`/reconnect reads authoritative state. The controller
  requests GPU cells exactly as the in-process driver does today (`k8s_job_cell_runner`).
- **Fallback: durable local driver.** When no controller host is available, the local driver
  runs under `setsid` (not `nohup` — `nohup` dies on Claude-Code teardown, per memory), with
  `OPENRESEARCH_HARDEXIT_CLEANUP` on so a hard exit reaps children, and periodic run-state
  checkpoints so a relaunch resumes rather than restarts.
- **State authority.** Run-state (`rlm_state/`, `demo_status.json`, ledgers) is
  checkpoint-written and, for the controller path, mirrored to the run bucket so the driver's
  host is stateless and replaceable.
- **Single-writer lease — no split-brain (closes finding #8).** Exactly one driver owns a run at
  a time via a generation-numbered ownership lease in the run bucket
  (`runs/<id>/rlm_state/owner.lease`, acquired by compare-and-swap on the GCS object generation
  with a TTL/heartbeat). The controller acquires the lease at start; the laptop launcher holds it
  only until hand-off, then releases; `--resume`/reconnect may drive **only** if it can acquire
  the lease. A driver that finds a newer generation exits without writing. This removes the
  laptop+controller double-drive / duplicate-job / racing-final-report hazard.
- **Identity, creds, egress (explicit preconditions, not assumptions).** The controller runs
  under a Workload-Identity KSA with the same IAM the local orchestrator uses today (GCS RW + GKE
  job submit — `gke_job_backend.py:15`); LLM credentials are mounted from a secret, never baked
  into the image; the controller pod needs the same egress (arXiv / GitHub / model hubs) the
  laptop driver has. A missing precondition fails loud at controller-submit, not mid-run.

This is the only substantial new build in Track D; the rest is promotion.

### 3.2 Make the working path the default (G2)

- Route `sandbox=gcp` monolithic **commands-only (uncellified)** projects through cell-synth by
  **defaulting `OPENRESEARCH_GKE_SYNTH_CELL` ON** (cellified projects already use the working
  route), and default `OPENRESEARCH_EXECUTE_SYNTH` + `OPENRESEARCH_FRAMEWORK_IMAGES` ON for
  execute-mode/known-framework repos.
- The monolithic `k8s_job_backend.exec` staging path is **guarded**: if it would run for
  `gcp`/`gke` without staging code, it fails loud (`monolithic_exec_unstaged`) instead of
  silently producing a scratch pod. (The non-flag companion `_default_gpu_sku` fix and the
  `cd <host_path>` smoke-skip are already landed and stay.)

### 3.3 Curated guards → default-ON, behind the A/B + σ-gate (G3)

Classify the built guards by flip risk and promote in tiers:

- **Tier-0 (structural, safe to default-ON immediately — no run-outcome change):**
  `HARDEXIT_CLEANUP`, `IMPL_ABANDON_GUARD`, `GKE_SYNTH_CELL`, `CELL_RESUME_AUTO`. These change
  reliability, not which runs dispatch or what they score; they ship default-ON with a
  regression-test pass.
- **Audited-flip — `PREFLIGHT_UNION_SCOPE` (closes finding #7):** it *widens* the scanned file
  set, which is outcome-changing in BOTH directions — it fixes the narrow-scan false-block (real
  `from_pretrained` living in `train_cell.py`) but could *newly* block a faithful helper
  `model.py` that trips a hard violation (`primitives.py:6870`). So it is **not** free Tier-0: it
  flips ON only after the §4 bidirectional false-block audit shows it reduces net false-blocks
  without introducing new ones.
- **Tier-1 (score-changing honesty vetoes — need the ≥3 paired-A/B + σ-gate before default-ON):**
  `PER_MODEL_STATUS_GATE`, `ZERO_METRICS_GUARD`, `EVAL_PROVENANCE_GUARD`, `ENV_LIVENESS_GATE`,
  `NO_LEARNING_SIGNAL_GATE`, `LEAF_EVIDENCE_GATE` (the verdict-level evidence gate is already
  default-ON). Each is a fail-closed veto that can only turn a fabricated/dead result
  *repairable* — but because it changes the shipped result it must clear the A/B gate. Track D
  runs the A/B (§5) and flips per-guard as each clears.
- **Coordination with Track A:** the Tier-1 vetoes feed `VerdictAuthority` as `inconclusive`
  inputs (dead env / no-learning) or `contradicted`-eligibility inputs. The flip order is:
  Track A's `VerdictAuthority` lands (behind its flag) → Tier-0 default-ON → Tier-1 A/B → flip.

### 3.4 Checkpoint / pure-resume (G4)

- Default `OPENRESEARCH_CELL_RESUME_AUTO` ON: `run_experiment` stabilizes `run_id` to
  `ctx.project_id` and auto-arms `OPENRESEARCH_RESUME_CELLS`, so a 1/N-cell failure resumes
  only the failed cell.
- **Pure-resume fix:** on resume, a leftover K8s Job name is reconciled (adopt-if-matching /
  delete-and-recreate) instead of 409-colliding; `--resume` on the controller path re-attaches
  to the controller/cells rather than re-running the pipeline.
- The synthesized/execute cells checkpoint periodically (step-level) so a preempt loses ≤ the
  checkpoint interval, not the whole cell.
- **Deterministic adoption (closes finding #9).** Job names are derived deterministically from
  `(run_id, cell_id, attempt)` instead of the current fresh UUID (`k8s_job_backend.py:120`), and
  a write-ahead **submit-intent** row is persisted to the run bucket *before* the K8s submit
  (mirroring the campaign ledger's write-ahead intent). So a controller that crashes after submit
  but before persisting can, on restart, find the intent + the deterministically-named Job and
  **adopt** it (in-memory active-job tracking at `k8s_job_backend.py:800` is no longer the only
  source of truth) rather than launching a duplicate.

## 4. The default-flip discipline (the crux)

Flipping guards ON is the highest-leverage, lowest-effort win — but a wrong flip (e.g. a
guard that false-blocks faithful code) wastes GPU and mislabels a good run. Rules:

- **Every flip is a default change, not a code change** — off-state stays byte-identical; a
  flip is a one-line default plus the A/B evidence.
- **Tier-0 vs audited-flip vs Tier-1** (§3.3) gates *which* flip needs what: Tier-0
  structural/reliability flips ship with a regression pass; `PREFLIGHT_UNION_SCOPE` needs the
  bidirectional false-block audit below; Tier-1 score-changing vetoes need ≥3 paired SDAR A/B
  runs + the grader-σ gate (`data/grader_calibration.json`, σ≤0.02 already met at samples=1).
- **Bidirectional false-block audit for `PREFLIGHT_UNION_SCOPE`** — it is an *audited-flip*, not
  Tier-0: widening the scan is outcome-changing both ways. The audit must prove BOTH (a) a
  faithful multi-file impl (SDAR/UCPO shape) that the narrow scan false-blocked now passes, AND
  (b) no faithful helper-file-heavy impl is *newly* blocked — while a true surrogate (no
  `from_pretrained` anywhere) still blocks. It flips ON only if net false-blocks drop with no new ones.
- **Loud, never silent** — a promoted guard that fires emits its existing `run_warning`; a
  flip that regresses the A/B is reverted, logged, not forced.

## 5. Testing & acceptance

- **Durability drill (headline):** launch a GCP run, kill the launcher host mid-training,
  reconnect → the run continues to finalize (controller path) or resumes from checkpoint
  (fallback path), producing a real metric. Today this run dies; success = it finishes.
- **Default-path test:** a plain `sandbox=gcp` monolithic project stages code into the pod and
  runs (no `monolithic_exec_unstaged`) without any hand-set env vars.
- **Resume test:** a 1/N-cell OOM re-runs only the failed cell; a leftover Job name does not
  409.
- **Guard A/B:** `scripts/ab_compare.py` paired runs (guard-on vs guard-off) per Tier-1 guard,
  σ-gated, before its default flip; `OPENRESEARCH_REQUIRE_STAMPED_AB` enforced.
- **False-block regression:** faithful SDAR/UCPO impl passes `PREFLIGHT_UNION_SCOPE`; true
  surrogate blocks.
- **Off-state byte-identical:** every promoted flag, when explicitly set `0`, reproduces prior
  behavior.

## 6. Rollout sequence

1. Land durable orchestration (§3.1) behind `OPENRESEARCH_DURABLE_CONTROLLER` (default-OFF),
   validate the durability drill, then default-ON for `sandbox=gcp`.
2. Default-ON the Tier-0 flags (§3.3) with the regression pass; run the `PREFLIGHT_UNION_SCOPE`
   bidirectional false-block audit (§4) and flip it only if it passes.
3. Default-ON cell-synth path (§3.2) + the pure-resume fix (§3.4).
4. Run the Tier-1 guard A/B (§5) and flip each guard as it clears the σ-gate.

## 7. Interfaces / touch points

- New: a controller-Job submitter + reconnect/resume reader (wraps the existing
  `k8s_job_cell_runner` submit path; run-state to the run bucket).
- Changed defaults only (no signature changes) for the built flags in §3.3/§3.4.
- Guarded `k8s_job_backend.exec` for `gcp`/`gke` unstaged (§3.2).

## 8. Risks & mitigations

- **R1 — a promoted honesty guard false-blocks a faithful run.** Mitigation: Tier-0/Tier-1
  split; the A/B + σ-gate; the `PREFLIGHT_UNION_SCOPE` false-block regression; every guard is
  a *repairable* veto (the repair loop re-drives), not a hard fail.
- **R2 — a CPU-class paper is sent to `sandbox=gcp`** (CPU-triage deferred) and pays the GPU
  failure surface. Mitigation: document the `sandbox=local` routing rule; surface a
  `cpu_class_on_gpu` advisory warning when a paper with no GPU signal runs on `gcp` (cheap,
  non-blocking; full CPU-triage is a later spec).
- **R3 — the controller path adds infra surface** (a controller Job, bucket state).
  Mitigation: the fallback durable-local driver keeps a no-cluster path working; controller
  state is just the existing `runs/<id>/` mirrored to GCS.
- **R4 — cost visibility stays blind on the GPU path** (node-hours absent from the ledger).
  Mitigation: out of scope for the reliability flips, flagged as a known gap; the durability
  work makes node lifetime observable (controller owns node request/teardown), a prerequisite
  for a later cost-accounting spec.

## 9. Out of scope (own specs / later)

- Azure live target (bootstrap the AKS cluster + GPU quota + first real run).
- In-cloud CPU-tier triage / cheap-fail-fast feasibility before GPU lease.
- GPU node-hour cost accounting into the ledger.
