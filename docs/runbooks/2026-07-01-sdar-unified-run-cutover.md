# SDAR unified-run cutover — operator runbook (Phase 1f)

> **Doc status:** Operator runbook · 2026-07-01 · the cutover procedure for the
> paper-agnostic multi-cloud reproduction stack (Phases 1a–1f). Spec:
> `docs/history/specs/2026-07-01-paper-agnostic-multicloud-reproduction-and-self-improvement-design.md`.

This is the **strangler-fig cutover** from the bespoke SDAR-on-GCP bash to the
unified `ReproductionRun` controller. Phases 1a–1e are **code-complete and merged**
(all flag-gated default-OFF ⇒ the live path is byte-identical today). Phase 1f — the
actual A/B validation, the default-flip, and thinning the bash — is **operator work
that spends real GPU money**, so it lives here as a procedure, not as an automated
step.

## What's code-complete (merged, default-OFF)

| Layer | What | Where | Flag |
|---|---|---|---|
| 1a | `EnvironmentAdapter` seam + 3 SDAR adapters + `AssetCache` | `backend/services/runtime/env_adapters/`, `asset_cache.py` | — (refactor; `env_cache` facade byte-identical) |
| 1b | `RunBudget.max_gpu_hours`, `RunPlan.required_assets`, `FeasibilityTriage`, `estimate_scope_cost` | `run_plan.py`, `feasibility_triage.py`, `budget.py` | unwired |
| 1c | `ComputeProvider` + `CloudProfile` + `VmComputeProvider(gcp)` + `ClusterComputeProvider` + `ReproductionRun` | `compute_provider.py`, `cloud_profile.py`, `vm_compute_provider.py`, `cluster_compute_provider.py`, `reproduction_run.py` | `OPENRESEARCH_UNIFIED_RUN` (entrypoint), `OPENRESEARCH_CLOUD_FAILOVER` (failover) |
| 1d | `CredentialBroker`, generic `AssetResolver`, `cpu_warm_disk_then_gpu_attach` tiering | `credential_broker.py`, `asset_resolver.py`, `vm_compute_provider.py` | unwired |
| 1e | `FailureAttribution`, `ExperienceMemory` (global-infra store), `held_out_gate` | `failure_attribution.py`, `experience_memory.py`, `held_out_gate.py` | `OPENRESEARCH_EXPERIENCE_MEMORY` |
| 1f | `build_reproduction_run` composition root | `unified_run.py` | `OPENRESEARCH_UNIFIED_RUN` |

Every unit ships hermetic tests; the composition root's FINALIZE test proves the
whole stack runs end-to-end against a `FakeComputeProvider` with **zero cloud spend**.

## The flags (all default-OFF ⇒ unset = today's behavior)

- `OPENRESEARCH_UNIFIED_RUN=1` — opt into the `ReproductionRun` controller path (via
  `unified_run.build_reproduction_run`). Unset ⇒ today's bash / `run.py` path, unchanged.
- `OPENRESEARCH_CLOUD_FAILOVER=gcp,azure` — on a cloud run, try GCP then fall back to
  AKS on a `backend_unavailable` `SandboxRuntimeError`. Empty/unset ⇒ single-cloud, byte-identical.
- `OPENRESEARCH_EXPERIENCE_MEMORY=1` — advisory cross-run memory hints. Unset ⇒ no hints.
- (Phase-1b/1d knobs are consumed by the controller, not the live bash; they are inert until `OPENRESEARCH_UNIFIED_RUN` is on.)

## The cutover procedure (the ≥3-paired-run discipline this repo mandates)

Do NOT flip `OPENRESEARCH_UNIFIED_RUN` to a default without completing this.

1. **Characterize (done):** the 45 `env_cache` tests + the provisioning characterization
   suite pin the old behavior; the adapter refactor kept them green unchanged.
2. **Golden-command parity (done, no GPU):** `VmComputeProvider`'s golden-command tests
   (`tests/services/runtime/test_vm_compute_provider.py`) assert the emitted `gcloud`
   argv matches the bash's effective lifecycle. Re-run before any live A/B:
   `.venv/bin/python -m pytest tests/services/runtime/test_vm_compute_provider.py -q`.
3. **Operator paired A/B (spends GPU — YOUR step):** run SDAR (1-model, smallest-two
   scope) BOTH ways on the SAME GCP config, ≥3 paired runs:
   - **control:** the current bash (`scripts/sdar_gcp_optimal_run.sh`), as today.
   - **candidate:** the controller — `OPENRESEARCH_UNIFIED_RUN=1` driving
     `build_reproduction_run(paper_id="2605.15155", scope=<smallest-two>, budget=<RunBudget>, cloud="gcp", ...)`
     then `.run()`.
   Compare `final_report.json` (score, verdict, cost, wall-clock) with
   `scripts/ab_compare.py` (the existing paired-report tool). Parity = the controller
   produces an equivalent report at equivalent cost, and its emitted gcloud lifecycle
   matches step 2.
4. **Flip default → thin the bash:** only after ≥3 paired runs show parity, make
   `OPENRESEARCH_UNIFIED_RUN` the default for SDAR and reduce
   `scripts/sdar_gcp_*.sh` to a thin wrapper (or retire it). Keep the bash as the
   fallback until the flip holds across a few real runs.

## GCP→Azure failover (operator note)

Provision-time failover is live behind `OPENRESEARCH_CLOUD_FAILOVER`. Preconditions to
actually exercise it: both a reachable GKE cluster (`ensure_gcp_available`) and a
reachable AKS cluster (`ensure_azure_available`) — provisioned via `infra/gcp/` +
`infra/azure/bicep/`. With neither cloud available the selector raises
`backend_unavailable` after trying both (honest, not a hang). Mid-run failover (GCP
dies mid-training) is out of scope — this is provision-time only.

## Honest boundaries (what remains beyond Phase 1f)

- **Live-GPU validation is un-run:** the golden-command + FINALIZE tests are hermetic;
  no controller-driven live GCP `final_report.json` exists yet. Step 3 above produces
  the first one.
- **`ClusterComputeProvider` full run integration** (streaming, real cell dispatch through
  the provider) stays on the existing Job path; the provider ships the shape + failover.
- **CredentialBroker wiring into `VmComputeProvider.stage`** — `stage` already redacts
  by-construction (ships a scratch-file path, never a raw `.env`); folding the broker in
  is a follow-up consolidation, not a gap.
- **The `cpu_warm_disk_then_gpu_attach` two-VM handoff** is argv-parity-tested but its
  live GCP validation (+ making it the on-demand-A100 default over `stage_on_gpu`) is
  operator work.
- **North-star (Phase 2+):** the in-cluster control-plane service, validated Azure VM,
  and the staged harness self-edit tier remain future work per the spec.
