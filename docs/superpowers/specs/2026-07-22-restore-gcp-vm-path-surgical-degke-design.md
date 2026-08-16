<!-- doc-meta: status=current; last-verified=2026-07-22 -->
# Restore GCP-without-GKE (single-VM) as the GCP path — design

**Date:** 2026-07-22 · **Author:** operator + Claude · **Scope decision:** surgical (approved)

## Problem

GPU reproduction on GCP has been broken since the stricter split-identity **GKE**
backend became the default (landed on `main` 2026-07-17, author `sww35`/`lolout1`).
Pods fail before running because two IAM grants the GKE design needs were never applied
live and the operator account (`aayush@deepinvent.ai`) lacks `setIamPolicy` to add them:

1. `roles/artifactregistry.reader` on repo `reprolab` for node SA `deepinv-gke-node@…`
   → `ErrImagePull` (pod never starts).
2. `roles/iam.workloadIdentityUser` on `deepinv-workload@…` for
   `serviceAccount:deepinvent-ext-ut.svc.id.goog[reprolab/reprolab-sa]`
   → `getAccessToken` 403 (pod can't reach GCS).

The project author's guidance (2026-07-21): *"all I did was enable gke as default — just
switch it back … you can use GCP without gke."* That non-GKE path is real and already in
the codebase; this design restores it as the GCP execution path. The `kubectl`-transport
"local-transport" branch is explicitly **rejected** as a workaround and is not part of this.

## Decision

**Surgical.** GCP reproduction runs on the single-VM path (`VmComputeProvider`); the
GKE/K8s code stays in the tree but parked. **Azure/AKS is untouched** — it shares the same
K8s runner (`k8s_job_backend.py`, `k8s_job_cell_runner.py`), so a full K8s removal would
break Azure, ~40 tests, and ~100 config keys. Not doing that.

## Architecture — the GCP-without-GKE execution model

`backend/services/runtime/vm_compute_provider.py` (`VmComputeProvider`, ~lines 174–779)
implements the `ComputeProvider` lifecycle (`compute_provider.py:102–163`):
`preflight → provision_cpu → stage → acquire_gpu → launch → watch → collect →
release_gpu → teardown`. Concretely it shells `gcloud compute instances create`, `scp`
(stage/collect), `ssh` (launch + status poll), and `instances stop/delete`.

- One GCE **GPU VM** is provisioned; code is `scp`'d in; the reproduction runs **inside the
  VM as `--sandbox local`**; artifacts are `scp`'d back; the VM is torn down.
- **No Kubernetes, no Artifact Registry pull, no Workload Identity, no GCS** → structurally
  sidesteps **both** blocked IAM grants. Operator GCE permissions verified present
  (`compute.instances.create/delete/setMetadata/get/setTags`, `compute.disks.create`).
- Driven by the campaign **unified** driver: `attempt_driver.py::UnifiedRunDriver.launch()`
  → `unified_run.build_reproduction_run(cloud="gcp", …)` →
  `VmComputeProvider(CloudProfile(cloud="gcp", vm=VmSpec(...)))` →
  `ReproductionRun.run()` state machine.
- **Distinct from `--sandbox gcp`**, which always routes to `GkeJobBackend`
  (`primitives.py:3117–3122`). The VM path is reached via the campaign, not the sandbox
  flag.

## What changes (surgical)

**(a) Finalize the un-GKE cleanup already in the working tree.** The current tree
(branch `remove-runpod-railway-cleanup`) already deletes RunPod/Brev backends and flips the
default `gcp`→`local` at all three entry points (`agents/execution.py:59`,
`config.py:259`, `services/events/live_runs.py:180`) plus a configurable
`autonomous_sandbox` (`config.py:274`). Implementation step 1 is to **review this diff with
the operator and commit it cleanly** — this *is* "revert enable-gke-as-default." (Note:
there is no clean 2-commit `git revert` because `main` was rebased past the original flip;
this finalizes the in-flight revert instead.)

**(b) Fail-loud GKE entry (recommended, small).** `reproduce --sandbox gcp`/`gke` should
error with a clear message — "GKE is parked; use `campaign --campaign-driver unified` for
the GCP VM path" — so nobody silently re-hits the broken GKE default. Prevents the exact
recurrence that caused this incident. Gated so it does not affect Azure.

**(c) No functional change** to `VmComputeProvider` (already 40+ tests), Azure/AKS, or the
shared K8s runner. GKE-specific code (`gke_job_backend.py`, `gke_cell_synth.py`,
`docker/gke-*`, `infra/gcp/`) stays present but parked.

## Non-goals / explicitly NOT changing

- Not deleting the shared K8s runner or Azure/AKS path.
- Not deleting GKE code or `infra/gcp` Terraform (parked, may be revived if IAM is fixed).
- Not using the `OPENRESEARCH_GKE_LOCAL_TRANSPORT` kubectl workaround (rejected).
- Not asking the admin for IAM grants (not required for the VM path).

## Testing

Live end-to-end on a **small, cheap** paper with tight money caps, provisioning a **fresh
VM under the operator's own SSH key** (via `OPENRESEARCH_GCP_*`) rather than depending on
the default `sdar-a100-od` / `abheekp` instance:

```
python -m backend.cli campaign <small_paper> --campaign-driver unified \
  --sandbox local --billing-sandbox gcp \
  --max-llm-usd <X> --max-gpu-usd <Y> --max-gpu-hours <Z>
```

Pass = the VM provisions, stages, trains, collects, tears down, and a real
`final_report.json` + artifacts return locally. Watch: `gcloud compute instances list`
shows the VM up then gone (no stray billing). Hermetic side: existing
`tests/rlm/test_vm_compute_provider.py` (+ the fail-loud test from (b)) stay green.

**Prerequisites to settle in the plan:** the target VM/zone/machine-type + SSH key
(`OPENRESEARCH_GCP_PROJECT/INSTANCE/ZONE/SSH_USER/REMOTE_DIR`); a small test paper; GPU
capacity in the chosen zone (L4/A100 stockout is zonal — pick a zone with capacity).

## Risks

- **Stray VM billing** if teardown fails → the path caps `--instance-termination-action=STOP`
  + `max_run_duration_s`; plan adds a post-run `instances list` check.
- **Zonal GPU stockout** → pick/verify a zone at launch; VmComputeProvider detects capacity
  signatures.
- **Uncommitted-tree risk** → step 1 reviews the ~60-file cleanup diff before committing;
  do not blind-commit.

## Key references (file:line)

- VM path: `backend/services/runtime/vm_compute_provider.py`, `compute_provider.py:102–163`,
  `cloud_profile.py`.
- Driver/selection: `backend/agents/rlm/attempt_driver.py:394–482`, `unified_run.py:64–113`,
  `reproduction_run.py`.
- GKE routing (parked): `backend/agents/rlm/primitives.py:3117–3122`.
- Default entry points: `backend/agents/execution.py:59`, `backend/config.py:259,274`,
  `backend/services/events/live_runs.py:180`.
- IAM diagnosis: `.claude/worktrees/gke-local-transport/docs/runbooks/2026-07-20-gke-local-transport-handoff.md`
  (§2/§3), memory `gcp-gke-setup`.
