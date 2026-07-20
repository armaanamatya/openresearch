# Scheduler authority and cloud preflight — 2026-07-19

## Status

No chargeable reproduction was launched. `OPENRESEARCH_SCHEDULER_TREE=1` remains
the only runnable scheduler mode. `OPENRESEARCH_SCHEDULER_AUTHORITATIVE` is an
explicit fail-closed audit seam: the current campaign has no provenance-validated
defining metric at a paper-pinned optimizer step, no checkpoint lineage, and no
branch queue/frozen pool to carry out an ASHA decision. The paired A/B gate
therefore rejects every `applied:true` action and cannot approve a default flip.

## Read-only preflight observations

On 2026-07-19 the local gcloud active identity was
`thisisaayushbaniya@gmail.com`. A read-only
`gcloud container clusters list --project deepinvent-ext-ut` returned HTTP 403;
`kubectl config current-context` reported no current context; and
`GOOGLE_APPLICATION_CREDENTIALS` was unset. Do not use this identity to create a
cluster or launch a billed job.

The CLI accepts `auto`, `local`, `docker`, `runpod`, `azure`, `gcp`, and `gke` for
`--sandbox`; it does not have an `aws` GPU backend. AWS GPU execution would need a
new EKS/Batch/EC2 backend and is not a valid value for the current command.

## Exact unblocks before any real run

1. Authenticate gcloud as an identity with `deepinvent-ext-ut` GKE and billing
   permissions, then retrieve a kubectl context for the recreated `deepinv-gke`
   cluster. Recreate it only through the reviewed six-gate procedure in
   `docs/runbooks/2026-07-17-deepinv-gke-l4-validation-handoff.md`; use an L4 pool
   if A100 stock is unavailable.
2. Configure the verified GKE cell-matrix route (`cells.json`, `train_cell.py`,
   `GKE_SYNTH_CELL`) and resolve the documented job-name 409 retry collision before
   launch.
3. Add the prerequisite authority runtime: frozen paper-step ladder,
   provenance-validated metric/checkpoint receipts, durable branch queue/frozen
   pool, and factual branch-lineage event emitter. Until then, do not call an arm
   "authoritative" and do not set `applied:true`.
4. For a shadow run, pass explicit LLM/GPU dollar and GPU-hour caps. Verify spend
   from provider/token records and `kubectl get nodes,pods`; neither
   `cost_ledger.jsonl` nor `demo_status.json` includes Foundry spend or idle GPU
   node time.

## Monitoring contract after those unblocks

Record the project ID and run `asha_shadow_report.py` on every ~40-minute tick;
also inspect `demo_status.json`, `experiment_runs.jsonl`, and live nodes/pods.
Append verified cost and terminal evidence to this dated runbook and the scheduler
memory. A 40-minute scheduler/wakeup facility was not available in this session,
so no fictitious recurring monitor was registered.
