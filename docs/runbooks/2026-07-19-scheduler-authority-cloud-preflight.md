# Scheduler authority and cloud preflight — 2026-07-19

## Status

No chargeable reproduction has been launched in this workstream.
`OPENRESEARCH_SCHEDULER_TREE=1` remains the only runnable scheduler mode.
`OPENRESEARCH_SCHEDULER_AUTHORITATIVE` is still an explicit fail-closed audit
seam: it preserves the cohort decision and records `applied:false` because the
campaign runtime has not yet produced frozen optimizer-step ladders, verified
metric/checkpoint receipts, a durable queue/frozen pool, or controller-emitted
branch transitions.

The offline paired-A/B gate is now stricter than the current producer. A future
applied action must bind a verified receipt, a closed-world grade-free ASHA
decision artifact that the gate recomputes, and a matching controller-owned
`branch-tree:<campaign>` EventStore event. It also rejects reuse of a receipt
across the entire manifest. This is evidence for an operator review only; it
cannot flip a default and it is not permission to claim an authority-on run.

The CLI now has an inert `--sandbox aws` EKS+S3 cell-matrix foundation. It is
not yet an available AWS environment: no EKS cluster, namespace, IRSA role,
S3 bucket, ECR digest, credentials, or configured cost metadata was discovered
or created in this workstream. Generic EKS sandbox/exec is intentionally
disabled; AWS jobs can use only the S3/IRSA cell-matrix route.

## Read-only preflight observations

On 2026-07-19 the local gcloud active identity was
`thisisaayushbaniya@gmail.com`. A read-only
`gcloud container clusters list --project deepinvent-ext-ut` returned HTTP 403;
`kubectl config current-context` reported no current context; and
`GOOGLE_APPLICATION_CREDENTIALS` and `ANTHROPIC_API_KEY` were unset. Do not use
this identity to create a cluster or launch a billed job; a no-credit Anthropic
key would not become an OAuth fallback. The documented `deepinv-gke` cluster was
deleted on 2026-07-17.

AWS is not launch-ready: the AWS CLI is absent, `AWS_PROFILE` and
`AWS_WEB_IDENTITY_TOKEN_FILE` are unset, and the repository `.venv` currently
lacks `boto3`. The bounded `aws-preflight` therefore stops locally before any
AWS API call with the typed missing-`boto3` error. Do not set static credentials
in a run spec or worker pod; the EKS cell contract requires IRSA. The controller
dependency is declared in `backend/requirements.txt` / `pyproject.toml`.

## Exact unblocks before any real run

1. Authenticate gcloud as an identity with `deepinvent-ext-ut` GKE and billing
   permissions, then retrieve a kubectl context for the recreated `deepinv-gke`
   cluster. Recreate it only through the six-gate procedure in
   `docs/runbooks/2026-07-17-deepinv-gke-l4-validation-handoff.md`; use an L4
   pool if A100 stock is unavailable.
2. Configure the verified GKE cell-matrix route (`cells.json`, `train_cell.py`,
   `OPENRESEARCH_GKE_SYNTH_CELL`) and resolve the documented job-name 409 retry
   collision before launch.
3. For AWS, provision and review an EKS namespace with a least-privilege IRSA
   ServiceAccount, NVIDIA GPU device plugin, a one-GPU-per-node labeled pool,
   S3 project/run-prefix policy, and a pinned ECR cell image. Set only verified
   `OPENRESEARCH_AWS_*` settings, including a whole-node `AWS_GPU_USD_PER_HOUR`.
   Install the declared Python dependency into the launch environment, configure
   an operator AWS identity/kube context, then run the explicit bounded preflight:

   ```bash
   python -m backend.cli aws-preflight --project-id <project> --run-id <probe-run>
   ```

   It checks controller identity/context and creates one no-GPU Job that proves
   the **pod** IRSA STS plus scoped S3 Put/Get/List/Delete, then foreground-
   deletes the Job. A controller STS check alone is not proof of pod access.
4. Implement the prerequisite authority runtime: frozen paper-step ladder,
   provenance-validated metric/checkpoint receipts, durable branch queue and
   frozen pool, grade-free decision-evidence writer, and factual controller
   branch-lineage emitter. Until then, do not call an arm authoritative and do
   not set `applied:true`.
5. For every shadow run, pass explicit LLM/GPU dollar and GPU-hour caps. Verify
   spend from provider/token records and `kubectl get nodes,pods`; neither
   `cost_ledger.jsonl` nor `demo_status.json` includes Foundry spend or idle GPU
   node time.

## Monitoring contract after those unblocks

Record each project ID, then on every ~40-minute tick run
`asha_shadow_report.py` for the arm and inspect `demo_status.json`,
`experiment_runs.jsonl`, and live nodes/pods. Append verified cost, node idle
time, terminal evidence, and A/B evidence to this dated runbook and the
scheduler memory. This tool environment has no recurring wakeup capability, so
no fictitious monitor has been registered; an operator or supported scheduler
must trigger the next tick.
