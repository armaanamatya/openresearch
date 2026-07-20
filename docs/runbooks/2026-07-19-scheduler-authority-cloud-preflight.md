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

### Authorized GCP recheck — 2026-07-19

The local machine retains a second authenticated account,
`aayush@deepinvent.ai`.  Read-only calls made with that account, without
changing the global gcloud configuration, establish that
`deepinvent-ext-ut` is active and billed, both historical artifact buckets
exist, the pinned `gke-cell-base:v1` image exists in Artifact Registry, and
the regional quotas are currently unused (8 L4, 8 A100, and 4 A100-80 GPUs).
`gcloud container clusters list` remains empty: `deepinv-gke` has not been
recreated.

This recheck also found the gate that must be repaired before cluster creation:
`deepinv-workload@deepinvent-ext-ut.iam.gserviceaccount.com` has
`roles/storage.objectAdmin` on `deepinvent-ext-ut-reprolab-artifacts`, but its
service-account IAM policy has no Kubernetes Workload Identity member.  The
available account is a project Editor and can use the project, but direct
permission checks show it lacks both `iam.serviceAccounts.setIamPolicy` and
`resourcemanager.projects.setIamPolicy`.  It therefore cannot restore the
required binding or grant the custom node service account its node role.

No cluster was created from this account: creating one before those bindings
exist would knowingly fail gate 6 (provenance artifact upload) and leave a
billed, nonfunctional control plane.  A project IAM administrator must first
apply the three bindings in the next section; then this account can continue
with the scale-to-zero L4 recreation and the six-gate validation.

On 2026-07-19 the local gcloud active identity was
`thisisaayushbaniya@gmail.com`. A read-only
`gcloud container clusters list --project deepinvent-ext-ut` returned HTTP 403;
`kubectl config current-context` reported no current context; and
`GOOGLE_APPLICATION_CREDENTIALS` and `ANTHROPIC_API_KEY` were unset. Do not use
this identity to create a cluster or launch a billed job; a no-credit Anthropic
key would not become an OAuth fallback. The documented `deepinv-gke` cluster was
deleted on 2026-07-17. The current local `ensure_gcp_available()` gate stops
even earlier, at missing `OPENRESEARCH_GCP_GCS_BUCKET`, so it made no cloud call.

### Live CPU-only Workload Identity probe — 2026-07-19

At the operator's explicit request to "run and try", a bounded diagnostic was
run against the real `deepinvent-ext-ut` project with the already authenticated
`aayush@deepinvent.ai` account. This was **not** a reproduction or scheduler
campaign: it used no LLM, no GPU node pool, no paper, no artifact write, and no
static credentials. Its only cloud action was a CPU GKE control-plane/identity
probe, followed by teardown.

1. Created temporary zonal Standard cluster
   `deepinv-gke-smoke-20260719` in `us-central1-a`, with a one-node
   `e2-standard-2` CPU pool, IP aliasing, no master-authorized networks, and
   workload pool `deepinvent-ext-ut.svc.id.goog`.
2. The cluster creation completed and the CPU node became `Ready`. The initial
   default node had metadata concealment, so a first probe could not access the
   GKE metadata server. This is a node-pool configuration fact, not an IAM
   success. The default pool was replaced by a CPU-only `wi-cpu` pool explicitly
   configured with `--workload-metadata=GKE_METADATA` before the decisive probe.
3. Created only `reprolab/reprolab-sa`, annotated with
   `iam.gke.io/gcp-service-account:
   deepinv-workload@deepinvent-ext-ut.iam.gserviceaccount.com`. A bounded Job
   used the public `google/cloud-sdk:slim` image and made no GCS write.
4. The corrected-node direct metadata probe returned HTTP 200 for the intended
   GSA email, proving KSA-to-GSA selection reached the expected identity. Its
   token request returned HTTP 403 with the exact live error:

   ```text
   Unable to generate access token; IAM returned 403 Forbidden:
   Permission 'iam.serviceAccounts.getAccessToken' denied.
   This error could be caused by a missing IAM policy binding on the target
   IAM service account.
   ```

This confirms the prior read-only finding with a real pod: the missing
`roles/iam.workloadIdentityUser` binding blocks the artifact path before any
GCS operation, exactly as the evidence gate requires. It is not a Claude-vs-
Codex issue, a GPU-capacity issue, or a scheduler defect. Do not substitute node
credentials, static service-account keys, or unauthenticated artifact storage.
Cluster teardown was submitted immediately after the probe. Delete operation
`operation-1784522366167-2a78d7e2-c3a1-4946-80c2-dc35004e84e9` reached `DONE`;
a subsequent explicit project listing showed no clusters and no
`gke-deepinv-gke-smoke*` Compute Engine nodes. The diagnostic is closed and
left no running GKE or node resource.

AWS is not launch-ready: the AWS CLI is absent, `AWS_PROFILE` and
`AWS_WEB_IDENTITY_TOKEN_FILE` are unset. `boto3` was installed into the local
repository venv for the hermetic controller path, and the bounded
`aws-preflight` now stops locally at the first real configuration gate:
missing `OPENRESEARCH_AWS_S3_BUCKET`. It makes no AWS API call in that state.
Do not set static credentials in a run spec or worker pod; the EKS cell contract
requires IRSA. The controller dependency is declared in
`backend/requirements.txt` / `pyproject.toml`.

## Exact unblocks before any real run

1. A project IAM administrator must restore the Workload Identity and custom
   node-service-account bindings (no static key fallback):

   ```bash
   gcloud iam service-accounts add-iam-policy-binding \
     deepinv-workload@deepinvent-ext-ut.iam.gserviceaccount.com \
     --project deepinvent-ext-ut \
     --member='serviceAccount:deepinvent-ext-ut.svc.id.goog[reprolab/reprolab-sa]' \
     --role=roles/iam.workloadIdentityUser

   gcloud projects add-iam-policy-binding deepinvent-ext-ut \
     --member='serviceAccount:deepinv-gke-node@deepinvent-ext-ut.iam.gserviceaccount.com' \
     --role=roles/container.defaultNodeServiceAccount

   gcloud projects add-iam-policy-binding deepinvent-ext-ut \
     --member='serviceAccount:deepinv-gke-node@deepinvent-ext-ut.iam.gserviceaccount.com' \
     --role=roles/artifactregistry.reader
   ```

   Preserve `deepinv-workload`'s existing bucket-scoped
   `roles/storage.objectAdmin`; it is the GSA for the `reprolab/reprolab-sa`
   Kubernetes ServiceAccount.  Verify the first binding before creating a
   cluster.
2. Set the intended `OPENRESEARCH_GCP_PROJECT`, `OPENRESEARCH_GCP_GCS_BUCKET`,
   pinned `OPENRESEARCH_GCP_BASE_IMAGE`, namespace, and ServiceAccount; then
   authenticate gcloud as an identity with `deepinvent-ext-ut` GKE and billing
   permissions and retrieve a kubectl context for the recreated `deepinv-gke`
   cluster. Recreate it only through the six-gate procedure in
   `docs/runbooks/2026-07-17-deepinv-gke-l4-validation-handoff.md`; use an L4
   pool if A100 stock is unavailable.
3. Configure the verified GKE cell-matrix route (`cells.json`, `train_cell.py`,
   `OPENRESEARCH_GKE_SYNTH_CELL`). The generic Job 409 collision is now
   fail-closed: full run/cell/config identity must match before active/success
   adoption; terminal conflicts never issue an unreserved retry. Ensure cluster
   RBAC grants Job create/patch rights only to the controller ServiceAccount.
4. For AWS, provision and review an EKS namespace with a least-privilege IRSA
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
5. Implement the prerequisite authority runtime: frozen paper-step ladder,
   provenance-validated metric/checkpoint receipts, durable branch queue and
   frozen pool, grade-free decision-evidence writer, and factual controller
   branch-lineage emitter. Until then, do not call an arm authoritative and do
   not set `applied:true`.
6. For every shadow run, pass explicit LLM/GPU dollar and GPU-hour caps. Verify
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
