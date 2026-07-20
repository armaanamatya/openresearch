# GKE identity rebuild investigation — handoff (2026-07-19)

## Executive status

The scheduler code and hermetic tests can continue. A real GCP GPU campaign is not safe to launch: a pod in the exact historical deepinv-gke identity topology cannot mint its short-lived GCP access token, so it cannot return provenance-valid artifacts to GCS.

This is a current authorization condition, not a stale local login, an agent choice, GPU capacity, or a scheduler defect. The available evidence does not establish that the operator changed GCP and does not identify an actor or time for an IAM change.

No GCP project, service-account, bucket, or IAM policy was changed in this workstream. Two temporary CPU-only GKE clusters were created for identity diagnostics and then deleted. Final project checks showed no clusters or probe nodes. Neither probe used a GPU, LLM, paper campaign, GCS artifact write, or static credential.

See the companion cloud preflight runbook for the current operator checklist.

## What previously ran

- deepinv-gke scheduled real GPU Jobs on 2026-07-17 and an L4 node scaled from zero.
- prj_resnetgcp10 recorded the same metadata-token error described below before this workstream.
- prj_resnetgcp11 and prj_resnetgcp12 later reached cell execution; gcp12 timed out after a long L4 training attempt.
- The July handoff says the final L4 path wrote artifacts, but there is no preserved IAM-policy snapshot proving the exact binding state at that time.
- deepinv-gke was deleted externally around 2026-07-17 22:00 UTC. That deletion is separate from the current GSA policy state.
- An older, separate openresearch-gpu cluster was deleted on 2026-07-11. It used a legacy node-service-account plus storage-rw route to a different bucket. It was not a Claude credential and is not a safe fallback for the current evidence contract.

## Current verified facts

### Authentication and permission boundary

The aayush@deepinvent.ai account was explicitly refreshed for both gcloud and ADC:

    gcloud auth login aayush@deepinvent.ai --force
    gcloud config set account aayush@deepinvent.ai
    gcloud config set project deepinvent-ext-ut
    gcloud auth application-default login
    gcloud auth application-default set-quota-project deepinvent-ext-ut

The account can create/delete a GKE cluster and attach deepinv-gke-node (iam.serviceAccounts.actAs). A direct read-only IAM test on deepinv-workload returned iam.serviceAccounts.getIamPolicy but not iam.serviceAccounts.setIamPolicy. Fresh authentication therefore does not grant the policy-write permission needed to repair the binding.

### IAM and storage

- deepinv-workload@deepinvent-ext-ut.iam.gserviceaccount.com currently has only an etag in its IAM policy: it has no roles/iam.workloadIdentityUser member for serviceAccount:deepinvent-ext-ut.svc.id.goog[reprolab/reprolab-sa].
- The workload GSA retains roles/storage.objectAdmin on gs://deepinvent-ext-ut-reprolab-artifacts.
- That bucket has uniform bucket-level access and public-access prevention enforced; there is no legacy ACL fallback for the node service account.
- The custom node service account is attachable, but current visible project/bucket policy exposes no write role for it on the current artifact bucket.

### Terraform-state evidence

The versioned object gs://deepinvent-ext-ut-tfstate/deepinvent/state/default.tfstate#1784270037040593 was written 2026-07-17 06:33 UTC. It records:

- cluster deepinv-gke and workload pool deepinvent-ext-ut.svc.id.goog;
- deepinv-gke-node attached to system and GPU pools; and
- deepinv-workload plus bucket object-admin access.

It records no google_service_account_iam_member or google_project_iam_member resource. This proves only that Terraform did not manage the Workload Identity binding in that state; it does not prove whether a manual binding existed later. The available audit lookup did not return a matching event, so no actor/time can be attributed.

## Probes performed

### Probe 1: baseline Workload Identity smoke

Temporary cluster: deepinv-gke-smoke-20260719 in us-central1-a.

1. Created a one-node e2-standard-2 CPU cluster with workload pool deepinvent-ext-ut.svc.id.goog.
2. The initial default node concealed the metadata server. It was replaced by a CPU-only wi-cpu pool configured with GKE_METADATA.
3. Created only reprolab/reprolab-sa, annotated it to deepinv-workload, and ran a public google/cloud-sdk:slim Job. The Job made no GCS call or write.
4. Metadata returned the intended GSA email with HTTP 200, then the token endpoint returned HTTP 403: Permission iam.serviceAccounts.getAccessToken denied.
5. Delete operation operation-1784522366167-2a78d7e2-c3a1-4946-80c2-dc35004e84e9 reached DONE; no cluster or corresponding node remained.

This showed the current Workload Identity failure but did not yet match the historical custom node identity.

### Probe 2: exact historical identity topology

Temporary cluster: deepinv-gke-historical-probe-20260719 in us-central1-a.

This decisive retry used:

- one e2-standard-2 CPU node;
- workload pool deepinvent-ext-ut.svc.id.goog;
- custom node service account deepinv-gke-node;
- cloud-platform OAuth scope plus GKE_METADATA; and
- reprolab/reprolab-sa annotated to deepinv-workload.

The cluster and node became Running/Ready. A public, no-write metadata Job returned:

    HTTP 200: deepinv-workload@deepinvent-ext-ut.iam.gserviceaccount.com

    HTTP 403: Unable to generate access token; IAM returned 403 Forbidden:
    Permission iam.serviceAccounts.getAccessToken denied.
    This error could be caused by a missing IAM policy binding on the target IAM service account.

This eliminates the temporary default-node identity as an explanation. The exact historical node configuration is accepted by GKE, but the keyless GSA exchange is currently unauthorized.

Delete operation operation-1784524836593-f46bb931-2854-4ef2-8ecd-c83732bc89f7 reached DONE. Explicit project listings then returned no GKE clusters and no gke-deepinv-gke-historic nodes.

## Issues encountered and disposition

| Issue | Evidence | Disposition |
|---|---|---|
| Wrong gcloud account / account-blind auth-plugin cache | Historical 404/403 Job submission failures. | Reauthenticate as aayush@deepinvent.ai; after a future get-credentials, bake CLOUDSDK_CORE_ACCOUNT into kubeconfig exec.env. Not the current pod-token failure. |
| ADC quota project mismatch | gcloud warning after account switch. | ADC refreshed and quota project set to deepinvent-ext-ut. |
| Default-node metadata concealment | First probe could not reach metadata server. | Use GKE_METADATA. The exact historical probe verified the custom-node topology. |
| A100 capacity stockout | Historical A100-40 scale-up failure. | Use an L4 pool only after identity passes. |
| GPU budget reservation | Historical run refused at projected cap. | Set strict GPU dollar, GPU-hour, and wall-clock caps. |
| 409 Job-name retry collision | Historical retry collided with failed Job. | Cloud runner was hardened; run hermetic collision tests before a future cloud launch. |
| Workload Identity token minting | Both probes returned live iam.serviceAccounts.getAccessToken 403. | Current hard gate. Do not launch a paid GPU/LLM reproduction. |
| Attribution | Terraform did not manage the binding; no matching audit event was returned. | Do not blame the operator or claim a binding deletion. |
| AWS readiness | No AWS CLI/profile/web identity/EKS/S3 runtime discovered. | AWS controller is hermetic only; no AWS cloud run was attempted. |

## Exact next IAM action

An IAM administrator must restore the keyless Workload Identity binding:

    gcloud iam service-accounts add-iam-policy-binding \
      deepinv-workload@deepinvent-ext-ut.iam.gserviceaccount.com \
      --project deepinvent-ext-ut \
      --member='serviceAccount:deepinvent-ext-ut.svc.id.goog[reprolab/reprolab-sa]' \
      --role=roles/iam.workloadIdentityUser

The current operator account cannot perform that action because it lacks iam.serviceAccounts.setIamPolicy. This is not a static key, node-credential, or broad project-access workaround.

Before production image pulls, also validate the custom node service account has the expected GKE and Artifact Registry roles. Retain the workload GSA's existing bucket-scoped object-admin role.

## Safe restart sequence after the binding exists

1. Read the GSA policy and verify the exact KSA member appears.
2. Recreate deepinv-gke through the six-gate GKE handoff. Use a scale-to-zero L4 pool if A100 capacity is unavailable.
3. Retrieve credentials and bake aayush@deepinvent.ai into kubeconfig exec.env.
4. Run one capped CPU/no-write metadata probe, then one capped cell-matrix GCS marker preflight; delete the marker Job afterward.
5. Run one strict-cap L4 cell-matrix shadow-scheduler reproduction. Verify GCS metrics/provenance, nodes/pods, and provider/token spend. Do not treat a zero cost ledger as proof of zero spend.
6. Authority-on remains prohibited until three valid paired A/B runs, the grader-sigma gate, and the scheduler runtime evidence prerequisites are satisfied.

## Scheduler implementation handoff

Worktree: .claude/worktrees/scheduler-authority-enrichment
Branch: scheduler-authority-runtime

Scheduler foundations, enrichments, cloud-controller hardening, and the authority audit seam were committed in the existing milestone series, including fee8a5c6, 66786bf8, 9194bf17, 7cb6f420, b146696d, and the final diagnosis correction 08d42210.

OPENRESEARCH_SCHEDULER_AUTHORITATIVE remains default-off and audit-only, recording applied:false. This is intentional: verified receipts, frozen step ladders, durable branch queue/frozen pool, controller-owned lineage emission, and valid paired cloud evidence are required before an action may apply.

Previously completed verification includes the campaign/ASHA/GKE/AWS hermetic suite (685 passed, 1 legacy warning). The latest documentation guards are green:

    PYTHONPATH="$PWD" .venv/bin/python -m pytest \
      tests/test_flag_registry_fresh.py tests/test_claude_md_fidelity.py \
      -p no:cacheprovider -q

### Publication and implementation status — 2026-07-20

The committed worktree branch is now published at
`origin/scheduler-authority-runtime` as
[draft PR #10](https://github.com/Deepinvent/scientific_article_generator/pull/10).
It was deliberately published as a review handoff only. A simulated merge
against current `origin/main` has eight content conflicts, including campaign,
Kubernetes runner, CLI, flag-registry, and campaign-CLI test files. Do not merge
the PR until it has been rebased and every conflict has been reviewed.

The current focused hermetic command passed **352 tests** across the ASHA core,
authority gate, scheduler evidence, campaign regressions, flag registry, and
documentation-fidelity guards. This does not make the authority flag live. An
independent runtime audit reconfirmed that the serial campaign still lacks a
runtime producer for the frozen paper-step ladder and verified resumable
metric/checkpoint receipt; it also lacks a durable branch queue/frozen pool,
grade-free closed-world decision artifact, and controller-owned idempotent
action/event executor. Thus the flag must remain default-OFF and audit-only
(`applied:false`). No GCP or AWS resource was created or changed while
publishing or documenting this status.
