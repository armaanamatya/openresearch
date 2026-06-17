# reprolab-gke-cell-base

Base Docker image for ReproLab GKE single-cell training Jobs.

## Overview

Each Google Kubernetes Engine (GKE) Job runs one SDAR training cell inside
this image.  The entrypoint (`gke_cell_entrypoint.py`) handles:

1. Downloading project code from Google Cloud Storage.
2. Setting `HF_HOME` / `TRANSFORMERS_CACHE` / `PIP_CACHE_DIR` under the shared
   cache volume (`/mnt/reprolab-cache`) so weights are shared across cells.
3. `pip install`-ing project requirements through the cached mount (fast after
   the first run).
4. Executing `train_cell.py` **unchanged** with all `OPENRESEARCH_CELL_*` env vars
   injected by the orchestrator.
5. On CUDA OOM: self-retrying with the shrink ladder:
   - Attempt 0: original params
   - Attempt 1: `OPENRESEARCH_CELL_BATCH_SCALE=0.5` + `OPENRESEARCH_CELL_GRAD_CHECKPOINT=1`
   - Attempt 2+: `OPENRESEARCH_CELL_BATCH_SCALE=0.25` + `OPENRESEARCH_CELL_GRAD_CHECKPOINT=1`
6. Uploading `metrics.json`, per-attempt logs, and `status.json` to GCS.

## Build and push — BLOCKED

`docker build` and `gcloud auth configure-docker` could not be run in this
session (no Docker daemon, no GCP credentials).  Commands to execute when
tooling is available:

```bash
# Authenticate docker against the Artifact Registry host (one-time)
gcloud auth configure-docker REGION-docker.pkg.dev

# Build locally
docker build -t reprolab-gke-cell-base:latest docker/gke-cell-base/

# Tag for your Artifact Registry repo
AR_HOST=REGION-docker.pkg.dev          # e.g. us-central1-docker.pkg.dev
AR_REPO=your-gcp-project/REPO          # e.g. my-proj/reprolab
docker tag reprolab-gke-cell-base:latest ${AR_HOST}/${AR_REPO}/gke-cell-base:latest

# Push
docker push ${AR_HOST}/${AR_REPO}/gke-cell-base:latest
```

Set `OPENRESEARCH_GCP_BASE_IMAGE` (or `settings.gcp_base_image`) to the
resulting fully-qualified image reference.

## Environment variable contract

All variables are injected by `k8s_job_cell_runner.py` into the K8s Job spec.

| Variable | Required | Description |
|---|---|---|
| `OPENRESEARCH_CELL_ID` | Yes | Unique cell identifier (e.g. `qwen3_1b-sdar-alfworld-42`) |
| `OPENRESEARCH_CELL_PARAMS` | Yes | JSON-encoded cell params passed to `train_cell.py` |
| `OPENRESEARCH_CELL_OUTPUT_DIR` | Auto | Set by wrapper to internal scratch dir |
| `OPENRESEARCH_CELL_BATCH_SCALE` | Auto | Injected per-attempt by OOM ladder |
| `OPENRESEARCH_CELL_GRAD_CHECKPOINT` | Auto | Injected per-attempt by OOM ladder |
| `OPENRESEARCH_CELL_MAX_OOM_RETRIES` | No | Default 2; number of OOM retries after original attempt |
| `OPENRESEARCH_GCP_GCS_BUCKET` | Yes | GCS bucket name (artifact bus) |
| `OPENRESEARCH_GCP_PROJECT` | No | GCP project ID; optional — inferred from ADC / the metadata server when unset (the manifest does NOT inject it) |
| `OPENRESEARCH_BLOB_CODE_PREFIX` | Yes | Object-path prefix for uploaded project code (cloud-neutral) |
| `OPENRESEARCH_BLOB_OUTPUT_PREFIX` | Yes | Object-path prefix for cell output artifacts (cloud-neutral) |
| `OPENRESEARCH_CACHE_MOUNT` | No | Shared cache mount root; default `/mnt/reprolab-cache` |

Application Default Credentials (Workload Identity) are used for all GCS I/O —
no static storage key (service-account JSON) is ever present in the container.
The GKE Kubernetes service account is annotated to impersonate a GCP service
account, and the pod receives short-lived credentials from the metadata server.

## Exit code contract

The orchestrator (`k8s_job_cell_runner.py`) maps these exact codes:

| Exit | `status.json.outcome` | Meaning |
|---|---|---|
| 0 | `ok` | metrics parsed and all artifacts uploaded |
| 40 | `bootstrap_error` | GCS pull, cache setup, or pip install failure |
| 41 | `error` | Non-OOM trainer failure |
| 42 | `oom_shrink_exhausted` | Final OOM after full shrink ladder |
| 43 | `metrics_invalid` | Trainer exited 0 but metrics.json is missing/invalid |
| 44 | `artifact_upload_error` | Training succeeded but sentinel could not be uploaded |

`podFailurePolicy` rules in the K8s Job manifest map exit codes 40–44 to
`FailJob` immediately, preventing `backoffLimit` retries for application-level
failures.  `backoffLimit` retries are reserved for infrastructure failures
(node preemption, kubelet restart, etc.).

## GCS artifact layout

```
runs/<run_id>/code/                  # project code uploaded by orchestrator
runs/<run_id>/cells/<cell_id>/
  metrics.json                       # flat per-cell metrics
  status.json                        # sentinel (outcome, exit_code, timestamps)
  logs/
    attempt-0.log
    attempt-1.log
    ...
```

## Testability

The pure functions in `gke_cell_entrypoint.py` can be tested without a GPU,
without a Docker daemon, and without the `google-cloud-storage` package
installed:

- `plan_attempts(max_oom_retries)` — shrink ladder config list
- `is_oom(stderr)` — OOM pattern detection
- `classify_outcome(exit_code, stderr, metrics, is_last_attempt)` — exit code mapping
- `build_sentinel(...)` — status.json builder

See `tests/services/runtime/test_gke_cell_entrypoint.py`.
