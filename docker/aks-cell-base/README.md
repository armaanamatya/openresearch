# reprolab-aks-cell-base

Base Docker image for ReproLab AKS single-cell training Jobs.

## Overview

Each Azure Kubernetes Service (AKS) Job runs one SDAR training cell inside
this image.  The entrypoint (`aks_cell_entrypoint.py`) handles:

1. Downloading project code from Azure Blob Storage.
2. Setting `HF_HOME` / `TRANSFORMERS_CACHE` / `PIP_CACHE_DIR` under the
   Azure Files PVC (`/mnt/reprolab-cache`) so weights are shared across cells.
3. `pip install`-ing project requirements through the cached mount (fast after
   the first run).
4. Executing `train_cell.py` **unchanged** with all `REPROLAB_CELL_*` env vars
   injected by the orchestrator.
5. On CUDA OOM: self-retrying with the shrink ladder:
   - Attempt 0: original params
   - Attempt 1: `REPROLAB_CELL_BATCH_SCALE=0.5` + `REPROLAB_CELL_GRAD_CHECKPOINT=1`
   - Attempt 2+: `REPROLAB_CELL_BATCH_SCALE=0.25` + `REPROLAB_CELL_GRAD_CHECKPOINT=1`
6. Uploading `metrics.json`, per-attempt logs, and `status.json` to Blob.

## Build and push — BLOCKED

`docker build` and `az acr login` could not be run in this session (no Docker
daemon, no Azure credentials).  Commands to execute when tooling is available:

```bash
# Build locally
docker build -t reprolab-aks-cell-base:latest docker/aks-cell-base/

# Tag for your ACR
ACR_SERVER=<your-acr-login-server>  # e.g. reprolabacr.azurecr.io
docker tag reprolab-aks-cell-base:latest ${ACR_SERVER}/reprolab/aks-cell-base:latest

# Authenticate and push
az acr login --name <your-acr-name>
docker push ${ACR_SERVER}/reprolab/aks-cell-base:latest
```

Set `REPROLAB_AZURE_BASE_IMAGE` (or `settings.azure_base_image`) to the
resulting fully-qualified image reference.

## Environment variable contract

All variables are injected by `k8s_job_cell_runner.py` into the K8s Job spec.

| Variable | Required | Description |
|---|---|---|
| `REPROLAB_CELL_ID` | Yes | Unique cell identifier (e.g. `qwen3_1b-sdar-alfworld-42`) |
| `REPROLAB_CELL_PARAMS` | Yes | JSON-encoded cell params passed to `train_cell.py` |
| `REPROLAB_CELL_OUTPUT_DIR` | Auto | Set by wrapper to internal scratch dir |
| `REPROLAB_CELL_BATCH_SCALE` | Auto | Injected per-attempt by OOM ladder |
| `REPROLAB_CELL_GRAD_CHECKPOINT` | Auto | Injected per-attempt by OOM ladder |
| `REPROLAB_CELL_MAX_OOM_RETRIES` | No | Default 2; number of OOM retries after original attempt |
| `REPROLAB_AZURE_STORAGE_ACCOUNT` | Yes | Azure storage account name |
| `REPROLAB_AZURE_BLOB_CONTAINER` | Yes | Blob container name (artifact bus) |
| `REPROLAB_BLOB_CODE_PREFIX` | Yes | Blob prefix for uploaded project code |
| `REPROLAB_BLOB_OUTPUT_PREFIX` | Yes | Blob prefix for cell output artifacts |
| `REPROLAB_CACHE_MOUNT` | No | Azure Files mount root; default `/mnt/reprolab-cache` |

`DefaultAzureCredential` (workload identity) is used for all Blob I/O — no
static storage keys are ever present in the container.

## Exit code contract

The orchestrator (`k8s_job_cell_runner.py`) maps these exact codes:

| Exit | `status.json.outcome` | Meaning |
|---|---|---|
| 0 | `ok` | metrics parsed and all artifacts uploaded |
| 40 | `bootstrap_error` | Blob pull, cache setup, or pip install failure |
| 41 | `error` | Non-OOM trainer failure |
| 42 | `oom_shrink_exhausted` | Final OOM after full shrink ladder |
| 43 | `metrics_invalid` | Trainer exited 0 but metrics.json is missing/invalid |
| 44 | `artifact_upload_error` | Training succeeded but sentinel could not be uploaded |

`podFailurePolicy` rules in the K8s Job manifest map exit codes 40–44 to
`FailJob` immediately, preventing `backoffLimit` retries for application-level
failures.  `backoffLimit` retries are reserved for infrastructure failures
(node preemption, kubelet restart, etc.).

## Blob artifact layout

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

The pure functions in `aks_cell_entrypoint.py` can be tested without a GPU,
without a Docker daemon, and without the `azure` package installed:

- `plan_attempts(max_oom_retries)` — shrink ladder config list
- `is_oom(stderr)` — OOM pattern detection
- `classify_outcome(exit_code, stderr, metrics, is_last_attempt)` — exit code mapping
- `build_sentinel(...)` — status.json builder

See `tests/services/runtime/test_aks_cell_entrypoint.py`.
