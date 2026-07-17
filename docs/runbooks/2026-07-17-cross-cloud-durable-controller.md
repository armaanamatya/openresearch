# Cross-Cloud Durable Controller Runbook

Use this path when the reproduction must continue after the initiating laptop
disconnects. The dynamic controller is supported for campaign-backed RLM runs
on GKE and AKS.

## Provision Once

1. Deploy the cloud L1 infrastructure and install the matching Helm chart.
2. Enable the chart's `orchestrator.enabled=true` resources. This creates the
   `reprolab-orchestrator` ServiceAccount, namespaced Job RBAC, and cloud
   SecretProviderClass. The Deployment and CronJob may remain disabled.
3. Provision the RWX `reprolab-cache` PVC. It is mandatory for controller state,
   not an optional performance cache on this path.
4. Put the configured provider credentials in Secret Manager or Key Vault
   under the names expected by the chart. Autonomous mode requires
   `azure-foundry-api-key` because its root is `opus-foundry`.
5. Build and push `docker/orchestrator/Dockerfile` with a git-SHA tag. The image
   contains the backend, campaign profiles, Node, and the pinned Claude CLI.
6. Build and push the separate pinned GPU cell image.

GCP additionally needs the orchestrator GSA and Secret Manager module enabled.
Azure needs the orchestrator managed identity, Key Vault values, and workload
identity label/annotation chain configured by the chart.

For autonomous mode, enable the chart's `orchestrator.azureFoundry.enabled`
setting and set its non-secret `endpoint`. Optional `anthropicEndpoint`,
`opusModel`, and `sonnetModel` values override the derived route/default model
names. The GCP Secret Manager Terraform module creates the Foundry secret name;
the secret value is always added out of band. On Azure, add it with:

```bash
az keyvault secret set --vault-name VAULT \
  --name azure-foundry-api-key --value "$AZURE_FOUNDRY_API_KEY"
```

## Launcher Environment

Set the selected cloud's normal cluster, object-store, namespace, and cell-image
settings. Then set:

```bash
export OPENRESEARCH_DURABLE_CONTROLLER=1
export OPENRESEARCH_CAMPAIGN_MAX_LLM_USD=50
export OPENRESEARCH_CAMPAIGN_MAX_GPU_USD=100
export OPENRESEARCH_CAMPAIGN_MAX_GPU_HOURS=24
export AZURE_FOUNDRY_ENDPOINT=https://RESOURCE.services.ai.azure.com
```

The launcher forwards Foundry endpoint/model coordinates but never the API key.
The controller Job reads `azure-foundry-api-key` from its CSI volume.

For GCP:

```bash
export OPENRESEARCH_GCP_ORCHESTRATOR_IMAGE=us-central1-docker.pkg.dev/PROJECT/reprolab/reprolab-orchestrator:sha-COMMIT
gcloud auth application-default login
gcloud container clusters get-credentials CLUSTER --region REGION --project PROJECT
```

For Azure:

```bash
export OPENRESEARCH_AZURE_ORCHESTRATOR_IMAGE=REGISTRY.azurecr.io/reprolab-orchestrator:sha-COMMIT
az login
az aks get-credentials --resource-group RG --name CLUSTER
```

`OPENRESEARCH_{GCP|AZURE}_CONTROLLER_IMAGE` is an explicit rollout override.
Floating or untagged images are rejected. Optional overrides include
`OPENRESEARCH_CONTROLLER_RUNS_ROOT`, `OPENRESEARCH_CONTROLLER_READY_TIMEOUT_S`,
`OPENRESEARCH_CONTROLLER_BACKOFF_LIMIT`, and cloud-specific CPU pool labels.

## Start And Disconnect

Start only the local API, submit the paper, and wait for the 202 response. The
response is not returned until the controller pod is Running.

```bash
.venv/bin/uvicorn backend.app:create_app --factory --port 8000

curl --fail-with-body -X POST http://127.0.0.1:8000/runs/upload \
  -F paper=@paper.pdf \
  -F mode=rlm \
  -F sandbox=gcp \
  -F autonomous=true
```

Use `sandbox=azure` to keep the autonomous profile on AKS; otherwise autonomous
mode defaults to GCP. After the response contains `controller.jobName`, the API
and laptop are no longer in the execution path.

## Observe

```bash
kubectl get jobs,pods -n reprolab -l app=reprolab-controller
kubectl logs -n reprolab job/CONTROLLER_JOB -f
```

Cloud state paths:

- lease: `runs/<project>/rlm_state/owner.lease`
- staged inputs: `runs/<project>/controller-input/`
- status/report checkpoints: `runs/<project>/controller-state/`

The RWX PVC remains authoritative for full campaign resume state. Object-store
checkpoints are the operator/status surface, not a complete replacement for the
PVC.

## Failure Handling

- `ControllerNotReady`: the failed Job was confirmed absent. Fix the reported
  image/PVC/CSI/scheduling issue and resubmit remotely.
- `ControllerStuck`: remote liveness is ambiguous. Inspect/delete the named Job
  before retrying; never start a local driver for the same project.
- exit 2 / paused: provide the required operator decision, then launch a remote
  resume. Kubernetes intentionally does not retry it.
- exit 3 / money halt: repair or raise the explicit budget only after reviewing
  the ledger. Kubernetes intentionally does not retry it.
- expired lease: a sweeper/relaunch with a new owner increments the fence and
  reaps older-fence controller and cell Jobs after takeover.

## Paper-Coverage Pilot

Do not use a single ML benchmark as evidence of broad scientific coverage. Run
a budgeted, pre-registered pilot with at least these strata:

1. classical vision or tabular ML with public data and a short baseline;
2. modern transformer/RL work with an official repository and GPU cells;
3. graph or time-series ML with nonstandard dependencies;
4. computational biology using public data, such as single-cell classification,
   protein sequence prediction, or genomic variant modeling;
5. a paper with a deliberately unreproducible wet-lab claim.

For each paper record ingestion success, environment build success, executed
claim count, evidence-backed verdict, declared reductions, unresolved assets,
cost, wall time, and restart recovery. Wet-lab procedures must be reported as
out-of-scope gaps; only their computational analyses can be reproduced here.
