# Cross-Cloud Durable Controller

**Status:** implemented, live-cloud drill pending
**Clouds:** GKE/GCS and AKS/Azure Blob
**Entry path:** HTTP `StartRunRequest(mode="rlm", sandbox="gcp"|"azure")`

## Goal

The initiating laptop submits one controller Job, waits until its pod is
actually running, and may then disconnect. The controller owns the campaign,
object-store lease, persistent run state, and GPU/CPU cell Jobs until the
campaign reaches a terminal state.

## Runtime Shape

1. The API prepares the source PDF and optional run spec locally, then uploads
   both to the selected cloud's object store.
2. The launcher acquires an object-store CAS lease with a unique launch owner,
   builds a fenced CPU-only Job, submits it, and waits for a Running pod.
3. The Job mounts the `reprolab-cache` RWX PVC and the cloud Secrets Store CSI
   provider, downloads its staged inputs, and reacquires the same owner token.
   Autonomous Foundry credentials are mounted from `azure-foundry-api-key`;
   endpoint/model coordinates are non-secret Job configuration.
4. The in-pod supervisor reaps older-fence Jobs, starts `backend.cli campaign
   --resume --sandbox <cloud>`, renews the lease every 60 seconds, and stops the
   child process group if renewal fails.
5. Campaign state stays on the RWX PVC. Small operator-facing state files are
   also checkpointed to `runs/<project>/controller-state/` in GCS or Blob.
6. Kubernetes retries genuine process crashes. Exit 2 (operator pause) and exit
   3 (money halt) are normalized to Job success so `backoffLimit` cannot turn a
   deliberate stop into repeated spending.

## Correctness Invariants

- **Unique ownership:** a launch UUID is stable across retries of one Job but
  differs across laptops and sweeper takeovers. A second launcher cannot pose
  as a restart of the first.
- **Two tokens:** the storage generation/ETag changes on every heartbeat;
  `fence_epoch` changes only on a different-owner takeover. Job names and labels
  use the stable fence.
- **Fail closed:** once durable mode is selected, lease, storage, readiness, or
  Kubernetes errors never fall back to a laptop process.
- **Ambiguous submit:** a create error is retryable only after the Job is
  confirmed deleted. An unconfirmed result raises `ControllerStuck`.
- **Real readiness:** Job `active` is insufficient. Handoff succeeds only when
  a controller pod is Running/Succeeded.
- **Bounded heartbeat:** heartbeat interval is positive and at most one third
  of the 180-second lease TTL.
- **Pinned control plane:** the controller requires the full orchestrator image
  with a pinned tag or digest and never falls back to a GPU cell image.
- **Explicit money limits:** LLM dollars, GPU dollars, and GPU hours are required
  before submission.
- **No literal credentials in Job specs:** provider keys are projected from the
  configured cloud Secrets Store CSI class.
- **Request fidelity:** root model, execution mode, GPU mode, and compute
  minimization are persisted in each attempt directive and explicitly forwarded
  to every `reproduce` child.

## Scope

The durable HTTP path drives campaign-backed `mode="rlm"`. `rdr` and
`rlm-pure` continue through their existing process paths until they have an
equivalent resumable outer ledger. The campaign and ingestion stack accepts
arXiv IDs, DOIs, and PDFs; scientific-domain breadth is validated separately
from controller durability.

## Validation Boundary

Hermetic tests cover CAS races, takeover fencing, pod readiness, input staging,
secret-free manifests, heartbeat loss, exit normalization, and both cloud
selection paths. A real GKE and AKS drill is still required before claiming
production availability. Paper coverage claims require a measured corpus; code
and unit tests alone do not establish “vast majority” reproduction rates.
