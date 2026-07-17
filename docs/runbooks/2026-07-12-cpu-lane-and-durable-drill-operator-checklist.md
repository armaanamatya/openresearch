# CPU Cloud Lane + Durable-Controller Drill — Operator Checklist

- **Date:** 2026-07-12
- **Applies to:** the durable-controller + CPU-lane implementation on
  `feat/gke-gpu-path-reproduction-reliability` (commits `c4a2fb6a`, `2eac72f4`, `f5b09f94`,
  `c12462b9`, `04406fd0`).
- **Design/plan:** `docs/superpowers/specs/2026-07-12-cloud-native-durable-and-cpu-lane-design.md`,
  `docs/superpowers/plans/2026-07-12-cloud-native-durable-and-cpu-lane.md`.
- **Scope:** everything below is an OPERATOR step (live GKE + real A100 money) that an agent cannot
  run. The code is landed, flag-gated default-OFF, and hermetically tested; these steps activate and
  certify it.

---

## 1. Provision the CPU node pool (serves BOTH the controller and CPU cells)

The durable controller is a CPU workload, and CPU-class papers (e.g. Adam) run as CPU Jobs — one
scale-to-zero CPU pool serves both:

```bash
gcloud container node-pools create cpu-pool \
  --cluster openresearch-gpu --region us-central1 \
  --machine-type e2-standard-8 --num-nodes 0 \
  --enable-autoscaling --min-nodes 0 --max-nodes 3 \
  --node-labels reprolab/pool=cpu
```

The label must match `OPENRESEARCH_CPU_POOL_LABEL` (default `reprolab/pool=cpu`).

## 2. Controller ServiceAccount RBAC

The in-cluster controller Pod needs GCS access (the lease + artifacts) and `batch/jobs`
create/list/delete (to launch + reap cell Jobs). Reuse the `reprolab-sa` Workload Identity binding
from the 2026-07-07 GKE bring-up; confirm the KSA can `create/list/delete` `jobs.batch` in the
`reprolab` namespace.

## 3. Deployment flags (this is how "default and on" is achieved — Option Y)

The code keeps every flag default-OFF (byte-identical) to preserve the repo invariant. Set these in
the **gcp deployment env / run-spec** so gcp runs are durable + CPU-cloud by default:

| Flag | Set to | Effect |
|---|---|---|
| `OPENRESEARCH_DURABLE_CONTROLLER` | `1` | route gcp runs through the durable controller |
| `OPENRESEARCH_CPU_CLOUD_CELLS` | `1` | route CPU-class cells to CPU Jobs |
| `OPENRESEARCH_CPU_POOL_LABEL` | `reprolab/pool=cpu` | must match the pool label from §1 |
| `OPENRESEARCH_CONTROLLER_READY_TIMEOUT_S` | `180` (default) | readiness gate before fallback |
| `OPENRESEARCH_CONTROLLER_BACKOFF_LIMIT` | `3` (default) | K8s-native controller restart budget |
| `OPENRESEARCH_GCP_GCS_BUCKET` | (your bucket) | required — else the durable submit fails soft to Popen |

With `OPENRESEARCH_DURABLE_CONTROLLER=0` (or unset), a gcp run is byte-identical to today
(`subprocess.Popen`). Any misconfiguration (no bucket, cluster unreachable, controller Pod not
ready-and-confirmed) degrades to a local run with a `durable_controller_fallback` warning — it never
crashes or splits-brain.

## 4. The Pod-kill durability drill (headline validation — certifies "proven")

Run a `sandbox=gcp` reproduction with the flags above, then:

1. **Kill the controller Pod mid-run.** Confirm a successor Pod (K8s `backoffLimit` restart, OR the
   sweeper) **reacquires the lease** and resumes the SAME campaign lineage (`--resume`), finalizing a
   real metric. Confirm the successor's `fence_epoch` is preserved on a same-owner restart and
   **bumped** on a takeover.
2. **Reaper.** Confirm the successor reaps the predecessor's older-fence cell Jobs (no orphaned A100
   bills) — `kubectl get jobs -l reprolab/project=<id>` shows only current-fence Jobs.
3. **No split-brain.** Confirm a superseded controller's writes are refused (its lease token is no
   longer current) and that the local fallback never runs while a remote Job is live.
4. **CPU-cloud metric neutrality.** Run Adam (arXiv 1412.6980) on gcp with `OPENRESEARCH_CPU_CLOUD_CELLS=1`
   and confirm it reproduces the `best_runs/adam` baseline within tolerance — the CPU Job path is
   measurement-identical to the local path. (Command-identity is unit-tested; this drill certifies
   env parity.)

Passing §4 authorizes treating the durable path as "proven" and, if desired, tightening the fallback.

## 5. What stays out (design non-goals, not lazy deferrals)

- **Azure/AKS lease parity** — the WS3 lease is GCS-only by explicit design non-goal; an Azure Blob
  ETag adapter is a separate future adapter.
- **WS-Ext (`result_fidelity` kind measurement)** — a verdict-red-line change; see the handoff
  `docs/runbooks/2026-07-12-phase3-ws3-ws2-implementation-handoff.md` §3b for the corrected anchors +
  the frozen-Adam validation it requires before flipping on.
