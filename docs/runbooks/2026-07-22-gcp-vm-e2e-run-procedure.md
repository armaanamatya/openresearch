<!-- doc-meta: status=current; last-verified=2026-07-22 -->
# GCP single-VM end-to-end run procedure (canonical)

**Date:** 2026-07-22 · **Status:** current — this is the go-forward "normal" GCP GPU path.

This is the canonical how-to for running a reproduction end-to-end on **GCP using the
single-VM `VmComputeProvider` path** (no Kubernetes). It supersedes the older GKE-first GCP
runbooks for day-to-day use.

## Where this fits (read first)

- **GKE is parked.** As of 2026-07-22, `--sandbox gcp` / `--sandbox gke` **fails loud**
  (`_backend_for_sandbox_mode` raises: *"GKE parked — use the campaign VM path"*). Set
  `OPENRESEARCH_ALLOW_GKE=1` to revive it, but only once the two blocked IAM grants are
  fixed (artifactregistry.reader + workloadIdentityUser). Design + rationale:
  [`docs/superpowers/specs/2026-07-22-restore-gcp-vm-path-surgical-degke-design.md`](../superpowers/specs/2026-07-22-restore-gcp-vm-path-surgical-degke-design.md).
- **The supported GCP GPU path is the single-VM `VmComputeProvider`**
  (`backend/services/runtime/vm_compute_provider.py`), reached via the **campaign**, NOT via
  the `--sandbox` flag. It provisions one GCE GPU VM, `scp`s the code in, runs the
  reproduction *inside* the VM as `--sandbox local`, `scp`s artifacts back, and tears the VM
  down. No Kubernetes, no Artifact Registry pull, no Workload Identity, no GCS — so it
  sidesteps both blocked IAM grants.
- Cloud posture (2026-07-22): primary clouds are **GCP + Azure**; RunPod, Brev, and Railway
  were removed; the default sandbox is `local`.

## Prerequisites checklist

Verified present on the operator box on 2026-07-22:

- [ ] `gcloud` authenticated as `aayush@deepinvent.ai`, project `deepinvent-ext-ut`.
- [ ] Application Default Credentials present (`gcloud auth application-default login`).
- [ ] Default zone `us-central1-b` (or another zone you have GPU capacity in).
- [ ] GPU quota in `us-central1` (verified unused on 2026-07-22): `NVIDIA_A100_80GB=4`,
      `NVIDIA_A100` (40GB) `=8`, `NVIDIA_L4=8`. **Zonal stockout is real** — pick a zone with
      capacity; `VmComputeProvider` classifies stockout stderr signatures.
- [ ] Operator has the GCE permissions the path needs
      (`compute.instances.create/delete/setMetadata/get/setTags`, `compute.disks.create`).

Quick checks:

```bash
gcloud config list account project
gcloud auth application-default print-access-token >/dev/null && echo "ADC OK"
gcloud compute regions describe us-central1 \
  --format='table(quotas.filter("metric~NVIDIA").flatten())'
```

## LLM configuration — Azure Foundry is the live path

As of 2026-07-22 the only working LLM auth surface is **Azure Foundry (OAuth-free)**:

- The `.env` OpenAI key is **DEAD** (401 — stale `sk-svcacct-`, rotated per the 2026-07-21
  secret-leak audit). Anthropic is empty/unset. There is no `claude` OAuth session.
- Foundry was validated live (HTTP 200) on 2026-07-22.

So runs use Foundry for both the root model and the sub-agents:

```
--model grok            # or opus-foundry / sonnet-foundry
--models executor=grok,verifier=grok,grader=grok
```

**Caveat:** grok/foundry is NOT SDAR-paper-validated, so the run emits an advisory
`role_model_fidelity` warning. That is fine for plumbing smokes; for a canonical fidelity
result use an SDAR-validated model once one is available on a working auth surface.

Foundry LLM spend is **invisible in `cost_ledger.jsonl`** (`opus-foundry`/`sonnet-foundry`
rows log 0 tokens and $0). A `$0` ledger is NOT proof of $0 — track real LLM spend via Azure
Cost Management (`az`), not the ledger.

## Command shape

The single-VM path is driven by the **`unified` campaign driver** with a **local** run
sandbox and a **gcp billing sandbox** (the `--billing-sandbox gcp` is what routes provisioning
through `VmComputeProvider`; `--sandbox gcp` would instead hit the parked/fail-loud GKE path):

```bash
python -m backend.cli campaign <paper> \
  --campaign-driver unified \
  --sandbox local \
  --billing-sandbox gcp \
  --max-llm-usd <X> --max-gpu-usd <Y> --max-gpu-hours <Z>
```

All three money caps are **REQUIRED** (three independent meters). The path arms
`--instance-termination-action=STOP` + `max_run_duration_s` as a control-plane cost ceiling.

### Fresh-VM identity (override the stale defaults)

`VmComputeProvider`'s built-in defaults point at an **OLD 4×A100 instance** `sdar-a100-od` /
SSH user `abheekp` that **no longer exists**. For a fresh VM under your own SSH key, override
the identity via env (these are read dynamically on each access — shell wins):

```bash
export OPENRESEARCH_GCP_PROJECT=deepinvent-ext-ut
export OPENRESEARCH_GCP_INSTANCE=<fresh-vm-name>        # e.g. gcp-vm-smoke-20260722
export OPENRESEARCH_GCP_ZONE=us-central1-b
export OPENRESEARCH_GCP_SSH_USER=<your-oslogin-user>   # derive from `gcloud compute os-login describe-profile --format='value(posixAccounts[0].username)'`
export OPENRESEARCH_REMOTE_DIR=/home/<your-oslogin-user>/openresearch
```

> **Placeholder to fill:** `<your-oslogin-user>` — derive it from your OS Login profile
> (command above) or from the local part of your `gcloud` account; do NOT reuse the retired
> `abheekp`. `OPENRESEARCH_REMOTE_DIR` should live under that same user's home.

Machine type: the campaign path builds `VmSpec` with only `max_run_duration_s`, so it falls back
to the **expensive 4×A100-80 default (`a2-highgpu-4g`, ~$14/hr)**. Select a cheaper machine with
the env override (added 2026-07-22, commit `aa205f94`):

```bash
export OPENRESEARCH_GCP_GPU_MACHINE_TYPE=g2-standard-8   # 1×L4-24GB (~$0.70/hr)
# or a2-ultragpu-1g / a2-highgpu-1g for 1×A100-80
```

- **`g2-standard-8`** — 1×L4-24GB, ~$0.70/hr. Cheap plumbing smoke.
- **`a2-ultragpu-1g` / `a2-highgpu-1g`** — 1×A100-80GB. SDAR canonical fidelity run.
- **Default (unset) is `a2-highgpu-4g`** (4×A100-80GB) — expensive; do NOT use for a smoke.

## Two run tiers

### 1. Cheap L4 plumbing smoke (~$3–5)

Proves the full lifecycle — provision → stage → train → collect → teardown — and that a real
`final_report.json` returns locally. Use a small paper, an L4 machine type, and tight caps.

```bash
export OPENRESEARCH_GCP_PROJECT=deepinvent-ext-ut
export OPENRESEARCH_GCP_INSTANCE=gcp-vm-smoke-20260722
export OPENRESEARCH_GCP_ZONE=us-central1-b
export OPENRESEARCH_GCP_SSH_USER=<your-oslogin-user>
export OPENRESEARCH_REMOTE_DIR=/home/<your-oslogin-user>/openresearch

python -m backend.cli campaign <small_paper> \
  --campaign-driver unified \
  --sandbox local \
  --billing-sandbox gcp \
  --model grok \
  --models executor=grok,verifier=grok,grader=grok \
  --max-llm-usd 5 --max-gpu-usd 5 --max-gpu-hours 2
```

(The `g2-standard-8` / 1×L4 machine type is selected through the unified driver's `VmSpec`;
confirm the resolved machine type in the run's provisioning log before the create call bills.)

**Pass criteria:** the VM provisions, stages, trains, collects, tears down, and a real
`runs/<project_id>/final_report.json` + artifacts land locally.

### 2. A100 SDAR smallest-two (canonical fidelity — pricier)

The canonical fidelity validation: SDAR (arXiv **2605.15155**), smallest-two models
(**Qwen3-1.7B + Qwen2.5-3B**), on a 1×A100-80 machine type (`a2-ultragpu-1g` /
`a2-highgpu-1g`). Same command shape; swap the paper to `2605.15155`, use the A100 machine
type, and set caps sized for A100 node-hours (higher `--max-gpu-usd` / `--max-gpu-hours`).

## Stray-VM post-check (ALWAYS)

The single risk of this path is **stray VM billing if teardown fails**. The path caps with
`--instance-termination-action=STOP` + `max_run_duration_s`, but **always** post-check after
every run to confirm no VM was left running:

```bash
gcloud compute instances list --project deepinvent-ext-ut
# Expect: your run's VM STOPPED/absent. If it is still RUNNING, stop it:
gcloud compute instances stop <fresh-vm-name> --zone us-central1-b --project deepinvent-ext-ut
```

Watch the VM come up and then go away during a healthy run — no lingering RUNNING instance
means no stray billing.

## Validated direct recipe (proven green end-to-end 2026-07-22)

> **This is the recipe that actually ran the full pipeline on a GCP GPU and returned a real
> `final_report.json`.** The campaign/`VmComputeProvider` path above is currently **SDAR-specific**
> and does NOT run an arbitrary paper as-is (`scripts/sdar_gcp_run.sh` hardcodes SDAR — see
> Gotchas). Until a generic VM launcher lands, run any non-SDAR paper with these direct steps.

**Result 2026-07-22:** Adam (arXiv 1412.6980) on a fresh 1×L4 VM — the GCP path was validated
end-to-end (ingest → workspace → agent pipeline → experiment on the GPU → `final_report.json` +
on-disk evidence). The reproduction *verdict* was `failed` **only** because the `grok` executor
doesn't emit the harness's `commands.json` manifest — a model-quality issue, not infra. Use a
validated executor (Sonnet) on a working auth surface for a passing result.

```bash
Z=us-central1-a          # a zone with GPU capacity (see the stockout gotcha)
VM=openresearch-vmrun

# 1. Provision with AUTO-DELETE safety (nothing lingers). List image families first:
#    gcloud compute images list --project=deeplearning-platform-release --filter='family~cu12'
gcloud compute instances create $VM --project=deepinvent-ext-ut --zone=$Z \
  --machine-type=g2-standard-8 \                                     # 1xL4-24GB; a2-ultragpu-1g for 1xA100
  --image-family=common-cu129-ubuntu-2204-nvidia-580 --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE --boot-disk-size=200GB --boot-disk-type=pd-ssd \
  --max-run-duration=7200s --instance-termination-action=DELETE \   # HARD 2h auto-delete
  --scopes=cloud-platform

# 2. Wait for SSH (--quiet avoids the key-passphrase prompt hanging).
gcloud compute ssh $VM --zone=$Z --quiet --command='nvidia-smi -L; python3 --version'

# 3. Stage code. DO NOT `--exclude=runs` — it also drops backend/services/runs/.
tar czf /tmp/or.tgz --exclude=node_modules --exclude=.git --exclude=.venv --exclude=__pycache__ \
  backend scripts configs pyproject.toml
gcloud compute scp /tmp/or.tgz $VM:~/ --zone=$Z --quiet

# 4. Bootstrap py3.12 + orchestrator deps (uv is fast).
gcloud compute ssh $VM --zone=$Z --quiet --command='
  mkdir -p ~/or && tar xzf ~/or.tgz -C ~/or && cd ~/or
  curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH
  uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r backend/requirements.txt'

# 5. Stage LLM creds WITHOUT echoing secrets (Foundry is the live surface).
grep -E "^AZURE_FOUNDRY_(API_KEY|ENDPOINT|DEPLOYMENT)=" .env > /tmp/vm.env
printf 'OPENRESEARCH_MIN_DISK_GB=0\nANTHROPIC_API_KEY=\nOPENAI_API_KEY=\n' >> /tmp/vm.env
gcloud compute scp /tmp/vm.env $VM:~/or/.env --zone=$Z --quiet; rm -f /tmp/vm.env

# 6. Run the reproduction INSIDE the VM as --sandbox local, detached.
gcloud compute ssh $VM --zone=$Z --quiet --command='
  cd ~/or && export PATH=$HOME/.local/bin:$PATH
  nohup .venv/bin/python -m backend.cli reproduce <PAPER_ID> \
    --mode rlm --sandbox local --model grok \
    --models executor=grok,verifier=grok,grader=grok \
    --force-single-gpu --max-wall-clock 3000 --project-id run1 \
    > ~/reproduce.log 2>&1 & echo pid=$!'

# 7. Monitor until runs/run1/final_report.json exists (poll ~every 90s).
gcloud compute ssh $VM --zone=$Z --quiet --command='cd ~/or; tail -5 ~/reproduce.log; ls runs/run1/final_report.json 2>/dev/null'

# 8. Pull results, then DELETE the VM (or let the 2h auto-delete fire).
gcloud compute scp $VM:~/or/runs/run1/final_report.json ./ --zone=$Z --quiet
gcloud compute instances delete $VM --zone=$Z --quiet
gcloud compute instances list --project=deepinvent-ext-ut    # confirm no stray VM
```

## Gotchas & failure modes — every issue hit on 2026-07-22, with the fix

| Symptom | Root cause | Fix |
|---|---|---|
| `image family ... was not found` | DLVM image families rotate | List first (`gcloud compute images list --project=deeplearning-platform-release --filter='family~cu12'`); current is `common-cu129-ubuntu-2204-nvidia-580` |
| create fails `STOCKOUT` / `does not have enough resources` | Zonal GPU stockout (L4 was out in `us-central1-b`) | Loop zones until one has capacity (`-a`, `-c`, `-f`, other regions). L4 is offered in us-central1-a/b/c |
| "no A100/L4 quota!" panic | You truncated the quota list with `head` | Grep the FULL `regions describe … --flatten='quotas[]'`. Real quota (2026-07-22): A100-80=4, A100-40=8, L4=8 |
| `timeout: command not found` | macOS has no GNU `timeout` | Don't wrap `gcloud` in `timeout`; rely on its own connection timeout |
| Run dies instantly / LLM 401 | `.env` OpenAI + Anthropic keys are dead (rotated in the 2026-07-21 leak) | **Validate LLM auth BEFORE provisioning** (cheap ping). Foundry is the live surface |
| `ModuleNotFoundError: backend.services.runs` | `tar --exclude=runs` also excludes `backend/services/runs/` | When tarring only `backend/`, drop the `runs` exclude entirely |
| Reproduces SDAR instead of your paper | `scripts/sdar_gcp_run.sh` hardcodes `2605.15155`; `VmComputeProvider.launch` treats `paper_id` as a hint only | Run `reproduce <paper>` directly on the VM (recipe above), or build a generic VM launcher |
| Campaign VM path always 4×A100 | `attempt_driver` builds `VmSpec` with no machine type → default `a2-highgpu-4g` | Set `OPENRESEARCH_GCP_GPU_MACHINE_TYPE=g2-standard-8` (commit `aa205f94`) |
| SSH hangs on a passphrase prompt | first `gcloud compute ssh` generates a key interactively | Always pass `--quiet` |
| `--sandbox gcp/gke` raises "GKE parked" | Intentional fail-loud (commit `86c00abe`) | Use the VM path; set `OPENRESEARCH_ALLOW_GKE=1` only after the IAM grants are fixed |
| verdict `failed`, evidence `no commands.json` | `grok` is NOT a validated executor (doesn't emit the cell/commands manifest) | Use a validated executor (Sonnet) via a working Anthropic key/OAuth |
| Fear of stray VM billing | teardown / self-stop can be missed | Provision with `--max-run-duration=Ns --instance-termination-action=DELETE`; ALWAYS `gcloud compute instances list` after |
| `add-iam-policy-binding` PERMISSION_DENIED | operator account lacks `setIamPolicy` (only the owner has it) — this is exactly why GKE is blocked | The single-VM path needs NO IAM grants (uses the instance SA) — that is the whole reason it's the go-forward path |

**One-line rule:** GCP GPU work runs on the **single-VM path** (fresh VM + `reproduce --sandbox
local` + auto-delete), never GKE. Before you provision, verify three things — **LLM creds live,
GPU quota present, chosen zone has capacity** — or you pay to spin up a VM that dies.

## History — previous GCP approaches that FAILED (do not repeat)

The single-VM path is the go-forward answer *because* every other GCP approach hit a wall. The
dead-ends, so nobody re-treads them:

| Approach tried | Why it failed | Source |
|---|---|---|
| **GKE as the default sandbox** (landed on `main` 2026-07-17) | The GKE backend needs two IAM grants that were never applied live and the operator account **cannot `setIamPolicy`**: `artifactregistry.reader` on the node SA (→ `ErrImagePull`, pod never starts) and `workloadIdentityUser` on the workload SA (→ `getAccessToken` 403, pod can't reach GCS). Every pod failed before running. | de-GKE spec §Problem; memory `gcp-gke-setup` |
| **The `kubectl` "local-transport" workaround** | Shipped code + evidence through the K8s control plane (`kubectl cp`) to dodge the broken Workload-Identity/GCS path. Rejected: it works *around* the real IAM gap, produces a **non-authoritative** run with no GCS provenance, and can't back a reproduction claim. | audit `2026-07-21-things-we-tried-and-got-wrong.md` (X1) |
| **Just granting the IAM ourselves** | Ran the two `gcloud … add-iam-policy-binding` commands → both `PERMISSION_DENIED` (aayush is `roles/editor`; only the owner `aljo` holds `setIamPolicy`). Check roles before attempting a grant. | audit (L3) |
| **`gcp_gpu_skus` left at the default `["gcp_a100_80x8"]`** | The phantom 8×A100 SKU doesn't match the cluster's real `reprolab/sku` pool labels → `GpuResolutionError` under `--force-single-gpu`, or an unschedulable Pending Job. The #1 GKE footgun. | runbook `2026-07-08-multipaper-gcp-overnight.md` + runtime CLAUDE.md |
| **`--vram-gb 80` on an 80GB-max A100 fleet** | The resolver inflates by ×1.25 headroom → needs 100GB → no SKU exists → `GpuResolutionError` on *every* call → the run can never get a GPU (the root "recovers" the traceback, then burns LLM budget toward a doomed experiment). Pick `N` so `ceil(N×1.25)` lands inside a real SKU. | 2026-07-08 incident + runtime CLAUDE.md |
| **New GKE node pool without the workload-identity SA** | A pool created without `--service-account <node-sa>` gets the `default` compute SA → the pod's GCS write fails `getAccessToken denied` **after** a full (billed) training run — looks like a code bug. Node-pool SA can't change in place → delete + recreate. | 2026-07-17 validation |
| **GKE auth via the account-blind plugin cache** | `gke-gcloud-auth-plugin` caches tokens keyed by cluster, not account; if the active account reverts to a no-IAM one, Job creation 404s/403s (masked as "1 cell failed with non-OOM errors"). Fix: bake `CLOUDSDK_CORE_ACCOUNT` into kubeconfig `exec.env` — and re-apply it, since every `get-credentials` wipes it. | 2026-07-17 handoff |
| **Failed cell Job not auto-deleted → 409** | A failed `reprolab-cell-*` Job isn't cleaned up, so a retry hits `409 Conflict` and the run can't self-heal — a known `k8s_job_cell_runner` gap (delete-then-create / unique names is the fix). | runtime CLAUDE.md |
| **Concluding "no straight-GCP path exists"** | Searched by guessed filenames (`gce`, `compute_instance`) and missed `vm_compute_provider.py` — the GCE-VM path that had been the original SDAR-on-GCP route all along. Lesson: search by **capability**, not assumed names. | audit (L1) |
| **Presenting a partial cell-matrix run as a "reproduction"** | A direct GPU cell-matrix controller ran 3/9 cells, one seed, no root model / scheduler / rubric — **no `final_report.json`, no score** — but was implied as paper progress. An executor smoke is not a reproduction. | audit (X2) |

**The through-line:** every GKE failure traces to IAM / Workload-Identity the operator can't
grant. The single-VM path exists precisely to need **none** of it (it uses the instance service
account directly). So when someone proposes "let's just fix GKE," the honest cost is *the owner
(`aljo`) granting two IAM bindings + a cluster recreate + a paid L4 smoke* — not a code change.

## References

- Design/rationale (de-GKE, surgical):
  [`docs/superpowers/specs/2026-07-22-restore-gcp-vm-path-surgical-degke-design.md`](../superpowers/specs/2026-07-22-restore-gcp-vm-path-surgical-degke-design.md)
- Provider code: `backend/services/runtime/vm_compute_provider.py`
- Runtime-layer reference: [`backend/services/runtime/CLAUDE.md`](../../backend/services/runtime/CLAUDE.md)
  (see *Straight-GCP single-VM path*)
- Historical single-VM SDAR runbooks (kept as history):
  `docs/runbooks/2026-06-26-sdar-gcp-e2e-runlog.md`,
  `docs/runbooks/2026-07-01-sdar-gcp-reproduction-walkthrough.md`,
  `docs/runbooks/2026-07-08-multipaper-gcp-overnight.md`
