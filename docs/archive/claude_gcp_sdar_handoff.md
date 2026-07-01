# Claude Handoff — GCP SDAR / Grok Run

## Current state

- GPU VM is stopped: `sdar-a100-8g` is `TERMINATED`.
- VM is cheaper spot/preemptible capacity: `SPOT`, `preemptible=True`, `a2-highgpu-8g`.
- No local `gcp_sdar_preflight`, `sdar_gcp_assets`, `backend.cli reproduce`, or `batch_reproduce` process is running.
- Do not launch the full SDAR run until the GCP asset preflight returns GREEN.

Verify billing/process state:

```bash
CLOUDSDK_CONFIG=/home/abheekp/.config/gcloud \
gcloud compute instances describe sdar-a100-8g \
  --zone=us-central1-c --project=deepinvent-ext-ut \
  --format='table(name,status,scheduling.provisioningModel,scheduling.preemptible,machineType.basename())'

pgrep -af 'gcp_sdar_preflight.sh|sdar_gcp_assets.py|backend.cli reproduce|batch_reproduce.py' || true
```

## What changed

- Added `backend/requirements-sdar.txt`: explicit SDAR-only ML/env stack.
- Added `scripts/sdar_gcp_assets.py`: installs/checks SDAR deps, warms HF model/dataset caches, provisions ALFWorld/WebShop/Search-QA, writes `runs/.cache/sdar_gcp.env`.
- Added `scripts/gcp_sdar_preflight.sh`: `status/start/sync/check/prepare/stop` wrapper for the spot VM.
- `scripts/gcp_sdar_preflight.sh` now:
  - refuses non-spot GPU VMs by default with `OPENRESEARCH_REQUIRE_SPOT=true`;
  - waits through `STAGING/PROVISIONING` instead of issuing duplicate starts;
  - syncs source without `runs/`, venvs, `__pycache__`, or `.pyc`;
  - syncs `.env` when present;
  - bootstraps remote `.venv` + core requirements before SDAR asset prep.
- Updated `docs/runbooks/2026-06-16-sdar-on-gcp-a100-vm.md`, `CHANGELOG.md`, `docs/archive/learn.md`, and `issues.md`.

Validation already run locally:

```bash
bash -n scripts/gcp_sdar_preflight.sh
bash -n scripts/cancel_gcp_sdar_run.sh
python -m py_compile scripts/sdar_gcp_assets.py
git diff --check -- scripts/gcp_sdar_preflight.sh scripts/sdar_gcp_assets.py backend/requirements-sdar.txt
```

## Important blockers / caveats

- `.claude/skills/iterate.md` is missing in this repo; continue using repo runbooks and direct code inspection.
- Full asset prep has **not completed** yet. It was interrupted to stop billing.
- The previous Grok/Foundry run failed before useful GPU training due to missing SDAR assets:
  - `alfworld-download` missing;
  - `web_agent_site` / WebShop server not importable/running;
  - `transformers` stack missing/broken in generated cells.
- The earlier failed run project was `prj_77d3388db8e90d24`; final score was `0.0`.

## Safe next steps

1. Confirm the VM is stopped.
2. Run the gated prep only:

```bash
CLOUDSDK_CONFIG=/home/abheekp/.config/gcloud scripts/gcp_sdar_preflight.sh prepare
```

3. If and only if prep returns GREEN, launch the Grok run with the generated env file sourced:

```bash
set -a
. runs/.cache/sdar_gcp.env
set +a

env -u ANTHROPIC_API_KEY .venv/bin/python -m backend.cli reproduce 2605.15155 \
  --mode rlm --sandbox local --model grok \
  --models executor=grok,grader=grok,verifier=grok \
  --gpu-mode max --gpu-parallelism multi --vram-gb 40 \
  --no-force-single-gpu --max-wall-clock 86400
```

4. Stop billing immediately when done or blocked:

```bash
scripts/cancel_gcp_sdar_run.sh --stop-vm
```

## Operating rule

No full SDAR run until `scripts/gcp_sdar_preflight.sh prepare` is GREEN on the spot VM.
