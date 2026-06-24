# SDAR on a GCP A100 VM — launch runbook (2026-06-16)

Run the SDAR reproduction (arXiv 2605.15155) on a bare GCP **A2 GPU VM** using the
`local` sandbox + Claude subscription OAuth. LLM cost **$0** (OAuth subscription); you
pay only the A100 VM hours. This is the SDAR baseline handoff command's auth shape with
`--sandbox local` instead of `runpod`.

Current production preference: use **spot/preemptible A100 capacity** unless an
operator explicitly opts into on-demand. `scripts/gcp_sdar_preflight.sh` enforces
that by default (`OPENRESEARCH_REQUIRE_SPOT=true`) and refuses to start a non-spot
GPU VM. The existing `sdar-a100-8g` VM has been verified as `SPOT`, `preemptible=True`,
`a2-highgpu-8g`.

> Why `--sandbox local` and not `--sandbox gcp`: your `gcp_info.md` provisions a single
> long-lived **VM**, which the harness drives as host subprocesses (one-GPU-per-cell
> `run_matrix`) with no Docker/K8s. `--sandbox gcp` (GKE Jobs) targets a managed cluster,
> a different execution model. For a VM, `local` is the ready, fully-tested path.

Compute sizing (see the seed/compute analysis): **4–8× A100-40GB is enough; H100s are not
needed.** Each SDAR cell is sized 8–32 GB, embarrassingly parallel one-GPU-per-cell.
Quota note from `gcp_info.md`: 40GB A100 quota = 8 available, **80GB A100 quota = 0**.

---

## 0. Prerequisites (local workstation)

```bash
gcloud config list --format='text(core.account,core.project,compute.region,compute.zone)'
# expect: account=abheek@deepinvent.ai project=deepinvent-ext-ut region=us-central1 zone=us-central1-c
```

---

## 1. Provision the VM  (print & confirm before running — this spends money)

Start with 4 GPUs; scale to `a2-highgpu-8g` only after SSH + `nvidia-smi` work.

```bash
gcloud compute instances create a100-train-4g \
  --project=deepinvent-ext-ut \
  --zone=us-central1-c \
  --machine-type=a2-highgpu-4g \
  --maintenance-policy=TERMINATE \
  --image-family=pytorch-2-9-cu129-ubuntu-2404-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=300GB \
  --boot-disk-type=pd-ssd \
  --scopes=https://www.googleapis.com/auth/cloud-platform
```

## 2. SSH in and verify the GPUs

```bash
gcloud compute ssh a100-train-4g --zone=us-central1-c
nvidia-smi            # expect 4× A100-SXM4-40GB
```

## 3. Clone the repo and build the Python env

The fresh clone ships no venv; the harness needs Python 3.12+ (`rlms==0.1.1`).

```bash
git clone git@github.com:armaanamatya/openresearch.git
cd openresearch
git checkout feat/azure-bicep-canonical-aoai-hardening   # has GCP backend + everything merged

# uv is the supported path; falls back to python3.12 -m venv if uv absent
uv venv --python 3.12 .venv 2>/dev/null || python3.12 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt
.venv/bin/pip install -r backend/requirements-dev.txt   # parallel test runners (optional)
```

## 3b. Full-scope SDAR asset preflight — mandatory before a paid run

The full SDAR scope is not just "Python + GPUs". It needs the Qwen HF weights,
Search-QA datasets, ALFWorld game data, and the WebShop server module ready
before the RLM loop submits cells. Run the GCP preflight/warmer once on the VM:

```bash
scripts/gcp_sdar_preflight.sh prepare
```

This starts the VM if needed, syncs the repo, installs `backend/requirements-sdar.txt`,
warms HuggingFace model/dataset caches under `runs/.cache/`, provisions
ALFWorld/WebShop/Search-QA through `EnvCacheManager`, verifies 8 visible GPUs, and
writes:

```bash
runs/.cache/sdar_gcp.env
```

Source that file for any direct launch:

```bash
set -a
. runs/.cache/sdar_gcp.env
set +a
```

Fast validation without downloads:

```bash
scripts/gcp_sdar_preflight.sh check
```

If this is RED, do not launch the reproduction. Fix the missing package/data/server
first; otherwise the failure will happen mid-run while the A100 VM is billing.

The wrapper supports:

```bash
scripts/gcp_sdar_preflight.sh status   # includes spot/preemptible status
scripts/gcp_sdar_preflight.sh start    # refuses non-spot by default
scripts/gcp_sdar_preflight.sh stop
```

## 4. Authenticate — Claude subscription OAuth (zero LLM cost)

The harness detects OAuth by the **existence of `~/.claude/.credentials.json`**
(`factory.py::_has_claude_subscription_oauth`), so we want the credential *file* on the
VM, not just an env token.

```bash
# install the Claude CLI (the claude-agent-sdk shells out to it)
npm i -g @anthropic-ai/claude-code   # or the native installer

# headless login: prints a URL, open it in YOUR laptop browser, authorize,
# then paste the returned code back into this SSH session.
claude login
#   -> writes ~/.claude/.credentials.json on the VM

claude /status                  # confirm: "Authenticated with Claude subscription"
claude --print "ping"           # proves the subscription works
```

**Critical — unset any stale API key** (a no-credit `ANTHROPIC_API_KEY` silently shadows
OAuth → every run dies at the first Sonnet sub-call with `400 credit balance too low`,
`cost_usd=0.0`):

```bash
unset ANTHROPIC_API_KEY
# also make sure .env does not set a no-credit key:
grep -n '^ANTHROPIC_API_KEY=' .env 2>/dev/null   # should be empty/blank
# belt-and-suspenders: fail fast if OAuth isn't detected
export OPENRESEARCH_LLM_AUTH_STRATEGY=oauth_only
```

## 5. Smoke / validation run first — smallest-two scope (~2–5 h on 4 GPUs)

Validate the pipeline end-to-end cheaply (catches WebShop-install + 7B-on-40GB risks)
before committing to the full matrix. Pin to the two smallest models.

```bash
export OPENRESEARCH_BASELINE_EXTRA_GUIDANCE="SCOPE: reproduce SDAR using ONLY the two SMALLEST model variants the paper tests — Qwen3-1.7B and Qwen2.5-3B-Instruct. SKIP Qwen2.5-7B entirely. Use the real pretrained weights from HuggingFace (no surrogate) and the real ALFWorld + Search-QA + WebShop datasets, but evaluate on a small representative slice (e.g. 32 tasks per env) to keep wall-clock practical. Report results for both 1.7B and 3B."

# one run that sees all 4 GPUs; the cell scheduler runs cells one-GPU-each in parallel
env -u ANTHROPIC_API_KEY .venv/bin/python scripts/batch_reproduce.py 2605.15155 \
  --gpus-per-run 4 \
  --model claude-oauth
```

Equivalent single-CLI form (no batch scheduler):

```bash
env -u ANTHROPIC_API_KEY .venv/bin/python -m backend.cli reproduce 2605.15155 \
  --mode rlm --sandbox local --model claude-oauth \
  --max-wall-clock 18000 --max-usd 20
```

## 6. Full matrix run (after smoke passes)

Scale to 8 GPUs (`a2-highgpu-8g`, ~2× faster — the workload is embarrassingly parallel
across cells, so more GPUs is the right lever, not H100s). Drop the smallest-two scope
guidance to run all three models; expect **~4–7 h direct compute, longer through the RLM
iteration loop** (single seed; see the seed note below).

```bash
unset OPENRESEARCH_BASELINE_EXTRA_GUIDANCE
env -u ANTHROPIC_API_KEY .venv/bin/python scripts/batch_reproduce.py 2605.15155 \
  --gpus-per-run 8 \
  --model claude-oauth \
  --max-wall-clock 86400
```

For the Grok / Azure AI Foundry path, keep the same preflight requirement and
launch with the cached env file loaded:

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

---

## Notes & gotchas

- **Seeds.** The official SDAR scripts use a **single** seed (`env.seed=0`), not multiple.
  The SDAR `default_scope` in `paper_hints.py` is now `seeds=[0]` (the faithful, ~3× cheaper
  scope; was `[42,43,44]`). Multi-seed is NOT required to match the paper — widen via
  `--scope-spec` only if you explicitly want a variance study.
- **cu129 vs env_pin (cu121).** The DLVM image is torch 2.9 / CUDA 12.9; the harness
  `env_pin` installs a cu121 torch core into the per-run venv. cu121 wheels run
  forward-compatibly on the nvidia-580 driver, so this is normally fine. If you hit a CUDA
  lib conflict at import, disable it: `export OPENRESEARCH_DISABLE_ENV_PIN=1` (uses the
  agent's pins / the DLVM torch).
- **7B on 40GB.** The 7B cell sits at the 40GB memory edge (≈32 GB × 1.25 headroom). The
  built-in OOM shrink-retry (batch 0.5→0.25 + grad-ckpt) handles it; or request
  `NVIDIA_A100_80GB_GPUS` quota (us-central1-c only) and use `a2-ultragpu`.
- **WebShop** is install-risk (Java simulator); if it fails it's recorded as a documented
  gap, not a crash.
- **Watch a live run:** `tail -f runs/<id>/code/.exec_live.log` (local-sandbox streaming).

## Cleanup (stop when idle, delete when done — GPU VMs bill while running)

```bash
gcloud compute instances stop a100-train-4g --zone=us-central1-c
gcloud compute instances delete a100-train-4g --zone=us-central1-c
```
