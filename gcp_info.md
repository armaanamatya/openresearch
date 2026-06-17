# GCP GPU Notes

Last verified: 2026-06-16

Purpose: read this before touching Google Cloud GPU infra. Do not create paid GPU resources without explicit confirmation.

## Project

```text
account: abheek@deepinvent.ai
project: deepinvent-ext-ut
region:  us-central1
zone:    us-central1-c
gcloud:  Google Cloud SDK 573.0.0
```

Verify:

```bash
gcloud config list --format='text(core.account,core.project,compute.region,compute.zone)'
```

## Target

Use one A2 VM for A100 training.

| Need | Machine type | GPU | vCPU | RAM |
|---|---:|---|---:|---:|
| 4x A100 40GB | `a2-highgpu-4g` | 4x 40GB | 48 | 340GB |
| 8x A100 40GB | `a2-highgpu-8g` | 8x 40GB | 96 | 680GB |
| 4x A100 80GB | `a2-ultragpu-4g` | 4x 80GB | 48 | 680GB |
| 8x A100 80GB | `a2-ultragpu-8g` | 8x 80GB | 96 | 1360GB |

Current usable zones checked by `gcloud`:

```text
A100 40GB: us-central1-b, us-central1-c, us-central1-f
A100 80GB: us-central1-c
```

Start with `a2-highgpu-4g`; move to 8 GPUs only after SSH, image boot, and `nvidia-smi` work.

## Current Quota

Checked in `us-central1` on 2026-06-16:

| Metric | Limit | Used |
|---|---:|---:|
| `NVIDIA_A100_GPUS` | 8 | 0 |
| `PREEMPTIBLE_NVIDIA_A100_GPUS` | 16 | 0 |
| `NVIDIA_A100_80GB_GPUS` | 0 | 0 |
| `PREEMPTIBLE_NVIDIA_A100_80GB_GPUS` | 0 | 0 |

Implication: 40GB A100 quota can fit one 8-GPU A2 VM quota-wise. 80GB A100 quota is unavailable.

Recheck:

```bash
gcloud compute regions describe us-central1 --format=json \
| jq -r '.quotas[] | select(.metric | test("NVIDIA_A100")) | [.metric, .limit, .usage] | @tsv'
```

Quota is regional permission, not live zonal capacity. Capacity errors still require trying another valid zone.

## Image

Use the current PyTorch Deep Learning VM family:

```text
pytorch-2-9-cu129-ubuntu-2404-nvidia-580
```

Latest image resolved on 2026-06-16:

```text
pytorch-2-9-cu129-ubuntu-2404-nvidia-580-v20260616
```

Recheck:

```bash
gcloud compute images describe-from-family \
  pytorch-2-9-cu129-ubuntu-2404-nvidia-580 \
  --project deeplearning-platform-release \
  --format='value(name)'
```

## Create Test VM

Print and confirm before running:

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

Then:

```bash
gcloud compute ssh a100-train-4g --zone=us-central1-c
nvidia-smi
```

## Scale Options

Change only the name, zone, machine type, and disk size:

```text
a100-train-8g       us-central1-b|c|f  a2-highgpu-8g   500GB
a100-80gb-train-4g  us-central1-c      a2-ultragpu-4g  500GB  requires 80GB quota
a100-80gb-train-8g  us-central1-c      a2-ultragpu-8g  500GB  requires 80GB quota
```

## Cleanup

Stop when idle; delete when done.

```bash
gcloud compute instances list
gcloud compute instances stop INSTANCE --zone=ZONE
gcloud compute instances delete INSTANCE --zone=ZONE
```

## Failure Checks

```text
Invalid region: use us-central1 for regional commands, not us-central1-c.
Quota exceeded: recheck regional NVIDIA_A100* quotas.
Capacity exhausted: try another valid zone for the same machine type.
Auth failed: run gcloud auth login and gcloud config set project deepinvent-ext-ut.
Python ADC failed: run gcloud auth application-default login.
```

## Rules

- Never commit `~/.config/gcloud/`, `.boto`, or `application_default_credentials.json`.
- Never hardcode credentials in scripts or docs.
- Never create multiple GPU VMs without confirmation.
- Always stop or delete unused GPU VMs.
