# GCP GPU Setup

Operator guide for spinning up A100 GPU VMs on Google Cloud for training runs.

> **Read this before touching Google Cloud GPU infra.** Do not create paid GPU
> resources without explicit confirmation. Last verified: 2026-06-16.

Replace the `<...>` placeholders with your own values (set once via
`gcloud config set ...`); never commit real project IDs, account emails, or
credentials.

## 1. Project

```text
account: <YOUR_ACCOUNT_EMAIL>
project: <YOUR_GCP_PROJECT>
region:  us-central1
zone:    us-central1-c
gcloud:  Google Cloud SDK 573.0.0
```

Verify your active config:

```bash
gcloud config list --format='text(core.account,core.project,compute.region,compute.zone)'
```

## 2. Machine target

Use one A2 VM for A100 training.

| Need          | Machine type     | GPU       | vCPU | RAM     |
| ------------- | ---------------- | --------- | ---: | ------- |
| 4× A100 40 GB | `a2-highgpu-4g`  | 4× 40 GB  |   48 | 340 GB  |
| 8× A100 40 GB | `a2-highgpu-8g`  | 8× 40 GB  |   96 | 680 GB  |
| 4× A100 80 GB | `a2-ultragpu-4g` | 4× 80 GB  |   48 | 680 GB  |
| 8× A100 80 GB | `a2-ultragpu-8g` | 8× 80 GB  |   96 | 1360 GB |

Zones with A100 availability (checked via `gcloud`):

```text
A100 40GB: us-central1-b, us-central1-c, us-central1-f
A100 80GB: us-central1-c
```

Start with `a2-highgpu-4g`; move to 8 GPUs only after SSH, image boot, and
`nvidia-smi` all work.

## 3. Quota

Checked in `us-central1` on 2026-06-16:

| Metric                             | Limit | Used |
| ---------------------------------- | ----: | ---: |
| `NVIDIA_A100_GPUS`                 |     8 |    0 |
| `PREEMPTIBLE_NVIDIA_A100_GPUS`     |    16 |    0 |
| `NVIDIA_A100_80GB_GPUS`            |     0 |    0 |
| `PREEMPTIBLE_NVIDIA_A100_80GB_GPUS`|     0 |    0 |

Implication: 40 GB A100 quota fits one 8-GPU A2 VM; 80 GB A100 quota is
unavailable until granted.

Recheck:

```bash
gcloud compute regions describe us-central1 --format=json \
| jq -r '.quotas[] | select(.metric | test("NVIDIA_A100")) | [.metric, .limit, .usage] | @tsv'
```

Quota is a regional *permission*, not live zonal *capacity* — capacity errors
still require trying another valid zone.

## 4. Image

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

## 5. Create a test VM

Print and confirm the command before running it:

```bash
gcloud compute instances create a100-train-4g \
  --project=<YOUR_GCP_PROJECT> \
  --zone=us-central1-c \
  --machine-type=a2-highgpu-4g \
  --maintenance-policy=TERMINATE \
  --image-family=pytorch-2-9-cu129-ubuntu-2404-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=300GB \
  --boot-disk-type=pd-ssd \
  --scopes=https://www.googleapis.com/auth/cloud-platform
```

Then SSH in and confirm the GPUs are visible:

```bash
gcloud compute ssh a100-train-4g --zone=us-central1-c
nvidia-smi
```

## 6. Scale options

Change only the name, zone, machine type, and disk size:

```text
a100-train-8g       us-central1-b|c|f   a2-highgpu-8g    500GB
a100-80gb-train-4g  us-central1-c       a2-ultragpu-4g   500GB   requires 80GB quota
a100-80gb-train-8g  us-central1-c       a2-ultragpu-8g   500GB   requires 80GB quota
```

## 7. Cleanup

Stop when idle; delete when done.

```bash
gcloud compute instances list
gcloud compute instances stop   INSTANCE --zone=ZONE
gcloud compute instances delete INSTANCE --zone=ZONE
```

## 8. Troubleshooting

| Symptom              | Fix                                                              |
| -------------------- | --------------------------------------------------------------- |
| Invalid region       | Use `us-central1` for regional commands, not `us-central1-c`.    |
| Quota exceeded       | Recheck the regional `NVIDIA_A100*` quotas (§3).                 |
| Capacity exhausted   | Try another valid zone for the same machine type.               |
| Auth failed          | `gcloud auth login` && `gcloud config set project <PROJECT>`.   |
| Python ADC failed    | `gcloud auth application-default login`.                         |

## Rules

- Never commit `~/.config/gcloud/`, `.boto`, or `application_default_credentials.json`.
- Never hardcode credentials — or real project IDs / account emails — in scripts or docs.
- Never create multiple GPU VMs without confirmation.
- Always stop or delete unused GPU VMs.
