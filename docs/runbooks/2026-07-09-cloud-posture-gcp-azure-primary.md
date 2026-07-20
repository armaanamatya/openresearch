# Runbook: GCP/Azure as Primary Clouds, RunPod Legacy (2026-07-09)

**Spec:** `docs/superpowers/specs/2026-07-09-cloud-posture-gcp-azure-primary-design.md`

## Decision

GCP (GKE) and Azure (AKS) are now the primary supported backends for GPU campaigns.
RunPod is legacy — still usable via `--sandbox runpod` but never selected automatically.
`--sandbox auto` resolves to docker/local ONLY; it never escalates to a paid remote backend.

## The 6 changes and their operator knobs

| # | Change | Operator knob |
|---|--------|---------------|
| 1 | Foundry LLM rows price via `FOUNDRY_ALIASES` — no more $0 in ledger | None (automatic via `pricing.FOUNDRY_ALIASES`) |
| 2 | Explicit `--vram-gb` is used verbatim; skips the 1.25× headroom factor | `--vram-gb <N>` |
| 3 | `gcp_gpu_skus` mismatch fails loud at GCP preflight; default `["gcp_a100_80x8"]` is Terraform-synced | `OPENRESEARCH_GCP_GPU_SKUS=gcp_a100_80x8,...` |
| 4 | `--sandbox auto` resolves to docker/local only; `DEFAULT_SANDBOX_MODE = auto` | `--sandbox gcp\|azure\|runpod` for cloud |
| 5 | `--sandbox runpod` logs a one-line legacy notice | None (informational) |
| 6 | A cell that would breach `--max-run-gpu-usd` mid-flight is killed (`gpu_budget_exceeded`) | `--max-run-gpu-usd <$>` |

## Recommended campaign launch defaults (GCP)

```bash
python -m backend.cli campaign <arxiv-id> \
  --sandbox gcp \
  --model opus-foundry \
  --force-single-gpu \
  --max-run-gpu-usd 40 \
  --max-gpu-hours 8
```

With recommended env:
```bash
export OPENRESEARCH_GKE_SYNTH_CELL=1      # auto-generate cells.json + train_cell.py
export OPENRESEARCH_FEASIBILITY_SCOPE=1   # pre-GPU feasibility triage
```

## Recommended campaign launch defaults (Azure)

```bash
python -m backend.cli campaign <arxiv-id> \
  --sandbox azure \
  --model opus-foundry \
  --max-run-gpu-usd 40 \
  --max-gpu-hours 8
```

## Cost visibility reminders

- `cost_ledger.jsonl` is now accurate for Foundry models (no longer $0) — but GPU-idle
  time is still invisible. Verify real GPU spend via `kubectl get nodes`.
- `--max-run-gpu-usd` caps per-cell spend mid-flight; set it on every overnight campaign.
- `OPENRESEARCH_GCP_GPU_SKUS` must match the Terraform-provisioned node pool; a mismatch
  now fails loud at preflight (`validate_gcp_skus_against_cluster`) rather than silently
  routing to the wrong hardware.
