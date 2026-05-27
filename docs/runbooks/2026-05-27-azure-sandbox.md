# Azure compute sandbox — operator runbook

**Status:** Shipped 2026-05-27, branch `azure`.
**Related spec:** [`docs/superpowers/specs/2026-05-27-azure-sandbox-prompt.md`](../superpowers/specs/2026-05-27-azure-sandbox-prompt.md).

## What this gives you

`--sandbox azure` (CLI), `REPROLAB_DEFAULT_SANDBOX=azure` (env), or `{"sandbox":"azure"}` (HTTP) provisions an Azure Virtual Machine with a GPU SKU, uploads your project, runs commands over SSH, syncs artifacts back, and tears the VM down. The contract matches `--sandbox runpod` and `--sandbox brev` exactly — every primitive (`exec`, `copy_in`, `copy_out`, `destroy`, `probe_alive`, `soft_recover`) behaves identically.

**Independent of Azure OpenAI.** The `azure-gpt-4o` LLM root model (in `backend/agents/rlm/models.py`) and the Azure VM sandbox use separate auth surfaces. Having one configured does NOT auto-enable the other.

## Quickstart

```bash
# 1. Auth (any of: az login, service principal, managed identity)
az login

# 2. Generate an SSH key for VM access (one-time)
ssh-keygen -t ed25519 -f ~/.ssh/azure_ed25519

# 3. Set env in .env
cat >> .env <<EOF
REPROLAB_AZURE_SUBSCRIPTION_ID=<your-sub-id>
REPROLAB_AZURE_REGION=eastus
REPROLAB_AZURE_VM_SIZE=Standard_NC6s_v3
REPROLAB_AZURE_SSH_KEY_PATH=~/.ssh/azure_ed25519
EOF

# 4. Preflight (free)
scripts/azure_check.sh

# 5. Optional: actual VM smoke (~$0.01)
scripts/azure_check.sh --start-vm

# 6. Run a paper
python -m backend.cli reproduce 2605.15155 --sandbox azure --max-usd 5
```

## Cost expectations

| SKU (typical) | $/hr (eastus, on-demand) | Use case |
|---|---|---|
| `Standard_B1s` | $0.01 | smoke tests only (no GPU) |
| `Standard_NV6ads_A10_v5` | $0.91 | 24 GB A10 — small/medium papers |
| `Standard_NC6s_v3` | $3.06 | V100 16 GB — legacy default |
| `Standard_NC24ads_A100_v4` | $3.67 | A100 80 GB |
| `Standard_NC40ads_H100_v5` | $6.98 | H100 80 GB |
| `Standard_ND96isr_H200_v5` | $84.00 | 8× H200 — big runs only |

Cheapest dev configuration today is: **OpenAI for the root model ($1/run with `--model gpt-5`) + Azure A10 for compute ($0.91/hr × ~1 h = $0.91)** ≈ **$2/run total**. Subscription OAuth for Sonnet sub-agents stays $0 if you've run `claude login`.

`REPROLAB_MAX_RUN_GPU_USD` (default 10.0) bounds total spend per run. `REPROLAB_MAX_GPU_USD_PER_HOUR` (default 10.0) bounds per-hour spend. Both apply identically to RunPod and Azure.

## Auth surface table

| Auth mode | Required env vars | Notes |
|---|---|---|
| **`az login` (interactive)** | `REPROLAB_AZURE_SUBSCRIPTION_ID` | Easiest for local dev. Token cached in `~/.azure/`. |
| **Service principal** | `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, `AZURE_TENANT_ID`, `REPROLAB_AZURE_SUBSCRIPTION_ID` | Recommended for CI / headless prod. |
| **Managed identity** | `REPROLAB_AZURE_SUBSCRIPTION_ID` | Works automatically on Azure VMs that have an attached MI. |
| **VS Code / Azure Developer CLI** | `REPROLAB_AZURE_SUBSCRIPTION_ID` | DefaultAzureCredential picks these up if present. |

All four resolve through `azure.identity.DefaultAzureCredential` in priority order. If multiple are present, the SDK uses the first one that succeeds.

## Resource group strategy

| `REPROLAB_AZURE_RESOURCE_GROUP` | Behavior |
|---|---|
| **Empty (default)** | Ephemeral per-run RG named `reprolab-<run-id>-<rand6>`. `destroy()` cascade-deletes the whole RG, which cleans up vnet/NSG/disk/NIC/IP atomically. |
| **Populated** | Reuses the named RG. `destroy()` only deletes the VM + its owned nested resources (NIC, public IP, OS disk, NSG, vnet). The RG itself is preserved. |

Use a preexisting RG when corp policy requires resources to live in named RGs only.

## Logs and observability

Every run writes:

| File | Format | What it captures |
|---|---|---|
| `runs/<id>/azure_resource_log.jsonl` | JSON Lines | Every ARM call (`op`, `resource_id`, `region`, `sku`, `status`, `latency_ms`, `error`). One line per call. |
| `runs/<id>/azure_provisioning.log` | text | SSH-wait progress, image download, boot-state polling. Line-buffered, `tail -f`-friendly. |
| `runs/<id>/cost_ledger.jsonl` | JSON Lines | Existing cost-ledger entries get `cloud="azure"` and `vm_size` fields. |
| `runs/<id>/artifacts/` | directory | Same as RunPod — `/artifacts` on the VM is rsynced here after each `exec()`. |

## Common failures and fixes

### `QuotaExceeded` on first GPU VM

You need to file a quota increase. Azure ships zero GPU quota by default on most subscriptions. Open a support ticket via the Azure portal, "Subscription → Usage + quotas → Compute → Request increase". Picking quota family `NCASv3_T4` is fast (~minutes); GPU families like `NCv3` or `NDv4` need human review (~hours).

### `SkuNotAvailable` in your region

Not every region has every SKU. The preflight script (`scripts/azure_check.sh`) probes this before each run. Common workarounds:

- Switch region: `REPROLAB_AZURE_REGION=southcentralus` or `westus2`
- Switch SKU family: A100 (`NCadsA100v4`) vs H100 (`NCadsH100v5`)
- Use the resolver's escalation ladder — it walks up the Azure catalog automatically on OOM

### `SshConnectionTimeout` 30-60s after VM creation

The NSG ships open on port 22 from `0.0.0.0/0` by default — corp policy may block egress to non-RFC1918 IPs. Either:

- Add your corp egress range to the NSG (edit `_create_nsg` in `azure_backend.py`)
- Use a VPN that puts you on an allowed range

### Leaked VMs

The combination of `asyncio.shield` on destroy + atexit handler + the periodic `pod_sweep_scheduler` (which sweeps Azure too as of this branch) means leaks are rare. But to clean up manually:

```bash
# List all reprolab-tagged Azure resources:
az resource list --tag reprolab=true -o table

# Nuke ephemeral RGs (the most common leak):
az group list --tag reprolab=true --query "[].name" -o tsv | \
  xargs -n1 -I{} az group delete -n {} --yes --no-wait

# Or nuke individual VMs:
az vm list --query "[?tags.reprolab=='true'].{rg:resourceGroup, name:name}" -o tsv | \
  xargs -n2 -I{} sh -c 'az vm delete -g $1 -n $2 --yes --no-wait' _ {}
```

### `credit balance too low` or `unauthorized` from the LLM root mid-run

This is **the Azure OpenAI auth surface, not the compute sandbox**. See `CLAUDE.md` § "RLM auth — two surfaces" for the disentangling. The fix is independent: refill Azure OpenAI credits, or switch `--model` to `claude-oauth` / `gpt-5`. The VM sandbox continues to work either way.

## Sweep cadence

`backend/services/runtime/pod_sweep_scheduler.py` runs both the RunPod and Azure sweepers on the same interval. Defaults:

- Interval: `REPROLAB_POD_SWEEP_INTERVAL_S` (default 1800s = 30 min)
- Age threshold: `REPROLAB_POD_SWEEP_MAX_AGE_S` (default 7200s = 2 h)
- Disable: `REPROLAB_POD_SWEEP_ENABLED=false`

The Azure sweeper is a silent no-op when `REPROLAB_AZURE_SUBSCRIPTION_ID` is unset.

## Limitations (v1)

- **No spot VMs / low-priority.** Regular on-demand only.
- **No managed identity-based VM-to-VM auth.** Use SSH keys, same as RunPod.
- **No Azure ML Jobs / Batch / AKS.** Plain Compute VMs only.
- **One sandbox per run.** Can't run half on RunPod and half on Azure.
- **GPU count is fixed by VM SKU.** `REPROLAB_FORCE_SINGLE_GPU=true` is a no-op on Azure because the SKU dictates the GPU count (e.g. `Standard_ND96isr_H200_v5` is always 8× H200). To run on a single GPU, choose a single-GPU SKU like `Standard_NC40ads_H100_v5` (1× H100) or `Standard_NV6ads_A10_v5` (1× A10). The `AZURE_SKU_BY_SHORT_NAME` map in `backend/services/runtime/azure_backend.py` defaults every short_name to a single-GPU SKU.
- **Storage: ephemeral OS disk only by default.** `REPROLAB_AZURE_DATA_DISK_GB > 0` attaches a managed disk, but it dies with the VM (single-run lifetime). Future v2 may add Azure Files / Blob for persistent artifact storage.

## API version floor

The backend relies on `azure-mgmt-compute>=30,<32`, which ships API version 2023-09-01 by default. That version supports:

- Managed boot diagnostics via `diagnostics_profile.boot_diagnostics.enabled = True` (no `storage_uri` required)
- Standard SKU public IPs with `Static` allocation
- `StandardSSD_LRS` and `Premium_LRS` for OS disks
- Explicit `os_disk.name` override

If you pin to an older `azure-mgmt-compute` version (pre-2020-12-01 API), managed boot diagnostics fall back to needing an explicit storage account URI — set `REPROLAB_AZURE_BOOT_DIAGNOSTICS=false` to disable or pin a newer SDK.

## Future v2 work (not in this PR)

- Switching `REPROLAB_DEFAULT_SANDBOX` from `runpod` to `azure` after a bake period.
- Multi-cloud orchestration (run half on RunPod, half on Azure).
- Spot / low-priority VM support with eviction handling.
- Azure Files-backed artifact persistence across runs.
- A dedicated `/runs/<id>/azure-status` endpoint mirroring `runpod-status` for the lab UI.
