# Azure sandbox — fix-pass plan (2026-05-27)

**Context:** Initial Azure sandbox implementation landed under the `azure` branch (PR forthcoming). A second analytical pass surfaced 9 concrete bugs + 1 documentation gap. This document itemizes each, plus the exact fix and a per-item verification check.

**Goal:** ship Azure with no known correctness or cost bugs; tests cover every fix; commit + push.

---

## Bug table

| # | Severity | File | Symptom | Fix |
|---|---|---|---|---|
| 1 | **HIGH** | `azure_backend.py:_create_vm` | `Premium_LRS` hardcoded in `os_disk.managed_disk.storage_account_type`; incompatible with B-series VMs (used by the live smoke). Real Azure rejects with `"VMSizeNotSupportingPremiumStorage"`. | Make `REPROLAB_AZURE_OS_DISK_TIER` configurable; default to `StandardSSD_LRS` which works on every VM SKU and is ~3× cheaper than Premium for ephemeral OS disks. |
| 2 | **HIGH** | `azure_backend.py:_delete_vm_piecemeal` | OS-disk name `f"{vm_name}-osdisk"` doesn't match Azure's auto-generated `{vm}_OsDisk_1_<guid>` form. Piecemeal delete returns 404 → orphaned managed disk → ongoing $/GB-month bill on preexisting-RG runs. | Set `os_disk.name = f"{vm_name}-osdisk"` explicitly in `_create_vm` params so the piecemeal delete finds it. |
| 3 | **MEDIUM** | `azure_backend.py:_create_public_ip` | `Basic` SKU + `Dynamic` allocation. Azure announced `Basic` SKU deprecation by 2025-09-30; new subscriptions can't create it. | Switch to `Standard` SKU + `Static` allocation. Compatible everywhere, forward-compatible. |
| 4 | **MEDIUM** | `azure_backend.py:_delete_vm` | `await poller.result()` on RG cascade — Azure RG delete takes 5-15 min. Blocks `destroy()` and stalls the run. | Issue `begin_delete` and immediately return. Don't await `.result()`. The cascade continues server-side. |
| 5 | **LOW** | `azure_backend.py:_create_nsg` | Standard PIP requires NSG to explicitly permit inbound — current `AllowSSH` rule is correct, but no `DenyAllInbound` lower-priority rule needed (Azure adds it implicitly). No code change. | Verify only — no fix. |
| 6 | **LOW** | `azure_backend.py:_provision_vm` | `vnet_id = getattr(subnet, "id", "").rsplit("/subnets/", 1)[0]` — fragile string parse. | Capture `vnet.id` directly from the `_create_vnet` return value; pass it through. |
| 7 | **HIGH** | `scripts/azure_check.sh:smoke` | Inherits FIX 1 bug — B1s + Premium_LRS = failure. | Resolved by FIX 1. Verify the smoke passes the env through correctly. |
| 8 | **LOW** | `azure_backend.py:_create_vm` | `diagnostics_profile.boot_diagnostics = {"enabled": True}` — modern API (2020-12-01+) accepts this for managed boot diagnostics. Older API versions need `storage_uri`. | Use `compute_client.api_version` default which is recent enough. Document the API floor in the runbook. |
| 9 | **DOC** | `docs/runbooks/2026-05-27-azure-sandbox.md` | Doesn't explain that Azure VM SKUs have fixed GPU counts — `REPROLAB_FORCE_SINGLE_GPU=true` is a no-op on Azure since the SKU itself fixes the count. | Add a "GPU count is fixed by VM SKU" note in the runbook. |
| 10 | **TEST** | `tests/services/runtime/test_azure_backend.py` | Doesn't verify the create params for storage tier, OS disk name, PIP SKU, or that RG cascade is non-blocking. | Add `test_create_uses_configured_storage_tier`, `test_create_sets_explicit_os_disk_name`, `test_public_ip_uses_standard_sku_static`, `test_destroy_does_not_block_on_rg_cascade`. |

---

## Fix recipe

### FIX 1 — Storage tier configurable

1. Add `azure_os_disk_tier: str = "StandardSSD_LRS"` to `backend/config.py`.
2. Add `REPROLAB_AZURE_OS_DISK_TIER=StandardSSD_LRS` to `.env.example`.
3. Add `os_disk_tier` parameter to `AzureBackend.__init__` (default `"StandardSSD_LRS"`).
4. Pass through from `_backend_for_sandbox_mode` in `primitives.py`.
5. Use `self.os_disk_tier` in `_create_vm`'s `managed_disk.storage_account_type`.

### FIX 2 — Explicit OS disk name

In `_create_vm` params, add to `storage_profile.os_disk`:
```python
"name": f"{vm_name}-osdisk",
```
Now the piecemeal `compute.disks.begin_delete(rg_name, f"{vm_name}-osdisk")` matches.

### FIX 3 — Standard public IP

`_create_public_ip` params:
```python
"public_ip_allocation_method": "Static",   # was "Dynamic"
"sku": {"name": "Standard"},               # was "Basic"
```

### FIX 4 — Non-blocking RG cascade

In `_delete_vm` ephemeral branch:
```python
# Issue the cascade delete; do NOT await poller.result().  Azure RG
# delete takes 5-15 minutes; blocking destroy on it stalls the run.
poller = ...resource_groups.begin_delete(rg_name)
# Optional: brief wait_for status with a tight timeout to confirm
# Azure accepted the request, then return.
```

### FIX 5 — Verify NSG/NIC/PIP SKU alignment

After FIX 3 (Standard PIP), NIC must also be Standard-compatible. The NIC params don't have a `sku` field; the SKU is implicit. Verify no error in test.

### FIX 6 — Robust vnet_id

`_create_vnet` already returns the vnet object first. Capture it:
```python
return (vnet, subnet)
```
Then in `_provision_vm`:
```python
vnet, subnet = await self._create_vnet(...)
...
vnet_id=getattr(vnet, "id", ""),
```

### FIX 7 — Live smoke

After FIX 1, B1s + StandardSSD_LRS works. Verify the smoke env passes through. No code change beyond FIX 1.

### FIX 8 — Boot diagnostics

Current `{"enabled": self.boot_diagnostics}` is correct for API version 2020-12-01+. `azure-mgmt-compute>=30` defaults to 2023-09-01 which is fine. Document the floor.

### FIX 9 — GPU-count doc

Add to runbook §Limitations.

### FIX 10 — Tests

Four new tests:
- `test_create_uses_configured_storage_tier` — assert `storage_account_type == os_disk_tier`
- `test_create_sets_explicit_os_disk_name` — assert `os_disk.name == "{vm_name}-osdisk"`
- `test_public_ip_uses_standard_sku_static` — assert `sku.name == "Standard"` and allocation = Static
- `test_destroy_ephemeral_rg_does_not_block` — verify destroy returns quickly (no `poller.result()` await on the RG delete poller)

---

## Verification

After all fixes:
1. `pytest .claude/worktrees/azure/tests/services/runtime/test_azure_backend.py` — all green
2. `pytest .claude/worktrees/azure/tests/services/runtime/ .claude/worktrees/azure/tests/rlm/ .claude/worktrees/azure/tests/test_pod_sweep_scheduler.py` — no new regressions
3. The 9 main-branch-confirmed pre-existing failures remain (not ours to fix here)

## Sequencing

Apply fixes in dependency order: FIX 6 (helper) → FIX 1 (storage tier) → FIX 2 (osdisk name) → FIX 3 (PIP) → FIX 4 (non-blocking) → FIX 5 (verify) → FIX 8 (verify) → FIX 9 (doc) → FIX 10 (tests).

Single commit at the end covering the full landing.

---

## Push

```bash
# Ensure clean diff
git status

# Stage everything (use specific paths to avoid sweeping in pre-existing diffs
# like data/calibration.json)
git add backend/ tests/ docs/ scripts/ start.sh CLAUDE.md .env.example pyproject.toml

# Commit with a focused message
git commit -m "..."   # HEREDOC, Co-Authored-By footer

# Push to a new remote branch
git push -u origin azure
```
