# Azure GPU Compute Backend — Scope (2026-06-02)

**Status:** SCOPE / not built. Companion to the LLM-on-Azure path (already wired: `--model azure`, `azure_openai_client.py`). This doc covers the *compute* surface only — the `describe_capacity` Azure stub that raises `NotImplementedError` (`backend/services/runtime/gpu_capacity.py::_describe_azure`).

**Confirm before building (one line, not a blocker):** the existing scaffolding is ambiguous about IaaS vs PaaS — `_AZURE_VM_SKU_CATALOG` + `OPENRESEARCH_AZURE_VM_SIZE` lean **VM/IaaS**, but `OPENRESEARCH_AZURE_IMAGE=mcr.microsoft.com/azureml/curated/...` + datastore-mount language in `baseline_implementation.py::_resolve_cloud_hardware` lean **Azure ML/PaaS**. This scope assumes **IaaS (Azure GPU VM over SSH)** — see §2. If the operator actually wants AML managed jobs, see §6 (different, larger shape).

## 1. Center of gravity (verified)

The RLM cloud experiment path is: `run_experiment` (`backend/agents/rlm/primitives.py`) → `_execute_in_sandbox` (~`:2795`) → an internal **backend factory** (~`:2189–2236`) that today constructs `RunpodBackend(run_budget, gpu_plan)` and raises "not supported in the RLM" for unknown `sandbox_mode`. Cloud backends do **not** use `gpu_cell_runner.run_matrix` (that path is gated on `backend_kind in ("local","docker")`); they run the agent's monolithic `commands.json` remotely and recover via **SKU escalation**, not one-cell-per-GPU placement.

→ The work is: implement `AzureBackend(RuntimeBackend)` mirroring `RunpodBackend`, add one branch to that factory, and feed an Azure SKU into `gpu_plan` so pricing/capacity/escalation all light up. ~80% is a structural clone of `runpod_backend.py`.

## 2. Backend (IaaS, SSH-exec) — the bulk of the work

New `backend/services/runtime/azure_backend.py` implementing the `RuntimeBackend` ABC (`backend/services/runtime/interface.py`, 5 methods): `create_sandbox / exec / copy_out / copy_in / destroy`. Mirror `RunpodBackend` patterns exactly:
- **Provision:** create an NC/ND-series GPU VM (`az` SDK or REST). Inject SSH public key (`OPENRESEARCH_RUNPOD_SSH_KEY_PATH` analog → `OPENRESEARCH_AZURE_SSH_KEY_PATH`). Poll for boot + SSH ready with a deadline.
- **exec/copy:** identical `asyncssh` SSH+SFTP path as RunPod (`/work`↔project_root, `/artifacts`↔artifact_root, per-run venv bootstrap).
- **destroy:** `asyncio.shield` + owned-instance allowlist (`_owned_vm_ids`) so a cancel never leaks a paid GPU VM.
- **error classes:** map 401/403 → non-retryable; transient 5xx / boot-timeout → retryable (mirror the RunPod-500 latch handling).

## 3. Wiring checklist (cite by symbol — verify lines on edit)

- `backend/agents/rlm/primitives.py` — add `azure` branch to the `_execute_in_sandbox` backend factory (~`:2228`) → `AzureBackend(run_budget, gpu_plan)`. **This is the one edit that actually enables RLM Azure runs.**
- `backend/agents/execution.py::SandboxMode` (`:32`) — add `azure = "azure"`; add `ensure_azure_available()` branch to `ensure_sandbox_mode_available`.
- `backend/cli.py` — add `"azure"` to the `--sandbox` choices tuple + help text.
- `backend/config.py` — add `"azure"` to `default_sandbox` / `force_sandbox` Literals.
- `backend/services/runtime/__init__.py` — export `AzureBackend`, `ensure_azure_available`.
- `backend/agents/baseline_implementation.py::_COST_BEARING_SANDBOXES` (`:1999`) — add `"azure"` (it bills).
- `backend/services/runtime/gpu_capacity.py::_describe_azure` (`:176`) — replace the stub. `_backend_kind` already routes `"azure"` (`:107`); since cloud capacity derives from `ctx.gpu_plan`, this can largely **delegate to `_describe_cloud(ctx, "azure")`** once gpu_plan carries an Azure SKU.
- **Azure SKU catalog + pricing + ladder** — the real net-new data. Add Azure entries (`Standard_NC*_A100_v4`, `Standard_ND96*_H100_v5`, …) with `vram_gb`, `gpu_count`, `usd_per_hr`, and an escalation `ladder` to `gpu_catalog`/`gpu_resolver` (today RunPod-shaped) + `services/pricing/catalog.py`. Reuse `_AZURE_VM_SKU_CATALOG` (GPU model/count/VRAM) already in `baseline_implementation.py` as the seed.
- Reuse existing env vars: `OPENRESEARCH_AZURE_VM_SIZE / _REGION / _IMAGE / _DATA_DISK_GB / _DATASTORE_GB / _DATASTORE_MOUNT` (+ new `_SUBSCRIPTION_ID / _RESOURCE_GROUP / _SSH_KEY_PATH`).
- Preflight: new `scripts/azure_check.sh` (sub/RG/quota/SSH-key) invoked from `start.sh` when sandbox=`azure`; add Azure asserts to `tests/test_provider_credentials.py`.

## 4. Risks (first-class, not footnotes)

1. **Azure GPU quota = 0 by default.** NC/ND vCPU quota needs a support-ticket increase (hours–days) before *any* run works. Blocks first-run regardless of code. Document in the runbook; surface a clear preflight error.
2. **No cloud OOM prevention.** `gpu_cell_runner` one-cell-per-GPU is local/docker-only. A single multi-GPU Azure VM running the monolithic `train.py` reintroduces exactly the single-card collapse the 2026-05-31 OOM spec fixed for local. Azure inherits the **cloud escalate path, which that spec marks "designed but not the verified target."** So Azure ships on an unverified recovery path — call this out; first SDAR run must be watched.
3. **Anchors drift.** Line numbers here are from a single read; edit by symbol.

## 5. Effort

Backend clone + SSH lifecycle (≈1–1.5 days incl. tests, the bulk) · SKU catalog/pricing/ladder + `_describe_azure` (≈0.5 day) · wiring + config + preflight + creds test (≈0.5 day) · one watched live SDAR smoke on a real GPU VM (cost). **~2.5–3 days + quota lead time.**

## 6. Out of scope / alternative

- **Azure ML (PaaS) managed jobs** — submit-job/poll/fetch-outputs does **not** fit the SSH-exec `RuntimeBackend` ABC; it is a separate execution layer and a materially larger effort. Only pursue if the operator confirms they want AML (the curated-image/datastore scaffolding hints at it). Not covered here.
- Spot/low-priority VM cost optimization; multi-VM distributed (torchrun across VMs); running `run_matrix` remotely over SSH on a multi-GPU VM (would restore cloud OOM-prevention — a worthwhile v2).
