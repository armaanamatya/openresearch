# Azure compute sandbox — implementation prompt

**Worktree:** `/Volumes/CS_Stuff/openresearch/.claude/worktrees/azure`
**Branch:** `azure` (off `main` @ `022e3f4`)
**Date:** 2026-05-27
**Scope:** Add Azure as a first-class `--sandbox` backend, mirroring the existing RunPod / Brev surface. Must be **additive and backward-compatible** with `main` — do not regress RunPod, Brev, local, or docker paths.

---

## 1. Goal (one paragraph)

Add a new `RuntimeBackend` implementation, `AzureBackend`, that provisions an Azure VM with attached GPU(s), uploads the project tree, executes commands over SSH (matching the RunPod / Brev contract exactly), streams stdout/stderr through the existing SSE bridge, persists Azure-side resource state under `runs/<id>/`, and tears the VM down on `destroy()`. The user-visible surface is `--sandbox azure` (CLI), `REPROLAB_DEFAULT_SANDBOX=azure` (env), and `{"sandbox": "azure"}` (HTTP). Every existing RunPod-specific code path that branches on the string `"runpod"` must learn the string `"azure"`. Keep the RunPod default; Azure is opt-in until validated.

You are NOT replacing RunPod. You are adding a peer. `main` must keep passing tests on the RunPod path after this change.

---

## 2. Hard constraints

- **Backward compat:** `--sandbox runpod`, `--sandbox brev`, `--sandbox docker`, `--sandbox local`, and `--sandbox auto` must all behave identically to `main` after the diff. Don't touch their code paths except to add `azure` to the `Literal`/`choices`/`enum` they already enumerate.
- **No new abstractions for the sake of it.** Use the existing `RuntimeBackend` ABC at `backend/services/runtime/interface.py`. Don't refactor the ABC. Don't introduce a "cloud provider" mega-abstraction.
- **Mirror Brev, not RunPod, as the structural template.** `brev_backend.py` is the most recent peer (~949 LOC) and already documents the SSH/SFTP pattern, owned-instance allowlist, `asyncio.shield` on destroy, TOFU host-key pinning, and terminal-state detection. Copy that shape. RunPod (1452 LOC) is older, has more bespoke pod-attach/persistent-pod logic, and is heavier than you need.
- **Auth surface:** use the Azure SDK (`azure-identity` + `azure-mgmt-compute` + `azure-mgmt-network` + `azure-mgmt-resource`). Default credential chain via `DefaultAzureCredential`, with explicit `AZURE_CLIENT_ID` / `AZURE_CLIENT_SECRET` / `AZURE_TENANT_ID` (service principal) as the documented production path. **Do not** shell out to `az` CLI — Azure has a proper SDK; the Brev CLI-shellout was a workaround for Brev not having a documented REST surface, which does not apply here.
- **GPU VM, not Azure ML Job.** Azure ML batch jobs are declarative and don't fit our live-`exec`/`copy_in`/`copy_out` contract. Provision an Azure Virtual Machine with a GPU SKU (`NC`/`ND`/`NCads`-series), SSH in, behave exactly like RunPod's pod model.
- **Region + SKU come from env.** Don't hardcode `eastus` / `Standard_NC6s_v3`. They must be `REPROLAB_AZURE_REGION` / `REPROLAB_AZURE_VM_SIZE`, defaulting to values that work for most subscriptions.
- **Cost containment:** `destroy()` must run under `asyncio.shield` and must delete the VM, OS disk, NIC, public IP, and ephemeral resource group (if we created one). A leaked Azure VM at $3/hr is a worse outage than a leaked $0.34/hr RunPod pod. Mirror RunPod's `_register_atexit_cleanup` pattern.
- **Logs:** every Azure ARM call writes a structured line to `runs/<project_id>/azure_resource_log.jsonl` with `{ts, op, resource_type, resource_id, region, sku, status, latency_ms, error?}`. Provisioning console output goes to `runs/<project_id>/azure_provisioning.log` (line-buffered, tail-followable). Cost-ledger entries get `cloud="azure"`.
- **SSE bridge is unchanged.** stdout/stderr from `exec()` already routes through `sse_bridge.sanitize_iteration`. Do not bypass it. Do not invent new event types — `primitive_call` with `sandbox="azure"` is enough.
- **Tests must run without an Azure subscription.** All Azure SDK calls go through a thin wrapper that's monkeypatched in unit tests. The real Azure path is gated by `REPROLAB_AZURE_SUBSCRIPTION_ID` being set + an opt-in `pytest -m azure_live` marker.

---

## 3. Where to look first (read these before writing code)

| File | What it tells you |
|---|---|
| `backend/services/runtime/interface.py` | The `RuntimeBackend` ABC. `create_sandbox`, `exec`, `copy_in`, `copy_out`, `destroy`, `probe_alive`, `soft_recover`. ~169 LOC, don't change it. |
| `backend/services/runtime/brev_backend.py` | Structural template. ~949 LOC. SSH + SFTP via `asyncssh`. Owned-instance allowlist. `asyncio.shield` on destroy. Read this end-to-end before writing `azure_backend.py`. |
| `backend/services/runtime/runpod_backend.py` | Reference for the more bespoke bits: artifact sync, port-mapping discovery, atexit cleanup, soft-recover, probe-alive. Especially the `_sync_artifacts_to_host` flow (lines ~799–895) and `_delete_pod` (~1039). |
| `backend/services/runtime/__init__.py` | Re-export point. Add `AzureBackend` + `ensure_azure_available` here. |
| `backend/agents/execution.py` | `SandboxMode` enum (line ~43), `resolve_sandbox_mode` (line ~232), `ensure_sandbox_mode_available` (~271). Extend, don't rewrite. |
| `backend/config.py` | `Settings.default_sandbox` (`Literal["auto","local","docker","runpod"]`, line ~130) + `force_sandbox`. Add `"azure"` to both literals. Add `azure_*` fields. |
| `backend/cli.py` | `--sandbox` choices at line ~1331. Add `"azure"`. |
| `backend/app.py` | Lines 262, 301, 405 each branch on the literal `"runpod"`. Add `"azure"` to each. Also line 405 (`REPROLAB_RUNPOD_API_KEY` presence check for the lab UI's enabled-sandboxes list) — add an `AZURE_*` parallel. |
| `backend/agents/baseline_implementation.py` | Lines 714–838 already contain `_AZURE_VM_SKU_CATALOG` (Azure ML VM SKU → (GPU model, count, VRAM)) and `_AZURE_*` env-var dispatch in `_describe_gpu_for_prompt`. This is **prompt text only** — the codegen agent already knows how to write Azure-aware training scripts. You're adding the *execution backend* underneath. Don't duplicate the catalog; import it. |
| `backend/agents/rlm/models.py` | Already has `azure-gpt-4o` for the root LLM. Do NOT touch — that's a different Azure surface (LLM, not compute). The two are independent. |
| `scripts/runpod_check.sh` | Template for the new `scripts/azure_check.sh`. Same exit-code discipline (0 green, 2 missing env, 3 auth fail, 4 SKU not available in region, 5 SSH key bad, 6 end-to-end smoke fail). |
| `start.sh` | Calls `scripts/runpod_check.sh` when sandbox=runpod. Add the parallel azure branch. |
| `.env.example` | Lines 68–110 are the canonical RunPod env block. Lines 121–124 are the (LLM-only) Azure block. Add a new "Azure compute" block right after the RunPod one, mirroring its structure. |

---

## 4. Files to create

```
backend/services/runtime/azure_backend.py            # the new backend (~600–900 LOC)
scripts/azure_check.sh                                # preflight (auth + SKU + SSH key)
tests/services/runtime/test_azure_backend.py          # unit tests w/ monkeypatched SDK
tests/services/runtime/test_azure_backend_live.py     # opt-in real-Azure smoke (@pytest.mark.azure_live)
docs/runbooks/2026-05-27-azure-sandbox.md             # user-facing runbook: env, costs, common errors
```

## 5. Files to modify (additive only)

```
backend/services/runtime/__init__.py        # export AzureBackend, ensure_azure_available
backend/services/runtime/service.py         # if it has a backend dispatcher, add "azure" → AzureBackend
backend/services/runtime/pod_sweeper.py     # add a parallel azure_sweeper.py OR genericize — see §8
backend/config.py                            # +Literal "azure", +azure_* Settings fields, +force_sandbox literal
backend/agents/execution.py                 # SandboxMode.azure, ensure_sandbox_mode_available branch
backend/cli.py                              # --sandbox choices, --vram-gb still works for azure
backend/app.py                              # lines 262, 301, 405: add "azure" alongside "runpod"
backend/agents/rlm/failure_classifier.py    # line 65: generalize the "persistent pip cache" hint
backend/agents/rlm/primitives.py            # lines 2092, 2604: REPROLAB_RUNPOD_AUTO_FALLBACK → also honor REPROLAB_AZURE_AUTO_FALLBACK (or rename to REPROLAB_SANDBOX_AUTO_FALLBACK, keeping the old name as an alias)
backend/agents/baseline_implementation.py   # confirm _AZURE_VM_SKU_CATALOG is reachable from azure_backend.py; move shared bits to gpu_catalog.py if needed (don't reshape its prompt-emitting role)
backend/services/runtime/gpu_resolver.py    # the dynamic-GPU plan (`gpu_plan.json`) must produce an Azure VM size when sandbox=azure. Extend the resolver's catalog query, don't fork it
backend/services/runtime/gpu_catalog.py     # add Azure SKUs alongside RunPod SKUs OR add an Azure catalog with the same shape (8 SKUs analogous to RTX 4090 → H200)
.env.example                                # new "Azure compute" block
start.sh                                    # add azure preflight branch
```

---

## 6. The `AzureBackend` contract — exact methods

Use `brev_backend.py` as the line-by-line template and replace the Brev CLI calls with Azure SDK calls. The public method signatures are fixed by the ABC:

```python
class AzureBackend(RuntimeBackend):
    def __init__(
        self,
        *,
        subscription_id: str,
        region: str,
        vm_size: str,                  # e.g. "Standard_NC6s_v3"
        image: str = "",               # Azure VM image URN, e.g. "microsoft-dsvm:ubuntu-hpc:2204:latest"
        resource_group: str | None,    # None → create ephemeral RG per run
        ssh_key_path: str | Path | None,
        ssh_public_key: str = "",
        ssh_user: str = "azureuser",
        os_disk_gb: int = 100,
        data_disk_gb: int = 0,         # 0 = no extra disk
        boot_diagnostics: bool = True, # write serial console to a managed storage account
        delete_on_destroy: bool = True,
        tags: dict[str, str] | None,   # always include {"reprolab": "true", "project_id": ..., "run_id": ...}
        max_boot_seconds: float = 600,
        log_dir: Path | None,          # → runs/<id>/  for azure_resource_log.jsonl + azure_provisioning.log
    ) -> None: ...

    async def create_sandbox(self, config: SandboxConfig) -> Sandbox: ...
    async def exec(self, sandbox: Sandbox, command: str, timeout: int) -> ExecResult: ...
    async def copy_in(self, sandbox: Sandbox, path: str, data: bytes) -> None: ...
    async def copy_out(self, sandbox: Sandbox, path: str) -> bytes: ...
    async def destroy(self, sandbox: Sandbox) -> None: ...
    async def probe_alive(self, sandbox: Sandbox, *, timeout: float = 10.0) -> bool: ...
    async def soft_recover(self, sandbox: Sandbox) -> bool: ...
```

Inside `create_sandbox`:
1. Resolve credentials via `DefaultAzureCredential` (cached). Fail fast with `SandboxRuntimeError(backend_unavailable, retryable=False)` if no creds.
2. Create or reuse `resource_group` (if `None`, create `reprolab-<run_id>` and remember-to-delete it on destroy).
3. Create vnet/subnet (small `/29`), NSG with port 22 inbound from `*` (or restrict to caller IP if discoverable), public IP, NIC.
4. Create the VM with the configured SKU + image, inject the SSH public key via `osProfile.linuxConfiguration.ssh.publicKeys`.
5. Poll until the VM is `PowerState/running` (provisioning state separate). Cap at `max_boot_seconds`. Terminal states (`Failed`, `Deallocated`) → abort with `cause_kind=backend_unavailable`.
6. Open SSH (`asyncssh`, ed25519, known-hosts-policy = TOFU pin per session — copy from `brev_backend.py:_connect_ssh`).
7. `mkdir -p /home/azureuser/reprolab/code /home/azureuser/reprolab/artifacts`, then SFTP-upload the project tree (`_upload_directory` from Brev).
8. Optionally `apt-get update && apt-get install -y python3-pip` if the image isn't the DSVM (most CUDA images already have it; check first to save 30s).
9. Return a `Sandbox` whose `sandbox_id` is the VM's full ARM resource ID. Stash the resource group, NIC, public IP, OS disk IDs in a private dict keyed by `sandbox_id` so `destroy()` can find them.

Inside `destroy`:
1. Wrap the whole body in `asyncio.shield`.
2. Close the SSH connection (suppress all errors).
3. Sync `/artifacts` to `runs/<id>/artifacts/` via SFTP (mirror RunPod's `_sync_artifacts_to_host`). Allow 5 min, then give up.
4. Delete in this order: VM → NIC → public IP → OS disk → data disk(s) → NSG → vnet → ephemeral RG. Each delete swallows 404. Each delete writes a line to `azure_resource_log.jsonl`.
5. If the RG was ephemeral, the cleanest path is `resource_client.resource_groups.begin_delete(rg_name)` which cascades — use that instead of step 4 piecemeal; piecemeal is the fallback when the RG was preexisting.
6. Remove from the owned-instance allowlist.
7. Register an `atexit` handler at create time that calls `destroy()` synchronously (use `subprocess`-less ARM REST calls; the SDK has sync clients too — see `runpod_backend.py:_cleanup_atexit` for the pattern).

Inside `probe_alive`:
- Open a **fresh** SSH channel (not the cached one — that may be the wedged one). Run `true` with `timeout=timeout`. If it returns, alive.

Inside `soft_recover`:
- On the cached SSH channel, run `pkill -9 -f 'python|train.py|run_experiment'`. Return True on success. Do NOT delete the VM here — that's `destroy`'s job.

---

## 7. Env vars (drop into `.env.example`)

```bash
# ---------------------------------------------------------------------------
# Azure compute sandbox (--sandbox azure)
#
# Azure VM-based GPU execution.  Auth defaults to DefaultAzureCredential,
# which works with `az login` for local dev and service-principal env vars
# in CI/prod.  For headless prod, set AZURE_CLIENT_ID + AZURE_CLIENT_SECRET
# + AZURE_TENANT_ID (service principal).
#
# Cost note: Azure GPU VMs are $1–$15/hr depending on SKU.  Set
# REPROLAB_MAX_RUN_GPU_USD to bound spend; the existing RunBudget gate
# applies identically to Azure VMs.
# ---------------------------------------------------------------------------
REPROLAB_AZURE_SUBSCRIPTION_ID=
REPROLAB_AZURE_REGION=eastus
REPROLAB_AZURE_VM_SIZE=Standard_NC6s_v3
REPROLAB_AZURE_IMAGE=microsoft-dsvm:ubuntu-hpc:2204:latest
REPROLAB_AZURE_RESOURCE_GROUP=                        # empty → ephemeral per-run RG
REPROLAB_AZURE_OS_DISK_GB=100
REPROLAB_AZURE_DATA_DISK_GB=0
REPROLAB_AZURE_SSH_KEY_PATH=
REPROLAB_AZURE_SSH_PUBLIC_KEY=
REPROLAB_AZURE_SSH_USER=azureuser
REPROLAB_AZURE_BOOT_DIAGNOSTICS=true
REPROLAB_AZURE_DELETE_ON_DESTROY=true                  # parallel to REPROLAB_RUNPOD_DELETE_ON_DESTROY
REPROLAB_AZURE_AUTO_FALLBACK=false                     # parallel to REPROLAB_RUNPOD_AUTO_FALLBACK
REPROLAB_AZURE_BOOTSTRAP_COMMAND=
REPROLAB_AZURE_MAX_BOOT_SECONDS=600
REPROLAB_AZURE_TAGS=                                   # comma-separated k=v pairs, merged with reprolab defaults

# Service-principal auth (optional; DefaultAzureCredential picks these up):
# AZURE_CLIENT_ID=
# AZURE_CLIENT_SECRET=
# AZURE_TENANT_ID=
```

---

## 8. Pod sweeper — generalize, don't fork

`backend/services/runtime/pod_sweeper.py` currently sweeps stale RunPod pods owned by `REPROLAB_RUNPOD_API_KEY`. Two acceptable approaches:

**(a) Preferred:** rename internals to a neutral concept (sweep stale *sandboxes*), then have it dispatch to per-cloud sweepers. Add `azure_sweeper.py` with the same interface. Keep `pod_sweeper.py` as a façade that loops over enabled clouds. Document the rename in the runbook.

**(b) Acceptable:** leave `pod_sweeper.py` alone and add a sibling `azure_sweeper.py` + `pod_sweep_scheduler.py` learns to call both. Less elegant but lower-risk.

The sweeper finds stale resources by tag (`reprolab=true` + `created_at < now-24h`) and deletes them. Test in unit mode by monkeypatching `ComputeManagementClient.virtual_machines.list_all`.

---

## 9. Dynamic GPU resolution (spec 2026-05-23 — `gpu_resolver.py` + `gpu_catalog.py`)

The existing `resolve_gpu_requirements(...)` produces a `gpu_plan.json` with a RunPod SKU string. When `sandbox=azure`, it must instead produce an Azure VM size string from a catalog that maps the same hardware-clue space (VRAM-GB, count, compute capability) onto Azure SKUs.

Approach:
- Add `_AZURE_GPU_CATALOG` to `gpu_catalog.py` with ~8 SKUs analogous to the existing RunPod ladder, from cheapest (e.g. `Standard_NC4as_T4_v3`, $0.53/hr, 16GB T4) to highest (`Standard_ND96asr_v4` / `Standard_ND96isr_H100_v5`, 8×A100/8×H100). Source: Azure VM sizes documentation, GPU-accelerated section (verify via context7 if available).
- The resolver picks the cheapest SKU that meets `vram_gb * headroom_multiplier` and `gpu_count`.
- Escalation ladder works identically: on CUDA OOM, step up to the next SKU in the catalog, up to `REPROLAB_DYNAMIC_GPU_MAX_ESCALATIONS`.
- `REPROLAB_MAX_GPU_USD_PER_HOUR` and `REPROLAB_MAX_RUN_GPU_USD` apply unchanged.
- `gpu_plan.json` gains a `cloud: "azure" | "runpod"` field for clarity. Don't break readers — make it optional and default to the existing implicit cloud.

---

## 10. SSE / logging requirements

- Every Azure ARM call → one line in `runs/<project_id>/azure_resource_log.jsonl`:
  ```json
  {"ts":"2026-05-27T15:33:12Z","op":"VirtualMachines.begin_create_or_update","resource_type":"vm","resource_id":"/subscriptions/.../Microsoft.Compute/virtualMachines/reprolab-xyz","region":"eastus","sku":"Standard_NC6s_v3","status":"InProgress","latency_ms":423}
  ```
- Provisioning console (boot-wait stdout, image download, ssh-wait) → `runs/<project_id>/azure_provisioning.log`, line-buffered, follows `tail -f`.
- The existing `cost_ledger.jsonl` gets entries tagged `cloud="azure"` with `vm_size` and `region` fields. The existing `RunBudget.check_run_gpu_usd` is auth/cloud-agnostic — don't change its contract.
- The lab UI's "enabled sandboxes" list (`backend/app.py:405`) currently returns `runpod` only when `REPROLAB_RUNPOD_API_KEY` is set. Add the same gate for azure: return `azure` only when `REPROLAB_AZURE_SUBSCRIPTION_ID` is set AND credentials resolve via `DefaultAzureCredential` (cheap probe — call `SubscriptionClient.subscriptions.get`).

---

## 11. Tests — what must exist

**Unit (must pass in CI without an Azure subscription):**
1. `test_azure_backend_create_sandbox_happy_path` — monkeypatch `ComputeManagementClient`, assert VM/NIC/IP/disk creation order, assert resource log written.
2. `test_azure_backend_destroy_deletes_in_correct_order` — assert RG cascade delete is called when RG was ephemeral; piecemeal otherwise.
3. `test_azure_backend_destroy_is_asyncio_shielded` — cancel the task mid-destroy, assert delete still completes.
4. `test_azure_backend_terminal_state_aborts_boot_wait` — VM enters `Failed`, assert `SandboxRuntimeError(backend_unavailable)`.
5. `test_azure_backend_owned_instance_allowlist` — destroy a VM created by a different `AzureBackend` instance → refuses.
6. `test_azure_backend_atexit_cleanup_registered` — assert `atexit.register` was called.
7. `test_ensure_azure_available_missing_creds_raises` — no `REPROLAB_AZURE_SUBSCRIPTION_ID` → typed error.
8. `test_config_default_sandbox_accepts_azure` — `Settings(default_sandbox="azure")` constructs.
9. `test_cli_sandbox_choice_includes_azure` — argparse parses `--sandbox azure`.
10. `test_app_runs_endpoint_accepts_sandbox_azure` — `POST /runs {"sandbox":"azure"}` routes to AzureBackend (with the SDK monkeypatched).
11. `test_pod_sweeper_sweeps_azure_tagged_resources` — stale `reprolab=true` Azure VMs older than 24h get deleted (mocked).
12. `test_gpu_resolver_emits_azure_sku_when_sandbox_azure` — feed VRAM=24, count=1, expect `Standard_NC*` not `RTX 4090`.

**Live (opt-in, `pytest -m azure_live`):**
- `test_azure_end_to_end_smoke` — provision smallest CPU SKU (`Standard_B1s` — no GPU, ~$0.01/hr), run `echo hello`, destroy. Should complete in <3 min and cost <$0.005.
- Mark with `@pytest.mark.skipif(not os.environ.get("REPROLAB_AZURE_LIVE_TESTS"), reason="costs money")`.

**Regression:** the entire existing RunPod test suite must still pass unchanged. If you touch `pod_sweeper.py` (option 8a), add `test_pod_sweeper_runpod_still_works` to lock in the old behavior.

---

## 12. Dependencies to add

`backend/requirements.txt`:
```
azure-identity>=1.16,<2
azure-mgmt-compute>=30,<32
azure-mgmt-network>=25,<27
azure-mgmt-resource>=23,<25
```

Do **not** add `azure-cli`. We don't shell out.

Do **not** pin to a single point version — Azure SDKs respect semver well enough; constrain to a major-version window.

---

## 13. `scripts/azure_check.sh` — preflight contract

Mirror `scripts/runpod_check.sh` exactly:

```
Usage:
  scripts/azure_check.sh                # preflight only (read-only, free)
  scripts/azure_check.sh --start-vm     # provision smallest CPU SKU, ssh, destroy (~$0.005)

Exit codes:
  0  green
  1  bad usage
  2  required env var missing (AZURE_SUBSCRIPTION_ID, etc.)
  3  Azure auth failed (DefaultAzureCredential.get_token raised)
  4  configured VM_SIZE not available in REGION (capability query)
  5  SSH key missing / wrong permissions / mismatched pair
  6  --start-vm end-to-end smoke failed
```

Hook it into `start.sh` so `REPROLAB_DEFAULT_SANDBOX=azure ./start.sh` runs the preflight before bringing up uvicorn, matching the RunPod branch. Allow `START_SKIP_PREFLIGHT=1` to bypass (already wired for RunPod).

---

## 14. Runbook (`docs/runbooks/2026-05-27-azure-sandbox.md`) must cover

- Cheapest dev configuration ("CPU smoke + GPT-5 root model = ~$0.05/run; full SDAR repro on `Standard_NC6s_v3` = ~$3/run").
- Common errors and what they mean: `QuotaExceeded` (file a quota increase, this is the #1 Azure GPU gotcha), `SkuNotAvailable` (try a different region), `SshConnectionTimeout` (NSG misconfiguration or boot still in progress).
- How to clean up leaked VMs manually: `az vm list --tag reprolab=true -o table` + `az group delete -n reprolab-<run_id>`.
- The auth surface table: which env var goes with which auth mode (service principal vs. `az login` vs. managed identity).
- The CLAUDE.md "RLM auth — two surfaces" section needs an update: Azure is now a *third* dimension (compute sandbox), independent of the Azure OpenAI LLM provider. Make that explicit.

---

## 15. CLAUDE.md updates (project-level)

After implementation, add to `/Volumes/CS_Stuff/openresearch/CLAUDE.md` under "## Sandboxes":

> `--sandbox azure` provisions an Azure VM via the Compute SDK (no `az` CLI dependency). Requires `REPROLAB_AZURE_SUBSCRIPTION_ID` + credentials resolvable by `DefaultAzureCredential`. Defaults to `Standard_NC6s_v3` in `eastus`; override via `REPROLAB_AZURE_VM_SIZE` / `REPROLAB_AZURE_REGION`. Tear-down is best-effort under `asyncio.shield` plus an `atexit` hook — leaked VMs are very expensive, so `REPROLAB_AZURE_DELETE_ON_DESTROY=true` is the default and `scripts/azure_check.sh` includes a `--cleanup-stale` flag. Costs are bounded by the same `REPROLAB_MAX_RUN_GPU_USD` gate as RunPod.

And under "### RLM auth — two surfaces" add a third bullet noting that "Azure OpenAI (LLM root)" and "Azure VM (compute sandbox)" are **independent** — having one configured does not auto-enable the other.

---

## 16. Definition of done (checklist — use this as your pre-PR audit)

- [ ] `pytest tests/` green on the existing RunPod path (no regressions).
- [ ] `pytest tests/services/runtime/test_azure_backend.py` green (unit, no Azure subscription needed).
- [ ] `python -m backend.cli reproduce 2605.15155 --sandbox azure --max-usd 5` runs end-to-end against a real Azure subscription and produces `final_report.json` (manual smoke; document the run id in the PR description).
- [ ] `python -m backend.cli reproduce 2605.15155 --sandbox runpod --max-usd 5` STILL works (regression check).
- [ ] `frontend/` builds clean (`npm run build`) and the lab UI shows `azure` in the sandbox dropdown when `REPROLAB_AZURE_SUBSCRIPTION_ID` is set.
- [ ] `runs/<id>/azure_resource_log.jsonl` and `runs/<id>/azure_provisioning.log` exist after an Azure run.
- [ ] `cost_ledger.jsonl` entries from the Azure run carry `cloud="azure"`.
- [ ] `scripts/azure_check.sh` (preflight only) exits 0 on a properly configured machine, with informative non-zero codes on misconfiguration.
- [ ] `.env.example` has the Azure compute block.
- [ ] CLAUDE.md updated.
- [ ] Runbook written.
- [ ] No `az` CLI shellouts anywhere.
- [ ] All Azure resources from the smoke run are deleted (verify with `az resource list --tag reprolab=true -o table`).

---

## 17. Out of scope (don't do these in this PR)

- Switching the default `REPROLAB_DEFAULT_SANDBOX` from `runpod` to `azure`. Keep RunPod the default until Azure has bake time.
- Multi-cloud orchestration (run half on RunPod, half on Azure). One sandbox per run, as today.
- Spot VMs / low-priority VMs. Stick to regular on-demand for v1 — spot adds eviction handling complexity.
- Managed Identity-based VM-to-VM auth. Use SSH keys, same as RunPod.
- Azure ML Jobs / Azure Batch / AKS. Plain VMs only.
- A unified "cloud provider" abstraction. Two cloud backends do not justify a base class; three would be the threshold.

---

## 18. Implementation order (recommended; not enforced)

1. Read `brev_backend.py` end-to-end (it's the closest analog).
2. Write the `.env.example` block + `backend/config.py` Settings fields. Confirm `Settings()` loads with the new fields present and absent.
3. Write the `AzureBackend` skeleton with all five ABC methods raising `NotImplementedError`. Wire it into `__init__.py`, `execution.py`, `cli.py`, `app.py`. Confirm `--sandbox azure` parses and reaches the backend's `create_sandbox` before failing.
4. Implement `create_sandbox` + `destroy` with the SDK. Get unit tests passing.
5. Implement `exec` + `copy_in` + `copy_out` via `asyncssh` (lift wholesale from `brev_backend.py`).
6. Implement `probe_alive` + `soft_recover`.
7. Wire `gpu_resolver.py` + `gpu_catalog.py` for Azure SKUs.
8. Wire `pod_sweeper.py` (option 8a or 8b — pick one and document).
9. Write `scripts/azure_check.sh`.
10. Write the unit tests, then the live smoke test.
11. Real Azure smoke run on SDAR (2605.15155, smallest-two scope per CLAUDE.md).
12. Runbook + CLAUDE.md update.
13. PR with the §16 checklist filled in.

---

## 19. Useful one-liners while debugging

```bash
# Confirm DefaultAzureCredential works:
python -c "from azure.identity import DefaultAzureCredential; DefaultAzureCredential().get_token('https://management.azure.com/.default'); print('ok')"

# List your subscriptions:
python -c "from azure.identity import DefaultAzureCredential; from azure.mgmt.subscription import SubscriptionClient; [print(s.subscription_id, s.display_name) for s in SubscriptionClient(DefaultAzureCredential()).subscriptions.list()]"

# Find available GPU SKUs in a region:
python -c "import os; from azure.identity import DefaultAzureCredential; from azure.mgmt.compute import ComputeManagementClient; c=ComputeManagementClient(DefaultAzureCredential(), os.environ['REPROLAB_AZURE_SUBSCRIPTION_ID']); [print(s.name) for s in c.resource_skus.list() if 'NC' in (s.name or '') and any(l.location==os.environ['REPROLAB_AZURE_REGION'] for l in (s.locations or []) ) ]"

# Find leaked reprolab resources:
az resource list --tag reprolab=true -o table

# Nuke everything tagged reprolab=true (DESTRUCTIVE):
az group list --tag reprolab=true --query "[].name" -o tsv | xargs -n1 -I{} az group delete -n {} --yes --no-wait
```

---

## 20. Questions the implementer should ask before writing code

(Surface these in chat before starting — don't guess.)

1. Does the user have an Azure subscription ready, or should the prompt include "fail fast and ask the user to run `az login` first"?
2. Quota: does the chosen region/SKU need a quota increase request first? GPU SKUs almost always do on a fresh subscription.
3. Networking: is the SSH inbound rule allowed to be `0.0.0.0/0` (matches RunPod's posture), or does corp policy require a CIDR restriction?
4. Resource group strategy: is the ephemeral-per-run RG acceptable, or does the subscription require a preexisting RG?
5. Should artifacts also land in an Azure Storage account (durable across VM deletion) in addition to the local `runs/<id>/artifacts/` sync? Probably not for v1, but the runbook should call it out as a v2 follow-up.

---

*This prompt was generated by a prior planning session. The structure mirrors `docs/superpowers/specs/2026-05-26-gepa-phase0-audit.md` and follows project convention for spec docs.*
