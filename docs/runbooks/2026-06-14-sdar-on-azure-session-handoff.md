# SDAR-on-Azure — session handoff

> **Purpose:** everything a fresh session needs to drive the first real SDAR GPU
> run on Azure. Self-contained; read this first, open the linked docs only when a
> step needs their detail.
> **Date:** 2026-06-14 · **Branch:** `feat/azure-bicep-canonical-aoai-hardening`

---

## 0. TL;DR / status

The setup is **built, reviewed, and committed**. It is **NOT yet run** — blocked on
**3 admin-gated actions only an AIONIC subscription admin can do** (§4). Once those
land, the run is four scripts: bootstrap → preflight → run → monitor (§5). Profile:
**local reasoning loop (claude-oauth) + GPU training cells on Azure AKS, blob-only,
1× A100-80GB, smallest-two, capped ~$15-25.**

## 1. Locked decisions — do NOT re-litigate

| Axis | Decision | Why |
|---|---|---|
| Topology | **Local orchestrator + Azure GPU cells** | Reasoning loop runs on WSL; only training cells become AKS Jobs. Avoids the missing orchestrator image, KeyVault, CSI, in-cluster pod. |
| Reasoning auth | **claude-oauth root + Sonnet-OAuth executor** | The only profile validated to climb SDAR's rubric. OAuth can't run in AKS pods — fine, it's local. (AOAI `--model azure` has **zero deployments**, so it has no target anyway.) |
| Compute | **1× A100-80GB** (`azure_a100_80` = `Standard_NC24ads_A100_v4`, $3.67/hr), scale-to-zero, maxNodes=1 | 80 GB is the floor for Qwen2.5-3B full-param GRPO RL + vLLM rollout; 24 GB would force LoRA and break fidelity. |
| Scope | Qwen3-1.7B + Qwen2.5-3B-Instruct, seed 42, ~32-task slice/env, GRPO + SDAR; **7B honest-omitted** | Cost-bounded faithful subset. Caps: 6h wall-clock / `--max-usd 30` / `--max-run-gpu-usd 25`. |

## 2. What's already committed — do NOT redo

Commit on `feat/azure-bicep-canonical-aoai-hardening` (HEAD), author `lolout1`:
- **Backend:** `backend/agents/rlm/k8s_job_cell_runner.py` — cache volume is now
  `emptyDir` when `azure_files_cache_enabled=false`/no share (was an unconditional
  PVC mount that hung every Pod `Pending`); PVC claimName aligned to the Helm name
  `reprolab-cache` (cross-layer bug fixed). `backend/config.py` — new
  `azure_files_cache_enabled` (default True). Guard test
  `tests/agents/rlm/test_aks_cache_volume.py` (5 cases, incl. a cross-layer drift pin).
- **Helm:** `pvc-cache.yaml` + `storageclass-files.yaml` gated behind
  `storage.filesCache.enabled` (`values.yaml` default true).
- **Scripts:** `scripts/azure_sdar_{preflight,bootstrap_cluster,run,monitor}.sh`.
- **Docs:** spec `docs/superpowers/specs/2026-06-14-sdar-on-azure-run-design.md`,
  operator runbook `docs/runbooks/2026-06-14-sdar-on-azure-run.md`, plan
  `docs/superpowers/plans/2026-06-14-sdar-on-azure-run.md`.
- **Verified:** 138 backend tests green, ruff clean, `helm lint`/`bicep build` pass,
  scripts syntax-checked, Codex-reviewed (its findings are all folded in).

## 3. Architecture (6 lines)

`backend.cli reproduce 2605.15155 --sandbox azure --model claude-oauth` runs the RLM
root + Sonnet sub-agents **locally**. On `run_experiment`, the **cells route**
(`code/cells.json` + `code/train_cell.py`) makes `AksJobBackend` +
`k8s_job_cell_runner.run_matrix` submit **one K8s GPU Job per cell** to `sciart-aks`
(ns `reprolab`, SA `reprolab-sa` via workload identity, nodeSelector
`reprolab/sku=azure_a100_80`, `nvidia.com/gpu=1`). Code up / `metrics.json` back via
Blob (`reprolab-artifacts`). HF cache → ephemeral `emptyDir`. No Files/KeyVault/CSI.

## 4. BLOCKED ON — 3 hard admin gates (operator/admin, not codeable)

Start the first two now; they have lead time.

1. **GPU quota** (support ticket): `Standard NCADS_A100_v4 Family vCPUs` ≥ 24 in
   **westus3** (currently 0/0 → no GPU job schedules). Portal → Subscriptions → AIONIC
   → Usage + quotas → filter `NCADS_A100_v4` → Request increase.
2. **AKS RBAC** (a subscription admin must run; you can't self-assign): grant your
   objectId `Azure Kubernetes Service RBAC Cluster Admin` on `sciart-aks` — exact
   `az role assignment create` in runbook §2. Needed for kubectl/helm/Job submission
   (cluster has `disableLocalAccounts` + Entra RBAC → also `kubelogin`).
3. **Redeploy** (cluster torn down 06-13): the proven `az stack group create`
   (`deployKeyVault=false deployStorageKeyOperatorRole=false`) — runbook §2 / the
   06-13 deploy handoff §3.

**Dropped by the blob-only design** (do NOT chase these): Storage-Account-Key-Operator
role + `Microsoft.KeyVault` provider registration — only for the persistent Files
cache / Stream E orchestrator, neither of which this path uses.

## 5. Run it (once gates clear)

```bash
# one-time, in order (full command text: runbook §3)
az login --use-device-code
az aks get-credentials -g rg-sciartgen-external -n sciart-aks
kubelogin convert-kubeconfig -l azurecli            # az aks install-cli if missing
scripts/azure_wire_env.sh rg-sciartgen-external openresearch-l1 .env.azure
scripts/azure_build_cell_image.sh sciartacr         # then add printed OPENRESEARCH_AZURE_BASE_IMAGE= to .env.azure
scripts/azure_sdar_bootstrap_cluster.sh             # namespace + cell SA + NVIDIA plugin (orchestrator+Files OFF)

# run
scripts/azure_sdar_preflight.sh                     # MUST be GREEN — refuses to launch otherwise
scripts/azure_sdar_run.sh                           # the capped launcher (preflight runs again inside)
scripts/azure_sdar_monitor.sh                        # in a 2nd terminal; auto-detects latest run

# retry only failed cells (cheap, reuses ok cells):
scripts/azure_sdar_run.sh .env.azure --resume-cells
```

The launcher pins: `--sandbox azure --model claude-oauth --paper-hint 2605.15155
--scope-spec '{"models":["Qwen3-1.7B","Qwen2.5-3B-Instruct"],"seeds":[42]}'
--force-single-gpu` + the caps, with `OPENRESEARCH_AZURE_GPU_SKUS='["azure_a100_80"]'`
and `OPENRESEARCH_AZURE_FILES_CACHE_ENABLED=false` (blob-only).

## 6. Critical gotchas / invariants (these bite)

- **claude-oauth auth:** a no-credit `ANTHROPIC_API_KEY` silently beats OAuth →
  `400 credit balance too low` at the first Sonnet call. The run script `unset`s it;
  if launching by hand, `unset ANTHROPIC_API_KEY OPENAI_API_KEY` first. A working
  `claude --print ping` proves the subscription is live.
- **Shell wins over .env:** a stale exported `OPENAI_API_KEY`/`ANTHROPIC_API_KEY`
  shadows `.env`. Keep them unset for this profile.
- **Blob-only is load-bearing:** the cell cache is `emptyDir` (weights re-download per
  cell — fine for smallest-two). Do NOT "fix" this by provisioning Files unless you
  also get the Key-Operator grant.
- **Cells route gate:** the per-GPU-cell scheduler only engages if the agent emits
  BOTH `code/cells.json` and `code/train_cell.py`; else it falls back to the monolithic
  path. Don't drop one without the other.
- **SDAR fidelity (the rubric inspects literally):** `g_t = σ(β·Δ_t)`, stop-gradient on
  the gate, `λ=0.1`, `β=10`, GRPO loss ADDED to gated OPSD; **real** Qwen weights. Model
  ids are canonical: **`Qwen/Qwen3-1.7B` (NO `-Instruct`)**, `Qwen/Qwen2.5-3B-Instruct`
  (keeps it). All encoded in `docs/papers/2605.15155.yaml` + `paper_hints.py` and
  enforced by the preflight invariant guard.
- **Don't expect a rubric pass first try.** Success for *this* milestone = a real GPU
  Job runs a cell and `metrics.json` returns via Blob. The rubric is the paper's own
  difficulty; iterate with `--resume-cells`.

## 7. If it fails — triage (full matrix: runbook §7)

| Symptom | Cause → fix |
|---|---|
| Pods stuck `Pending` | GPU quota 0 (§4.1) or pool missing `reprolab/sku=azure_a100_80` label → quota ticket / redeploy |
| kubectl/helm `forbidden`/401 | AKS RBAC not granted (§4.2) → admin assigns, re-run `kubelogin convert-kubeconfig -l azurecli` |
| Blob 403 on artifact I/O | workload MI `sciart-workload-mi` lacks `Storage Blob Data Contributor` on `sciartgenreprolab` |
| `partial` verdict + `SDK success-with-no-text` | claude-oauth/SDK auth, **not** Docker → `claude login`, unset `ANTHROPIC_API_KEY` |
| `ErrImagePull` | `OPENRESEARCH_AZURE_BASE_IMAGE` tag not in ACR → rebuild via `azure_build_cell_image.sh`, confirm AcrPull |

## 8. Process constraints (for the next agent)

- **Opus plans + reviews every diff; Sonnet executes; Codex reviews.** Verify diffs,
  not summaries. Use the `/implement` skill for impl, never Codex for writing code.
- **Commit infrequently** at milestones; no Conventional-Commits prefix; **no
  `Co-Authored-By`/AI trailer**; author = local default `lolout1 / appradhann@gmail.com`
  (never `-c user.email=sww35`).
- **Push only to `origin` (openresearch) on explicit request**; `deepinvent` mirror only
  on explicit request; never `replix`. This branch is currently **unpushed**.
- **Secrets → Key Vault, never git/params/`.env`-in-git.** `.env.azure` (resource names,
  no keys) is gitignored. Resource/sub identifiers are not secrets; live keys are.

## 9. Canonical docs

- Operator command reference: `docs/runbooks/2026-06-14-sdar-on-azure-run.md`
- Design spec: `docs/superpowers/specs/2026-06-14-sdar-on-azure-run-design.md`
- Implementation plan: `docs/superpowers/plans/2026-06-14-sdar-on-azure-run.md`
- IaC deploy + redeploy: `docs/runbooks/2026-06-13-azure-aionic-deploy-handoff.md`
- Paper detail + debug history: `docs/runbooks/2026-05-23-sdar-baseline-handoff.md`
- Azure backend design: `docs/superpowers/specs/2026-06-03-azure-aks-gpu-backend-design.md`
- Backend entry points: `backend/services/runtime/aks_job_backend.py`,
  `backend/agents/rlm/k8s_job_cell_runner.py`, `backend/agents/rlm/executor.py`.

## 10. Next-session prompt (paste to start fresh)

```
Continue the SDAR-on-Azure run on branch feat/azure-bicep-canonical-aoai-hardening.
Read docs/runbooks/2026-06-14-sdar-on-azure-session-handoff.md first — it has the
full context. The setup is committed and Codex-reviewed; we are blocked on 3 admin
gates (GPU quota, AKS RBAC, redeploy — handoff §4). When I tell you those are done,
drive the run via the scripts in §5 (bootstrap → preflight → run → monitor) and
babysit it. Profile: local claude-oauth orchestrator + Azure AKS GPU cells, blob-only,
smallest-two Qwen3-1.7B + Qwen2.5-3B on 1x A100-80GB, capped ~$15-25. Honor the
process constraints in §8 (Opus plans+reviews, Sonnet executes, commit infrequently,
push only on request, no AI trailer).
```
