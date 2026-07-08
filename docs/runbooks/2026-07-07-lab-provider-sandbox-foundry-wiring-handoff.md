<!-- doc-meta: status=current; last-verified=2026-07-07 -->
# Handoff — Wire the `/lab` LLM-provider + Sub-agent-auth pickers (add Foundry) + GCP/Azure sandbox, and the GCP A100 run setup

> **Date:** 2026-07-07 · **Status:** IN PROGRESS (recon done + plan confirmed; implementation NOT started).
> Self-contained: a fresh session should be able to finish the feature and know the environment state from this doc alone.

## 0. Status header (5-second read)

**Done + pushed to `deepinvent/main`** (SHAs in §3): PR-A skills, PR-B autonomous-UI merge, dynamic-Foundry fix, GPU-count+cost-cap+full-stack-`start.sh`+`gcp_ready.sh`. **Environment ready** for a real GCP A100 run (all local `.env` + cloud infra done, §4). **In progress / NEXT:** implement the `/lab` picker feature — add **Foundry** to "LLM provider" + "Sub-agent auth", add **GCP/Azure** to "Sandbox", and **wire the two pickers** (they are inert today). Recon complete, plan user-confirmed; no code written yet.

---

## 1. THE ACTIVE TASK — `/lab` provider/sandbox/Foundry wiring (implement this)

### 1.1 What the user asked (verbatim, this session)
- *"add Foundry as an option for LLM provider + sub-agent auth … also gcp should be an option as well as azure for sandbox fix this robust optimal solution this is in localhost:3000/lab in the ui/ux ensure it integrates seamlessly with backend robust optimal /implement"*
- Confirmed via question: **"Yes wire them (option 1) and they should actually be the default now"** — i.e. make the pickers functional (not cosmetic) AND make Foundry the **pre-selected default** when Foundry creds exist.
- *"you can prepare handoff doc for this as well if needed to preserve context"* → this doc.

### 1.2 THE decisive recon finding (do not re-derive)
`root_provider` and `subagent_auth` on `StartRunRequest` are **INERT** — forwarded from the UI into the request but **never consumed** to set the root model or executor auth. So the "LLM provider" and "Sub-agent auth" radios in `/lab` are **decorative today**. The task is (a) wire them, (b) add Foundry, (c) add GCP/Azure sandbox options.

- `sandbox` is **already backend-ready**: `SandboxMode = Literal["auto","docker","local","runpod","azure","gcp"]` (`backend/services/events/live_runs.py:43`) and `DemoSandboxMode` (frontend `demo-run-types.ts:7`) already include `azure`+`gcp`. The UI just omits them from the radio list. **No backend change needed for sandbox** (gcp → `GkeJobBackend`, azure → Azure AKS).

### 1.3 How the root model actually reaches the child (verified)
- `StartRunRequest.model: ModelChoice = "sonnet"` where **`ModelChoice = str`** (`live_runs.py:45,182`) — a free string, so it can hold any root token (`opus-foundry`, `claude-oauth`, `gpt-5`, `azure`, …).
- `_python_script` (`live_runs.py:1775-1796`): `request.model=="sonnet"` → `_model_id` = anthropic/openai default model; `=="opus"` → reasoning model; **else `_model_id = request.model` (pass-through, line 1796)**. So `opus-foundry`/`sonnet-foundry`/`claude-oauth` pass straight through.
- `_model_id` → `common["model"]` (`live_runs.py:~1824`) → serialized as `config = json.loads(...)` (a JSON dict, ~line 1863) → `cmd_reproduce(...)` in the child. **The root model reaches the child via the `config` dict, NOT an env var.** The child's `resolve_root_model` (`backend/agents/rlm/models.py`) resolves that token.
- **Autonomous mode already does exactly this:** `apply_autonomous_profile_override` (`live_runs.py:498-519`) does `model_copy(update={"sandbox":"gcp","model":"opus-foundry","run_spec":...})`.

### 1.4 Override chain + the correct injection point (verified)
`_start_python_run` (`live_runs.py:846-852`), in order:
1. `apply_sandbox_override(request, settings.force_sandbox)` (deployment force)
2. `apply_provider_override(request, settings.force_llm_provider)` (deployment force)
3. `apply_autonomous_profile_override(request)` — **LAST, wins** (per-request opt-in)

**Inject a new `apply_picker_overrides(request)` BEFORE step 3** (i.e. between line 848 and 852), so the picker drives the run by default but the autonomous override still wins. Model each existing `apply_*_override` (they `return request.model_copy(update={...})`, `live_runs.py:467-519`).

### 1.5 Executor / sub-agent auth (the second wire)
- `OPENRESEARCH_ROLE_MODELS` / `llm_auth_strategy` are **NOT set from the request anywhere in `live_runs.py`** (grep is empty) → `subagent_auth` is fully inert. The child resolves sub-roles from the `OPENRESEARCH_ROLE_MODELS` env (`role_models.resolve_role_models`).
- Wire `subagent_auth` by adding an env in **`_subprocess_env`** (`live_runs.py:535-610`, the block that already sets `OPENRESEARCH_GPU_COUNT` etc. around line 595-600): `subagent_auth=="foundry"` → `env["OPENRESEARCH_ROLE_MODELS"]='{"executor":"sonnet-foundry"}'` (or merge with any existing); `anthropic_oauth`/`anthropic_api` → set the auth strategy (`OPENRESEARCH_LLM_AUTH_STRATEGY=oauth_only|api_only` — **VERIFY the exact env name in the child**; see §7 open item).
- **`_subprocess_env` env vs the `config` dict:** root model rides the `config` dict (§1.3); executor auth rides env. Both reach the same child process.

### 1.6 The token matrix (root_provider [+ model] → request.model root token)
Confirmed catalog tokens (from `resolve_root_model` / `models.ROOT_MODELS`, exercised extensively this session):
`opus-foundry` (Opus 4.8 via Foundry), `sonnet-foundry` (Sonnet 5 via Foundry), `claude-oauth` (Sonnet via OAuth), `claude` (Opus via ANTHROPIC_API_KEY), `gpt-5` (OPENAI), `azure`/`azure-gpt-4o` (Azure OpenAI), `qwen3-coder-featherless` (Featherless), `azure-foundry`/`grok` (OpenAI-compat Foundry).

**Mapping to implement (user-confirmed Foundry semantics = opus root / sonnet executor, matching the autonomous profile):**
| `root_provider` | + `model` | → `request.model` token |
|---|---|---|
| `foundry` | `opus` | `opus-foundry` |
| `foundry` | `sonnet` (default) | `sonnet-foundry` |
| `anthropic_oauth` | any | `claude-oauth` |
| `anthropic_api` | `opus`/`sonnet` | leave as-is (`opus`/`sonnet`; existing `_python_script` maps via ANTHROPIC_API_KEY) |
| `openai_api` | any | `gpt-5` |
| `azure_openai` | any | `azure` |
| `featherless` | any | `qwen3-coder-featherless` |

`subagent_auth`: `foundry` → executor `sonnet-foundry`; `anthropic_oauth` → OAuth; `anthropic_api` → API key.

> The Explore agent **a86109454f92ec013** ("Trace root-model + executor auth resolution") was launched to CONFIRM this matrix + the exact `OPENRESEARCH_LLM_AUTH_STRATEGY` env name + the run-spec-vs-picker precedence. It may still be running — check `/workflows` or re-launch the same prompt (it's in this conversation) before finalizing the matrix. Everything above is verified except the auth-strategy env name and the run-spec-pins-model precedence.

### 1.7 Files to touch (exact)
**Frontend** (`frontend/src/`):
- `lib/demo/demo-run-types.ts`: `RootProvider` (lines 293-301) add `| "foundry"`; `SubagentAuth` (303) add `| "foundry"`. `DemoSandboxMode` (7) already has `gcp`/`azure` — no change.
- `components/lab/upload-view.tsx`: `SANDBOX_OPTIONS` (line 15) add `{value:"gcp",…}` + `{value:"azure",…}` (mirror the runpod entry at line 17; hints: gcp = "GKE A100 pool, scale-to-zero"; azure = "Azure AKS (needs AKS infra)"). Provider label map (lines 33-34) add `foundry: "Foundry (Opus/Sonnet)"`. Optional: a Foundry cred hint mirroring `showAzureFields`/`showAnthropicKey` (lines 184-186). The provider/subagent radios render from `authStatus` (lines 405-475) — adding to types + auth-status surfaces them automatically.
- `components/lab/lab-shell.tsx`: default/persisted-choice logic (lines 172-179) already falls back to `authStatus.defaults.root_provider` — so making Foundry the default is a **backend** change (§factory.py below), not here. Verify the saved-choice guard handles the new value.
- `components/lab/upload-view.test.tsx` + `hooks/use-run.test.ts`: extend for the new options (the `gpuCount` tests added this session are the pattern).

**Backend**:
- `backend/agents/runtime/factory.py` `aggregate_auth_status()` (line 454, body ~470-520): add `"foundry"` to the `providers` dict (`available: has_foundry_credentials()`, detail "AZURE_FOUNDRY_API_KEY set/missing"), add `"foundry"` to `subagent_auth` dict, and make `defaults.root_provider`/`defaults.subagent_auth` = `foundry` when foundry creds present (prepend to the existing `default_root`/`default_subagent` if-chains at ~470-486). `has_foundry_credentials` is already imported/available in factory.py (used by `make_runtime` foundry branch ~535).
- `backend/services/events/live_runs.py`: NEW `apply_picker_overrides(request)` near the other `apply_*_override` (467-519); call it in `_start_python_run` before line 852. It sets `request.model` from `(root_provider, model)` per §1.6, and returns via `model_copy`. Add the `subagent_auth`→`OPENRESEARCH_ROLE_MODELS`/auth-strategy env in `_subprocess_env` (~595-600).
- `backend/app.py`: `root_provider`/`subagent_auth` are already forwarded (lines 649-650 arxiv, 709-710 upload; `StartArxivRunRequest` 1195-1196). No change unless a new field is needed (it isn't).

### 1.8 Precedence rules (design decision — do not reverse)
- Autonomous override (`model=opus-foundry`, `sandbox=gcp`) **always wins** (applied last) — DO NOT let the picker override it.
- Only set `request.model` from the picker when a picker value is present; when `root_provider` is None/unset, leave `request.model` untouched → **byte-identical to today** (off-state invariant, mandatory per CLAUDE.md flag rules).
- Verify: does a `--run-spec` that pins the root model override `request.model`, or vice versa? (Open item §7 — the Explore agent is checking.)

### 1.9 Verify (acceptance for the feature)
```bash
cd /home/abheekp/openresearch
# backend
.venv/bin/python -m pytest tests/services/events/ tests/routes/test_advanced_field_forwarding.py -q   # + new picker-wiring tests
.venv/bin/python -c "from backend.config import Settings; from backend.agents.runtime.factory import aggregate_auth_status; print(aggregate_auth_status()['providers'].get('foundry'), aggregate_auth_status()['defaults'])"
# frontend (node 20 via nvm — system node v21 is out of range)
source ~/.nvm/nvm.sh && nvm use 20.20.2
cd frontend && npx tsc --noEmit && npm test -- <touched test files>
# off-state: root_provider=None → request.model unchanged (byte-identical)
```

---

## 2. Task inventory

### 2.1 DONE + PUSHED to `deepinvent/main` (verified)
| Commit | What | Verified |
|---|---|---|
| `006812a4` | **PR-A merge** — relevance-gated skill selection + SDAR execute-mode prereqs (reconcile → main) | 48 tests, merge-tree conflict-free |
| `1e687552` | **PR-B merge** — autonomous-upload UI + spec_validator (feat/autonomous-upload-ui → main); 9-file Anthropic-Foundry conflict cluster resolved toward main's canonical layer | 174 tests, primitive count 19 |
| `b9c6df99` | **Dynamic Foundry** — `_build_llm_client` gains an `anthropic-foundry` branch (root primitive client → Foundry via `AnthropicMessagesClient`); `normalize_anthropic_base_url` hardened (scheme-less/case/query) | 293 tests, tsc+Next build, ruff |
| `fcba19ca` | **GPU-count end-to-end + per-run cost cap + full-stack `start.sh` + `scripts/gcp_ready.sh`** | 135 backend + 482 frontend, tsc+build |

Feature detail for `fcba19ca`: user-selectable GPU count 1–8 (`/lab` field + `/abs` picker → `OPENRESEARCH_GPU_COUNT` → `settings.gpu_count` → `gpu_resolver.gpu_count_override` pins `GpuPlan.gpu_count`, relaxing `force_single_gpu`, capping to SKU physical → k8s `nvidia.com/gpu`); `GkeJobBackend._enforce_run_budget` (was stored-but-never-checked); `GpuStatusStrip` on the session view; `gcp_ready.sh` preflight.

### 2.2 DONE — environment/infra (local, NOT committed; see §4)
- Primary dir `/home/abheekp/openresearch` switched from `reconcile` → `main` (`fcba19ca`), fully provisioned (`.venv`, `.env`, `node_modules` synced). Reconcile external-runs WIP preserved in **`git stash@{0}`**.
- Scratch worktree `/home/abheekp/openresearch-main` **removed**.
- GCP A100 run path configured + verified READY (`gcp_ready.sh` all-green).

### 2.3 IN PROGRESS — the `/lab` feature (§1). Recon done, plan confirmed, **0 lines written**.

### 2.4 Planned / offered but NOT started
- **`scripts/gcp_provision_a100.sh`** + a GCP-A100-setup runbook — offered to capture the (currently local-only) A100 pool provisioning + `.env` GCP config so it's reproducible. User hasn't said yes/no. (The pool-create commands are in §4.3.)
- **SDAR A/B (~$30)** — long-standing, prepped, never launched (VM `sdar-2model-a`, `scripts/sdar_phase1_foundry.sh`). Separate spend.
- **A100-80 quota bump 4→8** — offered (needed for `gcp_a100_80x8`). Not filed.
- **repo-first / kimik2 branches** — genuinely unmerged; user said **"leave both for now"** (repo-first is ~10 real commits on a 1425-commit divergent base → needs rebase, not merge; kimik2 = obsolete May spike).

---

## 3. Required context (coordinates — do NOT re-derive)

- **Remote:** push ONLY to `deepinvent` = `git@github.com:Deepinvent/scientific_article_generator.git`. `origin` (armaanamatya/openresearch) is stale — ignore the "ahead of origin by N" message. `deepinvent/main` = `fcba19ca`.
- **Branch/worktree state:** primary `/home/abheekp/openresearch` on `main`. Parked worktrees: `-autonomous-ui` (846d4e06, merged), `-gke`, `-lab-k2`, `-lab-ui-fix`, `-repo-first` (48a86368, unmerged), `-tree-search`. Stash: `stash@{0}` = "external-runs WIP + loose files parked before main switch (2026-07-06)".
- **Author/commit rules:** author `lolout1 <appradhann@gmail.com>` (local git config — never `-c user.email=…`); **no `Co-Authored-By`/Claude/AI attribution**; no Conventional-Commit prefixes; descriptive present-tense headline; commit at milestones; push only when asked.
- **Test env:** backend `.venv/bin/python -m pytest`. Frontend needs **nvm node 20.20.2** (`source ~/.nvm/nvm.sh && nvm use 20.20.2`) — **system node v21.7.3 is OUT of Next's `≥20.19<21 || ≥22.12` range**; `.nvmrc` pins `22` which is NOT installed. Ruff: `uvx ruff@0.15.16 check <files>`.
- **Run the app:** `./start.sh` (now full-stack: backend :8000 + frontend :3000, nvm-node-select, watchdog; `START_BACKEND_ONLY=1`/`START_FRONTEND_ONLY=1`/`START_SKIP_PREFLIGHT=1`). Verified it boots (backend `/openapi.json` 200, frontend `/` 200).
- **Foundry endpoint memory:** the Anthropic-Foundry base MUST end at `…/anthropic` (SDK appends `/v1/messages`); `…/anthropic/v1` double-`/v1`s → 404. Funded Opus 4.8 + Sonnet 5, `x-api-key=AZURE_FOUNDRY_API_KEY`.

---

## 4. GCP A100 run setup — current state (LOCAL, not committed)

### 4.1 `.env` additions (gitignored — holds secrets + this machine's setup)
```
OPENRESEARCH_DEFAULT_SANDBOX=gcp
OPENRESEARCH_GCP_PROJECT=deepinvent-ext-ut
OPENRESEARCH_GCP_GCS_BUCKET=deepinvent-ext-ut-sdar-runs
OPENRESEARCH_GCP_BASE_IMAGE=us-central1-docker.pkg.dev/deepinvent-ext-ut/reprolab/gke-cell-base:v1
OPENRESEARCH_GCP_GPU_SKUS=["gcp_a100_80","gcp_a100_80x2","gcp_a100_80x4"]
```
Also present (funded): `AZURE_FOUNDRY_API_KEY`, `AZURE_FOUNDRY_ENDPOINT`. Missing/empty: runpod key (fine). `OPENRESEARCH_GCP_GPU_SKUS` **must be JSON-array form** — single-string errors (`pydantic_settings` parse error).

### 4.2 Cloud resources (deepinvent-ext-ut)
- GKE cluster **`openresearch-gpu`**, zone **`us-central1-a`**, RUNNING.
- Node pools: `default-pool` (e2-small), `gpu-l4` (g2-standard-8 — **note: had NO `reprolab/sku` label, so it never scheduled Jobs**), and the 3 A100 pools created this session: **`a100-80-1g`/`-2g`/`-4g`** (a2-ultragpu-1g/2g/4g, `nvidia-a100-80gb` ×1/2/4, **scale-to-zero, $0 idle, 0 quota when idle**, labels `reprolab/sku=gcp_a100_80`/`_80x2`/`_80x4`, taint `nvidia.com/gpu=present:NoSchedule`, driver DEFAULT).
- Bucket `gs://deepinvent-ext-ut-sdar-runs`, base image `…/reprolab/gke-cell-base:v1`.
- **Quota (us-central1):** A100-80GB=**4** (so `gcp_a100_80x8` NOT possible w/o bump), A100-40GB=8, A2_CPUS=48. Usage 0.
- `gke-gcloud-auth-plugin` installed at **`~/.local/bin/gke-gcloud-auth-plugin`** (on default PATH) — see gotcha §6.
- kube-context now `gke_deepinvent-ext-ut_us-central1-a_openresearch-gpu` (was Azure `sciart-aks`).

### 4.3 Reproduce the A100 pools (the exact commands run)
```bash
for spec in a100-80-1g:a2-ultragpu-1g:1:gcp_a100_80 a100-80-2g:a2-ultragpu-2g:2:gcp_a100_80x2 a100-80-4g:a2-ultragpu-4g:4:gcp_a100_80x4; do
  IFS=: read name mt cnt sku <<<"$spec"
  gcloud container node-pools create "$name" --cluster openresearch-gpu --zone us-central1-a --project deepinvent-ext-ut \
    --machine-type "$mt" --accelerator "type=nvidia-a100-80gb,count=$cnt,gpu-driver-version=default" \
    --num-nodes 0 --enable-autoscaling --min-nodes 0 --max-nodes 1 \
    --node-labels "reprolab/sku=$sku" --node-taints "nvidia.com/gpu=present:NoSchedule" \
    --disk-size 100 --disk-type pd-balanced --image-type COS_CONTAINERD --async
done
```
Readiness check: `bash scripts/gcp_ready.sh` (fixes kube-context + reports all gates). Was READY at last run.

---

## 5. Decisions & why (don't reverse)
- **Merge resolved toward main's canonical Anthropic-Foundry layer** (dropped the UI branch's parallel Foundry copies) — verified UI's `foundry_anthropic.py` normalizer was buggy (`/anthropic/v1` → 404) and its patch lacked the fail-closed cred guard. UI's copies were superseded, not fixes.
- **A100 node pools pre-created as scale-to-zero** (not per-run) — pools cost $0 at 0 nodes and creating per-run adds ~3–5 min latency for no savings; the GPU *node* is what's created on-demand (GKE autoscale on Job dispatch). User questioned this; confirmed correct.
- **GPU_SKUS set to L4 first, then A100** — L4 matched the only existing pool; switched to A100 after the user said "we need A100s like previously." The agentic GPU determination (`resolve_gpu_requirements`) is CORE/in-main (not a missing merge); it can only pick SKUs that are *provisioned as node pools*, hence the pool ladder.
- **Foundry = opus root / sonnet executor** for the picker mapping — matches the autonomous profile exactly.
- **Wire the pickers as the default behavior** (user: "they should actually be the default now") — the inert-picker state is a pre-existing gap being fixed.

## 6. Gotchas discovered
- `gke-gcloud-auth-plugin` can't `gcloud components install` (snap-managed gcloud) and isn't in apt (no Google Cloud apt repo). **Fix used (no sudo):** download the official `.deb` from `packages.cloud.google.com/apt` (find via the `dists/cloud-sdk/main/binary-amd64/Packages` index), `dpkg-deb -x`, copy the binary to `~/.local/bin/` (on PATH). The backend's kubernetes python client + kubectl find it there.
- `OPENRESEARCH_GCP_GPU_SKUS` requires **JSON-array** env form (`["gcp_a100_80x4"]`); a bare `gcp_a100_80x4` throws a pydantic-settings parse error.
- `gpu-l4` pool had **no `reprolab/sku` label** → Jobs targeting it hang Pending. Always create GPU pools with `--node-labels reprolab/sku=<short_name>`.
- The `openresearch-gpu` cluster's default GPU pool is **L4, not A100**; `gcp_gpu_skus` default is `gcp_a100_80x8` → mismatch would hang Pending until you set the SKU config to match real pools.
- Frontend build/test: **must use nvm node 20** (system v21 breaks it).
- `configs/external_runs.json` is an untracked leftover from the external-runs WIP — harmless, `rm` for a clean tree.

## 7. Open questions / to confirm (Explore agent a86109454f92ec013 is on it)
1. Exact env var name for executor OAuth-vs-API auth strategy (`OPENRESEARCH_LLM_AUTH_STRATEGY`? verify against `role_models`/`grader_transport`).
2. Precedence: does a `--run-spec` that pins `OPENRESEARCH_RLM_ROOT_MODEL` override `request.model` (config-dict), or vice versa? Confirm the picker doesn't silently lose to a run-spec.
3. Whether `subagent_auth` should also gate the *grader/verifier* transports or only the executor (user said "sub-agent auth" broadly).

## 8. Next immediate action
Check the Explore agent **a86109454f92ec013** result (`/workflows`, or re-launch its prompt from the transcript) to lock the token-matrix precedence + the auth-strategy env name (§1.6/§7). THEN implement §1.7 — recommended fan-out: one Sonnet agent for the frontend (types + `SANDBOX_OPTIONS` + labels + tests), one for backend (`factory.py` auth-status Foundry+default + `live_runs.py` `apply_picker_overrides` + `_subprocess_env` executor env + tests). Opus authors the `apply_picker_overrides` matrix (§1.6) + reviews every diff. Verify per §1.9, commit (author lolout1, no trailer), push to `deepinvent/main` on the user's OK.

---

## DURABLE-FACT FLAGS (promote to memory/CLAUDE.md — not just this doc)
- **GCP A100 run infra** (project `deepinvent-ext-ut`, cluster `openresearch-gpu`@us-central1-a, bucket `deepinvent-ext-ut-sdar-runs`, base image `…/reprolab/gke-cell-base:v1`, A100-80 pools 1/2/4 scale-to-zero, quota 4×A100-80 / 8×A100-40, `gke-gcloud-auth-plugin` no-sudo install to `~/.local/bin`). → new `reference`/`project` memory. Extends `[[gcp_a100_capacity_quota]]`, `[[project_gcp_gke_backend]]`, `[[sdar_gcp_run_hardening]]`.
- **`OPENRESEARCH_GCP_GPU_SKUS` must be JSON-array env form; gpu_skus must match real provisioned node pools** (else Pending). → nested `backend/services/runtime/CLAUDE.md`.
- **The `/lab` `root_provider`/`subagent_auth` pickers were inert** until this task wires them — once wired, update `backend/agents/rlm/CLAUDE.md` (or the frontend one) so it's not re-flagged as a bug.
- **Primary working dir is now `main`** (not reconcile); external-runs WIP in `stash@{0}`. → transient, but note if the user asks "where's my external-runs work".
