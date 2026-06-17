# SDAR on Azure (local-orchestrator) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `--sandbox azure` path actually run the SDAR paper blob-only with a local RLM orchestrator, and ship the operator setup/run/monitor scripts + handoff runbook.

**Architecture:** Two small backend/Helm edits remove the hard Azure-Files dependency (emptyDir cache fallback), then four bash scripts (preflight → bootstrap → run → monitor) drive a capped smallest-two SDAR run on 1× A100-80GB AKS GPU Jobs. Reasoning loop (claude-oauth root + Sonnet executor) stays local; only training cells dispatch to AKS.

**Tech Stack:** Python 3.12 (pydantic Settings, pytest), Kubernetes Python client (lazy-imported), Azure CLI / kubectl / helm / jq, Bicep + Helm IaC.

**Spec:** `docs/superpowers/specs/2026-06-14-sdar-on-azure-run-design.md`.

> **COMMIT POLICY (overrides the skill's per-task commit):** Do NOT `git commit` per task. Make ONE milestone commit at the very end (Task 10) bundling the spec, this plan, the backend/Helm edits, the four scripts, and the runbook. Author = local default (lolout1 / appradhann@gmail.com). No Conventional-Commits prefix, no Co-Authored-By trailer. Run all tasks on the current branch `feat/azure-bicep-canonical-aoai-hardening`.

---

## File structure

| File | Action | Responsibility |
|---|---|---|
| `backend/config.py` | Modify (near line 361) | Add `azure_files_cache_enabled: bool` setting. |
| `backend/agents/rlm/k8s_job_cell_runner.py` | Modify (`_build_job_manifest` ~422-470/555-560, call site ~1025) | Conditional cache volume: PVC when enabled+share set, else `emptyDir`. |
| `tests/agents/rlm/test_aks_cache_volume.py` | Create | Guard test: PVC vs emptyDir vs identical mount path. |
| `infra/azure/helm/templates/pvc-cache.yaml` | Modify | Gate behind `.Values.storage.filesCache.enabled`. |
| `infra/azure/helm/templates/storageclass-files.yaml` | Modify | Same gate. |
| `infra/azure/helm/values.yaml` | Modify | Add `storage.filesCache.enabled: true` default. |
| `scripts/azure_sdar_preflight.sh` | Create | Read-only green/red prerequisite validator. |
| `scripts/azure_sdar_bootstrap_cluster.sh` | Create | One-time Helm scaffold (orchestrator+Files off). |
| `scripts/azure_sdar_run.sh` | Create | The pinned, capped SDAR launcher. |
| `scripts/azure_sdar_monitor.sh` | Create | Live babysit loop. |
| `docs/runbooks/2026-06-14-sdar-on-azure-run.md` | Create | Operator handoff playbook. |

---

## Task 1: Add the `azure_files_cache_enabled` setting

**Files:**
- Modify: `backend/config.py` (immediately after the `azure_files_share` field; the Azure block is ~line 240-300)

- [ ] **Step 1: Find the anchor.** Run: `grep -n "azure_files_share" backend/config.py` — note the line. Insert the new field right after it.

- [ ] **Step 2: Add the field.** Insert (match the file's `Field(...)` style and indentation):

```python
    azure_files_cache_enabled: bool = Field(
        default=True,
        description=(
            "When True (default), AKS cell Jobs mount the Azure Files PVC "
            "(<namespace>-files-pvc) at azure_cache_mount_path as the HF/pip "
            "cache. When False — or when azure_files_share is empty — the cell "
            "Job uses an ephemeral emptyDir instead, so no Files share / "
            "Storage Account Key Operator grant is required (blob-only path)."
        ),
    )
```

- [ ] **Step 3: Verify it loads.** Run:
```bash
.venv/bin/python -c "from backend.config import get_settings; print(get_settings().azure_files_cache_enabled)"
```
Expected: `True`

---

## Task 2: Conditional cache volume in the AKS Job manifest (TDD)

**Files:**
- Test: `tests/agents/rlm/test_aks_cache_volume.py` (create)
- Modify: `backend/agents/rlm/k8s_job_cell_runner.py`

- [ ] **Step 1: Write the failing test.** Create `tests/agents/rlm/test_aks_cache_volume.py`:

```python
"""Guard: the AKS cell Job cache volume is a PVC when Files is enabled, an
emptyDir fallback otherwise — with an identical mount path either way.

Pins the blob-only invariant from the 2026-06-14 SDAR-on-Azure spec §4.1: a
cell Pod must never hard-depend on a provisioned Azure Files PVC.
"""
from __future__ import annotations

from backend.agents.rlm.k8s_job_cell_runner import _build_job_manifest

_MOUNT = "/mnt/reprolab-cache"


def _manifest(*, files_cache_enabled: bool, files_share: str):
    return _build_job_manifest(
        job_name="job-x",
        namespace="reprolab",
        service_account="reprolab-sa",
        node_pool_name="gpua100",
        base_image="acr.azurecr.io/reprolab-cell:abc123",
        storage_account="sacct",
        blob_container="reprolab-artifacts",
        files_share=files_share,
        cell_id="qwen3_1_7b__sdar__alfworld__s42",
        cell_params_json="{}",
        output_blob_prefix="runs/x/cells/c/out",
        code_blob_prefix="runs/x/code",
        active_deadline_seconds=3600,
        max_oom_retries=2,
        fingerprint=None,
        now_iso=None,
        cache_mount_path=_MOUNT,
        files_cache_enabled=files_cache_enabled,
    )


def _volume(manifest):
    return manifest["spec"]["template"]["spec"]["volumes"][0]


def _mount_path(manifest):
    vm = manifest["spec"]["template"]["spec"]["containers"][0]["volumeMounts"][0]
    return vm["mountPath"]


def test_pvc_volume_when_files_cache_enabled():
    m = _manifest(files_cache_enabled=True, files_share="reprolab-cache")
    vol = _volume(m)
    assert "persistentVolumeClaim" in vol
    assert vol["persistentVolumeClaim"]["claimName"] == "reprolab-files-pvc"
    assert "emptyDir" not in vol
    assert _mount_path(m) == _MOUNT


def test_emptydir_when_files_cache_disabled():
    m = _manifest(files_cache_enabled=False, files_share="reprolab-cache")
    vol = _volume(m)
    assert "emptyDir" in vol
    assert "persistentVolumeClaim" not in vol
    assert _mount_path(m) == _MOUNT


def test_emptydir_when_share_empty_even_if_enabled():
    m = _manifest(files_cache_enabled=True, files_share="")
    vol = _volume(m)
    assert "emptyDir" in vol
    assert "persistentVolumeClaim" not in vol
    assert _mount_path(m) == _MOUNT


def test_volume_name_is_stable_across_both_modes():
    on = _volume(_manifest(files_cache_enabled=True, files_share="reprolab-cache"))
    off = _volume(_manifest(files_cache_enabled=False, files_share="reprolab-cache"))
    assert on["name"] == off["name"] == "reprolab-cache"
```

- [ ] **Step 2: Run it — expect FAIL.** Run:
```bash
.venv/bin/python -m pytest tests/agents/rlm/test_aks_cache_volume.py -v
```
Expected: FAIL — `_build_job_manifest() got an unexpected keyword argument 'files_cache_enabled'` (and the emptyDir tests fail because the current code is unconditional PVC).

- [ ] **Step 3: Add the `files_cache_enabled` param.** In `backend/agents/rlm/k8s_job_cell_runner.py`, add to the `_build_job_manifest` signature (after `default_sku: str = "azure_a100_80",`, before the closing `) -> dict[str, Any]:`):

```python
    # Spec 2026-06-14 §4.1: blob-only fallback. When False (or files_share is
    # empty) the cache volume is an ephemeral emptyDir, not the Azure Files PVC.
    files_cache_enabled: bool = True,
```

- [ ] **Step 4: Add the volume-spec helper.** Insert this module-level helper just above `def _build_job_manifest(` (around line 421):

```python
def _cache_volume_spec(
    *, namespace: str, files_share: str, files_cache_enabled: bool
) -> dict[str, Any]:
    """Return the K8s volume dict named 'reprolab-cache'.

    PVC (<namespace>-files-pvc) when the Azure Files cache is enabled AND a
    share name is configured; otherwise an ephemeral emptyDir so the cell Pod
    never blocks on a missing PVC (spec 2026-06-14 §4.1, blob-only path).
    """
    if files_cache_enabled and files_share.strip():
        return {
            "name": "reprolab-cache",
            "persistentVolumeClaim": {"claimName": f"{namespace}-files-pvc"},
        }
    return {"name": "reprolab-cache", "emptyDir": {}}
```

- [ ] **Step 5: Use the helper.** In `_build_job_manifest`, delete the line `pvc_name = f"{namespace}-files-pvc"` (~line 470) and replace the hardcoded `volumes` list in `pod_template["spec"]` (~lines 555-560):

```python
            "volumes": [
                {
                    "name": "reprolab-cache",
                    "persistentVolumeClaim": {"claimName": pvc_name},
                }
            ],
```
with:
```python
            "volumes": [
                _cache_volume_spec(
                    namespace=namespace,
                    files_share=files_share,
                    files_cache_enabled=files_cache_enabled,
                )
            ],
```

- [ ] **Step 6: Run the test — expect PASS.** Run:
```bash
.venv/bin/python -m pytest tests/agents/rlm/test_aks_cache_volume.py -v
```
Expected: 4 passed.

- [ ] **Step 7: Thread the setting from the call site.** In the `_build_job_manifest(...)` call (~line 1017-1053), add — right after the `cache_mount_path=...` line (~1038):

```python
            files_cache_enabled=bool(_setting("azure_files_cache_enabled", True)),
```

- [ ] **Step 8: Regression-check the existing Azure tests.** Run:
```bash
.venv/bin/python -m pytest tests/agents/rlm/test_k8s_job_cell_runner.py tests/agents/rlm/test_azure_cell_integration.py tests/agents/rlm/test_azure_env_contract.py -q
```
Expected: all pass (no behavior change for the default enabled+share path).

---

## Task 3: Gate the Helm Files PVC/StorageClass

**Files:**
- Modify: `infra/azure/helm/values.yaml`
- Modify: `infra/azure/helm/templates/pvc-cache.yaml`
- Modify: `infra/azure/helm/templates/storageclass-files.yaml`

- [ ] **Step 1: Read the current storage values block.** Run:
```bash
grep -n "storage:" -A 12 infra/azure/helm/values.yaml
```
Note the exact indentation under `storage:`.

- [ ] **Step 2: Add the toggle to values.yaml.** Under the existing `storage:` map, add (matching indentation):

```yaml
  # Spec 2026-06-14 §4.2: set false for the blob-only path (no Azure Files
  # PVC/StorageClass, so no Storage Account Key Operator grant needed).
  filesCache:
    enabled: true
```

- [ ] **Step 3: Wrap pvc-cache.yaml.** Open `infra/azure/helm/templates/pvc-cache.yaml`. Wrap the ENTIRE document body in a guard. First line of the file becomes:
```yaml
{{- if .Values.storage.filesCache.enabled }}
```
and append as the last line:
```yaml
{{- end }}
```

- [ ] **Step 4: Wrap storageclass-files.yaml.** Identically wrap `infra/azure/helm/templates/storageclass-files.yaml`:
```yaml
{{- if .Values.storage.filesCache.enabled }}
```
… existing content …
```yaml
{{- end }}
```

- [ ] **Step 5: Verify both render states.** Run:
```bash
helm template reprolab-aks infra/azure/helm --set storage.filesCache.enabled=true  | grep -c "kind: PersistentVolumeClaim"
helm template reprolab-aks infra/azure/helm --set storage.filesCache.enabled=false | grep -c "kind: PersistentVolumeClaim"
```
Expected: first prints `1` (PVC present), second prints `0` (PVC gated out). If `helm` is not installed, defer this to Task 9's lint step and note it.

- [ ] **Step 6: Lint the chart.** Run: `helm lint infra/azure/helm`
Expected: `1 chart(s) linted, 0 chart(s) failed`.

---

## Task 4: `scripts/azure_sdar_preflight.sh`

**Files:**
- Create: `scripts/azure_sdar_preflight.sh`

- [ ] **Step 1: Write the script.** Create the file with exactly:

```bash
#!/usr/bin/env bash
# azure_sdar_preflight.sh — read-only green/red gate for an SDAR-on-Azure run.
# Exits non-zero if ANY hard check fails. Run before azure_sdar_run.sh.
#
# Usage: scripts/azure_sdar_preflight.sh [env-file=.env.azure]
set -uo pipefail

ENV_FILE="${1:-.env.azure}"
[[ -f "$ENV_FILE" ]] && set -a && . "$ENV_FILE" && set +a

RG="${OPENRESEARCH_AZURE_RESOURCE_GROUP:-rg-sciartgen-external}"
CLUSTER="${OPENRESEARCH_AZURE_AKS_CLUSTER:-sciart-aks}"
NS="${OPENRESEARCH_AZURE_NAMESPACE:-reprolab}"
SA="${OPENRESEARCH_AZURE_SERVICE_ACCOUNT:-reprolab-sa}"
SKU="${OPENRESEARCH_AZURE_GPU_SKUS:-azure_a100_80}"; SKU="${SKU//[\[\]\"]/}"; SKU="${SKU%%,*}"
REGION="${OPENRESEARCH_AZURE_REGION:-westus3}"
QUOTA_FAMILY="Standard NCADS_A100_v4 Family vCPUs"
QUOTA_MIN=24

fails=0; warns=0
ok()   { printf '  \033[32m[OK]\033[0m   %s\n' "$1"; }
bad()  { printf '  \033[31m[FAIL]\033[0m %s\n' "$1"; fails=$((fails+1)); }
warn() { printf '  \033[33m[WARN]\033[0m %s\n' "$1"; warns=$((warns+1)); }
have() { command -v "$1" >/dev/null 2>&1; }

echo "== SDAR-on-Azure preflight (rg=$RG cluster=$CLUSTER ns=$NS sku=$SKU) =="

# 1. tooling
for t in az kubectl jq; do have "$t" && ok "tool: $t" || bad "tool missing: $t"; done
have kubelogin || warn "kubelogin missing (needed for disableLocalAccounts clusters): az aks install-cli"

# 2. az login + subscription
if az account show >/dev/null 2>&1; then ok "az logged in (sub $(az account show --query id -o tsv))"
else bad "not logged in — run: az login --use-device-code"; fi

# 3. cluster reachable
if kubectl get nodes >/dev/null 2>&1; then ok "kubectl reaches $CLUSTER ($(kubectl get nodes --no-headers 2>/dev/null | wc -l) node(s))"
else bad "kubectl cannot reach cluster — run: az aks get-credentials -g $RG -n $CLUSTER && kubelogin convert-kubeconfig -l azurecli"; fi

# 4. GPU quota (the #1 blocker; was 0/0)
LIMIT=$(az vm list-usage -l "$REGION" --query "[?localName=='$QUOTA_FAMILY'].limit | [0]" -o tsv 2>/dev/null || echo "")
if [[ -z "$LIMIT" ]]; then warn "could not read $QUOTA_FAMILY quota (need Reader on the sub)"
elif (( ${LIMIT%.*} >= QUOTA_MIN )); then ok "GPU quota: $QUOTA_FAMILY limit=$LIMIT (>= $QUOTA_MIN)"
else bad "GPU quota too low: $QUOTA_FAMILY limit=$LIMIT (< $QUOTA_MIN) — open a quota ticket"; fi

# 5. operator AKS-RBAC (can submit Jobs)
if kubectl auth can-i create jobs -n "$NS" >/dev/null 2>&1; then ok "RBAC: can create jobs in $NS"
else bad "RBAC: cannot create jobs in $NS — assign 'Azure Kubernetes Service RBAC Cluster Admin' on $CLUSTER"; fi

# 6. namespace + SA (Helm bootstrap ran)
if kubectl get ns "$NS" >/dev/null 2>&1; then ok "namespace $NS exists"
else bad "namespace $NS missing — run scripts/azure_sdar_bootstrap_cluster.sh"; fi
if kubectl get sa "$SA" -n "$NS" >/dev/null 2>&1; then
  cid=$(kubectl get sa "$SA" -n "$NS" -o jsonpath='{.metadata.annotations.azure\.workload\.identity/client-id}' 2>/dev/null)
  [[ -n "$cid" ]] && ok "SA $SA has workload-identity client-id" || bad "SA $SA missing workload-identity annotation"
else bad "SA $SA missing in $NS — run scripts/azure_sdar_bootstrap_cluster.sh"; fi

# 7. GPU node-pool label exists on a (scale-to-zero) pool
if az aks nodepool list -g "$RG" --cluster-name "$CLUSTER" --query "[?nodeLabels.\"reprolab/sku\"=='$SKU'] | [0].name" -o tsv 2>/dev/null | grep -q .; then
  ok "GPU pool with label reprolab/sku=$SKU present"
else bad "no node pool labelled reprolab/sku=$SKU — redeploy the Bicep stack"; fi

# 8. ACR + pinned cell image tag
if [[ -n "${OPENRESEARCH_AZURE_BASE_IMAGE:-}" ]]; then
  acr="${OPENRESEARCH_AZURE_ACR_LOGIN_SERVER%%.*}"; repo="${OPENRESEARCH_AZURE_BASE_IMAGE#*/}"; repo="${repo%%:*}"; tag="${OPENRESEARCH_AZURE_BASE_IMAGE##*:}"
  if az acr repository show-tags -n "$acr" --repository "$repo" -o tsv 2>/dev/null | grep -qx "$tag"; then ok "cell image present: $OPENRESEARCH_AZURE_BASE_IMAGE"
  else bad "cell image tag missing in ACR: $OPENRESEARCH_AZURE_BASE_IMAGE — run scripts/azure_build_cell_image.sh"; fi
else bad "OPENRESEARCH_AZURE_BASE_IMAGE unset — build + pin the cell image"; fi

# 9. storage account + container
if [[ -n "${OPENRESEARCH_AZURE_STORAGE_ACCOUNT:-}" ]] && az storage account show -n "$OPENRESEARCH_AZURE_STORAGE_ACCOUNT" -g "$RG" >/dev/null 2>&1; then
  ok "storage account $OPENRESEARCH_AZURE_STORAGE_ACCOUNT exists"
else bad "storage account ${OPENRESEARCH_AZURE_STORAGE_ACCOUNT:-<unset>} not found"; fi

# 10. python deps in the venv
if .venv/bin/python -c "import kubernetes, azure.identity, azure.storage.blob" 2>/dev/null; then ok "python deps: kubernetes, azure-identity, azure-storage-blob"
else bad "python deps missing — pip install -r backend/requirements.txt into .venv"; fi

# 11. claude-oauth root surface alive
if have claude && claude --print "ping" >/dev/null 2>&1; then ok "claude-oauth root surface responds"
else warn "claude --print ping failed — confirm 'claude login' (root model is claude-oauth)"; fi

echo "== preflight: $fails fail(s), $warns warn(s) =="
(( fails == 0 )) || { echo "RED — resolve the [FAIL] items above before running."; exit 1; }
echo "GREEN — safe to run scripts/azure_sdar_run.sh"
```

- [ ] **Step 2: chmod + syntax-check.** Run:
```bash
chmod +x scripts/azure_sdar_preflight.sh && bash -n scripts/azure_sdar_preflight.sh && echo OK
```
Expected: `OK`. If `shellcheck` is installed: `shellcheck scripts/azure_sdar_preflight.sh` (warnings acceptable; no errors).

- [ ] **Step 3: Dry behavior check (no Azure needed).** Run: `scripts/azure_sdar_preflight.sh /nonexistent.env`
Expected: it runs to completion and exits non-zero with `[FAIL]`/`[WARN]` lines (proves the checklist + exit logic work without crashing).

---

## Task 5: `scripts/azure_sdar_bootstrap_cluster.sh`

**Files:**
- Create: `scripts/azure_sdar_bootstrap_cluster.sh`

- [ ] **Step 1: Confirm the Helm value keys.** Run:
```bash
grep -nE "^namespace:|^serviceAccountName:|workloadIdentity:|clientId:|^storage:|orchestrator:" infra/azure/helm/values.yaml
```
Confirm the keys used in the `--set` flags below match (adjust paths if the chart differs — e.g. `serviceAccount.name` vs `serviceAccountName`).

- [ ] **Step 2: Write the script.** Create `scripts/azure_sdar_bootstrap_cluster.sh`:

```bash
#!/usr/bin/env bash
# azure_sdar_bootstrap_cluster.sh — one-time, idempotent cluster scaffold for
# the local-orchestrator SDAR path: namespace + cell SA (workload identity) +
# NVIDIA device plugin + RBAC + quota. NO orchestrator, NO KeyVault, NO Files.
# Requires the AKS RBAC Cluster Admin grant.
#
# Usage: scripts/azure_sdar_bootstrap_cluster.sh [env-file=.env.azure]
set -euo pipefail

ENV_FILE="${1:-.env.azure}"
[[ -f "$ENV_FILE" ]] && set -a && . "$ENV_FILE" && set +a

RG="${OPENRESEARCH_AZURE_RESOURCE_GROUP:-rg-sciartgen-external}"
WI_MI_NAME="${OPENRESEARCH_AZURE_WORKLOAD_MI_NAME:-sciart-workload-mi}"
RELEASE="reprolab-aks"
CHART="infra/azure/helm"

command -v helm >/dev/null || { echo "helm not installed: https://helm.sh/docs/intro/install/"; exit 1; }

echo "Resolving workload-identity client id from MI '$WI_MI_NAME' in $RG ..."
WI_CLIENT_ID="$(az identity show -g "$RG" -n "$WI_MI_NAME" --query clientId -o tsv)"
[[ -n "$WI_CLIENT_ID" ]] || { echo "could not resolve clientId for $WI_MI_NAME"; exit 1; }

echo "helm upgrade --install $RELEASE (orchestrator OFF, filesCache OFF) ..."
helm upgrade --install "$RELEASE" "$CHART" \
  --set orchestrator.enabled=false \
  --set storage.filesCache.enabled=false \
  --set workloadIdentity.clientId="$WI_CLIENT_ID" \
  --wait --timeout 5m

echo "Done. Scaffolded namespace + cell SA + NVIDIA device plugin + RBAC + quota."
echo "Verify: kubectl get sa,ns,daemonset -A | grep -E 'reprolab|nvidia'"
```

- [ ] **Step 3: chmod + syntax-check.** Run:
```bash
chmod +x scripts/azure_sdar_bootstrap_cluster.sh && bash -n scripts/azure_sdar_bootstrap_cluster.sh && echo OK
```
Expected: `OK`.

---

## Task 6: `scripts/azure_sdar_run.sh`

**Files:**
- Create: `scripts/azure_sdar_run.sh`

- [ ] **Step 1: Write the script.** Create `scripts/azure_sdar_run.sh`:

```bash
#!/usr/bin/env bash
# azure_sdar_run.sh — launch the capped smallest-two SDAR reproduction with GPU
# cells on Azure AKS and the reasoning loop local (claude-oauth + Sonnet).
# Runs the preflight first and aborts if it is RED.
#
# Usage: scripts/azure_sdar_run.sh [env-file=.env.azure]
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_FILE="${1:-.env.azure}"
[[ -f "$ENV_FILE" ]] && set -a && . "$ENV_FILE" && set +a

# OAuth on both reasoning surfaces — clear any shadowing API keys (CLAUDE.md gotcha).
unset OPENAI_API_KEY ANTHROPIC_API_KEY OPENRESEARCH_FORCE_SANDBOX 2>/dev/null || true

echo "== preflight =="
scripts/azure_sdar_preflight.sh "$ENV_FILE" || { echo "Preflight RED — aborting."; exit 1; }

export OPENRESEARCH_AZURE_GPU_SKUS='["azure_a100_80"]'
export OPENRESEARCH_AZURE_FILES_CACHE_ENABLED=false   # blob-only (spec §4.1)
export OPENRESEARCH_ACCELERATOR=off                   # accel needs OpenAI; off
export OPENRESEARCH_DYNAMIC_GPU=true
export OPENRESEARCH_BASELINE_EXTRA_GUIDANCE="SCOPE: reproduce SDAR using ONLY the two smallest model variants — Qwen3-1.7B and Qwen2.5-3B-Instruct. Honest-omit Qwen2.5-7B (declare in metrics.json['omitted']). Use real pretrained HF weights (no surrogate) and the real ALFWorld + Search-QA + WebShop datasets, but evaluate on a small representative slice (~32 tasks/env) to keep wall-clock practical on a single A100-80GB. Run the GRPO baseline and the proposed SDAR; the ablations (OPSD, Skill-SD, GRPO+OPSD, RLSD) may be code-present and declared. Report per_model results for 1.7B and 3B."

LOG="runs/sdar-azure-$(date +%s).log"
echo "== launch (log → $LOG) =="
.venv/bin/python -m backend.cli reproduce 2605.15155 \
  --mode rlm --sandbox azure --model claude-oauth --paper-hint 2605.15155 \
  --scope-spec '{"models":["Qwen3-1.7B","Qwen2.5-3B-Instruct"],"seeds":[42]}' \
  --force-single-gpu \
  --max-wall-clock 21600 --max-usd 30 --max-run-gpu-usd 25 --max-gpu-usd-per-hour 4 \
  2>&1 | tee "$LOG"

echo "== done. Monitor with: scripts/azure_sdar_monitor.sh =="
```

- [ ] **Step 2: chmod + syntax-check.** Run:
```bash
chmod +x scripts/azure_sdar_run.sh && bash -n scripts/azure_sdar_run.sh && echo OK
```
Expected: `OK`.

- [ ] **Step 3: Confirm every CLI flag is real.** Run:
```bash
.venv/bin/python -m backend.cli reproduce --help 2>&1 | grep -E -- "--scope-spec|--paper-hint|--force-single-gpu|--max-run-gpu-usd|--max-gpu-usd-per-hour|--max-wall-clock|--max-usd|--sandbox|--model"
```
Expected: every flag in the launch command appears. (They were verified present in `backend/cli.py` while authoring this plan.)

---

## Task 7: `scripts/azure_sdar_monitor.sh`

**Files:**
- Create: `scripts/azure_sdar_monitor.sh`

- [ ] **Step 1: Write the script.** Create `scripts/azure_sdar_monitor.sh`:

```bash
#!/usr/bin/env bash
# azure_sdar_monitor.sh — live babysit loop for an SDAR-on-Azure run.
# Usage: scripts/azure_sdar_monitor.sh [project_id] [interval_s=30]
set -uo pipefail
cd "$(dirname "$0")/.."

NS="${OPENRESEARCH_AZURE_NAMESPACE:-reprolab}"
INTERVAL="${2:-30}"
PID="${1:-}"
if [[ -z "$PID" ]]; then
  PID="$(ls -dt runs/prj_* 2>/dev/null | head -1 | xargs -r basename)"
fi
[[ -n "$PID" ]] || { echo "no project id; pass one explicitly"; exit 1; }
RUN="runs/$PID"
echo "Monitoring $PID (ns=$NS, every ${INTERVAL}s; Ctrl-C to stop)"

while true; do
  clear 2>/dev/null || true
  echo "=== $PID @ $(date -u '+%H:%M:%SZ') ==="
  echo "--- AKS jobs/pods ($NS) ---"
  kubectl get jobs,pods -n "$NS" 2>/dev/null | head -20 || echo "(kubectl unavailable)"
  echo "--- rubric / iteration ---"
  if [[ -f "$RUN/dashboard_events.jsonl" ]]; then
    grep -E '"(rubric_score|repl_iteration)"' "$RUN/dashboard_events.jsonl" 2>/dev/null | tail -1 \
      | jq -rc '{type, score:(.score // .overall_score // .data.score), iter:(.iteration // .data.iteration)}' 2>/dev/null || echo "(no rubric event yet)"
  fi
  echo "--- cost (USD) ---"
  if [[ -f "$RUN/cost_ledger.jsonl" ]]; then
    jq -s 'map(.usd // .cost_usd // 0) | add' "$RUN/cost_ledger.jsonl" 2>/dev/null || echo "?"
  fi
  echo "--- warnings ---"
  [[ -f "$RUN/dashboard_events.jsonl" ]] && grep -E '"(run_warning|gpu_escalated|capacity_exhausted)"' "$RUN/dashboard_events.jsonl" 2>/dev/null | tail -3
  echo "--- exec tail ---"
  [[ -f "$RUN/code/.exec_live.log" ]] && tail -4 "$RUN/code/.exec_live.log" 2>/dev/null
  echo
  echo "(recover failed cells:  scripts/azure_sdar_run.sh  then add --resume-cells)"
  sleep "$INTERVAL"
done
```

- [ ] **Step 2: chmod + syntax-check.** Run:
```bash
chmod +x scripts/azure_sdar_monitor.sh && bash -n scripts/azure_sdar_monitor.sh && echo OK
```
Expected: `OK`.

---

## Task 8: `docs/runbooks/2026-06-14-sdar-on-azure-run.md`

**Files:**
- Create: `docs/runbooks/2026-06-14-sdar-on-azure-run.md`

- [ ] **Step 1: Write the runbook** with these exact sections and commands (fill each from the spec + the 06-13 deploy handoff; every command must be copy-pasteable):

  1. **Header** — date 2026-06-14, status, pairs-with the spec + 06-13 handoff + SDAR baseline handoff.
  2. **§1 What this does** — local orchestrator (claude-oauth + Sonnet) + GPU cells on AKS, blob-only, smallest-two, 1× A100-80GB, capped ~$15-25.
  3. **§2 Admin-gated prerequisites (your end)** — the table from spec §6: GPU quota ticket (`Standard NCADS_A100_v4 Family vCPUs ≥ 24`, westus3); AKS RBAC (`az role assignment create --role "Azure Kubernetes Service RBAC Cluster Admin" --assignee <objectId> --scope .../managedClusters/sciart-aks`); redeploy `az stack group create` (deployKeyVault=false deployStorageKeyOperatorRole=false — copy from 06-13 handoff §3). State Key-Operator + KeyVault are NOT needed (blob-only).
  4. **§3 One-time setup** — `az login` + `az aks get-credentials -g rg-sciartgen-external -n sciart-aks` + `kubelogin convert-kubeconfig -l azurecli`; `scripts/azure_wire_env.sh rg-sciartgen-external openresearch-l1 .env.azure`; `scripts/azure_build_cell_image.sh sciartacr` then add the printed `OPENRESEARCH_AZURE_BASE_IMAGE=` to `.env.azure`; `scripts/azure_sdar_bootstrap_cluster.sh`.
  5. **§4 Run** — `scripts/azure_sdar_preflight.sh` (must be GREEN) → `scripts/azure_sdar_run.sh` → in another shell `scripts/azure_sdar_monitor.sh`.
  6. **§5 Cost table** — A100-80 @ $3.67/hr; serial cells; the $25 run cap and $4/hr per-GPU cap; scale-to-zero idle = $0.
  7. **§6 Teardown** — `az stack group delete --name openresearch-l1 -g rg-sciartgen-external --action-on-unmanage deleteResources --yes` (stops all spend; keeps RG + AOAI + tfstate).
  8. **§7 Troubleshooting matrix** — Pods stuck Pending → GPU quota or missing `reprolab/sku` label; `forbidden`/auth errors → AKS RBAC not granted; Blob 403 → workload MI lacks Storage Blob Data Contributor; `partial` verdict with `SDK success-with-no-text` → claude-oauth/SDK auth, NOT Docker; cell image pull error → tag not in ACR.

- [ ] **Step 2: Link it** from `docs/runbooks/2026-06-13-azure-aionic-deploy-handoff.md` (add a "Next: run SDAR → 2026-06-14-sdar-on-azure-run.md" line near its §5) and verify the file renders (no broken local links): `grep -n "2026-06-14-sdar-on-azure-run" docs/runbooks/*.md`.

---

## Task 9: Full verification sweep

- [ ] **Step 1: Backend tests + lint.** Run:
```bash
.venv/bin/python -m pytest tests/agents/rlm/test_aks_cache_volume.py tests/agents/rlm/test_k8s_job_cell_runner.py tests/agents/rlm/test_azure_cell_integration.py tests/agents/rlm/test_azure_env_contract.py tests/rlm/test_azure_wiring.py -q
uvx ruff@0.15.16 check backend/agents/rlm/k8s_job_cell_runner.py backend/config.py tests/agents/rlm/test_aks_cache_volume.py
```
Expected: all tests pass; ruff clean.

- [ ] **Step 2: Helm + Bicep static checks.** Run:
```bash
helm lint infra/azure/helm
helm template reprolab-aks infra/azure/helm --set storage.filesCache.enabled=false | grep -c "PersistentVolumeClaim"   # expect 0
az bicep build --file infra/azure/bicep/infra.bicep --stdout >/dev/null && echo "bicep OK"
```
Expected: chart lints; PVC count 0 when gated off; bicep builds. (Skip a tool only if it is not installed; note which were skipped.)

- [ ] **Step 3: Shell syntax.** Run:
```bash
for s in scripts/azure_sdar_*.sh; do bash -n "$s" && echo "$s OK"; done
```
Expected: all four `OK`.

---

## Task 10: Milestone commit (single)

- [ ] **Step 1: Stage the deliverables only** (no `runs/` noise):
```bash
git add docs/superpowers/specs/2026-06-14-sdar-on-azure-run-design.md \
        docs/superpowers/plans/2026-06-14-sdar-on-azure-run.md \
        docs/runbooks/2026-06-14-sdar-on-azure-run.md \
        docs/runbooks/2026-06-13-azure-aionic-deploy-handoff.md \
        backend/config.py backend/agents/rlm/k8s_job_cell_runner.py \
        tests/agents/rlm/test_aks_cache_volume.py \
        infra/azure/helm/values.yaml \
        infra/azure/helm/templates/pvc-cache.yaml \
        infra/azure/helm/templates/storageclass-files.yaml \
        scripts/azure_sdar_preflight.sh scripts/azure_sdar_bootstrap_cluster.sh \
        scripts/azure_sdar_run.sh scripts/azure_sdar_monitor.sh
```

- [ ] **Step 2: Commit** (descriptive headline, no prefix, no AI trailer):
```bash
git commit -m "Add the blob-only SDAR-on-Azure run path: emptyDir cell-cache fallback + operator scripts and runbook

Make --sandbox azure runnable end-to-end without a provisioned Azure Files
share: the AKS cell Job now falls back to an ephemeral emptyDir cache when
azure_files_cache_enabled is false or no share is set (was an unconditional
PVC mount that hung every Pod Pending), and the Helm Files PVC/StorageClass
are gated behind storage.filesCache.enabled. Ships preflight/bootstrap/run/
monitor scripts and the operator handoff runbook for a capped smallest-two
(Qwen3-1.7B + Qwen2.5-3B) SDAR run on 1x A100-80GB with a local claude-oauth
orchestrator. Drops the Storage Account Key Operator grant from the critical
path; hard admin gates reduce to GPU quota + AKS RBAC + redeploy."
```

- [ ] **Step 3: Confirm clean.** Run: `git status --short` — expect no staged deliverables remaining; only pre-existing `runs/` noise untracked.

---

## Self-review (completed by author)

- **Spec coverage:** §4.1 → Tasks 1-2; §4.2 → Task 3; §5.1-5.4 → Tasks 4-7; §5.5 → Task 8; §6 → Task 8 §2; §7 → Task 9. All spec sections mapped.
- **Placeholders:** none — full code/commands in every step; Task 8 lists exact section content + commands.
- **Type/name consistency:** `azure_files_cache_enabled` (config + `_setting` key + manifest param), `files_cache_enabled` (manifest/helper param), `_cache_volume_spec` (defined Task 2 Step 4, used Step 5), `storage.filesCache.enabled` (values + both templates + bootstrap + monitor) — consistent across tasks.
- **Verification reality:** shell tasks use `bash -n` + flag-existence checks (can't unit-test bash); backend task is full TDD; Helm/Bicep use lint/template/build. Live-Azure checks are deferred to the operator (preflight), as the spec requires.
