# Cloud-Native Durable Controller + CPU Cloud Lane — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the durable GKE controller real, safe, and ON-by-default for `sandbox=gcp`, and add a dedicated CPU Job lane so CPU-class papers run on cloud with no laptop — both fail-soft to today's local behavior.

**Architecture:** A stable `fence_epoch` (renew-invariant) is added to `BlobLease` so a controller can fence and reap Jobs without reaping its own after a heartbeat. `_submit_durable_controller` becomes a real, takeover-safe submit (acquire → is_current → submit → ready → reap → record handle) with pre-submit-only local fallback. A deterministic `cpu_class` classifier routes CPU-only cells to a GPU-free K8s Job manifest; all-cells infra failure reruns locally.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic `BaseModel`, Kubernetes `batch/v1` Job dicts (no SDK at import), GCS CAS via `gcs_blob`, pytest (socket-hermetic via `--disable-socket`).

## Global Constraints

- New capability = `feature_flags.env_truthy("FLAG")`, **default-safe**, byte-identical when the operational default is not engaged (non-gcp or explicit opt-out). Copy this convention verbatim.
- **No new primitive.** `PRIMITIVE_REGISTRY` stays **19** (`tests/rlm/test_registry.py`).
- Tests are **socket-hermetic** — never dial out; inject fakes. Full suite: `.venv/bin/python -m pytest tests/ -n auto`. Lint: `uvx ruff@0.15.16 check .`.
- **Metric neutrality:** a CPU cloud cell must issue the identical cell command as its local counterpart; env parity is drill-certified, not asserted here.
- **Never split-brain:** local fallback is reachable only when no remote controller Job is live.
- Subagent guardrails: forbidden git state commands (`commit`/`add`/`amend`/`checkout`/`switch`/`stash`/`reset`/`rebase`/`merge`/`rm`/`clean`); write ONLY the task's allowlisted files; **never edit or delete an existing test** — STOP and report; return a structured summary. The lead verifies the git footprint + re-runs tests before committing.
- After ANY touched file, run the tripwires: `tests/agents/rlm/test_single_verdict_authority_guard.py` and `tests/rlm/test_registry.py`.
- Spec: `docs/superpowers/specs/2026-07-12-cloud-native-durable-and-cpu-lane-design.md`.

---

## File Structure

| File | Responsibility | New/Modify |
|---|---|---|
| `backend/services/runtime/blob_lease.py` | add `fence_epoch` to `LeaseToken`+payload; `reap_stale_fence_epochs` | Modify |
| `backend/agents/rlm/controller_launch.py` | pure CPU-by-construction controller Job manifest builder | New |
| `backend/agents/rlm/cpu_class.py` | deterministic `requires_gpu` classifier | New |
| `backend/services/events/live_runs.py` | `ControllerHandle`, real `_submit_durable_controller`, fallback, default-for-gcp, handle-aware liveness | Modify (hot) |
| `backend/agents/rlm/k8s_job_cell_runner.py` | `accelerator="cpu"` manifest branch + CPU routing + all-infra-fail signal | Modify (hot) |
| `backend/agents/rlm/run_controller.py` | `build_controller_command` → controller entrypoint; `durable_controller_default_for_sandbox` | Modify |
| `backend/agents/rlm/feature_flags.py` | flag readers (if a helper is warranted) | Modify (small) |
| `docs/runbooks/2026-07-12-cpu-lane-and-durable-drill-operator-checklist.md` | operator commands + drill | New doc |

---

## Phase A — Stable fence epoch (everything fences on it)

### Task A1: Add renew-invariant `fence_epoch` to `BlobLease`

**Files:**
- Modify: `backend/services/runtime/blob_lease.py`
- Test: `tests/services/runtime/test_blob_lease.py` (ADD new tests only; do not edit existing ones)

**Interfaces:**
- Produces: `LeaseToken(run_id, generation, owner_id, acquired_epoch, fence_epoch: int)`; `BlobLease.reap_stale_fence_epochs(run_id, token, *, list_jobs: Callable[[str], list[tuple[str,int]]], delete_job: Callable[[str], None]) -> int` where `list_jobs` returns `(job_name, fence_epoch)` and jobs with `fence_epoch < token.fence_epoch` are deleted.
- Consumes: existing `gcs_blob.read_bytes_with_generation` / `upload_bytes(if_generation_match=...)`.

**Fence semantics:** first acquire → `fence_epoch=1`; same-owner reacquire/renew → **preserve**; takeover (different owner, expired) → `old_fence_epoch + 1`.

- [ ] **Step 1: Write failing tests (ADD to the file, new functions only)**

```python
def test_first_acquire_sets_fence_epoch_1(fake_gcs):
    lease = BlobLease(bucket="b", client=fake_gcs)
    tok = lease.acquire("run1", "ownerA", now_epoch=100.0)
    assert tok is not None and tok.fence_epoch == 1

def test_renew_preserves_fence_epoch(fake_gcs):
    lease = BlobLease(bucket="b", client=fake_gcs)
    tok = lease.acquire("run1", "ownerA", now_epoch=100.0)
    tok2 = lease.renew(tok, now_epoch=160.0)
    assert tok2 is not None
    assert tok2.fence_epoch == tok.fence_epoch      # stable across heartbeat
    assert tok2.generation != tok.generation        # CAS version advanced

def test_same_owner_reacquire_preserves_fence_epoch(fake_gcs):
    lease = BlobLease(bucket="b", client=fake_gcs)
    tok = lease.acquire("run1", "ownerA", now_epoch=100.0)
    tok2 = lease.acquire("run1", "ownerA", now_epoch=160.0)  # restart, stable owner_id
    assert tok2.fence_epoch == tok.fence_epoch

def test_takeover_bumps_fence_epoch(fake_gcs):
    lease = BlobLease(bucket="b", client=fake_gcs)
    tok = lease.acquire("run1", "ownerA", now_epoch=100.0)
    # ownerB takes over after TTL (LEASE_TTL_S=180)
    tok_b = lease.acquire("run1", "ownerB", now_epoch=100.0 + LEASE_TTL_S + 1)
    assert tok_b is not None and tok_b.fence_epoch == tok.fence_epoch + 1

def test_reap_stale_fence_epochs_keys_on_fence_not_generation(fake_gcs):
    lease = BlobLease(bucket="b", client=fake_gcs)
    tok = lease.acquire("run1", "ownerA", now_epoch=100.0)
    tok = lease.renew(tok, now_epoch=160.0)   # generation advanced, fence stable
    deleted = []
    jobs = [("job-fe1", 1), ("job-fe2", 2)]   # (name, fence_epoch); tok.fence_epoch==1
    n = lease.reap_stale_fence_epochs(
        "run1", tok, list_jobs=lambda r: jobs, delete_job=deleted.append)
    assert n == 0 and deleted == []           # own current-epoch jobs NOT reaped after renew
```

- [ ] **Step 2: Run to verify they fail** — `.venv/bin/python -m pytest tests/services/runtime/test_blob_lease.py -k fence -v` → FAIL (`fence_epoch` / `reap_stale_fence_epochs` missing).

- [ ] **Step 3: Implement**

Add `fence_epoch: int` to `LeaseToken`. In `_encode_lease`/`_decode_lease` add `"fence_epoch"`. In `acquire`:
```python
# no existing lease
if existing is None:
    new_gen = self._write(..., fence_epoch=1, if_generation_match=0)
    ... return LeaseToken(..., fence_epoch=1)
# existing:
prev_fence = int(record.get("fence_epoch", 1))
if current_owner == owner_id:
    fence_epoch = prev_fence                      # reacquire — preserve
else:  # takeover (expired path only reaches here)
    fence_epoch = prev_fence + 1
new_gen = self._write(..., fence_epoch=fence_epoch, if_generation_match=current_gen)
... return LeaseToken(..., fence_epoch=fence_epoch)
```
In `renew`, thread `fence_epoch=token.fence_epoch` into `_write` and `dataclasses.replace(token, generation=new_gen)` (fence_epoch already preserved). `_write` gains a `fence_epoch: int` kwarg and includes it in the payload. Add:
```python
def reap_stale_fence_epochs(self, run_id, token, *, list_jobs, delete_job) -> int:
    try:
        jobs = list_jobs(run_id)
    except Exception as exc:
        logger.warning("reap_stale_fence_epochs(%s): list_jobs failed: %s", run_id, exc)
        return 0
    deleted = 0
    for job_name, fe in jobs:
        if fe >= token.fence_epoch:
            continue
        try:
            delete_job(job_name); deleted += 1
        except Exception as exc:
            logger.warning("reap stale fence job %s failed: %s", job_name, exc)
    return deleted
```
Leave `reap_older_generations` untouched (existing tests keep passing).

- [ ] **Step 4: Run** — `.venv/bin/python -m pytest tests/services/runtime/test_blob_lease.py -v` → PASS (new + existing).

- [ ] **Step 5: Lead reviews diff + commits** — `git add backend/services/runtime/blob_lease.py tests/services/runtime/test_blob_lease.py && git commit -m "Add renew-invariant fence_epoch + reap_stale_fence_epochs to BlobLease (WS3 fence correctness)"`

---

## Phase B — Controller Job builder + entrypoint

### Task B1: `controller_launch.build_controller_job_manifest` (pure, CPU-by-construction)

**Files:**
- Create: `backend/agents/rlm/controller_launch.py`
- Test: `tests/agents/rlm/test_controller_launch.py`

**Interfaces:**
- Produces: `build_controller_job_manifest(*, paper, project_id, fence_epoch, image, cpu_pool_label, namespace, service_account, env, backoff_limit=3, command) -> dict`. `cpu_pool_label` is a `"key=value"` string; `command` is the argv list from the controller entrypoint (Task B2). Returns a `batch/v1` Job dict with **no** `nvidia.com/gpu` toleration/resources and a CPU `nodeSelector`.

- [ ] **Step 1: Write failing test**

```python
from backend.agents.rlm.controller_launch import build_controller_job_manifest

def test_controller_manifest_is_cpu_only():
    m = build_controller_job_manifest(
        paper="1412.6980", project_id="prj_x", fence_epoch=2,
        image="img:v1", cpu_pool_label="reprolab/pool=cpu", namespace="default",
        service_account="reprolab-sa", env={"K": "V"}, command=["python", "-m", "x"])
    pod = m["spec"]["template"]["spec"]
    text = str(m)
    assert "nvidia.com/gpu" not in text                 # no GPU anywhere
    assert pod["nodeSelector"] == {"reprolab/pool": "cpu"}
    assert m["metadata"]["name"].endswith("-fe2") or "fe2" in m["metadata"]["name"]
    assert pod["containers"][0]["command"] == ["python", "-m", "x"]
    assert m["spec"]["backoffLimit"] == 3
```

- [ ] **Step 2: Run to verify it fails** — `pytest tests/agents/rlm/test_controller_launch.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement** (pure dict; job name embeds `fence_epoch` as `fe<N>`):

```python
"""Pure builder for the durable-controller K8s Job (CPU-only, fenced)."""
from __future__ import annotations

def _cpu_node_selector(label: str) -> dict[str, str]:
    k, _, v = label.partition("=")
    return {k: v}

def build_controller_job_manifest(*, paper, project_id, fence_epoch, image,
                                  cpu_pool_label, namespace, service_account,
                                  env, command, backoff_limit=3) -> dict:
    job_name = f"controller-{project_id}-fe{fence_epoch}"
    env_list = [{"name": k, "value": str(v)} for k, v in sorted(env.items())]
    return {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {"name": job_name, "namespace": namespace,
                     "labels": {"app": "reprolab-controller",
                                "reprolab/project": project_id,
                                "reprolab/fence-epoch": str(fence_epoch)}},
        "spec": {
            "backoffLimit": backoff_limit,
            "template": {"metadata": {"labels": {"app": "reprolab-controller"}},
                "spec": {
                    "serviceAccountName": service_account,
                    "restartPolicy": "Never",
                    "nodeSelector": _cpu_node_selector(cpu_pool_label),
                    "containers": [{"name": "controller", "image": image,
                                    "command": command, "env": env_list,
                                    "resources": {"requests": {"cpu": "2", "memory": "8Gi"}}}],
                }}},
    }
```

- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Lead reviews + commits** — `git commit -m "Add pure CPU-only fenced controller Job manifest builder"`

### Task B2: Fence threading via env fallback + `durable_controller_default_for_sandbox`

**Recon deviation (simplification):** the fence already threads through the existing
`k8s_job_cell_runner` `fence_generation` ContextVar (read once at `run_matrix` :1908, embedded by
`_job_name`→`fenced_job_name`, labelled `reprolab-generation`). So **no `controller_entry` wrapper
module** is needed and `build_controller_command` is **unchanged** (still the `campaign` CLI). Instead
the submit (C3) stamps `OPENRESEARCH_DURABLE_CONTROLLER=1` + `OPENRESEARCH_CELL_FENCE_EPOCH=<fence_epoch>`
into the controller Job env, and `_get_fence_generation()` gains an **env fallback** so the in-Pod
campaign fences its cell Jobs by the stable `fence_epoch` with zero wrapper code. The carrier
(`fence_generation`) now conveys the stable fence_epoch value — an explicit ContextVar binding still
wins over the env fallback.

**Files:**
- Modify: `backend/agents/rlm/run_controller.py` (add helper), `backend/agents/rlm/k8s_job_cell_runner.py` (`_get_fence_generation` env fallback)
- Test: `tests/agents/rlm/test_run_controller.py`, `tests/agents/rlm/test_k8s_job_cell_runner.py` (ADD new tests only)

**Interfaces:**
- Produces: `durable_controller_default_for_sandbox(sandbox: str) -> bool` (True for `"gcp"` unless `OPENRESEARCH_DURABLE_CONTROLLER=0`); `_get_fence_generation()` reads `OPENRESEARCH_CELL_FENCE_EPOCH` when the ContextVar is unbound. `build_controller_command` unchanged.

- [ ] **Step 1: Write failing tests**

```python
from backend.agents.rlm import run_controller as rc

def test_build_controller_command_targets_entrypoint():
    cmd = rc.build_controller_command("1412.6980", "prj_x")
    assert cmd[:3] == ["python", "-m", "backend.agents.rlm.controller_entry"]
    assert "--resume" in cmd and "prj_x" in cmd

def test_durable_default_on_for_gcp_off_for_others(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_DURABLE_CONTROLLER", raising=False)
    assert rc.durable_controller_default_for_sandbox("gcp") is True
    assert rc.durable_controller_default_for_sandbox("local") is False

def test_durable_opt_out_with_zero(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_DURABLE_CONTROLLER", "0")
    assert rc.durable_controller_default_for_sandbox("gcp") is False
```

- [ ] **Step 2: Run to verify fail** → FAIL.
- [ ] **Step 3: Implement** — retarget `build_controller_command`; add:
```python
def durable_controller_default_for_sandbox(sandbox: str) -> bool:
    if sandbox != "gcp":
        return False
    raw = os.environ.get("OPENRESEARCH_DURABLE_CONTROLLER", "").strip().lower()
    return raw not in ("0", "false", "no")   # default-ON for gcp, explicit-0 opts out
```
Create `backend/agents/rlm/controller_entry.py` that reads the fence from `acquire`/`renew` (drill-time wiring) and sets `os.environ["OPENRESEARCH_CELL_FENCE_EPOCH"]` before invoking `backend.cli` campaign. Keep `durable_controller_enabled()` env-only/default-false (unchanged).

- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Lead reviews + commits** — `git commit -m "Retarget controller command to fence-threading entrypoint; add durable default-for-gcp helper"`

---

## Phase C — Durable submit + handle state (`live_runs.py`, hot; single owner)

### Task C1: `ControllerHandle` field + handle-aware liveness

**Files:**
- Modify: `backend/services/events/live_runs.py` (`LiveRunState` at :337, `_load_run` at :1113, liveness at :1122)
- Test: `tests/services/events/test_live_runs_durable_seam.py` (ADD new tests only)

**Interfaces:**
- Produces: `LiveRunState.controller: ControllerHandle | None = None` where `ControllerHandle(BaseModel)` = `{job_name: str, fence_epoch: int, submitted_epoch: float}`; a run with `status in {"queued","running"}` and a present `controller` is **active** regardless of `pid`.

- [ ] **Step 1: Write failing test**

```python
def test_running_durable_run_with_controller_is_active(tmp_runs):
    # a durable run persists pid=None + controller handle
    state = make_state(status="running", pid=None,
                       controller={"job_name": "controller-prj_x-fe1",
                                   "fence_epoch": 1, "submitted_epoch": 100.0})
    persist(state, tmp_runs)
    loaded = LiveRuns(tmp_runs)._load_run("prj_x")
    assert loaded is not None and loaded.status == "running"   # NOT marked dead
```

- [ ] **Step 2: Run to verify fail** → FAIL (`controller` unknown / run marked interrupted).
- [ ] **Step 3: Implement** — add `ControllerHandle(BaseModel)` + `controller` field; in `_load_run` (:1122) treat `status in {"queued","running"} and controller is not None` as active before the `_pid_exists(pid)` check; mirror in the idempotency guard at :1022 and get_run.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Lead reviews + commits** — `git commit -m "Add ControllerHandle to LiveRunState; treat handle-bearing run as active (pid=None safe)"`

### Task C2: Default-ON-for-gcp in `_should_use_durable_controller`

**Files:** Modify `backend/services/events/live_runs.py` (:953). Test: same file as C1.

- [ ] **Step 1: Failing test**
```python
def test_should_use_durable_defaults_on_for_gcp(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_DURABLE_CONTROLLER", raising=False)
    lr = LiveRuns(tmp)
    assert lr._should_use_durable_controller(req(sandbox="gcp")) is True
    assert lr._should_use_durable_controller(req(sandbox="local")) is False
```
- [ ] **Step 2: Run → FAIL** (currently keys on `durable_controller_enabled()` default-false).
- [ ] **Step 3: Implement** — replace body with `return run_controller.durable_controller_default_for_sandbox(request.sandbox)`.
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Lead reviews + commits** — `git commit -m "Default durable controller ON for sandbox=gcp (opt-out via =0)"`

### Task C3: Real `_submit_durable_controller` (takeover-safe) + `_ControllerCluster` seam

**Files:** Modify `backend/services/events/live_runs.py` (:973). Test: same file as C1.

**Interfaces:**
- Consumes: `BlobLease` (fence_epoch), `controller_launch.build_controller_job_manifest`, `run_controller.build_controller_command`.
- Produces: `_ControllerCluster` protocol with `acquire_lease`, `is_current`, `submit_job`, `wait_ready`, `delete_job_confirmed`, `list_jobs`, `reap`, `now`; `_ControllerNotReady` + `_ControllerStuck` exceptions. Ordering: **acquire → is_current → submit → wait_ready → (is_current) reap → record handle**. Reap uses `BlobLease.reap_stale_fence_epochs`.

- [ ] **Step 1: Write failing tests** (fakes injected):
```python
def test_submit_happy_path_records_handle(fake_cluster):
    lr = LiveRuns(tmp, controller_cluster=fake_cluster)
    state = await lr._submit_durable_controller(req(sandbox="gcp"), project_id="prj_x", uploaded_paper=None)
    assert state.controller.fence_epoch == fake_cluster.token.fence_epoch
    assert fake_cluster.submitted and fake_cluster.reaped_after_ready
    assert fake_cluster.reaped_order == ["submit", "ready", "reap"]

def test_lease_none_adopts_no_submit(fake_cluster_locked):
    lr = LiveRuns(tmp, controller_cluster=fake_cluster_locked)  # acquire → None
    state = await lr._submit_durable_controller(...)            # returns existing, no submit
    assert not fake_cluster_locked.submitted

def test_not_ready_confirmed_delete_raises_notready(fake_cluster_slow):
    with pytest.raises(_ControllerNotReady):
        await lr._submit_durable_controller(...)
    assert fake_cluster_slow.deleted_confirmed

def test_not_ready_unconfirmed_delete_raises_stuck(fake_cluster_stuck):
    with pytest.raises(_ControllerStuck):
        await lr._submit_durable_controller(...)
```
- [ ] **Step 2: Run → FAIL** (still `NotImplementedError`).
- [ ] **Step 3: Implement** the §3.2 sequence exactly; inject `_ControllerCluster` via constructor (default `None` → a real GCS/K8s-backed impl built lazily, never constructed under test).
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Lead reviews + commits** — `git commit -m "Implement takeover-safe durable controller submit (acquire/is_current/submit/ready/reap/handle)"`

### Task C4: Pre-submit-only graceful fallback

**Files:** Modify `backend/services/events/live_runs.py` (:1062 call site). Test: same file as C1.

- [ ] **Step 1: Failing test**
```python
async def test_pre_submit_error_falls_back_to_popen(fake_cluster_acquire_raises, capwarn):
    lr = LiveRuns(tmp, controller_cluster=fake_cluster_acquire_raises)
    state = await lr._start_python_run(req(sandbox="gcp"), project_id="prj_x", uploaded_paper=None)
    assert state.pid is not None                          # local Popen ran
    assert "durable_controller_fallback" in capwarn.text

async def test_controller_stuck_does_not_fall_back(fake_cluster_stuck):
    with pytest.raises(_ControllerStuck):                 # never split-brain
        await lr._start_python_run(req(sandbox="gcp"), ...)
```
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** the try/except at :1062 per §3.3 (`_ControllerStuck` re-raises; every other exception logs `durable_controller_fallback` + falls through to Popen).
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Lead reviews + commits** — `git commit -m "Add pre-submit-only durable-controller fallback; fail loud on stuck remote Job"`

---

## Phase D — CPU cloud lane

### Task D1: `cpu_class.requires_gpu` classifier

**Files:** Create `backend/agents/rlm/cpu_class.py`. Test: `tests/agents/rlm/test_cpu_class.py`.

**Interfaces:**
- Produces: `requires_gpu(cell: dict, *, trusted_cpu: bool = False) -> bool`; `run_is_cpu_class(cells: list[dict], *, trusted_cpu: bool = False) -> bool` (True only if every cell is CPU). `_GPU_FRAMEWORKS = {"verl", ...}`.

- [ ] **Step 1: Failing tests**
```python
from backend.agents.rlm.cpu_class import requires_gpu, run_is_cpu_class

def test_hard_gpu_signal_overrides_cpu_declaration():
    assert requires_gpu({"accelerator": "cpu", "est_vram_gb": 40}) is True   # GPU wins + warns
    assert requires_gpu({"accelerator": "cpu", "framework": "verl"}) is True

def test_trusted_cpu_declaration_routes_cpu():
    assert requires_gpu({"accelerator": "cpu"}, trusted_cpu=True) is False

def test_unknown_defaults_gpu():
    assert requires_gpu({}) is True

def test_run_cpu_class_requires_all_cpu():
    assert run_is_cpu_class([{"accelerator": "cpu"}, {"accelerator": "cpu"}], trusted_cpu=True) is True
    assert run_is_cpu_class([{"accelerator": "cpu"}, {"est_vram_gb": 24}], trusted_cpu=True) is False
```
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement**
```python
_GPU_FRAMEWORKS = {"verl"}
def _hard_gpu(cell):
    if float(cell.get("est_vram_gb") or 0) > 0: return True
    if str(cell.get("framework") or cell.get("image_key") or "").lower() in _GPU_FRAMEWORKS: return True
    if cell.get("distributed") or cell.get("nproc_per_node"): return True
    return False
def requires_gpu(cell, *, trusted_cpu=False):
    if _hard_gpu(cell):
        return True                          # hard signal wins (caller emits cpu_gpu_conflict warning)
    acc = str(cell.get("accelerator") or "").lower()
    if acc == "cpu" and (trusted_cpu or True):  # no hard GPU signal → honor cpu
        return False
    if acc == "gpu":
        return True
    return True                              # unknown ⇒ conservative GPU
def run_is_cpu_class(cells, *, trusted_cpu=False):
    return bool(cells) and all(not requires_gpu(c, trusted_cpu=trusted_cpu) for c in cells)
```
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Lead reviews + commits** — `git commit -m "Add deterministic CPU-class classifier (hard GPU signals override soft cpu)"`

### Task D2: `accelerator="cpu"` manifest branch in `k8s_job_cell_runner._build_job_manifest`

**Files:** Modify `backend/agents/rlm/k8s_job_cell_runner.py` (:640, GPU block :779-857). Test: `tests/agents/rlm/test_k8s_job_cell_runner.py` (ADD new tests only).

**Interfaces:**
- Consumes: a new `accelerator: str = "gpu"` kwarg threaded into `_build_job_manifest` (default `"gpu"` ⇒ byte-identical). Branch **before** any gpu_count logic.

- [ ] **Step 1: Failing test**
```python
def test_cpu_manifest_omits_gpu_and_uses_cpu_pool():
    m = _build_job_manifest(..., accelerator="cpu", cpu_pool_label="reprolab/pool=cpu")
    text = str(m)
    assert "nvidia.com/gpu" not in text
    assert "OPENRESEARCH_CELL_GPU_COUNT" not in text
    assert m_pod(m)["nodeSelector"] == {"reprolab/pool": "cpu"}

def test_gpu_manifest_byte_identical_when_accelerator_default():
    a = _build_job_manifest(..., gpu_plan=plan1)              # no accelerator arg
    b = _build_job_manifest(..., gpu_plan=plan1, accelerator="gpu")
    assert a == b                                             # golden: default unchanged
```
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** — early `if accelerator == "cpu":` branch that builds a CPU pod (no toleration, no `nvidia.com/gpu` resources, no `OPENRESEARCH_CELL_GPU_COUNT` env, CPU `nodeSelector` + CPU/mem requests); the existing GPU path is untouched when `accelerator=="gpu"`.
- [ ] **Step 4: Run → PASS** (+ existing tests green).
- [ ] **Step 5: Lead reviews + commits** — `git commit -m "Add CPU accelerator branch to cell Job manifest (GPU path byte-identical)"`

### Task D3: CPU routing + all-cells-infra-failure local fallback

**Files:** Modify `backend/agents/rlm/k8s_job_cell_runner.py` (`run_matrix`); the run-level routing in `primitives.py::_execute_cell_matrix` or `run_experiment`. Test: `tests/agents/rlm/test_k8s_job_cell_runner.py` (ADD).

**Interfaces:**
- Consumes: `cpu_class.run_is_cpu_class` + `OPENRESEARCH_CPU_CLOUD_CELLS`. Produces: when every submitted CPU cell returns `status == STATUS_ERROR` with an infra reason, `run_matrix` (or its caller) signals a typed `cpu_cloud_all_infra_failed` so the caller reruns the SAME cells locally with a `cpu_cloud_fallback` warning — NOT aggregated as an experiment failure.

- [ ] **Step 1: Failing test**
```python
def test_all_cpu_cells_infra_error_triggers_local_rerun(fake_k8s_all_error, capwarn):
    results = run_cpu_matrix_with_fallback(cells_cpu, k8s=fake_k8s_all_error, local=fake_local_ok)
    assert fake_local_ok.ran_cells == cells_cpu             # same cells rerun locally
    assert "cpu_cloud_fallback" in capwarn.text

def test_partial_cpu_success_is_not_fallback(fake_k8s_one_ok):
    results = run_cpu_matrix_with_fallback(cells_cpu, k8s=fake_k8s_one_ok, local=fake_local_ok)
    assert not fake_local_ok.ran_cells                      # a real result is kept, no rerun
```
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** — detect `all(r.status == STATUS_ERROR for r in cpu_results)` with an infra reason set; on all-infra-failure, rerun the identical cells via the local matrix path + emit `cpu_cloud_fallback`. A single successful cell suppresses fallback.
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Lead reviews + commits** — `git commit -m "Route CPU-class cells to CPU Job; local rerun on all-cells infra failure"`

### Task D4: `OPENRESEARCH_CPU_CLOUD_CELLS` default-ON-for-gcp

**Files:** Modify `backend/agents/rlm/feature_flags.py` + the routing gate in D3. Test: `tests/agents/rlm/test_cpu_class.py` (ADD).

- [ ] **Step 1: Failing test** — flag unset + `sandbox=gcp` ⇒ CPU cells route to cloud; `=0` ⇒ local; `sandbox=local` ⇒ local (byte-identical).
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** `cpu_cloud_cells_default_for_sandbox(sandbox)` mirroring `durable_controller_default_for_sandbox` (gcp default-ON, `=0` opt-out).
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Lead reviews + commits** — `git commit -m "Default CPU cloud cells ON for sandbox=gcp (opt-out via =0)"`

---

## Phase E — Operator runbook + final verification

### Task E1: Operator checklist doc

**Files:** Create `docs/runbooks/2026-07-12-cpu-lane-and-durable-drill-operator-checklist.md`.

- [ ] **Step 1:** Write the doc: the CPU node-pool `gcloud` command (spec §8.1), the `reprolab-sa` KSA/RBAC binding (§8.2), and the **Pod-kill durability drill** acceptance criteria (§8.3): kill the controller Pod mid-run, assert a successor reacquires the lease (fence_epoch bumps), reaps the predecessor's older-fence Jobs, resumes the same lineage with no local fallback; and a CPU-cloud Adam run reproduces `best_runs/adam` within tolerance.
- [ ] **Step 2: Lead reviews + commits** — `git commit -m "Add CPU-lane + durable-controller operator checklist and drill acceptance"`

### Task E2: Full-suite + off-state verification (lead-run, no commit of its own)

- [ ] **Step 1:** `.venv/bin/python -m pytest tests/ -n auto` → establish that the only failures are the known-pre-existing baseline (3 × `TestResolveAuto` + the credential/OCR probes per the 2026-07-11 handoff), no new regressions.
- [ ] **Step 2:** `uvx ruff@0.15.16 check .` → clean.
- [ ] **Step 3: Off-state proof** — with `OPENRESEARCH_DURABLE_CONTROLLER=0` and `OPENRESEARCH_CPU_CLOUD_CELLS=0`, assert durable → Popen and CPU cells → local (byte-identical). Run tripwires `tests/agents/rlm/test_single_verdict_authority_guard.py` + `tests/rlm/test_registry.py` (still 19).
- [ ] **Step 4:** `.venv/bin/python scripts/gen_flag_registry.py && .venv/bin/python scripts/gen_flag_registry.py --check` → registry current for the two new flags.

---

## Self-Review

**Spec coverage:** §3.0 fence→A1; §3.1 controller manifest→B1; §3.2 submit→C3; §3.3 fallback→C4; §3.4 classifier→D1; §3.5 CPU manifest→D2; §3.6 entrypoint→B2; §3.7 handle→C1; §5 CPU fail-soft→D3; §6 flags→C2/B2/D4; §8 operator→E1; §9 tests→each task + E2. No gap.

**Placeholder scan:** every code step carries real code; no TBD/TODO. The one deliberately deferred item (autonomous resubmit sweeper, spec §7) is out of scope by design, logged not stubbed.

**Type consistency:** `fence_epoch: int` on `LeaseToken` is produced in A1 and consumed in B1/B2/C3; `ControllerHandle{job_name,fence_epoch,submitted_epoch}` produced in C1, consumed in C3; `requires_gpu(cell,*,trusted_cpu)` / `run_is_cpu_class` consistent D1→D3; `accelerator="gpu"` default keeps the GPU manifest byte-identical (D2 golden test). `durable_controller_default_for_sandbox` produced B2, consumed C2; `cpu_cloud_cells_default_for_sandbox` produced/consumed D4.
