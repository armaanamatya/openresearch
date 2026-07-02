# Phase 1c — Unified ComputeProvider + ReproductionRun + both providers + GCP→Azure failover — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add the Layer-A whole-run lifecycle abstraction (`ComputeProvider` + `ReproductionRun` state machine) that wires the Phase-1b gates, plus **both** concrete providers — single-VM `VmComputeProvider(gcp)` and cluster `ClusterComputeProvider` (wrapping the existing GKE/AKS Job backends, no SSH) — and a **GCP→Azure failover** selector. All default-OFF ⇒ the live run path is byte-identical today.

**Architecture:** One `ComputeProvider` ABC with typed tier ops (`provision_cpu`/`acquire_gpu`/`release_gpu`/`collect`/`recover`) + a typed `ComputeLease{cpu,disk,gpu}`. Cloud differences live in a neutral `CloudProfile{k8s: CloudSpec|None, vm: VmSpec|None}` (wraps the existing K8s `CloudSpec`, never extends it). `ReproductionRun` is a checkpointed, fail-soft state machine (RESOLVE→TRIAGE→PROVISION_CPU→PROVISION_ASSETS→GREEN_GATE→ACQUIRE_GPU→RUN→WATCH→COLLECT→RELEASE_GPU→FINALIZE→TEARDOWN) that reaches `ACQUIRE_GPU` **only after** triage+budget+GREEN pass. `VmComputeProvider` lifts the SDAR gcloud bash; `ClusterComputeProvider` wraps `GkeJobBackend`/`AksJobBackend` and consumes the failover selector.

**Tech Stack:** Python 3.12 / floor 3.11; `pytest` (socket-hermetic); stdlib + existing schemas. No new third-party deps. No live cloud calls in tests (fakes/injection).

## Global Constraints

- **Everything default-OFF ⇒ byte-identical live path.** The controller is opt-in via `OPENRESEARCH_UNIFIED_RUN=1`; failover via `OPENRESEARCH_CLOUD_FAILOVER` (comma list, e.g. `gcp,azure`; empty = OFF). Unset = today's bash-VM / existing Job dispatch, unchanged. **Do NOT modify `backend/agents/rlm/run.py`'s live path** (the controller is a peer entrypoint, not a rewrite).
- **Do NOT extend the K8s `CloudSpec`** (`k8s_job_backend.py`). `CloudProfile`/`VmSpec` are a new neutral module that *wraps/references* it.
- **Hermetic only.** No real `gcloud`/`kubectl`/cloud SDK calls in tests — inject fakes. Live GPU/cluster validation is operator-gated (never CI), same discipline as `gke_check.sh`. `VmComputeProvider` command builders are asserted by golden-command tests (argv comparison), never executed.
- **CredentialBroker precondition (VM path):** `VmComputeProvider` must NOT ship a raw `.env`/OAuth token in any staged bundle or command (the current bash SCPs a raw `.env` — do NOT reproduce that). Phase 1c stages via a redaction-safe seam; the real `CredentialBroker` is Phase 1d. A redaction test enforces "no secret-shaped value in a staged bundle or logged command."
- **Fail-soft:** every state transition + provider op degrades to a safe terminal (provisioning failure → `Exclusion`/plan report; execution failure → repairable), never an unhandled raise.
- Env-var naming canonically `OPENRESEARCH_*`. Run tests with `.venv/bin/python -m pytest <path> -q`; lint `uvx ruff@0.15.16 check <path>`.
- **Commits milestone-level** (one at the end of Phase 1c, or split VM/cluster if large); no Conventional-Commits prefix; no `Co-Authored-By` trailer; author `lolout1`. Confirm before committing.

## Component → file map

| Component | New/extends | Where |
|---|---|---|
| `ComputeProvider` ABC + `ComputeLease`/`CapacityReport`/`RunHandle`/`RunStatus`/`ReportBundle` | NEW | `backend/services/runtime/compute_provider.py` |
| `CloudProfile{k8s,vm}` + `VmSpec` | NEW (neutral) | `backend/services/runtime/cloud_profile.py` |
| `select_backend_with_failover` + `failover_preference` | NEW | `backend/services/runtime/cloud_failover.py` |
| `ReproductionRun` state machine + `FakeComputeProvider` | NEW | `backend/agents/rlm/reproduction_run.py` (+ `FakeComputeProvider` in the test module) |
| `VmComputeProvider(gcp)` | NEW | `backend/services/runtime/vm_compute_provider.py` |
| `ClusterComputeProvider` | NEW | `backend/services/runtime/cluster_compute_provider.py` |

---

## Unit A — GCP→Azure failover selector

**Files:** Create `backend/services/runtime/cloud_failover.py`; Test `tests/services/runtime/test_cloud_failover.py`.

**Interfaces:**
```python
@dataclass(frozen=True)
class BackendSelection:
    backend: object            # the constructed RuntimeBackend
    cloud: str                 # "gcp" | "azure" — the cloud that won
    attempts: tuple[tuple[str, str], ...]   # ((cloud, "ok"|"<err>"), ...) in order tried

def failover_preference() -> list[str]:
    """Parse OPENRESEARCH_CLOUD_FAILOVER (comma list, e.g. 'gcp,azure'); [] when unset/empty."""

def select_backend_with_failover(
    preference: list[str], *, run_budget=None, gpu_plan=None,
    availability: dict[str, Callable[[], None]] | None = None,   # cloud -> ensure_*_available (injected in tests)
    backend_factory: Callable[..., object] | None = None,        # (cloud, run_budget, gpu_plan) -> backend (injected)
) -> BackendSelection:
    """Try each cloud in order: call its ensure_available; on SandboxRuntimeError
    (backend_unavailable) or a capacity signal, record + try the next; else build
    + return its backend. Raises SandboxRuntimeError(backend_unavailable) only when
    EVERY cloud fails (message lists the per-cloud reasons)."""
```
Defaults (real path): `availability = {"gcp": ensure_gcp_available, "azure": ensure_azure_available}` (lazy-imported from `gke_job_backend`/`aks_job_backend`); `backend_factory` wraps `primitives._backend_for_sandbox_mode(SandboxMode(cloud), ...)`.

**Behavior:** provision-time failover only (v1). A `backend_unavailable`/`capacity_exhausted`-classed `SandboxRuntimeError` from `ensure_available` (or the factory) → try next cloud. Any OTHER error propagates (a real bug shouldn't be masked as "cloud down"). Mid-run failover (GCP dies mid-training) is out of scope — note it in the module docstring.

- [ ] **Step 1: Write the failing tests**

```python
import pytest
from backend.services.runtime.cloud_failover import (
    select_backend_with_failover, failover_preference, BackendSelection,
)
from backend.services.runtime.interface import SandboxRuntimeError, RuntimeCauseKind


def _down():
    def _raise():
        raise SandboxRuntimeError(RuntimeCauseKind.backend_unavailable, "simulated cloud down")
    return _raise


def test_failover_picks_azure_when_gcp_down():
    made = []
    sel = select_backend_with_failover(
        ["gcp", "azure"],
        availability={"gcp": _down(), "azure": lambda: None},
        backend_factory=lambda cloud, **_: made.append(cloud) or f"backend:{cloud}",
    )
    assert isinstance(sel, BackendSelection)
    assert sel.cloud == "azure" and sel.backend == "backend:azure"
    assert made == ["azure"]                          # gcp never built (unavailable)
    assert sel.attempts[0][0] == "gcp" and "unavailable" in sel.attempts[0][1].lower()


def test_first_cloud_wins_when_available():
    sel = select_backend_with_failover(
        ["gcp", "azure"],
        availability={"gcp": lambda: None, "azure": lambda: None},
        backend_factory=lambda cloud, **_: f"backend:{cloud}",
    )
    assert sel.cloud == "gcp"


def test_all_clouds_down_raises():
    with pytest.raises(SandboxRuntimeError):
        select_backend_with_failover(
            ["gcp", "azure"],
            availability={"gcp": _down(), "azure": _down()},
            backend_factory=lambda cloud, **_: f"backend:{cloud}",
        )


def test_non_capacity_error_propagates_not_failed_over():
    made = []
    with pytest.raises(RuntimeError):
        select_backend_with_failover(
            ["gcp", "azure"],
            availability={"gcp": lambda: (_ for _ in ()).throw(RuntimeError("bug")), "azure": lambda: None},
            backend_factory=lambda cloud, **_: made.append(cloud) or f"backend:{cloud}",
        )
    assert made == []                                 # a real bug is not masked as "cloud down"


def test_preference_parsing(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_CLOUD_FAILOVER", raising=False)
    assert failover_preference() == []
    monkeypatch.setenv("OPENRESEARCH_CLOUD_FAILOVER", "gcp, azure")
    assert failover_preference() == ["gcp", "azure"]
```

- [ ] **Step 2:** Run → FAIL (module missing). `.venv/bin/python -m pytest tests/services/runtime/test_cloud_failover.py -q`
- [ ] **Step 3:** Implement `cloud_failover.py`. Confirm the real class name + "kind" attribute of the credential/capacity error by reading `backend/agents/execution.py` (`SandboxRuntimeError`) + `gke_job_backend.ensure_gcp_available`/`aks_job_backend.ensure_azure_available` (what they raise). Classify "cloud down" = the error kind used for `backend_unavailable`/`capacity_exhausted`; everything else propagates.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5 (wiring, flag-gated, byte-identical OFF):** find the callers of `primitives._backend_for_sandbox_mode` for a cloud sandbox; where the run first selects a cloud backend, add: `if failover_preference(): backend = select_backend_with_failover(failover_preference(), ...).backend`. Guard so an empty preference is the exact current call (byte-identical). Run the touching test module(s) + a broad `tests/agents/rlm/ -q -k "sandbox or backend"` smoke. Report which call site was wired.

---

## Unit B — `ComputeProvider` foundation + neutral `CloudProfile`

**Files:** Create `backend/services/runtime/compute_provider.py`, `backend/services/runtime/cloud_profile.py`; Test `tests/services/runtime/test_compute_provider.py`, `tests/services/runtime/test_cloud_profile.py`.

**Interfaces (`compute_provider.py`):**
```python
@dataclass(frozen=True)
class ComputeLease:
    cloud: str = ""                      # "gcp" | "azure"
    cpu: object | None = None            # opaque handle (VM name / cluster ctx)
    disk: object | None = None
    gpu: object | None = None
    ref: str = ""                        # stable id for recover() after a crash
    meta: dict = field(default_factory=dict)

@dataclass(frozen=True)
class CapacityReport:
    available: bool
    reason: str = ""
    est_wait_s: float | None = None

@dataclass(frozen=True)
class RunHandle:
    id: str
    lease: ComputeLease
    meta: dict = field(default_factory=dict)

@dataclass(frozen=True)
class RunStatus:
    state: str                           # "running" | "terminal" | "stalled" | "stopped_uncollected"
    detail: str = ""
    synced: bool = False                 # True once artifacts are off-box for this poll

@dataclass(frozen=True)
class ReportBundle:
    ok: bool
    report_path: str | None = None
    artifacts: dict = field(default_factory=dict)
    reason: str = ""

class ComputeProvider(ABC):
    @abstractmethod
    def preflight(self, plan) -> CapacityReport: ...      # quota/capacity — NO lease, NO billing
    @abstractmethod
    def provision_cpu(self, plan) -> ComputeLease: ...     # cheap CPU tier (+cache disk); NO gpu
    def stage(self, lease: ComputeLease, bundle, run_spec) -> None: ...      # ship code+spec (default no-op)
    @abstractmethod
    def acquire_gpu(self, lease: ComputeLease) -> ComputeLease: ...          # ARMS ceiling; billing starts
    @abstractmethod
    def launch(self, lease: ComputeLease, run_spec) -> RunHandle: ...
    @abstractmethod
    def watch(self, handle: RunHandle) -> "Iterator[RunStatus]": ...         # stream + periodic sync
    @abstractmethod
    def collect(self, handle: RunHandle) -> ReportBundle: ...
    @abstractmethod
    def release_gpu(self, lease: ComputeLease) -> None: ...
    @abstractmethod
    def teardown(self, lease: ComputeLease, *, reason: str) -> None: ...
    def recover(self, lease_ref: str) -> "ReportBundle | None": return None  # default: unrecoverable
```

**Interfaces (`cloud_profile.py`):** neutral — imports the existing `CloudSpec` for `k8s`, defines `VmSpec` as pure config (the provider owns the command logic).
```python
@dataclass(frozen=True)
class VmSpec:
    zone: str = ""
    cpu_machine_type: str = ""
    gpu_machine_type: str = ""
    accelerator_type: str = ""
    accelerator_count: int = 0
    image: str = ""                      # or machine_image
    machine_image: str = ""
    cache_disk_name: str = ""
    tiering_strategy: str = "stage_on_gpu"   # | "machine_type_flip" | "cpu_warm_disk_then_gpu_attach"
    max_run_duration_s: int | None = None    # control-plane cost ceiling (gcloud max-run-duration=STOP)
    capacity_signatures: tuple[str, ...] = ()  # stderr substrings meaning STOCKOUT / ZONE_RESOURCE_POOL_EXHAUSTED

@dataclass(frozen=True)
class CloudProfile:
    cloud: str                           # "gcp" | "azure"
    k8s: object | None = None            # the EXISTING k8s_job_backend.CloudSpec, unchanged (typed loosely to avoid import coupling)
    vm: VmSpec | None = None
```

- [ ] **Step 1: Write the failing tests** (`test_compute_provider.py`)

```python
from backend.services.runtime.compute_provider import (
    ComputeProvider, ComputeLease, CapacityReport, RunHandle, RunStatus, ReportBundle,
)


def test_lease_and_report_shapes():
    lease = ComputeLease(cloud="gcp", cpu="vm-1", ref="run-7")
    assert lease.gpu is None and lease.ref == "run-7"
    assert CapacityReport(available=False, reason="stockout").available is False
    assert ReportBundle(ok=True, report_path="/r").report_path == "/r"


def test_abstract_methods_enforced():
    class _Partial(ComputeProvider):
        pass
    import pytest
    with pytest.raises(TypeError):        # cannot instantiate without the abstract methods
        _Partial()


def test_default_stage_and_recover_are_safe():
    class _Min(ComputeProvider):
        def preflight(self, plan): return CapacityReport(available=True)
        def provision_cpu(self, plan): return ComputeLease(cloud="x")
        def acquire_gpu(self, lease): return lease
        def launch(self, lease, run_spec): return RunHandle(id="h", lease=lease)
        def watch(self, handle): yield RunStatus(state="terminal")
        def collect(self, handle): return ReportBundle(ok=True)
        def release_gpu(self, lease): pass
        def teardown(self, lease, *, reason): pass
    p = _Min()
    assert p.stage(ComputeLease(), None, None) is None      # default no-op
    assert p.recover("ref") is None                         # default unrecoverable
```

(`test_cloud_profile.py`)
```python
from backend.services.runtime.cloud_profile import CloudProfile, VmSpec

def test_vmspec_defaults_stage_on_gpu():
    assert VmSpec().tiering_strategy == "stage_on_gpu"

def test_cloud_profile_holds_k8s_and_vm():
    from backend.services.runtime.gke_job_backend import _GCP_CLOUD   # the existing K8s CloudSpec
    prof = CloudProfile(cloud="gcp", k8s=_GCP_CLOUD, vm=VmSpec(zone="us-central1-b", accelerator_count=1))
    assert prof.cloud == "gcp" and prof.k8s is _GCP_CLOUD and prof.vm.zone == "us-central1-b"
```

- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement both modules (pure dataclasses + ABC). `cloud_profile.py` imports nothing from `k8s_job_backend` at module scope beyond what the type hint needs — keep `k8s` typed as `object | None` to avoid import coupling (the test imports the real `_GCP_CLOUD` to prove it fits).
- [ ] **Step 4:** Run → PASS. Lint clean.

---

## Unit C — `ReproductionRun` state machine + `FakeComputeProvider`

**Files:** Create `backend/agents/rlm/reproduction_run.py`; Test `tests/rlm/test_reproduction_run.py` (defines `FakeComputeProvider`).

**Consumes:** `ComputeProvider` (U-B) · `FeasibilityTriage`/`TriageDecision`/`RunPlan` (Phase 1b) · `RunBudget` (Phase 1b).

**Interfaces:**
```python
@dataclass(frozen=True)
class ReproductionOutcome:
    state: str                 # terminal state: "FINALIZE" | "PLAN_ONLY" | "TEARDOWN" | "RECOVERED"
    decision: str              # triage decision
    report: "ReportBundle | None"
    reasons: tuple[str, ...]
    gpu_acquired: bool         # invariant witness: True only if the gates passed

class ReproductionRun:
    STATES = ("RESOLVE","TRIAGE","PROVISION_CPU","PROVISION_ASSETS","GREEN_GATE",
              "ACQUIRE_GPU","RUN","WATCH","COLLECT","RELEASE_GPU","FINALIZE","TEARDOWN")
    def __init__(self, *, plan: "RunPlan", provider: "ComputeProvider", triage: "FeasibilityTriage",
                 sku, state_dir: "Path", green_gate: "Callable[[ComputeLease], bool] | None" = None,
                 sync_each_watch: bool = True, clock: "Callable[[], float]" = time.monotonic) -> None: ...
    def run(self) -> ReproductionOutcome: ...     # drive the machine; checkpoint each transition; fail-soft
```

**The invariants to ENCODE + TEST (this is the point of the unit):**
1. **No GPU before the gates.** `acquire_gpu` is called ONLY after `TRIAGE ∈ {PROCEED, DOWN_SCOPE}` AND `GREEN_GATE` passes AND budget ok. A `PLAN_ONLY` triage → terminal `PLAN_ONLY`, `provider.acquire_gpu` NEVER called, `gpu_acquired == False`.
2. **DOWN_SCOPE** adopts `TriageDecision.scope` before proceeding.
3. **Graceful teardown:** COLLECT → RELEASE_GPU → TEARDOWN in that order on a normal terminal.
4. **Emergency shutdown:** if `watch` yields `state == "stopped_uncollected"` (VM killed by ceiling/watchdog, bypassing COLLECT), the run calls `provider.recover(lease.ref)`; a non-None bundle → terminal `RECOVERED`; None → terminal TEARDOWN with the last synced state.
5. **Periodic sync witness:** each `watch` `RunStatus.synced` is recorded (so an emergency stop degrades to "last synced," never "strand").
6. **Budget breach at WATCH** (`plan.budget.check_gpu_hours`/`check_run_gpu_usd` raises) → emergency shutdown after a sync.
7. **Checkpointed:** after each transition write `{state, decision, lease_ref}` to `state_dir/reproduction_run.json` atomically (resume-safe); fail-soft (a write error never aborts the run).

`FakeComputeProvider` (in the test module) is scriptable: `stockout_polls`, `watch_sequence: list[RunStatus]`, `recover_bundle: ReportBundle|None`, and records `self.calls: list[str]` (the ordered method names) so tests assert the lifecycle order + that `acquire_gpu` is/ isn't present.

- [ ] **Step 1: Write the failing tests** (representative — implement all 7 invariants)

```python
from pathlib import Path
from backend.services.runtime.compute_provider import (
    ComputeProvider, ComputeLease, CapacityReport, RunHandle, RunStatus, ReportBundle,
)
from backend.agents.rlm.reproduction_run import ReproductionRun, ReproductionOutcome
from backend.services.runtime.feasibility_triage import FeasibilityTriage
from backend.services.runtime.run_plan import RunPlan, RequiredAsset
from backend.services.runtime.gpu_catalog import find_by_alias
from backend.agents.resilience.budget import RunBudget
from backend.agents.schemas import ScopeSpec

_SKU = find_by_alias("rtx4090")


class FakeComputeProvider(ComputeProvider):
    def __init__(self, *, watch_sequence=None, recover_bundle=None):
        self.calls = []
        self._watch = watch_sequence or [RunStatus(state="terminal", synced=True)]
        self._recover = recover_bundle
    def preflight(self, plan): self.calls.append("preflight"); return CapacityReport(available=True)
    def provision_cpu(self, plan): self.calls.append("provision_cpu"); return ComputeLease(cloud="gcp", ref="r1")
    def stage(self, lease, bundle, run_spec): self.calls.append("stage")
    def acquire_gpu(self, lease): self.calls.append("acquire_gpu"); return lease
    def launch(self, lease, run_spec): self.calls.append("launch"); return RunHandle(id="h", lease=lease)
    def watch(self, handle):
        self.calls.append("watch")
        for s in self._watch: yield s
    def collect(self, handle): self.calls.append("collect"); return ReportBundle(ok=True, report_path="/r")
    def release_gpu(self, lease): self.calls.append("release_gpu")
    def teardown(self, lease, *, reason): self.calls.append(f"teardown:{reason}")
    def recover(self, lease_ref): self.calls.append("recover"); return self._recover


def _plan(scope, budget):
    return RunPlan(scope=scope, budget=budget,
                   required_assets=(RequiredAsset("dataset", "alfworld"),))


def test_happy_path_reaches_finalize_in_order(tmp_path: Path):
    prov = FakeComputeProvider()
    run = ReproductionRun(
        plan=_plan(ScopeSpec(models=["qwen3-1.7b"], datasets=[{"name": "alfworld"}], seeds=[0]),
                   RunBudget(max_gpu_hours=100.0)),
        provider=prov, triage=FeasibilityTriage(reachability_probe=lambda a: "reachable"),
        sku=_SKU, state_dir=tmp_path, green_gate=lambda lease: True)
    out = run.run()
    assert out.state == "FINALIZE" and out.gpu_acquired is True
    # ACQUIRE_GPU comes AFTER provision + green, and COLLECT before RELEASE_GPU before TEARDOWN.
    assert prov.calls.index("acquire_gpu") > prov.calls.index("provision_cpu")
    teardown_idx = next(i for i, c in enumerate(prov.calls) if c.startswith("teardown"))
    assert prov.calls.index("collect") < prov.calls.index("release_gpu") < teardown_idx


def test_plan_only_never_acquires_gpu(tmp_path: Path):
    prov = FakeComputeProvider()
    run = ReproductionRun(
        plan=_plan(ScopeSpec(models=["qwen2.5-7b"], datasets=[{"name": "alfworld"}], seeds=[0]),
                   RunBudget(max_gpu_hours=0.0001)),          # infeasible → PLAN_ONLY
        provider=prov, triage=FeasibilityTriage(reachability_probe=lambda a: "reachable"),
        sku=_SKU, state_dir=tmp_path)
    out = run.run()
    assert out.state == "PLAN_ONLY" and out.gpu_acquired is False
    assert "acquire_gpu" not in prov.calls and "provision_cpu" not in prov.calls


def test_emergency_stop_recovers(tmp_path: Path):
    prov = FakeComputeProvider(
        watch_sequence=[RunStatus(state="running", synced=True),
                        RunStatus(state="stopped_uncollected", synced=True)],
        recover_bundle=ReportBundle(ok=True, report_path="/recovered"))
    run = ReproductionRun(
        plan=_plan(ScopeSpec(models=["qwen3-1.7b"], datasets=[{"name": "alfworld"}], seeds=[0]),
                   RunBudget(max_gpu_hours=100.0)),
        provider=prov, triage=FeasibilityTriage(reachability_probe=lambda a: "reachable"),
        sku=_SKU, state_dir=tmp_path, green_gate=lambda lease: True)
    out = run.run()
    assert out.state == "RECOVERED" and out.report.report_path == "/recovered"
    assert "recover" in prov.calls


def test_green_red_aborts_before_gpu(tmp_path: Path):
    prov = FakeComputeProvider()
    run = ReproductionRun(
        plan=_plan(ScopeSpec(models=["qwen3-1.7b"], datasets=[{"name": "alfworld"}], seeds=[0]),
                   RunBudget(max_gpu_hours=100.0)),
        provider=prov, triage=FeasibilityTriage(reachability_probe=lambda a: "reachable"),
        sku=_SKU, state_dir=tmp_path, green_gate=lambda lease: False)   # RED
    out = run.run()
    assert out.gpu_acquired is False and "acquire_gpu" not in prov.calls
    assert (tmp_path / "reproduction_run.json").exists()               # checkpointed
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement `reproduction_run.py` (drive `STATES`; call `triage.triage(plan, sku)` at TRIAGE; on PLAN_ONLY set terminal + return without provisioning; adopt DOWN_SCOPE scope; call `provider.preflight`→`provision_cpu`→`stage`; GREEN_GATE via the injected `green_gate` (default: assets present — for the fake, injected True/False); ACQUIRE_GPU→launch→watch loop (record `synced`, run budget checks, detect `stopped_uncollected`→recover); COLLECT→RELEASE_GPU→FINALIZE, or emergency→recover/TEARDOWN; checkpoint each transition atomically; wrap every provider call so a raise degrades to TEARDOWN, never propagates). **Step 4:** Run → PASS. Lint clean.

---

## Unit D — `VmComputeProvider(gcp)` + golden-command parity test

**Files:** Create `backend/services/runtime/vm_compute_provider.py`; Test `tests/services/runtime/test_vm_compute_provider.py`.

**Consumes:** `ComputeProvider` (U-B) · `CloudProfile`/`VmSpec` (U-B).

**Read the source bash to port (the golden reference):** `scripts/gcp_sdar_preflight.sh` (the `gcloud compute instances create` + capacity-poll + cache-disk + `stage_on_gpu` + the raw `.env` SCP you must NOT reproduce), `scripts/sdar_gcp_run.sh` (ssh launch of `backend.cli reproduce --sandbox local`), `scripts/sdar_gcp_watch.sh` (poll + periodic pull + stop), `scripts/sdar_gcp_optimal_run.sh` (stage order), `scripts/cancel_gcp_sdar_run.sh` (stop/delete). Extract the exact `gcloud`/`ssh`/`scp` argv shapes.

**Golden reference (extracted from the bash — the argv shapes the golden-command test asserts):**
```
# 13 gcloud verbs to cover: compute instances {create, describe, start, stop, set-scheduling,
#   set-machine-type, attach-disk}, compute disks {create, describe}, compute {ssh, scp},
#   compute machine-images create
# CREATE (machine-image path, gcp_sdar_preflight.sh:204-210):
gcloud compute instances create <INSTANCE> --zone <ZONE> --machine-type <GPU_MT> \
  --source-machine-image <MI_NAME> --maintenance-policy TERMINATE --no-restart-on-failure \
  --max-run-duration=<MAX_RUN_DUR> --instance-termination-action=STOP
# CREATE (image-family path, :212-216):
gcloud compute instances create <INSTANCE> --zone <ZONE> --machine-type <GPU_MT> \
  --image-family <CREATE_IMAGE> --image-project deeplearning-platform-release \
  --maintenance-policy TERMINATE --boot-disk-size 1000 --boot-disk-type pd-ssd \
  --metadata install-nvidia-driver=True --no-restart-on-failure \
  --max-run-duration=<MAX_RUN_DUR> --instance-termination-action=STOP
# CAPACITY SIGNATURES (stderr grep -qiE, preflight:219): STOCKOUT | enough resources | EXHAUSTED | currently unavailable
# STOP (sdar_gcp_watch.sh:101-112, with local-ssd retry):
gcloud --project <PROJECT> compute instances stop <INSTANCE> --zone <ZONE> --quiet
#   on 'local ssd|cannot be stopped' stderr → retry with --discard-local-ssd=true
# DESCRIBE status (optimal_run:182): gcloud compute instances describe <INSTANCE> --zone <ZONE> --format='value(status)'
# CACHE DISK (preflight:300,312): gcloud compute disks create <DISK> --zone <ZONE> --size=1000GB --type=pd-ssd ;
#   gcloud compute instances attach-disk <INSTANCE> --zone <ZONE> --disk <DISK> --mode=rw --device-name=sdar-cache
# SCP run_spec (preflight:659): gcloud compute scp --zone <ZONE> --project <PROJECT> --quiet <spec> <USER>@<INSTANCE>:<REMOTE>/runs/.cache/run_spec.json
# LAUNCH (preflight:675, detached): ssh ... "cd <REMOTE> && ( setsid nohup bash scripts/sdar_gcp_run.sh --run-spec runs/.cache/run_spec.json > runs/sdar_gcp_run.out 2>&1 </dev/null & )"
#   → which runs: backend.cli reproduce 2605.15155 --mode rlm --sandbox local --model <ROOT> --project-id <PID> ...
# stage_on_gpu (preflight:386): for STANDARD (on-demand) provisioning, stage on the GPU machine type
#   (GPU forces --maintenance-policy=TERMINATE; e2 STANDARD needs MIGRATE; no valid intermediate → no cheap CPU tier)
# THE RAW-.env SCP TO AVOID (preflight:351,372, dotglob): the sync SCPs `.env` (OAuth token) to the VM.
#   VmComputeProvider.stage MUST NOT do this — redact secret-shaped values (see the redaction test).
# CONFIG DEFAULTS: ZONE=us-central1-b PROJECT=deepinvent-ext-ut INSTANCE=sdar-a100-od
#   GPU_MT=a2-highgpu-4g (4×A100-80GB) MAX_RUN_DUR=100800s CPU_MT=e2-standard-16
```

**Design:** `VmComputeProvider(profile: CloudProfile)` implements every `ComputeProvider` method as a **pure argv builder + an injected runner** (`runner: Callable[[list[str]], ExecResult]`, default a real subprocess wrapper; tests inject a recording fake). NOTHING executes in tests. `tiering_strategy="stage_on_gpu"` default (provision on the GPU type, warm there — matches the current script). The command builders read `profile.vm` (`VmSpec`).

- `preflight` → `gcloud compute instances describe`/quota probe argv; classify a `capacity_signatures` stderr as unavailable.
- `provision_cpu` → under `stage_on_gpu`, this folds into the GPU-type create (no separate cheap tier — document it); the create argv = the golden `gcloud compute instances create` (machine type, accelerator, image/machine-image, disk, zone, `--maintenance-policy=TERMINATE`, `--max-run-duration=<s>,--instance-termination-action=STOP`).
- `stage` → **redaction-safe** code+run-spec push (NO raw `.env`; a `--run-spec` JSON with secret-shaped values dropped). Redaction test enforces it.
- `acquire_gpu` → under `stage_on_gpu`, arms the max-run-duration ceiling (already on the create) — effectively a no-op returning the lease with `gpu` set; document the fold.
- `launch` → `ssh ... backend.cli reproduce ... --sandbox local` (detached) argv.
- `watch` → poll (`gcloud compute instances describe` status) + periodic artifact `scp`/gsutil pull; yield `RunStatus`; detect a non-RUNNING VM as `stopped_uncollected`.
- `collect` → final artifact pull argv → `ReportBundle`.
- `release_gpu`/`teardown` → `gcloud compute instances stop`/`delete` argv (disk persists).
- `recover` → re-pull from a stopped-but-not-deleted VM's boot disk.

- [ ] **Step 1: Golden-command tests** — assert the emitted argv equals the bash's effective commands (with secrets redacted). E.g.:

```python
def test_create_argv_matches_bash_shape(tmp_path):
    calls = []
    from backend.services.runtime.cloud_profile import CloudProfile, VmSpec
    from backend.services.runtime.vm_compute_provider import VmComputeProvider
    prof = CloudProfile(cloud="gcp", vm=VmSpec(
        zone="us-central1-b", gpu_machine_type="a2-highgpu-4g",
        accelerator_type="nvidia-tesla-a100", accelerator_count=4,
        machine_image="sdar-ultra", max_run_duration_s=21600,
        capacity_signatures=("ZONE_RESOURCE_POOL_EXHAUSTED", "does not have enough resources")))
    prov = VmComputeProvider(prof, runner=lambda argv: calls.append(argv) or _ok())
    prov.provision_cpu(_run_plan())
    create = next(a for a in calls if a[:3] == ["gcloud", "compute", "instances"] and "create" in a)
    assert "--zone" in create and "us-central1-b" in create
    assert any("a100" in x for x in create) and "--max-run-duration=21600s" in " ".join(create) or "21600" in " ".join(create)


def test_stage_never_ships_a_secret(tmp_path):
    calls = []
    prov = VmComputeProvider(_gcp_profile(), runner=lambda argv: calls.append(argv) or _ok())
    prov.stage(_lease(), bundle={"env": {"ANTHROPIC_API_KEY": "sk-secret-xyz", "SEED": "0"}}, run_spec={})
    blob = repr(calls)
    assert "sk-secret-xyz" not in blob and "ANTHROPIC_API_KEY" not in blob   # redaction
    assert "SEED" in blob or "0" in blob                                     # non-secret survives


def test_capacity_signature_classified_unavailable():
    prov = VmComputeProvider(_gcp_profile(), runner=lambda argv: _err("ZONE_RESOURCE_POOL_EXHAUSTED"))
    rep = prov.preflight(_run_plan())
    assert rep.available is False and "exhaust" in rep.reason.lower()
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement `vm_compute_provider.py` (pure argv builders + injected runner; read the bash scripts for exact argv; redaction in `stage` via a `KEY|TOKEN|SECRET|PASSWORD`-drop over env). **Step 4:** Run → PASS. Lint clean. Note in the module docstring: live-VM parity (bash A/B) is the operator-gated Phase-1f step; this unit is argv-parity only.

---

## Unit E — `ClusterComputeProvider` + failover wiring

**Files:** Create `backend/services/runtime/cluster_compute_provider.py`; Test `tests/services/runtime/test_cluster_compute_provider.py`.

**Consumes:** `ComputeProvider` (U-B) · `select_backend_with_failover` (U-A) · the existing `GkeJobBackend`/`AksJobBackend` + `ensure_gcp_available`/`ensure_azure_available`.

**Design:** the cluster topology has no VM bracket — GPU is per-Job autoscaled and the orchestrator already runs in-cluster. So `ClusterComputeProvider(preference: list[str])` maps the `ComputeProvider` ops onto the Job model:
- `preflight` → try `select_backend_with_failover(preference)` in "probe mode": returns `CapacityReport(available=True, reason=<chosen cloud>)` or `available=False` when all clouds are down.
- `provision_cpu` → resolve + cache the winning backend via `select_backend_with_failover` (the failover happens HERE); the returned `ComputeLease.cloud` records which cloud won. No GPU billing (nodes autoscale per-Job).
- `acquire_gpu` → **no-op** returning the lease (GPU is acquired per-cell-Job by the K8s autoscaler, not a VM bracket) — document the fold.
- `launch`/`watch`/`collect` → in Phase 1c these are thin: `launch` records the run (the existing cell dispatch already runs via the backend); `watch` yields a single terminal `RunStatus` (full streaming integration is deferred — the existing Job path already streams); `collect` pulls the report via the object store. Keep them minimal + honest (docstring: full cluster-run integration is deferred; this unit delivers the provider shape + failover).
- `release_gpu`/`teardown` → no-op (Jobs self-clean via `ttl_seconds`); `recover` → re-pull the report from the object store.

The VALUE of this unit in Phase 1c = the **failover** (provision_cpu picks GCP, falls back to AKS) behind the `ComputeProvider` shape, hermetically tested. Full cluster-run execution stays on the existing Job path.

- [ ] **Step 1: Write the failing tests**

```python
from backend.services.runtime.cluster_compute_provider import ClusterComputeProvider
from backend.services.runtime.compute_provider import ComputeLease, CapacityReport
from backend.services.runtime.interface import SandboxRuntimeError, RuntimeCauseKind


def _down():
    def _r(): raise SandboxRuntimeError(RuntimeCauseKind.backend_unavailable, "down")
    return _r


def test_provision_fails_over_gcp_to_azure():
    prov = ClusterComputeProvider(
        preference=["gcp", "azure"],
        availability={"gcp": _down(), "azure": lambda: None},
        backend_factory=lambda cloud, **_: f"backend:{cloud}")
    lease = prov.provision_cpu(_run_plan())
    assert isinstance(lease, ComputeLease) and lease.cloud == "azure"


def test_preflight_reports_unavailable_when_all_down():
    prov = ClusterComputeProvider(
        preference=["gcp", "azure"],
        availability={"gcp": _down(), "azure": _down()},
        backend_factory=lambda cloud, **_: f"backend:{cloud}")
    assert prov.preflight(_run_plan()).available is False


def test_acquire_gpu_is_noop_no_vm_bracket():
    prov = ClusterComputeProvider(preference=["gcp"], availability={"gcp": lambda: None},
                                  backend_factory=lambda cloud, **_: f"backend:{cloud}")
    lease = prov.provision_cpu(_run_plan())
    assert prov.acquire_gpu(lease) is lease        # no VM GPU bracket on the cluster path
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement `cluster_compute_provider.py` (constructor takes `preference` + injectable `availability`/`backend_factory` threaded to `select_backend_with_failover`; the ops as designed above). **Step 4:** Run → PASS. Lint clean.

---

## Validation (whole phase)

- [ ] Full new suite: `.venv/bin/python -m pytest tests/services/runtime/test_cloud_failover.py tests/services/runtime/test_compute_provider.py tests/services/runtime/test_cloud_profile.py tests/services/runtime/test_vm_compute_provider.py tests/services/runtime/test_cluster_compute_provider.py tests/rlm/test_reproduction_run.py -q`
- [ ] Broad regression (collateral from the failover wiring): `.venv/bin/python -m pytest tests/services/runtime/ tests/agents/ -q -x` (or a scoped subset if the full set is slow).
- [ ] Ruff clean on all new files.
- [ ] Import smoke: `.venv/bin/python -c "import backend.agents.rlm.run, backend.cli; print('ok')"` (the live path must still import with the failover wiring present but the flag OFF).
- [ ] Confirm default-OFF byte-identical: with `OPENRESEARCH_CLOUD_FAILOVER` unset + `OPENRESEARCH_UNIFIED_RUN` unset, the backend-selection call site produces the same backend as before (a test or a grep-proof that the wiring is guarded).
- [ ] Docs: CLAUDE.md "Where to look first" note (ComputeProvider seam + failover flag) + memory update.

## Self-Review (against the spec §5.1–5.5, §13)

- §5.2 `ComputeProvider` typed tier ops + `ComputeLease{cpu,disk,gpu}` → U-B. ✓ Neutral `CloudProfile{k8s,vm}` wrapping (not extending) the K8s `CloudSpec` → U-B. ✓
- §5.3 `ReproductionRun` state machine, `ACQUIRE_GPU` gated on triage+budget+GREEN → U-C (invariant tests). ✓
- §5.4 graceful vs emergency teardown + periodic sync + `recover()` → U-C (emergency/recover tests). ✓
- §5.5 `stage_on_gpu` default (VM) → U-D. ✓
- §9 `FakeVmComputeProvider`-style full-lifecycle hermetic test + golden-command parity → U-C `FakeComputeProvider` + U-D golden test. ✓
- **User asks:** no-SSH cluster path (already in-cluster) exposed as a provider + **GCP→Azure failover** → U-A + U-E. ✓
- **Deferred (honest):** live-VM bash A/B parity (1f, operator-gated) · full cluster-run streaming integration (existing Job path already does it) · `CredentialBroker` (1d — U-D stages redaction-safe as the precondition) · `cpu_warm_disk_then_gpu_attach` real CPU tier (1d) · Azure VM provider (experimental). None block Phase 1c.
