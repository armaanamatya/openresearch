"""Unit tests for the checkpointed, fail-soft ReproductionRun state machine (Phase 1c, Unit C).

Hermetic: no network, no LLM, no real GPU/cloud. ``FakeComputeProvider`` is
the only ``ComputeProvider`` implementation exercised here -- it records
every call (``self.calls``) so the lifecycle ORDER is directly assertable,
and is scriptable (``watch_sequence``/``recover_bundle``) so the emergency
and budget-breach branches are reachable deterministically. ``clock`` is
always injected where elapsed-time accounting matters, never real time.

Covers all 7 invariants from the Phase 1c plan's Unit C:
  1. No GPU before the gates            -> test_happy_path_reaches_finalize_in_order
                                             test_plan_only_never_acquires_gpu
  2. DOWN_SCOPE adopts the trimmed scope -> test_down_scope_adopts_trimmed_scope
  3. Graceful terminal order             -> test_happy_path_reaches_finalize_in_order
  4. Emergency shutdown + recover()      -> test_emergency_stop_recovers
  5. Periodic sync witness               -> test_periodic_sync_witness_recorded_across_watch_polls
                                             (also exercised incidentally by #4/#6)
  6. Budget breach at WATCH              -> test_budget_breach_at_watch_triggers_emergency_shutdown
  7. Checkpointed (+ fail-soft)          -> test_green_red_aborts_before_gpu (write happens)
                                             test_checkpoint_write_failure_is_fail_soft (write fails)

Plus one bonus test for the module's overarching hard constraint ("wrap every
provider call so a raise degrades to a safe TEARDOWN terminal, never
propagates out of run()"): test_unexpected_exception_never_propagates_out_of_run.
"""

from __future__ import annotations

from pathlib import Path

from backend.agents.resilience.budget import RunBudget
from backend.agents.rlm.reproduction_run import ReproductionOutcome, ReproductionRun
from backend.agents.schemas import ScopeSpec
from backend.services.runtime.compute_provider import (
    CapacityReport,
    ComputeLease,
    ComputeProvider,
    ReportBundle,
    RunHandle,
    RunStatus,
)
from backend.services.runtime.feasibility_triage import FeasibilityTriage
from backend.services.runtime.gpu_catalog import find_by_alias
from backend.services.runtime.run_plan import RequiredAsset, RunPlan

_SKU = find_by_alias("rtx4090")


class FakeComputeProvider(ComputeProvider):
    """Scriptable ``ComputeProvider`` test double.

    ``watch_sequence`` scripts the polls ``watch()`` yields; ``recover_bundle``
    scripts what ``recover()`` returns (``None`` = unrecoverable, the
    ``ComputeProvider`` default). ``self.calls`` is the ordered list of
    method names invoked -- the primary lifecycle-order witness.
    ``provision_cpu_plans`` additionally records every ``plan`` passed to
    ``provision_cpu`` so DOWN_SCOPE adoption (invariant 2) is directly
    observable rather than merely inferred from the outcome.
    """

    def __init__(self, *, watch_sequence=None, recover_bundle=None):
        self.calls: list[str] = []
        self.provision_cpu_plans: list[object] = []
        self._watch = watch_sequence or [RunStatus(state="terminal", synced=True)]
        self._recover = recover_bundle

    def preflight(self, plan):
        self.calls.append("preflight")
        return CapacityReport(available=True)

    def provision_cpu(self, plan):
        self.calls.append("provision_cpu")
        self.provision_cpu_plans.append(plan)
        return ComputeLease(cloud="gcp", ref="r1")

    def stage(self, lease, bundle, run_spec):
        self.calls.append("stage")

    def acquire_gpu(self, lease):
        self.calls.append("acquire_gpu")
        return lease

    def launch(self, lease, run_spec):
        self.calls.append("launch")
        return RunHandle(id="h", lease=lease)

    def watch(self, handle):
        self.calls.append("watch")
        for s in self._watch:
            yield s

    def collect(self, handle):
        self.calls.append("collect")
        return ReportBundle(ok=True, report_path="/r")

    def release_gpu(self, lease):
        self.calls.append("release_gpu")

    def teardown(self, lease, *, reason):
        self.calls.append(f"teardown:{reason}")

    def recover(self, lease_ref):
        self.calls.append("recover")
        return self._recover


class _SequenceClock:
    """Deterministic injectable clock: pops each value in order, then repeats the last."""

    def __init__(self, values):
        self._values = list(values)

    def __call__(self) -> float:
        if len(self._values) > 1:
            return self._values.pop(0)
        return self._values[0]


class _RaisingTriage:
    """A triage stand-in whose .triage() raises something other than BudgetExhausted.

    Used only to prove run()'s outer safety net -- a bug in this module's own
    orchestration code, not a ComputeProvider call -- still degrades to a
    safe TEARDOWN instead of propagating.
    """

    def triage(self, plan, sku):
        raise RuntimeError("boom")


def _plan(scope, budget):
    return RunPlan(
        scope=scope, budget=budget, required_assets=(RequiredAsset("dataset", "alfworld"),)
    )


# ---------------------------------------------------------------------------
# Invariants 1 + 3: no GPU before the gates; graceful COLLECT -> RELEASE_GPU -> TEARDOWN order
# ---------------------------------------------------------------------------


def test_happy_path_reaches_finalize_in_order(tmp_path: Path):
    prov = FakeComputeProvider()
    run = ReproductionRun(
        plan=_plan(
            ScopeSpec(models=["qwen3-1.7b"], datasets=[{"name": "alfworld"}], seeds=[0]),
            RunBudget(max_gpu_hours=100.0),
        ),
        provider=prov,
        triage=FeasibilityTriage(reachability_probe=lambda a: "reachable"),
        sku=_SKU,
        state_dir=tmp_path,
        green_gate=lambda lease: True,
    )
    out = run.run()
    assert isinstance(out, ReproductionOutcome)
    assert out.state == "FINALIZE" and out.gpu_acquired is True
    # ACQUIRE_GPU comes AFTER provision + green, and COLLECT before RELEASE_GPU before TEARDOWN.
    assert prov.calls.index("acquire_gpu") > prov.calls.index("provision_cpu")
    teardown_idx = next(i for i, c in enumerate(prov.calls) if c.startswith("teardown"))
    assert prov.calls.index("collect") < prov.calls.index("release_gpu") < teardown_idx


def test_plan_only_never_acquires_gpu(tmp_path: Path):
    prov = FakeComputeProvider()
    run = ReproductionRun(
        plan=_plan(
            ScopeSpec(models=["qwen2.5-7b"], datasets=[{"name": "alfworld"}], seeds=[0]),
            RunBudget(max_gpu_hours=0.0001),  # infeasible → PLAN_ONLY
        ),
        provider=prov,
        triage=FeasibilityTriage(reachability_probe=lambda a: "reachable"),
        sku=_SKU,
        state_dir=tmp_path,
    )
    out = run.run()
    assert out.state == "PLAN_ONLY" and out.gpu_acquired is False
    assert "acquire_gpu" not in prov.calls and "provision_cpu" not in prov.calls


# ---------------------------------------------------------------------------
# Invariant 2: DOWN_SCOPE adopts the trimmed scope before proceeding
# ---------------------------------------------------------------------------


def test_down_scope_adopts_trimmed_scope(tmp_path: Path):
    prov = FakeComputeProvider()
    scope = ScopeSpec(
        models=["qwen2.5-7b", "qwen3-1.7b"], datasets=[{"name": "alfworld"}], seeds=[0]
    )
    # Full 2-model scope costs ~1.0 gpu-h; the smallest-model trim costs
    # ~0.33 gpu-h -- a budget of 0.5 fits the trim but not the full scope,
    # which is exactly the DOWN_SCOPE decision boundary (see
    # feasibility_triage._trim_scope).
    run = ReproductionRun(
        plan=_plan(scope, RunBudget(max_gpu_hours=0.5)),
        provider=prov,
        triage=FeasibilityTriage(reachability_probe=lambda a: "reachable"),
        sku=_SKU,
        state_dir=tmp_path,
        green_gate=lambda lease: True,
    )
    out = run.run()
    assert out.decision == "DOWN_SCOPE"
    assert out.state == "FINALIZE" and out.gpu_acquired is True
    # The adopted (trimmed) scope -- not the original 2-model scope -- is
    # what actually reached provision_cpu.
    assert len(prov.provision_cpu_plans) == 1
    adopted_scope = prov.provision_cpu_plans[0].scope
    assert adopted_scope.models == ["qwen3-1.7b"]


# ---------------------------------------------------------------------------
# Invariant 4 (+5): emergency shutdown recovers artifacts when the compute
# vanishes before COLLECT could run
# ---------------------------------------------------------------------------


def test_emergency_stop_recovers(tmp_path: Path):
    prov = FakeComputeProvider(
        watch_sequence=[
            RunStatus(state="running", synced=True),
            RunStatus(state="stopped_uncollected", synced=True),
        ],
        recover_bundle=ReportBundle(ok=True, report_path="/recovered"),
    )
    run = ReproductionRun(
        plan=_plan(
            ScopeSpec(models=["qwen3-1.7b"], datasets=[{"name": "alfworld"}], seeds=[0]),
            RunBudget(max_gpu_hours=100.0),
        ),
        provider=prov,
        triage=FeasibilityTriage(reachability_probe=lambda a: "reachable"),
        sku=_SKU,
        state_dir=tmp_path,
        green_gate=lambda lease: True,
    )
    out = run.run()
    assert out.state == "RECOVERED" and out.report.report_path == "/recovered"
    assert "recover" in prov.calls
    assert "collect" not in prov.calls  # COLLECT is bypassed on an emergency stop
    assert run.sync_log == [True, True]  # last-synced witness recorded before the stop


# ---------------------------------------------------------------------------
# Invariant 1 (negative, via GREEN_GATE) + 7 (checkpoint file exists)
# ---------------------------------------------------------------------------


def test_green_red_aborts_before_gpu(tmp_path: Path):
    prov = FakeComputeProvider()
    run = ReproductionRun(
        plan=_plan(
            ScopeSpec(models=["qwen3-1.7b"], datasets=[{"name": "alfworld"}], seeds=[0]),
            RunBudget(max_gpu_hours=100.0),
        ),
        provider=prov,
        triage=FeasibilityTriage(reachability_probe=lambda a: "reachable"),
        sku=_SKU,
        state_dir=tmp_path,
        green_gate=lambda lease: False,  # RED
    )
    out = run.run()
    assert out.gpu_acquired is False and "acquire_gpu" not in prov.calls
    assert (tmp_path / "reproduction_run.json").exists()  # checkpointed


# ---------------------------------------------------------------------------
# Invariant 6: a budget breach observed mid-WATCH is also an emergency shutdown
# ---------------------------------------------------------------------------


def test_budget_breach_at_watch_triggers_emergency_shutdown(tmp_path: Path):
    prov = FakeComputeProvider(
        watch_sequence=[
            RunStatus(state="running", synced=True),
            RunStatus(state="running", synced=True),
        ],
    )
    # gpu_start=0.0, poll #1 now=0.0 (0h elapsed, ok), poll #2 now=7200.0
    # (2h elapsed >= the 1.0h cap) -> BudgetExhausted on the second poll.
    clock = _SequenceClock([0.0, 0.0, 7200.0])
    run = ReproductionRun(
        plan=_plan(
            ScopeSpec(models=["qwen3-1.7b"], datasets=[{"name": "alfworld"}], seeds=[0]),
            RunBudget(max_gpu_hours=1.0),
        ),
        provider=prov,
        triage=FeasibilityTriage(reachability_probe=lambda a: "reachable"),
        sku=_SKU,
        state_dir=tmp_path,
        green_gate=lambda lease: True,
        clock=clock,
    )
    out = run.run()
    assert out.state == "TEARDOWN"
    assert out.gpu_acquired is True  # GPU billing had genuinely started
    assert "collect" not in prov.calls  # emergency bypasses graceful COLLECT
    assert run.sync_log == [True, True]  # a sync was recorded before the breach fired
    assert any("budget" in r.lower() for r in out.reasons)


# ---------------------------------------------------------------------------
# Invariant 5: every watch poll's synced flag is recorded, in order
# ---------------------------------------------------------------------------


def test_periodic_sync_witness_recorded_across_watch_polls(tmp_path: Path):
    prov = FakeComputeProvider(
        watch_sequence=[
            RunStatus(state="running", synced=False),
            RunStatus(state="running", synced=True),
            RunStatus(state="terminal", synced=True),
        ],
    )
    run = ReproductionRun(
        plan=_plan(
            ScopeSpec(models=["qwen3-1.7b"], datasets=[{"name": "alfworld"}], seeds=[0]),
            RunBudget(max_gpu_hours=100.0),
        ),
        provider=prov,
        triage=FeasibilityTriage(reachability_probe=lambda a: "reachable"),
        sku=_SKU,
        state_dir=tmp_path,
        green_gate=lambda lease: True,
    )
    out = run.run()
    assert out.state == "FINALIZE"
    # Every poll's synced flag, in order -- including the one False -- so an
    # emergency stop can distinguish "last synced" from "stranded."
    assert run.sync_log == [False, True, True]


# ---------------------------------------------------------------------------
# Invariant 7: checkpointed, and checkpoint failures are fail-soft
# ---------------------------------------------------------------------------


def test_checkpoint_write_failure_is_fail_soft(tmp_path: Path):
    bogus_state_dir = tmp_path / "not_a_dir"
    bogus_state_dir.write_text("i am a file, not a directory")  # mkdir() on this always raises

    prov = FakeComputeProvider()
    run = ReproductionRun(
        plan=_plan(
            ScopeSpec(models=["qwen3-1.7b"], datasets=[{"name": "alfworld"}], seeds=[0]),
            RunBudget(max_gpu_hours=100.0),
        ),
        provider=prov,
        triage=FeasibilityTriage(reachability_probe=lambda a: "reachable"),
        sku=_SKU,
        state_dir=bogus_state_dir,
        green_gate=lambda lease: True,
    )
    out = run.run()  # every _checkpoint() call fails internally; the run must still complete
    assert out.state == "FINALIZE" and out.gpu_acquired is True
    assert not bogus_state_dir.is_dir()  # confirms checkpointing genuinely never succeeded


# ---------------------------------------------------------------------------
# Bonus: the overarching fail-soft hard constraint (never propagate out of run())
# ---------------------------------------------------------------------------


def test_unexpected_exception_never_propagates_out_of_run(tmp_path: Path):
    prov = FakeComputeProvider()
    run = ReproductionRun(
        plan=_plan(
            ScopeSpec(models=["qwen3-1.7b"], datasets=[{"name": "alfworld"}], seeds=[0]),
            RunBudget(max_gpu_hours=100.0),
        ),
        provider=prov,
        triage=_RaisingTriage(),
        sku=_SKU,
        state_dir=tmp_path,
    )
    out = run.run()  # must not raise, even though _RaisingTriage.triage() does
    assert out.state == "TEARDOWN" and out.gpu_acquired is False
    assert "acquire_gpu" not in prov.calls
