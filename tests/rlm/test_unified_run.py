"""Tests for the opt-in composition root (Phase 1f).

``build_reproduction_run`` assembles the already-built Phase-1a..1e pieces
(``RunPlan``/``extract_required_assets``, ``FeasibilityTriage``,
``VmComputeProvider``/``CloudProfile``, ``gpu_catalog``, ``ReproductionRun``)
into one constructible, runnable thing. Hermetic: ``FakeComputeProvider``
never shells out to gcloud/ssh -- the FINALIZE composition test proves the
whole stack drives a real ``ReproductionRun.run()`` state machine to
completion with ZERO cloud spend.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.agents.resilience.budget import RunBudget
from backend.agents.rlm.reproduction_run import ReproductionOutcome, ReproductionRun
from backend.agents.rlm.unified_run import build_reproduction_run, unified_run_enabled
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

_SKU = find_by_alias("l4")


class FakeComputeProvider(ComputeProvider):
    """Minimal recording ``ComputeProvider`` double for the composition-root test.

    Trimmed copy of ``tests/rlm/test_reproduction_run.py``'s
    ``FakeComputeProvider``: a single terminal ``watch`` poll and an ok
    ``collect()`` bundle are enough to drive a happy-path run to FINALIZE.
    ``self.calls`` records the ordered method-call names invoked.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def preflight(self, plan):
        self.calls.append("preflight")
        return CapacityReport(available=True)

    def provision_cpu(self, plan):
        self.calls.append("provision_cpu")
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
        yield RunStatus(state="terminal", synced=True)

    def collect(self, handle):
        self.calls.append("collect")
        return ReportBundle(ok=True, report_path="/r")

    def release_gpu(self, lease):
        self.calls.append("release_gpu")

    def teardown(self, lease, *, reason):
        self.calls.append(f"teardown:{reason}")


# ---------------------------------------------------------------------------
# unified_run_enabled(): flag reading
# ---------------------------------------------------------------------------


def test_unified_run_enabled_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENRESEARCH_UNIFIED_RUN", raising=False)
    assert unified_run_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "True", "TRUE", "yes", "Yes"])
def test_unified_run_enabled_truthy_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("OPENRESEARCH_UNIFIED_RUN", value)
    assert unified_run_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "", "banana"])
def test_unified_run_enabled_falsy_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("OPENRESEARCH_UNIFIED_RUN", value)
    assert unified_run_enabled() is False


# ---------------------------------------------------------------------------
# build_reproduction_run(): pure assembly
# ---------------------------------------------------------------------------


def test_build_reproduction_run_returns_reproduction_run(tmp_path: Path) -> None:
    run = build_reproduction_run(
        paper_id="2605.15155",
        state_dir=tmp_path,
        provider=FakeComputeProvider(),
        triage=FeasibilityTriage(reachability_probe=lambda a: "reachable"),
        sku=_SKU,
    )
    assert isinstance(run, ReproductionRun)


def test_build_reproduction_run_supplies_defaults_when_omitted(tmp_path: Path) -> None:
    """No provider/triage/sku override -> the assembler wires real (inert)
    defaults. Constructing (never running) a ``VmComputeProvider`` builds no
    argv and makes no gcloud call, so this stays hermetic."""
    run = build_reproduction_run(paper_id="2605.15155", state_dir=tmp_path)
    assert isinstance(run, ReproductionRun)


# ---------------------------------------------------------------------------
# FINALIZE composition test: the whole Phase-1a..1e stack composes end-to-end
# ---------------------------------------------------------------------------


def test_build_reproduction_run_reaches_finalize(tmp_path: Path) -> None:
    prov = FakeComputeProvider()
    run = build_reproduction_run(
        paper_id="2605.15155",
        state_dir=tmp_path,
        scope=ScopeSpec(models=["qwen3-1.7b"], datasets=[{"name": "alfworld"}], seeds=[0]),
        budget=RunBudget(max_gpu_hours=100.0),
        provider=prov,
        triage=FeasibilityTriage(reachability_probe=lambda a: "reachable"),
        sku=_SKU,
        green_gate=lambda lease: True,
    )
    assert isinstance(run, ReproductionRun)

    out = run.run()

    assert isinstance(out, ReproductionOutcome)
    assert out.state == "FINALIZE"
    assert out.gpu_acquired is True
    assert out.decision == "PROCEED"
    assert out.report is not None and out.report.ok is True
    # ACQUIRE_GPU only after the gates, COLLECT before RELEASE_GPU before TEARDOWN.
    assert prov.calls.index("acquire_gpu") > prov.calls.index("provision_cpu")
    assert prov.calls.index("collect") < prov.calls.index("release_gpu")
