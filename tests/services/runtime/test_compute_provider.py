"""Unit tests for the ComputeProvider foundation (Phase 1c, Unit B)."""

import pytest

from backend.services.runtime.compute_provider import (
    CapacityReport,
    ComputeLease,
    ComputeProvider,
    ReportBundle,
    RunHandle,
    RunStatus,
)


def test_lease_and_report_shapes():
    lease = ComputeLease(cloud="gcp", cpu="vm-1", ref="run-7")
    assert lease.gpu is None and lease.ref == "run-7"
    assert CapacityReport(available=False, reason="stockout").available is False
    assert ReportBundle(ok=True, report_path="/r").report_path == "/r"


def test_abstract_methods_enforced():
    class _Partial(ComputeProvider):
        pass

    with pytest.raises(TypeError):  # cannot instantiate without the abstract methods
        _Partial()


def test_default_stage_and_recover_are_safe():
    class _Min(ComputeProvider):
        def preflight(self, plan):
            return CapacityReport(available=True)

        def provision_cpu(self, plan):
            return ComputeLease(cloud="x")

        def acquire_gpu(self, lease):
            return lease

        def launch(self, lease, run_spec):
            return RunHandle(id="h", lease=lease)

        def watch(self, handle):
            yield RunStatus(state="terminal")

        def collect(self, handle):
            return ReportBundle(ok=True)

        def release_gpu(self, lease):
            pass

        def teardown(self, lease, *, reason):
            pass

    p = _Min()
    assert p.stage(ComputeLease(), None, None) is None  # default no-op
    assert p.recover("ref") is None  # default unrecoverable
