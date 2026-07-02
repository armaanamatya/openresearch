"""tests/rlm/test_unified_paired_driver.py — UnifiedRunDriver + PairedDriver
(spec §7; Codex F8).

``UnifiedRunDriver`` wraps ``build_reproduction_run(...).run()`` (the opt-in
unified composition-root path): ``launch()`` builds + stashes the run object
without ever invoking ``.run()`` (which is synchronous), and ``await_result``
(called right after, same process, by the campaign loop) invokes ``.run()``
and maps the outcome. A resumed campaign process constructs a FRESH driver
with an empty stash, so ``await_result`` on an old handle falls through to
assess-from-disk semantics instead of re-running anything.

``PairedDriver`` alternates between a live and a unified delegate by
attempt_n parity and requires an explicit ``operator_ack=True`` (F8: no
default-flip may ever construct a working paired driver).

All hermetic: tmp_path-only, no real subprocess, no real ReproductionRun
(``build_run`` is always an injected fake), no sockets, no GPU, no LLM.
"""

from __future__ import annotations

import json
import types

import pytest

from backend.agents.rlm import attempt_driver as ad
from backend.agents.rlm.attempt_driver import (
    AttemptHandle,
    DriverError,
    PairedDriver,
    UnifiedRunDriver,
)
from backend.agents.rlm.campaign_policy import AttemptEnvelope

# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------


def _envelope(**overrides) -> AttemptEnvelope:
    base = dict(llm_usd=5.0, gpu_usd=8.0, gpu_hours=2.0, wall_s=3600.0, vm_ceiling_s=7200.0)
    base.update(overrides)
    return AttemptEnvelope(**base)


def _directives(**overrides):
    base = dict(
        attempt_n=1,
        project_id="prj_unified",
        paper_ref="2605.15155",
        run_spec_path=None,
        enforcement={"effective_wall_s": 3000.0, "vm_ceiling_s": 6000.0},
        seed_pointer=None,
        seed_lineage=None,
        target_floor=None,
        scope_spec=None,
        extra_guidance="",
        envelope=_envelope(),
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


class FakeClock:
    def __init__(self, start: float = 2_000_000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class FakeReproductionRun:
    """Stand-in for ``ReproductionRun``: records every ``.run()``/abort-like
    call; ``run_result`` is returned verbatim (or raised, if an Exception)."""

    def __init__(self, run_result=None, *, raises: Exception | None = None) -> None:
        self.run_result = run_result
        self.raises = raises
        self.run_calls = 0
        self.abort_calls: list = []
        self.teardown_calls: list = []

    def run(self):
        self.run_calls += 1
        if self.raises is not None:
            raise self.raises
        return self.run_result


class AbortableFakeRun(FakeReproductionRun):
    def abort(self) -> None:
        self.abort_calls.append("abort")


class TeardownOnlyFakeRun(FakeReproductionRun):
    def teardown(self) -> None:
        self.teardown_calls.append("teardown")


def _outcome(report_path: str | None, *, has_report: bool = True):
    report = types.SimpleNamespace(report_path=report_path) if has_report else None
    return types.SimpleNamespace(state="FINALIZE", report=report)


def _build_run_factory(run_obj, captured: dict | None = None):
    def _build_run(**kwargs):
        if captured is not None:
            captured.update(kwargs)
        return run_obj

    return _build_run


# ---------------------------------------------------------------------------
# UnifiedRunDriver.launch — envelope/enforcement -> RunBudget/VmSpec mapping
# ---------------------------------------------------------------------------


def test_launch_maps_envelope_and_enforcement_to_run_budget_and_vm_spec(tmp_path):
    """F2/F3 mapping test at the unified boundary: llm->max_usd,
    gpu_usd->max_run_gpu_usd, gpu_hours->max_gpu_hours,
    enforcement.effective_wall_s->max_wall_clock_seconds,
    enforcement.vm_ceiling_s->VmSpec.max_run_duration_s. NEVER
    envelope.wall_s/vm_ceiling_s (those are pre-co-tightening)."""
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    captured: dict = {}
    run_obj = FakeReproductionRun(run_result=_outcome("r.json"))
    driver = UnifiedRunDriver(runs_root=runs_root, build_run=_build_run_factory(run_obj, captured))

    envelope = _envelope(llm_usd=11.0, gpu_usd=22.0, gpu_hours=3.5, wall_s=999.0, vm_ceiling_s=888.0)
    directives = _directives(
        project_id="prj_map",
        paper_ref="2605.15155",
        envelope=envelope,
        enforcement={"effective_wall_s": 4321.0, "vm_ceiling_s": 6543.0},
    )

    driver.launch(directives)

    assert captured["paper_id"] == "2605.15155"
    assert captured["state_dir"] == runs_root / "prj_map"

    budget = captured["budget"]
    assert budget.max_usd == 11.0
    assert budget.max_run_gpu_usd == 22.0
    assert budget.max_gpu_hours == 3.5
    assert budget.max_wall_clock_seconds == 4321.0  # enforcement, NOT envelope.wall_s (999.0)

    vm_spec = captured["vm_spec"]
    assert vm_spec.max_run_duration_s == 6543  # enforcement.vm_ceiling_s, NOT envelope's 888.0


def test_launch_handles_missing_enforcement_keys(tmp_path):
    """A bare enforcement dict (no effective_wall_s/vm_ceiling_s, e.g. a
    minimal test double) must not raise -- both map to None."""
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    captured: dict = {}
    run_obj = FakeReproductionRun(run_result=_outcome("r.json"))
    driver = UnifiedRunDriver(runs_root=runs_root, build_run=_build_run_factory(run_obj, captured))

    directives = _directives(project_id="prj_bare", enforcement={})
    driver.launch(directives)

    assert captured["budget"].max_wall_clock_seconds is None
    assert captured["vm_spec"].max_run_duration_s is None


# ---------------------------------------------------------------------------
# UnifiedRunDriver.launch — never calls .run(), returns handle correctly
# ---------------------------------------------------------------------------


def test_launch_never_calls_run_returns_synchronous_handle(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    run_obj = FakeReproductionRun(run_result=_outcome("r.json"))
    driver = UnifiedRunDriver(
        runs_root=runs_root, build_run=_build_run_factory(run_obj), clock=FakeClock(5_000_000.0)
    )

    directives = _directives(project_id="prj_sync", attempt_n=4)
    handle = driver.launch(directives)

    assert run_obj.run_calls == 0  # .run() is NEVER invoked from launch()
    assert handle.pid is None
    assert handle.lease_ref is None
    assert handle.driver == "unified"
    assert handle.attempt_n == 4
    assert handle.project_id == "prj_sync"
    assert handle.run_dir == str(runs_root / "prj_sync")
    assert handle.launched_at == 5_000_000.0


def test_launch_force_quarantines_before_build_and_stages_seed_marker(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    project_id = "prj_quarantine"
    run_dir = runs_root / project_id
    code_dir = run_dir / "code"
    code_dir.mkdir(parents=True)
    (code_dir / "train.py").write_text("# stale residue")

    order: list[str] = []
    real_force_archive = ad.force_archive_incomplete

    def _tracking_force_archive(project_id_arg, root, *, reason):
        order.append("archive")
        return real_force_archive(project_id_arg, root, reason=reason)

    monkeypatch.setattr(ad, "force_archive_incomplete", _tracking_force_archive)

    def _build_run(**kwargs):
        order.append("build")
        return FakeReproductionRun(run_result=_outcome("r.json"))

    driver = UnifiedRunDriver(runs_root=runs_root, build_run=_build_run)
    directives = _directives(
        project_id=project_id, seed_pointer=str(tmp_path / "seed_code"), seed_lineage="champion", target_floor=0.5
    )

    driver.launch(directives)

    assert order == ["archive", "build"]
    assert not (run_dir / "code").exists()  # quarantined away before build

    marker = json.loads((run_dir / "campaign" / "seed_staging.json").read_text(encoding="utf-8"))
    assert marker["source_code_dir"] == str(tmp_path / "seed_code")
    assert marker["target_floor"] == 0.5
    assert marker["lineage"] == "champion"


# ---------------------------------------------------------------------------
# UnifiedRunDriver.await_result — outcome mapping
# ---------------------------------------------------------------------------


def test_await_result_report_present_maps_completed(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    run_obj = FakeReproductionRun(run_result=_outcome("/abs/final_report.json"))
    driver = UnifiedRunDriver(runs_root=runs_root, build_run=_build_run_factory(run_obj))

    handle = driver.launch(_directives(project_id="prj_present"))
    result = driver.await_result(handle)

    assert run_obj.run_calls == 1
    assert result.exit_condition == "completed"
    assert result.report_path == "/abs/final_report.json"
    assert result.run_dir == handle.run_dir


def test_await_result_plan_only_maps_completed_with_no_report_path(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    run_obj = FakeReproductionRun(run_result=_outcome(None, has_report=False))
    driver = UnifiedRunDriver(runs_root=runs_root, build_run=_build_run_factory(run_obj))

    handle = driver.launch(_directives(project_id="prj_planonly"))
    result = driver.await_result(handle)

    assert result.exit_condition == "completed"
    assert result.report_path is None


def test_await_result_run_exception_raises_driver_error(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    run_obj = FakeReproductionRun(raises=RuntimeError("boom"))
    driver = UnifiedRunDriver(runs_root=runs_root, build_run=_build_run_factory(run_obj))

    handle = driver.launch(_directives(project_id="prj_boom"))

    with pytest.raises(DriverError):
        driver.await_result(handle)


def test_await_result_resumed_handle_never_reruns(tmp_path):
    """A fresh driver in a NEW process (empty stash) reading a handle from a
    prior process must assess-from-disk, never call .run() on anything."""
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "prj_resumed"
    (run_dir).mkdir(parents=True)
    (run_dir / "final_report.json").write_text(json.dumps({"score": 1}), encoding="utf-8")

    fresh_driver = UnifiedRunDriver(runs_root=runs_root)  # no build_run ever called
    handle = AttemptHandle(
        attempt_n=2, project_id="prj_resumed", run_dir=str(run_dir), driver="unified",
        pid=None, launched_at=1.0, lease_ref=None,
    )

    result = fresh_driver.await_result(handle)

    assert result.exit_condition == "already_dead"
    assert result.report_path == str(run_dir / "final_report.json")


def test_await_result_resumed_handle_no_report_yet(tmp_path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "prj_resumed_bare"
    run_dir.mkdir(parents=True)

    fresh_driver = UnifiedRunDriver(runs_root=runs_root)
    handle = AttemptHandle(
        attempt_n=1, project_id="prj_resumed_bare", run_dir=str(run_dir), driver="unified",
        pid=None, launched_at=1.0, lease_ref=None,
    )

    result = fresh_driver.await_result(handle)

    assert result.exit_condition == "already_dead"
    assert result.report_path is None


def test_await_result_pops_stash_second_call_is_resumed_semantics(tmp_path):
    """Awaiting the same handle twice must not re-invoke .run() the second
    time -- the stash is consumed on first await."""
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    run_obj = FakeReproductionRun(run_result=_outcome("r.json"))
    driver = UnifiedRunDriver(runs_root=runs_root, build_run=_build_run_factory(run_obj))

    handle = driver.launch(_directives(project_id="prj_twice"))
    first = driver.await_result(handle)
    second = driver.await_result(handle)

    assert run_obj.run_calls == 1
    assert first.exit_condition == "completed"
    assert second.exit_condition == "already_dead"


# ---------------------------------------------------------------------------
# UnifiedRunDriver.abort
# ---------------------------------------------------------------------------


def test_abort_calls_stashed_run_abort_method(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    run_obj = AbortableFakeRun(run_result=_outcome("r.json"))
    driver = UnifiedRunDriver(runs_root=runs_root, build_run=_build_run_factory(run_obj))

    handle = driver.launch(_directives(project_id="prj_abort"))
    driver.abort(handle, reason="operator_stop")

    assert run_obj.abort_calls == ["abort"]
    # Stash consumed by abort -- a subsequent await treats it as resumed.
    result = driver.await_result(handle)
    assert result.exit_condition == "already_dead"
    assert run_obj.run_calls == 0


def test_abort_falls_back_to_teardown_when_no_abort_method(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    run_obj = TeardownOnlyFakeRun(run_result=_outcome("r.json"))
    driver = UnifiedRunDriver(runs_root=runs_root, build_run=_build_run_factory(run_obj))

    handle = driver.launch(_directives(project_id="prj_teardown"))
    driver.abort(handle, reason="x")

    assert run_obj.teardown_calls == ["teardown"]


def test_abort_noop_when_nothing_stashed(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    driver = UnifiedRunDriver(runs_root=runs_root)

    handle = AttemptHandle(
        attempt_n=1, project_id="prj_noop", run_dir=str(runs_root / "prj_noop"), driver="unified",
        pid=None, launched_at=1.0, lease_ref=None,
    )
    driver.abort(handle, reason="x")  # must not raise


def test_abort_swallows_delegate_exception(tmp_path):
    """Best-effort: a raising abort()/teardown() on the stashed object must
    not propagate -- the VM-side ceiling is the real backstop."""
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    class RaisingAbortRun(FakeReproductionRun):
        def abort(self):
            raise RuntimeError("provider unreachable")

    run_obj = RaisingAbortRun(run_result=_outcome("r.json"))
    driver = UnifiedRunDriver(runs_root=runs_root, build_run=_build_run_factory(run_obj))

    handle = driver.launch(_directives(project_id="prj_raising_abort"))
    driver.abort(handle, reason="x")  # must not raise


# ---------------------------------------------------------------------------
# PairedDriver — F8
# ---------------------------------------------------------------------------


class _RecordingDriver:
    kind = "recording"

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.launch_calls: list = []
        self.await_calls: list = []
        self.abort_calls: list = []

    def launch(self, directives):
        self.launch_calls.append(directives.attempt_n)
        return AttemptHandle(
            attempt_n=directives.attempt_n, project_id=directives.project_id,
            run_dir=f"/runs/{directives.project_id}", driver=self.kind,
            pid=None, launched_at=0.0, lease_ref=None,
        )

    def await_result(self, handle):
        self.await_calls.append(handle.attempt_n)
        return ad.AttemptRawResult(run_dir=handle.run_dir, report_path=None, exit_condition="completed")

    def abort(self, handle, *, reason):
        self.abort_calls.append((handle.attempt_n, reason))


def test_paired_requires_explicit_true_ack():
    live, unified = _RecordingDriver("live"), _RecordingDriver("unified")
    with pytest.raises(DriverError):
        PairedDriver(live=live, unified=unified, operator_ack=False)


def test_paired_ack_has_no_default():
    live, unified = _RecordingDriver("live"), _RecordingDriver("unified")
    with pytest.raises(TypeError):
        PairedDriver(live=live, unified=unified)  # type: ignore[call-arg]


def test_paired_alternates_by_attempt_parity():
    live, unified = _RecordingDriver("live"), _RecordingDriver("unified")
    driver = PairedDriver(live=live, unified=unified, operator_ack=True)

    for n in (1, 2, 3, 4, 5):
        driver.launch(_directives(attempt_n=n, project_id=f"prj_{n}"))

    assert live.launch_calls == [1, 3, 5]
    assert unified.launch_calls == [2, 4]


def test_paired_handle_driver_reflects_delegate_kind():
    live, unified = _RecordingDriver("live"), _RecordingDriver("unified")
    driver = PairedDriver(live=live, unified=unified, operator_ack=True)

    odd_handle = driver.launch(_directives(attempt_n=1, project_id="prj_odd"))
    even_handle = driver.launch(_directives(attempt_n=2, project_id="prj_even"))

    assert odd_handle.driver == "live"
    assert even_handle.driver == "unified"


def test_paired_await_result_and_abort_route_by_attempt_n():
    live, unified = _RecordingDriver("live"), _RecordingDriver("unified")
    driver = PairedDriver(live=live, unified=unified, operator_ack=True)

    odd_handle = driver.launch(_directives(attempt_n=1, project_id="prj_odd"))
    even_handle = driver.launch(_directives(attempt_n=2, project_id="prj_even"))

    driver.await_result(odd_handle)
    driver.await_result(even_handle)
    driver.abort(odd_handle, reason="r1")
    driver.abort(even_handle, reason="r2")

    assert live.await_calls == [1]
    assert unified.await_calls == [2]
    assert live.abort_calls == [(1, "r1")]
    assert unified.abort_calls == [(2, "r2")]
