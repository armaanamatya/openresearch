"""Tests for catalogue bugs I7 and I12.

I7 — ``--sandbox`` was a no-op for RLM ``run_experiment``:
  ``RunContext`` had no ``sandbox_mode`` field, ``run_pipeline_rlm`` never stored
  the flag, and ``_execute_in_sandbox`` hardcoded ``LocalDockerBackend()``.

I12 — ``ThreadPoolExecutor`` could block past its timeout on shutdown:
  ``with ThreadPoolExecutor(...) as pool:`` calls ``pool.shutdown(wait=True)`` on
  ``__exit__``, so a wedged worker thread blocks the primitive far past the
  ``TimeoutError`` that was already raised by ``.result(timeout=...)``.
"""

from __future__ import annotations

import concurrent.futures
import json
import threading
import time

import pytest

import backend.agents.rlm.primitives as primitives
from backend.agents.rlm.context import RunContext
from backend.agents.rlm.primitives import run_experiment


# ---------------------------------------------------------------------------
# I7 — sandbox_mode field on RunContext + backend selection
# ---------------------------------------------------------------------------


def test_run_context_accepts_sandbox_mode(tmp_path):
    """I7: RunContext accepts a sandbox_mode kwarg and stores it."""
    from backend.agents.dashboard_emitter import DashboardEmitter
    from backend.agents.resilience.cost import RunCostLedger
    from backend.agents.execution import SandboxMode

    project_dir = tmp_path / "prj"
    project_dir.mkdir()
    ctx = RunContext(
        project_id="prj",
        project_dir=project_dir,
        runs_root=tmp_path,
        dashboard=DashboardEmitter("prj", tmp_path),
        cost_ledger=RunCostLedger.load_jsonl(
            project_dir / "cost_ledger.jsonl", project_id="prj", attach_path=True
        ),
        llm_client=object(),
        provider="anthropic",
        model="test-model",
        sandbox_mode=SandboxMode.docker,
    )
    assert ctx.sandbox_mode is SandboxMode.docker


def test_run_context_sandbox_mode_default_is_none(tmp_path):
    """I7 back-compat: RunContext without sandbox_mode kwarg still works; defaults to None."""
    from backend.agents.dashboard_emitter import DashboardEmitter
    from backend.agents.resilience.cost import RunCostLedger

    project_dir = tmp_path / "prj"
    project_dir.mkdir()
    ctx = RunContext(
        project_id="prj",
        project_dir=project_dir,
        runs_root=tmp_path,
        dashboard=DashboardEmitter("prj", tmp_path),
        cost_ledger=RunCostLedger.load_jsonl(
            project_dir / "cost_ledger.jsonl", project_id="prj", attach_path=True
        ),
        llm_client=object(),
        provider="anthropic",
        model="test-model",
    )
    assert ctx.sandbox_mode is None


def test_backend_for_sandbox_mode_docker_returns_local_docker_backend():
    """I7: docker mode returns LocalDockerBackend (behaviour-identical to the hardcoded path)."""
    from backend.agents.execution import SandboxMode
    from backend.services.runtime.local_docker import LocalDockerBackend
    from backend.agents.rlm.primitives import _backend_for_sandbox_mode

    backend = _backend_for_sandbox_mode(SandboxMode.docker)
    assert isinstance(backend, LocalDockerBackend)


def test_backend_for_sandbox_mode_none_returns_local_docker_backend():
    """I7: None (no --sandbox flag) returns LocalDockerBackend."""
    from backend.services.runtime.local_docker import LocalDockerBackend
    from backend.agents.rlm.primitives import _backend_for_sandbox_mode

    backend = _backend_for_sandbox_mode(None)
    assert isinstance(backend, LocalDockerBackend)


def test_backend_for_sandbox_mode_unsupported_falls_back_with_warning(caplog):
    """I7: an unsupported mode (simulate) falls back to LocalDockerBackend with a WARNING.

    Note: SandboxMode.runpod and SandboxMode.local are now fully wired (RunpodBackend
    and LocalProcessBackend respectively; see test_runpod_wiring.py and
    test_local_sandbox_routing.py).  This test uses SandboxMode.simulate, which
    remains an unsupported / fallback-only mode in the RLM path.
    """
    import logging
    from backend.agents.execution import SandboxMode
    from backend.services.runtime.local_docker import LocalDockerBackend
    from backend.agents.rlm.primitives import _backend_for_sandbox_mode

    with caplog.at_level(logging.WARNING, logger="backend.agents.rlm.primitives"):
        backend = _backend_for_sandbox_mode(SandboxMode.simulate)

    assert isinstance(backend, LocalDockerBackend)
    assert any("not supported" in r.message for r in caplog.records)


def test_run_experiment_threads_sandbox_mode_to_execute_in_sandbox(
    make_context, tmp_path, monkeypatch
):
    """I7: run_experiment passes ctx.sandbox_mode into _execute_in_sandbox."""
    from backend.agents.execution import SandboxMode

    ctx = make_context(tmp_path)
    ctx.sandbox_mode = SandboxMode.docker

    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "commands.json").write_text(json.dumps(["python train.py"]))

    captured = {}

    async def fake_exec(code_path, env_id, commands, *, project_id, run_id,
                        sandbox_mode=None, run_budget=None, gpu_plan=None, gpu_mode=None, gpu_device_ids=()):
        captured["sandbox_mode"] = sandbox_mode
        return {"metrics": {}, "success": True, "logs": ""}

    monkeypatch.setattr(primitives, "_execute_in_sandbox", fake_exec)
    result = run_experiment(str(code_dir), "reprolab/test:env", ctx=ctx)

    assert result["success"] is True
    assert captured.get("sandbox_mode") is SandboxMode.docker


# ---------------------------------------------------------------------------
# I12 — ThreadPoolExecutor shutdown must not block past timeout
# ---------------------------------------------------------------------------


def test_run_experiment_does_not_block_on_shutdown_when_worker_wedges(
    make_context, tmp_path, monkeypatch
):
    """I12 (run_experiment): a wedged worker thread must not block the primitive at cleanup.

    Symptom: ``with ThreadPoolExecutor(...) as pool:`` exits by calling
    ``pool.shutdown(wait=True)``, which hangs until the wedged thread finishes.
    Fix: ``pool.shutdown(wait=False, cancel_futures=True)`` in a ``finally`` block.

    The test submits a blocking task, forces the ``.result(timeout=...)`` to time
    out, and asserts the function returns quickly (< 3 s) even though the worker
    thread is still blocked.  The blocking event is set afterward so the leaked
    thread can exit cleanly.
    """
    ctx = make_context(tmp_path)
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "commands.json").write_text(json.dumps(["python train.py"]))

    # An event the fake worker blocks on indefinitely.
    unblock = threading.Event()

    async def wedging_exec(code_path, env_id, commands, *, project_id, run_id,
                           sandbox_mode=None, run_budget=None, gpu_plan=None, gpu_mode=None, gpu_device_ids=()):
        # Block until the test releases us — simulates a wedged Docker call.
        unblock.wait()
        return {"metrics": {}, "success": True, "logs": ""}

    monkeypatch.setattr(primitives, "_execute_in_sandbox", wedging_exec)

    # Give run_experiment a deadline of ~0 s (1 s is the clamped minimum in
    # _timeout_for) so it immediately hits the TimeoutError path.
    from datetime import datetime, timezone, timedelta
    ctx.deadline_utc = datetime.now(timezone.utc) - timedelta(seconds=60)

    start = time.monotonic()
    result = run_experiment(str(code_dir), "reprolab/test:env", ctx=ctx)
    elapsed = time.monotonic() - start

    # Release the wedged worker so the thread can exit cleanly.
    unblock.set()

    assert result["success"] is False
    assert "timed out" in result.get("error", "")
    # The primitive must have returned well before any worker-thread join.
    # A generous 4 s bound — the clamped 1 s timeout + 3 s margin covers slow CI.
    assert elapsed < 4.0, (
        f"run_experiment blocked for {elapsed:.2f} s — "
        "pool.shutdown(wait=True) likely still in effect"
    )


def test_implement_baseline_does_not_block_on_shutdown_when_worker_wedges(
    make_context, tmp_path, monkeypatch
):
    """I12 (implement_baseline): same non-blocking shutdown guarantee."""
    from backend.agents.rlm.primitives import implement_baseline

    ctx = make_context(tmp_path)

    unblock = threading.Event()

    async def wedging_run(*args, **kwargs):
        unblock.wait()
        # Return a minimal object with commands_to_run
        from types import SimpleNamespace
        return SimpleNamespace(commands_to_run=["python train.py"])

    monkeypatch.setattr(primitives, "_run_baseline_with_sdk", wedging_run)

    from datetime import datetime, timezone, timedelta
    ctx.deadline_utc = datetime.now(timezone.utc) - timedelta(seconds=60)

    plan = {
        "paper_claim_map": {"core_contribution": "test"},
        "environment_spec": {},
    }

    start = time.monotonic()
    result = implement_baseline(plan, ctx=ctx)
    elapsed = time.monotonic() - start

    unblock.set()

    assert isinstance(result, dict)
    assert result.get("success") is False
    assert "timed out" in result.get("error", "")
    assert elapsed < 4.0, (
        f"implement_baseline blocked for {elapsed:.2f} s — "
        "pool.shutdown(wait=True) likely still in effect"
    )
