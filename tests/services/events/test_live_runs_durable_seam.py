"""Hermetic OFF+ON tests for the WS3 durable-controller submit seam.

Phase-3 (owner 3) lands only the *seam* in `live_runs.py`: a pure decision
predicate (`_should_use_durable_controller`) plus an injectable submit hook
(`_submit_durable_controller`) whose default body is a fail-loud
`NotImplementedError` stub. The real cluster submit (a GKE Deployment via
`run_controller.build_controller_command` + `acquire_drive_lease`, recording
the controller handle into `demo_status.json`) needs a live GKE cluster and
is operator/drill-gated — explicitly out of scope here (see
`.superpowers/sdd/phase3-owner3-live_runs-seam.md`).

These tests prove:
  - the predicate is pure and flag-gated (`OPENRESEARCH_DURABLE_CONTROLLER`);
  - flag OFF (default) is byte-identical to today: `_start_python_run` always
    falls through to the existing `subprocess.Popen` reproduce path — even
    for `sandbox="gcp"` — and never calls the controller hook;
  - flag ON + `sandbox="gcp"` short-circuits to the controller hook instead
    of spawning a local subprocess;
  - flag ON + any other sandbox still uses the local Popen path;
  - the unpatched hook itself raises `NotImplementedError`.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import subprocess
from pathlib import Path

import pytest

import backend.services.events.live_runs as lr_module
from backend.services.events.live_runs import FileLiveRunService, LiveRunState, StartRunRequest

# Matches the `_NONEXISTENT_PID` convention already established in
# tests/services/events/test_live_runs_watchdog.py: never a real pid, so the
# fire-and-forget `_stderr_watchdog` task started after a faked Popen call
# exits cleanly on its first liveness check rather than polling forever.
_FAKE_PID = 99999999

_ALL_SANDBOXES: tuple[str, ...] = ("gcp", "local", "runpod", "azure", "docker", "auto")


def _reset_settings_cache() -> None:
    import backend.config as _config

    _config._settings_cache = None


@pytest.fixture(autouse=True)
def _clean_flag(monkeypatch):
    """Every test starts from a known-clean flag/override state.

    `OPENRESEARCH_DURABLE_CONTROLLER` is unset by default (ON tests opt in
    explicitly); the unrelated force-sandbox/force-provider overrides are
    also cleared so an ambient dev-shell env var can't silently change which
    branch `_start_python_run` takes in these tests.
    """
    monkeypatch.delenv("OPENRESEARCH_DURABLE_CONTROLLER", raising=False)
    monkeypatch.delenv("OPENRESEARCH_FORCE_SANDBOX", raising=False)
    monkeypatch.delenv("OPENRESEARCH_FORCE_LLM_PROVIDER", raising=False)
    _reset_settings_cache()
    yield
    _reset_settings_cache()


class _FakeProcess:
    """Stand-in for `Popen`, including the credential-handoff stdin pipe."""

    def __init__(self, pid: int, *, stdin_is_pipe: bool) -> None:
        self.pid = pid
        self.stdin = io.BytesIO() if stdin_is_pipe else None


def _recording_popen(calls: list[dict]):
    """Build a fake `subprocess.Popen` that records its call and spawns nothing."""

    def _fake(argv, **kwargs):
        calls.append({"argv": argv, "kwargs": kwargs})
        return _FakeProcess(
            _FAKE_PID,
            stdin_is_pipe=kwargs.get("stdin") == subprocess.PIPE,
        )

    return _fake


async def _cancel_named_tasks(name: str) -> None:
    """Cancel+drain any pending task with the given name.

    The unpatched Popen path in `_start_python_run` fires off a background
    `_stderr_watchdog` task (`name=f"stderr-watchdog-{project_id}"`) that the
    caller never awaits. With `_FAKE_PID` it would self-exit on its first
    liveness check anyway, but tests should not leave asyncio tasks pending
    across the test boundary.
    """
    pending = [t for t in asyncio.all_tasks() if t.get_name() == name]
    for task in pending:
        task.cancel()
    for task in pending:
        with contextlib.suppress(asyncio.CancelledError):
            await task


# ---------------------------------------------------------------------------
# Predicate — pure, no I/O, no monkeypatching required.
# ---------------------------------------------------------------------------


def test_predicate_false_when_flag_unset_for_every_sandbox(tmp_path: Path) -> None:
    service = FileLiveRunService(runs_root=tmp_path)
    for sandbox in _ALL_SANDBOXES:
        request = StartRunRequest(sandbox=sandbox)
        assert service._should_use_durable_controller(request) is False, sandbox


def test_predicate_true_only_for_gcp_when_flag_on(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENRESEARCH_DURABLE_CONTROLLER", "1")
    service = FileLiveRunService(runs_root=tmp_path)

    assert service._should_use_durable_controller(StartRunRequest(sandbox="gcp")) is True
    for sandbox in ("local", "runpod", "azure", "docker", "auto"):
        request = StartRunRequest(sandbox=sandbox)
        assert service._should_use_durable_controller(request) is False, sandbox


# ---------------------------------------------------------------------------
# Real submit hook — injected fake cluster (the real GCS/K8s cluster is
# drill-exercised; here we drive the takeover-safe ordering + handle recording).
# ---------------------------------------------------------------------------


class _FakeToken:
    def __init__(self, fence_epoch=2):
        self.fence_epoch = fence_epoch
        self.generation = 1
        self.acquired_epoch = 100.0
        self.owner_id = "owner"


class _FakeCluster:
    def __init__(self, *, token=None, ready=True, delete_ok=True):
        self._token = token if token is not None else _FakeToken()
        self._ready = ready
        self._delete_ok = delete_ok

    def now(self):
        return 100.0

    def acquire(self, project_id, owner_id, now_epoch):
        return self._token

    def is_current(self, token):
        return True

    def submit(self, manifest):
        pass

    def wait_ready(self, job_name, *, timeout_s):
        return self._ready

    def delete_confirmed(self, job_name):
        return self._delete_ok

    def reap(self, project_id, token):
        return 0


@pytest.mark.asyncio
async def test_durable_submit_records_handle_via_start(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENRESEARCH_DURABLE_CONTROLLER", "1")
    service = FileLiveRunService(
        runs_root=tmp_path, repo_root=tmp_path, controller_cluster=_FakeCluster()
    )
    popen_calls: list[dict] = []
    monkeypatch.setattr(lr_module.subprocess, "Popen", _recording_popen(popen_calls))

    result = await service._start_python_run(
        StartRunRequest(sandbox="gcp"), project_id="prj_dur", uploaded_paper=None,
    )
    assert popen_calls == []  # durable path, no local subprocess
    assert result.controller is not None
    assert result.controller.jobName == "controller-prj_dur-fe2"
    assert result.controller.fenceEpoch == 2
    assert result.pid is None


@pytest.mark.asyncio
async def test_durable_not_ready_confirmed_delete_falls_back_to_popen(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("OPENRESEARCH_DURABLE_CONTROLLER", "1")
    service = FileLiveRunService(
        runs_root=tmp_path,
        repo_root=tmp_path,
        controller_cluster=_FakeCluster(ready=False, delete_ok=True),
    )
    popen_calls: list[dict] = []
    monkeypatch.setattr(lr_module.subprocess, "Popen", _recording_popen(popen_calls))

    project_id = "prj_notready"
    try:
        result = await service._start_python_run(
            StartRunRequest(sandbox="gcp"), project_id=project_id, uploaded_paper=None,
        )
    finally:
        await _cancel_named_tasks(f"stderr-watchdog-{project_id}")
    assert len(popen_calls) == 1  # confirmed-deleted → safe local fallback
    assert result.pid == _FAKE_PID


@pytest.mark.asyncio
async def test_durable_stuck_fails_loud_no_fallback(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENRESEARCH_DURABLE_CONTROLLER", "1")
    service = FileLiveRunService(
        runs_root=tmp_path,
        repo_root=tmp_path,
        controller_cluster=_FakeCluster(ready=False, delete_ok=False),
    )
    popen_calls: list[dict] = []
    monkeypatch.setattr(lr_module.subprocess, "Popen", _recording_popen(popen_calls))

    from backend.agents.rlm.controller_cluster import ControllerStuck

    with pytest.raises(ControllerStuck):
        await service._start_python_run(
            StartRunRequest(sandbox="gcp"), project_id="prj_stuck", uploaded_paper=None,
        )
    assert popen_calls == []  # never fall back while a remote Job may be live


# ---------------------------------------------------------------------------
# Branch OFF — flag unset: byte-identical to today. Even sandbox="gcp" falls
# through to the existing subprocess.Popen path; the controller hook is
# never invoked.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_python_run_flag_off_uses_popen_not_controller(
    tmp_path: Path, monkeypatch
) -> None:
    # repo_root=tmp_path (empty) => _prepare_source_artifacts falls back to
    # its synthetic-minimal-PDF path instead of copying the multi-MB repo
    # fixture, keeping this fully-driven test fast and independent of what
    # fixture files happen to exist at the real repo root.
    service = FileLiveRunService(runs_root=tmp_path, repo_root=tmp_path)

    popen_calls: list[dict] = []
    monkeypatch.setattr(lr_module.subprocess, "Popen", _recording_popen(popen_calls))

    controller_calls: list[str] = []

    async def _fake_controller(self, request, *, project_id, uploaded_paper):
        controller_calls.append(project_id)
        return LiveRunState(
            projectId=project_id, outputDir="unused", runMode=request.mode,
            status="queued", payload=None, log="",
        )

    monkeypatch.setattr(FileLiveRunService, "_submit_durable_controller", _fake_controller)

    project_id = "prj_off_test"
    request = StartRunRequest(sandbox="gcp")
    try:
        result = await service._start_python_run(
            request, project_id=project_id, uploaded_paper=None,
        )
    finally:
        await _cancel_named_tasks(f"stderr-watchdog-{project_id}")

    assert len(popen_calls) == 1
    assert controller_calls == []
    assert isinstance(result, LiveRunState)
    assert result.pid == _FAKE_PID


# ---------------------------------------------------------------------------
# Branch ON + sandbox="gcp" — the controller hook is invoked instead of the
# local subprocess; Popen is never called.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_python_run_flag_on_gcp_uses_controller_not_popen(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("OPENRESEARCH_DURABLE_CONTROLLER", "1")
    service = FileLiveRunService(runs_root=tmp_path, repo_root=tmp_path)

    popen_calls: list[dict] = []
    monkeypatch.setattr(lr_module.subprocess, "Popen", _recording_popen(popen_calls))

    controller_calls: list[dict] = []

    async def _fake_controller(self, request, *, project_id, uploaded_paper):
        controller_calls.append({"project_id": project_id, "sandbox": request.sandbox})
        return LiveRunState(
            projectId=project_id,
            outputDir=str(self.runs_root / project_id),
            runMode=request.mode,
            status="queued",
            payload=None,
            log="",
        )

    monkeypatch.setattr(FileLiveRunService, "_submit_durable_controller", _fake_controller)

    project_id = "prj_on_test"
    request = StartRunRequest(sandbox="gcp")
    result = await service._start_python_run(
        request, project_id=project_id, uploaded_paper=None,
    )

    assert controller_calls == [{"project_id": project_id, "sandbox": "gcp"}]
    assert popen_calls == []
    assert isinstance(result, LiveRunState)
    assert result.status == "queued"


# ---------------------------------------------------------------------------
# Branch ON but sandbox != "gcp" — the flag alone is not enough; still Popen.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_python_run_flag_on_non_gcp_still_uses_popen(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("OPENRESEARCH_DURABLE_CONTROLLER", "1")
    service = FileLiveRunService(runs_root=tmp_path, repo_root=tmp_path)

    popen_calls: list[dict] = []
    monkeypatch.setattr(lr_module.subprocess, "Popen", _recording_popen(popen_calls))

    controller_calls: list[str] = []

    async def _fake_controller(self, request, *, project_id, uploaded_paper):
        controller_calls.append(project_id)
        return LiveRunState(
            projectId=project_id, outputDir="unused", runMode=request.mode,
            status="queued", payload=None, log="",
        )

    monkeypatch.setattr(FileLiveRunService, "_submit_durable_controller", _fake_controller)

    project_id = "prj_on_local_test"
    request = StartRunRequest(sandbox="local")
    try:
        result = await service._start_python_run(
            request, project_id=project_id, uploaded_paper=None,
        )
    finally:
        await _cancel_named_tasks(f"stderr-watchdog-{project_id}")

    assert len(popen_calls) == 1
    assert controller_calls == []
    assert isinstance(result, LiveRunState)
