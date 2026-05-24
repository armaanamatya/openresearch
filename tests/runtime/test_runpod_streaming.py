"""Unit tests for RunpodBackend.exec streaming behaviour.

Mirror of LocalDockerBackend.exec's exec.log streaming: every line written
to stdout/stderr on the remote pod is appended to ``<artifact_root>/exec.log``
as it arrives, so a wedged remote process (NCCL deadlock, dataset download
stuck, matplotlib OOM) leaves a readable host file even when the
``asyncio.wait_for`` timeout fires.  Without this, a buffered SSH command
that never returns gives zero visibility into what the pod was doing —
exactly the 50-minute "stuck on L40S" symptom.

These tests are pure unit tests — no real SSH or network.  The asyncssh
process / readline surface is fully mocked.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from backend.services.runtime.interface import (
    RuntimeCauseKind,
    Sandbox,
    SandboxConfig,
)
from backend.services.runtime.runpod_backend import RunpodBackend, _RunpodConnection


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_sandbox(tmp_path: Path) -> Sandbox:
    config = SandboxConfig(
        project_id="proj",
        run_id="run",
        project_root=tmp_path,
        artifact_root=tmp_path / "artifacts",
    )
    return Sandbox(
        sandbox_id="test-pod",
        name="test-pod",
        image="test-image",
        config=config,
        created_at=datetime(2026, 5, 24, 12, 0, 0, tzinfo=timezone.utc),
    )


class _FakeReader:
    """Mock SSHReader: hands out queued lines on readline(), then EOF."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._closed = False

    async def push(self, line: str | None) -> None:
        await self._queue.put(line)

    async def readline(self) -> str:
        if self._closed:
            return ""
        item = await self._queue.get()
        if item is None:  # EOF sentinel
            self._closed = True
            return ""
        return item

    def close(self) -> None:
        self._closed = True


class _FakeProcess:
    """Mock asyncssh SSHClientProcess.

    Behaviour is driven by ``mode``:
      - "fast"    — emits stdout lines, then exits cleanly with returncode=0.
      - "hang"    — emits a few lines, then blocks forever on process.wait().
                    Used to force the asyncio.wait_for timeout path.
      - "failed"  — emits stdout + stderr, then exits with returncode=2.
    """

    def __init__(self, *, mode: str, lines: list[tuple[str, str]] | None = None) -> None:
        self.stdout = _FakeReader()
        self.stderr = _FakeReader()
        self.exit_status: int | None = None
        self._mode = mode
        self._lines = lines or [("stdout", "hello\n"), ("stdout", "world\n")]
        self._wait_event = asyncio.Event()
        self.terminate_called = False
        self.close_called = False
        # Kick off the line-emitting task right away so readers see lines
        # before wait() returns.
        self._driver = asyncio.create_task(self._drive())

    async def _drive(self) -> None:
        for stream, text in self._lines:
            if stream == "stdout":
                await self.stdout.push(text)
            else:
                await self.stderr.push(text)
            # Yield to the event loop so the drain task can process this line
            # before the next one arrives — that's what proves "incremental".
            await asyncio.sleep(0.01)
        if self._mode == "fast":
            await self.stdout.push(None)
            await self.stderr.push(None)
            self.exit_status = 0
            self._wait_event.set()
        elif self._mode == "failed":
            await self.stdout.push(None)
            await self.stderr.push(None)
            self.exit_status = 2
            self._wait_event.set()
        # mode == "hang": never push EOF, never set the wait event.

    async def wait(self) -> int | None:
        await self._wait_event.wait()
        return self.exit_status

    def terminate(self) -> None:
        self.terminate_called = True
        # Simulate the channel closing: push EOF so drain tasks exit.
        self.stdout.close()
        self.stderr.close()
        try:
            self.stdout._queue.put_nowait(None)
        except Exception:
            pass
        try:
            self.stderr._queue.put_nowait(None)
        except Exception:
            pass
        # Unblock wait() so the awaiter doesn't deadlock.
        if not self._wait_event.is_set():
            self._wait_event.set()

    def close(self) -> None:
        self.close_called = True


class _FakeSSHConnection:
    """Mock asyncssh SSHClientConnection exposing only what RunpodBackend uses."""

    def __init__(self, process: _FakeProcess) -> None:
        self._process = process
        self.created_with: str | None = None

    async def create_process(self, command: str) -> _FakeProcess:
        self.created_with = command
        return self._process

    def is_closed(self) -> bool:
        return False


def _build_backend(sandbox: Sandbox, conn: _FakeSSHConnection) -> RunpodBackend:
    backend = RunpodBackend(api_key="dummy")
    backend._ssh_clients[sandbox.sandbox_id] = conn
    backend._connections[sandbox.sandbox_id] = _RunpodConnection(
        pod_id=sandbox.sandbox_id,
        public_ip="1.2.3.4",
        ssh_port=22,
        remote_base="/workspace",
        remote_workdir="/workspace/work",
        remote_artifacts_dir="/workspace/artifacts",
    )
    backend._owned_pod_ids = {sandbox.sandbox_id}

    # Stub the artifact sync so we don't try to SFTP-walk a fake pod.
    async def _noop_sync(_sandbox: Sandbox) -> None:
        return None

    backend._sync_artifacts_to_host = _noop_sync  # type: ignore[assignment]
    backend._sync_artifacts_to_host_quietly = _noop_sync  # type: ignore[assignment]
    return backend


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_writes_log_file_line_by_line_during_execution(tmp_path):
    """Each stdout/stderr line MUST hit <artifact_root>/exec.log before wait() returns.

    This is the load-bearing guarantee. We assert by checking the file
    grows monotonically AS the process emits lines, not just after wait().
    """
    sandbox = _make_sandbox(tmp_path)
    artifact_root = sandbox.config.resolved_artifact_root()
    log_path = artifact_root / "exec.log"

    proc = _FakeProcess(
        mode="fast",
        lines=[
            ("stdout", "epoch 1/3\n"),
            ("stdout", "epoch 2/3\n"),
            ("stderr", "warning: low memory\n"),
            ("stdout", "epoch 3/3\n"),
        ],
    )
    conn = _FakeSSHConnection(proc)
    backend = _build_backend(sandbox, conn)

    # Sample log file size during execution. Spawn the exec call as a task
    # so we can poll the file while the process is still emitting.
    sizes: list[int] = []

    async def _sampler() -> None:
        # Wait until the file exists, then take ~10 samples while the proc
        # is running (proc.exit_status stays None until the last line emits).
        deadline = asyncio.get_running_loop().time() + 2.0
        while asyncio.get_running_loop().time() < deadline:
            if log_path.exists():
                sizes.append(log_path.stat().st_size)
                if proc.exit_status is not None:
                    return
            await asyncio.sleep(0.005)

    exec_task = asyncio.create_task(backend.exec(sandbox, "true", timeout=10))
    sampler_task = asyncio.create_task(_sampler())
    result = await exec_task
    await sampler_task

    # Clean success: returncode 0, no timeout flag, captured both streams.
    assert result.exit_code == 0
    assert result.timed_out is False
    assert "epoch 1/3" in result.stdout
    assert "epoch 2/3" in result.stdout
    assert "epoch 3/3" in result.stdout
    assert "warning: low memory" in result.stderr

    # Log file must exist, carry the command header, and contain every line.
    assert log_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert ">>> true" in log_text
    assert "epoch 1/3" in log_text
    assert "warning: low memory" in log_text
    assert "epoch 3/3" in log_text

    # The "incremental" assertion: at least two distinct file sizes were
    # observed during execution. Otherwise the file was written all at once
    # at the end (regression — that's the buffered behaviour we're fixing).
    distinct = sorted(set(sizes))
    assert len(distinct) >= 2, (
        f"exec.log was not written incrementally: only sizes {distinct} observed. "
        "If this is 1 element the streaming regressed to buffer-then-flush."
    )


@pytest.mark.asyncio
async def test_streaming_returns_captured_tail_on_timeout(tmp_path):
    """On forced timeout the ExecResult.stdout MUST contain the captured tail.

    This is the visibility guarantee: a wedged remote process leaves
    readable log evidence rather than an empty string.
    """
    sandbox = _make_sandbox(tmp_path)
    artifact_root = sandbox.config.resolved_artifact_root()
    log_path = artifact_root / "exec.log"

    proc = _FakeProcess(
        mode="hang",
        lines=[
            ("stdout", "loading dataset...\n"),
            ("stdout", "downloading shard 1/10\n"),
            ("stderr", "huggingface_hub: connection slow\n"),
        ],
    )
    conn = _FakeSSHConnection(proc)
    backend = _build_backend(sandbox, conn)

    # Tiny timeout — process emits 3 lines then hangs; wait_for fires fast.
    result = await backend.exec(sandbox, "python train.py", timeout=1)

    assert result.timed_out is True
    assert result.exit_code is None
    assert result.cause_kind == RuntimeCauseKind.exec_timeout
    # The load-bearing assertion: stdout is NOT empty — it has the captured tail.
    assert result.stdout, "Timed-out exec returned empty stdout — visibility regression."
    assert "loading dataset" in result.stdout or "shard 1/10" in result.stdout
    # The log file was written before the hang and is still on disk.
    assert log_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert "loading dataset" in log_text
    assert "huggingface_hub: connection slow" in log_text
    # The remote process was asked to terminate.
    assert proc.terminate_called is True


@pytest.mark.asyncio
async def test_streaming_clean_success_matches_log_file(tmp_path):
    """On clean exit, the returned stdout/stderr match the on-disk log file content.

    Sanity check: no data loss between memory chunks and the host file.
    """
    sandbox = _make_sandbox(tmp_path)
    log_path = sandbox.config.resolved_artifact_root() / "exec.log"

    proc = _FakeProcess(
        mode="fast",
        lines=[
            ("stdout", "alpha\n"),
            ("stdout", "beta\n"),
            ("stderr", "warn: deprecated flag\n"),
            ("stdout", "gamma\n"),
        ],
    )
    conn = _FakeSSHConnection(proc)
    backend = _build_backend(sandbox, conn)

    result = await backend.exec(sandbox, "echo run", timeout=5)
    assert result.exit_code == 0
    assert result.timed_out is False
    log_text = log_path.read_text(encoding="utf-8")
    # Every stdout line is in result.stdout and in the file.
    for token in ("alpha", "beta", "gamma"):
        assert token in result.stdout, f"missing {token!r} from stdout"
        assert token in log_text, f"missing {token!r} from log"
    # stderr too.
    assert "warn: deprecated flag" in result.stderr
    assert "warn: deprecated flag" in log_text


@pytest.mark.asyncio
async def test_streaming_nonzero_exit_marked_command_failed(tmp_path):
    """A clean wait() with non-zero exit code sets cause_kind=command_failed."""
    sandbox = _make_sandbox(tmp_path)
    proc = _FakeProcess(
        mode="failed",
        lines=[
            ("stdout", "starting...\n"),
            ("stderr", "ERROR: missing dependency\n"),
        ],
    )
    conn = _FakeSSHConnection(proc)
    backend = _build_backend(sandbox, conn)

    result = await backend.exec(sandbox, "pip check", timeout=5)
    assert result.exit_code == 2
    assert result.timed_out is False
    assert result.cause_kind == RuntimeCauseKind.command_failed
    assert "missing dependency" in result.stderr


@pytest.mark.asyncio
async def test_streaming_falls_back_when_create_process_missing(tmp_path):
    """A legacy/fake SSH connection without create_process uses the buffered path.

    The buffered path is what the old test doubles in tests/services/runtime/
    rely on, so we must not break it.
    """
    sandbox = _make_sandbox(tmp_path)

    class _BufferedOnlyConn:
        def __init__(self) -> None:
            self.run_called_with: str | None = None

        def is_closed(self) -> bool:
            return False

        async def run(self, command: str, check: bool = False) -> Any:
            self.run_called_with = command
            class _R:
                returncode = 0
                stdout = "buffered output\n"
                stderr = ""
            return _R()

    conn = _BufferedOnlyConn()
    backend = _build_backend(sandbox, conn)

    result = await backend.exec(sandbox, "echo legacy", timeout=5)
    assert result.exit_code == 0
    assert "buffered output" in result.stdout
    assert conn.run_called_with is not None
    assert "echo legacy" in conn.run_called_with


@pytest.mark.asyncio
async def test_streaming_continues_when_log_file_open_fails(tmp_path, monkeypatch):
    """If <artifact_root>/exec.log cannot be opened, exec MUST still succeed.

    Fail-soft contract: an IO error on the log file falls back to in-memory
    buffering rather than crashing the exec call.
    """
    sandbox = _make_sandbox(tmp_path)
    # Pre-create the artifact dir as a FILE (not a directory) so .open("ab")
    # on a child path will OSError out — that simulates a read-only / corrupt
    # artifact volume.
    artifact_root = sandbox.config.resolved_artifact_root()
    artifact_root.mkdir(parents=True, exist_ok=True)
    log_path = artifact_root / "exec.log"

    real_open = Path.open

    def _raising_open(self: Path, *args: Any, **kwargs: Any):
        if self == log_path:
            raise OSError("disk full")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _raising_open)

    proc = _FakeProcess(
        mode="fast",
        lines=[("stdout", "still works\n")],
    )
    conn = _FakeSSHConnection(proc)
    backend = _build_backend(sandbox, conn)

    result = await backend.exec(sandbox, "echo guarded", timeout=5)
    # exec did NOT crash on the OSError; the in-memory path still captured.
    assert result.exit_code == 0
    assert "still works" in result.stdout
