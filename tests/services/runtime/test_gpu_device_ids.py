"""Tests for gpu_device_ids wiring through SandboxConfig, _gpu_device_request,
and LocalProcessBackend.exec.

Three responsibilities pinned here:

  1. SandboxConfig.gpu_device_ids round-trips correctly through pydantic.
  2. _gpu_device_request builds the right device-request object for both
     explicit device IDs and the all-GPUs fallback.
  3. LocalProcessBackend.exec pins CUDA_VISIBLE_DEVICES from
     config.gpu_device_ids when it spawns the subprocess.

The async exec test wraps the coroutine with asyncio.run() inside a
sync test function — the pattern used throughout this repository (see
tests/rlm/test_sandbox_identity_authority.py, tests/rlm/test_run.py,
and tests/rlm/test_run_integration.py).  No pytest-asyncio marker is
needed or used.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from backend.services.runtime.interface import (
    Sandbox,
    SandboxConfig,
)
from backend.services.runtime.local_docker import _gpu_device_request
from backend.services.runtime.local_process import LocalProcessBackend


class _FakeStream:
    """A pipe that is at EOF immediately (the streaming drain loop reads b'')."""

    async def read(self, n: int = -1) -> bytes:
        return b""


# ---------------------------------------------------------------------------
# 1. SandboxConfig.gpu_device_ids round-trip
# ---------------------------------------------------------------------------


def test_sandbox_config_accepts_gpu_device_ids(tmp_path: Path) -> None:
    """SandboxConfig accepts a tuple of GPU UUIDs and round-trips them unchanged."""
    cfg = SandboxConfig(
        project_id="p",
        run_id="r",
        project_root=tmp_path,
        gpu_device_ids=("GPU-a", "GPU-b"),
    )
    assert cfg.gpu_device_ids == ("GPU-a", "GPU-b")


def test_sandbox_config_gpu_device_ids_default_empty(tmp_path: Path) -> None:
    """SandboxConfig.gpu_device_ids defaults to an empty tuple."""
    cfg = SandboxConfig(
        project_id="p",
        run_id="r",
        project_root=tmp_path,
    )
    assert cfg.gpu_device_ids == ()


def test_sandbox_config_single_device_id(tmp_path: Path) -> None:
    """A single-element tuple is preserved."""
    cfg = SandboxConfig(
        project_id="p",
        run_id="r",
        project_root=tmp_path,
        gpu_device_ids=("GPU-deadbeef-0000-0000-0000-000000000001",),
    )
    assert cfg.gpu_device_ids == ("GPU-deadbeef-0000-0000-0000-000000000001",)


# ---------------------------------------------------------------------------
# 2. _gpu_device_request
# ---------------------------------------------------------------------------


def test_gpu_device_request_with_explicit_ids() -> None:
    """_gpu_device_request with non-empty device_ids produces device_ids attribute
    and no count attribute (count must be None or absent)."""
    req = _gpu_device_request(("GPU-a",))
    # Works whether docker SDK is present (real DeviceRequest) or not (SimpleNamespace)
    assert hasattr(req, "device_ids"), "result must have a device_ids attribute"
    assert list(getattr(req, "device_ids")) == ["GPU-a"]
    # count must not be set to -1 when explicit IDs are provided
    count_val = getattr(req, "count", None)
    assert count_val != -1, (
        "count should be None/unset when explicit device IDs are given; "
        f"got count={count_val!r}"
    )


def test_gpu_device_request_empty_uses_count_minus_one() -> None:
    """_gpu_device_request with no device IDs produces count=-1 (all GPUs)."""
    req = _gpu_device_request(())
    assert hasattr(req, "count"), "result must have a count attribute"
    assert req.count == -1


def test_gpu_device_request_multiple_ids() -> None:
    """_gpu_device_request with multiple IDs preserves order."""
    req = _gpu_device_request(("GPU-a", "GPU-b", "GPU-c"))
    assert list(getattr(req, "device_ids")) == ["GPU-a", "GPU-b", "GPU-c"]


# ---------------------------------------------------------------------------
# 3. LocalProcessBackend.exec — CUDA_VISIBLE_DEVICES pinning
# ---------------------------------------------------------------------------


def _make_sandbox_with_gpus(
    tmp_path: Path,
    gpu_device_ids: tuple[str, ...],
) -> Sandbox:
    """Helper: build a Sandbox whose config has the given GPU device IDs."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = SandboxConfig(
        project_id="test-proj",
        run_id="test-run",
        project_root=tmp_path,
        gpu_device_ids=gpu_device_ids,
    )
    return Sandbox(
        sandbox_id="local-test-proj-test-run",
        name="local-test-proj-test-run",
        image="local-process",
        config=cfg,
        created_at=datetime(2026, 5, 29, 0, 0, 0, tzinfo=timezone.utc),
    )


def test_exec_sets_cuda_visible_devices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """LocalProcessBackend.exec must set CUDA_VISIBLE_DEVICES from gpu_device_ids.

    asyncio.create_subprocess_shell is monkeypatched to capture the env kwarg
    and return a fake process (stdout=b"", stderr=b"", returncode=0) without
    spawning a real subprocess.
    """
    captured_env: dict[str, str] = {}

    class _FakeProcess:
        returncode = 0
        pid = 4000000  # nonexistent pid; liveness polls (cpu/gpu) fail-soft

        def __init__(self):
            self.stdout = _FakeStream()
            self.stderr = _FakeStream()

        async def wait(self):
            return 0

        def kill(self):
            pass

    async def _fake_create_subprocess_shell(cmd, **kwargs):  # type: ignore[override]
        captured_env.update(kwargs.get("env") or {})
        return _FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_shell", _fake_create_subprocess_shell)

    sandbox = _make_sandbox_with_gpus(tmp_path, ("GPU-a", "GPU-b"))
    backend = LocalProcessBackend()

    asyncio.run(backend.exec(sandbox, "true", timeout=5))

    assert "CUDA_VISIBLE_DEVICES" in captured_env, (
        "exec must set CUDA_VISIBLE_DEVICES in the subprocess environment "
        "when gpu_device_ids is non-empty"
    )
    assert captured_env["CUDA_VISIBLE_DEVICES"] == "GPU-a,GPU-b", (
        f"Expected 'GPU-a,GPU-b' but got {captured_env['CUDA_VISIBLE_DEVICES']!r}"
    )


def test_exec_no_cuda_visible_when_no_device_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """LocalProcessBackend.exec must NOT force CUDA_VISIBLE_DEVICES when gpu_device_ids is empty.

    The test only verifies the code-path does not inject a forced value;
    it does not assert the env is completely absent (the host environment
    may already have CUDA_VISIBLE_DEVICES set, which is fine).
    """
    captured_env: dict[str, str] = {}
    initial_cvd = "host-value"

    class _FakeProcess:
        returncode = 0
        pid = 4000000  # nonexistent pid; liveness polls (cpu/gpu) fail-soft

        def __init__(self):
            self.stdout = _FakeStream()
            self.stderr = _FakeStream()

        async def wait(self):
            return 0

        def kill(self):
            pass

    async def _fake_create_subprocess_shell(cmd, **kwargs):
        captured_env.update(kwargs.get("env") or {})
        return _FakeProcess()

    # Seed the subprocess env to verify our code does not override it
    monkeypatch.setattr(asyncio, "create_subprocess_shell", _fake_create_subprocess_shell)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", initial_cvd)

    sandbox = _make_sandbox_with_gpus(tmp_path, ())  # no explicit device IDs
    backend = LocalProcessBackend()

    asyncio.run(backend.exec(sandbox, "true", timeout=5))

    # When gpu_device_ids is empty, we must NOT have overwritten CUDA_VISIBLE_DEVICES
    # with a forced value; whatever was in os.environ (initial_cvd) should still be
    # present (or it may be absent if the host env had it unset — either is acceptable).
    # The critical invariant is that the value was not changed to something new by our code.
    if "CUDA_VISIBLE_DEVICES" in captured_env:
        assert captured_env["CUDA_VISIBLE_DEVICES"] == initial_cvd, (
            "exec must not override CUDA_VISIBLE_DEVICES when gpu_device_ids is empty"
        )


def test_exec_sets_cuda_device_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """exec must also set CUDA_DEVICE_ORDER=PCI_BUS_ID when pinning GPU IDs."""
    captured_env: dict[str, str] = {}

    class _FakeProcess:
        returncode = 0
        pid = 4000000  # nonexistent pid; liveness polls (cpu/gpu) fail-soft

        def __init__(self):
            self.stdout = _FakeStream()
            self.stderr = _FakeStream()

        async def wait(self):
            return 0

        def kill(self):
            pass

    async def _fake_create_subprocess_shell(cmd, **kwargs):
        captured_env.update(kwargs.get("env") or {})
        return _FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_shell", _fake_create_subprocess_shell)

    sandbox = _make_sandbox_with_gpus(tmp_path, ("GPU-x",))
    backend = LocalProcessBackend()

    asyncio.run(backend.exec(sandbox, "true", timeout=5))

    assert captured_env.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID", (
        "exec must set CUDA_DEVICE_ORDER=PCI_BUS_ID alongside CUDA_VISIBLE_DEVICES"
    )
