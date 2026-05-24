"""Tests for backend.services.runtime.gpu_resolution.

Three responsibilities pinned here:

  1. ``is_gpu_passthrough_mode`` — pure predicate, no side-effects.
  2. ``host_supports_nvidia_gpu`` — degrades gracefully when nvidia-smi is
     missing, broken, or empty. Caches its result.
  3. ``effective_gpu_mode`` + ``select_torch_index_url`` — compose 1 and 2
     so a ``--gpu-mode prefer`` request on a GPU-less host falls back to the
     CPU wheel instead of installing 2 GB of CUDA torch into a container
     that will never see a CUDA device.
"""

from __future__ import annotations

import subprocess

import pytest

from backend.services.runtime import gpu_resolution as gr


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    """Reset the nvidia-smi probe cache between tests."""
    gr._probe_nvidia_smi.cache_clear()
    try:
        yield
    finally:
        gr._probe_nvidia_smi.cache_clear()


# ---------------------------------------------------------------------------
# is_gpu_passthrough_mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["prefer", "max", "PREFER", "Max"])
def test_passthrough_modes_pass(mode: str) -> None:
    assert gr.is_gpu_passthrough_mode(mode) is True


@pytest.mark.parametrize("mode", ["off", "auto", "OFF", "Auto", "", None, "unknown"])
def test_non_passthrough_modes_fail(mode) -> None:
    assert gr.is_gpu_passthrough_mode(mode) is False


# ---------------------------------------------------------------------------
# host_supports_nvidia_gpu — the probe
# ---------------------------------------------------------------------------


class _FakeCompletedProcess:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_host_no_nvidia_smi_binary(monkeypatch) -> None:
    def _missing(*_args, **_kwargs):
        raise FileNotFoundError("nvidia-smi not on PATH")
    monkeypatch.setattr(gr.subprocess, "run", _missing)
    assert gr.host_supports_nvidia_gpu() is False


def test_host_nvidia_smi_timeout(monkeypatch) -> None:
    def _timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=3)
    monkeypatch.setattr(gr.subprocess, "run", _timeout)
    assert gr.host_supports_nvidia_gpu() is False


def test_host_nvidia_smi_nonzero_exit(monkeypatch) -> None:
    monkeypatch.setattr(
        gr.subprocess, "run",
        lambda *a, **k: _FakeCompletedProcess(returncode=1, stderr="No devices"),
    )
    assert gr.host_supports_nvidia_gpu() is False


def test_host_nvidia_smi_empty_output(monkeypatch) -> None:
    monkeypatch.setattr(
        gr.subprocess, "run",
        lambda *a, **k: _FakeCompletedProcess(returncode=0, stdout=""),
    )
    assert gr.host_supports_nvidia_gpu() is False


def test_host_has_one_gpu(monkeypatch) -> None:
    monkeypatch.setattr(
        gr.subprocess, "run",
        lambda *a, **k: _FakeCompletedProcess(
            returncode=0,
            stdout="GPU 0: NVIDIA GeForce RTX 2060 (UUID: GPU-abc123)\n",
        ),
    )
    assert gr.host_supports_nvidia_gpu() is True


def test_host_has_two_gpus(monkeypatch) -> None:
    monkeypatch.setattr(
        gr.subprocess, "run",
        lambda *a, **k: _FakeCompletedProcess(
            returncode=0,
            stdout=(
                "GPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-abc)\n"
                "GPU 1: NVIDIA A100-SXM4-80GB (UUID: GPU-def)\n"
            ),
        ),
    )
    assert gr.host_supports_nvidia_gpu() is True


def test_probe_is_cached(monkeypatch) -> None:
    calls = {"n": 0}

    def _counting(*_args, **_kwargs):
        calls["n"] += 1
        return _FakeCompletedProcess(returncode=0, stdout="GPU 0: ...\n")

    monkeypatch.setattr(gr.subprocess, "run", _counting)
    for _ in range(5):
        gr.host_supports_nvidia_gpu()
    assert calls["n"] == 1, "nvidia-smi must be probed only once per process"


# ---------------------------------------------------------------------------
# effective_gpu_mode — composition
# ---------------------------------------------------------------------------


def test_effective_mode_passthrough_on_gpu_host(monkeypatch) -> None:
    monkeypatch.setattr(
        gr.subprocess, "run",
        lambda *a, **k: _FakeCompletedProcess(returncode=0, stdout="GPU 0: ...\n"),
    )
    assert gr.effective_gpu_mode("prefer") == "prefer"
    assert gr.effective_gpu_mode("max") == "max"


def test_effective_mode_downgrades_on_no_gpu(monkeypatch) -> None:
    monkeypatch.setattr(
        gr.subprocess, "run",
        lambda *a, **k: _FakeCompletedProcess(returncode=1),
    )
    assert gr.effective_gpu_mode("prefer") == "auto"
    assert gr.effective_gpu_mode("max") == "auto"


def test_effective_mode_passes_through_non_gpu_requests(monkeypatch) -> None:
    monkeypatch.setattr(
        gr.subprocess, "run",
        lambda *a, **k: _FakeCompletedProcess(returncode=0, stdout="GPU 0: ...\n"),
    )
    assert gr.effective_gpu_mode("off") == "off"
    assert gr.effective_gpu_mode("auto") == "auto"
    assert gr.effective_gpu_mode(None) == "auto"


# ---------------------------------------------------------------------------
# select_torch_index_url — end-to-end policy
# ---------------------------------------------------------------------------


def test_index_url_is_none_for_gpu_request_on_gpu_host(monkeypatch) -> None:
    monkeypatch.setattr(
        gr.subprocess, "run",
        lambda *a, **k: _FakeCompletedProcess(returncode=0, stdout="GPU 0: ...\n"),
    )
    assert gr.select_torch_index_url("prefer") is None
    assert gr.select_torch_index_url("max") is None


def test_index_url_falls_back_to_cpu_for_gpu_request_on_no_gpu_host(monkeypatch) -> None:
    """The headline guarantee: a --gpu-mode prefer run on a GPU-less host
    must NOT install CUDA torch into the container."""
    monkeypatch.setattr(
        gr.subprocess, "run",
        lambda *a, **k: _FakeCompletedProcess(returncode=1),
    )
    assert gr.select_torch_index_url("prefer") == gr.CPU_TORCH_INDEX_URL


def test_index_url_is_cpu_for_off_auto_none(monkeypatch) -> None:
    monkeypatch.setattr(
        gr.subprocess, "run",
        lambda *a, **k: _FakeCompletedProcess(returncode=0, stdout="GPU 0: ...\n"),
    )
    assert gr.select_torch_index_url("off") == gr.CPU_TORCH_INDEX_URL
    assert gr.select_torch_index_url("auto") == gr.CPU_TORCH_INDEX_URL
    assert gr.select_torch_index_url(None) == gr.CPU_TORCH_INDEX_URL
