"""C3 — OOM mitigation: advisory → enforced (2026-06-16).

REPROLAB_OOM_ENFORCE default OFF → only the advisory batch-scale is set and only
the base OOM signatures match (byte-for-byte today). ON → an enforced per-GPU
memory-fraction shim is injected + broadened OOM signatures match.
"""

from __future__ import annotations

import pytest

from backend.agents.rlm import gpu_cell_runner as gcr


@pytest.fixture(autouse=True)
def _clear(monkeypatch):
    monkeypatch.delenv("REPROLAB_OOM_ENFORCE", raising=False)


def test_base_signatures_always_match():
    assert gcr._is_oom("RuntimeError: CUDA out of memory.") is True
    assert gcr._is_oom("OutOfMemoryError: ...") is True


def test_extra_signatures_gated_by_flag(monkeypatch):
    msg = "RuntimeError: CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate"
    assert gcr._is_oom(msg) is False  # not a base signature → today
    monkeypatch.setenv("REPROLAB_OOM_ENFORCE", "1")
    assert gcr._is_oom(msg) is True  # broadened set matches when enforced


def test_non_oom_never_matches(monkeypatch):
    monkeypatch.setenv("REPROLAB_OOM_ENFORCE", "1")
    assert gcr._is_oom("ValueError: shape mismatch") is False


def test_inject_memcap_writes_shim_and_env(tmp_path):
    env: dict = {}
    gcr._inject_oom_memcap(env, tmp_path, 0.5)
    shim = tmp_path / "_oom_shim" / "sitecustomize.py"
    assert shim.exists()
    assert "set_per_process_memory_fraction" in shim.read_text(encoding="utf-8")
    assert env["REPROLAB_CELL_MEM_FRACTION"] == "0.5"
    assert str(tmp_path / "_oom_shim") in env["PYTHONPATH"]


def test_memcap_fraction_clamped(tmp_path):
    env: dict = {}
    gcr._inject_oom_memcap(env, tmp_path, 2.0)  # > 1.0 clamps to 1.0
    assert env["REPROLAB_CELL_MEM_FRACTION"] == "1.0"
