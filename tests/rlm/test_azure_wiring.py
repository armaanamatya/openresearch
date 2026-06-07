"""
Regression + wiring tests for --sandbox azure routing.

Guards (a) local/runpod/docker still use gpu_cell_runner unchanged,
(b) azure uses k8s_job_cell_runner with identical kwargs,
(c) SandboxMode.azure and ensure_sandbox_mode_available wiring,
(d) build_environment azure short-circuit returns no-op success.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from backend.agents.execution import SandboxMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(sandbox_mode: SandboxMode) -> SimpleNamespace:
    """Minimal RunContext duck-type for _execute_cell_matrix tests."""
    return SimpleNamespace(
        sandbox_mode=sandbox_mode,
        run_budget=None,
        _event_sink=None,
        gpu_device_ids=None,
        gpu_plan=None,
    )


# ---------------------------------------------------------------------------
# (c) SandboxMode.azure exists
# ---------------------------------------------------------------------------

def test_sandbox_mode_azure_member_exists():
    """SandboxMode.azure must be a valid enum member."""
    assert SandboxMode.azure is SandboxMode("azure")
    assert SandboxMode.azure.value == "azure"


def test_ensure_sandbox_mode_available_azure_calls_ensure_azure_available():
    """ensure_sandbox_mode_available('azure') must call ensure_azure_available."""
    with patch("backend.services.runtime.ensure_azure_available") as mock_ensure:
        from backend.agents.execution import ensure_sandbox_mode_available

        ensure_sandbox_mode_available(SandboxMode.azure)
        mock_ensure.assert_called_once()


def test_ensure_sandbox_mode_available_runpod_not_affected(monkeypatch):
    """Regression: runpod path still calls ensure_runpod_available, not ensure_azure_available."""
    monkeypatch.setenv("REPROLAB_RUNPOD_API_KEY", "fake-key")
    with (
        patch("backend.services.runtime.ensure_runpod_available") as mock_runpod,
        patch("backend.services.runtime.ensure_azure_available") as mock_azure,
    ):
        from backend.agents.execution import ensure_sandbox_mode_available

        ensure_sandbox_mode_available(SandboxMode.runpod)
        mock_runpod.assert_called_once()
        mock_azure.assert_not_called()


# ---------------------------------------------------------------------------
# (d) build_environment azure short-circuit
# ---------------------------------------------------------------------------

def test_build_environment_azure_returns_noop_success():
    """build_environment with sandbox_mode=azure must return ok=True, skipped=True
    without touching any docker client."""
    from backend.agents.rlm.primitives import build_environment

    ctx = _make_ctx(SandboxMode.azure)
    result = build_environment({"dockerfile": "FROM python:3.11"}, ctx=ctx)

    assert result.get("ok") is True, f"Expected ok=True, got {result}"
    assert result.get("skipped") is True, f"Expected skipped=True, got {result}"
    assert result.get("image_tag") == "", f"Expected image_tag='', got {result}"
    assert "azure" in result.get("note", "").lower(), (
        f"Note should mention azure, got: {result.get('note')}"
    )


def test_build_environment_local_short_circuit_unchanged():
    """Regression: local short-circuit is NOT broken by the azure branch."""
    from backend.agents.rlm.primitives import build_environment

    ctx = _make_ctx(SandboxMode.local)
    result = build_environment({"dockerfile": "FROM python:3.11"}, ctx=ctx)

    assert result.get("ok") is True
    assert result.get("skipped") is True
    assert "local" in result.get("note", "").lower()


# ---------------------------------------------------------------------------
# (a) Regression: local/runpod/docker use gpu_cell_runner unchanged
# (b) Azure uses k8s_job_cell_runner with identical kwargs
# ---------------------------------------------------------------------------

def _build_cells_json_env(tmp_path):
    """Write a minimal cells.json + train_cell.py so _execute_cell_matrix can parse them."""
    import json

    code = tmp_path / "code"
    code.mkdir()
    cells = [{"id": "c1", "model": "m1", "env": "e1", "baseline": "b1",
               "est_vram_gb": 4, "dataset_url": ""}]
    (code / "cells.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    (code / "train_cell.py").write_text("# stub", encoding="utf-8")
    return code


def _make_gpu_capacity_stub(vram: float = 80.0, n: int = 1):
    from backend.services.runtime.gpu_capacity import GpuCapacity

    return GpuCapacity(
        "azure", num_gpus=n, per_gpu_vram_gb=vram,
        free_gpu_ids=tuple(str(i) for i in range(n)),
        can_escalate=False,
    )


def _stub_cell_matrix_helpers():
    """Return monkeypatches that make the cell_matrix helpers harmless stubs."""
    return {
        "backend.agents.rlm.cell_matrix.capacity_gate": lambda cells, *a, **kw: (cells, [], []),
        "backend.agents.rlm.cell_matrix.dataset_url_preflight": lambda cells, **kw: (cells, [], []),
        "backend.agents.rlm.cell_matrix.aggregate_cell_metrics": lambda *a, **kw: {},
        "backend.agents.rlm.cell_fingerprint.compute_fingerprint": lambda *a, **kw: "fp",
    }


@pytest.mark.parametrize("mode", [SandboxMode.local, SandboxMode.docker, SandboxMode.runpod])
def test_non_azure_modes_use_gpu_cell_runner(tmp_path, mode):
    """Regression guard: local/docker/runpod must call gpu_cell_runner.run_matrix,
    NOT k8s_job_cell_runner.run_matrix.

    Because these imports are lazy (inside _execute_cell_matrix), we patch the
    module objects in backend.agents.rlm directly.
    """
    from backend.agents.rlm import gpu_cell_runner as _gpu, k8s_job_cell_runner as _k8s

    code = _build_cells_json_env(tmp_path)
    ctx = _make_ctx(mode)
    caps = _make_gpu_capacity_stub()

    mock_gpu = MagicMock(return_value={"c1": {"status": "ok", "metrics": {}}})
    mock_k8s = MagicMock(return_value={"c1": {"status": "ok", "metrics": {}}})

    with (
        patch.object(_gpu, "run_matrix", mock_gpu),
        patch.object(_k8s, "run_matrix", mock_k8s),
        patch("backend.agents.rlm.primitives._apply_operator_scope", lambda m, _: m),
        patch("backend.agents.rlm.primitives._summarize_cell_logs", return_value=""),
        patch("backend.agents.rlm.primitives._emit_dashboard_event"),
        patch("backend.agents.rlm.cell_matrix.capacity_gate",
              lambda cells, *a, **kw: (cells, [], [])),
        patch("backend.agents.rlm.cell_matrix.dataset_url_preflight",
              lambda cells, **kw: (cells, [], [])),
        patch("backend.agents.rlm.cell_matrix.aggregate_cell_metrics",
              lambda *a, **kw: {}),
        patch("backend.agents.rlm.cell_fingerprint.compute_fingerprint",
              lambda *a, **kw: "fp"),
    ):
        from backend.agents.rlm.primitives import _execute_cell_matrix
        _execute_cell_matrix(ctx, str(code), caps, timeout_s=60.0, run_id="r1")

    mock_gpu.assert_called_once()
    mock_k8s.assert_not_called()


def test_azure_mode_uses_k8s_runner_with_identical_kwargs(tmp_path):
    """azure must call k8s_job_cell_runner.run_matrix with the same kwargs shape
    (not gpu_cell_runner) — identical args minus the runner name."""
    from backend.agents.rlm import gpu_cell_runner as _gpu, k8s_job_cell_runner as _k8s

    code = _build_cells_json_env(tmp_path)
    ctx = _make_ctx(SandboxMode.azure)
    caps = _make_gpu_capacity_stub()

    mock_gpu = MagicMock(return_value={})
    mock_k8s = MagicMock(return_value={"c1": {"status": "ok", "metrics": {}}})

    with (
        patch.object(_gpu, "run_matrix", mock_gpu),
        patch.object(_k8s, "run_matrix", mock_k8s),
        patch("backend.agents.rlm.primitives._apply_operator_scope", lambda m, _: m),
        patch("backend.agents.rlm.primitives._summarize_cell_logs", return_value=""),
        patch("backend.agents.rlm.primitives._emit_dashboard_event"),
        patch("backend.agents.rlm.cell_matrix.capacity_gate",
              lambda cells, *a, **kw: (cells, [], [])),
        patch("backend.agents.rlm.cell_matrix.dataset_url_preflight",
              lambda cells, **kw: (cells, [], [])),
        patch("backend.agents.rlm.cell_matrix.aggregate_cell_metrics",
              lambda *a, **kw: {}),
        patch("backend.agents.rlm.cell_fingerprint.compute_fingerprint",
              lambda *a, **kw: "fp"),
    ):
        from backend.agents.rlm.primitives import _execute_cell_matrix
        _execute_cell_matrix(ctx, str(code), caps, timeout_s=60.0, run_id="r1")

    mock_k8s.assert_called_once()
    mock_gpu.assert_not_called()

    # Verify the key kwargs are present and correct
    _, kw = mock_k8s.call_args
    assert "output_root" in kw
    assert "per_cell_timeout_s" in kw
    assert kw["per_cell_timeout_s"] == pytest.approx(60.0)
    assert "overall_timeout_s" in kw
    assert "gpus_per_cell" in kw
    assert "fingerprints" in kw
    assert "force_cells" in kw
    assert "now_iso" in kw


# ---------------------------------------------------------------------------
# (a+b) _backend_for_sandbox_mode azure branch
# ---------------------------------------------------------------------------

def test_backend_for_sandbox_mode_azure_returns_aks_backend():
    """sandbox_mode='azure' must construct AksJobBackend, not LocalDockerBackend."""
    with patch("backend.services.runtime.ensure_azure_available", lambda: None):
        from backend.agents.rlm.primitives import _backend_for_sandbox_mode
        from backend.services.runtime.aks_job_backend import AksJobBackend

        backend = _backend_for_sandbox_mode(SandboxMode.azure, run_budget=None)
        assert isinstance(backend, AksJobBackend), (
            f"Expected AksJobBackend, got {type(backend).__name__}"
        )


def test_backend_for_sandbox_mode_local_unchanged():
    """Regression: local still returns LocalProcessBackend."""
    from backend.agents.rlm.primitives import _backend_for_sandbox_mode
    from backend.services.runtime.local_process import LocalProcessBackend

    backend = _backend_for_sandbox_mode(SandboxMode.local, run_budget=None)
    assert isinstance(backend, LocalProcessBackend)


def test_backend_for_sandbox_mode_docker_unchanged():
    """Regression: docker still returns LocalDockerBackend."""
    from backend.agents.rlm.primitives import _backend_for_sandbox_mode
    from backend.services.runtime.local_docker import LocalDockerBackend

    backend = _backend_for_sandbox_mode(SandboxMode.docker, run_budget=None)
    assert isinstance(backend, LocalDockerBackend)
