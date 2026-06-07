"""
Regression + wiring tests for --sandbox azure routing.

Guards (a) local/runpod/docker still use gpu_cell_runner unchanged,
(b) azure uses k8s_job_cell_runner with identical kwargs,
(c) SandboxMode.azure and ensure_sandbox_mode_available wiring,
(d) build_environment azure short-circuit returns no-op success,
(e) _backend_for_sandbox_mode azure passes gpu_plan to AksJobBackend,
(f) resolve_gpu_requirements selects provider='azure' for azure mode,
(g) _execute_cell_matrix azure passes gpu_plan to bind_run_context.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from backend.agents.execution import SandboxMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_gpu_plan_dict(
    *,
    short_name: str = "azure_nc24ads_a100",
    runpod_id: str = "Standard_NC24ads_A100_v4",
    vram_gb: int = 80,
    gpu_count: int = 1,
    cloud_type: str = "ONDEMAND",
    sku_usd_per_hr: float = 3.67,
    source: str = "paper",
) -> dict:
    """Build a valid dict matching GpuPlan's schema (all required fields present)."""
    from datetime import datetime, timezone
    return {
        "short_name": short_name,
        "runpod_id": runpod_id,
        "vram_gb": vram_gb,
        "gpu_count": gpu_count,
        "cloud_type": cloud_type,
        "sku_usd_per_hr": sku_usd_per_hr,
        "total_usd_per_hr": sku_usd_per_hr * gpu_count,
        "container_disk_gb": 100,
        "volume_gb": 200,
        "source": source,
        "requirements": {
            "estimated_vram_gb": vram_gb,
            "confidence": 0.9,
            "reasoning": "test",
        },
        "resolved_at": datetime.now(timezone.utc).isoformat(),
    }


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
    from contextlib import contextmanager

    from backend.agents.rlm import gpu_cell_runner as _gpu, k8s_job_cell_runner as _k8s

    code = _build_cells_json_env(tmp_path)
    ctx = _make_ctx(SandboxMode.azure)
    caps = _make_gpu_capacity_stub()

    mock_gpu = MagicMock(return_value={})
    mock_k8s = MagicMock(return_value={"c1": {"status": "ok", "metrics": {}}})

    @contextmanager
    def _fake_bind(*, run_budget=None, event_sink=None, gpu_plan=None):
        yield

    with (
        patch.object(_gpu, "run_matrix", mock_gpu),
        patch.object(_k8s, "run_matrix", mock_k8s),
        patch.object(_k8s, "bind_run_context", _fake_bind),
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


# ---------------------------------------------------------------------------
# (e) _backend_for_sandbox_mode azure+gpu_plan threading
# ---------------------------------------------------------------------------

def test_backend_for_sandbox_mode_azure_threads_gpu_plan():
    """sandbox_mode='azure' with a gpu_plan must pass it into AksJobBackend._gpu_plan."""
    from types import SimpleNamespace

    from backend.agents.rlm.primitives import _backend_for_sandbox_mode
    from backend.services.runtime.aks_job_backend import AksJobBackend

    plan = SimpleNamespace(short_name="azure_nc24ads_a100", gpu_count=1)

    with patch("backend.services.runtime.ensure_azure_available", lambda: None):
        backend = _backend_for_sandbox_mode(SandboxMode.azure, run_budget=None, gpu_plan=plan)

    assert isinstance(backend, AksJobBackend)
    assert backend._gpu_plan is plan


def test_backend_for_sandbox_mode_azure_no_gpu_plan_still_works():
    """sandbox_mode='azure' with gpu_plan=None constructs AksJobBackend without error."""
    from backend.agents.rlm.primitives import _backend_for_sandbox_mode
    from backend.services.runtime.aks_job_backend import AksJobBackend

    with patch("backend.services.runtime.ensure_azure_available", lambda: None):
        backend = _backend_for_sandbox_mode(SandboxMode.azure, run_budget=None, gpu_plan=None)

    assert isinstance(backend, AksJobBackend)
    assert backend._gpu_plan is None


# ---------------------------------------------------------------------------
# (f) resolve_gpu_requirements provider selection
# ---------------------------------------------------------------------------

def test_resolve_gpu_requirements_azure_mode_calls_resolve_with_azure_provider(tmp_path, monkeypatch):
    """resolve_gpu_requirements with ctx.sandbox_mode=SandboxMode.azure must call
    gpu_resolver.resolve(..., provider='azure', cloud_types=('ONDEMAND',))."""
    from unittest.mock import MagicMock

    ctx = SimpleNamespace(
        sandbox_mode=SandboxMode.azure,
        project_dir=tmp_path,
        vram_override=None,
    )

    captured: list[dict] = []

    # Use SimpleNamespace as a fake GpuPlan — the resolver returns it, we just
    # check that resolve() was called with the right kwargs.
    fake_plan = SimpleNamespace(
        short_name="azure_nc24ads_a100",
        gpu_count=1,
        source="paper",
        model_dump=lambda mode=None: _make_valid_gpu_plan_dict(),
    )

    def _fake_resolve(req, *, provider="runpod", cloud_types=("COMMUNITY",), **kw):
        captured.append({"provider": provider, "cloud_types": cloud_types})
        return fake_plan

    monkeypatch.setattr("backend.services.runtime.gpu_resolver.resolve", _fake_resolve)
    # Patch get_settings to avoid real config I/O.
    fake_settings = MagicMock()
    fake_settings.dynamic_gpu_enabled = True
    fake_settings.force_single_gpu = True
    fake_settings.max_gpu_usd_per_hour = None
    fake_settings.dynamic_gpu_headroom = 1.25
    fake_settings.dynamic_gpu_fallback_vram_gb = 16
    fake_settings.runpod_cloud_type = "COMMUNITY"
    monkeypatch.setattr("backend.config.get_settings", lambda: fake_settings)

    from backend.agents.rlm.primitives import resolve_gpu_requirements
    from backend.agents.schemas import GpuRequirements

    req = GpuRequirements(estimated_vram_gb=40, confidence=0.9, reasoning="needs A100")
    resolve_gpu_requirements(req, ctx=ctx)

    assert len(captured) == 1, f"resolve was not called exactly once: {captured}"
    assert captured[0]["provider"] == "azure", f"Expected provider='azure', got {captured[0]['provider']!r}"
    assert captured[0]["cloud_types"] == ("ONDEMAND",), (
        f"Expected cloud_types=('ONDEMAND',), got {captured[0]['cloud_types']!r}"
    )


def test_resolve_gpu_requirements_runpod_mode_uses_runpod_provider(tmp_path, monkeypatch):
    """Regression: resolve_gpu_requirements with sandbox_mode=runpod must NOT pass provider='azure'."""
    from unittest.mock import MagicMock

    ctx = SimpleNamespace(
        sandbox_mode=SandboxMode.runpod,
        project_dir=tmp_path,
        vram_override=None,
    )

    captured: list[dict] = []

    fake_plan = SimpleNamespace(
        short_name="rtx4090",
        gpu_count=1,
        source="paper",
        model_dump=lambda mode=None: _make_valid_gpu_plan_dict(
            short_name="rtx4090", runpod_id="NVIDIA GeForce RTX 4090",
            vram_gb=24, cloud_type="COMMUNITY", sku_usd_per_hr=0.34,
        ),
    )

    def _fake_resolve(req, *, provider="runpod", cloud_types=("COMMUNITY",), **kw):
        captured.append({"provider": provider, "cloud_types": cloud_types})
        return fake_plan

    monkeypatch.setattr("backend.services.runtime.gpu_resolver.resolve", _fake_resolve)
    fake_settings = MagicMock()
    fake_settings.dynamic_gpu_enabled = True
    fake_settings.force_single_gpu = True
    fake_settings.max_gpu_usd_per_hour = None
    fake_settings.dynamic_gpu_headroom = 1.25
    fake_settings.dynamic_gpu_fallback_vram_gb = 16
    fake_settings.runpod_cloud_type = "COMMUNITY"
    monkeypatch.setattr("backend.config.get_settings", lambda: fake_settings)

    from backend.agents.rlm.primitives import resolve_gpu_requirements
    from backend.agents.schemas import GpuRequirements

    req = GpuRequirements(estimated_vram_gb=20, confidence=0.8, reasoning="small GPU")
    resolve_gpu_requirements(req, ctx=ctx)

    assert len(captured) == 1
    assert captured[0]["provider"] == "runpod", (
        f"Regression: expected provider='runpod' for runpod mode, got {captured[0]['provider']!r}"
    )
    assert "azure" not in str(captured[0]["cloud_types"]).lower(), (
        f"Regression: runpod path must not use ONDEMAND cloud_types, got {captured[0]['cloud_types']!r}"
    )


def test_resolve_gpu_requirements_local_mode_uses_runpod_provider(tmp_path, monkeypatch):
    """Regression: local sandbox_mode must use provider='runpod' (not azure)."""
    from unittest.mock import MagicMock

    ctx = SimpleNamespace(
        sandbox_mode=SandboxMode.local,
        project_dir=tmp_path,
        vram_override=None,
    )
    captured: list[dict] = []
    fake_plan = SimpleNamespace(
        short_name="rtx4090",
        gpu_count=1,
        source="paper",
        model_dump=lambda mode=None: _make_valid_gpu_plan_dict(
            short_name="rtx4090", runpod_id="NVIDIA GeForce RTX 4090",
            vram_gb=24, cloud_type="COMMUNITY", sku_usd_per_hr=0.34,
        ),
    )

    def _fake_resolve(req, *, provider="runpod", cloud_types=("COMMUNITY",), **kw):
        captured.append({"provider": provider})
        return fake_plan

    monkeypatch.setattr("backend.services.runtime.gpu_resolver.resolve", _fake_resolve)
    fake_settings = MagicMock()
    fake_settings.dynamic_gpu_enabled = True
    fake_settings.force_single_gpu = True
    fake_settings.max_gpu_usd_per_hour = None
    fake_settings.dynamic_gpu_headroom = 1.25
    fake_settings.dynamic_gpu_fallback_vram_gb = 16
    fake_settings.runpod_cloud_type = "COMMUNITY"
    monkeypatch.setattr("backend.config.get_settings", lambda: fake_settings)

    from backend.agents.rlm.primitives import resolve_gpu_requirements
    from backend.agents.schemas import GpuRequirements

    req = GpuRequirements(estimated_vram_gb=16, confidence=0.7, reasoning="local run")
    resolve_gpu_requirements(req, ctx=ctx)

    assert captured[0]["provider"] == "runpod"


# ---------------------------------------------------------------------------
# (g) _execute_cell_matrix azure branch: bind_run_context called with gpu_plan
# ---------------------------------------------------------------------------

def test_execute_cell_matrix_azure_passes_gpu_plan_to_bind_run_context(tmp_path, monkeypatch):
    """_execute_cell_matrix for azure must call k8s_job_cell_runner.bind_run_context
    with gpu_plan loaded from gpu_plan.json."""
    import json
    from contextlib import contextmanager
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from backend.agents.schemas import GpuPlan
    from backend.services.runtime.gpu_capacity import GpuCapacity

    # Write a gpu_plan.json into rlm_state/ (all required GpuPlan fields).
    rlm_state = tmp_path / "rlm_state"
    rlm_state.mkdir(parents=True)
    plan_dict = _make_valid_gpu_plan_dict(
        short_name="azure_nc24ads_a100",
        runpod_id="Standard_NC24ads_A100_v4",
        vram_gb=80,
        gpu_count=1,
        cloud_type="ONDEMAND",
        sku_usd_per_hr=3.67,
    )
    (rlm_state / "gpu_plan.json").write_text(json.dumps(plan_dict), encoding="utf-8")

    code = tmp_path / "code"
    code.mkdir()
    cells = [{"id": "c1", "model": "m1", "env": "e1", "baseline": "b1",
               "est_vram_gb": 4, "dataset_url": ""}]
    (code / "cells.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    (code / "train_cell.py").write_text("# stub", encoding="utf-8")

    ctx = SimpleNamespace(
        sandbox_mode=SandboxMode.azure,
        project_dir=tmp_path,
        run_budget=None,
        _event_sink=None,
        gpu_device_ids=None,
    )
    caps = GpuCapacity("azure", num_gpus=1, per_gpu_vram_gb=80.0,
                       free_gpu_ids=("0",), can_escalate=False)

    bind_kwargs_captured: list[dict] = []

    @contextmanager
    def _fake_bind_run_context(*, run_budget=None, event_sink=None, gpu_plan=None):
        bind_kwargs_captured.append({
            "run_budget": run_budget,
            "event_sink": event_sink,
            "gpu_plan": gpu_plan,
        })
        yield

    from backend.agents.rlm import k8s_job_cell_runner as _k8s

    mock_run_matrix = MagicMock(return_value={"c1": {"status": "ok", "metrics": {}}})

    with (
        patch.object(_k8s, "bind_run_context", _fake_bind_run_context),
        patch.object(_k8s, "run_matrix", mock_run_matrix),
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

    assert len(bind_kwargs_captured) == 1, "bind_run_context must be called exactly once"
    gpu_plan_passed = bind_kwargs_captured[0]["gpu_plan"]
    assert gpu_plan_passed is not None, "gpu_plan must not be None when gpu_plan.json exists"
    assert isinstance(gpu_plan_passed, GpuPlan), (
        f"Expected GpuPlan instance, got {type(gpu_plan_passed).__name__}"
    )
    assert gpu_plan_passed.short_name == "azure_nc24ads_a100"


def test_execute_cell_matrix_azure_passes_none_gpu_plan_when_no_file(tmp_path):
    """_execute_cell_matrix for azure passes gpu_plan=None when gpu_plan.json is absent."""
    import json
    from contextlib import contextmanager
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from backend.services.runtime.gpu_capacity import GpuCapacity

    # Do NOT create rlm_state/gpu_plan.json
    code = tmp_path / "code"
    code.mkdir()
    cells = [{"id": "c1", "model": "m1", "env": "e1", "baseline": "b1",
               "est_vram_gb": 4, "dataset_url": ""}]
    (code / "cells.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    (code / "train_cell.py").write_text("# stub", encoding="utf-8")

    ctx = SimpleNamespace(
        sandbox_mode=SandboxMode.azure,
        project_dir=tmp_path,
        run_budget=None,
        _event_sink=None,
        gpu_device_ids=None,
    )
    caps = GpuCapacity("azure", num_gpus=1, per_gpu_vram_gb=80.0,
                       free_gpu_ids=("0",), can_escalate=False)

    bind_kwargs_captured: list[dict] = []

    @contextmanager
    def _fake_bind_run_context(*, run_budget=None, event_sink=None, gpu_plan=None):
        bind_kwargs_captured.append({"gpu_plan": gpu_plan})
        yield

    from backend.agents.rlm import k8s_job_cell_runner as _k8s

    with (
        patch.object(_k8s, "bind_run_context", _fake_bind_run_context),
        patch.object(_k8s, "run_matrix", MagicMock(return_value={"c1": {"status": "ok", "metrics": {}}})),
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

    assert len(bind_kwargs_captured) == 1
    assert bind_kwargs_captured[0]["gpu_plan"] is None


def test_execute_cell_matrix_non_azure_no_gpu_plan_plumbing(tmp_path):
    """Regression: non-azure (_execute_cell_matrix) must use gpu_cell_runner with
    NO gpu_plan plumbing into bind_run_context."""
    import json
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from backend.agents.rlm import gpu_cell_runner as _gpu, k8s_job_cell_runner as _k8s
    from backend.services.runtime.gpu_capacity import GpuCapacity

    code = tmp_path / "code"
    code.mkdir()
    cells = [{"id": "c1", "model": "m1", "env": "e1", "baseline": "b1",
               "est_vram_gb": 4, "dataset_url": ""}]
    (code / "cells.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    (code / "train_cell.py").write_text("# stub", encoding="utf-8")

    ctx = SimpleNamespace(
        sandbox_mode=SandboxMode.local,
        project_dir=tmp_path,
        run_budget=None,
        _event_sink=None,
        gpu_device_ids=None,
    )
    caps = GpuCapacity("local", num_gpus=1, per_gpu_vram_gb=24.0,
                       free_gpu_ids=("0",), can_escalate=False)

    mock_gpu_run = MagicMock(return_value={"c1": {"status": "ok", "metrics": {}}})
    mock_k8s_bind = MagicMock()

    with (
        patch.object(_gpu, "run_matrix", mock_gpu_run),
        patch.object(_k8s, "bind_run_context", mock_k8s_bind),
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

    mock_gpu_run.assert_called_once()
    mock_k8s_bind.assert_not_called()
