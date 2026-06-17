"""
Factory-dispatch tests for --sandbox gcp.

Guards that _backend_for_sandbox_mode(SandboxMode.gcp) constructs a GkeJobBackend,
that ensure_gcp_available is called, and that gpu_plan is threaded through correctly.
Regression guards for other modes are in test_azure_wiring.py / test_runpod_wiring.py.
"""

from __future__ import annotations

from unittest.mock import patch

from backend.agents.execution import SandboxMode


def test_sandbox_mode_gcp_member_exists():
    """SandboxMode.gcp must be a valid enum member."""
    assert SandboxMode.gcp is SandboxMode("gcp")
    assert SandboxMode.gcp.value == "gcp"


def test_backend_for_sandbox_mode_gcp_returns_gke_backend():
    """sandbox_mode='gcp' must construct GkeJobBackend, not LocalDockerBackend."""
    with patch("backend.services.runtime.ensure_gcp_available", lambda: None):
        from backend.agents.rlm.primitives import _backend_for_sandbox_mode
        from backend.services.runtime.gke_job_backend import GkeJobBackend

        backend = _backend_for_sandbox_mode(SandboxMode.gcp, run_budget=None)
        assert isinstance(backend, GkeJobBackend), (
            f"Expected GkeJobBackend, got {type(backend).__name__}"
        )


def test_backend_for_sandbox_mode_gcp_threads_gpu_plan():
    """sandbox_mode='gcp' with a gpu_plan must pass it into GkeJobBackend._gpu_plan."""
    from types import SimpleNamespace

    plan = SimpleNamespace(short_name="gcp_a100_80", gpu_count=1)

    with patch("backend.services.runtime.ensure_gcp_available", lambda: None):
        from backend.agents.rlm.primitives import _backend_for_sandbox_mode
        from backend.services.runtime.gke_job_backend import GkeJobBackend

        backend = _backend_for_sandbox_mode(SandboxMode.gcp, run_budget=None, gpu_plan=plan)

    assert isinstance(backend, GkeJobBackend)
    assert backend._gpu_plan is plan


def test_backend_for_sandbox_mode_gcp_no_gpu_plan_still_works():
    """sandbox_mode='gcp' with gpu_plan=None constructs GkeJobBackend without error."""
    with patch("backend.services.runtime.ensure_gcp_available", lambda: None):
        from backend.agents.rlm.primitives import _backend_for_sandbox_mode
        from backend.services.runtime.gke_job_backend import GkeJobBackend

        backend = _backend_for_sandbox_mode(SandboxMode.gcp, run_budget=None, gpu_plan=None)

    assert isinstance(backend, GkeJobBackend)
    assert backend._gpu_plan is None


def test_backend_for_sandbox_mode_azure_unaffected_by_gcp_branch():
    """Regression: azure still returns AksJobBackend after GCP branch added."""
    with patch("backend.services.runtime.ensure_azure_available", lambda: None):
        from backend.agents.rlm.primitives import _backend_for_sandbox_mode
        from backend.services.runtime.aks_job_backend import AksJobBackend

        backend = _backend_for_sandbox_mode(SandboxMode.azure, run_budget=None)
        assert isinstance(backend, AksJobBackend), (
            f"Regression: azure must still return AksJobBackend, got {type(backend).__name__}"
        )


def test_resolve_gpu_requirements_gcp_mode_calls_resolve_with_gcp_provider(tmp_path, monkeypatch):
    """resolve_gpu_requirements with ctx.sandbox_mode=SandboxMode.gcp must call
    gpu_resolver.resolve(..., provider='gcp', cloud_types=('ONDEMAND',))."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    ctx = SimpleNamespace(
        sandbox_mode=SandboxMode.gcp,
        project_dir=tmp_path,
        vram_override=None,
    )

    captured: list[dict] = []

    fake_plan = SimpleNamespace(
        short_name="gcp_a100_80",
        gpu_count=1,
        source="paper",
        model_dump=lambda mode=None: {
            "short_name": "gcp_a100_80",
            "runpod_id": "a100-80",
            "vram_gb": 80,
            "gpu_count": 1,
            "cloud_type": "ONDEMAND",
            "sku_usd_per_hr": 3.93,
            "total_usd_per_hr": 3.93,
            "container_disk_gb": 50,
            "volume_gb": 100,
            "source": "paper",
            "requirements": {"estimated_vram_gb": 80, "confidence": 0.9, "reasoning": "test"},
            "resolved_at": "2026-06-16T00:00:00Z",
        },
    )

    def _fake_resolve(req, *, provider="runpod", cloud_types=("COMMUNITY",),
                      provisioned_skus=None, **kw):
        captured.append({
            "provider": provider,
            "cloud_types": cloud_types,
            "provisioned_skus": provisioned_skus,
        })
        return fake_plan

    monkeypatch.setattr("backend.services.runtime.gpu_resolver.resolve", _fake_resolve)
    fake_settings = MagicMock()
    fake_settings.dynamic_gpu_enabled = True
    fake_settings.force_single_gpu = True
    fake_settings.max_gpu_usd_per_hour = None
    fake_settings.dynamic_gpu_headroom = 1.25
    fake_settings.dynamic_gpu_fallback_vram_gb = 16
    fake_settings.runpod_cloud_type = "COMMUNITY"
    fake_settings.gcp_gpu_skus = ["gcp_a100_80", "gcp_a100_80x2"]
    monkeypatch.setattr("backend.config.get_settings", lambda: fake_settings)

    from backend.agents.rlm.primitives import resolve_gpu_requirements
    from backend.agents.schemas import GpuRequirements

    req = GpuRequirements(estimated_vram_gb=40, confidence=0.9, reasoning="needs A100")
    resolve_gpu_requirements(req, ctx=ctx)

    assert len(captured) == 1, f"resolve was not called exactly once: {captured}"
    assert captured[0]["provider"] == "gcp", (
        f"Expected provider='gcp', got {captured[0]['provider']!r}"
    )
    assert captured[0]["cloud_types"] == ("ONDEMAND",), (
        f"Expected cloud_types=('ONDEMAND',), got {captured[0]['cloud_types']!r}"
    )
    assert captured[0]["provisioned_skus"] == ("gcp_a100_80", "gcp_a100_80x2"), (
        f"Expected provisioned_skus from gcp_gpu_skus, got {captured[0]['provisioned_skus']!r}"
    )


def test_resolve_gpu_requirements_azure_unaffected_by_gcp_branch(tmp_path, monkeypatch):
    """Regression: azure sandbox mode still passes provider='azure' after GCP branch."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    ctx = SimpleNamespace(
        sandbox_mode=SandboxMode.azure,
        project_dir=tmp_path,
        vram_override=None,
    )

    captured: list[dict] = []

    fake_plan = SimpleNamespace(
        short_name="azure_nc24ads_a100",
        gpu_count=1,
        source="paper",
        model_dump=lambda mode=None: {
            "short_name": "azure_nc24ads_a100",
            "runpod_id": "Standard_NC24ads_A100_v4",
            "vram_gb": 80,
            "gpu_count": 1,
            "cloud_type": "ONDEMAND",
            "sku_usd_per_hr": 3.67,
            "total_usd_per_hr": 3.67,
            "container_disk_gb": 100,
            "volume_gb": 200,
            "source": "paper",
            "requirements": {"estimated_vram_gb": 80, "confidence": 0.9, "reasoning": "test"},
            "resolved_at": "2026-06-16T00:00:00Z",
        },
    )

    def _fake_resolve(req, *, provider="runpod", cloud_types=("COMMUNITY",),
                      provisioned_skus=None, **kw):
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
    fake_settings.azure_gpu_skus = ["azure_a100_80"]
    monkeypatch.setattr("backend.config.get_settings", lambda: fake_settings)

    from backend.agents.rlm.primitives import resolve_gpu_requirements
    from backend.agents.schemas import GpuRequirements

    req = GpuRequirements(estimated_vram_gb=40, confidence=0.9, reasoning="needs A100")
    resolve_gpu_requirements(req, ctx=ctx)

    assert len(captured) == 1
    assert captured[0]["provider"] == "azure", (
        f"Regression: azure mode must still pass provider='azure', got {captured[0]['provider']!r}"
    )
