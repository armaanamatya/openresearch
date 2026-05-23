from __future__ import annotations

import os

import pytest

from backend.config import Settings


def _reset_settings_cache():
    """Drop the cached Settings singleton so the next Settings() call rebuilds it."""
    import backend.config as _config
    _config._settings_cache = None


@pytest.fixture(autouse=True)
def _isolate_settings_cache():
    _reset_settings_cache()
    try:
        yield
    finally:
        _reset_settings_cache()


def test_default_values_match_spec():
    s = Settings()
    assert s.dynamic_gpu_enabled is True
    assert s.force_single_gpu is True
    assert s.max_gpu_usd_per_hour == pytest.approx(10.0)
    assert s.max_run_gpu_usd == pytest.approx(10.0)
    assert s.dynamic_gpu_headroom == pytest.approx(1.25)
    assert s.dynamic_gpu_fallback_vram_gb == 24
    assert s.dynamic_gpu_max_escalations == 2


def test_env_var_overrides(monkeypatch):
    monkeypatch.setenv("REPROLAB_DYNAMIC_GPU_ENABLED", "false")
    monkeypatch.setenv("REPROLAB_FORCE_SINGLE_GPU", "false")
    monkeypatch.setenv("REPROLAB_MAX_GPU_USD_PER_HOUR", "2.5")
    monkeypatch.setenv("REPROLAB_MAX_RUN_GPU_USD", "3.0")
    monkeypatch.setenv("REPROLAB_DYNAMIC_GPU_HEADROOM", "1.5")
    monkeypatch.setenv("REPROLAB_DYNAMIC_GPU_FALLBACK_VRAM_GB", "40")
    monkeypatch.setenv("REPROLAB_DYNAMIC_GPU_MAX_ESCALATIONS", "3")
    s = Settings()
    assert s.dynamic_gpu_enabled is False
    assert s.force_single_gpu is False
    assert s.max_gpu_usd_per_hour == pytest.approx(2.5)
    assert s.max_run_gpu_usd == pytest.approx(3.0)
    assert s.dynamic_gpu_headroom == pytest.approx(1.5)
    assert s.dynamic_gpu_fallback_vram_gb == 40
    assert s.dynamic_gpu_max_escalations == 3


def test_empty_cost_caps_treated_as_no_cap(monkeypatch):
    monkeypatch.setenv("REPROLAB_MAX_GPU_USD_PER_HOUR", "0")
    monkeypatch.setenv("REPROLAB_MAX_RUN_GPU_USD", "0")
    s = Settings()
    # 0 is allowed numerically; consumer treats 0 as "no cap" (see resolver test).
    assert s.max_gpu_usd_per_hour == 0.0
    # Spec says 0 = no cap; implementation uses 0.0 as the no-cap sentinel.
    assert s.max_run_gpu_usd == 0.0
