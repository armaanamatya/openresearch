from __future__ import annotations


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


def test_default_values_match_spec(monkeypatch):
    # A pytest plugin (deepeval) calls load_dotenv() on the repo .env, leaking
    # its values into os.environ for the whole session — which shadows the code
    # defaults this test verifies (e.g. .env sets OPENRESEARCH_DYNAMIC_GPU_ENABLED=
    # false, and env > .env > default). Clear the asserted vars from os.environ
    # AND disable the .env file so we assert the true Field defaults, hermetically.
    for _k in (
        "OPENRESEARCH_DYNAMIC_GPU_ENABLED", "OPENRESEARCH_FORCE_SINGLE_GPU",
        "OPENRESEARCH_MAX_GPU_USD_PER_HOUR", "OPENRESEARCH_MAX_RUN_GPU_USD",
        "OPENRESEARCH_DYNAMIC_GPU_HEADROOM", "OPENRESEARCH_DYNAMIC_GPU_FALLBACK_VRAM_GB",
        "OPENRESEARCH_DYNAMIC_GPU_MAX_ESCALATIONS",
    ):
        monkeypatch.delenv(_k, raising=False)
    s = Settings(_env_file=None)
    assert s.dynamic_gpu_enabled is True
    assert s.force_single_gpu is True
    assert s.max_gpu_usd_per_hour == pytest.approx(10.0)
    assert s.max_run_gpu_usd == pytest.approx(10.0)
    assert s.dynamic_gpu_headroom == pytest.approx(1.25)
    assert s.dynamic_gpu_fallback_vram_gb == 24
    assert s.dynamic_gpu_max_escalations == 2


def test_env_var_overrides(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_DYNAMIC_GPU_ENABLED", "false")
    monkeypatch.setenv("OPENRESEARCH_FORCE_SINGLE_GPU", "false")
    monkeypatch.setenv("OPENRESEARCH_MAX_GPU_USD_PER_HOUR", "2.5")
    monkeypatch.setenv("OPENRESEARCH_MAX_RUN_GPU_USD", "3.0")
    monkeypatch.setenv("OPENRESEARCH_DYNAMIC_GPU_HEADROOM", "1.5")
    monkeypatch.setenv("OPENRESEARCH_DYNAMIC_GPU_FALLBACK_VRAM_GB", "40")
    monkeypatch.setenv("OPENRESEARCH_DYNAMIC_GPU_MAX_ESCALATIONS", "3")
    s = Settings()
    assert s.dynamic_gpu_enabled is False
    assert s.force_single_gpu is False
    assert s.max_gpu_usd_per_hour == pytest.approx(2.5)
    assert s.max_run_gpu_usd == pytest.approx(3.0)
    assert s.dynamic_gpu_headroom == pytest.approx(1.5)
    assert s.dynamic_gpu_fallback_vram_gb == 40
    assert s.dynamic_gpu_max_escalations == 3


def test_empty_cost_caps_treated_as_no_cap(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_MAX_GPU_USD_PER_HOUR", "0")
    monkeypatch.setenv("OPENRESEARCH_MAX_RUN_GPU_USD", "0")
    s = Settings()
    # 0 is allowed numerically; consumer treats 0 as "no cap" (see resolver test).
    assert s.max_gpu_usd_per_hour == 0.0
    # Spec says 0 = no cap; implementation uses 0.0 as the no-cap sentinel.
    assert s.max_run_gpu_usd == 0.0


def test_default_runpod_image_has_cuda_dev_headers():
    # The earlier swap to -runtime- (Lane D) broke pip installs that need to
    # JIT-compile against CUDA dev headers (bitsandbytes, flash-attn, deepspeed).
    # Reverted to -devel- after the SDAR run silently failed on a chained
    # `pip install bitsandbytes && python train.py`. Override via the env var
    # when a paper genuinely doesn't need dev headers.
    #
    # Assert the class DEFAULT, not Settings() — the latter picks up a local
    # .env OPENRESEARCH_RUNPOD_IMAGE override, which makes this host-dependent.
    default_image = Settings.model_fields["runpod_image"].default
    assert "devel" in default_image and "runtime" not in default_image
