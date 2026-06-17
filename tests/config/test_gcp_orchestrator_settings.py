"""
Tests for the GCP orchestrator/secret-store settings added in Stream A (and
the CLAUDE_CODE_OAUTH_TOKEN field added in Stream B).

Verifies that:
- All new gcp_orchestrator_* / claude_code_oauth_token fields exist with
  documented defaults and round-trip via OPENRESEARCH_GCP_* / CLAUDE_CODE_*
  env vars.
- Existing gcp_* fields are untouched (no regression).
"""

from __future__ import annotations

import pytest

from backend.config import Settings


def _reset_settings_cache():
    import backend.config as _config
    if hasattr(_config, "_settings_cache"):
        _config._settings_cache = None


@pytest.fixture(autouse=True)
def _isolate_settings_cache():
    _reset_settings_cache()
    yield
    _reset_settings_cache()


_NEW_GCP_ENVVARS = [
    "OPENRESEARCH_GCP_ORCHESTRATOR_IMAGE",
    "OPENRESEARCH_GCP_CSI_MOUNT_PATH",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "OPENRESEARCH_CLAUDE_CODE_OAUTH_TOKEN",
]


@pytest.fixture
def clean_new_gcp_env(monkeypatch):
    """Remove the new env vars so defaults are exercised."""
    for k in _NEW_GCP_ENVVARS:
        monkeypatch.delenv(k, raising=False)


def test_gcp_orchestrator_image_default(clean_new_gcp_env):
    """gcp_orchestrator_image defaults to empty — operator must supply a pinned tag."""
    s = Settings(_env_file=None)
    assert s.gcp_orchestrator_image == ""


def test_gcp_csi_mount_path_default(clean_new_gcp_env):
    """gcp_csi_mount_path defaults to /mnt/sm-secrets."""
    s = Settings(_env_file=None)
    assert s.gcp_csi_mount_path == "/mnt/sm-secrets"


def test_claude_code_oauth_token_default(clean_new_gcp_env):
    """claude_code_oauth_token defaults to empty string."""
    s = Settings(_env_file=None)
    assert s.claude_code_oauth_token == ""


def test_gcp_orchestrator_image_env_override(monkeypatch):
    """OPENRESEARCH_GCP_ORCHESTRATOR_IMAGE overrides gcp_orchestrator_image."""
    monkeypatch.setenv(
        "OPENRESEARCH_GCP_ORCHESTRATOR_IMAGE",
        "us-central1-docker.pkg.dev/my-project/reprolab/reprolab-orchestrator:sha-abc1234",
    )
    s = Settings(_env_file=None)
    assert s.gcp_orchestrator_image == "us-central1-docker.pkg.dev/my-project/reprolab/reprolab-orchestrator:sha-abc1234"


def test_gcp_csi_mount_path_env_override(monkeypatch):
    """OPENRESEARCH_GCP_CSI_MOUNT_PATH overrides gcp_csi_mount_path."""
    monkeypatch.setenv("OPENRESEARCH_GCP_CSI_MOUNT_PATH", "/mnt/custom-secrets")
    s = Settings(_env_file=None)
    assert s.gcp_csi_mount_path == "/mnt/custom-secrets"


def test_claude_code_oauth_token_env_override(monkeypatch):
    """CLAUDE_CODE_OAUTH_TOKEN (unprefixed) overrides claude_code_oauth_token."""
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "tok_test_abc123")
    s = Settings(_env_file=None)
    assert s.claude_code_oauth_token == "tok_test_abc123"


def test_claude_code_oauth_token_prefixed_env_override(monkeypatch):
    """OPENRESEARCH_CLAUDE_CODE_OAUTH_TOKEN also works."""
    monkeypatch.setenv("OPENRESEARCH_CLAUDE_CODE_OAUTH_TOKEN", "tok_prefixed_xyz")
    s = Settings(_env_file=None)
    assert s.claude_code_oauth_token == "tok_prefixed_xyz"


def test_existing_gcp_fields_unchanged(clean_new_gcp_env):
    """Regression: the existing gcp_* fields retain their defaults."""
    s = Settings(_env_file=None)
    assert s.gcp_project == ""
    assert s.gcp_region == "us-central1"
    assert s.gcp_gcs_bucket == ""
    assert s.gcp_namespace == "reprolab"
    assert s.gcp_service_account == "reprolab-sa"
    assert s.gcp_per_gpu_vram_gb == pytest.approx(80.0)
    assert s.gcp_max_nodes == 4
    assert s.gcp_base_image == ""
    assert s.gcp_gpu_skus == ["gcp_a100_80"]
    assert s.gcp_ttl_seconds_after_finished == 3600
    assert s.gcp_job_backoff_limit == 0
    assert s.gcp_cache_mount_path == "/mnt/reprolab-cache"
    assert s.gcp_watch_poll_interval_s == pytest.approx(5.0)
    assert s.gcp_oom_batch_scale_step1 == pytest.approx(0.5)
    assert s.gcp_oom_batch_scale_floor == pytest.approx(0.25)
    assert s.gcp_bootstrap_pip_timeout_s == 600
