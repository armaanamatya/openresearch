"""
Tests for azure_* Settings fields.

Verifies that:
- All azure_* fields exist with documented defaults.
- The Literal fields accept "azure" as a valid value.
- Env-var overrides work.
"""

from __future__ import annotations


import pytest

from backend.config import Settings


def _reset_settings_cache():
    import backend.config as _config
    _config._settings_cache = None


@pytest.fixture(autouse=True)
def _isolate_settings_cache():
    _reset_settings_cache()
    yield
    _reset_settings_cache()


_AZURE_ENVVARS = [
    "OPENRESEARCH_AZURE_RESOURCE_GROUP",
    "OPENRESEARCH_AZURE_REGION",
    "OPENRESEARCH_AZURE_STORAGE_ACCOUNT",
    "OPENRESEARCH_AZURE_BLOB_CONTAINER",
    "OPENRESEARCH_AZURE_FILES_SHARE",
    "OPENRESEARCH_AZURE_ACR_LOGIN_SERVER",
    "OPENRESEARCH_AZURE_AKS_CLUSTER",
    "OPENRESEARCH_AZURE_NAMESPACE",
    "OPENRESEARCH_AZURE_SERVICE_ACCOUNT",
    "OPENRESEARCH_AZURE_NODE_POOL_NAME",
    "OPENRESEARCH_AZURE_PER_GPU_VRAM_GB",
    "OPENRESEARCH_AZURE_MAX_NODES",
    "OPENRESEARCH_AZURE_BASE_IMAGE",
    "OPENRESEARCH_AZURE_GPU_USD_PER_HOUR",
    "OPENRESEARCH_AZURE_BOOT_TIMEOUT_SECONDS",
    "OPENRESEARCH_AZURE_PENDING_TIMEOUT_SECONDS",
    "OPENRESEARCH_AZURE_GPU_SKUS",
    "OPENRESEARCH_AZURE_TTL_SECONDS_AFTER_FINISHED",
    "OPENRESEARCH_AZURE_JOB_BACKOFF_LIMIT",
    "OPENRESEARCH_AZURE_CACHE_MOUNT_PATH",
    "OPENRESEARCH_AZURE_WATCH_POLL_INTERVAL_S",
    "OPENRESEARCH_AZURE_OOM_BATCH_SCALE_STEP1",
    "OPENRESEARCH_AZURE_OOM_BATCH_SCALE_FLOOR",
    "OPENRESEARCH_AZURE_BOOTSTRAP_PIP_TIMEOUT_S",
]


@pytest.fixture
def clean_azure_env(monkeypatch):
    """Remove all OPENRESEARCH_AZURE_* env vars so defaults are exercised."""
    for k in _AZURE_ENVVARS:
        monkeypatch.delenv(k, raising=False)


def test_azure_fields_exist_with_defaults(clean_azure_env):
    """All azure_* fields must exist and have the documented defaults."""
    s = Settings(_env_file=None)

    assert s.azure_resource_group == ""
    assert s.azure_region == "eastus"
    assert s.azure_storage_account == ""
    assert s.azure_blob_container == "reprolab-artifacts"
    assert s.azure_files_share == "reprolab-cache"
    assert s.azure_acr_login_server == ""
    assert s.azure_aks_cluster == ""
    assert s.azure_namespace == "reprolab"
    # Canonical SA name — matches Terraform/tfvars/Helm federated-credential subject
    assert s.azure_service_account == "reprolab-sa"
    assert s.azure_node_pool_name == "gpua100"
    assert s.azure_per_gpu_vram_gb == pytest.approx(80.0)
    # Canonical concurrency cap — the runner's _SETTINGS_DEFAULTS must match this
    assert s.azure_max_nodes == 4
    assert s.azure_base_image == ""
    assert s.azure_gpu_usd_per_hour == pytest.approx(3.67)
    assert s.azure_boot_timeout_seconds == 900
    # Cold-start from zero can take 10-12 min; 900s killed legitimate scale-up
    assert s.azure_pending_timeout_seconds == 1500
    assert s.azure_gpu_skus == ["azure_a100_80"]
    # New operational knob defaults
    assert s.azure_ttl_seconds_after_finished == 3600
    assert s.azure_job_backoff_limit == 0
    assert s.azure_cache_mount_path == "/mnt/reprolab-cache"
    assert s.azure_watch_poll_interval_s == pytest.approx(5.0)
    assert s.azure_oom_batch_scale_step1 == pytest.approx(0.5)
    assert s.azure_oom_batch_scale_floor == pytest.approx(0.25)
    assert s.azure_bootstrap_pip_timeout_s == 600


def test_default_sandbox_literal_accepts_azure(clean_azure_env):
    """default_sandbox Literal must accept 'azure' without a ValidationError."""
    s = Settings(_env_file=None, default_sandbox="azure")
    assert s.default_sandbox == "azure"


def test_force_sandbox_literal_accepts_azure(clean_azure_env):
    """force_sandbox Literal must accept 'azure' without a ValidationError."""
    s = Settings(_env_file=None, force_sandbox="azure")
    assert s.force_sandbox == "azure"


def test_azure_env_var_overrides(monkeypatch):
    """Env-var overrides must reach the Settings fields."""
    monkeypatch.setenv("OPENRESEARCH_AZURE_RESOURCE_GROUP", "my-rg")
    monkeypatch.setenv("OPENRESEARCH_AZURE_REGION", "westus2")
    monkeypatch.setenv("OPENRESEARCH_AZURE_STORAGE_ACCOUNT", "mysa")
    monkeypatch.setenv("OPENRESEARCH_AZURE_BLOB_CONTAINER", "mycontainer")
    monkeypatch.setenv("OPENRESEARCH_AZURE_FILES_SHARE", "myshare")
    monkeypatch.setenv("OPENRESEARCH_AZURE_ACR_LOGIN_SERVER", "my.azurecr.io")
    monkeypatch.setenv("OPENRESEARCH_AZURE_AKS_CLUSTER", "my-aks")
    monkeypatch.setenv("OPENRESEARCH_AZURE_NAMESPACE", "myns")
    monkeypatch.setenv("OPENRESEARCH_AZURE_SERVICE_ACCOUNT", "mysa")
    monkeypatch.setenv("OPENRESEARCH_AZURE_NODE_POOL_NAME", "mygpu")
    monkeypatch.setenv("OPENRESEARCH_AZURE_PER_GPU_VRAM_GB", "40.0")
    monkeypatch.setenv("OPENRESEARCH_AZURE_MAX_NODES", "8")
    monkeypatch.setenv("OPENRESEARCH_AZURE_BASE_IMAGE", "my.azurecr.io/reprolab:latest")
    monkeypatch.setenv("OPENRESEARCH_AZURE_GPU_USD_PER_HOUR", "3.50")
    monkeypatch.setenv("OPENRESEARCH_AZURE_BOOT_TIMEOUT_SECONDS", "600")
    monkeypatch.setenv("OPENRESEARCH_AZURE_PENDING_TIMEOUT_SECONDS", "300")

    s = Settings()
    assert s.azure_resource_group == "my-rg"
    assert s.azure_region == "westus2"
    assert s.azure_storage_account == "mysa"
    assert s.azure_blob_container == "mycontainer"
    assert s.azure_files_share == "myshare"
    assert s.azure_acr_login_server == "my.azurecr.io"
    assert s.azure_aks_cluster == "my-aks"
    assert s.azure_namespace == "myns"
    assert s.azure_node_pool_name == "mygpu"
    assert s.azure_per_gpu_vram_gb == pytest.approx(40.0)
    assert s.azure_max_nodes == 8
    assert s.azure_base_image == "my.azurecr.io/reprolab:latest"
    assert s.azure_gpu_usd_per_hour == pytest.approx(3.50)
    assert s.azure_boot_timeout_seconds == 600
    assert s.azure_pending_timeout_seconds == 300


def test_existing_literal_values_still_valid(clean_azure_env):
    """Regression: the existing Literal values for default_sandbox + force_sandbox still work."""
    for val in ("auto", "local", "docker", "runpod"):
        s = Settings(_env_file=None, default_sandbox=val)
        assert s.default_sandbox == val

    for val in ("", "auto", "local", "docker", "runpod"):
        s = Settings(_env_file=None, force_sandbox=val)
        assert s.force_sandbox == val


def test_azure_gpu_skus_default(clean_azure_env):
    """azure_gpu_skus must default to a single-element list containing 'a100_80'."""
    s = Settings(_env_file=None)
    assert s.azure_gpu_skus == ["azure_a100_80"]
    assert isinstance(s.azure_gpu_skus, list)


def test_azure_gpu_skus_json_env_override(monkeypatch):
    """OPENRESEARCH_AZURE_GPU_SKUS as a JSON array env var parses to the right list."""
    monkeypatch.setenv("OPENRESEARCH_AZURE_GPU_SKUS", '["azure_a100_80","azure_a100_80x2"]')
    s = Settings(_env_file=None)
    assert s.azure_gpu_skus == ["azure_a100_80", "azure_a100_80x2"]


def test_azure_gpu_skus_single_element_override(monkeypatch):
    """OPENRESEARCH_AZURE_GPU_SKUS with a single SKU parses correctly."""
    monkeypatch.setenv("OPENRESEARCH_AZURE_GPU_SKUS", '["v100_32"]')
    s = Settings(_env_file=None)
    assert s.azure_gpu_skus == ["v100_32"]


def test_azure_max_nodes_canonical_cap(clean_azure_env):
    """azure_max_nodes default == 4, the canonical concurrency cap.

    The runner's _SETTINGS_DEFAULTS must match this value.  If this test
    fails, a drift fix agent updated the runner without updating config.py
    (or vice-versa).
    """
    s = Settings(_env_file=None)
    assert s.azure_max_nodes == 4


def test_azure_new_fields_env_overrides(monkeypatch):
    """Env-var overrides for the 7 new operational knob fields."""
    monkeypatch.setenv("OPENRESEARCH_AZURE_TTL_SECONDS_AFTER_FINISHED", "7200")
    monkeypatch.setenv("OPENRESEARCH_AZURE_JOB_BACKOFF_LIMIT", "2")
    monkeypatch.setenv("OPENRESEARCH_AZURE_CACHE_MOUNT_PATH", "/mnt/custom-cache")
    monkeypatch.setenv("OPENRESEARCH_AZURE_WATCH_POLL_INTERVAL_S", "10.0")
    monkeypatch.setenv("OPENRESEARCH_AZURE_OOM_BATCH_SCALE_STEP1", "0.75")
    monkeypatch.setenv("OPENRESEARCH_AZURE_OOM_BATCH_SCALE_FLOOR", "0.5")
    monkeypatch.setenv("OPENRESEARCH_AZURE_BOOTSTRAP_PIP_TIMEOUT_S", "300")

    s = Settings(_env_file=None)
    assert s.azure_ttl_seconds_after_finished == 7200
    assert s.azure_job_backoff_limit == 2
    assert s.azure_cache_mount_path == "/mnt/custom-cache"
    assert s.azure_watch_poll_interval_s == pytest.approx(10.0)
    assert s.azure_oom_batch_scale_step1 == pytest.approx(0.75)
    assert s.azure_oom_batch_scale_floor == pytest.approx(0.5)
    assert s.azure_bootstrap_pip_timeout_s == 300


def test_azure_service_account_drift_fix(clean_azure_env):
    """azure_service_account must be 'reprolab-sa', not 'reprolab-runner'.

    The federated-credential subject in Terraform/tfvars/Helm is:
      system:serviceaccount:<ns>:reprolab-sa
    Using 'reprolab-runner' would cause every workload-identity token request
    to fail with a 401.
    """
    s = Settings(_env_file=None)
    assert s.azure_service_account == "reprolab-sa"
    assert s.azure_service_account != "reprolab-runner"


def test_azure_pending_timeout_cold_start(clean_azure_env):
    """azure_pending_timeout_seconds must be >= 1500.

    AKS GPU cold-start from zero can take 10-12 minutes (node provisioning +
    image pull + driver init).  900s killed legitimate scale-up as
    capacity_exhausted.  The canonical floor is 1500s.
    """
    s = Settings(_env_file=None)
    assert s.azure_pending_timeout_seconds >= 1500


def test_azure_oom_batch_scale_invariant(clean_azure_env):
    """OOM batch scale floor must be <= step1 (both in (0, 1])."""
    s = Settings(_env_file=None)
    assert 0 < s.azure_oom_batch_scale_floor <= s.azure_oom_batch_scale_step1 <= 1.0


def test_azure_base_image_default_empty(clean_azure_env):
    """azure_base_image defaults to empty — operator must supply a pinned tag."""
    s = Settings(_env_file=None)
    assert s.azure_base_image == ""
