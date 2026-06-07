"""
Tests for azure_* Settings fields.

Verifies that:
- All azure_* fields exist with documented defaults.
- The Literal fields accept "azure" as a valid value.
- Env-var overrides work.
"""

from __future__ import annotations

import os

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
    "REPROLAB_AZURE_RESOURCE_GROUP",
    "REPROLAB_AZURE_REGION",
    "REPROLAB_AZURE_STORAGE_ACCOUNT",
    "REPROLAB_AZURE_BLOB_CONTAINER",
    "REPROLAB_AZURE_FILES_SHARE",
    "REPROLAB_AZURE_ACR_LOGIN_SERVER",
    "REPROLAB_AZURE_AKS_CLUSTER",
    "REPROLAB_AZURE_NAMESPACE",
    "REPROLAB_AZURE_SERVICE_ACCOUNT",
    "REPROLAB_AZURE_NODE_POOL_NAME",
    "REPROLAB_AZURE_PER_GPU_VRAM_GB",
    "REPROLAB_AZURE_MAX_NODES",
    "REPROLAB_AZURE_BASE_IMAGE",
    "REPROLAB_AZURE_GPU_USD_PER_HOUR",
    "REPROLAB_AZURE_BOOT_TIMEOUT_SECONDS",
    "REPROLAB_AZURE_PENDING_TIMEOUT_SECONDS",
    "REPROLAB_AZURE_GPU_SKUS",
]


@pytest.fixture
def clean_azure_env(monkeypatch):
    """Remove all REPROLAB_AZURE_* env vars so defaults are exercised."""
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
    assert s.azure_service_account == "reprolab-runner"
    assert s.azure_node_pool_name == "gpua100"
    assert s.azure_per_gpu_vram_gb == pytest.approx(80.0)
    assert s.azure_max_nodes == 4
    assert s.azure_base_image == ""
    assert s.azure_gpu_usd_per_hour == pytest.approx(3.67)
    assert s.azure_boot_timeout_seconds == 900
    assert s.azure_pending_timeout_seconds == 900
    assert s.azure_gpu_skus == ["azure_a100_80"]


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
    monkeypatch.setenv("REPROLAB_AZURE_RESOURCE_GROUP", "my-rg")
    monkeypatch.setenv("REPROLAB_AZURE_REGION", "westus2")
    monkeypatch.setenv("REPROLAB_AZURE_STORAGE_ACCOUNT", "mysa")
    monkeypatch.setenv("REPROLAB_AZURE_BLOB_CONTAINER", "mycontainer")
    monkeypatch.setenv("REPROLAB_AZURE_FILES_SHARE", "myshare")
    monkeypatch.setenv("REPROLAB_AZURE_ACR_LOGIN_SERVER", "my.azurecr.io")
    monkeypatch.setenv("REPROLAB_AZURE_AKS_CLUSTER", "my-aks")
    monkeypatch.setenv("REPROLAB_AZURE_NAMESPACE", "myns")
    monkeypatch.setenv("REPROLAB_AZURE_SERVICE_ACCOUNT", "mysa")
    monkeypatch.setenv("REPROLAB_AZURE_NODE_POOL_NAME", "mygpu")
    monkeypatch.setenv("REPROLAB_AZURE_PER_GPU_VRAM_GB", "40.0")
    monkeypatch.setenv("REPROLAB_AZURE_MAX_NODES", "8")
    monkeypatch.setenv("REPROLAB_AZURE_BASE_IMAGE", "my.azurecr.io/reprolab:latest")
    monkeypatch.setenv("REPROLAB_AZURE_GPU_USD_PER_HOUR", "3.50")
    monkeypatch.setenv("REPROLAB_AZURE_BOOT_TIMEOUT_SECONDS", "600")
    monkeypatch.setenv("REPROLAB_AZURE_PENDING_TIMEOUT_SECONDS", "300")

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
    """REPROLAB_AZURE_GPU_SKUS as a JSON array env var parses to the right list."""
    monkeypatch.setenv("REPROLAB_AZURE_GPU_SKUS", '["azure_a100_80","azure_a100_80x2"]')
    s = Settings(_env_file=None)
    assert s.azure_gpu_skus == ["azure_a100_80", "azure_a100_80x2"]


def test_azure_gpu_skus_single_element_override(monkeypatch):
    """REPROLAB_AZURE_GPU_SKUS with a single SKU parses correctly."""
    monkeypatch.setenv("REPROLAB_AZURE_GPU_SKUS", '["v100_32"]')
    s = Settings(_env_file=None)
    assert s.azure_gpu_skus == ["v100_32"]
