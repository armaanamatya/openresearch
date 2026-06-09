"""Shared fixtures for tests/agents/rlm/.

Provides a module-level autouse patch for k8s_job_cell_runner._setting so
that every test in this directory gets safe Azure defaults (in particular a
non-empty azure_base_image, which _build_job_manifest now rejects as empty
per P1-fix-5).  Individual tests that need different values override via
their own monkeypatch.setattr calls, which take priority over this fixture.
"""
from __future__ import annotations

from typing import Any

import pytest


_DEFAULT_AZURE_TEST_SETTINGS: dict[str, Any] = {
    "azure_namespace": "reprolab",
    "azure_service_account": "reprolab-sa",
    "azure_node_pool_name": "gpunodes",
    # P1-fix-5: must be non-empty to pass the _build_job_manifest guard.
    # Tests don't pull this image; it just needs to be a valid non-empty string.
    "azure_base_image": "test-registry.io/reprolab-aks-cell:test",
    "azure_storage_account": "testacct",
    "azure_blob_container": "testctr",
    "azure_files_share": "test-share",
    "azure_max_nodes": 4,
    "azure_gpu_usd_per_hour": 3.5,
    "azure_pending_timeout_seconds": 900,
    "azure_boot_timeout_seconds": 900,
    "azure_gpu_skus": ["azure_a100_80"],
    "dynamic_gpu_max_escalations": 2,
    "azure_ttl_seconds_after_finished": 3600,
    "azure_job_backoff_limit": 0,
    "azure_cache_mount_path": "/mnt/reprolab-cache",
    # Fast poll for tests — avoids real sleep(5) in _watch_job.
    "azure_watch_poll_interval_s": 0.001,
}


@pytest.fixture(autouse=True)
def _azure_test_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch k8s_job_cell_runner._setting with safe test defaults.

    Individual tests that need different settings can override via:
        monkeypatch.setattr(kjcr, "_setting", lambda name, default=None: {...}.get(name, default))
    The most-recently-applied monkeypatch.setattr wins.
    """
    try:
        import backend.agents.rlm.k8s_job_cell_runner as kjcr
    except ImportError:
        return  # module not available (slice not landed) — skip silently

    monkeypatch.setattr(
        kjcr,
        "_setting",
        lambda name, default=None: _DEFAULT_AZURE_TEST_SETTINGS.get(name, default),
    )
