"""Regression tests for the GCP sandbox hardening fixes.

Covers:
1. build_environment with sandbox_mode="gcp" returns {ok: True, skipped: True}
   without invoking docker (Fix 1).
2. The cell-route admission list includes "gcp" by default — OPENRESEARCH_GCP_CELL_ROUTE
   defaults to "1" (Fix 3).
3. _build_job_manifest under the GCP prefix with gcp_files_cache_enabled=False
   produces a cache volume that is an emptyDir, NOT a persistentVolumeClaim (Fix 5).

All tests are hermetic — no network, no subprocess, no real K8s or GCS.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import backend.agents.rlm.k8s_job_cell_runner as kjcr
from backend.agents.rlm import primitives


# ---------------------------------------------------------------------------
# Fix 1: build_environment GCP no-op
# ---------------------------------------------------------------------------

class TestBuildEnvironmentGcpNoop:
    """build_environment must short-circuit for sandbox_mode='gcp' without docker."""

    def test_gcp_returns_ok_skipped(self):
        ctx = SimpleNamespace(sandbox_mode="gcp")
        result = primitives.build_environment({"dockerfile": "FROM ubuntu:22.04\n"}, ctx=ctx)
        assert result["ok"] is True
        assert result["skipped"] is True

    def test_gcp_note_mentions_artifact_registry(self):
        ctx = SimpleNamespace(sandbox_mode="gcp")
        result = primitives.build_environment({"dockerfile": "FROM ubuntu:22.04\n"}, ctx=ctx)
        note = result.get("note", "").lower()
        assert "gcp" in note or "artifact" in note

    def test_gcp_never_invokes_docker(self, monkeypatch):
        """Confirm docker is never touched — monkeypatch the client factory to assert."""
        docker_called = {"called": False}

        def _fail_if_called(*a, **k):
            docker_called["called"] = True
            raise RuntimeError("docker client should not be created for gcp sandbox")

        monkeypatch.setattr(primitives, "_docker_client", _fail_if_called, raising=False)
        ctx = SimpleNamespace(sandbox_mode="gcp")
        result = primitives.build_environment({"dockerfile": "FROM ubuntu:22.04\n"}, ctx=ctx)
        assert result["ok"] is True
        assert docker_called["called"] is False


# ---------------------------------------------------------------------------
# Fix 3: cell-route includes "gcp" by default
# ---------------------------------------------------------------------------

class TestGcpCellRouteAdmission:
    """OPENRESEARCH_GCP_CELL_ROUTE defaults to '1' → 'gcp' is admitted."""

    def _make_code(self, tmp_path: Path) -> Path:
        code = tmp_path / "code"
        code.mkdir()
        cells = [{"id": "c1", "model_key": "m", "baseline": "b", "env": "e",
                  "seed": 0, "est_vram_gb": 1.0}]
        (code / "cells.json").write_text(json.dumps({"cells": cells}))
        (code / "train_cell.py").write_text("# stub\n")
        return code

    def _gcp_caps(self):
        return SimpleNamespace(
            backend_kind="gcp", num_gpus=1, per_gpu_vram_gb=80.0,
            free_gpu_ids=("GPU-0",), is_empty=False,
        )

    def _ctx(self, tmp_path: Path):
        return SimpleNamespace(
            project_id="prj_test", project_dir=tmp_path, run_id="prj_test-abc",
            gpu_device_ids=(), sandbox_mode="gcp",
        )

    def test_gcp_in_default_cell_route_kinds(self, monkeypatch, tmp_path):
        """Without any env override, 'gcp' must appear in the cell-route kinds."""
        code = self._make_code(tmp_path)
        spy = {"reached": False}

        def fake_execute(ctx, code_path, caps, *, timeout_s, run_id):
            spy["reached"] = True
            return {"success": False, "metrics": {}, "failure_class": "test_stub"}

        monkeypatch.setattr(primitives, "_execute_cell_matrix", fake_execute)
        monkeypatch.setattr(primitives, "_emit_dashboard_event", lambda *a, **k: None)
        monkeypatch.delenv("OPENRESEARCH_GCP_CELL_ROUTE", raising=False)

        with patch("backend.services.runtime.gpu_capacity.describe_capacity",
                   return_value=self._gcp_caps()), \
             patch("backend.agents.rlm.pre_flight_validator.validate_code_pre_flight",
                   return_value=[]):
            primitives.run_experiment(str(code), env_id="", ctx=self._ctx(tmp_path))

        assert spy["reached"] is True, (
            "GCP backend should have reached _execute_cell_matrix but did not — "
            "'gcp' is missing from _cell_route_kinds"
        )

    def test_gcp_cell_route_disabled_by_env(self, monkeypatch, tmp_path):
        """OPENRESEARCH_GCP_CELL_ROUTE=0 must prevent 'gcp' from being admitted."""
        code = self._make_code(tmp_path)
        spy = {"reached": False}

        def fake_execute(ctx, code_path, caps, *, timeout_s, run_id):
            spy["reached"] = True
            return {"success": False, "metrics": {}, "failure_class": "test_stub"}

        monkeypatch.setattr(primitives, "_execute_cell_matrix", fake_execute)
        monkeypatch.setattr(primitives, "_emit_dashboard_event", lambda *a, **k: None)
        monkeypatch.setenv("OPENRESEARCH_GCP_CELL_ROUTE", "0")

        with patch("backend.services.runtime.gpu_capacity.describe_capacity",
                   return_value=self._gcp_caps()), \
             patch("backend.agents.rlm.pre_flight_validator.validate_code_pre_flight",
                   return_value=[]):
            primitives.run_experiment(str(code), env_id="", ctx=self._ctx(tmp_path))

        assert spy["reached"] is False, (
            "OPENRESEARCH_GCP_CELL_ROUTE=0 should suppress the gcp cell route"
        )


# ---------------------------------------------------------------------------
# Fix 5 + Fix 4: gcp_files_cache_enabled defaults False → emptyDir
# ---------------------------------------------------------------------------

def _fake_gcp_settings(**overrides) -> Any:
    """Return a SimpleNamespace with GCP defaults (using post-Fix-4 key names)."""
    defaults: dict[str, Any] = {
        "gcp_namespace": "reprolab",
        "gcp_service_account": "reprolab-sa",
        "gcp_node_pool_name": "gpua100",
        "gcp_base_image": "us-docker.pkg.dev/my-proj/repo/reprolab:v1",
        "gcp_gcs_bucket": "my-reprolab-bucket",
        "gcp_max_nodes": 4,
        "gcp_gpu_usd_per_hour": 3.67,
        "gcp_pending_timeout_seconds": 900,
        "gcp_gpu_skus": ["gcp_a100_80"],
        "gcp_ttl_seconds_after_finished": 3600,
        "gcp_job_backoff_limit": 0,
        "gcp_cache_mount_path": "/mnt/reprolab-cache",
        "gcp_watch_poll_interval_s": 5.0,
        # Post-Fix-4: keys match config.py fields (no "cell_" infix).
        "gcp_oom_batch_scale_step1": 0.5,
        "gcp_oom_batch_scale_floor": 0.25,
        "gcp_bootstrap_pip_timeout_s": 600,
        # Fix 5: default is False — cells use emptyDir, not PVC.
        "gcp_files_cache_enabled": False,
        "dynamic_gpu_max_escalations": 2,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestGcpFilesCacheDefaultEmptyDir:
    """Fix 5: gcp_files_cache_enabled defaults False → cache volume is emptyDir, not PVC.

    Two complementary test layers:
    A) _cache_volume_spec and config default — conftest-immune, tests the plumbing
       directly without going through the _setting patch.
    B) _build_job_manifest with files_cache_enabled=False → emptyDir in the manifest.
    """

    # --- Layer A: _cache_volume_spec directly (conftest-immune) ---
    #
    # The conftest patches kjcr._setting for all tests in this directory (azure
    # defaults only). Testing _cloud_setting through the patched _setting would
    # see incorrect defaults. _cache_volume_spec is the function _build_job_manifest
    # calls with the resolved files_cache_enabled value — the right isolation boundary.

    def test_cache_volume_spec_false_gives_emptydir(self):
        """_cache_volume_spec(files_cache_enabled=False) → emptyDir."""
        vol = kjcr._cache_volume_spec(
            namespace="reprolab",
            files_share="reprolab-cache",
            files_cache_enabled=False,
        )
        assert "emptyDir" in vol, f"Expected emptyDir for disabled cache, got {vol}"
        assert "persistentVolumeClaim" not in vol

    def test_cache_volume_spec_true_gives_pvc(self):
        """_cache_volume_spec(files_cache_enabled=True) → PVC when share is non-empty."""
        vol = kjcr._cache_volume_spec(
            namespace="reprolab",
            files_share="reprolab-cache",
            files_cache_enabled=True,
        )
        assert "persistentVolumeClaim" in vol, f"Expected PVC for enabled cache, got {vol}"
        assert "emptyDir" not in vol

    def test_config_gcp_files_cache_enabled_default_is_false(self):
        """Fix 5: gcp_files_cache_enabled field defaults to False in config.py."""
        from backend.config import Settings
        s = Settings()
        assert s.gcp_files_cache_enabled is False, (
            "gcp_files_cache_enabled should default to False so GCP cells use "
            "emptyDir and never block on a missing optional Filestore PVC"
        )

    # --- Layer B: manifest builder integration ---

    def _build_manifest(self, *, files_cache_enabled: bool) -> dict[str, Any]:
        """Build a GCP manifest, passing files_cache_enabled directly."""
        fake_s = _fake_gcp_settings()
        original_get_settings = kjcr._get_settings
        original_object_store = kjcr._object_store

        class _FakeGcsStore:
            def upload_prefix(self, local_root, *, blob_prefix):
                return []
            def download_bytes(self, blob_name):
                return b"{}"
            def download_artifact(self, blob_name, destination):
                return Path(destination)

        try:
            kjcr._get_settings = lambda: fake_s  # type: ignore[assignment]
            kjcr._object_store = lambda: _FakeGcsStore()  # type: ignore[assignment]
            with kjcr._bind_settings_prefix("gcp"):
                return kjcr._build_job_manifest(
                    job_name="gcp-cache-test-job",
                    namespace="reprolab",
                    service_account="reprolab-sa",
                    node_pool_name="gpua100",
                    base_image="us-docker.pkg.dev/my-proj/repo/reprolab:v1",
                    storage_account="",
                    blob_container="",
                    files_share="reprolab-cache",
                    files_cache_enabled=files_cache_enabled,
                    cell_id="cell-cache-001",
                    cell_params_json='{"model": "qwen3-1.7b"}',
                    output_blob_prefix="runs/r1/cells",
                    code_blob_prefix="runs/r1/code",
                    active_deadline_seconds=3600,
                    max_oom_retries=2,
                    fingerprint="fp-cache",
                    now_iso="2026-06-16T00:00:00Z",
                    gpu_plan=None,
                    pod_template_extra_labels={},
                )
        finally:
            kjcr._get_settings = original_get_settings  # type: ignore[assignment]
            kjcr._object_store = original_object_store  # type: ignore[assignment]

    def _get_cache_volume(self, manifest: dict) -> dict | None:
        volumes = manifest["spec"]["template"]["spec"].get("volumes", [])
        for v in volumes:
            if v.get("name") == "reprolab-cache":
                return v
        return None

    def test_files_cache_disabled_uses_emptydir(self):
        manifest = self._build_manifest(files_cache_enabled=False)
        vol = self._get_cache_volume(manifest)
        assert vol is not None, "reprolab-cache volume should always be present"
        assert "emptyDir" in vol, (
            f"Expected emptyDir when files_cache_enabled=False, got: {vol}"
        )
        assert "persistentVolumeClaim" not in vol, (
            "PVC must not be used when files_cache_enabled=False"
        )

    def test_files_cache_enabled_uses_pvc(self):
        manifest = self._build_manifest(files_cache_enabled=True)
        vol = self._get_cache_volume(manifest)
        assert vol is not None, "reprolab-cache volume should always be present"
        assert "persistentVolumeClaim" in vol, (
            f"Expected PVC when files_cache_enabled=True, got: {vol}"
        )
