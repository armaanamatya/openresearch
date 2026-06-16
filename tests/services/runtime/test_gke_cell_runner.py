"""GKE-specific manifest assertions for k8s_job_cell_runner.

Verifies that with settings_prefix="gcp" bound via bind_run_context:
  (a) the pod template has NO azure.workload.identity/use label
  (b) the container env injects OPENRESEARCH_GCP_GCS_BUCKET (not
      OPENRESEARCH_AZURE_STORAGE_ACCOUNT)
  (c) nodeSelector uses reprolab/sku + nvidia.com/gpu resources stay present
  (d) default_sku resolves to gcp_a100_80

All tests are pure — no real K8s, no real GCS, no network.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import backend.agents.rlm.k8s_job_cell_runner as kjcr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_settings(**overrides) -> Any:
    """Return a SimpleNamespace settings object with GCP defaults + optional overrides.

    SimpleNamespace raises AttributeError for missing attrs (unlike MagicMock which
    returns a truthy Mock), so _setting()'s getattr fallback to _SETTINGS_DEFAULTS
    works correctly.
    """
    defaults = {
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
        "gcp_cell_oom_batch_scale_step1": 0.5,
        "gcp_cell_oom_batch_scale_floor": 0.25,
        "gcp_bootstrap_pip_timeout_s": 600,
        "dynamic_gpu_max_escalations": 2,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _build_gcp_manifest(
    *,
    base_image: str = "us-docker.pkg.dev/my-proj/repo/reprolab:v1",
    storage_account: str = "",
    blob_container: str = "",
    gcs_bucket: str = "my-reprolab-bucket",
    gpu_plan: Any = None,
    pod_template_extra_labels: dict | None = {},
) -> dict[str, Any]:
    """Build a manifest using the GCP code path (prefix="gcp" in context)."""
    fake_s = _fake_settings(gcp_gcs_bucket=gcs_bucket)

    # Monkeypatch _get_settings and _object_store to avoid real imports.
    original_get_settings = kjcr._get_settings
    original_object_store = kjcr._object_store

    class _FakeGcsStore:
        """Minimal fake GcsStore for upload/download without real GCS SDK."""
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
            manifest = kjcr._build_job_manifest(
                job_name="gke-test-job",
                namespace="reprolab",
                service_account="reprolab-sa",
                node_pool_name="gpua100",
                base_image=base_image,
                storage_account=storage_account,
                blob_container=blob_container,
                files_share="reprolab-cache",
                cell_id="cell-gke-001",
                cell_params_json='{"model": "qwen3-1.7b"}',
                output_blob_prefix="runs/r1/cells",
                code_blob_prefix="runs/r1/code",
                active_deadline_seconds=3600,
                max_oom_retries=2,
                fingerprint="fp-gke",
                now_iso="2026-06-16T00:00:00Z",
                gpu_plan=gpu_plan,
                pod_template_extra_labels=pod_template_extra_labels,
            )
    finally:
        kjcr._get_settings = original_get_settings  # type: ignore[assignment]
        kjcr._object_store = original_object_store  # type: ignore[assignment]

    return manifest


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGkeManifestNoAzureWiLabel:
    """(a) GKE manifest must NOT carry azure.workload.identity/use."""

    def test_no_azure_wi_label_when_pod_extra_empty(self):
        manifest = _build_gcp_manifest(pod_template_extra_labels={})
        pod_labels = manifest["spec"]["template"]["metadata"]["labels"]
        assert "azure.workload.identity/use" not in pod_labels, (
            f"azure WI label should not appear in GKE manifest, but found: {pod_labels}"
        )

    def test_base_app_label_still_present(self):
        manifest = _build_gcp_manifest(pod_template_extra_labels={})
        pod_labels = manifest["spec"]["template"]["metadata"]["labels"]
        assert pod_labels.get("app") == "reprolab-cell"

    def test_cell_id_label_still_present(self):
        manifest = _build_gcp_manifest(pod_template_extra_labels={})
        pod_labels = manifest["spec"]["template"]["metadata"]["labels"]
        assert "cell-id" in pod_labels


class TestGkeManifestGcsBucketEnvVar:
    """(b) GKE manifest injects OPENRESEARCH_GCP_GCS_BUCKET, not Azure storage vars."""

    def test_gcp_gcs_bucket_injected(self):
        manifest = _build_gcp_manifest(gcs_bucket="my-test-bucket")
        env_list = manifest["spec"]["template"]["spec"]["containers"][0]["env"]
        env_map = {e["name"]: e["value"] for e in env_list}

        assert "OPENRESEARCH_GCP_GCS_BUCKET" in env_map, (
            "OPENRESEARCH_GCP_GCS_BUCKET must be injected for GCP"
        )
        assert env_map["OPENRESEARCH_GCP_GCS_BUCKET"] == "my-test-bucket"

    def test_azure_storage_account_not_injected(self):
        manifest = _build_gcp_manifest()
        env_list = manifest["spec"]["template"]["spec"]["containers"][0]["env"]
        env_names = {e["name"] for e in env_list}

        assert "OPENRESEARCH_AZURE_STORAGE_ACCOUNT" not in env_names, (
            "OPENRESEARCH_AZURE_STORAGE_ACCOUNT must NOT appear in GKE manifests"
        )

    def test_azure_blob_container_not_injected(self):
        manifest = _build_gcp_manifest()
        env_list = manifest["spec"]["template"]["spec"]["containers"][0]["env"]
        env_names = {e["name"] for e in env_list}

        assert "OPENRESEARCH_AZURE_BLOB_CONTAINER" not in env_names, (
            "OPENRESEARCH_AZURE_BLOB_CONTAINER must NOT appear in GKE manifests"
        )

    def test_cloud_neutral_prefixes_still_injected(self):
        """OPENRESEARCH_BLOB_CODE_PREFIX / _OUTPUT_PREFIX stay for both clouds."""
        manifest = _build_gcp_manifest()
        env_list = manifest["spec"]["template"]["spec"]["containers"][0]["env"]
        env_names = {e["name"] for e in env_list}

        assert "OPENRESEARCH_BLOB_CODE_PREFIX" in env_names
        assert "OPENRESEARCH_BLOB_OUTPUT_PREFIX" in env_names
        assert "OPENRESEARCH_CACHE_MOUNT" in env_names


class TestGkeManifestNodeSelectorAndGpu:
    """(c) reprolab/sku nodeSelector + nvidia.com/gpu resources are present."""

    def test_no_gpu_plan_uses_gcp_a100_80_default(self):
        fake_s = _fake_settings()
        original_get_settings = kjcr._get_settings
        try:
            kjcr._get_settings = lambda: fake_s  # type: ignore[assignment]
            with kjcr.bind_run_context(settings_prefix="gcp"):
                manifest = kjcr._build_job_manifest(
                    job_name="gke-noplan-job",
                    namespace="reprolab",
                    service_account="reprolab-sa",
                    node_pool_name="gpua100",
                    base_image="us-docker.pkg.dev/proj/repo/img:v1",
                    storage_account="",
                    blob_container="",
                    files_share="reprolab-cache",
                    cell_id="cell-001",
                    cell_params_json="{}",
                    output_blob_prefix="runs/r1/cells",
                    code_blob_prefix="runs/r1/code",
                    active_deadline_seconds=3600,
                    max_oom_retries=0,
                    fingerprint=None,
                    now_iso=None,
                    gpu_plan=None,
                    pod_template_extra_labels={},
                )
        finally:
            kjcr._get_settings = original_get_settings  # type: ignore[assignment]

        pod_spec = manifest["spec"]["template"]["spec"]
        node_selector = pod_spec.get("nodeSelector", {})
        assert "reprolab/sku" in node_selector, (
            "nodeSelector must include reprolab/sku"
        )
        assert node_selector["reprolab/sku"] == "gcp_a100_80", (
            f"default_sku should be gcp_a100_80, got: {node_selector['reprolab/sku']}"
        )

    def test_nvidia_gpu_resource_in_container(self):
        manifest = _build_gcp_manifest(pod_template_extra_labels={})
        container = manifest["spec"]["template"]["spec"]["containers"][0]
        resources = container.get("resources", {})

        assert "nvidia.com/gpu" in resources.get("requests", {}), (
            "nvidia.com/gpu must appear in container resource requests"
        )
        assert "nvidia.com/gpu" in resources.get("limits", {}), (
            "nvidia.com/gpu must appear in container resource limits"
        )

    def test_gpu_toleration_present(self):
        manifest = _build_gcp_manifest(pod_template_extra_labels={})
        tolerations = manifest["spec"]["template"]["spec"].get("tolerations", [])
        tol_keys = {t.get("key") for t in tolerations}
        assert "nvidia.com/gpu" in tol_keys, (
            "nvidia.com/gpu toleration must be present for GPU nodes"
        )


class TestGkeDefaultSkuFromSettings:
    """(d) default_sku resolves to gcp_a100_80 from gcp_gpu_skus setting."""

    def test_gcp_gpu_skus_default_is_gcp_a100_80(self):
        """_cloud_setting('gpu_skus') with prefix=gcp returns ['gcp_a100_80']."""
        fake_s = _fake_settings()
        original_get_settings = kjcr._get_settings
        try:
            kjcr._get_settings = lambda: fake_s  # type: ignore[assignment]
            with kjcr.bind_run_context(settings_prefix="gcp"):
                gpu_skus = kjcr._cloud_setting("gpu_skus", [])
        finally:
            kjcr._get_settings = original_get_settings  # type: ignore[assignment]

        assert gpu_skus == ["gcp_a100_80"], (
            f"gcp_gpu_skus should default to ['gcp_a100_80'], got: {gpu_skus}"
        )

    def test_default_sku_in_manifest_is_gcp_a100_80(self):
        """When no gpu_plan is bound, the manifest nodeSelector should use gcp_a100_80."""
        manifest = _build_gcp_manifest(pod_template_extra_labels={})
        pod_spec = manifest["spec"]["template"]["spec"]
        node_selector = pod_spec.get("nodeSelector", {})
        assert node_selector.get("reprolab/sku") == "gcp_a100_80", (
            f"GKE default sku must be gcp_a100_80, got: {node_selector}"
        )


class TestGkeSettingsPrefixProductionNesting:
    """Guard the EXACT nesting primitives._execute_cell_matrix uses:

        with kjcr._bind_settings_prefix(<cloud>), kjcr.bind_run_context(...):
            kjcr.run_matrix(...)

    ``bind_run_context`` is entered with NO ``settings_prefix`` kwarg (its real
    call site passes only run_budget/event_sink/gpu_plan).  A regression here —
    bind_run_context clobbering the prefix back to its default — would silently
    route a GCP run through azure settings + AzureBlobStore.
    """

    def test_gcp_prefix_survives_bind_run_context_nesting(self):
        with kjcr._bind_settings_prefix("gcp"), kjcr.bind_run_context(
            run_budget=None, event_sink=None, gpu_plan=None
        ):
            assert kjcr._get_settings_prefix() == "gcp", (
                "GCP prefix was clobbered by the nested bind_run_context default — "
                "azure settings would be used for a GCP cell run"
            )

    def test_gcp_cloud_setting_resolves_under_nesting(self):
        fake_s = _fake_settings()
        original_get_settings = kjcr._get_settings
        try:
            kjcr._get_settings = lambda: fake_s  # type: ignore[assignment]
            with kjcr._bind_settings_prefix("gcp"), kjcr.bind_run_context(
                run_budget=None, event_sink=None, gpu_plan=None
            ):
                assert kjcr._cloud_setting("gpu_skus", []) == ["gcp_a100_80"]
                assert kjcr._cloud_setting("namespace", "") == "reprolab"
        finally:
            kjcr._get_settings = original_get_settings  # type: ignore[assignment]

    def test_azure_prefix_survives_bind_run_context_nesting(self):
        with kjcr._bind_settings_prefix("azure"), kjcr.bind_run_context(
            run_budget=None, event_sink=None, gpu_plan=None
        ):
            assert kjcr._get_settings_prefix() == "azure"

    def test_prefix_resets_to_azure_after_block(self):
        with kjcr._bind_settings_prefix("gcp"), kjcr.bind_run_context(
            run_budget=None, event_sink=None, gpu_plan=None
        ):
            pass
        assert kjcr._get_settings_prefix() == "azure", (
            "prefix ContextVar leaked past the with-block"
        )


class TestGkeSettingsPrefixIsolation:
    """Verify azure prefix is unaffected by the gcp prefix."""

    def test_azure_prefix_still_returns_azure_defaults(self):
        """bind_run_context(settings_prefix='azure') must yield azure defaults."""
        fake_s = SimpleNamespace(
            azure_namespace="reprolab",
            azure_service_account="reprolab-sa",
            azure_gpu_skus=["azure_a100_80"],
        )
        original_get_settings = kjcr._get_settings
        try:
            kjcr._get_settings = lambda: fake_s  # type: ignore[assignment]
            with kjcr.bind_run_context(settings_prefix="azure"):
                prefix = kjcr._get_settings_prefix()
                gpu_skus = kjcr._cloud_setting("gpu_skus", [])
        finally:
            kjcr._get_settings = original_get_settings  # type: ignore[assignment]

        assert prefix == "azure"
        assert gpu_skus == ["azure_a100_80"]
