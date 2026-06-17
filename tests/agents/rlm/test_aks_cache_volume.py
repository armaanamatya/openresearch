"""Guard: the AKS cell Job cache volume is a PVC when Files is enabled, an
emptyDir fallback otherwise — with an identical mount path either way.

Pins the blob-only invariant from the 2026-06-14 SDAR-on-Azure spec §4.1: a
cell Pod must never hard-depend on a provisioned Azure Files PVC.
"""
from __future__ import annotations

import pathlib

from backend.agents.rlm.k8s_job_cell_runner import _build_job_manifest

_MOUNT = "/mnt/reprolab-cache"


def _manifest(*, files_cache_enabled: bool, files_share: str):
    return _build_job_manifest(
        job_name="job-x",
        namespace="reprolab",
        service_account="reprolab-sa",
        node_pool_name="gpua100",
        base_image="acr.azurecr.io/reprolab-cell:abc123",
        storage_account="sacct",
        blob_container="reprolab-artifacts",
        files_share=files_share,
        cell_id="qwen3_1_7b__sdar__alfworld__s42",
        cell_params_json="{}",
        output_blob_prefix="runs/x/cells/c/out",
        code_blob_prefix="runs/x/code",
        active_deadline_seconds=3600,
        max_oom_retries=2,
        fingerprint=None,
        now_iso=None,
        cache_mount_path=_MOUNT,
        files_cache_enabled=files_cache_enabled,
    )


def _volume(manifest):
    return manifest["spec"]["template"]["spec"]["volumes"][0]


def _mount_path(manifest):
    vm = manifest["spec"]["template"]["spec"]["containers"][0]["volumeMounts"][0]
    return vm["mountPath"]


def test_pvc_volume_when_files_cache_enabled():
    m = _manifest(files_cache_enabled=True, files_share="reprolab-cache")
    vol = _volume(m)
    assert "persistentVolumeClaim" in vol
    assert vol["persistentVolumeClaim"]["claimName"] == "reprolab-cache"
    assert "emptyDir" not in vol
    assert _mount_path(m) == _MOUNT


def test_pvc_claim_name_matches_helm_chart():
    """Cross-layer pin: the runner's PVC claimName must equal the Helm-
    provisioned PVC metadata.name, so the two layers can never silently drift
    (the bug Codex caught: runner emitted 'reprolab-files-pvc' while the chart
    named the PVC 'reprolab-cache' → every Pod would hang Pending)."""
    m = _manifest(files_cache_enabled=True, files_share="reprolab-cache")
    claim = _volume(m)["persistentVolumeClaim"]["claimName"]
    helm = pathlib.Path("infra/azure/helm/templates/pvc-cache.yaml").read_text()
    assert f"name: {claim}" in helm, (
        f"runner claimName {claim!r} is not the PVC metadata.name in "
        f"pvc-cache.yaml — the cell Pod would bind a nonexistent PVC"
    )


def test_emptydir_when_files_cache_disabled():
    m = _manifest(files_cache_enabled=False, files_share="reprolab-cache")
    vol = _volume(m)
    assert "emptyDir" in vol
    assert "persistentVolumeClaim" not in vol
    assert _mount_path(m) == _MOUNT


def test_emptydir_when_share_empty_even_if_enabled():
    m = _manifest(files_cache_enabled=True, files_share="")
    vol = _volume(m)
    assert "emptyDir" in vol
    assert "persistentVolumeClaim" not in vol
    assert _mount_path(m) == _MOUNT


def test_volume_name_is_stable_across_both_modes():
    on = _volume(_manifest(files_cache_enabled=True, files_share="reprolab-cache"))
    off = _volume(_manifest(files_cache_enabled=False, files_share="reprolab-cache"))
    assert on["name"] == off["name"] == "reprolab-cache"
