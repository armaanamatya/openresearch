"""Unit tests for ``backend.agents.rlm.controller_launch`` — the pure,
CPU-only, fenced durable-controller Job manifest builder."""

from __future__ import annotations

import pytest

from backend.agents.rlm.controller_launch import build_controller_job_manifest


def _build(**over):
    kw = dict(
        paper="1412.6980",
        project_id="prj_x",
        fence_epoch=2,
        image="img:v1",
        cpu_pool_label="reprolab/pool=cpu",
        namespace="default",
        service_account="reprolab-sa",
        env={"K": "V"},
        command=["python", "-m", "backend.cli", "campaign", "1412.6980"],
    )
    kw.update(over)
    return build_controller_job_manifest(**kw)


def test_controller_manifest_is_cpu_only_no_gpu_anywhere():
    m = _build()
    assert "nvidia.com/gpu" not in str(m)


def test_controller_manifest_targets_cpu_pool():
    pod = _build()["spec"]["template"]["spec"]
    assert pod["nodeSelector"] == {"reprolab/pool": "cpu"}


def test_controller_manifest_name_embeds_fence_epoch():
    assert "fe2" in _build(fence_epoch=2)["metadata"]["name"]


def test_controller_manifest_runs_the_given_command():
    pod = _build()["spec"]["template"]["spec"]
    assert pod["containers"][0]["command"] == [
        "python", "-m", "backend.cli", "campaign", "1412.6980",
    ]


def test_controller_manifest_uses_service_account_and_backoff():
    m = _build(backoff_limit=3)
    pod = m["spec"]["template"]["spec"]
    assert pod["serviceAccountName"] == "reprolab-sa"
    assert m["spec"]["backoffLimit"] == 3
    assert pod["terminationGracePeriodSeconds"] == 30
    assert pod["securityContext"] == {
        "runAsNonRoot": True,
        "runAsUser": 10001,
        "runAsGroup": 10001,
        "fsGroup": 10001,
        "fsGroupChangePolicy": "OnRootMismatch",
    }


def test_controller_manifest_carries_env():
    pod = _build(env={"OPENRESEARCH_CELL_FENCE_EPOCH": "2", "A": "B"})["spec"]["template"]["spec"]
    names = {e["name"]: e["value"] for e in pod["containers"][0]["env"]}
    assert names["OPENRESEARCH_CELL_FENCE_EPOCH"] == "2"
    assert names["A"] == "B"


def test_controller_manifest_mounts_persistent_state_and_csi_secrets():
    pod = _build(
        runs_pvc_name="reprolab-cache",
        runs_mount_path="/mnt/reprolab",
        secret_provider_class="reprolab-orchestrator-sm",
    )["spec"]["template"]["spec"]
    assert pod["volumes"] == [
        {
            "name": "controller-state",
            "persistentVolumeClaim": {"claimName": "reprolab-cache"},
        },
        {
            "name": "orchestrator-secrets",
            "csi": {
                "driver": "secrets-store.csi.k8s.io",
                "readOnly": True,
                "volumeAttributes": {
                    "secretProviderClass": "reprolab-orchestrator-sm"
                },
            },
        },
    ]


def test_controller_manifest_rejects_literal_credentials():
    with pytest.raises(ValueError, match="must not embed credential"):
        _build(env={"ANTHROPIC_API_KEY": "secret"})


def test_controller_manifest_rejects_malformed_cpu_selector():
    with pytest.raises(ValueError, match="key=value"):
        _build(cpu_pool_label="cpu-only")


def test_controller_manifest_carries_reaper_fence_and_workload_identity_label():
    manifest = _build(pod_labels={"azure.workload.identity/use": "true"})
    assert manifest["metadata"]["labels"]["reprolab-generation"] == "2"
    assert (
        manifest["spec"]["template"]["metadata"]["labels"]
        ["azure.workload.identity/use"]
        == "true"
    )
