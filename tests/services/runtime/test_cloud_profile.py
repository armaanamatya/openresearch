"""Unit tests for the neutral CloudProfile/VmSpec (Phase 1c, Unit B)."""

from backend.services.runtime.cloud_profile import CloudProfile, VmSpec


def test_vmspec_defaults_stage_on_gpu():
    assert VmSpec().tiering_strategy == "stage_on_gpu"


def test_cloud_profile_holds_k8s_and_vm():
    from backend.services.runtime.gke_job_backend import _GCP_CLOUD  # the existing K8s CloudSpec

    prof = CloudProfile(cloud="gcp", k8s=_GCP_CLOUD, vm=VmSpec(zone="us-central1-b", accelerator_count=1))
    assert prof.cloud == "gcp" and prof.k8s is _GCP_CLOUD and prof.vm.zone == "us-central1-b"
