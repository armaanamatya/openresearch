"""Golden-command (argv) parity tests for ``VmComputeProvider`` (Phase 1c, Unit D).

Hermetic: every test injects a fake ``runner`` (a plain callable recording the
built argv and returning a canned :class:`VmExecResult`) -- NOTHING here
executes a real ``gcloud``/``ssh``/``scp`` subprocess. Assertions compare the
BUILT ARGV SHAPE against the bash scripts this module ports
(``scripts/gcp_sdar_preflight.sh``, ``scripts/sdar_gcp_optimal_run.sh``,
``scripts/sdar_gcp_run.sh``, ``scripts/sdar_gcp_watch.sh``,
``scripts/cancel_gcp_sdar_run.sh``), never against live gcloud output.

An ``autouse`` fixture clears the ``OPENRESEARCH_GCP_*``/``OPENRESEARCH_REMOTE_DIR``
env vars before every test so the provider's identity resolution (project/
instance/ssh-user/zone/remote-dir) is always the hardcoded CONFIG DEFAULT --
deterministic regardless of the host shell's own SDAR/GCP exports.
"""

import pytest

from backend.services.runtime.cloud_profile import CloudProfile, VmSpec
from backend.services.runtime.compute_provider import ComputeLease
from backend.services.runtime.run_plan import RunPlan
from backend.services.runtime.vm_compute_provider import VmComputeProvider, VmExecResult

_ENV_KEYS = (
    "OPENRESEARCH_GCP_PROJECT",
    "OPENRESEARCH_GCP_INSTANCE",
    "OPENRESEARCH_GCP_SSH_USER",
    "OPENRESEARCH_GCP_ZONE",
    "OPENRESEARCH_REMOTE_DIR",
    "OPENRESEARCH_GCP_CPU_INSTANCE",
)


@pytest.fixture(autouse=True)
def _clear_ambient_gcp_env(monkeypatch):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _run_plan() -> RunPlan:
    return RunPlan(paper_id="2605.15155")


def _gcp_profile() -> CloudProfile:
    return CloudProfile(
        cloud="gcp",
        vm=VmSpec(
            zone="us-central1-b",
            cpu_machine_type="e2-standard-16",
            gpu_machine_type="a2-highgpu-4g",
            accelerator_type="nvidia-tesla-a100",
            accelerator_count=4,
            machine_image="sdar-ultra",
            cache_disk_name="sdar-cache",
            max_run_duration_s=100800,
            capacity_signatures=(
                "STOCKOUT", "enough resources", "EXHAUSTED", "currently unavailable",
            ),
        ),
    )


def _lease() -> ComputeLease:
    return ComputeLease(cloud="gcp", cpu="sdar-a100-od", ref="sdar-a100-od")


def _ok() -> VmExecResult:
    return VmExecResult(returncode=0)


def _err(stderr: str) -> VmExecResult:
    return VmExecResult(returncode=1, stderr=stderr)


# ---------------------------------------------------------------------------
# The 3 mandated golden/redaction/capacity tests (verbatim from the plan)
# ---------------------------------------------------------------------------


def test_create_argv_matches_bash_shape(tmp_path):
    calls = []
    from backend.services.runtime.cloud_profile import CloudProfile, VmSpec
    from backend.services.runtime.vm_compute_provider import VmComputeProvider

    prof = CloudProfile(cloud="gcp", vm=VmSpec(
        zone="us-central1-b", gpu_machine_type="a2-highgpu-4g",
        accelerator_type="nvidia-tesla-a100", accelerator_count=4,
        machine_image="sdar-ultra", max_run_duration_s=21600,
        capacity_signatures=("ZONE_RESOURCE_POOL_EXHAUSTED", "does not have enough resources")))
    prov = VmComputeProvider(prof, runner=lambda argv: calls.append(argv) or _ok())
    prov.provision_cpu(_run_plan())
    create = next(a for a in calls if a[:3] == ["gcloud", "compute", "instances"] and "create" in a)
    assert "--zone" in create and "us-central1-b" in create
    assert any("a100" in x for x in create) and "--max-run-duration=21600s" in " ".join(create) or "21600" in " ".join(create)


def test_stage_never_ships_a_secret(tmp_path):
    calls = []
    prov = VmComputeProvider(_gcp_profile(), runner=lambda argv: calls.append(argv) or _ok())
    prov.stage(_lease(), bundle={"env": {"ANTHROPIC_API_KEY": "sk-secret-xyz", "SEED": "0"}}, run_spec={})
    blob = repr(calls)
    assert "sk-secret-xyz" not in blob and "ANTHROPIC_API_KEY" not in blob   # redaction
    assert "SEED" in blob or "0" in blob                                     # non-secret survives


def test_capacity_signature_classified_unavailable():
    prov = VmComputeProvider(_gcp_profile(), runner=lambda argv: _err("ZONE_RESOURCE_POOL_EXHAUSTED"))
    rep = prov.preflight(_run_plan())
    assert rep.available is False and "exhaust" in rep.reason.lower()


# ---------------------------------------------------------------------------
# Additional coverage: stop argv (release_gpu/teardown) + the backend.cli
# reproduce --sandbox local launch argv.
# ---------------------------------------------------------------------------


def test_release_gpu_stop_argv_matches_bash_shape():
    calls = []
    prov = VmComputeProvider(_gcp_profile(), runner=lambda argv: calls.append(argv) or _ok())
    prov.release_gpu(_lease())
    stop = next(a for a in calls if a[:3] == ["gcloud", "compute", "instances"] and "stop" in a)
    assert "sdar-a100-od" in stop
    assert "--zone" in stop and "us-central1-b" in stop
    assert "--quiet" in stop
    assert "--discard-local-ssd=true" not in stop  # plain stop succeeded -- no retry needed


def test_release_gpu_retries_with_discard_local_ssd_on_refusal():
    """Mirrors sdar_gcp_watch.sh's vm_stop: a mandatory-local-SSD refusal
    retries once with --discard-local-ssd=true."""
    calls = []

    def _runner(argv):
        calls.append(argv)
        if "--discard-local-ssd=true" in argv:
            return _ok()
        return _err(
            "ERROR: (gcloud.compute.instances.stop) Could not stop instance: "
            "local SSD data on this instance would be lost."
        )

    prov = VmComputeProvider(_gcp_profile(), runner=_runner)
    prov.release_gpu(_lease())
    assert len(calls) == 2
    assert "--discard-local-ssd=true" not in calls[0]
    assert "--discard-local-ssd=true" in calls[1]


def test_teardown_stops_and_never_deletes():
    """The ported bash never issues `instances delete` on the persistent,
    reusable SDAR VM -- teardown must mirror release_gpu (stop only)."""
    calls = []
    prov = VmComputeProvider(_gcp_profile(), runner=lambda argv: calls.append(argv) or _ok())
    prov.teardown(_lease(), reason="run_complete")
    assert any("stop" in a for a in calls)
    assert not any("delete" in a for a in calls)


def test_launch_argv_runs_backend_cli_reproduce_sandbox_local():
    calls = []
    prov = VmComputeProvider(_gcp_profile(), runner=lambda argv: calls.append(argv) or _ok())
    handle = prov.launch(
        _lease(),
        run_spec={
            "paper_id": "2605.15155", "mode": "rlm",
            "model": "foundry", "project_id": "sdar_gcp_e2e",
        },
    )
    ssh = next(a for a in calls if a[:3] == ["gcloud", "compute", "ssh"])
    remote_cmd = ssh[ssh.index("--command") + 1]
    assert "scripts/sdar_gcp_run.sh" in remote_cmd
    assert "setsid nohup" in remote_cmd and "&" in remote_cmd  # detached
    assert handle.meta["cli_argv_hint"] == [
        "backend.cli", "reproduce", "2605.15155",
        "--mode", "rlm", "--sandbox", "local",
        "--model", "foundry", "--project-id", "sdar_gcp_e2e",
    ]
    assert handle.id == "sdar_gcp_e2e"


# ---------------------------------------------------------------------------
# cpu_warm_disk_then_gpu_attach two-VM tiering (Phase 1d, Unit C)
# ---------------------------------------------------------------------------


def _cpu_warm_provider(runner) -> VmComputeProvider:
    prof = CloudProfile(cloud="gcp", vm=VmSpec(
        zone="us-central1-b", cpu_machine_type="e2-standard-16", gpu_machine_type="a2-highgpu-4g",
        machine_image="sdar-ultra", cache_disk_name="sdar-cache",
        tiering_strategy="cpu_warm_disk_then_gpu_attach"))
    return VmComputeProvider(prof, runner=runner)


def _stage_on_gpu_provider(runner) -> VmComputeProvider:
    return VmComputeProvider(_gcp_profile(), runner=runner)


def test_cpu_warm_disk_provision_uses_cheap_cpu_vm_and_attaches_disk(tmp_path):
    calls = []
    from backend.services.runtime.cloud_profile import CloudProfile, VmSpec
    from backend.services.runtime.vm_compute_provider import VmComputeProvider

    prof = CloudProfile(cloud="gcp", vm=VmSpec(
        zone="us-central1-b", cpu_machine_type="e2-standard-16", gpu_machine_type="a2-highgpu-4g",
        machine_image="sdar-ultra", cache_disk_name="sdar-cache",
        tiering_strategy="cpu_warm_disk_then_gpu_attach"))
    prov = VmComputeProvider(prof, runner=lambda a: calls.append(a) or _ok())
    lease = prov.provision_cpu(_run_plan())
    joined = [" ".join(a) for a in calls]
    assert any("e2-standard-16" in j and "instances create" in j for j in joined)   # cheap CPU VM
    assert any("disks create" in j and "sdar-cache" in j for j in joined) or \
           any("attach-disk" in j and "sdar-cache" in j for j in joined)            # disk warmed
    assert any("detach-disk" in j for j in joined)                                  # detached after warm
    assert lease.gpu is None and lease.disk == "sdar-cache"                          # no GPU billing yet


def test_cpu_warm_disk_acquire_gpu_attaches_warmed_disk(tmp_path):
    calls = []
    prov = _cpu_warm_provider(lambda a: calls.append(a) or _ok())
    lease = prov.provision_cpu(_run_plan()); calls.clear()
    lease = prov.acquire_gpu(lease)
    joined = [" ".join(a) for a in calls]
    assert any("a2-highgpu-4g" in j and "instances create" in j for j in joined)    # GPU VM created
    assert any("attach-disk" in j and "sdar-cache" in j for j in joined)            # warmed disk attached
    assert lease.gpu is not None


def test_stage_on_gpu_default_unchanged(tmp_path):
    # the default strategy still folds provision_cpu into the GPU create (1c behavior).
    calls = []
    prov = _stage_on_gpu_provider(lambda a: calls.append(a) or _ok())
    prov.provision_cpu(_run_plan())
    joined = [" ".join(a) for a in calls]
    assert any("a2-highgpu-4g" in j and "instances create" in j for j in joined)
    assert not any("e2-standard-16" in j for j in joined)                           # no cheap CPU VM under stage_on_gpu
