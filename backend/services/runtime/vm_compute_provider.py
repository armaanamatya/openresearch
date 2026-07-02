"""Single-VM GCP compute provider -- a pure argv-builder port of the SDAR bash
lifecycle scripts (Phase 1c, Unit D).

``VmComputeProvider`` implements every
:class:`~backend.services.runtime.compute_provider.ComputeProvider` method as
a **pure ``gcloud``/``ssh``/``scp`` argv builder** driven through an injected
``runner: Callable[[list[str]], VmExecResult]``. NOTHING executes a real
subprocess in a unit test -- ``runner`` defaults to a thin real-subprocess
wrapper (:func:`_default_subprocess_runner`) that only runs outside tests;
every test in ``tests/services/runtime/test_vm_compute_provider.py`` injects
a recording fake and asserts on the emitted argv shape (golden-command
parity), never on live gcloud output.

This module ports the single-VM SDAR-on-GCP operator scripts:
``scripts/gcp_sdar_preflight.sh`` (VM lifecycle actions -- start/stop/sync/
prepare/launch, cache-disk attach), ``scripts/sdar_gcp_optimal_run.sh`` (the
actual ``gcloud compute instances create`` argv shapes + STOCKOUT capacity-
signature classification -- despite the Unit D plan's inline comment
attributing those create lines to ``gcp_sdar_preflight.sh``, that script has
no ``instances create`` call anywhere in it; ``sdar_gcp_optimal_run.sh`` is
the real source, confirmed by reading both files start-to-end),
``scripts/sdar_gcp_run.sh`` (the detached ``backend.cli reproduce ...
--sandbox local`` launch), ``scripts/sdar_gcp_watch.sh`` (status poll +
report pull + stop-with-local-ssd-retry), and
``scripts/cancel_gcp_sdar_run.sh`` (stop).

``tiering_strategy="stage_on_gpu"`` (:class:`~backend.services.runtime.cloud_profile.VmSpec`'s
default) means ``provision_cpu`` FOLDS into the GPU-type ``gcloud compute
instances create`` -- there is no separate cheap CPU tier -- matching the
ported bash's own reasoning: a GPU machine type requires
``onHostMaintenance=TERMINATE`` while an on-demand e2 STANDARD CPU type
requires ``MIGRATE``, and there is no valid intermediate machine type to
flip between them, so SDAR stages directly on the billed GPU type.
``acquire_gpu`` is consequently a near no-op: the
``--max-run-duration``/``--instance-termination-action=STOP`` billing
ceiling was already armed by ``provision_cpu``'s create call, so
``acquire_gpu`` just stamps the lease's ``gpu`` slot to record that
GPU-billed compute is considered acquired from this point in the
``ReproductionRun`` state machine's lifecycle.

``tiering_strategy="cpu_warm_disk_then_gpu_attach"`` (Phase 1d, Unit C) is a
genuine two-VM handoff, ported from ``sdar_cpu_stage.sh``'s CPU-side warming
phases + ``gcp_sdar_preflight.sh``'s ``attach_and_mount_cache_disk`` same-
zone check: ``provision_cpu`` creates a SEPARATE, cheap CPU-only VM (no
accelerator, ``vm.cpu_machine_type``) that attaches the shared cache disk,
runs the ``.warm_ok``-sentinel-gated warm step over ssh, then detaches --
NO GPU billing starts here. ``acquire_gpu`` creates the actual GPU VM
(``vm.gpu_machine_type`` -- billing starts HERE, unlike ``stage_on_gpu``)
and re-attaches the already-warmed disk, same-zone-locked (a zonal
``pd-ssd`` cannot cross zones; a mismatch fails soft to stage_on_gpu-style
GPU-only provisioning rather than raising). See
``_provision_cpu_warm_disk``/``_acquire_gpu_warm_disk`` below.

``VmSpec.tiering_strategy="machine_type_flip"`` (which needs
``set-scheduling``/``set-machine-type`` to flip a STOPPED VM between a cheap
CPU type and the GPU type in place, mirroring ``gcp_sdar_preflight.sh``'s
``ensure_machine_type``) remains explicitly OUT OF SCOPE -- this provider
implements ``stage_on_gpu`` and ``cpu_warm_disk_then_gpu_attach`` only.

``--project`` is appended at the END of every built argv (after the
resource/verb/positional tokens are fully assembled) rather than immediately
after ``gcloud`` (as the ported bash's ``gcloud_base``/``G`` array literally
does, e.g. ``gcloud --project "$PROJECT" compute instances create ...``).
gcloud accepts global flags in any position, so this is a pure argv-SHAPE
choice -- every builder's argv predictably starts with ``["gcloud",
"compute", <resource>, ...]``, the shape the golden-command tests assert on
-- not a semantic deviation from the bash.

CredentialBroker precondition: ``stage`` never ships a raw ``.env`` (the
ported bash's ``sync_repo`` dotglob-SCPs ``.env``, including the OAuth
token -- this module deliberately does NOT reproduce that). Instead
``stage`` builds a redacted run-spec payload (any key matching
``KEY|TOKEN|SECRET|PASSWORD``, case-insensitively, is dropped), writes it to
a local scratch file, and ships ONLY that file's local PATH through the
injected runner's argv -- so no secret-shaped byte ever appears in a built
argv, regardless of redaction correctness in depth (defense in depth: even
an imperfect redaction can't leak into an argv it never touches). The real
``CredentialBroker`` lands in Phase 1d.

Argv-parity only: every test here asserts a BUILT ARGV SHAPE against an
injected fake, never against a live VM. Live-VM bash-vs-provider A/B parity
is the operator-gated Phase-1f step, out of scope for this unit.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, replace
from typing import Callable, Iterator

from backend.services.runtime.cloud_profile import CloudProfile, VmSpec
from backend.services.runtime.compute_provider import (
    CapacityReport,
    ComputeLease,
    ComputeProvider,
    ReportBundle,
    RunHandle,
    RunStatus,
)
from backend.services.runtime.interface import RuntimeCauseKind, SandboxRuntimeError

# --- CONFIG DEFAULTS -- mirror the ported bash's own `${VAR:-default}` fallbacks ---
_DEFAULT_PROJECT = "deepinvent-ext-ut"
_DEFAULT_INSTANCE = "sdar-a100-od"
_DEFAULT_SSH_USER = "abheekp"
_DEFAULT_REMOTE_DIR = "/home/abheekp/openresearch"
_DEFAULT_ZONE = "us-central1-b"
_DEFAULT_GPU_MACHINE_TYPE = "a2-highgpu-4g"  # 4x A100-80GB
_DEFAULT_CPU_MACHINE_TYPE = "e2-standard-16"  # cheap CPU tier (gcp_sdar_preflight.sh CPU_MACHINE_TYPE)
_DEFAULT_MAX_RUN_DURATION_S = 100800  # 28h hard ceiling (sdar_gcp_optimal_run.sh MAX_RUN_DUR)
_DEFAULT_IMAGE_PROJECT = "deeplearning-platform-release"
_DEFAULT_CACHE_MOUNT = "/mnt/sdar-cache"  # cpu_warm_disk_then_gpu_attach's .warm_ok sentinel lives here

# Capacity/stockout stderr substrings (case-insensitive), matching the bash's
# `grep -qiE 'STOCKOUT|enough resources|EXHAUSTED|currently unavailable'`.
_DEFAULT_CAPACITY_SIGNATURES: tuple[str, ...] = (
    "STOCKOUT",
    "enough resources",
    "EXHAUSTED",
    "currently unavailable",
)

# Secret-shaped env KEY names never allowed to reach a staged bundle/argv.
_SECRET_KEY_RE = re.compile(r"key|token|secret|password", re.IGNORECASE)

# A stop refused because of a mandatory local SSD -- retry with the discard flag.
_LOCAL_SSD_RETRY_RE = re.compile(r"local ssd|discard-local-ssd|cannot be stopped", re.IGNORECASE)


@dataclass(frozen=True)
class VmExecResult:
    """Minimal result the injected ``runner`` returns for one built argv.

    Deliberately NOT the pydantic ``backend.services.runtime.interface.ExecResult``
    -- that shape carries in-sandbox-exec-specific fields (timestamps, a
    ``command`` string, ``cause_kind``) oriented at sandboxed command
    execution. A VM lifecycle command is a bare subprocess invocation; this
    is the minimal shape a test fake needs to construct.
    """

    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


def _default_subprocess_runner(argv: list[str]) -> VmExecResult:
    """Real ``subprocess.run`` wrapper -- the default when no ``runner`` is injected.

    Never exercised by a unit test (every test constructs ``VmComputeProvider``
    with an explicit fake ``runner``); this only runs in a live operator
    invocation.
    """
    proc = subprocess.run(argv, capture_output=True, text=True)
    return VmExecResult(returncode=proc.returncode, stdout=proc.stdout or "", stderr=proc.stderr or "")


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value else default


def _redact_env(env: dict) -> dict:
    """Drop any key matching KEY|TOKEN|SECRET|PASSWORD (case-insensitive)."""
    if not isinstance(env, dict):
        return {}
    return {k: v for k, v in env.items() if not _SECRET_KEY_RE.search(str(k))}


class VmComputeProvider(ComputeProvider):
    """Single-VM GCP ``ComputeProvider`` -- pure argv builders + an injected runner.

    ``profile.vm`` (a :class:`~backend.services.runtime.cloud_profile.VmSpec`)
    supplies the per-run zone/machine-type/image/duration configuration; the
    GCP project/instance-name/SSH-user/remote-dir identity (which ``VmSpec``
    deliberately does not carry -- see ``cloud_profile.py``) is resolved from
    ``OPENRESEARCH_GCP_*``/``OPENRESEARCH_REMOTE_DIR`` env vars, falling back
    to the SAME literal defaults the ported bash scripts use. Resolution is a
    property (read dynamically on each access, never cached at construction),
    matching this repo's "shell wins" env-read idiom used elsewhere.
    """

    def __init__(
        self,
        profile: CloudProfile,
        *,
        runner: Callable[[list[str]], VmExecResult] | None = None,
        poll_interval_s: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.profile = profile
        self._vm: VmSpec = profile.vm or VmSpec()
        self._runner = runner or _default_subprocess_runner
        self._poll_interval_s = poll_interval_s
        self._sleep = sleep

    # ------------------------------------------------------------------
    # Identity / config resolution (properties: read dynamically, never cached)
    # ------------------------------------------------------------------

    @property
    def _project(self) -> str:
        return _env("OPENRESEARCH_GCP_PROJECT", _DEFAULT_PROJECT)

    @property
    def _instance(self) -> str:
        return _env("OPENRESEARCH_GCP_INSTANCE", _DEFAULT_INSTANCE)

    @property
    def _ssh_user(self) -> str:
        return _env("OPENRESEARCH_GCP_SSH_USER", _DEFAULT_SSH_USER)

    @property
    def _remote_dir(self) -> str:
        return _env("OPENRESEARCH_REMOTE_DIR", _DEFAULT_REMOTE_DIR)

    @property
    def _zone(self) -> str:
        return self._vm.zone or _env("OPENRESEARCH_GCP_ZONE", _DEFAULT_ZONE)

    @property
    def _gpu_machine_type(self) -> str:
        return self._vm.gpu_machine_type or _DEFAULT_GPU_MACHINE_TYPE

    @property
    def _cpu_machine_type(self) -> str:
        return self._vm.cpu_machine_type or _DEFAULT_CPU_MACHINE_TYPE

    @property
    def _cpu_instance(self) -> str:
        """Distinct instance name for the cheap CPU staging VM under
        ``cpu_warm_disk_then_gpu_attach`` -- ``self._instance`` is reserved
        for the GPU VM this strategy creates in ``acquire_gpu`` (unlike
        ``stage_on_gpu``, where ``self._instance`` IS the one VM that folds
        both roles). Defaults to ``<gpu-instance>-cpu``, independently
        overridable so an operator can point at an already-running CPU
        staging box without colliding with the GPU instance name.
        """
        return _env("OPENRESEARCH_GCP_CPU_INSTANCE", f"{self._instance}-cpu")

    @property
    def _max_run_duration_s(self) -> int:
        return self._vm.max_run_duration_s if self._vm.max_run_duration_s else _DEFAULT_MAX_RUN_DURATION_S

    # ------------------------------------------------------------------
    # argv helpers
    # ------------------------------------------------------------------

    def _gcloud(self, *args: str) -> list[str]:
        """``["gcloud", <args...>, "--project", <PROJECT>]``.

        Call this ONCE per command with the fully-assembled positional/flag
        tokens already in ``args`` -- see the module docstring for why
        ``--project`` is appended at the end rather than mirroring the bash's
        ``gcloud --project X ...`` prefix position.
        """
        return ["gcloud", *args, "--project", self._project]

    def _ssh_argv(self, remote_command: str, *, instance: str | None = None) -> list[str]:
        """``instance`` defaults to ``self._instance`` (every stage_on_gpu
        call site keeps that byte-identical behavior); the
        cpu_warm_disk_then_gpu_attach CPU warm step passes the CPU instance
        name explicitly since it targets a different VM."""
        target = instance or self._instance
        return self._gcloud(
            "compute", "ssh", f"{self._ssh_user}@{target}",
            "--zone", self._zone, "--quiet", "--command", remote_command,
        )

    def _run(self, argv: list[str]) -> VmExecResult:
        return self._runner(argv)

    def _match_capacity_signature(self, stderr: str) -> str | None:
        if not stderr:
            return None
        signatures = self._vm.capacity_signatures or _DEFAULT_CAPACITY_SIGNATURES
        lowered = stderr.lower()
        for sig in signatures:
            if sig.lower() in lowered:
                return sig
        return None

    def _create_instance_or_raise(self, argv: list[str]) -> VmExecResult:
        """Shared create-then-classify-failure helper for the NEW
        cpu_warm_disk_then_gpu_attach create call sites (CPU VM in
        ``_provision_cpu_warm_disk``, GPU VM in ``_acquire_gpu_warm_disk``).

        The stage_on_gpu branch of ``provision_cpu`` keeps its own inline
        copy of this exact error-classification pattern untouched, so this
        helper is additive only -- it can never regress stage_on_gpu's
        byte-identical argv/behavior.
        """
        result = self._run(argv)
        if result.returncode != 0:
            sig = self._match_capacity_signature(result.stderr)
            if sig is not None:
                raise SandboxRuntimeError(
                    RuntimeCauseKind.backend_unavailable,
                    f"gcp capacity signature matched during create: {sig}",
                )
            raise SandboxRuntimeError(
                RuntimeCauseKind.command_failed,
                f"gcloud compute instances create failed: {(result.stderr or '').strip()[:500]}",
            )
        return result

    # ------------------------------------------------------------------
    # ComputeProvider: preflight
    # ------------------------------------------------------------------

    def preflight(self, plan) -> CapacityReport:
        """Read-only ``describe`` probe -- NO lease, NO billing.

        A confirmed capacity signature in the probe's stderr (e.g. a
        stockout surfaced on a prior create attempt this process retries
        against) marks ``available=False``. Any other outcome -- including
        "instance not found" on a first-ever provision, which is normal, not
        a capacity signal -- defaults optimistic (``available=True``):
        preflight never invents a false blocker from an inconclusive read.
        """
        argv = self._gcloud(
            "compute", "instances", "describe", self._instance,
            "--zone", self._zone, "--format=value(status)",
        )
        result = self._run(argv)
        sig = self._match_capacity_signature(result.stderr)
        if sig is not None:
            return CapacityReport(available=False, reason=f"capacity signature matched: {sig}")
        return CapacityReport(available=True)

    # ------------------------------------------------------------------
    # ComputeProvider: provision_cpu (folds into the GPU-type create under stage_on_gpu)
    # ------------------------------------------------------------------

    def provision_cpu(self, plan) -> ComputeLease:
        """Create the instance directly on the GPU machine type.

        Under ``stage_on_gpu`` (this method's default/fallback branch)
        there is no separate cheap CPU tier -- see the module docstring for
        why. ``--max-run-duration``/``--instance-termination-action=STOP``
        on this create call is the control-plane cost ceiling;
        ``acquire_gpu`` later just stamps the lease (billing is already
        armed here).

        Under ``cpu_warm_disk_then_gpu_attach`` this delegates to
        :meth:`_provision_cpu_warm_disk` instead, which creates a genuinely
        separate, cheap CPU-only VM -- NO GPU billing starts here for that
        strategy (see that method's docstring).
        """
        if self._vm.tiering_strategy == "cpu_warm_disk_then_gpu_attach":
            return self._provision_cpu_warm_disk(plan)
        vm = self._vm
        args = [
            "compute", "instances", "create", self._instance,
            "--zone", self._zone, "--machine-type", self._gpu_machine_type,
        ]
        if vm.machine_image:
            args += [
                "--source-machine-image", vm.machine_image,
                "--maintenance-policy", "TERMINATE",
                "--no-restart-on-failure",
            ]
        elif vm.image:
            args += [
                "--image-family", vm.image,
                "--image-project", _DEFAULT_IMAGE_PROJECT,
                "--maintenance-policy", "TERMINATE",
                "--boot-disk-size", "1000", "--boot-disk-type", "pd-ssd",
                "--metadata", "install-nvidia-driver=True",
                "--no-restart-on-failure",
            ]
        else:
            raise ValueError(
                "VmSpec must set either `machine_image` or `image` before "
                "provision_cpu can build a `gcloud compute instances create` argv"
            )
        args += [
            f"--max-run-duration={self._max_run_duration_s}s",
            "--instance-termination-action=STOP",
        ]
        argv = self._gcloud(*args)

        result = self._run(argv)
        if result.returncode != 0:
            sig = self._match_capacity_signature(result.stderr)
            if sig is not None:
                raise SandboxRuntimeError(
                    RuntimeCauseKind.backend_unavailable,
                    f"gcp capacity signature matched during create: {sig}",
                )
            raise SandboxRuntimeError(
                RuntimeCauseKind.command_failed,
                f"gcloud compute instances create failed: {(result.stderr or '').strip()[:500]}",
            )

        lease = ComputeLease(
            cloud="gcp", cpu=self._instance, ref=self._instance,
            meta={"zone": self._zone, "project": self._project, "machine_type": self._gpu_machine_type},
        )
        if vm.cache_disk_name:
            lease = self._attach_cache_disk(lease)
        return lease

    def _provision_cpu_warm_disk(self, plan) -> ComputeLease:
        """``cpu_warm_disk_then_gpu_attach``: a cheap CPU VM (no accelerator,
        ``vm.cpu_machine_type``) warms the persistent cache disk, then
        detaches -- NO GPU billing starts here (mirrors
        ``sdar_cpu_stage.sh``'s ``.warm_ok``-sentinel-gated staging phase,
        invoked over ssh against the CPU instance).

        Detaching immediately after warming means the disk is never
        attached to two instances at once, so ``acquire_gpu`` can freely
        re-attach it to the GPU VM without a "disk already in use" error.
        """
        vm = self._vm
        cpu_instance = self._cpu_instance
        args = [
            "compute", "instances", "create", cpu_instance,
            "--zone", self._zone, "--machine-type", self._cpu_machine_type,
        ]
        if vm.machine_image:
            args += ["--source-machine-image", vm.machine_image]
        elif vm.image:
            args += [
                "--image-family", vm.image, "--image-project", _DEFAULT_IMAGE_PROJECT,
                "--boot-disk-size", "200", "--boot-disk-type", "pd-ssd",
            ]
        else:
            raise ValueError(
                "VmSpec must set either `machine_image` or `image` before "
                "provision_cpu can build the cpu_warm_disk_then_gpu_attach CPU-VM create argv"
            )
        self._create_instance_or_raise(self._gcloud(*args))

        lease = ComputeLease(
            cloud="gcp", cpu=cpu_instance, ref=cpu_instance,
            meta={
                "zone": self._zone, "project": self._project,
                "machine_type": self._cpu_machine_type,
                "strategy": "cpu_warm_disk_then_gpu_attach",
            },
        )
        if vm.cache_disk_name:
            lease = self._attach_cache_disk(lease, instance=cpu_instance)
            self._run(self._ssh_argv(self._warm_remote_command(), instance=cpu_instance))
            self._run(self._detach_disk_argv(vm.cache_disk_name, instance=cpu_instance))
        return lease

    def _warm_remote_command(self) -> str:
        """The ``.warm_ok``-sentinel-gated remote shell one-liner: skip
        re-staging if a prior warm already completed on this cache disk
        (mirrors ``sdar_cpu_stage.sh``'s own idempotent/resumable design --
        see that script's header)."""
        mount = _DEFAULT_CACHE_MOUNT
        remote_dir = self._remote_dir
        return (
            f"if [ -f {mount}/.warm_ok ]; then echo '[cpu-warm] already warm'; "
            f"else bash {remote_dir}/scripts/sdar_cpu_stage.sh; fi"
        )

    def _attach_cache_disk(self, lease: ComputeLease, *, instance: str | None = None) -> ComputeLease:
        """Idempotent, fail-soft (mirrors the bash: a create/attach error just
        falls back to boot-disk cache mode -- it never aborts provisioning).

        ``instance`` defaults to ``self._instance`` (the stage_on_gpu
        single-VM shape, unchanged); ``cpu_warm_disk_then_gpu_attach`` passes
        the CPU or GPU instance name explicitly since that strategy
        provisions two distinct VMs.
        """
        disk = self._vm.cache_disk_name
        target = instance or self._instance
        create_argv = self._gcloud(
            "compute", "disks", "create", disk,
            "--zone", self._zone, "--size=1000GB", "--type=pd-ssd",
        )
        self._run(create_argv)
        attach_argv = self._gcloud(
            "compute", "instances", "attach-disk", target,
            "--zone", self._zone, "--disk", disk,
            "--mode=rw", "--device-name=sdar-cache",
        )
        result = self._run(attach_argv)
        meta = dict(lease.meta)
        meta["cache_disk"] = disk
        return replace(lease, disk=(disk if result.returncode == 0 else lease.disk), meta=meta)

    def _detach_disk_argv(self, disk: str, *, instance: str | None = None) -> list[str]:
        """``gcloud compute instances detach-disk ... --disk <disk>`` --
        releases the cache disk from the CPU staging VM once warming
        completes, so ``acquire_gpu`` can re-attach it to the GPU VM
        without a "disk already in use by another instance" error.
        """
        target = instance or self._instance
        return self._gcloud(
            "compute", "instances", "detach-disk", target,
            "--zone", self._zone, "--disk", disk,
        )

    # ------------------------------------------------------------------
    # ComputeProvider: stage (redaction-safe)
    # ------------------------------------------------------------------

    def stage(self, lease: ComputeLease, bundle, run_spec) -> None:
        """Ship a redacted ``run_spec.json`` -- NEVER a raw ``.env``.

        The (redacted) payload is written to a LOCAL scratch file first, and
        only that file's PATH crosses into the ``scp`` argv passed to
        ``runner`` -- no env value or KEY name (secret or not) is ever
        embedded inline in a built argv, matching how the ported bash also
        SCPs a local ``run_spec.json`` file rather than inlining its content
        on the command line.
        """
        bundle = bundle if isinstance(bundle, dict) else {}
        run_spec = run_spec if isinstance(run_spec, dict) else {}
        payload: dict = {}
        payload.update(_redact_env(run_spec))
        payload.update(_redact_env(bundle.get("env", {})))

        remote_dir = self._remote_dir
        self._run(self._ssh_argv(f"mkdir -p {remote_dir}/runs/.cache"))

        local_path = self._write_local_spec(payload)
        scp_argv = self._gcloud(
            "compute", "scp", "--zone", self._zone, "--quiet",
            local_path, f"{self._ssh_user}@{self._instance}:{remote_dir}/runs/.cache/run_spec.json",
        )
        self._run(scp_argv)
        return None

    def _write_local_spec(self, payload: dict) -> str:
        fd, path = tempfile.mkstemp(prefix="run_spec_", suffix=".json")
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh)
        return path

    # ------------------------------------------------------------------
    # ComputeProvider: acquire_gpu (near no-op under stage_on_gpu -- see module docstring)
    # ------------------------------------------------------------------

    def acquire_gpu(self, lease: ComputeLease) -> ComputeLease:
        if self._vm.tiering_strategy == "cpu_warm_disk_then_gpu_attach":
            return self._acquire_gpu_warm_disk(lease)
        return replace(lease, gpu=self._instance)

    def _acquire_gpu_warm_disk(self, lease: ComputeLease) -> ComputeLease:
        """``cpu_warm_disk_then_gpu_attach``: create the GPU VM (billing
        starts HERE, unlike stage_on_gpu where it starts in
        ``provision_cpu``) and attach the already-warmed cache disk.

        Same-zone-locked: a zonal ``pd-ssd`` cannot cross zones (mirrors
        ``gcp_sdar_preflight.sh``'s ``attach_and_mount_cache_disk`` WARN
        path, lines ~289-294). If the disk's recorded zone (stamped into
        ``lease.meta["zone"]`` by ``_provision_cpu_warm_disk``) no longer
        matches the GPU VM's zone, fail soft: create the GPU VM anyway but
        skip the (impossible) cross-zone attach, degrading to
        stage_on_gpu-style GPU-only provisioning rather than raising.
        """
        vm = self._vm
        gpu_instance = self._instance
        args = [
            "compute", "instances", "create", gpu_instance,
            "--zone", self._zone, "--machine-type", self._gpu_machine_type,
        ]
        if vm.machine_image:
            args += [
                "--source-machine-image", vm.machine_image,
                "--maintenance-policy", "TERMINATE",
                "--no-restart-on-failure",
            ]
        elif vm.image:
            args += [
                "--image-family", vm.image,
                "--image-project", _DEFAULT_IMAGE_PROJECT,
                "--maintenance-policy", "TERMINATE",
                "--boot-disk-size", "1000", "--boot-disk-type", "pd-ssd",
                "--metadata", "install-nvidia-driver=True",
                "--no-restart-on-failure",
            ]
        else:
            raise ValueError(
                "VmSpec must set either `machine_image` or `image` before "
                "acquire_gpu can build the cpu_warm_disk_then_gpu_attach GPU-VM create argv"
            )
        args += [
            f"--max-run-duration={self._max_run_duration_s}s",
            "--instance-termination-action=STOP",
        ]
        self._create_instance_or_raise(self._gcloud(*args))

        meta = dict(lease.meta)
        meta.update({"zone": self._zone, "project": self._project, "machine_type": self._gpu_machine_type})
        new_lease = replace(lease, gpu=gpu_instance, ref=gpu_instance, meta=meta)

        disk = vm.cache_disk_name
        disk_zone = lease.meta.get("zone") if isinstance(lease.meta, dict) else None
        same_zone = disk_zone is None or disk_zone == self._zone
        if disk and same_zone:
            new_lease = self._attach_cache_disk(new_lease, instance=gpu_instance)
        return new_lease

    # ------------------------------------------------------------------
    # ComputeProvider: launch
    # ------------------------------------------------------------------

    def launch(self, lease: ComputeLease, run_spec) -> RunHandle:
        """Detached SSH launch of ``scripts/sdar_gcp_run.sh``, which in turn
        execs ``python -m backend.cli reproduce <paper_id> --mode <mode>
        --sandbox local --model <model> --project-id <project_id>``.

        The SSH wrapper argv mirrors the ported bash (``setsid nohup bash
        ... > runs/sdar_gcp_run.out 2>&1 </dev/null &``, detached so it
        survives the SSH session closing); the underlying CLI invocation
        shape is additionally surfaced as structured
        ``RunHandle.meta["cli_argv_hint"]`` so a caller/test can assert on it
        without parsing the remote shell string.
        """
        run_spec = run_spec if isinstance(run_spec, dict) else {}
        paper_id = run_spec.get("paper_id") or "2605.15155"
        mode = run_spec.get("mode") or "rlm"
        model = run_spec.get("model") or "foundry"
        project_id = run_spec.get("project_id") or "sdar_gcp_run"
        remote_dir = self._remote_dir

        remote_cmd = (
            f"cd {remote_dir} && "
            f"( setsid nohup bash scripts/sdar_gcp_run.sh "
            f"--run-spec {remote_dir}/runs/.cache/run_spec.json "
            f"> runs/sdar_gcp_run.out 2>&1 </dev/null & )"
        )
        self._run(self._ssh_argv(remote_cmd))

        cli_argv_hint = [
            "backend.cli", "reproduce", paper_id,
            "--mode", mode, "--sandbox", "local",
            "--model", model, "--project-id", project_id,
        ]
        return RunHandle(
            id=project_id,
            lease=lease,
            meta={"paper_id": paper_id, "mode": mode, "model": model, "cli_argv_hint": cli_argv_hint},
        )

    # ------------------------------------------------------------------
    # ComputeProvider: watch
    # ------------------------------------------------------------------

    def watch(self, handle: RunHandle) -> Iterator[RunStatus]:
        """Poll ``describe`` status until the VM leaves RUNNING.

        A non-RUNNING VM (stopped by the max-run-duration ceiling, a
        preemption, or an external actor) is classified
        ``stopped_uncollected`` -- the state machine driving this provider
        reacts by calling ``recover()`` rather than assuming a graceful
        COLLECT already ran. ``synced`` always reports ``False`` here: this
        method itself never pulls artifacts (that is ``collect``'s job), so
        nothing is off-box "as of this poll." Sleeps ``poll_interval_s``
        (constructor kwarg, default 30s; injectable ``sleep``) between polls
        while the VM stays RUNNING, and returns after the first non-RUNNING
        poll -- so a caller `for status in provider.watch(handle): ...`
        loop terminates on its own once the VM leaves service.
        """
        while True:
            argv = self._gcloud(
                "compute", "instances", "describe", self._instance,
                "--zone", self._zone, "--format=value(status)",
            )
            result = self._run(argv)
            status = (result.stdout or "").strip()
            if status == "RUNNING":
                run_status = RunStatus(state="running", detail=status, synced=False)
            elif status:
                run_status = RunStatus(state="stopped_uncollected", detail=status, synced=False)
            else:
                run_status = RunStatus(state="stalled", detail="empty status", synced=False)
            yield run_status
            if run_status.state != "running":
                return
            self._sleep(self._poll_interval_s)

    # ------------------------------------------------------------------
    # ComputeProvider: collect
    # ------------------------------------------------------------------

    def collect(self, handle: RunHandle) -> ReportBundle:
        """Tar the run's report artifacts remotely, then ``scp`` the tarball
        down (mirrors ``sdar_gcp_watch.sh``'s ``pull_report_and_log``)."""
        project_id = handle.id
        remote_dir = self._remote_dir
        remote_run_dir = f"{remote_dir}/runs/{project_id}"
        tgz_path = f"/tmp/{project_id}_report.tgz"
        tar_cmd = (
            f"cd {remote_run_dir} 2>/dev/null && tar czf {tgz_path} "
            f"final_report.json final_report.md demo_status.json cost_ledger.jsonl "
            f"experiment_runs.jsonl dashboard_events.jsonl code/metrics.json 2>/dev/null; echo tarred"
        )
        self._run(self._ssh_argv(tar_cmd))

        scp_argv = self._gcloud(
            "compute", "scp", "--zone", self._zone, "--quiet",
            f"{self._ssh_user}@{self._instance}:{tgz_path}", tgz_path,
        )
        result = self._run(scp_argv)
        ok = result.returncode == 0
        return ReportBundle(
            ok=ok,
            report_path=f"runs/{project_id}/final_report.json" if ok else None,
            artifacts={"tarball": tgz_path} if ok else {},
            reason="" if ok else (result.stderr or "collect failed").strip()[:500],
        )

    # ------------------------------------------------------------------
    # ComputeProvider: release_gpu / teardown (stop, with local-SSD retry)
    # ------------------------------------------------------------------

    def _stop_argv(self, *, discard_local_ssd: bool = False) -> list[str]:
        args = ["compute", "instances", "stop", self._instance, "--zone", self._zone, "--quiet"]
        if discard_local_ssd:
            args.append("--discard-local-ssd=true")
        return self._gcloud(*args)

    def _stop_with_retry(self) -> VmExecResult:
        """``gcloud compute instances stop`` -- an a2-ultragpu VM's mandatory
        local SSD refuses a plain stop; retry once with
        ``--discard-local-ssd=true`` (mirrors ``sdar_gcp_watch.sh``'s
        ``vm_stop``). The local SSD holds only ephemeral scratch; run
        artifacts live on the (persisted) boot disk."""
        result = self._run(self._stop_argv())
        if result.returncode != 0 and _LOCAL_SSD_RETRY_RE.search(result.stderr or ""):
            result = self._run(self._stop_argv(discard_local_ssd=True))
        return result

    def release_gpu(self, lease: ComputeLease) -> None:
        self._stop_with_retry()
        return None

    def teardown(self, lease: ComputeLease, *, reason: str) -> None:
        """Stop the VM (never delete it).

        The ported bash treats the named VM (``sdar-a100-od``) as a
        persistent, reusable resource across runs -- none of the source
        scripts ever issue ``gcloud compute instances delete``. So
        ``teardown`` mirrors ``release_gpu`` (stop, with the same local-SSD
        retry) rather than actually destroying the instance. ``reason`` is
        accepted for interface parity with ``ComputeProvider`` even though
        this implementation does not branch on it -- the VM is always just
        stopped, never deleted.
        """
        self._stop_with_retry()
        return None

    # ------------------------------------------------------------------
    # ComputeProvider: recover
    # ------------------------------------------------------------------

    def recover(self, lease_ref: str) -> ReportBundle | None:
        """Re-attach to a stopped-but-not-deleted VM by instance name and
        pull whatever the last run wrote to its (persisted) boot disk."""
        if not lease_ref:
            return None
        describe_argv = self._gcloud(
            "compute", "instances", "describe", lease_ref,
            "--zone", self._zone, "--format=value(status)",
        )
        result = self._run(describe_argv)
        status = (result.stdout or "").strip()
        if not status:
            return None
        if status != "RUNNING":
            start_argv = self._gcloud("compute", "instances", "start", lease_ref, "--zone", self._zone)
            self._run(start_argv)
        handle = RunHandle(id=lease_ref, lease=ComputeLease(cloud="gcp", cpu=lease_ref, ref=lease_ref))
        return self.collect(handle)


__all__ = ["VmComputeProvider", "VmExecResult"]
