"""Pure helpers for cross-cloud durable controller orchestration.

WS3 design (durable cloud-native orchestration):
``docs/superpowers/specs/2026-07-10-durable-cloud-native-orchestration-ws3-design.md``.

Everything here is a pure computation or a thin wrapper around an injected
lease. Cluster submission, in-pod heartbeats, and reaping are wired by
``controller_cluster``/``controller_entry`` for GKE and AKS.

Gated on ``OPENRESEARCH_DURABLE_CONTROLLER`` (default OFF); see
:func:`durable_controller_enabled`.
"""

from __future__ import annotations

from typing import Any, Protocol

from backend.agents.rlm.feature_flags import env_truthy

__all__ = [
    "durable_controller_enabled",
    "build_controller_command",
    "classify_controller_exit",
    "controller_job_exit_code",
    "validate_controller_image",
    "acquire_drive_lease",
]


def durable_controller_enabled() -> bool:
    """``OPENRESEARCH_DURABLE_CONTROLLER`` -- default OFF, read at call time.

    Stays env-only/default-false to keep the byte-identical-OFF invariant: cell
    fencing and the ``_should_use_durable_controller`` predicate both key on it.
    "On by default for gcp" is a DEPLOYMENT property — the gcp deployment env /
    run-spec sets ``=1`` — not a code default flip.
    """
    return env_truthy("OPENRESEARCH_DURABLE_CONTROLLER")


def build_controller_command(
    paper: str,
    project_id: str,
    *,
    cloud: str = "gcp",
    runs_root: str = "/mnt/reprolab/controller-runs",
    run_spec: str | None = None,
    root_model: str | None = None,
    execution_mode: str = "max",
    gpu_mode: str = "auto",
    minimize_compute: bool = False,
) -> list[str]:
    """The argv a durable controller Pod runs to drive one reproduction.

    The wrapper owns the cloud CAS lease and heartbeat, then launches the
    ledger-resumable ``campaign --resume`` child on the mounted runs PVC.
    """
    command = [
        "python",
        "-m",
        "backend.agents.rlm.controller_entry",
        "--cloud",
        cloud,
        "--paper",
        paper,
        "--project-id",
        project_id,
        "--runs-root",
        runs_root,
        "--execution-mode",
        execution_mode,
        "--gpu-mode",
        gpu_mode,
    ]
    if minimize_compute:
        command.append("--minimize-compute")
    if run_spec:
        command.extend(["--run-spec", run_spec])
    if root_model:
        command.extend(["--root-model", root_model])
    return command


def classify_controller_exit(code: int) -> str:
    """Map a controller process's exit code to a coarse lifecycle outcome.

    ``0`` -> ``"complete"`` (a terminal state was reached cleanly);
    ``2`` -> ``"paused"`` (the campaign CLI's own checkpoint/refusal
    convention -- resumable via ``--resume``); ``3`` -> ``"money_halt"``
    (the campaign CLI's own ``[MONEY-HALT]`` -- ``cmd_campaign`` catching
    ``CampaignLedgerError`` on ledger corruption/unwritability and halting
    BEFORE any further spend, ``backend/cli.py``'s ``cmd_campaign``). This is
    a deliberate fail-closed safety stop, not a process crash -- a durable
    reaper/restarter MUST NOT respawn/retry on ``"money_halt"`` the way it
    might for a genuine ``"crash"``; anything else (a real crash, ``137``/
    SIGKILL, an uncaught exception, ...) -> ``"crash"``.
    """
    if code == 0:
        return "complete"
    if code == 2:
        return "paused"
    if code == 3:
        return "money_halt"
    return "crash"


def controller_job_exit_code(code: int) -> int:
    """Map intentional campaign stops to Job success to suppress K8s retries.

    ``paused`` requires operator input and ``money_halt`` is deliberately
    fail-closed. Retrying either via ``backoffLimit`` would spend or loop
    without changing the condition. Genuine crashes retain their non-zero
    code and remain eligible for Kubernetes restart.
    """
    if classify_controller_exit(code) in {"paused", "money_halt"}:
        return 0
    return code


def validate_controller_image(image: str, *, env_name: str) -> str:
    """Require a non-floating image reference for a durable controller."""
    image = image.strip()
    final_component = image.rsplit("/", 1)[-1]
    pinned = "@sha256:" in image or ":" in final_component
    if not image or not pinned or final_component.endswith(":latest"):
        raise RuntimeError(
            f"durable controller requires {env_name} to be a pinned image "
            "tag or sha256 digest (never :latest)"
        )
    return image


class _AcquirableLease(Protocol):
    """Structural type for anything exposing a ``BlobLease.acquire``-shaped
    method -- lets :func:`acquire_drive_lease` be exercised against a plain
    test double as well as the real ``BlobLease``."""

    def acquire(self, run_id: str, owner_id: str, now_epoch: float) -> Any: ...


def acquire_drive_lease(
    lease: _AcquirableLease, run_id: str, owner_id: str, now_epoch: float
) -> Any:
    """Try to become (or remain) the sole driver of ``run_id``.

    Thin wrapper over ``lease.acquire(...)`` -- ``lease`` is duck-typed (any
    object exposing ``.acquire``, e.g. ``BlobLease`` or a test fake), so this
    is unit-testable without any cloud dependency. Returns whatever
    ``lease.acquire`` returns: a fence token on success, or ``None`` when
    another driver already owns ``run_id``. ``None`` means the caller MUST
    NOT drive -- it must not submit or adopt any Job, and must not write any
    evidence for ``run_id``.
    """
    return lease.acquire(run_id, owner_id, now_epoch)
