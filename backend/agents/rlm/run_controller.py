"""Pure/injectable driver-controller logic for WS3 durable orchestration.

WS3 design (durable cloud-native orchestration):
``docs/superpowers/specs/2026-07-10-durable-cloud-native-orchestration-ws3-design.md``.

**Phase-1 scope only.** Everything in this module is either a pure
computation or a thin wrapper around an injected dependency
(:func:`acquire_drive_lease` duck-types its ``lease`` argument so it is
unit-testable with a fake double, no cloud SDK required). The real GKE
Deployment submit that runs a durable controller Pod, and the reaper that
cleans up a superseded driver's Jobs, are wired in a LATER phase -- this
module makes no cluster call of any kind.

Gated on ``OPENRESEARCH_DURABLE_CONTROLLER`` (default OFF); see
:func:`durable_controller_enabled`.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

from backend.agents.rlm.feature_flags import env_truthy

__all__ = [
    "durable_controller_enabled",
    "durable_controller_default_for_sandbox",
    "build_controller_command",
    "classify_controller_exit",
    "acquire_drive_lease",
]


def durable_controller_enabled() -> bool:
    """``OPENRESEARCH_DURABLE_CONTROLLER`` -- default OFF, read at call time.

    This global helper stays env-only/default-false: cell-fencing helpers key on
    it, so it must NOT flip to default-on. The launch-boundary default lives in
    :func:`durable_controller_default_for_sandbox` instead.
    """
    return env_truthy("OPENRESEARCH_DURABLE_CONTROLLER")


def durable_controller_default_for_sandbox(sandbox: str) -> bool:
    """Launch-boundary default: durable controller ON for ``sandbox=="gcp"``.

    ON by default for gcp (operational, fail-soft path); ``OPENRESEARCH_DURABLE
    _CONTROLLER=0``/``false``/``no`` opts out. Non-gcp sandboxes are never
    durable, regardless of the flag, so a local/docker/runpod run is
    byte-identical to today.
    """
    if sandbox != "gcp":
        return False
    raw = os.environ.get("OPENRESEARCH_DURABLE_CONTROLLER", "").strip().lower()
    return raw not in ("0", "false", "no")


def build_controller_command(paper: str, project_id: str) -> list[str]:
    """The argv a durable controller Pod runs to drive one reproduction.

    Deliberately the ``campaign`` subcommand, not ``reproduce``: ``campaign``
    is the ledger-resumable outer driver (``campaign/{campaign.json,
    attempts.jsonl}``), so ``--resume`` lets a freshly (re)scheduled
    controller Pod pick a run back up after a Pod restart/reschedule instead
    of starting over.
    """
    return [
        "python",
        "-m",
        "backend.cli",
        "campaign",
        paper,
        "--project-id",
        project_id,
        "--resume",
    ]


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
