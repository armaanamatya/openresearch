"""Pure absolute-epoch deadline helpers for durable, restart-safe budgets.

WS3 (durable cloud-native orchestration) design
(``docs/history/specs/2026-07-10-durable-cloud-native-orchestration-ws3-design.md``):
when a controller pod is killed and a successor **adopts** an in-flight GPU
cell Job, the successor must inherit the run's REMAINING wall-clock budget,
not a fresh full one -- otherwise every restart doubles the GPU cost. Today's
deadlines are computed as ``time.monotonic() + timeout``, which resets to a
fresh full budget on every process restart since monotonic clocks are not
comparable across processes.

This module fixes that by recording an **absolute wall-clock (epoch)
deadline** instead: two downstream owners (``k8s_job_cell_runner._watch_job``,
``k8s_job_backend``) persist the record returned by :func:`make_deadline` to
the run bucket at submit time and re-read it (via :func:`parse`) on adopt, so
a successor computes the correct remaining budget via :func:`remaining_s`
regardless of how long the predecessor lived or when it died.

Clock discipline
-----------------
This module is entirely PURE: no cloud SDK calls, no I/O, and -- unlike most
of this package -- it does not even import ``time``. Every function that
reasons about "now" takes ``now_epoch: float`` explicitly from the caller
(mirrors :mod:`backend.services.runtime.blob_lease`'s clock discipline), so
deadline/expiry arithmetic is fully deterministic under test.
"""

from __future__ import annotations

import json

__all__ = [
    "make_deadline",
    "remaining_s",
    "is_expired",
    "serialize",
    "parse",
]

_DEADLINE_VERSION = 1


def make_deadline(now_epoch: float, budget_s: float) -> dict:
    """Build an absolute-epoch deadline record.

    ``deadline_epoch = now_epoch + max(0.0, budget_s)`` -- a negative
    ``budget_s`` clamps to ``0.0``, so the record is already expired as of
    ``now_epoch`` rather than yielding a deadline in the past relative to
    some other instant.

    Returns ``{"version": 1, "created_epoch": now_epoch, "budget_s":
    <clamped>, "deadline_epoch": <now_epoch + clamped>}``.
    """
    clamped_budget = max(0.0, budget_s)
    return {
        "version": _DEADLINE_VERSION,
        "created_epoch": now_epoch,
        "budget_s": clamped_budget,
        "deadline_epoch": now_epoch + clamped_budget,
    }


def remaining_s(record: dict, now_epoch: float) -> float:
    """Seconds left until ``record`` expires, as of ``now_epoch``.

    ``max(0.0, record["deadline_epoch"] - now_epoch)`` -- NEVER negative,
    so a caller can always treat this as a safe budget to pass onward
    (e.g. into a subprocess timeout) without an extra clamp of its own.
    """
    return max(0.0, record["deadline_epoch"] - now_epoch)


def is_expired(record: dict, now_epoch: float) -> bool:
    """Whether ``record`` has expired as of ``now_epoch``.

    ``now_epoch >= record["deadline_epoch"]`` -- the boundary instant
    itself counts as expired.
    """
    return now_epoch >= record["deadline_epoch"]


def serialize(record: dict) -> bytes:
    """Deterministic byte encoding of ``record`` for durable persistence.

    ``sort_keys=True`` so two calls on equal records always produce
    byte-identical output, regardless of key insertion order.
    """
    return json.dumps(record, sort_keys=True).encode("utf-8")


def parse(data: bytes) -> dict:
    """Inverse of :func:`serialize`."""
    return json.loads(data.decode("utf-8"))
