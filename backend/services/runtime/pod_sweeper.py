"""Sweeper for orphan RunPod pods.

A run that crashes before ``RunpodBackend.destroy`` can leave a billable
pod alive forever — the lifecycle's normal ``finally`` cleanup path is
bypassed.  Production also runs into pods that survive ``delete_on_destroy=
False`` operator overrides intended for debugging.

This module sweeps the RunPod account: list pods, identify those older
than a max-age threshold OR matching a name pattern that indicates they
belong to terminated reproductions, and delete them.

Public API:

  * :func:`list_pods` — thin wrapper over RunPod's GET /v1/pods returning
    a normalised list of dicts.
  * :func:`sweep_stale_pods` — delete every pod older than
    ``max_age_seconds``.  Returns a summary dict the caller can log.
  * CLI entry point: ``python -m backend.services.runtime.pod_sweeper``
    runs ``sweep_stale_pods`` with the operator-configured threshold.

Design contract:

  * Pure function calls — no global state.
  * Read-only by default — the sweep call takes a ``dry_run`` flag for
    safe inspection.
  * Fail-soft on individual pod failures — if one delete fails, the
    sweeper still continues the loop and reports per-pod outcomes.
  * Robust to RunPod's evolving response shapes (list vs ``{"pods": [...]}``).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Default max-age before a pod is considered stale + sweepable.
# 2 hours matches the documented production policy.
DEFAULT_MAX_AGE_SECONDS: int = 2 * 60 * 60

_RUNPOD_BASE_URL: str = "https://rest.runpod.io/v1"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PodInfo:
    """Normalised view of a RunPod pod."""

    id: str
    name: str
    status: str
    created_at: str
    age_seconds: int | None
    cost_per_hour: float | None
    gpu_type: str

    def is_stale(self, max_age_seconds: int) -> bool:
        if self.age_seconds is None:
            return False
        return self.age_seconds > max_age_seconds


@dataclass(slots=True)
class SweepReport:
    """Aggregated sweep outcome — caller logs or surfaces this."""

    swept: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)
    total_pods: int = 0
    estimated_savings_per_hour: float = 0.0

    def summary(self) -> str:
        bits = [f"{len(self.swept)}/{self.total_pods} swept"]
        if self.skipped:
            bits.append(f"{len(self.skipped)} skipped")
        if self.errors:
            bits.append(f"{len(self.errors)} errors")
        if self.estimated_savings_per_hour > 0:
            bits.append(f"saved ~${self.estimated_savings_per_hour:.2f}/hr")
        return ", ".join(bits)


# ---------------------------------------------------------------------------
# RunPod API helpers
# ---------------------------------------------------------------------------


def _api_key_from_env() -> str | None:
    return os.environ.get("REPROLAB_RUNPOD_API_KEY") or os.environ.get("RUNPOD_API_KEY")


def _parse_age_seconds(created_at: str | None) -> int | None:
    """Parse RunPod's ISO timestamp into an age in seconds.

    Returns None on any parse failure (RunPod has changed timestamp shapes
    historically — fail-soft instead of raising).
    """
    if not created_at:
        return None
    from datetime import datetime, timezone
    try:
        normalized = created_at.rstrip("Z")
        if "+" not in normalized and "-" not in normalized[10:]:
            normalized += "+00:00"
        elif normalized.endswith("UTC"):
            normalized = normalized[:-3].strip() + "+00:00"
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int((datetime.now(timezone.utc) - dt).total_seconds())
    except (ValueError, TypeError):
        return None


def list_pods(api_key: str | None = None, *, timeout: float = 10.0) -> list[PodInfo]:
    """Return every pod owned by the account behind ``api_key``.

    Falls back to ``REPROLAB_RUNPOD_API_KEY`` / ``RUNPOD_API_KEY`` env.
    Returns an empty list if the API key is missing (silent — the sweeper
    is harmless without credentials).
    """
    key = api_key or _api_key_from_env()
    if not key:
        return []
    headers = {"Authorization": f"Bearer {key}"}
    try:
        with httpx.Client(timeout=timeout, headers=headers) as client:
            resp = client.get(f"{_RUNPOD_BASE_URL}/pods")
    except httpx.HTTPError as exc:
        logger.warning("pod_sweeper.list_pods: HTTP error %s", exc)
        return []
    if resp.status_code != 200:
        logger.warning(
            "pod_sweeper.list_pods: HTTP %s — %s", resp.status_code, resp.text[:200]
        )
        return []
    try:
        body = resp.json()
    except ValueError:
        return []
    raw = body if isinstance(body, list) else body.get("pods") or []
    pods: list[PodInfo] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        pid = str(entry.get("id") or "")
        if not pid:
            continue
        pods.append(PodInfo(
            id=pid,
            name=str(entry.get("name") or ""),
            status=str(entry.get("desiredStatus") or "?"),
            created_at=str(entry.get("createdAt") or ""),
            age_seconds=_parse_age_seconds(entry.get("createdAt")),
            cost_per_hour=_parse_cost(entry.get("costPerHr")),
            gpu_type=str((entry.get("machine") or {}).get("gpuTypeId") or "?"),
        ))
    return pods


def _parse_cost(raw: Any) -> float | None:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def delete_pod(pod_id: str, api_key: str | None = None, *, timeout: float = 15.0) -> bool:
    """Delete a single pod via DELETE /v1/pods/<id>.  Returns success bool."""
    key = api_key or _api_key_from_env()
    if not key:
        return False
    headers = {"Authorization": f"Bearer {key}"}
    try:
        with httpx.Client(timeout=timeout, headers=headers) as client:
            resp = client.delete(f"{_RUNPOD_BASE_URL}/pods/{pod_id}")
    except httpx.HTTPError as exc:
        logger.warning("pod_sweeper.delete_pod %s: HTTP error %s", pod_id, exc)
        return False
    if resp.status_code not in (200, 202, 204):
        logger.warning(
            "pod_sweeper.delete_pod %s: HTTP %s — %s",
            pod_id, resp.status_code, resp.text[:200],
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def sweep_stale_pods(
    *,
    max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
    dry_run: bool = False,
    api_key: str | None = None,
    preserve_pod_ids: tuple[str, ...] = (),
) -> SweepReport:
    """Delete every pod older than ``max_age_seconds``.

    ``preserve_pod_ids`` lets the operator pin specific pods (e.g. an
    actively-running reproduction) so a routine sweep never kills them.

    ``dry_run=True`` makes the call safe for inspection — pods are
    classified but no DELETE requests are issued.

    Returns a ``SweepReport`` with per-pod outcomes and an estimated
    $/hour savings figure derived from the killed pods' costPerHr.
    """
    pods = list_pods(api_key=api_key)
    report = SweepReport(total_pods=len(pods))
    preserved = set(preserve_pod_ids)
    for pod in pods:
        if pod.id in preserved:
            report.skipped.append((pod.id, "preserved by caller"))
            continue
        if not pod.is_stale(max_age_seconds):
            age = "?" if pod.age_seconds is None else f"{pod.age_seconds // 60}min"
            report.skipped.append((pod.id, f"age {age} <= threshold"))
            continue
        if dry_run:
            report.swept.append(pod.id)
            if pod.cost_per_hour:
                report.estimated_savings_per_hour += pod.cost_per_hour
            continue
        ok = delete_pod(pod.id, api_key=api_key)
        if ok:
            report.swept.append(pod.id)
            if pod.cost_per_hour:
                report.estimated_savings_per_hour += pod.cost_per_hour
        else:
            report.errors.append((pod.id, "delete failed — see warning log"))
    logger.info("pod_sweeper: %s", report.summary())
    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _cli() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Sweep stale RunPod pods owned by REPROLAB_RUNPOD_API_KEY.",
    )
    parser.add_argument(
        "--max-age-seconds", type=int, default=DEFAULT_MAX_AGE_SECONDS,
        help="Delete pods older than this (default: 2h).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List what would be deleted without deleting.",
    )
    parser.add_argument(
        "--preserve", action="append", default=[],
        help="Pod ID to keep alive even if stale (may be repeated).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    report = sweep_stale_pods(
        max_age_seconds=args.max_age_seconds,
        dry_run=args.dry_run,
        preserve_pod_ids=tuple(args.preserve),
    )
    print(report.summary())
    for pid in report.swept:
        print(f"  {'WOULD DELETE' if args.dry_run else 'DELETED'} {pid}")
    for pid, reason in report.skipped:
        print(f"  SKIPPED {pid} — {reason}")
    for pid, reason in report.errors:
        print(f"  ERROR {pid} — {reason}")


if __name__ == "__main__":
    _cli()


__all__ = [
    "DEFAULT_MAX_AGE_SECONDS",
    "PodInfo",
    "SweepReport",
    "delete_pod",
    "list_pods",
    "sweep_stale_pods",
]
