"""Periodic background sweep of stale RunPod pods.

Wired into the FastAPI app lifespan (see backend/app.py). Calls
sweep_stale_pods on a configurable interval. Fail-soft: any exception is
logged but the scheduler keeps running.

Disabled when:
  - REPROLAB_RUNPOD_API_KEY is unset (no RunPod usage)
  - OPENRESEARCH_POD_SWEEP_ENABLED=false
"""
from __future__ import annotations

import asyncio
import logging
import os

from backend.services.runtime.pod_sweeper import sweep_stale_pods

logger = logging.getLogger(__name__)


class PodSweepScheduler:
    """Background asyncio task that runs sweep_stale_pods periodically."""

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._stop_event: asyncio.Event | None = None

    def _enabled(self) -> bool:
        if not os.environ.get("REPROLAB_RUNPOD_API_KEY"):
            return False
        val = os.environ.get("OPENRESEARCH_POD_SWEEP_ENABLED", "true").lower()
        if val in {"false", "0", "no", "off"}:
            return False
        return True

    def _interval_s(self) -> float:
        try:
            return float(os.environ.get("OPENRESEARCH_POD_SWEEP_INTERVAL_S", "1800"))
        except ValueError:
            return 1800.0

    def _max_age_s(self) -> int:
        try:
            return int(os.environ.get("OPENRESEARCH_POD_SWEEP_MAX_AGE_S", "7200"))
        except ValueError:
            return 7200

    async def start(self) -> None:
        if not self._enabled():
            logger.info(
                "pod_sweep_scheduler: disabled "
                "(no REPROLAB_RUNPOD_API_KEY or OPENRESEARCH_POD_SWEEP_ENABLED=false)"
            )
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        interval = self._interval_s()
        max_age = self._max_age_s()
        logger.info(
            "pod_sweep_scheduler: starting (interval=%.1fs, max_age=%ds)",
            interval,
            max_age,
        )
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                summary = sweep_stale_pods(max_age_seconds=max_age, dry_run=False)
                logger.info("pod_sweep_scheduler: sweep complete: %s", summary)
            except Exception as exc:
                logger.warning("pod_sweep_scheduler: sweep failed: %s", exc)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass  # interval elapsed; loop continues

    async def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                self._task.cancel()
        self._task = None
        self._stop_event = None


__all__ = ["PodSweepScheduler"]
