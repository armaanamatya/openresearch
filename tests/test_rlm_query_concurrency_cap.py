"""Phase 5 — code-level sub-RLM concurrency cap (FM-008).

The f848d7e system prompt tells the root to fan out up to 8 concurrent sub-RLMs,
but the cap was prompt-only. A module-level bounded semaphore in complete() bounds
concurrent OAuth/bundled-CLI sub-calls regardless of what the root does.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest


def _run_concurrent_completes(n_callers: int, monkeypatch) -> int:
    """Fire n_callers complete() calls concurrently; return the max observed in-flight."""
    from backend.services.context.workspace.tools import rlm_query as rq

    in_flight = 0
    max_in_flight = 0
    lock = threading.Lock()

    async def fake_async_complete(*, system, user):
        nonlocal in_flight, max_in_flight
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        try:
            await _async_sleep(0.15)
        finally:
            with lock:
                in_flight -= 1
        return ("ok", {})

    # avoid importing asyncio at module top for the helper
    import asyncio
    async def _async_sleep(s):
        await asyncio.sleep(s)

    client = rq.ClaudeLlmClient(model="m", max_turns=1)
    with patch.object(client, "_async_complete", fake_async_complete):
        with ThreadPoolExecutor(max_workers=n_callers) as pool:
            futures = [
                pool.submit(client.complete, system="s", user="u")
                for _ in range(n_callers)
            ]
            for f in futures:
                f.result()
    return max_in_flight


def test_concurrency_is_capped(monkeypatch):
    monkeypatch.setenv("REPROLAB_SUBRLM_MAX_CONCURRENCY", "2")
    # Force the semaphore to rebuild for this limit.
    from backend.services.context.workspace.tools import rlm_query as rq
    rq._subrlm_semaphore = None
    max_in_flight = _run_concurrent_completes(6, monkeypatch)
    assert max_in_flight <= 2, f"cap=2 but {max_in_flight} ran concurrently"


def test_concurrency_disabled_when_zero(monkeypatch):
    monkeypatch.setenv("REPROLAB_SUBRLM_MAX_CONCURRENCY", "0")
    from backend.services.context.workspace.tools import rlm_query as rq
    rq._subrlm_semaphore = None
    max_in_flight = _run_concurrent_completes(5, monkeypatch)
    assert max_in_flight >= 2, "cap=0 should not serialize calls"
