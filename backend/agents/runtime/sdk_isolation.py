"""Thread-isolated execution of claude-agent-sdk async calls.

The SDK has a known nested-generator aclose race (PR-μ runtime resilience
spec, 2026-05-26). When the SDK's internal generator is being closed while
another generator is mid-stream, Python raises RuntimeError("aclose():
asynchronous generator is already running") and the pipeline can unwind.

This module extracts the thread-isolation pattern used in rdr/agent.py and
rlm_query.py so every SDK call in the project inherits the workaround.

The aclose race always fires at cleanup AFTER the coroutine produced its
result — the result is already captured in `captured_result` before the
race fires. Genuine exceptions from inside the coroutine body propagate.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any, Coroutine, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_ACLOSE_MARKERS = (
    "aclose(): asynchronous generator is already running",
    "aclose(): synchronous generator already running",
)


def _is_aclose_race(exc: BaseException) -> bool:
    msg = str(exc)
    return any(marker in msg for marker in _ACLOSE_MARKERS)


async def run_isolated(coro: Coroutine[Any, Any, T]) -> T:
    """Run ``coro`` inside a dedicated worker thread with its own event loop.

    The aclose race that fires at generator cleanup is swallowed because it
    cannot affect the value the coroutine already returned. Any other exception
    from the coroutine body propagates to the caller unchanged.

    Set ``REPROLAB_SDK_ISOLATION_DISABLED=true`` to bypass the workaround and
    run the coroutine directly on the calling event loop — useful when
    debugging a suspected isolation-induced issue.
    """
    if os.environ.get("REPROLAB_SDK_ISOLATION_DISABLED", "").lower() in {"true", "1", "yes"}:
        return await coro

    captured_result: list[T] = []
    captured_exception: list[BaseException] = []
    done = threading.Event()

    def worker() -> None:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(coro)
                captured_result.append(result)
            except BaseException as exc:
                if _is_aclose_race(exc) and captured_result:
                    # Race fired AFTER result was returned — safe to swallow.
                    logger.debug("sdk_isolation: aclose race swallowed at cleanup")
                else:
                    captured_exception.append(exc)
            finally:
                try:
                    loop.close()
                except RuntimeError as exc:
                    if not _is_aclose_race(exc):
                        raise
                    # The race can also fire DURING loop.close() — swallow there too.
                    logger.debug("sdk_isolation: aclose race swallowed at loop close")
        finally:
            done.set()

    thread = threading.Thread(target=worker, name="sdk-isolation", daemon=True)
    thread.start()
    # Yield the calling event loop while we wait — non-blocking from caller's perspective.
    while not done.is_set():
        await asyncio.sleep(0.05)
    thread.join(timeout=1.0)

    if captured_exception:
        raise captured_exception[0]
    if not captured_result:
        raise RuntimeError("sdk_isolation worker completed without result or exception")
    return captured_result[0]


__all__ = ["run_isolated"]
