"""Thread-isolated SDK execution with PR-π aclose retry resilience.

Spec section: PR-π Module A — SDK call resilience.  The SDK can raise
``RuntimeError("aclose(): asynchronous generator is already running")`` either
after a result was captured or during streaming before any result exists.  This
module keeps the post-result swallow and retries pre-result races with a fresh
coroutine factory.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import os
import threading
import warnings
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, Literal, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_DEFAULT_MAX_RETRIES = 2
_STDERR_EXCERPT_CHARS = 4096

_ACLOSE_MARKERS = (
    "aclose(): asynchronous generator is already running",
    "aclose(): synchronous generator already running",
)

IsolationOutcomeKind = Literal[
    "ok",
    "aclose_post_result_swallowed",
    "aclose_pre_result_retried",
    "aclose_pre_result_exhausted",
    "real_exception",
]


class IsolationFailureKind(str, Enum):
    """Classify SDK isolation failures and recoverable aclose races."""

    ACLOSE_PRE_RESULT = "aclose_pre_result"
    ACLOSE_POST_RESULT = "aclose_post_result"
    REAL_EXCEPTION = "real_exception"


@dataclass(frozen=True)
class IsolationOutcome:
    """Structured diagnostics from the most recent isolated SDK call."""

    kind: IsolationOutcomeKind
    attempt_count: int
    stderr_excerpt: str = ""


class IsolationFailure(RuntimeError):
    """Raised when SDK isolation exhausts retryable pre-result aclose races.

    Pre: ``outcome.kind`` describes a terminal isolation failure.
    Post: callers can inspect ``.outcome`` and ``.kind`` without parsing text.
    Side effects: none.
    Exceptions raised: this exception is itself raised by ``run_isolated`` when
    a retryable SDK aclose race never reaches a result.
    """

    def __init__(self, message: str, *, outcome: IsolationOutcome) -> None:
        super().__init__(message)
        self.outcome = outcome
        self.kind = outcome.kind
        self.stderr_excerpt = outcome.stderr_excerpt


def _is_aclose_race(exc: BaseException) -> bool:
    msg = str(exc)
    return any(marker in msg for marker in _ACLOSE_MARKERS)


def is_aclose_race(exc: BaseException) -> bool:
    """Public predicate — True when ``exc`` is the SDK's aclose teardown race.

    Other SDK call sites (e.g. the RLM-root client in ``rlm_query.py``) reuse
    this so the definition of "an aclose race" lives in exactly one place.
    """
    return _is_aclose_race(exc)


def _max_retries_from_env() -> int:
    raw = os.environ.get("REPROLAB_SDK_MAX_RETRIES", "").strip()
    if not raw:
        return _DEFAULT_MAX_RETRIES
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("sdk_isolation: invalid REPROLAB_SDK_MAX_RETRIES=%r", raw)
        return _DEFAULT_MAX_RETRIES


def _stderr_excerpt(chunks: list[str]) -> str:
    return "\n".join(chunk for chunk in chunks if chunk)[-_STDERR_EXCERPT_CHARS:]


@dataclass
class _AttemptResult(Generic[T]):
    result: T | None = None
    exception: BaseException | None = None
    aclose_post_result: bool = False
    stderr: str = ""


class RunIsolated:
    """Configured callable returned by ``make_run_isolated``."""

    def __init__(self, *, max_retries: int | None = None, name: str = "sdk-isolation") -> None:
        self.max_retries = None if max_retries is None else max(0, int(max_retries))
        self.name = name
        self.last_outcome: IsolationOutcome | None = None

    async def __call__(
        self,
        coro_factory: Callable[[], Coroutine[Any, Any, T]] | Coroutine[Any, Any, T],
        *,
        max_retries: int | None = None,
        name: str | None = None,
    ) -> T:
        """Run one SDK coroutine in a worker thread with aclose retry semantics.

        Pre: ``coro_factory`` should return a fresh coroutine on each call.  A
        bare coroutine is accepted for one release with ``DeprecationWarning``,
        but cannot be retried because Python coroutines are single-use.
        Post: returns the coroutine result on success and updates
        ``last_outcome`` with structured diagnostics.
        Side effects: starts a daemon worker thread per attempt and captures the
        worker's stderr excerpt for diagnostics.
        Exceptions raised: propagates real coroutine exceptions unchanged;
        raises ``IsolationFailure`` when pre-result aclose races exhaust retry
        budget.
        """
        attempts_allowed = (
            _max_retries_from_env()
            if max_retries is None and self.max_retries is None
            else self.max_retries
            if max_retries is None
            else max(0, int(max_retries))
        )
        thread_name = name or self.name
        factory, retryable = self._coerce_factory(coro_factory)
        if not retryable:
            attempts_allowed = 0

        if os.environ.get("REPROLAB_SDK_ISOLATION_DISABLED", "").lower() in {"true", "1", "yes"}:
            result = await factory()
            self.last_outcome = IsolationOutcome(kind="ok", attempt_count=1)
            return result

        stderr_chunks: list[str] = []
        saw_retry = False
        attempt = 0
        while True:
            attempt += 1
            attempt_result = await self._run_once(factory, name=thread_name)
            stderr_chunks.append(attempt_result.stderr)

            if attempt_result.exception is None:
                kind: IsolationOutcomeKind = (
                    "aclose_post_result_swallowed"
                    if attempt_result.aclose_post_result
                    else "aclose_pre_result_retried"
                    if saw_retry
                    else "ok"
                )
                self.last_outcome = IsolationOutcome(
                    kind=kind,
                    attempt_count=attempt,
                    stderr_excerpt=_stderr_excerpt(stderr_chunks),
                )
                return attempt_result.result  # type: ignore[return-value]

            exc = attempt_result.exception
            if _is_aclose_race(exc):
                if attempt <= attempts_allowed:
                    saw_retry = True
                    logger.warning(
                        "sdk_isolation: pre-result aclose race on attempt %d; retrying",
                        attempt,
                    )
                    continue
                outcome = IsolationOutcome(
                    kind="aclose_pre_result_exhausted",
                    attempt_count=attempt,
                    stderr_excerpt=_stderr_excerpt(stderr_chunks),
                )
                self.last_outcome = outcome
                raise IsolationFailure("aclose_pre_result_exhausted", outcome=outcome) from exc

            self.last_outcome = IsolationOutcome(
                kind="real_exception",
                attempt_count=attempt,
                stderr_excerpt=_stderr_excerpt(stderr_chunks),
            )
            raise exc

    def _coerce_factory(
        self,
        value: Callable[[], Coroutine[Any, Any, T]] | Coroutine[Any, Any, T],
    ) -> tuple[Callable[[], Coroutine[Any, Any, T]], bool]:
        if asyncio.iscoroutine(value):
            warnings.warn(
                "run_isolated(coro) is deprecated; pass a zero-argument "
                "factory such as run_isolated(lambda: coro_func()) so SDK "
                "aclose retries can create a fresh coroutine.",
                DeprecationWarning,
                stacklevel=3,
            )

            async def _single_use() -> T:
                return await value

            return _single_use, False
        return value, True

    async def _run_once(
        self,
        coro_factory: Callable[[], Coroutine[Any, Any, T]],
        *,
        name: str,
    ) -> _AttemptResult[T]:
        captured_result: list[T] = []
        captured_exception: list[BaseException] = []
        aclose_post_result: list[bool] = []
        stderr_buffer = io.StringIO()
        done = threading.Event()

        def worker() -> None:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                with contextlib.redirect_stderr(stderr_buffer):
                    try:
                        result = loop.run_until_complete(coro_factory())
                        captured_result.append(result)
                    except BaseException as exc:
                        if _is_aclose_race(exc) and captured_result:
                            aclose_post_result.append(True)
                            logger.debug("sdk_isolation: aclose race swallowed at cleanup")
                        else:
                            captured_exception.append(exc)
                    finally:
                        try:
                            self._close_loop(loop)
                        except RuntimeError as exc:
                            if _is_aclose_race(exc) and captured_result:
                                aclose_post_result.append(True)
                                logger.debug("sdk_isolation: aclose race swallowed at loop close")
                            else:
                                captured_exception.append(exc)
            finally:
                done.set()

        thread = threading.Thread(target=worker, name=name, daemon=True)
        thread.start()
        while not done.is_set():
            await asyncio.sleep(0.05)
        thread.join(timeout=1.0)

        return _AttemptResult(
            result=captured_result[0] if captured_result else None,
            exception=captured_exception[0] if captured_exception else None,
            aclose_post_result=bool(aclose_post_result),
            stderr=stderr_buffer.getvalue(),
        )

    def _close_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        loop.close()


def make_run_isolated(
    *,
    max_retries: int | None = None,
    name: str = "sdk-isolation",
) -> RunIsolated:
    """Create a configured SDK isolation callable.

    Pre: ``max_retries`` is ``None`` to read ``REPROLAB_SDK_MAX_RETRIES`` or a
    non-negative integer retry budget.
    Post: returns a reusable callable whose ``last_outcome`` reflects the most
    recent call.
    Side effects: none until the returned callable is invoked.
    Exceptions raised: ``ValueError`` is not raised; negative retry values are
    clamped to zero.
    """
    return RunIsolated(max_retries=max_retries, name=name)


_default_runner = make_run_isolated()


async def run_isolated(
    coro_factory: Callable[[], Coroutine[Any, Any, T]] | Coroutine[Any, Any, T],
    *,
    max_retries: int | None = None,
    name: str = "sdk-isolation",
) -> T:
    """Run an SDK call in isolation using the module-level default runner.

    Pre: prefer a zero-argument coroutine factory; bare coroutines are accepted
    temporarily with ``DeprecationWarning`` for backwards compatibility.
    Post: returns the coroutine result and stores diagnostics on
    ``run_isolated.last_outcome``.
    Side effects: starts worker threads unless isolation is disabled via
    ``REPROLAB_SDK_ISOLATION_DISABLED``.
    Exceptions raised: propagates real exceptions; raises ``IsolationFailure``
    when retryable pre-result aclose races are exhausted.
    """
    runner = _default_runner if max_retries is None and name == "sdk-isolation" else make_run_isolated(
        max_retries=max_retries,
        name=name,
    )
    result = await runner(coro_factory)
    run_isolated.last_outcome = runner.last_outcome  # type: ignore[attr-defined]
    return result


run_isolated.last_outcome = None  # type: ignore[attr-defined]


__all__ = [
    "IsolationFailure",
    "IsolationFailureKind",
    "IsolationOutcome",
    "RunIsolated",
    "is_aclose_race",
    "make_run_isolated",
    "run_isolated",
]
