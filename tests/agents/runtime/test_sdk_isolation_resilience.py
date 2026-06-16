"""Tests for PR-π Module A SDK isolation retry resilience."""

from __future__ import annotations

import sys

import pytest

from backend.agents.runtime.sdk_isolation import (
    IsolationFailure,
    make_run_isolated,
    run_isolated,
)


_ACLOSE = "aclose(): asynchronous generator is already running"


@pytest.mark.asyncio
async def test_aclose_post_result_swallowed_returns_result(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = make_run_isolated(max_retries=0)
    original_close = runner._close_loop

    def close_then_race(loop) -> None:  # type: ignore[no-untyped-def]
        original_close(loop)
        raise RuntimeError(_ACLOSE)

    monkeypatch.setattr(runner, "_close_loop", close_then_race)

    async def coro() -> str:
        return "captured"

    result = await runner(coro)

    assert result == "captured"
    assert runner.last_outcome is not None
    assert runner.last_outcome.kind == "aclose_post_result_swallowed"
    assert runner.last_outcome.attempt_count == 1


@pytest.mark.asyncio
async def test_aclose_pre_result_retries_then_succeeds() -> None:
    runner = make_run_isolated(max_retries=1)
    attempts = 0

    def factory():
        nonlocal attempts
        attempts += 1

        async def coro() -> str:
            if attempts == 1:
                raise RuntimeError(_ACLOSE)
            return "ok"

        return coro()

    result = await runner(factory)

    assert result == "ok"
    assert attempts == 2
    assert runner.last_outcome is not None
    assert runner.last_outcome.kind == "aclose_pre_result_retried"
    assert runner.last_outcome.attempt_count == 2


@pytest.mark.asyncio
async def test_aclose_pre_result_exceeds_max_retries_raises() -> None:
    runner = make_run_isolated(max_retries=1)

    async def coro() -> str:
        raise RuntimeError(_ACLOSE)

    with pytest.raises(IsolationFailure) as excinfo:
        await runner(coro)

    assert excinfo.value.kind == "aclose_pre_result_exhausted"
    assert excinfo.value.outcome.attempt_count == 2


@pytest.mark.asyncio
async def test_real_exception_propagates_immediately() -> None:
    runner = make_run_isolated(max_retries=3)
    attempts = 0

    def factory():
        nonlocal attempts
        attempts += 1

        async def coro() -> str:
            raise ValueError("real error")

        return coro()

    with pytest.raises(ValueError, match="real error"):
        await runner(factory)

    assert attempts == 1
    assert runner.last_outcome is not None
    assert runner.last_outcome.kind == "real_exception"


@pytest.mark.asyncio
async def test_factory_called_fresh_each_retry() -> None:
    runner = make_run_isolated(max_retries=2)
    calls = 0

    def factory():
        nonlocal calls
        calls += 1

        async def coro() -> int:
            if calls < 3:
                raise RuntimeError(_ACLOSE)
            return calls

        return coro()

    result = await runner(factory)

    assert result == 3
    assert calls == 3
    assert runner.last_outcome is not None
    assert runner.last_outcome.attempt_count == 3


@pytest.mark.asyncio
async def test_stderr_excerpt_captured_on_failure() -> None:
    runner = make_run_isolated(max_retries=0)

    async def coro() -> str:
        print("original stderr line", file=sys.stderr)
        raise RuntimeError(_ACLOSE)

    with pytest.raises(IsolationFailure) as excinfo:
        await runner(coro)

    assert "original stderr line" in excinfo.value.stderr_excerpt
    assert excinfo.value.outcome.stderr_excerpt == excinfo.value.stderr_excerpt


@pytest.mark.asyncio
async def test_bare_coroutine_shim_warns_and_returns_value() -> None:
    async def coro() -> int:
        return 42

    with pytest.warns(DeprecationWarning):
        result = await run_isolated(coro())

    assert result == 42


@pytest.mark.asyncio
async def test_isolation_disabled_awaits_factory_directly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENRESEARCH_SDK_ISOLATION_DISABLED", "true")
    runner = make_run_isolated(max_retries=3)

    async def coro() -> str:
        return "direct"

    result = await runner(coro)

    assert result == "direct"
    assert runner.last_outcome is not None
    assert runner.last_outcome.kind == "ok"
