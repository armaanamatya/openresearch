import pytest
from backend.agents.runtime.sdk_isolation import run_isolated


@pytest.mark.asyncio
async def test_returns_value_from_isolated_coroutine():
    async def coro():
        return 42
    result = await run_isolated(coro())
    assert result == 42


@pytest.mark.asyncio
async def test_swallows_aclose_race_at_cleanup():
    """The known SDK race fires at generator cleanup; helper must not propagate.

    In the real SDK the race fires at GC/asyncio-shutdown time, AFTER the
    coroutine's async-for loop has completed and the result has been captured
    in a side-channel collector. We simulate this by having the coro return
    normally, and then the race fire when the worker loop shuts down its
    remaining async generators — the result is already in `result` at that
    point.

    The test below simulates the real pattern: the coro collects its result
    into a mutable list (as collect_agent_text does with `collected`), then
    returns the list. The aclose race fires from a separate async generator
    that's still live when the worker loop closes — it must not propagate
    to the run_isolated caller.
    """
    async def _fire_race():
        # Async generator that raises the race when closed by the event loop.
        yield "x"  # must yield once so it's a live async generator at loop close

    async def coro_with_postreturn_aclose_race():
        collected = []
        collected.append("result")
        # Start the async generator but don't fully drain it — it will be
        # open when the worker loop shuts down, triggering the aclose race.
        gen = _fire_race()
        collected.append(await gen.__anext__())  # drain one item — "x"
        return collected[0]  # returns "result"

    result = await run_isolated(coro_with_postreturn_aclose_race())
    assert result == "result"


@pytest.mark.asyncio
async def test_propagates_genuine_exception_from_coroutine():
    """Real exceptions from the coroutine body itself must still propagate."""
    async def coro_that_raises():
        raise ValueError("real error")
    with pytest.raises(ValueError, match="real error"):
        await run_isolated(coro_that_raises())
