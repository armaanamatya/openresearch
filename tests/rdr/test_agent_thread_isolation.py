"""Tests for the SDK thread-isolation helper in ``backend/agents/rdr/agent.py``.

Covers Workaround B from
docs/superpowers/specs/2026-05-22-sdk-aclose-investigation.md.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

import backend.agents.rdr.agent as agent_mod
from backend.agents.rdr.agent import (
    _run_sdk_in_thread,
    reproduce,
)
from backend.agents.rdr.models import (
    AgentContext,
    RubricLeaf,
    WorkCluster,
)


# ---------------------------------------------------------------------------
# Minimal synthetic helpers (mirrors test_agent.py)
# ---------------------------------------------------------------------------


def _make_leaf(leaf_id: str = "L1") -> RubricLeaf:
    return RubricLeaf(
        id=leaf_id,
        requirements="Implement the training loop.",
        weight=1.0,
        task_category="Code Development",
        paper_citations=[],
    )


def _make_cluster(cluster_id: str = "C1") -> WorkCluster:
    leaves = [_make_leaf()]
    return WorkCluster(
        id=cluster_id,
        title="Training loop",
        leaves=leaves,
        dominant_category="Code Development",
        weight=sum(lf.weight for lf in leaves),
        depends_on=[],
        paper_citations=[],
    )


def _make_agent_context() -> AgentContext:
    cluster = _make_cluster()
    leaf_contract = (
        f'This cluster ("{cluster.title}") is graded on {len(cluster.leaves)} '
        f"requirement(s) (total weight {cluster.weight:.1f}).\n"
        "Implement so that each is satisfiable by an automated reproducibility judge.\n\n"
        + "\n".join(
            f"[{i}] (weight {lf.weight:.1f}) {lf.requirements}"
            for i, lf in enumerate(cluster.leaves, start=1)
        )
    )
    return AgentContext(
        cluster=cluster,
        leaf_contract=leaf_contract,
        paper_sections=[],
        dependency_artifacts={},
        prior_feedback=None,
        working_summary="",
    )


# ---------------------------------------------------------------------------
# Test 1 — worker runs in a separate event loop; caller loop is unaffected
# ---------------------------------------------------------------------------


def test_thread_isolation_runs_sdk_in_separate_loop(tmp_path, monkeypatch):
    """_run_sdk_in_thread() executes collect_agent_text in a worker thread's own
    event loop and returns the correct result without disturbing any caller loop."""

    async def fake_collect(
        agent_id: str,
        prompt: str,
        *,
        project_dir: Path,
        model: str | None = None,
        provider: Any = None,
        runtime: Any = None,
        max_turns: Any = None,
    ) -> str:
        await asyncio.sleep(0.05)
        return "hello from worker"

    import backend.agents.runtime.invoke as invoke_mod
    monkeypatch.setattr(invoke_mod, "collect_agent_text", fake_collect)

    # Called from a synchronous context — no caller event loop.
    result = _run_sdk_in_thread(
        prompt="test prompt",
        code_dir=tmp_path,
        model=None,
        provider=None,
        runtime=None,
        max_turns=None,
        timeout_s=5.0,
    )

    assert result == "hello from worker"

    # Confirm there is no running event loop in *this* (synchronous) thread.
    # asyncio.get_event_loop().is_running() should be False outside an async frame.
    try:
        loop = asyncio.get_event_loop()
        assert not loop.is_running(), (
            "The caller's event loop must not be running after _run_sdk_in_thread returns"
        )
    except RuntimeError:
        # No event loop at all in this thread — also fine.
        pass


# ---------------------------------------------------------------------------
# Test 2 — timeout raises TimeoutError; monkeypatched slack keeps it fast
# ---------------------------------------------------------------------------


def test_thread_isolation_timeout(tmp_path, monkeypatch):
    """When collect_agent_text sleeps longer than timeout_s, a TimeoutError is raised.

    _THREAD_TEARDOWN_SLACK_S is monkeypatched to 0.2 so the test completes in
    ~0.5 + 0.2 = 0.7 s instead of waiting for the default 30 s slack.
    """
    # Reduce slack so the test is fast.
    monkeypatch.setattr(agent_mod, "_THREAD_TEARDOWN_SLACK_S", 0.2)

    async def slow_collect(
        agent_id: str,
        prompt: str,
        *,
        project_dir: Path,
        **kwargs: Any,
    ) -> str:
        await asyncio.sleep(10.0)
        return "should not reach here"

    import backend.agents.runtime.invoke as invoke_mod
    monkeypatch.setattr(invoke_mod, "collect_agent_text", slow_collect)

    with pytest.raises(TimeoutError):
        _run_sdk_in_thread(
            prompt="test prompt",
            code_dir=tmp_path,
            model=None,
            provider=None,
            runtime=None,
            max_turns=None,
            timeout_s=0.5,
        )


# ---------------------------------------------------------------------------
# Test 3 — non-timeout exceptions propagate unchanged
# ---------------------------------------------------------------------------


def test_thread_isolation_propagates_exceptions(tmp_path, monkeypatch):
    """Exceptions raised inside collect_agent_text propagate out of _run_sdk_in_thread."""

    async def raising_collect(
        agent_id: str,
        prompt: str,
        *,
        project_dir: Path,
        **kwargs: Any,
    ) -> str:
        raise RuntimeError("synthetic")

    import backend.agents.runtime.invoke as invoke_mod
    monkeypatch.setattr(invoke_mod, "collect_agent_text", raising_collect)

    with pytest.raises(RuntimeError, match="synthetic"):
        _run_sdk_in_thread(
            prompt="test prompt",
            code_dir=tmp_path,
            model=None,
            provider=None,
            runtime=None,
            max_turns=None,
            timeout_s=5.0,
        )


# ---------------------------------------------------------------------------
# Test 4 — reproduce() delegates to _run_sdk_in_thread (integration-ish)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reproduce_uses_thread_isolation(tmp_path, make_context, monkeypatch):
    """reproduce() delegates the SDK call to _run_sdk_in_thread, not the old direct-await path.

    Monkeypatches _run_sdk_in_thread itself to return a known string, then verifies
    that Artifacts.notes equals that string.
    """
    run_ctx = make_context(tmp_path, project_id="thread_iso_test")
    code_dir = run_ctx.project_dir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)

    # Write a commands.json so _reproduce_inner doesn't fail on parsing.
    (code_dir / "commands.json").write_text(json.dumps(["python run.py"]), encoding="utf-8")

    known_output = "thread-isolated agent output"

    def fake_run_sdk_in_thread(
        prompt: str,
        code_dir_arg: Path,
        model: Any,
        provider: Any,
        runtime: Any,
        max_turns: Any,
        timeout_s: float,
    ) -> str:
        return known_output

    monkeypatch.setattr(agent_mod, "_run_sdk_in_thread", fake_run_sdk_in_thread)

    ac = _make_agent_context()
    result = await reproduce(ac, ctx=run_ctx)

    assert not result.failed, f"Expected success but got error: {result.error}"
    assert result.notes == known_output, (
        f"Expected notes={known_output!r}; got {result.notes!r}"
    )


# ---------------------------------------------------------------------------
# Test 5 — non-blocking shutdown: hung worker does not stall the controller
# ---------------------------------------------------------------------------


def test_thread_isolation_unblocks_on_hung_worker(tmp_path, monkeypatch):
    """_run_sdk_in_thread() returns TimeoutError within a bounded wall-clock time
    even when the worker is permanently hung (simulating a WSL2 futex stall).

    If the old `with ThreadPoolExecutor(...) as ex` form were used, __exit__
    would call shutdown(wait=True), blocking until the hung worker returns
    (which is never). The test would hang indefinitely (or until pytest's own
    timeout). With shutdown(wait=False) the controller continues immediately
    after the timeout fires.

    The abandoned-worker behavior is intentional and not asserted here.
    """
    import time

    # Reduce timeout and slack so the whole thing completes in ~0.6 s
    # instead of waiting for the default 30 s slack.
    timeout_s = 0.3
    monkeypatch.setattr(agent_mod, "_THREAD_TEARDOWN_SLACK_S", 0.3)

    async def hung_collect(
        agent_id: str,
        prompt: str,
        *,
        project_dir: Path,
        **kwargs: Any,
    ) -> str:
        # An Event that is never set — simulates a worker stuck awaiting
        # SDK subprocess teardown with no cancellation escape.
        await asyncio.Event().wait()
        return "unreachable"

    import backend.agents.runtime.invoke as invoke_mod
    monkeypatch.setattr(invoke_mod, "collect_agent_text", hung_collect)

    t0 = time.perf_counter()
    with pytest.raises(TimeoutError):
        _run_sdk_in_thread(
            prompt="test prompt",
            code_dir=tmp_path,
            model=None,
            provider=None,
            runtime=None,
            max_turns=None,
            timeout_s=timeout_s,
        )
    elapsed = time.perf_counter() - t0

    # The call must complete well within the default agent timeout (5400 s)
    # and also within a generous tolerance above the expected window.
    # Expected: timeout_s + slack + epsilon ≈ 0.6 s. Cap at 1.5 s to
    # give CI some headroom without allowing a regression to go unnoticed.
    assert elapsed < 1.5, (
        f"_run_sdk_in_thread took {elapsed:.2f}s — expected <1.5s. "
        "Likely regression: shutdown(wait=True) is blocking on the hung worker."
    )
    # Guard: must complete far below _DEFAULT_AGENT_TIMEOUT_S to catch any
    # revert to the blocking form.
    assert elapsed < agent_mod._DEFAULT_AGENT_TIMEOUT_S, (
        f"elapsed {elapsed:.2f}s >= _DEFAULT_AGENT_TIMEOUT_S — "
        "the timeout is not working at all."
    )
