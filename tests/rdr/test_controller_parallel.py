"""Tests for parallel cluster execution in the RDR controller.

Cover the new ``cluster_concurrency`` arg added to ``run_rdr``: that Code
Development clusters dispatch concurrently, that Code Execution and Result
Analysis stay sequential, that per-cluster timeouts cancel only the offending
cluster, that shared state (done dict, file merge, cost ledger, dashboard
event log) survives parallel bursts, and that ``cluster_concurrency=1``
preserves the original sequential behaviour exactly.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any

import pytest

from backend.agents.rdr.controller import run_rdr, _split_clusters_by_parallelism
from backend.agents.rdr.models import Artifacts, WorkCluster

from tests.rdr.test_controller import (
    FakeBundle,
    _make_leaf,
    _patch_primitives,
    _patch_score,
    _rubric_tree_for,
    _FAKE_SCORES_HIGH,
)


# ---------------------------------------------------------------------------
# Helpers — controlled-timing reproduce_fn that records dispatch concurrency
# ---------------------------------------------------------------------------


def _make_timed_reproduce_fn(
    sleep_s: float = 0.2,
    *,
    files_per_cluster: dict[str, str] | None = None,
):
    """Return (async reproduce_fn, observer dict).

    The observer dict carries:
      - ``concurrent_peak``: the maximum number of clusters in-flight at any moment.
      - ``call_order``: list of cluster ids in call order.
      - ``finish_order``: list of cluster ids in finish order.
    """
    observer: dict[str, Any] = {
        "concurrent_peak": 0,
        "in_flight": 0,
        "call_order": [],
        "finish_order": [],
        "_lock": threading.Lock(),
    }

    async def _fn(agent_context: Any, *, ctx: Any) -> Artifacts:
        cid = agent_context.cluster.id
        with observer["_lock"]:
            observer["call_order"].append(cid)
            observer["in_flight"] += 1
            observer["concurrent_peak"] = max(
                observer["concurrent_peak"], observer["in_flight"]
            )
        try:
            await asyncio.sleep(sleep_s)
        finally:
            with observer["_lock"]:
                observer["in_flight"] -= 1
                observer["finish_order"].append(cid)
        return Artifacts(
            cluster_id=cid,
            files=(files_per_cluster or {}).get(cid, {f"{cid}.py": f"# {cid}"}),
            commands=[],
            notes="timed",
            failed=False,
            error="",
        )

    return _fn, observer


def _make_code_dev_bundle(n: int) -> FakeBundle:
    """Create a synthetic bundle of *n* Code Development leaves (one per cluster)."""
    leaves = [
        _make_leaf(f"leaf-{i}", weight=1.0, category="Code Development")
        for i in range(n)
    ]
    return FakeBundle(rubric_tree=_rubric_tree_for(leaves), leaves=leaves)


# ---------------------------------------------------------------------------
# Behavioural tests
# ---------------------------------------------------------------------------


def test_split_clusters_by_parallelism_categorises_correctly() -> None:
    # _make_cluster sets dominant_category="Code Development" by default,
    # so construct WorkCluster directly to exercise the non-default branches.
    code_dev = WorkCluster(
        id="cd", title="cd", leaves=[_make_leaf("a", 1.0, "Code Development")],
        dominant_category="Code Development", weight=1.0,
    )
    code_exec = WorkCluster(
        id="ce", title="ce", leaves=[_make_leaf("b", 1.0, "Code Execution")],
        dominant_category="Code Execution", weight=1.0,
    )
    result = WorkCluster(
        id="ra", title="ra", leaves=[_make_leaf("c", 1.0, "Result Analysis")],
        dominant_category="Result Analysis", weight=1.0,
    )
    parallel, sequential = _split_clusters_by_parallelism([code_dev, code_exec, result])
    assert parallel == [code_dev]
    assert sequential == [code_exec, result]


@pytest.mark.asyncio
async def test_code_dev_clusters_dispatch_concurrently(
    tmp_path: Path, make_context: Any, monkeypatch: Any
) -> None:
    """With cluster_concurrency=4 and 8 Code Dev clusters, peak in-flight ≥ 2.

    We assert ≥ 2 rather than == 4 to avoid flakiness from semaphore acquisition
    ordering — the contract under test is "actually parallel," not "exactly N."
    """
    bundle = _make_code_dev_bundle(8)
    reproduce_fn, observer = _make_timed_reproduce_fn(sleep_s=0.15)
    _patch_primitives(monkeypatch)
    _patch_score(monkeypatch, _FAKE_SCORES_HIGH)
    ctx = make_context(tmp_path, project_id="pb-parallel-codedev")
    result = await run_rdr(
        bundle,
        ctx=ctx,
        reproduce_fn=reproduce_fn,
        max_repair_iterations=0,
        cluster_concurrency=4,
    )
    assert result.clusters_total == 8
    assert observer["concurrent_peak"] >= 2
    # All 8 should have run (success path, no failures).
    assert len(observer["finish_order"]) == 8


@pytest.mark.asyncio
async def test_cluster_concurrency_1_runs_sequentially(
    tmp_path: Path, make_context: Any, monkeypatch: Any
) -> None:
    """cluster_concurrency=1 reproduces original sequential behaviour."""
    bundle = _make_code_dev_bundle(5)
    reproduce_fn, observer = _make_timed_reproduce_fn(sleep_s=0.05)
    _patch_primitives(monkeypatch)
    _patch_score(monkeypatch, _FAKE_SCORES_HIGH)
    ctx = make_context(tmp_path, project_id="pb-seq")
    await run_rdr(
        bundle,
        ctx=ctx,
        reproduce_fn=reproduce_fn,
        max_repair_iterations=0,
        cluster_concurrency=1,
    )
    assert observer["concurrent_peak"] == 1


@pytest.mark.asyncio
async def test_code_execution_clusters_stay_sequential_even_under_high_concurrency(
    tmp_path: Path, make_context: Any, monkeypatch: Any
) -> None:
    """Code Execution + Result Analysis are forced sequential regardless of concurrency."""
    # 3 Code Execution + 2 Result Analysis — all should run one-at-a-time.
    leaves = [
        _make_leaf("ce-1", 1.0, "Code Execution"),
        _make_leaf("ce-2", 1.0, "Code Execution"),
        _make_leaf("ce-3", 1.0, "Code Execution"),
        _make_leaf("ra-1", 1.0, "Result Analysis"),
        _make_leaf("ra-2", 1.0, "Result Analysis"),
    ]
    bundle = FakeBundle(rubric_tree=_rubric_tree_for(leaves), leaves=leaves)
    reproduce_fn, observer = _make_timed_reproduce_fn(sleep_s=0.05)
    _patch_primitives(monkeypatch)
    _patch_score(monkeypatch, _FAKE_SCORES_HIGH)
    ctx = make_context(tmp_path, project_id="pb-seq-tail")
    await run_rdr(
        bundle,
        ctx=ctx,
        reproduce_fn=reproduce_fn,
        max_repair_iterations=0,
        cluster_concurrency=16,
    )
    # All five clusters are non-Code-Dev → sequential tail → peak == 1.
    assert observer["concurrent_peak"] == 1


@pytest.mark.asyncio
async def test_parallel_dispatch_writes_one_checkpoint_per_cluster(
    tmp_path: Path, make_context: Any, monkeypatch: Any
) -> None:
    """All N parallel-dispatched clusters must produce a checkpoint file."""
    bundle = _make_code_dev_bundle(6)
    reproduce_fn, _ = _make_timed_reproduce_fn(sleep_s=0.0)
    _patch_primitives(monkeypatch)
    _patch_score(monkeypatch, _FAKE_SCORES_HIGH)
    ctx = make_context(tmp_path, project_id="pb-checkpoints")
    await run_rdr(
        bundle,
        ctx=ctx,
        reproduce_fn=reproduce_fn,
        max_repair_iterations=0,
        cluster_concurrency=4,
    )
    iterations_dir = ctx.project_dir / "iterations"
    checkpoints = sorted(iterations_dir.glob("cluster_*.json"))
    assert len(checkpoints) == 6
    # Each must parse + have a cluster_id.
    for p in checkpoints:
        data = json.loads(p.read_text(encoding="utf-8"))
        assert "cluster_id" in data and data["cluster_id"]


@pytest.mark.asyncio
async def test_per_cluster_timeout_isolates_failure(
    tmp_path: Path, make_context: Any, monkeypatch: Any
) -> None:
    """One slow cluster times out + fails; the rest still complete cleanly."""
    bundle = _make_code_dev_bundle(4)

    async def _flaky_reproduce(agent_context: Any, *, ctx: Any) -> Artifacts:
        cid = agent_context.cluster.id
        if cid == "leaf-2":
            # Far exceeds the test-tweaked timeout below.
            await asyncio.sleep(5.0)
        else:
            await asyncio.sleep(0.05)
        return Artifacts(
            cluster_id=cid,
            files={f"{cid}.py": f"# {cid}"},
            commands=[],
            notes="ok",
            failed=False,
            error="",
        )

    # Squeeze the per-cluster timeout to 0.5s — leaf-2 will exceed, others won't.
    monkeypatch.setattr(
        "backend.agents.rdr.controller._RDR_WATCHDOG_DEFAULT_S", 0.5
    )
    _patch_primitives(monkeypatch)
    _patch_score(monkeypatch, _FAKE_SCORES_HIGH)
    ctx = make_context(tmp_path, project_id="pb-timeout-iso")
    result = await run_rdr(
        bundle,
        ctx=ctx,
        reproduce_fn=_flaky_reproduce,
        max_repair_iterations=0,
        cluster_concurrency=4,
    )
    assert result.clusters_total == 4
    # leaf-2 should be the only failure — the timeout did NOT kill the process
    # or cancel its peers.
    assert result.clusters_failed == 1
    # The other three should have produced checkpoints with failed=False.
    iterations_dir = ctx.project_dir / "iterations"
    checkpoints = {
        json.loads(p.read_text(encoding="utf-8"))["cluster_id"]:
        json.loads(p.read_text(encoding="utf-8"))["failed"]
        for p in iterations_dir.glob("cluster_*.json")
    }
    assert checkpoints["leaf-2"] is True
    for cid, failed in checkpoints.items():
        if cid != "leaf-2":
            assert failed is False, f"peer cluster {cid} should have succeeded"


@pytest.mark.asyncio
async def test_same_path_writes_from_two_clusters_dont_corrupt(
    tmp_path: Path, make_context: Any, monkeypatch: Any
) -> None:
    """Two clusters writing the same path concurrently — file ends in a consistent state."""
    # 6 clusters all writing the same shared file with their cluster id as body.
    bundle = _make_code_dev_bundle(6)

    async def _shared_writer(agent_context: Any, *, ctx: Any) -> Artifacts:
        cid = agent_context.cluster.id
        await asyncio.sleep(0.01)
        return Artifacts(
            cluster_id=cid,
            # Same path from every cluster — last-completed wins by design.
            files={"shared.py": f"# written by {cid}\n"},
            commands=[],
            notes="",
            failed=False,
            error="",
        )

    _patch_primitives(monkeypatch)
    _patch_score(monkeypatch, _FAKE_SCORES_HIGH)
    ctx = make_context(tmp_path, project_id="pb-shared-write")
    await run_rdr(
        bundle,
        ctx=ctx,
        reproduce_fn=_shared_writer,
        max_repair_iterations=0,
        cluster_concurrency=6,
    )
    shared = (ctx.project_dir / "code" / "shared.py").read_text(encoding="utf-8")
    # Must be exactly one cluster's content (atomic merge under file_merge_lock),
    # never an interleaved mix of two clusters.
    assert shared.startswith("# written by leaf-") and shared.endswith("\n")
    assert shared.count("# written by") == 1


@pytest.mark.asyncio
async def test_dashboard_events_are_well_formed_under_parallel_burst(
    tmp_path: Path, make_context: Any, monkeypatch: Any
) -> None:
    """Every line in dashboard_events.jsonl must be valid JSON (no interleaved bytes)."""
    bundle = _make_code_dev_bundle(10)
    reproduce_fn, _ = _make_timed_reproduce_fn(sleep_s=0.02)
    _patch_primitives(monkeypatch)
    _patch_score(monkeypatch, _FAKE_SCORES_HIGH)
    ctx = make_context(tmp_path, project_id="pb-events")
    await run_rdr(
        bundle,
        ctx=ctx,
        reproduce_fn=reproduce_fn,
        max_repair_iterations=0,
        cluster_concurrency=10,
    )
    events_path = ctx.project_dir / "dashboard_events.jsonl"
    assert events_path.exists()
    lines = events_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) > 0
    # Every single line must be valid JSON — concurrent appends without the
    # DashboardEmitter lock would produce torn lines that fail to parse.
    for i, line in enumerate(lines):
        try:
            json.loads(line)
        except json.JSONDecodeError as exc:
            pytest.fail(f"torn jsonl on line {i}: {exc} — line={line!r}")


@pytest.mark.asyncio
async def test_done_dict_has_every_cluster_after_parallel_dispatch(
    tmp_path: Path, make_context: Any, monkeypatch: Any
) -> None:
    """Run with high concurrency and verify final_report reflects every cluster."""
    bundle = _make_code_dev_bundle(12)
    reproduce_fn, _ = _make_timed_reproduce_fn(sleep_s=0.01)
    _patch_primitives(monkeypatch)
    _patch_score(monkeypatch, _FAKE_SCORES_HIGH)
    ctx = make_context(tmp_path, project_id="pb-done-dict")
    result = await run_rdr(
        bundle,
        ctx=ctx,
        reproduce_fn=reproduce_fn,
        max_repair_iterations=0,
        cluster_concurrency=8,
    )
    final_report = json.loads(
        (ctx.project_dir / "final_report.json").read_text(encoding="utf-8")
    )
    # Summary mentions 12 clusters, 0 failed.
    assert "12 cluster" in final_report["reproduction_summary"]
    assert "0 failed" in final_report["reproduction_summary"]
    assert result.clusters_total == 12
    assert result.clusters_failed == 0
