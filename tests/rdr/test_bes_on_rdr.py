"""BES-on-RDR (spec 2026-06-07) — competing candidates + flag-off parity.

The MASTER gate ``bes_enabled`` (default False) makes every child flag inert, so
``run_rdr`` is byte-identical to today unless explicitly enabled. These tests pin:
  - ``select_best`` ranking semantics (candidates.py)
  - ``_dispatch_competing_candidates`` builds N isolated candidates + picks the winner
  - flag-off parity: dispatch count == one per cluster
  - master gate: ``bes_enabled=False`` overrides ``bes_candidates_per_cluster`` etc.
  - BES on: N candidates dispatched per Code-Dev cluster, scored statically
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from backend.agents.rdr.candidates import Candidate, select_best
from backend.agents.rdr.controller import _dispatch_competing_candidates, run_rdr
from backend.agents.rdr.decomposer import decompose
from backend.agents.rdr.models import AgentContext, Artifacts

from tests.rdr.test_controller import (
    FakeBundle,
    _FAKE_SCORES_HIGH,
    _make_cluster,
    _make_leaf,
    _patch_primitives,
    _patch_score,
    _rubric_tree_for,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BES_DEFAULTS = dict(
    bes_enabled=False,
    bes_candidates_per_cluster=1,
    bes_select_metric="cluster_score",
    bes_splice_enabled=False,
    rdr_preflight_gate=False,
    rdr_preflight_max_regens=1,
)


def _patch_bes_settings(monkeypatch: Any, **overrides: Any) -> SimpleNamespace:
    base = dict(_BES_DEFAULTS)
    base.update(overrides)
    stub = SimpleNamespace(**base)
    monkeypatch.setattr("backend.agents.rdr.controller.get_settings", lambda *a, **k: stub)
    return stub


def _code_dev_bundle(n: int) -> FakeBundle:
    leaves = [_make_leaf(f"leaf-{i}", 1.0, "Code Development") for i in range(n)]
    return FakeBundle(rubric_tree=_rubric_tree_for(leaves), leaves=leaves)


def _counting_reproduce():
    calls: dict[str, Any] = {"n": 0, "ids": []}

    async def _fn(agctx: Any, *, ctx: Any) -> Artifacts:
        calls["n"] += 1
        calls["ids"].append(agctx.cluster.id)
        cdir = getattr(agctx, "candidate_code_dir", None)
        if cdir is not None:
            cdir.mkdir(parents=True, exist_ok=True)
            (cdir / "train.py").write_text("# candidate", encoding="utf-8")
        return Artifacts(
            cluster_id=agctx.cluster.id,
            files={"train.py": "print('x')"},
            commands=["python train.py"],
            notes="", failed=False, error="",
        )

    return _fn, calls


# ---------------------------------------------------------------------------
# select_best (pure)
# ---------------------------------------------------------------------------


def _cand(cid: str, score: float | None, failed: bool = False, n_failed: int = 0) -> Candidate:
    return Candidate(
        candidate_id=cid, cluster_id="c", scratch_dir=Path("/tmp") / cid,
        artifacts=Artifacts(cluster_id="c", failed=failed),
        score=score, failed_leaves=["x"] * n_failed,
    )


def test_select_best_empty_pool_is_none():
    assert select_best([]) is None


def test_select_best_highest_score_wins():
    pool = [_cand("c#0", 0.3), _cand("c#1", 0.9), _cand("c#2", 0.5)]
    assert select_best(pool).candidate_id == "c#1"


def test_select_best_failed_ranks_below_scored():
    pool = [_cand("c#0", None, failed=True), _cand("c#1", 0.1)]
    assert select_best(pool).candidate_id == "c#1"


def test_select_best_tie_breaks_to_fewest_failed_then_earliest():
    # equal score, #1 has fewer failed leaves → wins
    pool = [_cand("c#0", 0.5, n_failed=3), _cand("c#1", 0.5, n_failed=1)]
    assert select_best(pool).candidate_id == "c#1"
    # full tie → earliest (max returns first element with the maximal key)
    pool2 = [_cand("c#0", 0.5, n_failed=1), _cand("c#1", 0.5, n_failed=1)]
    assert select_best(pool2).candidate_id == "c#0"


def test_select_best_failed_leaves_metric_prefers_fewest_failed():
    pool = [_cand("c#0", 0.9, n_failed=5), _cand("c#1", 0.4, n_failed=0)]
    # cluster_score metric → highest score
    assert select_best(pool, select_metric="cluster_score").candidate_id == "c#0"
    # failed_leaves metric → fewest failed
    assert select_best(pool, select_metric="failed_leaves").candidate_id == "c#1"


# ---------------------------------------------------------------------------
# _dispatch_competing_candidates (direct)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_competing_candidates_builds_n_and_picks_winner(tmp_path: Path):
    cluster = _make_cluster("c1", [_make_leaf("leaf-1", 1.0)])
    agctx = AgentContext(cluster=cluster, leaf_contract="", paper_sections=[])
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    fake_ctx = SimpleNamespace(project_id="t")
    emitted: list[tuple[str, dict]] = []

    n_repro = {"n": 0}

    async def reproduce(ac: Any, *, ctx: Any) -> Artifacts:
        n_repro["n"] += 1
        cdir = ac.candidate_code_dir
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "train.py").write_text("# x", encoding="utf-8")
        return Artifacts(cluster_id=ac.cluster.id, files={"train.py": "# x"},
                         commands=["python train.py"], notes="", failed=False, error="")

    def score_fn(cand_run_dir: Path, cl: Any) -> dict:
        # candidate #1 is the best; others poor.
        s = 0.9 if str(cand_run_dir).endswith("#1") else 0.1
        return {"leaf_scores": [{"id": "leaf-1", "score": s}]}

    art = await _dispatch_competing_candidates(
        cluster, agctx, n=3, reproduce=reproduce, ctx=fake_ctx, code_dir=code_dir,
        cluster_timeout_s=10.0, bes_score_fn=score_fn, select_metric="cluster_score",
        emit=lambda t, p: emitted.append((t, p)),
    )

    assert n_repro["n"] == 3                                   # N isolated attempts
    assert not art.failed
    for i in range(3):                                        # each candidate isolated
        assert (tmp_path / "candidates" / f"c1#{i}" / "code" / "train.py").exists()
    proposed = [p for t, p in emitted if t == "candidate_proposed"]
    assert len(proposed) == 3
    outcome = [p for t, p in emitted if t == "candidate_outcome"]
    assert outcome and outcome[0]["candidate_id"] == "c1#1"   # the high-scorer won


# ---------------------------------------------------------------------------
# Parity + master gate (through run_rdr)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bes_flags_off_dispatch_count_unchanged(tmp_path, make_context, monkeypatch):
    ctx = make_context(tmp_path)
    bundle = _code_dev_bundle(4)
    n_clusters = len(decompose(bundle.rubric()))
    _patch_primitives(monkeypatch)
    _patch_score(monkeypatch, _FAKE_SCORES_HIGH)
    _patch_bes_settings(monkeypatch, bes_enabled=False)
    reproduce, calls = _counting_reproduce()

    await run_rdr(bundle, ctx=ctx, reproduce_fn=reproduce, max_repair_iterations=0)

    assert calls["n"] == n_clusters                # exactly one dispatch per cluster
    assert not (ctx.project_dir / "candidates").exists()   # no candidate scratch dirs


@pytest.mark.asyncio
async def test_bes_master_gate_overrides_child_flags(tmp_path, make_context, monkeypatch):
    # bes_enabled=False MUST override bes_candidates_per_cluster=4 + splice=True.
    ctx = make_context(tmp_path)
    bundle = _code_dev_bundle(4)
    n_clusters = len(decompose(bundle.rubric()))
    _patch_primitives(monkeypatch)
    _patch_score(monkeypatch, _FAKE_SCORES_HIGH)
    _patch_bes_settings(monkeypatch, bes_enabled=False,
                        bes_candidates_per_cluster=4, bes_splice_enabled=True)
    reproduce, calls = _counting_reproduce()

    await run_rdr(bundle, ctx=ctx, reproduce_fn=reproduce, max_repair_iterations=0)

    assert calls["n"] == n_clusters                # master gate forces parity
    assert not (ctx.project_dir / "candidates").exists()


@pytest.mark.asyncio
async def test_bes_on_runs_n_candidates_per_codedev_cluster(tmp_path, make_context, monkeypatch):
    ctx = make_context(tmp_path)
    bundle = _code_dev_bundle(3)
    n_clusters = len(decompose(bundle.rubric()))   # all Code Development → all get BES
    _patch_primitives(monkeypatch)
    _patch_score(monkeypatch, _FAKE_SCORES_HIGH)
    _patch_bes_settings(monkeypatch, bes_enabled=True, bes_candidates_per_cluster=3)
    reproduce, calls = _counting_reproduce()

    await run_rdr(bundle, ctx=ctx, reproduce_fn=reproduce, max_repair_iterations=0)

    assert calls["n"] == 3 * n_clusters            # N candidates per cluster
    assert (ctx.project_dir / "candidates").exists()
