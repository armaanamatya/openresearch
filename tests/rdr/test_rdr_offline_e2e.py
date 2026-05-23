"""Offline end-to-end regression test for the RDR harness.

Exercises decompose + controller + context_engineer + report assembly on the
REAL ``sequential-neural-score-estimation`` PaperBench bundle. No LLM, no
Docker, no network. All I/O primitives are monkeypatched inside the
controller's namespace.

This is the integration-level backbone for Phase 5.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pytest

from backend.agents.rdr.controller import run_rdr
from backend.agents.rdr.models import Artifacts, RdrResult
from backend.evals.paperbench.bundle import load_paperbench_bundle

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUNDLE_ROOT = _REPO_ROOT / "third_party" / "paperbench" / "sequential-neural-score-estimation"

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def real_bundle():
    """Load the real PaperBench bundle from the vendored fixture."""
    return load_paperbench_bundle(_BUNDLE_ROOT)


def _fake_scores(overall: float = 0.80, leaf_count: int = 92, graded: int = 92) -> dict[str, Any]:
    """Build a plausible score dict.  leaf_scores intentionally empty — the
    controller's _cluster_score defaults absent leaves to 0.0, but for the
    'all high' scenario we pass overall_score directly and only need graded>0
    to flip status from 'partial' to 'completed'.
    """
    return {
        "overall_score": overall,
        "leaf_count": leaf_count,
        "graded": graded,
        "rubric_source": "paperbench_bundle",
        "leaf_scores": [],
    }


def _fake_reproduce_fn(
    files: dict[str, str] | None = None,
    commands: list[str] | None = None,
    failed: bool = False,
    error: str = "",
):
    """Return an async callable that yields deterministic Artifacts."""

    async def _fn(agent_context: Any, *, ctx: Any) -> Artifacts:
        return Artifacts(
            cluster_id=agent_context.cluster.id,
            files=files if files is not None else {"train.py": "# stub\npass"},
            commands=commands if commands is not None else ["python train.py"],
            notes="offline-e2e-stub",
            failed=failed,
            error=error,
        )

    return _fn


def _patch_primitives(monkeypatch: Any) -> None:
    """Monkeypatch detect_environment, build_environment, run_experiment in the
    controller's namespace — nothing real should run in offline tests."""
    monkeypatch.setattr(
        "backend.agents.rdr.controller.detect_environment",
        lambda spec, ctx: {"dockerfile": "FROM python:3.11", "python_version": "3.11"},
    )
    monkeypatch.setattr(
        "backend.agents.rdr.controller.build_environment",
        lambda spec, ctx: {"ok": True, "image_tag": "reprolab/test:stub", "error": "", "attempts": 1},
    )
    monkeypatch.setattr(
        "backend.agents.rdr.controller.run_experiment",
        lambda code_path, env_id, ctx: {"success": True, "metrics": {"accuracy": 0.99}, "logs": ""},
    )


# ---------------------------------------------------------------------------
# Test 1 — full contract on real bundle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_contract_real_bundle(
    tmp_path: Path, make_context: Any, monkeypatch: Any, real_bundle: Any
) -> None:
    """run_rdr returns a well-formed RdrResult and writes all required artifacts
    when given the real sequential-neural-score-estimation bundle."""
    ctx = make_context(tmp_path)
    _patch_primitives(monkeypatch)
    monkeypatch.setattr(
        "backend.agents.rdr.controller.score_reproduction",
        lambda rubric, run_dir, llm, **kwargs: _fake_scores(overall=0.80, graded=92),
    )

    result: RdrResult = await run_rdr(
        real_bundle,
        ctx=ctx,
        reproduce_fn=_fake_reproduce_fn(),
        max_repair_iterations=0,
        repair_target=0.6,
    )

    # --- RdrResult contract ---
    assert isinstance(result, RdrResult)
    assert result.project_id == ctx.project_id
    assert result.status in ("completed", "partial", "failed")
    assert result.rubric_score is not None
    assert result.rubric_score == pytest.approx(0.80)
    assert result.clusters_total == 27  # real bundle decomposes to 27 clusters
    assert result.clusters_failed == 0

    # --- final_report.json + final_report.md ---
    report_json = ctx.project_dir / "final_report.json"
    report_md = ctx.project_dir / "final_report.md"
    assert report_json.exists(), "final_report.json missing"
    assert report_md.exists(), "final_report.md missing"

    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert "verdict" in report
    assert "rubric" in report
    assert "reproduction_summary" in report
    assert report["rubric"]["overall_score"] == pytest.approx(0.80)

    # --- iterations/ — one file per cluster ---
    iter_dir = ctx.project_dir / "iterations"
    assert iter_dir.is_dir(), "iterations/ directory missing"
    iter_files = list(iter_dir.glob("cluster_*.json"))
    # Exactly one checkpoint per cluster from the initial pass
    assert len(iter_files) == 27, f"expected 27 checkpoint files, got {len(iter_files)}"

    for f in iter_files:
        data = json.loads(f.read_text(encoding="utf-8"))
        assert "cluster_id" in data
        assert "leaf_ids" in data
        assert "failed" in data
        assert "file_count" in data

    # --- repl_state.pickle ---
    pickle_path = ctx.project_dir / "repl_state.pickle"
    assert pickle_path.exists(), "repl_state.pickle missing"

    state = pickle.loads(pickle_path.read_bytes())
    assert isinstance(state, dict)
    assert "clusters_summary" in state
    assert "artifacts_summary" in state
    assert "scores" in state
    assert "repair_iterations" in state

    # --- corpus redaction: pickle must NOT contain the paper.md raw text ---
    paper_text = real_bundle.read_paper_markdown()
    # Use a known distinctive fragment from the paper text
    # (just check a long substring is absent from the raw bytes)
    fragment = paper_text[:200].encode("utf-8", errors="replace")
    raw_bytes = pickle_path.read_bytes()
    assert fragment not in raw_bytes, (
        "repl_state.pickle contains raw paper corpus text (corpus-redaction violation)"
    )

    # --- clusters_total matches the real decompose output ---
    assert result.clusters_total == 27


# ---------------------------------------------------------------------------
# Test 2 — repair iteration fires when first score is low
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repair_iteration_triggers_real_bundle(
    tmp_path: Path, make_context: Any, monkeypatch: Any, real_bundle: Any
) -> None:
    """repair_iterations >= 1 when the first score_reproduction returns low
    scores for all clusters (all leaf_scores absent → 0.0 per leaf)."""
    ctx = make_context(tmp_path)
    _patch_primitives(monkeypatch)

    call_count = [0]

    def _score_fn(rubric: Any, run_dir: Any, llm: Any) -> dict[str, Any]:
        call_count[0] += 1
        if call_count[0] == 1:
            # First call: all leaf_scores absent → _cluster_score → 0.0 for every cluster
            return _fake_scores(overall=0.05, graded=0)
        # Subsequent calls: high scores → no further repair
        return _fake_scores(overall=0.80, graded=92)

    monkeypatch.setattr("backend.agents.rdr.controller.score_reproduction", _score_fn)

    agent_calls: list[str] = []

    async def _counting_fn(agent_context: Any, *, ctx: Any) -> Artifacts:
        agent_calls.append(agent_context.cluster.id)
        return Artifacts(
            cluster_id=agent_context.cluster.id,
            files={"train.py": "# stub"},
            commands=["python train.py"],
            failed=False,
        )

    result: RdrResult = await run_rdr(
        real_bundle,
        ctx=ctx,
        reproduce_fn=_counting_fn,
        max_repair_iterations=2,
        repair_target=0.6,
    )

    assert result.repair_iterations >= 1, (
        f"expected at least 1 repair iteration, got {result.repair_iterations}"
    )
    # Initial pass: 27 clusters + at least one repair pass
    assert len(agent_calls) > 27


# ---------------------------------------------------------------------------
# Test 3 — failed reproduce_fn: run still completes, clusters_failed counted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_reproduce_fn_real_bundle(
    tmp_path: Path, make_context: Any, monkeypatch: Any, real_bundle: Any
) -> None:
    """When the reproduce_fn returns Artifacts(failed=True) for every cluster,
    the run still completes and clusters_failed == clusters_total."""
    ctx = make_context(tmp_path)
    _patch_primitives(monkeypatch)
    monkeypatch.setattr(
        "backend.agents.rdr.controller.score_reproduction",
        lambda rubric, run_dir, llm: _fake_scores(overall=0.0, graded=0),
    )

    result: RdrResult = await run_rdr(
        real_bundle,
        ctx=ctx,
        reproduce_fn=_fake_reproduce_fn(failed=True, error="simulated failure"),
        max_repair_iterations=0,
        repair_target=0.6,
    )

    assert isinstance(result, RdrResult)
    assert result.clusters_failed == 27  # all clusters failed
    assert result.clusters_total == 27
    # Run still writes final_report.json despite all clusters failing
    assert (ctx.project_dir / "final_report.json").exists()


# ---------------------------------------------------------------------------
# Test 4 — report's rubric key carries the score
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_report_rubric_carries_score_real_bundle(
    tmp_path: Path, make_context: Any, monkeypatch: Any, real_bundle: Any
) -> None:
    """final_report.json's rubric dict contains overall_score matching RdrResult."""
    ctx = make_context(tmp_path)
    _patch_primitives(monkeypatch)
    monkeypatch.setattr(
        "backend.agents.rdr.controller.score_reproduction",
        lambda rubric, run_dir, llm, **kwargs: _fake_scores(overall=0.72, graded=92),
    )

    result: RdrResult = await run_rdr(
        real_bundle,
        ctx=ctx,
        reproduce_fn=_fake_reproduce_fn(),
        max_repair_iterations=0,
    )

    assert result.rubric_score == pytest.approx(0.72)
    report = json.loads(
        (ctx.project_dir / "final_report.json").read_text(encoding="utf-8")
    )
    assert report["rubric"]["overall_score"] == pytest.approx(0.72)
