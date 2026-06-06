"""Tests for the Phase-4 RDR Controller (``backend/agents/rdr/controller.py``).

All Docker/LLM operations are monkeypatched so the test suite is fully
deterministic and does not require network, API keys, or Docker.
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.agents.rdr.controller import _ClusterWatchdog, _write_cluster_checkpoint, _write_repair_checkpoint, run_rdr
from backend.agents.rdr.models import Artifacts, RdrResult, RubricLeaf, WorkCluster


# ---------------------------------------------------------------------------
# Synthetic bundle helpers
# ---------------------------------------------------------------------------


def _make_leaf(
    lid: str,
    weight: float,
    category: str = "Code Development",
    requirements: str = "Implement the method.",
) -> RubricLeaf:
    return RubricLeaf(
        id=lid,
        requirements=requirements,
        weight=weight,
        task_category=category,
        paper_citations=[],
    )


def _make_cluster(
    cid: str,
    leaves: list[RubricLeaf],
    category: str = "Code Development",
) -> WorkCluster:
    return WorkCluster(
        id=cid,
        title=f"Cluster {cid}",
        leaves=leaves,
        dominant_category=category,
        weight=sum(l.weight for l in leaves),
        depends_on=[],
        paper_citations=[],
    )


def _rubric_tree_for(leaves: list[RubricLeaf]) -> dict[str, Any]:
    """Minimal rubric dict for ``score_reproduction``."""
    return {
        "id": "root",
        "requirements": "Root",
        "weight": sum(l.weight for l in leaves),
        "sub_tasks": [
            {
                "id": l.id,
                "requirements": l.requirements,
                "weight": l.weight,
                "task_category": l.task_category,
                "sub_tasks": [],
            }
            for l in leaves
        ],
    }


class FakeBundle:
    """Minimal PaperBenchBundle-compatible fake."""

    def __init__(
        self,
        rubric_tree: dict[str, Any] | None = None,
        paper_md: str = "# Paper\n\n## 1 Introduction\n\nHello.",
        meta: dict[str, Any] | None = None,
        *,
        leaves: list[RubricLeaf] | None = None,
    ) -> None:
        _leaves = leaves or [_make_leaf("leaf-1", 0.5), _make_leaf("leaf-2", 0.5)]
        self._rubric = rubric_tree or _rubric_tree_for(_leaves)
        self._paper = paper_md
        self._meta = meta or {"id": "test-paper", "title": "Test Paper"}

    def rubric(self) -> dict[str, Any]:
        return self._rubric

    def read_paper_markdown(self) -> str:
        return self._paper

    def metadata(self) -> dict[str, Any]:
        return self._meta


# ---------------------------------------------------------------------------
# Default monkeypatches (shared by most tests)
# ---------------------------------------------------------------------------

_FAKE_ENV_SPEC = {"dockerfile": "FROM python:3.11", "python_version": "3.11"}
_FAKE_BUILD_OK = {"ok": True, "image_tag": "openresearch/test:env-abc123", "error": "", "attempts": 1}
_FAKE_EXP_OK = {"success": True, "metrics": {"accuracy": 0.95}, "logs": ""}
_FAKE_SCORES_HIGH = {
    "overall_score": 0.85,
    "leaf_count": 2,
    "graded": 2,
    "rubric_source": "paperbench_bundle",
    "leaf_scores": [
        {"id": "leaf-1", "score": 0.9, "justification": "good"},
        {"id": "leaf-2", "score": 0.8, "justification": "ok"},
    ],
}
_FAKE_SCORES_LOW = {
    "overall_score": 0.10,
    "leaf_count": 2,
    "graded": 2,
    "rubric_source": "paperbench_bundle",
    "leaf_scores": [
        {"id": "leaf-1", "score": 0.1, "justification": "missing code"},
        {"id": "leaf-2", "score": 0.1, "justification": "missing code"},
    ],
}


def _make_reproduce_fn(files: dict[str, str] | None = None, commands: list[str] | None = None):
    """Return an async callable that returns a fixed Artifacts."""

    async def _fn(agent_context: Any, *, ctx: Any) -> Artifacts:
        return Artifacts(
            cluster_id=agent_context.cluster.id,
            files=files or {"train.py": "print('hello')"},
            commands=commands or ["python train.py"],
            notes="synthesised",
            failed=False,
            error="",
        )

    return _fn


def _patch_primitives(monkeypatch: Any, *, env_spec=None, build=None, exp=None) -> None:
    """Monkeypatch detect_environment, build_environment, run_experiment in controller."""
    monkeypatch.setattr(
        "backend.agents.rdr.controller.detect_environment",
        lambda spec, *, ctx: env_spec or _FAKE_ENV_SPEC,
    )
    monkeypatch.setattr(
        "backend.agents.rdr.controller.build_environment",
        lambda spec, *, ctx: build or _FAKE_BUILD_OK,
    )
    monkeypatch.setattr(
        "backend.agents.rdr.controller.run_experiment",
        lambda code_path, env_id, *, ctx: exp or _FAKE_EXP_OK,
    )


def _patch_score(monkeypatch: Any, scores: Any) -> None:
    """Monkeypatch score_reproduction in controller (not leaf_scorer directly)."""
    if callable(scores):
        monkeypatch.setattr(
            "backend.agents.rdr.controller.score_reproduction",
            scores,
        )
    else:
        monkeypatch.setattr(
            "backend.agents.rdr.controller.score_reproduction",
            lambda rubric, run_dir, llm, **kwargs: scores,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_loop_writes_required_artifacts(
    tmp_path: Path, make_context: Any, monkeypatch: Any
) -> None:
    """run_rdr runs end-to-end and writes all #62 DC#4 artifacts."""
    ctx = make_context(tmp_path)
    bundle = FakeBundle()

    _patch_primitives(monkeypatch)
    _patch_score(monkeypatch, _FAKE_SCORES_HIGH)

    # Monkeypatch decompose to use our fake rubric's leaves via the bundle
    result: RdrResult = await run_rdr(
        bundle,
        ctx=ctx,
        reproduce_fn=_make_reproduce_fn(),
        max_repair_iterations=1,
        repair_target=0.6,
    )

    assert isinstance(result, RdrResult)
    assert result.project_id == ctx.project_id

    # final_report.json + final_report.md
    assert (ctx.project_dir / "final_report.json").exists()
    assert (ctx.project_dir / "final_report.md").exists()

    # iterations/ directory with at least one file
    assert (ctx.project_dir / "iterations").is_dir()
    iter_files = list((ctx.project_dir / "iterations").glob("*.json"))
    assert len(iter_files) > 0

    # repl_state.pickle
    assert (ctx.project_dir / "repl_state.pickle").exists()
    state = pickle.loads((ctx.project_dir / "repl_state.pickle").read_bytes())
    assert "clusters_summary" in state
    assert "artifacts_summary" in state
    assert "scores" in state
    assert "repair_iterations" in state

    ledger_rows = [
        json.loads(line)
        for line in (ctx.project_dir / "cost_ledger.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["primitive"] for row in ledger_rows] == [
        "detect_environment",
        "build_environment",
        "run_experiment",
    ]
    assert all(
        {"primitive", "cost_usd", "tokens_in", "tokens_out", "timestamp"} <= row.keys()
        for row in ledger_rows
    )


@pytest.mark.asyncio
async def test_final_report_json_is_valid(
    tmp_path: Path, make_context: Any, monkeypatch: Any
) -> None:
    """final_report.json is parseable JSON with expected keys."""
    ctx = make_context(tmp_path)
    bundle = FakeBundle()
    _patch_primitives(monkeypatch)
    _patch_score(monkeypatch, _FAKE_SCORES_HIGH)

    await run_rdr(
        bundle,
        ctx=ctx,
        reproduce_fn=_make_reproduce_fn(),
        max_repair_iterations=0,
    )

    report = json.loads((ctx.project_dir / "final_report.json").read_text(encoding="utf-8"))
    assert "verdict" in report
    assert "rubric" in report
    assert "reproduction_summary" in report


@pytest.mark.asyncio
async def test_repair_triggers_on_low_scores(
    tmp_path: Path, make_context: Any, monkeypatch: Any
) -> None:
    """Repair loop fires when cluster scores are below repair_target."""
    ctx = make_context(tmp_path)
    leaves = [_make_leaf("leaf-1", 0.5), _make_leaf("leaf-2", 0.5)]
    bundle = FakeBundle(leaves=leaves)

    _patch_primitives(monkeypatch)

    # First call → low; second call → high (triggers exactly one repair iteration)
    call_count = [0]

    def _score_fn(rubric: Any, run_dir: Any, llm: Any) -> dict:
        call_count[0] += 1
        if call_count[0] == 1:
            return _FAKE_SCORES_LOW
        return _FAKE_SCORES_HIGH

    _patch_score(monkeypatch, _score_fn)

    agent_calls: list[str] = []

    async def _counting_reproduce(agent_context: Any, *, ctx: Any) -> Artifacts:
        agent_calls.append(agent_context.cluster.id)
        return Artifacts(
            cluster_id=agent_context.cluster.id,
            files={"train.py": "print('hi')"},
            commands=["python train.py"],
            failed=False,
        )

    result: RdrResult = await run_rdr(
        bundle,
        ctx=ctx,
        reproduce_fn=_counting_reproduce,
        max_repair_iterations=2,
        repair_target=0.6,
    )

    assert result.repair_iterations >= 1
    # The agent was called at least twice (initial pass + repair)
    assert len(agent_calls) >= 2


@pytest.mark.asyncio
async def test_no_repair_when_scores_high(
    tmp_path: Path, make_context: Any, monkeypatch: Any
) -> None:
    """repair_iterations == 0 when all cluster scores meet target from the start."""
    ctx = make_context(tmp_path)
    bundle = FakeBundle()
    _patch_primitives(monkeypatch)
    _patch_score(monkeypatch, _FAKE_SCORES_HIGH)

    result: RdrResult = await run_rdr(
        bundle,
        ctx=ctx,
        reproduce_fn=_make_reproduce_fn(),
        max_repair_iterations=2,
        repair_target=0.6,
    )

    assert result.repair_iterations == 0


@pytest.mark.asyncio
async def test_per_cluster_failsoft(
    tmp_path: Path, make_context: Any, monkeypatch: Any
) -> None:
    """A reproduce_fn that fails one cluster does not abort the run."""
    ctx = make_context(tmp_path)
    leaves = [_make_leaf("leaf-a", 0.4), _make_leaf("leaf-b", 0.6)]
    bundle = FakeBundle(leaves=leaves)

    _patch_primitives(monkeypatch)
    _patch_score(monkeypatch, _FAKE_SCORES_HIGH)

    call_idx = [0]

    async def _failing_once(agent_context: Any, *, ctx: Any) -> Artifacts:
        call_idx[0] += 1
        # Fail the very first cluster invocation
        if call_idx[0] == 1:
            return Artifacts(
                cluster_id=agent_context.cluster.id,
                failed=True,
                error="simulated agent error",
            )
        return Artifacts(
            cluster_id=agent_context.cluster.id,
            files={"train.py": "print('hi')"},
            commands=["python train.py"],
            failed=False,
        )

    result: RdrResult = await run_rdr(
        bundle,
        ctx=ctx,
        reproduce_fn=_failing_once,
        max_repair_iterations=0,
        repair_target=0.6,
    )

    # Run still completes
    assert isinstance(result, RdrResult)
    assert result.clusters_failed >= 1
    assert (ctx.project_dir / "final_report.json").exists()


@pytest.mark.asyncio
async def test_per_cluster_exception_failsoft(
    tmp_path: Path, make_context: Any, monkeypatch: Any
) -> None:
    """An exception raised by reproduce_fn is caught per-cluster; run still completes."""
    ctx = make_context(tmp_path)
    bundle = FakeBundle()
    _patch_primitives(monkeypatch)
    _patch_score(monkeypatch, _FAKE_SCORES_HIGH)

    async def _raises(agent_context: Any, *, ctx: Any) -> Artifacts:
        raise RuntimeError("agent exploded")

    result: RdrResult = await run_rdr(
        bundle,
        ctx=ctx,
        reproduce_fn=_raises,
        max_repair_iterations=0,
    )

    assert isinstance(result, RdrResult)
    assert result.clusters_failed >= 1


@pytest.mark.asyncio
async def test_verdict_reconciled_against_score(
    tmp_path: Path, make_context: Any, monkeypatch: Any
) -> None:
    """The final report verdict is reconciled with the rubric score."""
    ctx = make_context(tmp_path)
    bundle = FakeBundle()
    _patch_primitives(monkeypatch)
    # Very low score → verdict should be "failed"
    _patch_score(monkeypatch, {**_FAKE_SCORES_LOW, "overall_score": 0.0})

    result = await run_rdr(
        bundle,
        ctx=ctx,
        reproduce_fn=_make_reproduce_fn(),
        max_repair_iterations=0,
    )

    report = json.loads((ctx.project_dir / "final_report.json").read_text(encoding="utf-8"))
    # reconcile_verdict_with_score("partial", 0.0) → "failed"
    assert report["verdict"] == "failed"

    # Also check an intermediate score reconciles to partial
    _patch_score(monkeypatch, {**_FAKE_SCORES_LOW, "overall_score": 0.20})

    ctx2 = make_context(tmp_path, project_id="rdr_test_v2")
    result2 = await run_rdr(
        FakeBundle(),
        ctx=ctx2,
        reproduce_fn=_make_reproduce_fn(),
        max_repair_iterations=0,
    )
    report2 = json.loads((ctx2.project_dir / "final_report.json").read_text(encoding="utf-8"))
    assert report2["verdict"] == "partial"


@pytest.mark.asyncio
async def test_env_detect_fail_soft(
    tmp_path: Path, make_context: Any, monkeypatch: Any
) -> None:
    """Env detect failure → run still completes (no experiment, honest partial)."""
    ctx = make_context(tmp_path)
    bundle = FakeBundle()

    # detect_environment returns failure
    monkeypatch.setattr(
        "backend.agents.rdr.controller.detect_environment",
        lambda spec, *, ctx: {"success": False, "error": "no env"},
    )
    monkeypatch.setattr(
        "backend.agents.rdr.controller.build_environment",
        lambda spec, *, ctx: {"ok": False, "image_tag": "", "error": "skipped", "attempts": 0},
    )
    monkeypatch.setattr(
        "backend.agents.rdr.controller.run_experiment",
        lambda code_path, env_id, *, ctx: {"success": False, "metrics": {}},
    )
    _patch_score(monkeypatch, _FAKE_SCORES_HIGH)

    result = await run_rdr(
        bundle,
        ctx=ctx,
        reproduce_fn=_make_reproduce_fn(),
        max_repair_iterations=0,
    )

    assert isinstance(result, RdrResult)
    assert (ctx.project_dir / "final_report.json").exists()


@pytest.mark.asyncio
async def test_iterations_dir_has_one_json_per_cluster(
    tmp_path: Path, make_context: Any, monkeypatch: Any
) -> None:
    """Each cluster produces exactly one checkpoint JSON in iterations/."""
    ctx = make_context(tmp_path)
    leaves = [
        _make_leaf("l1", 0.3),
        _make_leaf("l2", 0.3),
        _make_leaf("l3", 0.4),
    ]
    bundle = FakeBundle(leaves=leaves)
    _patch_primitives(monkeypatch)
    _patch_score(monkeypatch, _FAKE_SCORES_HIGH)

    await run_rdr(
        bundle,
        ctx=ctx,
        reproduce_fn=_make_reproduce_fn(),
        max_repair_iterations=0,
    )

    # decompose on a flat rubric groups all leaves under a single-level tree;
    # the exact cluster count depends on the rubric shape but there is at least one.
    iter_files = list((ctx.project_dir / "iterations").glob("*.json"))
    assert len(iter_files) >= 1

    # Each checkpoint has the required keys
    for f in iter_files:
        data = json.loads(f.read_text(encoding="utf-8"))
        assert "cluster_id" in data
        assert "leaf_ids" in data
        assert "failed" in data
        assert "file_count" in data


@pytest.mark.asyncio
async def test_repl_state_no_corpus_leak(
    tmp_path: Path, make_context: Any, monkeypatch: Any
) -> None:
    """repl_state.pickle must not contain the raw paper markdown text."""
    sensitive_text = "VERY SECRET PAPER CONTENT that should not be pickled"
    ctx = make_context(tmp_path)
    bundle = FakeBundle(paper_md=f"# Paper\n\n{sensitive_text}")

    _patch_primitives(monkeypatch)
    _patch_score(monkeypatch, _FAKE_SCORES_HIGH)

    await run_rdr(
        bundle,
        ctx=ctx,
        reproduce_fn=_make_reproduce_fn(),
        max_repair_iterations=0,
    )

    raw_bytes = (ctx.project_dir / "repl_state.pickle").read_bytes()
    assert sensitive_text.encode() not in raw_bytes


@pytest.mark.asyncio
async def test_result_fields_consistent(
    tmp_path: Path, make_context: Any, monkeypatch: Any
) -> None:
    """RdrResult fields are consistent with the written final_report.json."""
    ctx = make_context(tmp_path)
    bundle = FakeBundle()
    _patch_primitives(monkeypatch)
    _patch_score(monkeypatch, _FAKE_SCORES_HIGH)

    result = await run_rdr(
        bundle,
        ctx=ctx,
        reproduce_fn=_make_reproduce_fn(),
        max_repair_iterations=0,
    )

    assert result.rubric_score is not None
    assert result.rubric_score == pytest.approx(_FAKE_SCORES_HIGH["overall_score"])
    assert result.final_report_path is not None
    assert Path(result.final_report_path).exists()

    report = json.loads(Path(result.final_report_path).read_text(encoding="utf-8"))
    assert report["rubric"]["overall_score"] == pytest.approx(result.rubric_score)


@pytest.mark.asyncio
async def test_determinism_no_llm_in_control_flow(
    tmp_path: Path, make_context: Any, monkeypatch: Any
) -> None:
    """The controller is deterministic: same fake agent → same result on two runs."""
    _patch_primitives(monkeypatch)
    _patch_score(monkeypatch, _FAKE_SCORES_HIGH)

    results = []
    for i in range(2):
        ctx = make_context(tmp_path, project_id=f"det_test_{i}")
        bundle = FakeBundle()
        res = await run_rdr(
            bundle,
            ctx=ctx,
            reproduce_fn=_make_reproduce_fn(),
            max_repair_iterations=0,
        )
        results.append(res)

    assert results[0].rubric_score == results[1].rubric_score
    assert results[0].clusters_total == results[1].clusters_total
    assert results[0].repair_iterations == results[1].repair_iterations


@pytest.mark.asyncio
async def test_commands_json_written_and_deduped(
    tmp_path: Path, make_context: Any, monkeypatch: Any
) -> None:
    """commands.json in code/ is written with deduplicated union of all cluster commands."""
    ctx = make_context(tmp_path)
    leaves = [_make_leaf("l1", 0.5), _make_leaf("l2", 0.5)]
    bundle = FakeBundle(leaves=leaves)
    _patch_primitives(monkeypatch)
    _patch_score(monkeypatch, _FAKE_SCORES_HIGH)

    call_idx = [0]

    async def _fn_with_commands(agent_context: Any, *, ctx: Any) -> Artifacts:
        call_idx[0] += 1
        # Both clusters emit the same command + one unique
        return Artifacts(
            cluster_id=agent_context.cluster.id,
            files={},
            commands=["python train.py", f"python eval_{call_idx[0]}.py"],
            failed=False,
        )

    await run_rdr(
        bundle,
        ctx=ctx,
        reproduce_fn=_fn_with_commands,
        max_repair_iterations=0,
    )

    cmds_path = ctx.project_dir / "code" / "commands.json"
    assert cmds_path.exists()
    cmds = json.loads(cmds_path.read_text(encoding="utf-8"))
    # "python train.py" deduplicated to one entry
    assert cmds.count("python train.py") == 1


# ---------------------------------------------------------------------------
# FIX 1: repl_state.pickle must not contain `notes`
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repl_state_no_notes_key(
    tmp_path: Path, make_context: Any, monkeypatch: Any
) -> None:
    """artifacts_summary entries in repl_state.pickle must not have a 'notes' key.

    The notes field is raw agent output and may echo paper-corpus text —
    it must never reach the pickle (corpus-leak redaction invariant).
    """
    ctx = make_context(tmp_path)
    bundle = FakeBundle()
    _patch_primitives(monkeypatch)
    _patch_score(monkeypatch, _FAKE_SCORES_HIGH)

    await run_rdr(
        bundle,
        ctx=ctx,
        reproduce_fn=_make_reproduce_fn(),
        max_repair_iterations=0,
    )

    state = pickle.loads((ctx.project_dir / "repl_state.pickle").read_bytes())
    for cid, entry in state["artifacts_summary"].items():
        assert "notes" not in entry, (
            f"artifacts_summary[{cid!r}] contains 'notes' — corpus-leak risk"
        )
    # Must still have the expected keys
    for entry in state["artifacts_summary"].values():
        assert "file_count" in entry
        assert "failed" in entry
        assert "command_count" in entry


# ---------------------------------------------------------------------------
# FIX 3: agent-authored code/Dockerfile is promoted to project_dir/Dockerfile
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_dockerfile_promoted_to_project_dir(
    tmp_path: Path, make_context: Any, monkeypatch: Any
) -> None:
    """When the agent writes code/Dockerfile, it is copied to project_dir/Dockerfile
    and used directly by build_environment (detect_environment is skipped).
    """
    ctx = make_context(tmp_path)
    bundle = FakeBundle()

    dockerfile_content = "FROM python:3.11-slim\nRUN pip install numpy\n"

    async def _reproduce_with_dockerfile(agent_context: Any, *, ctx: Any) -> Artifacts:
        # Simulate agent writing a Dockerfile into code/
        code_dir = ctx.project_dir / "code"
        code_dir.mkdir(parents=True, exist_ok=True)
        (code_dir / "Dockerfile").write_text(dockerfile_content, encoding="utf-8")
        return Artifacts(
            cluster_id=agent_context.cluster.id,
            files={"train.py": "print('hello')", "Dockerfile": dockerfile_content},
            commands=["python train.py"],
            failed=False,
        )

    detect_called = [False]
    build_called_with: list[dict] = []

    def _fake_detect(spec: Any, *, ctx: Any) -> dict:
        detect_called[0] = True
        return _FAKE_ENV_SPEC

    def _fake_build(spec: Any, *, ctx: Any) -> dict:
        build_called_with.append(dict(spec))
        return _FAKE_BUILD_OK

    monkeypatch.setattr("backend.agents.rdr.controller.detect_environment", _fake_detect)
    monkeypatch.setattr("backend.agents.rdr.controller.build_environment", _fake_build)
    monkeypatch.setattr(
        "backend.agents.rdr.controller.run_experiment",
        lambda code_path, env_id, *, ctx: _FAKE_EXP_OK,
    )
    _patch_score(monkeypatch, _FAKE_SCORES_HIGH)

    await run_rdr(
        bundle,
        ctx=ctx,
        reproduce_fn=_reproduce_with_dockerfile,
        max_repair_iterations=0,
    )

    # detect_environment should NOT have been called (agent supplied Dockerfile)
    assert not detect_called[0], "detect_environment was called despite agent-supplied Dockerfile"

    # project_dir/Dockerfile must exist and contain the promoted content
    promoted = ctx.project_dir / "Dockerfile"
    assert promoted.exists(), "project_dir/Dockerfile was not created"
    assert promoted.read_text(encoding="utf-8") == dockerfile_content

    # build_environment was called with a spec containing the dockerfile content
    assert build_called_with, "build_environment was never called"
    assert "dockerfile" in build_called_with[0], "build_environment spec missing 'dockerfile'"
    assert build_called_with[0]["dockerfile"] == dockerfile_content


@pytest.mark.asyncio
async def test_no_agent_dockerfile_falls_back_to_detect(
    tmp_path: Path, make_context: Any, monkeypatch: Any
) -> None:
    """When the agent does NOT write code/Dockerfile, detect_environment is called normally."""
    ctx = make_context(tmp_path)
    bundle = FakeBundle()

    detect_called = [False]

    def _fake_detect(spec: Any, *, ctx: Any) -> dict:
        detect_called[0] = True
        return _FAKE_ENV_SPEC

    monkeypatch.setattr("backend.agents.rdr.controller.detect_environment", _fake_detect)
    monkeypatch.setattr(
        "backend.agents.rdr.controller.build_environment",
        lambda spec, *, ctx: _FAKE_BUILD_OK,
    )
    monkeypatch.setattr(
        "backend.agents.rdr.controller.run_experiment",
        lambda code_path, env_id, *, ctx: _FAKE_EXP_OK,
    )
    _patch_score(monkeypatch, _FAKE_SCORES_HIGH)

    await run_rdr(
        bundle,
        ctx=ctx,
        reproduce_fn=_make_reproduce_fn(),  # no Dockerfile
        max_repair_iterations=0,
    )

    assert detect_called[0], "detect_environment should have been called when no code/Dockerfile"


# ---------------------------------------------------------------------------
# FIX 5: repair-pass checkpoints and accurate dispatch count
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repair_checkpoints_written(
    tmp_path: Path, make_context: Any, monkeypatch: Any
) -> None:
    """Repair-pass checkpoints repair_<n>_cluster_<id>.json are written for each
    re-dispatched cluster."""
    ctx = make_context(tmp_path)
    leaves = [_make_leaf("leaf-1", 0.5), _make_leaf("leaf-2", 0.5)]
    bundle = FakeBundle(leaves=leaves)

    _patch_primitives(monkeypatch)

    call_count = [0]

    def _score_fn(rubric: Any, run_dir: Any, llm: Any) -> dict:
        call_count[0] += 1
        if call_count[0] == 1:
            return _FAKE_SCORES_LOW
        return _FAKE_SCORES_HIGH

    _patch_score(monkeypatch, _score_fn)

    result: RdrResult = await run_rdr(
        bundle,
        ctx=ctx,
        reproduce_fn=_make_reproduce_fn(),
        max_repair_iterations=2,
        repair_target=0.6,
    )

    # At least one repair pass fired
    assert result.repair_iterations >= 1

    iterations_dir = ctx.project_dir / "iterations"
    repair_files = list(iterations_dir.glob("repair_*.json"))
    assert len(repair_files) >= 1, "No repair checkpoint files written"

    for rf in repair_files:
        data = json.loads(rf.read_text(encoding="utf-8"))
        assert "cluster_id" in data
        assert "repair_pass" in data
        assert isinstance(data["repair_pass"], int)
        assert data["repair_pass"] >= 1


@pytest.mark.asyncio
async def test_iterations_count_reflects_actual_dispatches(
    tmp_path: Path, make_context: Any, monkeypatch: Any
) -> None:
    """The ``iterations`` field in final_report.json equals the actual number of
    agent dispatches (initial + only weak clusters in repair), not the inflated
    ``len(clusters) + repair_iterations * len(clusters)`` upper bound.
    """
    ctx = make_context(tmp_path)
    # 4 leaves → likely 1 cluster (flat rubric), so initial=1, repair(weak)=1 → total=2
    leaves = [_make_leaf(f"leaf-{i}", 0.25) for i in range(4)]
    bundle = FakeBundle(leaves=leaves)

    _patch_primitives(monkeypatch)

    call_count = [0]

    def _score_fn(rubric: Any, run_dir: Any, llm: Any) -> dict:
        call_count[0] += 1
        # Low on first call → triggers one repair pass
        if call_count[0] == 1:
            return {
                "overall_score": 0.10,
                "leaf_count": 4,
                "graded": 4,
                "leaf_scores": [
                    {"id": f"leaf-{i}", "score": 0.1, "justification": "weak"}
                    for i in range(4)
                ],
            }
        return _FAKE_SCORES_HIGH

    _patch_score(monkeypatch, _score_fn)

    dispatch_count = [0]

    async def _counting_fn(agent_context: Any, *, ctx: Any) -> Artifacts:
        dispatch_count[0] += 1
        return Artifacts(
            cluster_id=agent_context.cluster.id,
            files={"train.py": "print('hi')"},
            commands=["python train.py"],
            failed=False,
        )

    result: RdrResult = await run_rdr(
        bundle,
        ctx=ctx,
        reproduce_fn=_counting_fn,
        max_repair_iterations=2,
        repair_target=0.6,
    )

    report = json.loads((ctx.project_dir / "final_report.json").read_text(encoding="utf-8"))
    # The report iterations must match the actual dispatch count
    assert report["iterations"] == dispatch_count[0], (
        f"report['iterations']={report['iterations']} != actual dispatches={dispatch_count[0]}"
    )
    # And it should NOT be the inflated upper-bound formula
    clusters_total = result.clusters_total
    inflated = clusters_total + result.repair_iterations * clusters_total
    if result.repair_iterations > 0:
        # If some clusters were weak, dispatch count < inflated when only a subset re-ran
        assert report["iterations"] <= inflated, (
            "dispatch count exceeded inflated upper bound (impossible)"
        )


# ---------------------------------------------------------------------------
# _ClusterWatchdog unit tests
# ---------------------------------------------------------------------------


def test_watchdog_disarms_on_success() -> None:
    """Arming then disarming before the timeout elapses must NOT call os._exit."""
    with patch("os._exit") as mock_exit:
        wd = _ClusterWatchdog(timeout_s=5.0, label="test_success")
        wd.arm()
        time.sleep(0.02)
        wd.disarm()
        # Give the timer a tiny extra window to fire if it were still ticking
        time.sleep(0.05)
        mock_exit.assert_not_called()


def test_watchdog_fires_on_timeout() -> None:
    """When the timeout elapses before disarm, os._exit(124) must be called."""
    exit_calls: list[int] = []

    def fake_exit(code: int) -> None:
        exit_calls.append(code)

    with patch("backend.agents.rdr.controller.os._exit", side_effect=fake_exit):
        wd = _ClusterWatchdog(timeout_s=0.05, label="test_fire")
        wd.arm()
        # Wait long enough for the timer thread to fire
        time.sleep(0.25)

    assert exit_calls == [124], f"Expected os._exit(124), got: {exit_calls}"


def test_watchdog_disarm_is_idempotent() -> None:
    """Calling disarm() multiple times must not raise."""
    wd = _ClusterWatchdog(timeout_s=10.0, label="test_idempotent")
    wd.disarm()  # disarm before arm — no-op
    wd.arm()
    wd.disarm()
    wd.disarm()  # second disarm — must be silent


# ---------------------------------------------------------------------------
# Finding 1: scorer exception propagation (CRITICAL)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_rdr_survives_scorer_exception(
    tmp_path: Path, make_context: Any, monkeypatch: Any
) -> None:
    """run_rdr returns an RdrResult even when score_reproduction raises.

    The result must carry rubric_score=0.0 and the run must still write
    final_report.json.
    """
    ctx = make_context(tmp_path)
    bundle = FakeBundle()
    _patch_primitives(monkeypatch)

    def _raising_scorer(rubric: Any, run_dir: Any, llm: Any) -> dict:
        raise RuntimeError("simulated scorer OOM")

    _patch_score(monkeypatch, _raising_scorer)

    result: RdrResult = await run_rdr(
        bundle,
        ctx=ctx,
        reproduce_fn=_make_reproduce_fn(),
        max_repair_iterations=0,
    )

    assert isinstance(result, RdrResult), "run_rdr must return RdrResult even on scorer failure"
    assert result.rubric_score == pytest.approx(0.0), (
        f"rubric_score should be 0.0 on scorer exception; got {result.rubric_score}"
    )
    # final_report.json must still be written
    assert (ctx.project_dir / "final_report.json").exists(), (
        "final_report.json was not written after scorer exception"
    )


# ---------------------------------------------------------------------------
# Finding 2: path-traversal in merge-write (CRITICAL)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_controller_merge_write_rejects_path_traversal(
    tmp_path: Path, make_context: Any, monkeypatch: Any, caplog: Any
) -> None:
    """Artifacts with escape paths (../…) are silently rejected; safe paths are written.

    Verifies both the initial-pass merge and that nothing escapes code_dir.
    """
    ctx = make_context(tmp_path)
    bundle = FakeBundle()
    _patch_primitives(monkeypatch)
    _patch_score(monkeypatch, _FAKE_SCORES_HIGH)

    async def _reproduce_with_traversal(agent_context: Any, *, ctx: Any) -> Artifacts:
        return Artifacts(
            cluster_id=agent_context.cluster.id,
            files={
                "../escape.txt": "evil traversal content",
                "../../backend/evil.py": "# should never appear",
                "good.py": "print('ok')",
            },
            commands=["python good.py"],
            failed=False,
        )

    with caplog.at_level(logging.WARNING, logger="backend.agents.rdr.controller"):
        result: RdrResult = await run_rdr(
            bundle,
            ctx=ctx,
            reproduce_fn=_reproduce_with_traversal,
            max_repair_iterations=0,
        )

    # The run must complete successfully (escape paths are silently rejected)
    assert isinstance(result, RdrResult)

    # Safe file was written
    assert (ctx.project_dir / "code" / "good.py").exists(), (
        "good.py should have been written into code_dir"
    )

    # Escape paths were NOT written anywhere outside code_dir
    assert not (tmp_path / "escape.txt").exists(), (
        "escape.txt must not have been written outside code_dir"
    )
    assert not (ctx.project_dir / "escape.txt").exists(), (
        "escape.txt must not have been written in project_dir"
    )

    # At least one warning about refusing to write was logged
    refuse_warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "refusing to write" in r.message
    ]
    assert refuse_warnings, (
        "Expected at least one 'refusing to write' warning for the escape paths"
    )


# ---------------------------------------------------------------------------
# Finding 5: watchdog writes emergency final_report before os._exit (IMPORTANT)
# ---------------------------------------------------------------------------


def test_watchdog_fire_writes_emergency_report(tmp_path: Path) -> None:
    """_ClusterWatchdog._fire writes a minimal final_report.json before os._exit(124)."""
    exit_calls: list[int] = []

    def fake_exit(code: int) -> None:
        exit_calls.append(code)

    project_dir = tmp_path / "watchdog_test_run"
    project_dir.mkdir()

    with patch("backend.agents.rdr.controller.os._exit", side_effect=fake_exit):
        wd = _ClusterWatchdog(
            timeout_s=0.05,
            label="test_fire_report",
            project_dir=project_dir,
        )
        wd.arm()
        time.sleep(0.25)

    # os._exit must still have been called
    assert exit_calls == [124], f"Expected os._exit(124), got: {exit_calls}"

    # final_report.json must have been written
    report_path = project_dir / "final_report.json"
    assert report_path.exists(), "final_report.json was not written by watchdog._fire"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report.get("status") == "watchdog_killed", (
        f"Expected status='watchdog_killed'; got {report.get('status')!r}"
    )
    assert report.get("verdict") == "failed"
    assert report.get("label") == "test_fire_report"


# ---------------------------------------------------------------------------
# Feature A: Retry-on-watchdog — resume tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_hydrates_done_from_checkpoints(
    tmp_path: Path, make_context: Any, monkeypatch: Any
) -> None:
    """resume=True loads existing cluster checkpoints and skips those clusters.

    Pre-creates a checkpoint for cluster 0 in iterations/ and asserts that
    the reproduce_fn is NOT called for that cluster (it is loaded from disk),
    while the second cluster IS reproduced normally.
    """
    ctx = make_context(tmp_path)
    # Two leaves → two clusters (flat rubric; decompose groups them)
    leaves = [_make_leaf("leaf-a", 0.5), _make_leaf("leaf-b", 0.5)]
    bundle = FakeBundle(leaves=leaves)
    _patch_primitives(monkeypatch)
    _patch_score(monkeypatch, _FAKE_SCORES_HIGH)

    # Run once with resume=False to get the cluster decomposition and create
    # a checkpoint — then we read back the cluster_id from the checkpoint.
    called_cluster_ids: list[str] = []

    async def _tracking_fn(agent_context: Any, *, ctx: Any) -> Artifacts:
        called_cluster_ids.append(agent_context.cluster.id)
        return Artifacts(
            cluster_id=agent_context.cluster.id,
            files={"train.py": "print('hello')"},
            commands=["python train.py"],
            failed=False,
        )

    # ---- First pass: capture cluster decomposition ----
    ctx_first = make_context(tmp_path, project_id="rdr_resume_first")
    await run_rdr(
        bundle,
        ctx=ctx_first,
        reproduce_fn=_tracking_fn,
        max_repair_iterations=0,
    )
    iterations_dir_first = ctx_first.project_dir / "iterations"
    initial_checkpoints = sorted(iterations_dir_first.glob("cluster_*.json"))
    assert initial_checkpoints, "Expected at least one cluster checkpoint from initial run"

    # Determine the first cluster's id from the checkpoint
    first_checkpoint_data = json.loads(initial_checkpoints[0].read_text(encoding="utf-8"))
    skipped_cluster_id = first_checkpoint_data["cluster_id"]

    # ---- Second pass: resume with pre-existing checkpoint for cluster 0 ----
    ctx_resume = make_context(tmp_path, project_id="rdr_resume_second")
    iterations_dir_resume = ctx_resume.project_dir / "iterations"
    iterations_dir_resume.mkdir(parents=True, exist_ok=True)

    # Manually plant the checkpoint for cluster 0
    checkpoint_name = initial_checkpoints[0].name
    checkpoint_dest = iterations_dir_resume / checkpoint_name
    checkpoint_dest.write_text(
        initial_checkpoints[0].read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    resume_called_ids: list[str] = []

    async def _must_not_call_skipped(agent_context: Any, *, ctx: Any) -> Artifacts:
        cid = agent_context.cluster.id
        resume_called_ids.append(cid)
        assert cid != skipped_cluster_id, (
            f"reproduce_fn was called for cluster {cid!r} which has an existing checkpoint!"
        )
        return Artifacts(
            cluster_id=cid,
            files={"train.py": "print('resumed')"},
            commands=["python train.py"],
            failed=False,
        )

    result = await run_rdr(
        bundle,
        ctx=ctx_resume,
        reproduce_fn=_must_not_call_skipped,
        max_repair_iterations=0,
        resume=True,
    )

    assert isinstance(result, RdrResult)
    # The skipped cluster must appear in done (hydrated from checkpoint)
    # and NOT appear in resume_called_ids.
    assert skipped_cluster_id not in resume_called_ids, (
        f"cluster {skipped_cluster_id!r} was reproduced despite having a checkpoint"
    )


@pytest.mark.asyncio
async def test_resume_with_missing_iterations_dir(
    tmp_path: Path, make_context: Any, monkeypatch: Any
) -> None:
    """resume=True but no iterations/ dir exists → proceeds as a fresh run.

    Verifies that the absence of checkpoints does not cause errors and that
    all clusters are reproduced normally.
    """
    ctx = make_context(tmp_path)
    bundle = FakeBundle()
    _patch_primitives(monkeypatch)
    _patch_score(monkeypatch, _FAKE_SCORES_HIGH)

    # Ensure iterations/ does NOT exist before calling run_rdr
    iterations_dir = ctx.project_dir / "iterations"
    assert not iterations_dir.exists(), "iterations/ should not exist for this test"

    dispatched: list[str] = []

    async def _tracking_fn(agent_context: Any, *, ctx: Any) -> Artifacts:
        dispatched.append(agent_context.cluster.id)
        return Artifacts(
            cluster_id=agent_context.cluster.id,
            files={"train.py": "print('fresh')"},
            commands=["python train.py"],
            failed=False,
        )

    result = await run_rdr(
        bundle,
        ctx=ctx,
        reproduce_fn=_tracking_fn,
        max_repair_iterations=0,
        resume=True,  # no checkpoints → fresh run
    )

    assert isinstance(result, RdrResult)
    # All clusters were reproduced (no skips)
    assert len(dispatched) >= 1, "Expected at least one cluster to be dispatched"
    assert (ctx.project_dir / "final_report.json").exists()


# ---------------------------------------------------------------------------
# Feature B: SSE / dashboard_event emission
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rdr_emits_lifecycle_events(
    tmp_path: Path, make_context: Any, monkeypatch: Any
) -> None:
    """run_rdr emits the expected lifecycle dashboard_event types in order.

    Uses a fake DashboardEmitter that records (event_type, payload) pairs.
    Asserts:
    - required event types appear in the expected order
    - no payload contains the raw paper text fixture (corpus-leak check)
    """
    paper_text = "SECRET PAPER CONTENT abcdef123"
    ctx = make_context(tmp_path)
    bundle = FakeBundle(paper_md=f"# Paper\n\n{paper_text}")
    _patch_primitives(monkeypatch)
    _patch_score(monkeypatch, _FAKE_SCORES_HIGH)

    emitted: list[tuple[str, dict]] = []

    class FakeEmitter:
        def emit(self, event_type: str, payload: dict) -> None:
            emitted.append((event_type, dict(payload)))

    # Inject the fake emitter into ctx
    ctx.dashboard = FakeEmitter()

    await run_rdr(
        bundle,
        ctx=ctx,
        reproduce_fn=_make_reproduce_fn(),
        max_repair_iterations=0,
    )

    event_types = [et for et, _ in emitted]

    # Required events must be present
    required = [
        "rdr_run_started",
        "rdr_cluster_started",
        "cluster_started",
        "rdr_cluster_completed",
        "cluster_artifact_emitted",
        "rdr_environment_started",
        "rdr_environment_completed",
        "rdr_experiment_started",
        "rdr_experiment_completed",
        "rdr_scoring_started",
        "rdr_scoring_completed",
        "cluster_scored",
        "rdr_run_completed",
    ]
    for ev in required:
        assert ev in event_types, f"Expected event {ev!r} but not found in {event_types}"

    # Order: run_started must come before cluster events, which precede scoring,
    # which precedes run_completed.
    def _first_idx(ev: str) -> int:
        return next((i for i, (et, _) in enumerate(emitted) if et == ev), -1)

    assert _first_idx("rdr_run_started") < _first_idx("rdr_cluster_started"), (
        "rdr_run_started must precede rdr_cluster_started"
    )
    assert _first_idx("cluster_started") < _first_idx("rdr_cluster_completed"), (
        "cluster_started must be emitted before cluster completion"
    )
    assert _first_idx("rdr_scoring_completed") < _first_idx("rdr_run_completed"), (
        "rdr_scoring_completed must precede rdr_run_completed"
    )

    # Corpus-leak check: no payload must contain the raw paper text
    for ev_type, payload in emitted:
        payload_str = json.dumps(payload)
        assert paper_text not in payload_str, (
            f"Event {ev_type!r} payload contains raw paper text — corpus-leak!"
        )


@pytest.mark.asyncio
async def test_rdr_emits_repair_dispatched_spec_event(
    tmp_path: Path, make_context: Any, monkeypatch: Any
) -> None:
    ctx = make_context(tmp_path)
    bundle = FakeBundle()
    _patch_primitives(monkeypatch)
    _patch_score(monkeypatch, _FAKE_SCORES_LOW)

    emitted: list[tuple[str, dict]] = []

    class FakeEmitter:
        def emit(self, event_type: str, payload: dict) -> None:
            emitted.append((event_type, dict(payload)))

    ctx.dashboard = FakeEmitter()

    await run_rdr(
        bundle,
        ctx=ctx,
        reproduce_fn=_make_reproduce_fn(),
        max_repair_iterations=1,
        repair_target=0.6,
    )

    repairs = [payload for event_type, payload in emitted if event_type == "repair_dispatched"]
    assert repairs, "repair_dispatched spec event was not emitted"
    assert repairs[0]["attempt"] == 1
    failed_leaf_ids = {
        leaf_id
        for payload in repairs
        for leaf_id in payload["failed_leaves"]
    }
    assert failed_leaf_ids == {"leaf-1", "leaf-2"}


@pytest.mark.asyncio
async def test_rdr_metricless_run_scores_degraded_and_reports_metadata(
    tmp_path: Path, make_context: Any, monkeypatch: Any
) -> None:
    ctx = make_context(tmp_path)
    bundle = FakeBundle()
    _patch_primitives(monkeypatch, exp={"success": False, "metrics": {}, "logs": ""})

    def _score_fn(rubric: dict, run_dir: Path, llm: Any, **kwargs: Any) -> dict:
        assert kwargs["degraded"] is True
        return {
            "overall_score": 0.35,
            "leaf_count": 2,
            "graded": 0,
            "rubric_source": "paperbench_bundle",
            "degraded": True,
            "target_score": None,
            "leaf_scores": [
                {"id": "leaf-1", "score": 0.35, "justification": "degraded"},
                {"id": "leaf-2", "score": 0.35, "justification": "degraded"},
            ],
        }

    _patch_score(monkeypatch, _score_fn)

    result = await run_rdr(
        bundle,
        ctx=ctx,
        reproduce_fn=_make_reproduce_fn(),
        max_repair_iterations=0,
    )

    assert result.rubric_score <= 0.35
    report = json.loads((ctx.project_dir / "final_report.json").read_text(encoding="utf-8"))
    assert report["mode"] == "rdr"
    assert report["degraded"] is True
    assert report["rubric"]["degraded"] is True
    assert report["rubric"]["overall_score"] <= 0.35
    assert report["models"]["planner"] == ctx.model
    assert report["started_at"]
    assert report["completed_at"]


@pytest.mark.asyncio
async def test_rdr_handles_none_dashboard(
    tmp_path: Path, make_context: Any, monkeypatch: Any
) -> None:
    """run_rdr with ctx.dashboard=None completes without raising."""
    ctx = make_context(tmp_path)
    bundle = FakeBundle()
    _patch_primitives(monkeypatch)
    _patch_score(monkeypatch, _FAKE_SCORES_HIGH)

    # Explicitly set dashboard to None
    ctx.dashboard = None

    result = await run_rdr(
        bundle,
        ctx=ctx,
        reproduce_fn=_make_reproduce_fn(),
        max_repair_iterations=0,
    )

    assert isinstance(result, RdrResult)
    assert (ctx.project_dir / "final_report.json").exists()


# ---------------------------------------------------------------------------
# Fix: _write_cluster_checkpoint persists art.error
# ---------------------------------------------------------------------------


def test_write_cluster_checkpoint_persists_error(tmp_path: Path) -> None:
    """_write_cluster_checkpoint includes art.error in the JSON payload.

    Previously art.error was dropped — cluster failures lost their error string
    unless someone grepped logs. This guards the fix.
    """
    iterations_dir = tmp_path / "iterations"
    leaf = _make_leaf("leaf-x", 0.5)
    cluster = _make_cluster("cluster-err", [leaf])
    art = Artifacts(
        cluster_id="cluster-err",
        failed=True,
        error="AuthenticationError: 401",
        notes="",
        files={},
        commands=[],
    )

    _write_cluster_checkpoint(iterations_dir, 0, cluster, art)

    checkpoint_files = list(iterations_dir.glob("cluster_*.json"))
    assert len(checkpoint_files) == 1, "Expected exactly one checkpoint file"

    payload = json.loads(checkpoint_files[0].read_text(encoding="utf-8"))
    assert "error" in payload, "Checkpoint JSON must contain 'error' key"
    assert payload["error"] == "AuthenticationError: 401", (
        f"Expected error='AuthenticationError: 401'; got {payload['error']!r}"
    )
    # Existing keys must still be present
    assert payload["cluster_id"] == "cluster-err"
    assert payload["failed"] is True


# ---------------------------------------------------------------------------
# Fix: _write_repair_checkpoint persists art.error (follow-up to 94db854)
# ---------------------------------------------------------------------------


def test_write_repair_checkpoint_persists_error(tmp_path: Path) -> None:
    """_write_repair_checkpoint includes art.error in the JSON payload.

    Analogous to test_write_cluster_checkpoint_persists_error but for the
    repair-pass checkpoint function.  Previously art.error was absent from
    repair_<N>_cluster_<id>.json so repair failures were silently dropped.
    """
    iterations_dir = tmp_path / "iterations"
    leaf = _make_leaf("leaf-r", 0.5)
    cluster = _make_cluster("cluster-repair-err", [leaf])
    art = Artifacts(
        cluster_id="cluster-repair-err",
        failed=True,
        error="OOM: cluster died",
        notes="",
        files={},
        commands=[],
    )

    _write_repair_checkpoint(iterations_dir, 1, cluster, art)

    repair_files = list(iterations_dir.glob("repair_*.json"))
    assert len(repair_files) == 1, "Expected exactly one repair checkpoint file"

    payload = json.loads(repair_files[0].read_text(encoding="utf-8"))
    assert "error" in payload, "Repair checkpoint JSON must contain 'error' key"
    assert payload["error"] == "OOM: cluster died", (
        f"Expected error='OOM: cluster died'; got {payload['error']!r}"
    )
    assert payload["cluster_id"] == "cluster-repair-err"
    assert payload["failed"] is True
    assert payload["repair_pass"] == 1


# ---------------------------------------------------------------------------
# Fix: cluster dispatch events include art.error (follow-up to 94db854)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cluster_dispatch_event_includes_error(
    tmp_path: Path, make_context: Any, monkeypatch: Any
) -> None:
    """rdr_cluster_completed and rdr_repair_cluster_completed events include 'error'.

    Runs the controller with a reproduce_fn that returns a failed Artifacts
    (error="X") for every cluster, then checks that at least one captured
    rdr_cluster_completed event payload carries "error": "X".  Also exercises
    the repair path (low initial scores) so rdr_repair_cluster_completed is
    emitted, and asserts its payload includes "error" too.
    """
    ctx = make_context(tmp_path)
    leaves = [_make_leaf("leaf-1", 0.5), _make_leaf("leaf-2", 0.5)]
    bundle = FakeBundle(leaves=leaves)
    _patch_primitives(monkeypatch)

    # Initial score low so a repair pass fires and we also get repair events.
    call_count = [0]

    def _score_fn(rubric: Any, run_dir: Any, llm: Any) -> dict:
        call_count[0] += 1
        if call_count[0] == 1:
            return _FAKE_SCORES_LOW
        return _FAKE_SCORES_HIGH

    _patch_score(monkeypatch, _score_fn)

    async def _failing_reproduce(agent_context: Any, *, ctx: Any) -> Artifacts:
        return Artifacts(
            cluster_id=agent_context.cluster.id,
            failed=True,
            error="X",
        )

    emitted: list[tuple[str, dict]] = []

    class FakeEmitter:
        def emit(self, event_type: str, payload: dict) -> None:
            emitted.append((event_type, dict(payload)))

    ctx.dashboard = FakeEmitter()

    await run_rdr(
        bundle,
        ctx=ctx,
        reproduce_fn=_failing_reproduce,
        max_repair_iterations=1,
        repair_target=0.6,
    )

    # rdr_cluster_completed payloads must all have "error"
    completed_events = [p for et, p in emitted if et == "rdr_cluster_completed"]
    assert completed_events, "Expected at least one rdr_cluster_completed event"
    for payload in completed_events:
        assert "error" in payload, (
            f"rdr_cluster_completed payload missing 'error': {payload!r}"
        )
        assert payload["error"] == "X", (
            f"Expected error='X' in rdr_cluster_completed; got {payload['error']!r}"
        )

    # rdr_repair_cluster_completed payloads must also have "error"
    repair_events = [p for et, p in emitted if et == "rdr_repair_cluster_completed"]
    assert repair_events, "Expected at least one rdr_repair_cluster_completed event"
    for payload in repair_events:
        assert "error" in payload, (
            f"rdr_repair_cluster_completed payload missing 'error': {payload!r}"
        )
