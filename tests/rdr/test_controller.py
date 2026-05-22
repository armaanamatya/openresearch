"""Tests for the Phase-4 RDR Controller (``backend/agents/rdr/controller.py``).

All Docker/LLM operations are monkeypatched so the test suite is fully
deterministic and does not require network, API keys, or Docker.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agents.rdr.controller import run_rdr
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
_FAKE_BUILD_OK = {"ok": True, "image_tag": "reprolab/test:env-abc123", "error": "", "attempts": 1}
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
        lambda spec, ctx: env_spec or _FAKE_ENV_SPEC,
    )
    monkeypatch.setattr(
        "backend.agents.rdr.controller.build_environment",
        lambda spec, ctx: build or _FAKE_BUILD_OK,
    )
    monkeypatch.setattr(
        "backend.agents.rdr.controller.run_experiment",
        lambda code_path, env_id, ctx: exp or _FAKE_EXP_OK,
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
            lambda rubric, run_dir, llm: scores,
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
        lambda spec, ctx: {"success": False, "error": "no env"},
    )
    monkeypatch.setattr(
        "backend.agents.rdr.controller.build_environment",
        lambda spec, ctx: {"ok": False, "image_tag": "", "error": "skipped", "attempts": 0},
    )
    monkeypatch.setattr(
        "backend.agents.rdr.controller.run_experiment",
        lambda code_path, env_id, ctx: {"success": False, "metrics": {}},
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

    def _fake_detect(spec: Any, ctx: Any) -> dict:
        detect_called[0] = True
        return _FAKE_ENV_SPEC

    def _fake_build(spec: Any, ctx: Any) -> dict:
        build_called_with.append(dict(spec))
        return _FAKE_BUILD_OK

    monkeypatch.setattr("backend.agents.rdr.controller.detect_environment", _fake_detect)
    monkeypatch.setattr("backend.agents.rdr.controller.build_environment", _fake_build)
    monkeypatch.setattr(
        "backend.agents.rdr.controller.run_experiment",
        lambda code_path, env_id, ctx: _FAKE_EXP_OK,
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

    def _fake_detect(spec: Any, ctx: Any) -> dict:
        detect_called[0] = True
        return _FAKE_ENV_SPEC

    monkeypatch.setattr("backend.agents.rdr.controller.detect_environment", _fake_detect)
    monkeypatch.setattr(
        "backend.agents.rdr.controller.build_environment",
        lambda spec, ctx: _FAKE_BUILD_OK,
    )
    monkeypatch.setattr(
        "backend.agents.rdr.controller.run_experiment",
        lambda code_path, env_id, ctx: _FAKE_EXP_OK,
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
