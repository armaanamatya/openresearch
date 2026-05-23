"""Tests for the Phase-3 Reproduction Agent (``backend/agents/rdr/agent.py``).

All tests mock ``collect_agent_text`` so that no real LLM or network calls
are made — the SDK infra is verified through call-argument inspection.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from backend.agents.rdr.agent import reproduce, _snapshot_repo_root_entries, _cleanup_repo_root_escape
from backend.agents.rdr.models import (
    AgentContext,
    Artifacts,
    CitedSection,
    RubricLeaf,
    WorkCluster,
)


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------


def _make_leaf(
    leaf_id: str = "L1",
    requirements: str = "Implement the training loop.",
    weight: float = 1.0,
    task_category: str = "Code Development",
    paper_citations: list[str] | None = None,
) -> RubricLeaf:
    return RubricLeaf(
        id=leaf_id,
        requirements=requirements,
        weight=weight,
        task_category=task_category,
        paper_citations=paper_citations or [],
    )


def _make_cluster(
    cluster_id: str = "C1",
    title: str = "Training loop",
    leaves: list[RubricLeaf] | None = None,
) -> WorkCluster:
    leaves = leaves or [_make_leaf()]
    return WorkCluster(
        id=cluster_id,
        title=title,
        leaves=leaves,
        dominant_category="Code Development",
        weight=sum(lf.weight for lf in leaves),
        depends_on=[],
        paper_citations=[],
    )


def _make_agent_context(
    cluster: WorkCluster | None = None,
    paper_sections: list[CitedSection] | None = None,
    dependency_artifacts: dict[str, str] | None = None,
    prior_feedback: str | None = None,
    working_summary: str = "",
) -> AgentContext:
    cluster = cluster or _make_cluster()
    leaf_contract = (
        f'This cluster ("{cluster.title}") is graded on {len(cluster.leaves)} '
        f'requirement(s) (total weight {cluster.weight:.1f}).\n'
        "Implement so that each is satisfiable by an automated reproducibility judge.\n\n"
        + "\n".join(
            f"[{i}] (weight {lf.weight:.1f}) {lf.requirements}"
            for i, lf in enumerate(cluster.leaves, start=1)
        )
    )
    return AgentContext(
        cluster=cluster,
        leaf_contract=leaf_contract,
        paper_sections=paper_sections or [],
        dependency_artifacts=dependency_artifacts or {},
        prior_feedback=prior_feedback,
        working_summary=working_summary,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ctx(tmp_path, make_context):
    return make_context(tmp_path)


# ---------------------------------------------------------------------------
# Happy-path test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path(tmp_path, make_context, monkeypatch):
    """reproduce() returns well-formed Artifacts from files the mock agent writes."""
    run_ctx = make_context(tmp_path, project_id="happy_test")
    code_dir = run_ctx.project_dir / "code"

    # The "agent" writes files and commands.json when it runs.
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
        # Write some artefacts to simulate code generation.
        (project_dir / "train.py").write_text("# training code\n", encoding="utf-8")
        (project_dir / "config.json").write_text('{"lr": 0.001}\n', encoding="utf-8")
        commands = ["python train.py", "python eval.py"]
        (project_dir / "commands.json").write_text(
            json.dumps(commands), encoding="utf-8"
        )
        return "Agent finished writing the training loop."

    monkeypatch.setattr(
        "backend.agents.rdr.agent.collect_agent_text",
        fake_collect,
        raising=False,
    )
    # Patch via the module's import path as used inside _reproduce_inner.
    import backend.agents.runtime.invoke as invoke_mod
    monkeypatch.setattr(invoke_mod, "collect_agent_text", fake_collect)

    ac = _make_agent_context()
    result = await reproduce(ac, ctx=run_ctx)

    assert not result.failed, f"Expected success but got error: {result.error}"
    assert result.cluster_id == ac.cluster.id

    # files should mirror what the mock wrote in code_dir
    assert "train.py" in result.files
    assert "config.json" in result.files
    assert "commands.json" in result.files
    assert "# training code" in result.files["train.py"]

    # commands parsed from commands.json
    assert result.commands == ["python train.py", "python eval.py"]

    # notes is the agent's text output
    assert "Agent finished" in result.notes


# ---------------------------------------------------------------------------
# Fail-soft test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fail_soft_on_sdk_error(tmp_path, make_context, monkeypatch):
    """reproduce() catches any exception and returns Artifacts(failed=True)."""
    run_ctx = make_context(tmp_path, project_id="fail_test")

    async def raising_collect(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError("simulated SDK failure")

    import backend.agents.runtime.invoke as invoke_mod
    monkeypatch.setattr(invoke_mod, "collect_agent_text", raising_collect)

    ac = _make_agent_context()
    result = await reproduce(ac, ctx=run_ctx)

    assert result.failed is True
    assert "RuntimeError" in result.error or "simulated SDK failure" in result.error
    assert result.files == {}
    assert result.commands == []
    assert result.cluster_id == ac.cluster.id


# ---------------------------------------------------------------------------
# Prompt content tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_contains_leaf_contract(tmp_path, make_context, monkeypatch):
    """The rendered prompt sent to the SDK contains the leaf_contract text."""
    run_ctx = make_context(tmp_path, project_id="prompt_leaf_test")
    captured: list[str] = []

    async def capturing_collect(
        agent_id: str,
        prompt: str,
        *,
        project_dir: Path,
        **kwargs: Any,
    ) -> str:
        captured.append(prompt)
        (project_dir / "commands.json").write_text("[]", encoding="utf-8")
        return ""

    import backend.agents.runtime.invoke as invoke_mod
    monkeypatch.setattr(invoke_mod, "collect_agent_text", capturing_collect)

    leaf = _make_leaf(requirements="Train the model using SGD with momentum.")
    cluster = _make_cluster(leaves=[leaf])
    ac = _make_agent_context(cluster=cluster)

    await reproduce(ac, ctx=run_ctx)

    assert captured, "collect_agent_text was never called"
    prompt = captured[0]
    # The leaf_contract text must appear in the prompt.
    assert ac.leaf_contract in prompt
    assert "Train the model using SGD with momentum." in prompt


@pytest.mark.asyncio
async def test_prompt_contains_paper_section(tmp_path, make_context, monkeypatch):
    """The rendered prompt includes at least one cited paper section excerpt."""
    run_ctx = make_context(tmp_path, project_id="prompt_section_test")
    captured: list[str] = []

    async def capturing_collect(
        agent_id: str,
        prompt: str,
        *,
        project_dir: Path,
        **kwargs: Any,
    ) -> str:
        captured.append(prompt)
        (project_dir / "commands.json").write_text("[]", encoding="utf-8")
        return ""

    import backend.agents.runtime.invoke as invoke_mod
    monkeypatch.setattr(invoke_mod, "collect_agent_text", capturing_collect)

    section = CitedSection(
        citation="Section 3",
        heading="3 Methodology",
        text="We use stochastic gradient descent with a cosine schedule.",
    )
    ac = _make_agent_context(paper_sections=[section])

    await reproduce(ac, ctx=run_ctx)

    assert captured
    prompt = captured[0]
    # Both the heading and the section text must appear.
    assert "3 Methodology" in prompt
    assert "stochastic gradient descent" in prompt


@pytest.mark.asyncio
async def test_prompt_contains_prior_feedback_when_set(
    tmp_path, make_context, monkeypatch
):
    """When prior_feedback is provided, the prompt includes it (repair pass)."""
    run_ctx = make_context(tmp_path, project_id="prompt_feedback_test")
    captured: list[str] = []

    async def capturing_collect(
        agent_id: str,
        prompt: str,
        *,
        project_dir: Path,
        **kwargs: Any,
    ) -> str:
        captured.append(prompt)
        (project_dir / "commands.json").write_text("[]", encoding="utf-8")
        return ""

    import backend.agents.runtime.invoke as invoke_mod
    monkeypatch.setattr(invoke_mod, "collect_agent_text", capturing_collect)

    feedback = "=== REPAIR FEEDBACK ===\nLeaf L1 scored 0.2 — fix the learning-rate schedule."
    ac = _make_agent_context(prior_feedback=feedback)

    await reproduce(ac, ctx=run_ctx)

    assert captured
    prompt = captured[0]
    assert feedback in prompt
    assert "repair" in prompt.lower()


@pytest.mark.asyncio
async def test_prompt_omits_feedback_section_when_none(
    tmp_path, make_context, monkeypatch
):
    """When prior_feedback is None (first pass), no 'Repair Instructions' section appears."""
    run_ctx = make_context(tmp_path, project_id="prompt_no_feedback_test")
    captured: list[str] = []

    async def capturing_collect(
        agent_id: str,
        prompt: str,
        *,
        project_dir: Path,
        **kwargs: Any,
    ) -> str:
        captured.append(prompt)
        (project_dir / "commands.json").write_text("[]", encoding="utf-8")
        return ""

    import backend.agents.runtime.invoke as invoke_mod
    monkeypatch.setattr(invoke_mod, "collect_agent_text", capturing_collect)

    ac = _make_agent_context(prior_feedback=None)

    await reproduce(ac, ctx=run_ctx)

    assert captured
    prompt = captured[0]
    assert "REPAIR FEEDBACK" not in prompt
    assert "Repair Instructions" not in prompt


# ---------------------------------------------------------------------------
# Dynamic provider / model test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_provider_and_model_flow_from_ctx(tmp_path, make_context, monkeypatch):
    """provider and model are taken from ctx, not hard-coded."""
    run_ctx = make_context(tmp_path, project_id="provider_test")
    # Patch ctx with explicit provider / agent_model values.
    run_ctx.provider = "openai"
    run_ctx.model = "gpt-4o"
    run_ctx.agent_model = "gpt-4o-mini"

    call_kwargs: dict[str, Any] = {}

    async def capturing_collect(
        agent_id: str,
        prompt: str,
        *,
        project_dir: Path,
        model: str | None = None,
        provider: Any = None,
        runtime: Any = None,
        max_turns: Any = None,
    ) -> str:
        call_kwargs["model"] = model
        call_kwargs["provider"] = provider
        (project_dir / "commands.json").write_text("[]", encoding="utf-8")
        return ""

    import backend.agents.runtime.invoke as invoke_mod
    monkeypatch.setattr(invoke_mod, "collect_agent_text", capturing_collect)

    ac = _make_agent_context()
    await reproduce(ac, ctx=run_ctx)

    # agent_model should take precedence over model.
    assert call_kwargs["model"] == "gpt-4o-mini"
    # provider must come from ctx, not be hard-coded.
    assert call_kwargs["provider"] == "openai"


@pytest.mark.asyncio
async def test_model_falls_back_to_ctx_model_when_agent_model_unset(
    tmp_path, make_context, monkeypatch
):
    """When agent_model is None, ctx.model is used."""
    run_ctx = make_context(tmp_path, project_id="model_fallback_test")
    run_ctx.provider = "anthropic"
    run_ctx.model = "claude-sonnet-4-6"
    run_ctx.agent_model = None

    call_kwargs: dict[str, Any] = {}

    async def capturing_collect(
        agent_id: str,
        prompt: str,
        *,
        project_dir: Path,
        model: str | None = None,
        provider: Any = None,
        **kwargs: Any,
    ) -> str:
        call_kwargs["model"] = model
        call_kwargs["provider"] = provider
        (project_dir / "commands.json").write_text("[]", encoding="utf-8")
        return ""

    import backend.agents.runtime.invoke as invoke_mod
    monkeypatch.setattr(invoke_mod, "collect_agent_text", capturing_collect)

    ac = _make_agent_context()
    await reproduce(ac, ctx=run_ctx)

    assert call_kwargs["model"] == "claude-sonnet-4-6"
    assert call_kwargs["provider"] == "anthropic"


def test_snapshot_excludes_ephemeral_and_cache_dirs(tmp_path):
    """``_snapshot_code_dir`` skips cache/build/ephemeral dirs so the repair
    loop never tries to rewrite e.g. Docker-created HuggingFace lock files."""
    from backend.agents.rdr.agent import _snapshot_code_dir

    code_dir = tmp_path / "code"
    code_dir.mkdir()
    # Source files — must be included.
    (code_dir / "train.py").write_text("print('hi')", encoding="utf-8")
    (code_dir / "configs").mkdir()
    (code_dir / "configs" / "model.yaml").write_text("lr: 0.001", encoding="utf-8")
    # Ephemeral / cache / hidden — must be excluded.
    (code_dir / "__pycache__").mkdir()
    (code_dir / "__pycache__" / "train.cpython-312.pyc").write_text("x", encoding="utf-8")
    (code_dir / "hf_cache" / ".locks" / "models--gpt2").mkdir(parents=True)
    (code_dir / "hf_cache" / ".locks" / "models--gpt2" / "abc.lock").write_text("", encoding="utf-8")
    (code_dir / ".git").mkdir()
    (code_dir / ".git" / "HEAD").write_text("ref", encoding="utf-8")
    (code_dir / "outputs").mkdir()
    (code_dir / "outputs" / "results.txt").write_text("acc=0.5", encoding="utf-8")
    (code_dir / "sbi-logs").mkdir()
    (code_dir / "sbi-logs" / "run.log").write_text("...", encoding="utf-8")

    snap = _snapshot_code_dir(code_dir)

    assert "train.py" in snap
    assert "configs/model.yaml" in snap
    assert not any("__pycache__" in p for p in snap), snap
    assert not any("hf_cache" in p for p in snap), snap
    assert not any(p.startswith(".git") or "/.git" in p for p in snap), snap
    assert not any("outputs" in p for p in snap), snap
    assert not any("sbi-logs" in p for p in snap), snap
    assert not any(p.endswith(".lock") for p in snap), snap
    assert not any(p.endswith(".pyc") for p in snap), snap


# ---------------------------------------------------------------------------
# CWD guard tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cwd_guard_restores_cwd_on_success(tmp_path, make_context, monkeypatch):
    """After a successful reproduce(), the working directory is restored to what it was before."""
    run_ctx = make_context(tmp_path, project_id="cwd_success_test")
    cwd_before = Path.cwd()

    async def fake_collect(
        agent_id: str,
        prompt: str,
        *,
        project_dir: Path,
        **kwargs: Any,
    ) -> str:
        (project_dir / "commands.json").write_text("[]", encoding="utf-8")
        return ""

    import backend.agents.runtime.invoke as invoke_mod
    monkeypatch.setattr(invoke_mod, "collect_agent_text", fake_collect)

    ac = _make_agent_context()
    await reproduce(ac, ctx=run_ctx)

    assert Path.cwd() == cwd_before


@pytest.mark.asyncio
async def test_cwd_guard_restores_cwd_on_failure(tmp_path, make_context, monkeypatch):
    """Even when collect_agent_text raises, the working directory is restored."""
    run_ctx = make_context(tmp_path, project_id="cwd_fail_test")
    cwd_before = Path.cwd()

    async def raising_collect(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError("simulated SDK failure")

    import backend.agents.runtime.invoke as invoke_mod
    monkeypatch.setattr(invoke_mod, "collect_agent_text", raising_collect)

    ac = _make_agent_context()
    result = await reproduce(ac, ctx=run_ctx)

    # The outer fail-soft wrapper returns Artifacts(failed=True) — CWD must still be restored.
    assert result.failed is True
    assert Path.cwd() == cwd_before


@pytest.mark.asyncio
async def test_cwd_guard_cleans_repo_root_escape(
    tmp_path: Path, make_context: Any, monkeypatch: Any, caplog: Any
) -> None:
    """When the agent creates a stray directory at the repo root, it is removed and a
    WARNING is logged.  Uses a monkeypatched _REPO_ROOT pointing at tmp_path so the
    real repo root is never touched."""
    import backend.agents.rdr.agent as agent_mod

    # Redirect _REPO_ROOT to a safe temp directory.
    fake_root = tmp_path / "fake_repo_root"
    fake_root.mkdir()
    (fake_root / "backend").mkdir()  # satisfies the validation check

    monkeypatch.setattr(agent_mod, "_REPO_ROOT", fake_root)

    run_ctx = make_context(tmp_path, project_id="cwd_escape_test")

    async def fake_collect_with_escape(
        agent_id: str,
        prompt: str,
        *,
        project_dir: Path,
        **kwargs: Any,
    ) -> str:
        # Simulate agent escaping out of code_dir and creating a dir at the (fake) repo root.
        stray = fake_root / "rdr_escape_test"
        stray.mkdir(exist_ok=True)
        (stray / "model.py").write_text("# stray file\n", encoding="utf-8")
        (project_dir / "commands.json").write_text("[]", encoding="utf-8")
        return ""

    import backend.agents.runtime.invoke as invoke_mod
    monkeypatch.setattr(invoke_mod, "collect_agent_text", fake_collect_with_escape)

    ac = _make_agent_context()
    with caplog.at_level(logging.WARNING, logger="backend.agents.rdr.agent"):
        result = await reproduce(ac, ctx=run_ctx)

    # reproduce() itself should succeed (stray dir is an agent behaviour issue, not a crash)
    assert not result.failed

    # The stray directory must have been removed by the guard
    assert not (fake_root / "rdr_escape_test").exists(), (
        "Guard did not remove the stray directory at the (fake) repo root"
    )

    # A WARNING must have been emitted
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("escaped" in r.message for r in warning_records), (
        f"Expected escape warning in logs; got: {[r.message for r in warning_records]}"
    )
