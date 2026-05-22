"""Tests for the Phase-3 Reproduction Agent (``backend/agents/rdr/agent.py``).

All tests mock ``collect_agent_text`` so that no real LLM or network calls
are made — the SDK infra is verified through call-argument inspection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from backend.agents.rdr.agent import reproduce
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
