"""Role-model tokens `opus-foundry` / `sonnet-foundry` (Anthropic-on-Foundry).

These tokens route the executor/verifier/grader sub-roles onto the Azure AI
Foundry Anthropic-compatible endpoint (``.../anthropic/v1/messages``) while
still being classified as the Claude family — the validated sub-role baseline,
so no fidelity warning fires. See ``docs/superpowers/plans/2026-07-05-
reliable-autonomous-reproduction-foundation.md`` Task 5.
"""
from __future__ import annotations

from backend.agents.rlm.role_models import PROVIDER_ANTHROPIC_FOUNDRY, resolve_role_models


def test_sonnet_foundry_executor_and_grader():
    sel = resolve_role_models(
        planner_token="claude-oauth",
        cli_models="executor=sonnet-foundry,grader=sonnet-foundry",
    )
    assert sel.executor.provider == PROVIDER_ANTHROPIC_FOUNDRY
    assert sel.executor.model == "claude-sonnet-5"
    assert sel.executor.family == "claude"
    assert sel.grader.model == "claude-sonnet-5"


def test_unset_roles_are_byte_identical_none():
    sel = resolve_role_models(planner_token="claude-oauth")
    assert sel.executor is None or sel.executor.provider != PROVIDER_ANTHROPIC_FOUNDRY
