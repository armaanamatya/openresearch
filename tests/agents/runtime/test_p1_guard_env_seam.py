"""P1 / #7 Unit C — the env-var guard seam.

collect_agent_text is the single chokepoint EVERY agent invocation flows through
(baseline-implementation, rdr, patch-mode, future callers). It seeds the
RuntimeGuard from OPENRESEARCH_BLOCKED_TERMS_JSON when the caller passes no explicit
blocklist, so the guard is uniform + un-forgettable — no per-caller threading.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator

from backend.agents.runtime.base import (
    AgentRuntimeSpec,
    ProviderName,
    StreamEvent,
    StreamText,
    blocked_terms_from_env,
)

_REPO = "https://github.com/BartekCupial/finetuning-RL-as-CL"


class _CapturingRuntime:
    """Records the guard terms on the spec handed to run_agent."""

    def __init__(self) -> None:
        self.captured_guard_terms: tuple[str, ...] | None = None

    @property
    def provider_name(self) -> ProviderName:
        return "anthropic"

    async def run_agent(self, *, agent: AgentRuntimeSpec, user_input: str) -> AsyncIterator[StreamEvent]:
        self.captured_guard_terms = agent.guard.blocked_terms
        yield StreamText("ok")


# --- blocked_terms_from_env() unit tests ---

def test_env_parser_empty_when_unset(monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_BLOCKED_TERMS_JSON", raising=False)
    assert blocked_terms_from_env() == ()


def test_env_parser_parses_list_and_strips(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_BLOCKED_TERMS_JSON", json.dumps([_REPO, "  ", "github.com/x/y"]))
    assert blocked_terms_from_env() == (_REPO, "github.com/x/y")


def test_env_parser_malformed_is_empty(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_BLOCKED_TERMS_JSON", "{bad json")
    assert blocked_terms_from_env() == ()


def test_env_parser_non_list_is_empty(monkeypatch):
    monkeypatch.setenv("OPENRESEARCH_BLOCKED_TERMS_JSON", json.dumps({"a": 1}))
    assert blocked_terms_from_env() == ()


# --- collect_agent_text seeds the guard from the env seam ---

def test_collect_agent_text_seeds_guard_from_env(monkeypatch, tmp_path: Path):
    """No explicit blocklist + env set ⇒ the agent spec's guard carries the terms."""
    monkeypatch.setenv("OPENRESEARCH_BLOCKED_TERMS_JSON", json.dumps([_REPO]))
    runtime = _CapturingRuntime()
    from backend.agents.runtime.invoke import collect_agent_text

    asyncio.run(collect_agent_text(
        "baseline-implementation", "build", project_dir=tmp_path, runtime=runtime,
    ))
    assert runtime.captured_guard_terms == (_REPO,)


def test_collect_agent_text_explicit_overrides_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("OPENRESEARCH_BLOCKED_TERMS_JSON", json.dumps(["github.com/from/env"]))
    runtime = _CapturingRuntime()
    from backend.agents.runtime.invoke import collect_agent_text

    asyncio.run(collect_agent_text(
        "baseline-implementation", "build", project_dir=tmp_path, runtime=runtime,
        blocked_terms=("github.com/explicit/win",),
    ))
    assert runtime.captured_guard_terms == ("github.com/explicit/win",)


def test_collect_agent_text_no_env_empty_guard(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("OPENRESEARCH_BLOCKED_TERMS_JSON", raising=False)
    runtime = _CapturingRuntime()
    from backend.agents.runtime.invoke import collect_agent_text

    asyncio.run(collect_agent_text(
        "baseline-implementation", "build", project_dir=tmp_path, runtime=runtime,
    ))
    assert runtime.captured_guard_terms == ()
