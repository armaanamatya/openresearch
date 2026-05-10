from __future__ import annotations

import asyncio

import pytest

from backend.agents.runtime.base import RuntimeGuard, RuntimeGuardViolation
from backend.agents.runtime.openai_runtime import _bash_tool, _unsupported_web_fetch_tool


def test_runtime_guard_normalizes_blocked_github_url() -> None:
    guard = RuntimeGuard(
        blocked_terms=("https://github.com/BartekCupial/finetuning-RL-as-CL",)
    )

    assert guard.find_blocked_term(
        "git clone https://github.com/BartekCupial/finetuning-RL-as-CL.git"
    )


def test_openai_adapter_blocks_guarded_bash_and_web_fetch(tmp_path) -> None:
    guard = RuntimeGuard(
        blocked_terms=("https://github.com/BartekCupial/finetuning-RL-as-CL",)
    )
    bash = _bash_tool(tmp_path, guard)
    web_fetch = _unsupported_web_fetch_tool(guard)

    with pytest.raises(RuntimeGuardViolation):
        asyncio.run(
            bash("git clone https://github.com/BartekCupial/finetuning-RL-as-CL.git")
        )

    with pytest.raises(RuntimeGuardViolation):
        web_fetch("https://github.com/BartekCupial/finetuning-RL-as-CL")


def test_runtime_guard_handles_arbitrary_text_with_brackets() -> None:
    """Regression: agent narration containing [brackets] caused
    ValueError: Invalid IPv6 URL in Python 3.12+ urllib.parse."""

    guard = RuntimeGuard(blocked_terms=("github.com/wolczyk/ftrl",))

    # Bracketed sequences that previously crashed urlparse.
    samples = [
        "Now I have sufficient information [some bracketed thing] to build PaperClaimMap",
        "checkpoint at [::1]:8080 — IPv6 literal in narration",
        "result: [unclosed bracket continues...",
        "[]",
        "https://[malformed",
    ]
    for sample in samples:
        # Must not raise.
        assert guard.find_blocked_term(sample) is None


def test_runtime_guard_normalizes_blocked_term_with_brackets() -> None:
    """Even a malformed blocked term must not crash term normalization."""

    guard = RuntimeGuard(blocked_terms=("github.com/foo[bar",))
    # find_blocked_term should not raise even on weird configured terms.
    assert guard.find_blocked_term("benign text") is None
    # The literal blocked term should still match its own substring.
    assert guard.find_blocked_term("see github.com/foo[bar in code") is not None
