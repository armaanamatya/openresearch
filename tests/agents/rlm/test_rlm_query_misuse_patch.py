"""BUG-NEW-033 (ported 2026-06-10): (slice, question) misuse auto-recovery.

The upstream ``rlm`` library exposes ``rlm_query(prompt, model=None)`` /
``llm_query(prompt, model=None)``. A root model calling the two-arg
``rlm_query(slice, question)`` form binds the question to ``model``, which
flows to ``claude --model "<question>"`` — the CLI error string then comes
back as the "answer" and poisons ``paper_claims`` (SDAR attempt 4: verdict
failed at score 0.0 with zero GPU spent). The patch detects a question-shaped
``model`` arg, composes a single prompt, and drops the bogus ``model=``.
"""

from __future__ import annotations

import pytest

from backend.agents.rlm.rlm_query_misuse_patch import (
    _looks_like_question,
    apply_rlm_query_misuse_patch,
)


class TestLooksLikeQuestion:
    @pytest.mark.parametrize(
        "value",
        [
            "What is the core algorithmic contribution of this paper?",
            "two words",
            "x" * 81,  # >80 chars, even without whitespace
        ],
    )
    def test_question_shaped_values(self, value):
        assert _looks_like_question(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            None,
            "gpt-5",
            "claude-sonnet-4-6",
            "qwen3-coder",
            "",
            42,  # non-str must never classify as question
        ],
    )
    def test_model_shaped_values(self, value):
        assert _looks_like_question(value) is False


@pytest.fixture()
def patched_repl(monkeypatch):
    """A fake LocalREPL with recording _rlm_query/_llm_query, then patched."""
    from rlm.environments import local_repl

    calls: list[tuple] = []

    def _fake_query(self, prompt, model=None, *args, **kwargs):
        calls.append((prompt, model))
        return f"answer-to:{prompt[:30]}"

    # Replace the real methods with the recorder, then apply the patch ON TOP
    # so the wrapper delegates to the recorder. monkeypatch restores the
    # originals afterwards (the production patch applied at run.py import
    # stays in place for the rest of the process either way — it is
    # behavior-preserving for correct calls).
    monkeypatch.setattr(local_repl.LocalREPL, "_rlm_query", _fake_query)
    monkeypatch.setattr(local_repl.LocalREPL, "_llm_query", _fake_query)
    apply_rlm_query_misuse_patch()

    repl = local_repl.LocalREPL.__new__(local_repl.LocalREPL)  # no __init__ needed
    return repl, calls


def test_two_arg_misuse_is_composed_and_model_dropped(patched_repl, capsys):
    repl, calls = patched_repl
    question = "What is the core algorithmic contribution of this paper?"

    result = repl._rlm_query("SLICE TEXT", question)

    assert calls == [(f"SLICE TEXT\n\nQuestion: {question}", None)]
    assert result.startswith("answer-to:")
    assert "rlm_query_misuse_patch" in capsys.readouterr().err


def test_correct_single_prompt_call_is_untouched(patched_repl, capsys):
    repl, calls = patched_repl

    repl._llm_query("SLICE\n\nQuestion: composed already")

    assert calls == [("SLICE\n\nQuestion: composed already", None)]
    assert capsys.readouterr().err == ""


def test_legitimate_model_override_is_untouched(patched_repl, capsys):
    repl, calls = patched_repl

    repl._rlm_query("a prompt", "gpt-5")

    assert calls == [("a prompt", "gpt-5")]
    assert capsys.readouterr().err == ""


def test_system_prompt_no_longer_teaches_two_arg_form():
    """The doc-fix half of BUG-NEW-033: the prompt must teach the composed
    single-prompt form, never `rlm_query(slice, question)` as an API."""
    import backend.agents.rlm.system_prompt as sp_mod
    from pathlib import Path

    text = Path(sp_mod.__file__).read_text(encoding="utf-8")
    assert "rlm_query(context_slice, query)" not in text
    assert "rlm_query(slice, specific_question)` over" not in text
    assert "NEVER call `rlm_query(slice, question)`" in text


def test_run_py_imports_the_patch_with_noqa():
    """Audit 2026-06-11: ruff --fix DELETED this side-effect import once
    (it lacked the noqa marker), silently disabling BUG-NEW-033 in
    production while these unit tests stayed green. Pin the wiring at the
    source level, noqa included, so an autofix breaks CI instead."""
    from pathlib import Path

    import backend.agents.rlm.run as run_mod

    src = Path(run_mod.__file__).read_text(encoding="utf-8")
    assert (
        "from backend.agents.rlm import rlm_query_misuse_patch "
        "as _rlm_query_misuse_patch  # noqa: F401"
    ) in src
