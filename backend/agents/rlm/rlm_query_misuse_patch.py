"""Auto-recover from the (slice, question) misuse of rlm_query / llm_query.

The upstream ``rlm`` library exposes ``rlm_query(prompt, model=None)`` and
``llm_query(prompt, model=None)`` as REPL globals. Both take a single composed
prompt — there is no two-argument (slice, question) form. When the root model
calls ``rlm_query(context, "What is the core algorithmic contribution...")``,
positional binding makes ``model=<the question text>``. That value flows into
``claude-agent-sdk`` → ``claude --model "<the question>"``, the CLI rejects the
"model name", and returns the error text *as the response body*. The library
treats the error string as a valid completion and the root model stores it as
the answer. Result on SDAR attempt 4: ``paper_claims.core_contribution`` was a
literal CLI error message, the implementer never learned the canonical Qwen
paths, every ``run_experiment`` pre-flight blocked, and the run ended
``verdict=failed`` with score 0.0 having spent zero GPU.

Two-pronged remediation:

  1. (this module) Patch ``LocalREPL._rlm_query`` / ``_llm_query`` to detect
     when the ``model`` arg looks like a question rather than a model name
     (contains whitespace OR exceeds 80 chars), auto-compose
     ``f"{prompt}\\n\\nQuestion: {model}"``, drop the bogus ``model=``, and
     emit a stderr warning. The patch preserves correct calls untouched.

  2. (sibling change) Doc-fix in ``system_prompt.py`` and ``primitives.py`` —
     stop teaching ``rlm_query(slice, specific_question)`` and start teaching
     ``rlm_query(f"<slice>\\n\\nQuestion: <q>")``. The patch is a backstop for
     models that follow the old prompt text from training.

Root cause: BUG-NEW-033 (2026-05-29 SDAR attempt 4 post-mortem).
"""
from __future__ import annotations

import sys

from rlm.environments import local_repl as _local_repl


# A "model name" in practice is short (≤80 chars), no whitespace, and looks
# like an identifier. A question is the opposite. We classify by the cheapest
# disambiguator that catches the bug: whitespace OR length.
def _looks_like_question(s: object) -> bool:
    if not isinstance(s, str):
        return False
    if len(s) > 80:
        return True
    if any(c.isspace() for c in s):
        return True
    return False


def _wrap(method_name: str) -> None:
    original = getattr(_local_repl.LocalREPL, method_name)

    def wrapper(self, prompt, model=None, *args, **kwargs):
        if _looks_like_question(model):
            sys.stderr.write(
                f"[rlm_query_misuse_patch] {method_name}: positional `model` arg "
                f"looks like a question ({len(model)} chars, contains whitespace) — "
                f"composing into single prompt and dropping bogus model=. "
                f"Correct API: {method_name.lstrip('_')}"
                f"(f'<slice>\\n\\nQuestion: <q>').\n"
            )
            composed = f"{prompt}\n\nQuestion: {model}"
            return original(self, composed, None, *args, **kwargs)
        return original(self, prompt, model, *args, **kwargs)

    wrapper.__name__ = method_name
    wrapper.__doc__ = (
        f"{(original.__doc__ or '').strip()}\n\n"
        f"Patched by rlm_query_misuse_patch — auto-composes when `model` "
        f"positional arg looks like a question (BUG-NEW-033)."
    )
    setattr(_local_repl.LocalREPL, method_name, wrapper)


def apply_rlm_query_misuse_patch() -> None:
    _wrap("_rlm_query")
    _wrap("_llm_query")


apply_rlm_query_misuse_patch()
