"""Extract RLM root code from ```python/```py fences AND native tool-call tokens.

The upstream rlm code-block extractor (``rlm.utils.parsing.find_code_blocks``)
matches ONLY ```repl-fenced blocks::

    pattern = r"```repl\\s*\\n(.*?)\\n```"

Two real-world models break that assumption; both abort the run via the
empty-code-block degenerate-loop detector (``run.py::_FatalBackendGateLogger``)
after ``OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD`` iterations even though the
root IS driving the loop correctly:

1. **Fence-tag drift.** Code-tuned models fence Python as ```python / ```py
   instead of ```repl (grok-4.3, 2026-06-18: died at iteration 3 emitting clean
   ```python the parser dropped). Fix: broaden the accepted tag to
   ``repl | python | py`` (case-insensitive). The tag stays REQUIRED — a bare
   ``` fence is never accepted, so a quoted traceback/paper slice is not
   mis-executed.

2. **Native tool-call channel.** Tool-trained models route REPL code through
   their native tool-call tokens instead of a markdown fence. Kimi-K2.6
   (Moonshot, 2026-06-18) emitted every iteration as::

       <|tool_calls_section_begin|><|tool_call_begin|>functions.repl:0
       <|tool_call_argument_begin|>{"code": "check_user_messages()\\n"}
       <|tool_call_end|><|tool_calls_section_end|>

   — i.e. the executable code lives in the JSON ``code`` argument of a
   ``functions.repl`` tool call, NOT a fence. ``find_code_blocks`` returned
   ``[]`` for all 3 iterations and the run died "pure prose" even though Kimi
   was correctly calling ``check_user_messages()``, reading ``paper_text``, and
   calling ``rlm_query(...)``. Fix: also extract the ``code`` arg of every
   ``repl`` tool call delimited by ``<|tool_call_begin|>…<|tool_call_end|>``.

Both fixes are strictly additive + model-agnostic: a model that emits neither a
broadened fence nor tool-call tokens (gpt-5 / claude on ```repl) matches exactly
as before and stays byte-identical. The patch can only turn a previously-IGNORED
block into an executed one — never the reverse.

``find_code_blocks`` is imported by-name into ``rlm.core.rlm`` (the loop's
caller at rlm/core/rlm.py:597), so we rebind BOTH the source module attribute and
that re-bound name.

Import once from run.py (after ``from rlm import RLM``). Mirror of
safe_builtins_patch.py.
"""
from __future__ import annotations

import json
import re

from rlm.utils import parsing as _parsing

# Identical structure to the upstream regex — REQUIRED tag, whitespace, a newline
# before the body, non-greedy DOTALL body, closing fence on its own line — with
# the tag broadened from ``repl`` to ``repl|python|py`` and matched
# case-insensitively (```Python / ```PY also accepted). No bare ``` on purpose.
_CODE_FENCE_PATTERN = re.compile(
    r"```(?:repl|python|py)\s*\n(.*?)\n```",
    re.DOTALL | re.IGNORECASE,
)

# Native tool-call channel (Kimi/Moonshot). Each call is delimited by
# ``<|tool_call_begin|>NAME<|tool_call_argument_begin|>ARG<|tool_call_end|>``
# where ARG is a JSON object carrying the REPL code under ``"code"``. Anchored on
# the special tokens, so a model that never emits them (gpt-5/claude/grok) never
# matches — strictly additive.
_TOOL_CALL_PATTERN = re.compile(
    r"<\|tool_call_begin\|>(?P<name>.*?)<\|tool_call_argument_begin\|>(?P<arg>.*?)<\|tool_call_end\|>",
    re.DOTALL,
)


def _code_from_tool_arg(arg: str) -> str | None:
    """Return the ``code`` string from a tool-call JSON argument, or None.

    Fail-soft: a malformed / non-``code`` argument yields None (skipped), never
    raises — a parser must never break the loop.
    """
    try:
        obj = json.loads(arg)
    except (ValueError, TypeError):
        return None
    if isinstance(obj, dict):
        code = obj.get("code")
        if isinstance(code, str) and code.strip():
            return code
    return None


def _find_tool_call_code(text: str) -> list[str]:
    """Extract the ``code`` arg of every ``repl`` native tool call in *text*."""
    if "<|tool_call_argument_begin|>" not in text:  # fast path: byte-identical for non-Kimi
        return []
    out: list[str] = []
    for m in _TOOL_CALL_PATTERN.finditer(text):
        # Only the repl/code tool carries executable REPL code (the model may
        # also name it ``functions.repl`` / ``repl:N`` — match on the substring).
        if "repl" not in m.group("name").lower():
            continue
        code = _code_from_tool_arg(m.group("arg").strip())
        if code:
            out.append(code.strip())
    return out


def _find_code_blocks(text: str) -> list[str]:
    """Drop-in replacement for ``rlm.utils.parsing.find_code_blocks``.

    Returns the stripped body of every repl/python/py-fenced block, followed by
    the ``code`` argument of every native ``repl`` tool call — so a root that
    fences its code AND a root that emits tool-call tokens both drive the loop.
    """
    blocks = [m.group(1).strip() for m in _CODE_FENCE_PATTERN.finditer(text)]
    blocks.extend(_find_tool_call_code(text))
    return blocks


def apply_code_fence_patch() -> None:
    _parsing.find_code_blocks = _find_code_blocks
    # rlm.core.rlm did ``from rlm.utils.parsing import find_code_blocks`` — that
    # binds its OWN module-level name, which is the one the loop actually calls.
    # Rebind it too (rlm.core.rlm is already imported via ``from rlm import RLM``).
    try:
        from rlm.core import rlm as _core_rlm

        _core_rlm.find_code_blocks = _find_code_blocks
    except Exception:  # noqa: BLE001 — never block import; the source patch still applies
        pass


apply_code_fence_patch()
