"""Accept ```python / ```py fences (not only ```repl) in RLM root responses.

The upstream rlm code-block extractor (``rlm.utils.parsing.find_code_blocks``)
matches ONLY ```repl-fenced blocks::

    pattern = r"```repl\\s*\\n(.*?)\\n```"

The system prompt asks the root to fence REPL code as ```repl, but code-tuned
models routinely fence Python as ```python (or ```py) instead — a natural,
near-irresistible habit. When the root does, the extractor returns ZERO blocks:
nothing executes, ``RLMIteration.code_blocks`` is empty, and the empty-code-block
degenerate-loop detector (``run.py::_FatalBackendGateLogger``) aborts the run as
"pure prose" after ``OPENRESEARCH_DEGENERATE_REFUSAL_THRESHOLD`` iterations — even
though the root is driving the loop correctly. Observed live (2026-06-18): a
grok-4.3 SDAR run died at iteration 3 with every iteration emitting a clean
```python block that the parser silently dropped.

Fix (strictly additive, model-agnostic): broaden the accepted fence tag to
``repl | python | py`` (case-insensitive). The tag stays REQUIRED — a bare ```
fence is NOT accepted, so a root quoting a traceback/paper text in a bare fence is
never mis-executed. Existing ```repl runs (gpt-5 / claude) match exactly as
before and stay byte-identical; the patch can only turn a previously-IGNORED
```python block into an executed one — never the reverse.

``find_code_blocks`` is imported by-name into ``rlm.core.rlm`` (the loop's
caller at rlm/core/rlm.py:597), so we rebind BOTH the source module attribute and
that re-bound name.

Import once from run.py (after ``from rlm import RLM``). Mirror of
safe_builtins_patch.py.
"""
from __future__ import annotations

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


def _find_code_blocks(text: str) -> list[str]:
    """Drop-in replacement for ``rlm.utils.parsing.find_code_blocks``.

    Returns the stripped body of every repl/python/py-fenced block, in order.
    """
    return [m.group(1).strip() for m in _CODE_FENCE_PATTERN.finditer(text)]


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
