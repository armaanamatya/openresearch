"""Guard the code-fence patch: ```python/```py must be extracted, bare ``` must not.

Root cause it locks in: the upstream rlm parser matched only ```repl, so a
grok-4.3 root that fenced every block as ```python had ZERO blocks extracted and
the empty-code-block degenerate detector killed the run at iteration 3 even though
the root was driving the loop. See backend/agents/rlm/code_fence_patch.py.
"""
from __future__ import annotations

# Importing the patch module applies it (mirror of safe_builtins_patch).
from backend.agents.rlm import code_fence_patch  # noqa: F401
from rlm.core import rlm as core_rlm
from rlm.utils import parsing


# The exact iteration-1 response from the dead grok-4.3 run (sdar_gcp_grok3).
_GROK_ITER1 = (
    "```python\n"
    'msgs = check_user_messages()\n'
    'print("User messages:", msgs)\n'
    "if not msgs:\n"
    '    print("No steering messages. Keys in context:", list(context.keys()))\n'
    "```"
)


def test_both_bindings_are_patched():
    # The loop calls rlm.core.rlm.find_code_blocks (imported by-name); the source
    # is rlm.utils.parsing.find_code_blocks. Both must point at the patch.
    assert parsing.find_code_blocks is code_fence_patch._find_code_blocks
    assert core_rlm.find_code_blocks is code_fence_patch._find_code_blocks


def test_extracts_grok_python_fence_regression():
    blocks = core_rlm.find_code_blocks(_GROK_ITER1)
    assert len(blocks) == 1
    assert blocks[0].startswith("msgs = check_user_messages()")
    assert "list(context.keys())" in blocks[0]


def test_still_extracts_repl_fence_backward_compat():
    text = "Thinking...\n```repl\nx = 1\nprint(x)\n```\nDone."
    assert parsing.find_code_blocks(text) == ["x = 1\nprint(x)"]


def test_extracts_py_and_case_insensitive():
    assert parsing.find_code_blocks("```py\na = 2\n```") == ["a = 2"]
    assert parsing.find_code_blocks("```Python\nb = 3\n```") == ["b = 3"]


def test_bare_fence_is_not_executed():
    # A traceback quoted in a BARE ``` fence must be skipped, so only the real
    # ```python block runs — the safety property that keeps the patch tight.
    text = (
        "Here is the error:\n"
        "```\nTraceback (most recent call last): boom\n```\n"
        "Now the fix:\n"
        "```python\nfixed = True\n```"
    )
    assert parsing.find_code_blocks(text) == ["fixed = True"]


def test_multiple_blocks_in_order():
    text = "```python\nstep = 1\n```\nmid\n```repl\nstep = 2\n```"
    assert parsing.find_code_blocks(text) == ["step = 1", "step = 2"]


def test_pure_prose_returns_empty():
    assert parsing.find_code_blocks("No code here, just prose about the paper.") == []
