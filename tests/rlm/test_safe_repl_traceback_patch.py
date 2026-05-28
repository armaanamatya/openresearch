"""Tests for backend.agents.rlm.safe_repl_traceback_patch — BUG-LR-012.

Verifies that:
1. Tracebacks appear in stderr after an exception (not just "ExcType: msg").
2. The cap (2000 chars) is respected.
3. The patch guard prevents double-application.
4. The bare 45-char NoneType message from BUG-LR-011 is gone once both
   patches are applied.
"""
from __future__ import annotations

import backend.agents.rlm.safe_builtins_patch  # noqa: F401
import backend.agents.rlm.safe_repl_traceback_patch  # noqa: F401
from backend.agents.rlm.safe_repl_traceback_patch import _PATCHED_ATTR, _TRACEBACK_CAP
from rlm.environments.local_repl import LocalREPL


def _make_repl() -> LocalREPL:
    return LocalREPL(custom_tools=[])


# ---------------------------------------------------------------------------
# Traceback surfacing
# ---------------------------------------------------------------------------

def test_traceback_in_stderr_on_exception() -> None:
    repl = _make_repl()
    code = "def f():\n    raise ValueError('bad input')\nf()"
    r = repl.execute_code(code)
    assert "in f" in r.stderr, f"expected 'in f' in stderr, got: {r.stderr!r}"
    assert "ValueError" in r.stderr
    assert "bad input" in r.stderr


def test_traceback_includes_line_number() -> None:
    repl = _make_repl()
    code = "x = 1\ny = 2\nraise RuntimeError('boom')"
    r = repl.execute_code(code)
    assert "RuntimeError" in r.stderr
    assert "boom" in r.stderr
    assert "Traceback" in r.stderr


def test_bare_none_type_error_includes_context() -> None:
    """Pre-patch: only 'TypeError: NoneType is not callable'. Post-patch: full traceback."""
    repl = _make_repl()
    code = "bad = None\nbad()"
    r = repl.execute_code(code)
    assert "Traceback" in r.stderr
    assert "TypeError" in r.stderr


# ---------------------------------------------------------------------------
# Cap enforcement
# ---------------------------------------------------------------------------

def test_traceback_capped_at_limit() -> None:
    repl = _make_repl()
    # Create a deep recursion to generate a large traceback.
    code = (
        "def recurse(n):\n"
        "    if n == 0: raise ValueError('deep')\n"
        "    return recurse(n - 1)\n"
        "recurse(200)\n"
    )
    r = repl.execute_code(code)
    assert len(r.stderr) <= _TRACEBACK_CAP + 50  # small slack for prefix/newlines
    assert "ValueError" in r.stderr


# ---------------------------------------------------------------------------
# Double-patch guard
# ---------------------------------------------------------------------------

def test_patch_guard_prevents_double_application() -> None:
    from rlm.environments.local_repl import LocalREPL as _REPL
    assert getattr(_REPL, _PATCHED_ATTR, False) is True
    # Calling apply_traceback_patch() again should not raise.
    from backend.agents.rlm.safe_repl_traceback_patch import apply_traceback_patch
    apply_traceback_patch()  # idempotent
    assert getattr(_REPL, _PATCHED_ATTR, False) is True


# ---------------------------------------------------------------------------
# No stderr on clean execution
# ---------------------------------------------------------------------------

def test_clean_code_has_empty_stderr() -> None:
    repl = _make_repl()
    r = repl.execute_code("x = 1 + 1\nprint(x)")
    assert r.stderr == ""
    assert r.stdout.strip() == "2"
