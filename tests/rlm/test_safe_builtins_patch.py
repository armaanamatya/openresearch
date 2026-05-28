"""Tests for backend.agents.rlm.safe_builtins_patch — BUG-LR-011.

Verifies that:
1. globals() / locals() are callable inside a LocalREPL after the patch.
2. eval / exec / compile / input remain blocked (still None).
3. The iter-1 snippet from prj_09047604e591d969 completes without error.
"""
from __future__ import annotations

import pytest

import backend.agents.rlm.safe_builtins_patch  # noqa: F401 — apply patch
from rlm.environments import local_repl as _local_repl
from rlm.environments.local_repl import LocalREPL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_repl() -> LocalREPL:
    return LocalREPL(custom_tools=[])


# ---------------------------------------------------------------------------
# Positive: globals / locals are restored
# ---------------------------------------------------------------------------

def test_globals_is_callable() -> None:
    repl = _make_repl()
    r = repl.execute_code("print(type(globals()).__name__)")
    assert r.stderr == "" or "NoneType" not in r.stderr, f"unexpected stderr: {r.stderr!r}"
    assert "dict" in r.stdout


def test_locals_is_callable() -> None:
    repl = _make_repl()
    r = repl.execute_code("x = 1\nprint(type(locals()).__name__)")
    assert r.stderr == "" or "NoneType" not in r.stderr, f"unexpected stderr: {r.stderr!r}"
    assert "dict" in r.stdout


def test_globals_get_idiom_works() -> None:
    """The exact idiom that triggered BUG-LR-011: globals().get("key", default)."""
    repl = _make_repl()
    r = repl.execute_code(
        'state = globals().get("report_state", {"iter": 0})\n'
        "print(state['iter'])"
    )
    assert "NoneType" not in r.stderr, f"unexpected error: {r.stderr!r}"
    assert r.stdout.strip() == "0"


def test_iter1_sdar_snippet_no_crash() -> None:
    """Replay the iter-1 snippet that caused the 2026-05-28 death-spiral."""
    repl = _make_repl()
    code = (
        "results = {}\n"
        'for n in ["check_user_messages", "understand_section"]:\n'
        "    obj = globals().get(n)\n"
        "    results[n] = (obj is not None)\n"
        "print(results)\n"
    )
    r = repl.execute_code(code)
    assert "NoneType" not in r.stderr, f"death-spiral snippet still crashes: {r.stderr!r}"


# ---------------------------------------------------------------------------
# Negative: genuine security boundary still intact
# ---------------------------------------------------------------------------

def test_eval_still_blocked() -> None:
    assert _local_repl._SAFE_BUILTINS.get("eval") is None


def test_exec_still_blocked() -> None:
    assert _local_repl._SAFE_BUILTINS.get("exec") is None


def test_compile_still_blocked() -> None:
    assert _local_repl._SAFE_BUILTINS.get("compile") is None


def test_input_still_blocked() -> None:
    assert _local_repl._SAFE_BUILTINS.get("input") is None


def test_globals_restored_to_builtin() -> None:
    import builtins
    assert _local_repl._SAFE_BUILTINS["globals"] is builtins.globals


def test_locals_restored_to_builtin() -> None:
    import builtins
    assert _local_repl._SAFE_BUILTINS["locals"] is builtins.locals
