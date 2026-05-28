"""Restore globals()/locals() inside rlm's LocalREPL sandbox.

The upstream rlm library blacklists ``globals`` and ``locals`` alongside
genuinely dangerous builtins (eval/exec/compile/input). Unlike those four,
``globals`` and ``locals`` are pure namespace getters with no code-execution
surface — Python's stdlib treats them as safe enough to ship in every REPL.

Blocking them by setting their entries to ``None`` means any model code that
calls ``globals().get("x", default)`` — a normal idiom for persisting state
between iterations — crashes with the bare message
``TypeError: 'NoneType' object is not callable``. The model sees no name
and no traceback (see safe_repl_traceback_patch.py for the latter half), so
it cannot diagnose what went wrong and typically spirals into a "primitives
unavailable" partial report.

This module patches the dict at import time. Import once from run.py BEFORE
``from rlm import RLM``.

Root cause: BUG-LR-011 (2026-05-28 SDAR death-spiral).
Design spec: docs/superpowers/specs/2026-05-28-rlm-stability-remediation-design.md
"""
from __future__ import annotations
import builtins as _builtins

from rlm.environments import local_repl as _local_repl


def apply_safe_builtins_patch() -> None:
    sb = _local_repl._SAFE_BUILTINS
    sb["globals"] = _builtins.globals
    sb["locals"] = _builtins.locals
    # NEVER restore eval/exec/compile/input — those are intentionally blocked.


apply_safe_builtins_patch()
