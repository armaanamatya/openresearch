"""Copyable base class for SDAR-style teacher/student environments.

The 2026-05-31 failure (`prj_09047604e591d969`).  Every `alfworld` cell of the
SDAR matrix died with::

    AttributeError: 'ALFWorldEnv' object has no attribute 'build_student_prompt'

The agent's trainer (`code/sdar/train.py`) called ``env.build_student_prompt(...)``
on an ``ALFWorldEnv`` that simply never defined it.  An ``AttributeError`` raised
*mid-grid*, after model load + a rollout had already burned GPU minutes, is the
worst possible place to discover a one-line interface gap: 18 cells, zero metrics,
rubric 0.0.

This module turns that runtime ``AttributeError`` into a **construction-time
``TypeError``**.  Any environment that subclasses :class:`BaseEnv` but forgets
``build_student_prompt`` / ``build_teacher_prompt`` cannot even be instantiated —
Python's ``ABCMeta`` refuses ``ALFWorldEnv()`` with::

    TypeError: Can't instantiate abstract class ALFWorldEnv with abstract
    methods build_student_prompt, build_teacher_prompt

…which surfaces at env-construction (cell start), before the model loads, and
names the exact missing method.  Paired with the AST pre-flight backstop
(`preflight_ast._check_env_interface_contract`) that catches a non-subclassing
``*Env`` *before* the grid runs at all.

Copyable helper — mirror of the ``gpu_cell_runner.py`` / ``rubric_guard.py``
pattern.  ``run_with_sdk`` copies this file into ``code/sdar_env_base.py`` and the
``implement_baseline`` prompt instructs the agent to::

    from sdar_env_base import BaseEnv

    class ALFWorldEnv(BaseEnv):
        def build_student_prompt(self, *args, **kwargs) -> str: ...
        def build_teacher_prompt(self, *args, **kwargs) -> str: ...

Zero non-stdlib dependencies, so the copy-and-paste route always works inside an
agent sandbox.  Auth-agnostic by construction (no provider branching, no LLM
calls).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

__all__ = ["BaseEnv"]


class BaseEnv(ABC):
    """Interface every SDAR teacher/student environment must satisfy.

    The signatures are deliberately permissive (``*args, **kwargs``): each
    environment (ALFWorld, WebShop, Search-QA, …) builds its prompts from
    whatever observation/state shape fits its own data.  The *only* contract
    this base enforces is that the two methods the SDAR trainer calls **exist**,
    so a missing one fails loudly at construction rather than mid-rollout.

    Subclasses are free to add ``reset`` / ``step`` / any env-specific surface;
    those are not abstracted here because they are not the cross-cutting
    invariant that broke the 2026-05-31 run.
    """

    @abstractmethod
    def build_student_prompt(self, *args: Any, **kwargs: Any) -> str:
        """Return the prompt shown to the *student* policy for one turn/episode.

        Called by the SDAR trainer on every rollout.  Must return a ``str``.
        """
        raise NotImplementedError

    @abstractmethod
    def build_teacher_prompt(self, *args: Any, **kwargs: Any) -> str:
        """Return the prompt shown to the *teacher* policy for self-distillation.

        Called by the SDAR trainer when computing the teacher/student gap.  Must
        return a ``str``.
        """
        raise NotImplementedError
