"""Tests for the SDAR env interface ABC (2026-05-31 OOM/GPU remediation, comp 2).

The contract: a ``*Env`` that subclasses :class:`BaseEnv` but forgets one of the
two required methods must fail at *construction* with a ``TypeError`` that names
the missing method — never mid-rollout with an ``AttributeError`` (the
2026-05-31 `prj_09047604e591d969` failure mode).
"""

from __future__ import annotations

import pytest

from backend.agents.rlm.sdar_env_base import BaseEnv


def test_complete_subclass_constructs_and_methods_work():
    class CompleteEnv(BaseEnv):
        def build_student_prompt(self, *args, **kwargs) -> str:
            return "student"

        def build_teacher_prompt(self, *args, **kwargs) -> str:
            return "teacher"

    env = CompleteEnv()  # must NOT raise
    assert env.build_student_prompt() == "student"
    assert env.build_teacher_prompt(anything=1) == "teacher"


def test_missing_student_prompt_raises_typeerror_on_construction():
    class MissingStudentEnv(BaseEnv):
        def build_teacher_prompt(self, *args, **kwargs) -> str:
            return "teacher"

    with pytest.raises(TypeError) as exc:
        MissingStudentEnv()
    # The error names the exact missing method — this is the whole point.
    assert "build_student_prompt" in str(exc.value)


def test_missing_teacher_prompt_raises_typeerror_on_construction():
    class MissingTeacherEnv(BaseEnv):
        def build_student_prompt(self, *args, **kwargs) -> str:
            return "student"

    with pytest.raises(TypeError) as exc:
        MissingTeacherEnv()
    assert "build_teacher_prompt" in str(exc.value)


def test_missing_both_raises_typeerror_naming_both():
    class EmptyEnv(BaseEnv):
        pass

    with pytest.raises(TypeError) as exc:
        EmptyEnv()
    msg = str(exc.value)
    assert "build_student_prompt" in msg
    assert "build_teacher_prompt" in msg


def test_base_env_itself_is_not_instantiable():
    with pytest.raises(TypeError):
        BaseEnv()


def test_copy_hook_lands_both_helpers_in_code_root(tmp_path):
    """run_with_sdk's copy hook must drop gpu_cell_runner.py + sdar_env_base.py
    into code/ so the agent's generated code can import them (comp 2b)."""
    from backend.agents.baseline_implementation import (
        _copy_harness_helpers_to_code_root,
        _HARNESS_CODE_HELPERS,
    )

    code_dir = tmp_path / "code"
    code_dir.mkdir()
    _copy_harness_helpers_to_code_root(code_dir)
    for helper in _HARNESS_CODE_HELPERS:
        assert (code_dir / helper).is_file(), f"{helper} not copied into code/"
    # The copied sdar_env_base is importable as a top-level module (zero deps).
    assert "build_student_prompt" in (code_dir / "sdar_env_base.py").read_text()


def test_copy_hook_is_fail_soft_on_unwritable_dest(tmp_path):
    """A copy failure must never raise — the agent can still emit the file itself."""
    from backend.agents.baseline_implementation import _copy_harness_helpers_to_code_root

    missing = tmp_path / "does" / "not" / "exist"  # parent absent → copy raises OSError
    _copy_harness_helpers_to_code_root(missing)  # must NOT raise


def test_permissive_signatures_accept_env_specific_params():
    """Each env builds prompts from its own observation shape — the ABC must not
    constrain the signature beyond 'the method exists and returns str'."""

    class WebShopEnv(BaseEnv):
        def build_student_prompt(self, observation, *, history=None) -> str:
            return f"obs={observation};hist={history}"

        def build_teacher_prompt(self, observation, gold_action) -> str:
            return f"obs={observation};gold={gold_action}"

    env = WebShopEnv()
    assert env.build_student_prompt("o1", history=["a"]) == "obs=o1;hist=['a']"
    assert env.build_teacher_prompt("o1", "buy") == "obs=o1;gold=buy"
