"""Tests for the SDAR teacher/student env interface pre-flight check.

Covers ``preflight_ast._check_env_interface_contract`` and its wiring into the
public ``scan_code_dir`` API.

The 2026-05-31 failure (`prj_09047604e591d969`): every `alfworld` cell of the SDAR
matrix died mid-grid with::

    AttributeError: 'ALFWorldEnv' object has no attribute 'build_student_prompt'

The bug slipped past pre-flight because the SDAR envs live at ``code/sdar/envs/*.py``
(two levels deep) while the other checks only glob one level. The new check walks
RECURSIVELY (``rglob``) and self-scopes to papers that actually use the contract.

Documented behavior decisions exercised here:
  * In play iff ANY file imports ``sdar_env_base`` / names ``BaseEnv`` / references
    ``build_student_prompt`` / ``build_teacher_prompt``. Otherwise: flag nothing.
  * A ``*Env`` that subclasses ``BaseEnv`` is NOT flagged — the ABC enforces the
    methods at construction (a loud, named ``TypeError``).
  * A ``*Env`` that neither subclasses ``BaseEnv`` nor defines the methods IS
    flagged when the contract is in play — including a class like ``ConfigEnv`` that
    merely ends in ``Env`` but is not really an environment. The agent fix is to
    subclass ``BaseEnv`` or rename the class so it no longer ends in ``Env``.
"""
from __future__ import annotations

from pathlib import Path

from backend.agents.rlm.preflight_ast import (
    PreflightViolation,
    _check_env_interface_contract,
    scan_code_dir,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(code_dir: Path, name: str, body: str) -> Path:
    p = code_dir / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


def _hard(vs: list[PreflightViolation]) -> list[PreflightViolation]:
    return [v for v in vs if v.severity == "hard"]


def _env_contract_hits(vs: list[PreflightViolation]) -> list[PreflightViolation]:
    """The env-contract check is the only one that uses this composite attr."""
    return [
        v
        for v in vs
        if v.missing_attr == "build_student_prompt/build_teacher_prompt"
    ]


# ---------------------------------------------------------------------------
# Case 1: recursive walk + self-scoping-by-reference (the 2026-05-31 bug)
# ---------------------------------------------------------------------------


def test_missing_env_methods_nested_in_play_by_reference_blocks(tmp_path: Path) -> None:
    """`sdar/envs/alfworld.py` defines a bare `ALFWorldEnv` (no BaseEnv, no methods)
    and `sdar/train.py` calls `env.build_student_prompt(...)`.

    The reference in train.py puts the contract "in play"; the nested env file is
    reached only by the recursive walk. EXACTLY ONE hard violation naming
    ALFWorldEnv at "sdar/envs/alfworld.py".
    """
    _write(tmp_path, "sdar/envs/alfworld.py", """\
class ALFWorldEnv:
    def __init__(self, data_path):
        self.data_path = data_path

    def reset(self):
        return "obs0"

    def step(self, action):
        return "obs1", 0.0, False
""")
    _write(tmp_path, "sdar/train.py", """\
def train(env):
    prompt = env.build_student_prompt("obs0")
    return prompt
""")

    out: list[PreflightViolation] = []
    _check_env_interface_contract(tmp_path, out)

    hits = _env_contract_hits(out)
    assert len(hits) == 1, f"Expected exactly one env-contract violation, got: {out}"
    v = hits[0]
    assert v.class_name == "ALFWorldEnv", f"Expected ALFWorldEnv, got {v.class_name}"
    assert v.file == "sdar/envs/alfworld.py", f"Expected nested rel path, got {v.file}"
    assert v.severity == "hard"
    assert v.line > 0, "Violation should carry the class def line number"
    # Detail should name the file:line, the class, and the trainer call.
    assert "ALFWorldEnv" in v.detail
    assert "sdar/envs/alfworld.py" in v.detail
    assert "build_student_prompt" in v.detail and "build_teacher_prompt" in v.detail
    # Suggested fix should point at BaseEnv + the copied helper.
    assert "BaseEnv" in v.suggested_fix
    assert "sdar_env_base" in v.suggested_fix


def test_scan_code_dir_surfaces_env_violation_end_to_end(tmp_path: Path) -> None:
    """Integration through the public API: scan_code_dir returns the env violation
    for the case-1 shape (recursive walk wired into scan_code_dir)."""
    _write(tmp_path, "sdar/envs/alfworld.py", """\
class ALFWorldEnv:
    def reset(self):
        return "obs0"
""")
    _write(tmp_path, "sdar/train.py", """\
def train(env):
    return env.build_student_prompt("obs0")
""")

    violations = scan_code_dir(tmp_path)
    hits = _env_contract_hits(_hard(violations))
    assert len(hits) == 1, f"Expected one env-contract violation, got: {violations}"
    assert hits[0].class_name == "ALFWorldEnv"
    assert hits[0].file == "sdar/envs/alfworld.py"


# ---------------------------------------------------------------------------
# Case 2: complete subclass with both methods → NO violation
# ---------------------------------------------------------------------------


def test_complete_subclass_with_both_methods_does_not_block(tmp_path: Path) -> None:
    """`class ALFWorldEnv(BaseEnv):` defining both methods, with the import →
    NO violation (the ABC is the enforcement; the AST backstop steps aside)."""
    _write(tmp_path, "sdar/envs/alfworld.py", """\
from sdar_env_base import BaseEnv


class ALFWorldEnv(BaseEnv):
    def __init__(self, data_path):
        self.data_path = data_path

    def build_student_prompt(self, *args, **kwargs) -> str:
        return "student"

    def build_teacher_prompt(self, *args, **kwargs) -> str:
        return "teacher"
""")
    _write(tmp_path, "sdar/train.py", """\
def train(env):
    return env.build_student_prompt("obs0"), env.build_teacher_prompt("obs0")
""")

    out: list[PreflightViolation] = []
    _check_env_interface_contract(tmp_path, out)
    assert _env_contract_hits(out) == [], f"Complete subclass should not block: {out}"

    # Also clean through the public API.
    assert _env_contract_hits(scan_code_dir(tmp_path)) == []


# ---------------------------------------------------------------------------
# Case 3: not in play (no contract signal anywhere) → NO violation
# ---------------------------------------------------------------------------


def test_unrelated_env_not_in_play_does_not_block(tmp_path: Path) -> None:
    """A bare `class ALFWorldEnv:` with no methods and no BaseEnv — but with NO
    reference to build_student_prompt/build_teacher_prompt and NO sdar_env_base /
    BaseEnv anywhere → self-scoping skips it. NO violation."""
    _write(tmp_path, "sdar/envs/alfworld.py", """\
class ALFWorldEnv:
    def __init__(self, data_path):
        self.data_path = data_path

    def reset(self):
        return "obs0"

    def step(self, action):
        return "obs1", 0.0, False
""")
    _write(tmp_path, "sdar/train.py", """\
def train(env):
    obs = env.reset()
    return obs
""")

    out: list[PreflightViolation] = []
    _check_env_interface_contract(tmp_path, out)
    assert out == [], f"Contract not in play → must flag nothing, got: {out}"

    # No env-contract violation through the public API either.
    assert _env_contract_hits(scan_code_dir(tmp_path)) == []


# ---------------------------------------------------------------------------
# Case 4: ConfigEnv (ends in Env, not really an env) while contract IS in play.
# Documented behavior: it WILL be flagged. The agent fix is to subclass BaseEnv
# or rename. The genuine SDAR env (subclasses BaseEnv) is NOT flagged.
# ---------------------------------------------------------------------------


def test_config_env_is_flagged_when_contract_in_play(tmp_path: Path) -> None:
    """When the contract is in play (a real env subclasses BaseEnv with both
    methods), a sibling `class ConfigEnv:` that merely ends in `Env` IS flagged —
    documented conservative-name behavior. The real env is left alone."""
    _write(tmp_path, "sdar/envs/webshop.py", """\
from sdar_env_base import BaseEnv


class WebShopEnv(BaseEnv):
    def build_student_prompt(self, *args, **kwargs) -> str:
        return "student"

    def build_teacher_prompt(self, *args, **kwargs) -> str:
        return "teacher"
""")
    _write(tmp_path, "sdar/config.py", """\
class ConfigEnv:
    def __init__(self):
        self.lr = 1e-5
        self.batch_size = 8
""")

    out: list[PreflightViolation] = []
    _check_env_interface_contract(tmp_path, out)
    hits = _env_contract_hits(out)
    flagged = {v.class_name for v in hits}
    assert flagged == {"ConfigEnv"}, (
        f"Only ConfigEnv should be flagged (WebShopEnv subclasses BaseEnv): {out}"
    )
    only = hits[0]
    assert only.file == "sdar/config.py"
    assert only.severity == "hard"


# ---------------------------------------------------------------------------
# Robustness: fail-soft, BaseEnv-itself-not-flagged, empty/missing dir.
# ---------------------------------------------------------------------------


def test_base_env_definition_itself_is_not_flagged(tmp_path: Path) -> None:
    """The BaseEnv ABC definition (ends in 'Env', defines the abstract methods)
    must never be flagged as a violation of its own contract."""
    _write(tmp_path, "sdar/sdar_env_base.py", """\
from abc import ABC, abstractmethod


class BaseEnv(ABC):
    @abstractmethod
    def build_student_prompt(self, *args, **kwargs) -> str:
        raise NotImplementedError

    @abstractmethod
    def build_teacher_prompt(self, *args, **kwargs) -> str:
        raise NotImplementedError
""")
    # A complete subclass keeps the contract in play AND clean.
    _write(tmp_path, "sdar/envs/alfworld.py", """\
from sdar_env_base import BaseEnv


class ALFWorldEnv(BaseEnv):
    def build_student_prompt(self, *args, **kwargs) -> str:
        return "s"

    def build_teacher_prompt(self, *args, **kwargs) -> str:
        return "t"
""")

    out: list[PreflightViolation] = []
    _check_env_interface_contract(tmp_path, out)
    assert _env_contract_hits(out) == [], f"BaseEnv itself must not be flagged: {out}"


def test_unparseable_file_is_failsoft(tmp_path: Path) -> None:
    """A file with a SyntaxError must not crash the env-contract check; the valid
    nested env is still evaluated."""
    _write(tmp_path, "sdar/broken.py", "def oops(:\n    pass\n")
    _write(tmp_path, "sdar/envs/alfworld.py", """\
class ALFWorldEnv:
    def reset(self):
        return "obs0"
""")
    _write(tmp_path, "sdar/train.py", """\
def train(env):
    return env.build_teacher_prompt("obs0")
""")

    out: list[PreflightViolation] = []
    _check_env_interface_contract(tmp_path, out)  # must not raise
    hits = _env_contract_hits(out)
    assert len(hits) == 1 and hits[0].class_name == "ALFWorldEnv", out


def test_empty_and_missing_dir_are_noops(tmp_path: Path) -> None:
    out: list[PreflightViolation] = []
    _check_env_interface_contract(tmp_path, out)  # empty dir
    assert out == []
    _check_env_interface_contract(
        Path("/tmp/__no_such_dir_env_contract_test__"), out
    )
    assert out == []


# ---------------------------------------------------------------------------
# Case 5 (2026-06-01): AgenticEnv subclasses + shipped concrete envs are valid
# bases. A *Env subclassing AgenticEnv (or a shipped env) inherits the prompt
# builders, so it must NOT be flagged even though it never defines them itself.
# ---------------------------------------------------------------------------


def test_agentic_env_subclass_not_flagged(tmp_path: Path) -> None:
    """`class SearchEnv(AgenticEnv):` implementing only reset/step is complete —
    AgenticEnv ships build_student_prompt/build_teacher_prompt. With the contract
    in play (the import + a trainer reference), there must be NO violation."""
    _write(tmp_path, "sdar/envs/search.py", """\
from sdar_env_base import AgenticEnv, StepResult


class SearchEnv(AgenticEnv):
    max_turns = 6

    def reset(self, *, seed=None, task=None) -> str:
        self._start_episode(system="search")
        self._record_obs("q")
        return "q"

    def step(self, action: str) -> StepResult:
        self._record_act(action)
        self._finish(1.0)
        return StepResult(observation="done", reward=1.0, done=True)
""")
    _write(tmp_path, "sdar/train.py", """\
def train(env):
    return env.build_student_prompt(), env.build_teacher_prompt()
""")

    out: list[PreflightViolation] = []
    _check_env_interface_contract(tmp_path, out)
    assert _env_contract_hits(out) == [], (
        f"AgenticEnv subclass implements the contract by inheritance: {out}"
    )
    assert _env_contract_hits(scan_code_dir(tmp_path)) == []


def test_subclass_of_shipped_env_not_flagged(tmp_path: Path) -> None:
    """A `*Env` subclassing a harness-shipped concrete env (SearchQAEnv /
    ALFWorldEnv / WebShopEnv) is not flagged — those bases already satisfy the
    contract. The contract is in play via the build_student_prompt reference."""
    _write(tmp_path, "sdar/envs/custom.py", """\
from search_qa_env import SearchQAEnv


class MySearchQAEnv(SearchQAEnv):
    pass
""")
    _write(tmp_path, "sdar/train.py", """\
def train(env):
    return env.build_student_prompt()
""")

    out: list[PreflightViolation] = []
    _check_env_interface_contract(tmp_path, out)
    assert _env_contract_hits(out) == [], (
        f"Subclass of a shipped env must not be flagged: {out}"
    )
