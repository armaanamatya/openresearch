"""Scoped AST pre-flight — catches agent code-writing bugs BEFORE sandbox dispatch.

Motivation: the 2026-05-26 VAE crash (F6) —

    AttributeError: 'WakeSleepVAE' object has no attribute 'reparameterize'

The agent wrote ``model.reparameterize(z, mu, log_var)`` on a class that had no such
method. The bug surfaced AFTER 3+ hours of sandbox training. This module catches the
shape in milliseconds via a narrow, conservative AST + symbol-table walk, before
``run_experiment`` dispatches ``commands.json`` to the sandbox.

Design contract (narrow scope — prefer false negatives over false positives):

WHAT BLOCKS:
  * Same-file, same-class missing attribute access — ``obj.method()`` called on an
    instance of a class defined in the same file that does not define ``method``.
  * Same-file undefined name — a bare name call with no definition anywhere in the
    file (NameError shapes).
  * Pure syntax error — Python won't execute a file that doesn't parse.
  * Import-from of a locally-defined symbol that doesn't exist — ``from .models
    import VAE`` but no class or function named ``VAE`` in models.py.

WHAT DOES NOT BLOCK (conservative — avoid false positives):
  * Dynamic attribute access — ``setattr(model, "reparameterize", ...)`` BEFORE the
    call → skip. ``__setattr__`` override / ``**kwargs``-driven ``__init__`` → skip.
  * Cross-file attribute lookups on classes imported from external libraries — no
    type stubs for torch/numpy, so we'd have near-100% false-positive rate.
  * Inherited methods — if the class inherits from another class in the same file,
    we check the parent; if the parent is external (library), we skip.
  * Anything the resolver is not 100% sure about — conservative default is no block.

Public API
----------
``scan_code_dir(code_dir) -> list[PreflightViolation]``

Integration: called from ``validate_code_pre_flight`` in ``pre_flight_validator.py``
via the γ.1 hook block. Results are appended to the violations list with ``severity``
matching the existing ``hard/soft`` policy.

``PreflightViolation`` (γ.1) is a separate dataclass from ``PreFlightViolation``
(pre_flight_validator.py) to keep the two modules' public APIs independent. The hook
block converts between them.
"""
from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PreflightViolation:
    """A single AST pre-flight violation found in agent-written code.

    Attributes
    ----------
    file : str
        Relative file name within the code directory.
    line : int
        1-based line number of the offending call or import.
    class_name : str | None
        Class name the missing attribute was called on (when applicable).
    missing_attr : str | None
        The attribute / method name that is not defined.
    suggested_fix : str
        Concrete, actionable fix the agent can apply on the next
        ``implement_baseline`` iteration.
    severity : "hard" | "soft"
        All γ.1 violations are ``hard`` — calling a non-existent method is
        a guaranteed ``AttributeError`` at runtime.
    detail : str
        Human-readable description of the issue.
    """

    file: str
    line: int
    class_name: str | None
    missing_attr: str | None
    suggested_fix: str
    severity: Literal["hard", "soft"]
    detail: str

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "line": self.line,
            "class_name": self.class_name,
            "missing_attr": self.missing_attr,
            "suggested_fix": self.suggested_fix,
            "severity": self.severity,
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
# Symbol collector
# ---------------------------------------------------------------------------


def _collect_class_members(tree: ast.AST, class_name: str) -> set[str]:
    """Return the set of method/attribute names defined on ``class_name``.

    Walks function definitions (methods), class-level ``self.<attr> = ...``
    assignments in ``__init__``, and class-level name assignments. Does NOT
    follow inheritance chains into external-library classes.
    """
    members: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name != class_name:
            continue
        # Class-level function definitions (methods).
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                members.add(item.name)
            # Class-level simple assignments: ``name: int = 0`` / ``name = 0``
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        members.add(target.id)
            elif isinstance(item, ast.AnnAssign):
                if isinstance(item.target, ast.Name):
                    members.add(item.target.id)
        # Walk __init__ to pick up self.<attr> = ... assignments.
        for item in ast.walk(node):
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if item.name != "__init__":
                continue
            for stmt in ast.walk(item):
                if not isinstance(stmt, ast.Assign):
                    continue
                for target in stmt.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        members.add(target.attr)
        break  # found the class — stop scanning
    return members


def _collect_parent_classes(tree: ast.AST, class_name: str) -> list[str]:
    """Return the names of direct base classes defined in the SAME file."""
    local_class_names: set[str] = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    }
    parents: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for base in node.bases:
            # Simple name: ``class Foo(Bar):``
            if isinstance(base, ast.Name) and base.id in local_class_names:
                parents.append(base.id)
            # Attribute: ``class Foo(module.Bar):`` — skip if not a local name
        break
    return parents


def _collect_all_class_members_with_inheritance(
    tree: ast.AST,
    class_name: str,
    _seen: frozenset[str] | None = None,
) -> set[str]:
    """Collect members of ``class_name`` including locally-defined parent classes.

    Stops at any base class that is NOT defined in the same file (external library).
    """
    if _seen is None:
        _seen = frozenset()
    if class_name in _seen:
        return set()  # cycle guard
    members = _collect_class_members(tree, class_name)
    for parent in _collect_parent_classes(tree, class_name):
        members |= _collect_all_class_members_with_inheritance(
            tree, parent, _seen | {class_name}
        )
    return members


def _collect_top_level_names(tree: ast.AST) -> set[str]:
    """Return all top-level (module scope) names defined in the file."""
    names: set[str] = set()
    for node in tree.body:  # type: ignore[attr-defined]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


# ---------------------------------------------------------------------------
# Dynamic-access detection (suppresses false positives)
# ---------------------------------------------------------------------------


def _has_setattr_before(tree: ast.AST, class_name: str, attr_name: str) -> bool:
    """Return True if ``setattr(<name>, <attr_name>, ...)`` or
    ``__setattr__`` is defined on the class.

    Also returns True if any class in the file overrides ``__getattr__``,
    which is a common pattern for dynamic proxies that add arbitrary attrs.
    """
    for node in ast.walk(tree):
        # setattr(obj, "<attr_name>", <value>) — any occurrence anywhere in the file.
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "setattr"
            and len(node.args) >= 2
        ):
            key_arg = node.args[1]
            if isinstance(key_arg, ast.Constant) and isinstance(key_arg.value, str):
                if key_arg.value == attr_name:
                    return True

        # Class defines __setattr__ or __getattr__ → dynamic, skip.
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name in {"__setattr__", "__getattr__", "__getattribute__"}:
                        return True
            # Check if __init__ uses **kwargs → could add attrs dynamically.
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "__init__":
                    if item.args.varkw is not None:
                        return True

    return False


# ---------------------------------------------------------------------------
# Local import checker
# ---------------------------------------------------------------------------


def _collect_local_py_names(code_dir: Path, module_stem: str) -> set[str]:
    """Return the top-level names exported from ``<module_stem>.py`` in code_dir.

    Returns an empty set if the file doesn't exist or can't be parsed — the
    caller treats empty set as "can't verify → no violation".
    """
    candidate = code_dir / f"{module_stem}.py"
    if not candidate.is_file():
        return set()
    try:
        source = candidate.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(candidate))
        return _collect_top_level_names(tree)
    except SyntaxError:
        return set()  # syntax error in the imported module is its own violation
    except Exception:  # noqa: BLE001
        return set()


# ---------------------------------------------------------------------------
# Per-check implementations
# ---------------------------------------------------------------------------


def _check_missing_attr_access(
    tree: ast.AST,
    path: Path,
    out: list[PreflightViolation],
) -> None:
    """Detect ``obj.method()`` where ``obj`` is an instance of a class
    defined in the same file that does NOT define ``method``.

    Scope (narrow):
      * Only checks attribute access where the object is a simple Name whose
        assignment we can resolve to a class instantiation in the same file.
      * Only checks the call's direct class — does NOT follow cross-file
        inheritance.
      * Skips if ``setattr`` or dynamic-access patterns are present.
    """
    # Step 1: Build a map from variable name → class name for simple assignments
    # like ``model = WakeSleepVAE(...)`` at ANY scope in the file.
    # We only map unambiguous cases: the RHS is a direct class call.
    var_to_class: dict[str, str] = {}

    # Collect all class names defined in THIS file.
    local_classes: set[str] = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    }
    if not local_classes:
        return  # nothing to check

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        # RHS must be a direct call to a locally-defined class name.
        if not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        cls_name: str | None = None
        if isinstance(func, ast.Name) and func.id in local_classes:
            cls_name = func.id
        elif isinstance(func, ast.Attribute) and func.attr in local_classes:
            cls_name = func.attr
        if cls_name is None:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_to_class[target.id] = cls_name

    if not var_to_class:
        return

    # Step 2: For each attribute access on a tracked variable, check the class.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if not isinstance(node.value, ast.Name):
            continue
        var_name = node.value.id
        cls_name = var_to_class.get(var_name)
        if cls_name is None:
            continue
        attr_name = node.attr
        # Don't flag dunder attributes — __class__, __dict__, etc. are always present.
        if attr_name.startswith("__") and attr_name.endswith("__"):
            continue
        # Collect all members (including inherited from same-file parents).
        members = _collect_all_class_members_with_inheritance(tree, cls_name)
        if attr_name in members:
            continue
        # Suppress if dynamic-access patterns exist.
        if _has_setattr_before(tree, cls_name, attr_name):
            continue
        # Only flag attribute access that is part of a call or an assignment —
        # pure attribute reads are common in isinstance checks and are low-value.
        # We want to flag the runtime-fatal case: calling a nonexistent method.
        lineno = getattr(node, "lineno", 0)
        out.append(PreflightViolation(
            file=path.name,
            line=lineno,
            class_name=cls_name,
            missing_attr=attr_name,
            suggested_fix=(
                f"Define `{attr_name}` on `{cls_name}` (e.g. "
                f"`def {attr_name}(self, ...)`) or remove the call. "
                f"The class is defined in `{path.name}` and has no `{attr_name}` method/attribute."
            ),
            severity="hard",
            detail=(
                f"{path.name}:{lineno}: `{var_name}.{attr_name}` called but "
                f"`{cls_name}` (defined in same file) does not define `{attr_name}`."
            ),
        ))


def _check_undefined_names(
    tree: ast.AST,
    path: Path,
    out: list[PreflightViolation],
) -> None:
    """Flag bare name calls (``name(...)``) where ``name`` is not defined
    anywhere visible in the module.

    Conservative: only flags direct top-level name calls where the name is
    not in the module's top-level scope AND not a Python builtin. Does NOT
    flag names that come from a ``*`` import (too many false positives).
    """
    # Builtins and common magic names we always allow.
    _BUILTINS = frozenset(dir(__builtins__) if isinstance(__builtins__, dict) else dir(__builtins__))  # type: ignore[arg-type]
    _ALWAYS_ALLOW = frozenset({
        # Python builtins + common globals
        "print", "range", "len", "list", "dict", "set", "tuple", "str", "int",
        "float", "bool", "type", "isinstance", "issubclass", "hasattr", "getattr",
        "setattr", "delattr", "super", "object", "None", "True", "False",
        "NotImplemented", "Ellipsis", "open", "zip", "map", "filter", "enumerate",
        "sorted", "reversed", "min", "max", "sum", "abs", "round", "pow",
        "vars", "dir", "id", "hash", "repr", "hex", "oct", "bin", "ord", "chr",
        "input", "iter", "next", "all", "any", "callable", "classmethod",
        "staticmethod", "property", "Exception", "ValueError", "TypeError",
        "KeyError", "IndexError", "RuntimeError", "NotImplementedError",
        "StopIteration", "AttributeError", "NameError", "OSError", "IOError",
        # Commonly starred-imported in ML code — suppress rather than false-positive.
        "nn", "F", "optim", "torch", "np", "pd", "os", "sys", "math",
        "logging", "json", "time", "copy", "re",
    }) | _BUILTINS

    # Check if there's a star import — if so, we can't know what names are available.
    has_star_import = any(
        isinstance(node, ast.ImportFrom)
        and any(alias.name == "*" for alias in node.names)
        for node in ast.walk(tree)
    )
    if has_star_import:
        return  # too many unknowns → skip this check entirely

    defined_names = _collect_top_level_names(tree)
    # Also collect names from comprehension scopes, nested functions etc.
    all_assigned: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            all_assigned.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # function param names are always "defined"
            for arg in node.args.args + node.args.posonlyargs + getattr(node.args, "kwonlyargs", []):
                all_assigned.add(arg.arg)
            if node.args.vararg:
                all_assigned.add(node.args.vararg.arg)
            if node.args.kwarg:
                all_assigned.add(node.args.kwarg.arg)
        elif isinstance(node, ast.For):
            if isinstance(node.target, ast.Name):
                all_assigned.add(node.target.id)
        elif isinstance(node, ast.ExceptHandler):
            if node.name:
                all_assigned.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for n in node.names:
                all_assigned.add(n)
        elif isinstance(node, ast.withitem):
            if isinstance(node.optional_vars, ast.Name):
                all_assigned.add(node.optional_vars.id)
        elif isinstance(node, ast.NamedExpr):
            if isinstance(node.target, ast.Name):
                all_assigned.add(node.target.id)

    visible = defined_names | all_assigned | _ALWAYS_ALLOW

    for node in ast.walk(tree):
        # Only flag direct name calls: ``foo(...)`` not ``obj.foo(...)``
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        if name in visible:
            continue
        lineno = getattr(node, "lineno", 0)
        out.append(PreflightViolation(
            file=path.name,
            line=lineno,
            class_name=None,
            missing_attr=name,
            suggested_fix=(
                f"Define `{name}` before calling it, or import it at the top "
                f"of `{path.name}`."
            ),
            severity="hard",
            detail=(
                f"{path.name}:{lineno}: `{name}(...)` called but `{name}` is "
                f"not defined in the module scope (potential NameError)."
            ),
        ))


def _check_local_import_from(
    tree: ast.AST,
    path: Path,
    code_dir: Path,
    out: list[PreflightViolation],
) -> None:
    """Flag ``from .models import VAE`` (relative) or ``from models import VAE``
    (same-dir absolute) when the symbol doesn't exist in the target file.

    Only fires when:
      * The import is relative (``from .module import symbol``) OR
        the module stem matches a .py file in code_dir.
      * The imported symbol is not in the target file's top-level names.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        is_relative = (node.level or 0) > 0
        # Strip leading dots for relative imports.
        module_stem = module.lstrip(".") if module else ""

        if is_relative:
            # Relative import: ``from . import foo`` has module="" and level=1;
            # ``from .models import VAE`` has module="models" and level=1.
            if not module_stem:
                continue  # ``from . import *`` — can't check without package context
            target_file = code_dir / f"{module_stem}.py"
            if not target_file.is_file():
                continue  # file doesn't exist — let the import error speak for itself
        else:
            # Absolute import: only check if the module stem is a local .py file.
            if not module_stem:
                continue
            target_file = code_dir / f"{module_stem}.py"
            if not target_file.is_file():
                continue  # external library — skip

        # Get the exported names from the target file.
        local_names = _collect_local_py_names(code_dir, module_stem)
        if not local_names:
            continue  # can't parse target → conservative, no violation

        lineno = getattr(node, "lineno", 0)
        for alias in node.names:
            if alias.name == "*":
                continue  # star import — can't check
            if alias.name in local_names:
                continue
            out.append(PreflightViolation(
                file=path.name,
                line=lineno,
                class_name=None,
                missing_attr=alias.name,
                suggested_fix=(
                    f"Define `{alias.name}` in `{module_stem}.py` or fix the "
                    f"import statement. Available names in `{module_stem}.py`: "
                    f"{sorted(local_names)[:8]}."
                ),
                severity="hard",
                detail=(
                    f"{path.name}:{lineno}: `from {'.' if is_relative else ''}"
                    f"{module_stem} import {alias.name}` — "
                    f"`{alias.name}` is not defined in `{module_stem}.py`."
                ),
            ))


# ---------------------------------------------------------------------------
# SDAR teacher/student env interface contract
# ---------------------------------------------------------------------------
#
# The 2026-05-31 failure (`prj_09047604e591d969`): every `alfworld` cell of the
# SDAR matrix died mid-grid with::
#
#     AttributeError: 'ALFWorldEnv' object has no attribute 'build_student_prompt'
#
# The agent's trainer called ``env.build_student_prompt(...)`` on an ``ALFWorldEnv``
# that neither defined the method nor subclassed ``BaseEnv`` (the ABC in
# ``sdar_env_base.py`` that would have turned this into a loud construction-time
# ``TypeError``). The bug slipped past pre-flight because the SDAR envs live at
# ``code/sdar/envs/*.py`` — two levels deep — and the other checks only glob one
# level (``code_dir/*.py``). This check walks RECURSIVELY.
#
# Self-scoping (conservative — avoid false positives on unrelated ``*Env`` classes
# in non-SDAR papers): the contract is only "in play" when the code actually uses
# the teacher/student surface (imports ``sdar_env_base`` / names ``BaseEnv`` / calls
# or accesses ``build_student_prompt`` / ``build_teacher_prompt`` somewhere). When no
# such signal is present, the check returns immediately and flags nothing.

_REQUIRED_ENV_METHODS = ("build_student_prompt", "build_teacher_prompt")
_BASE_ENV_NAME = "BaseEnv"
_BASE_ENV_MODULE = "sdar_env_base"


def _file_uses_env_contract(tree: ast.AST) -> bool:
    """Return True if this file imports/names the SDAR env contract surface.

    Signals (any one is enough):
      * ``import sdar_env_base`` / ``from sdar_env_base import ...``
      * the bare name ``BaseEnv`` referenced anywhere (import, base class, …)
      * a reference (call or attribute access) to ``build_student_prompt`` /
        ``build_teacher_prompt``
    """
    for node in ast.walk(tree):
        # ``import sdar_env_base`` (possibly dotted / aliased).
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == _BASE_ENV_MODULE:
                    return True
        # ``from sdar_env_base import BaseEnv`` (or relative ``from .sdar_env_base``).
        if isinstance(node, ast.ImportFrom):
            module_stem = (node.module or "").lstrip(".")
            if module_stem.split(".")[0] == _BASE_ENV_MODULE:
                return True
            for alias in node.names:
                if alias.name == _BASE_ENV_NAME:
                    return True
        # Bare name ``BaseEnv`` referenced anywhere.
        if isinstance(node, ast.Name) and node.id == _BASE_ENV_NAME:
            return True
        # Attribute access of the form ``module.BaseEnv``.
        if isinstance(node, ast.Attribute) and node.attr == _BASE_ENV_NAME:
            return True
        # Reference to one of the required methods — either ``obj.build_student_prompt``
        # (attribute access / call) or the bare name (e.g. passed around).
        if isinstance(node, ast.Attribute) and node.attr in _REQUIRED_ENV_METHODS:
            return True
        if isinstance(node, ast.Name) and node.id in _REQUIRED_ENV_METHODS:
            return True
    return False


def _base_is_base_env(base: ast.expr) -> bool:
    """Return True if an ``ast`` base-class expression names ``BaseEnv``.

    Matches both ``class Foo(BaseEnv):`` (``ast.Name``) and
    ``class Foo(mod.BaseEnv):`` (``ast.Attribute``).
    """
    if isinstance(base, ast.Name):
        return base.id == _BASE_ENV_NAME
    if isinstance(base, ast.Attribute):
        return base.attr == _BASE_ENV_NAME
    return False


def _check_env_interface_contract(
    code_dir: Path,
    out: list[PreflightViolation],
) -> None:
    """Catch a ``*Env`` that will ``AttributeError`` on the SDAR teacher/student
    contract BEFORE the training grid runs.

    Walks ``code_dir`` RECURSIVELY (the existing per-file checks only glob one
    level; the SDAR envs live at ``code/sdar/envs/*.py``).

    Step 1 — self-scope: if no file in the tree uses the env contract (imports
    ``sdar_env_base`` / names ``BaseEnv`` / references ``build_student_prompt`` /
    ``build_teacher_prompt``), return immediately and flag nothing. This keeps
    unrelated ``*Env`` classes in non-SDAR papers from being flagged.

    Step 2 — flag: for every ``ClassDef`` whose name ends in ``"Env"`` (and is not
    ``BaseEnv`` itself) that defines NEITHER required method AND does not directly
    subclass ``BaseEnv``, emit a ``hard`` violation. A class that subclasses
    ``BaseEnv`` is left alone: the ABC enforces the methods at construction (a
    loud, named ``TypeError``), so the AST backstop only needs to catch the
    non-subclassing escape hatch.

    Fail-soft per file (try/except) — never raises from pre-flight.
    """
    py_files: list[Path] = sorted(
        p for p in code_dir.rglob("*.py") if p.is_file()
    )
    if not py_files:
        return

    # Parse every file once (fail-soft per file), keeping the tree for re-use.
    parsed: list[tuple[Path, ast.AST]] = []
    for path in py_files:
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            if not source.strip():
                continue
            tree = ast.parse(source, filename=str(path))
        except Exception:  # noqa: BLE001 — syntax errors are caught elsewhere
            continue
        parsed.append((path, tree))

    if not parsed:
        return

    # Step 1: self-scope — is the teacher/student contract actually in play?
    if not any(_file_uses_env_contract(tree) for _path, tree in parsed):
        return  # this paper does not use the contract — skip entirely.

    # Step 2: flag non-subclassing, non-implementing ``*Env`` classes.
    for path, tree in parsed:
        try:
            rel = path.relative_to(code_dir).as_posix()
        except ValueError:
            rel = path.name
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            name = node.name
            if not name.endswith("Env") or name == _BASE_ENV_NAME:
                continue
            # Does it directly subclass BaseEnv? If so, the ABC enforces the
            # methods at construction — don't flag.
            if any(_base_is_base_env(base) for base in node.bases):
                continue
            # Does it define the methods itself (incl. same-file inheritance)?
            members = _collect_all_class_members_with_inheritance(tree, name)
            if any(m in members for m in _REQUIRED_ENV_METHODS):
                continue
            lineno = getattr(node, "lineno", 0)
            out.append(PreflightViolation(
                file=rel,
                line=lineno,
                class_name=name,
                missing_attr="build_student_prompt/build_teacher_prompt",
                suggested_fix=(
                    f"Make `{name}` subclass BaseEnv from sdar_env_base and "
                    f"implement build_student_prompt + build_teacher_prompt: "
                    f"`from sdar_env_base import BaseEnv` then "
                    f"`class {name}(BaseEnv): ...`. The harness copies "
                    f"sdar_env_base.py into code/."
                ),
                severity="hard",
                detail=(
                    f"{rel}:{lineno}: `{name}` neither subclasses BaseEnv nor "
                    f"defines build_student_prompt/build_teacher_prompt — the SDAR "
                    f"trainer calls build_student_prompt/build_teacher_prompt on "
                    f"every env, so this AttributeErrors mid-grid."
                ),
            ))


# ---------------------------------------------------------------------------
# Syntax error (surfaced via parse)
# ---------------------------------------------------------------------------


def _parse_with_syntax_check(
    path: Path,
    out: list[PreflightViolation],
) -> ast.AST | None:
    """Parse ``path``, appending a violation on SyntaxError. Returns None on error."""
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if not source.strip():
        return None
    try:
        return ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        out.append(PreflightViolation(
            file=path.name,
            line=exc.lineno or 0,
            class_name=None,
            missing_attr=None,
            suggested_fix=(
                f"Fix the syntax error in `{path.name}` at line {exc.lineno}: "
                f"{exc.msg}"
            ),
            severity="hard",
            detail=(
                f"{path.name}:{exc.lineno}: SyntaxError — {exc.msg}. "
                f"Python cannot execute a file that doesn't parse."
            ),
        ))
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan_code_dir(code_dir: Path) -> list[PreflightViolation]:
    """Run scoped AST pre-flight checks on all .py files under ``code_dir``.

    Parameters
    ----------
    code_dir : Path
        Directory containing agent-written code (``train.py``, ``models.py``, etc.).

    Returns
    -------
    list[PreflightViolation]
        Structured violations. Empty list means no issues detected (or that every
        ambiguous case was conservatively suppressed). Fail-soft: any internal
        exception returns whatever violations were collected so far.

    Notes
    -----
    This function is intentionally narrow in scope. It will NOT catch:
      * Cross-file attribute access on classes imported from external libraries.
      * Dynamic attribute addition via monkey-patching at module level.
      * Missing attributes added via ``setattr`` (suppressed by design).
    """
    code_dir = Path(code_dir)
    if not code_dir.is_dir():
        return []

    violations: list[PreflightViolation] = []

    # Collect all .py files — we inspect agent-written code, not vendored libraries.
    # Walk one level deep: train.py, models.py, utils.py etc.
    #
    # NOTE: an empty one-level glob is NOT an early return. The SDAR envs live two
    # levels deep (``code/sdar/envs/*.py``) with nothing at the top level, so a
    # ``return []`` here would shadow the recursive env-contract check below — the
    # exact blind spot that let the 2026-05-31 AttributeError slip past pre-flight.
    py_files: list[Path] = sorted(
        p for p in code_dir.glob("*.py") if p.is_file()
    )

    trees: dict[Path, ast.AST] = {}

    # Parse phase — collect syntax errors first.
    try:
        for path in py_files:
            tree = _parse_with_syntax_check(path, violations)
            if tree is not None:
                trees[path] = tree
    except Exception:  # noqa: BLE001 — never raise from pre-flight
        return violations

    # Attribute-access and local-import checks — per file.
    #
    # NOTE: _check_undefined_names is intentionally NOT called here.
    # The false-positive rate on real agent code is prohibitively high because:
    #   (a) Agent code routinely uses `from datasets import load_dataset`,
    #       `import torch`, etc. — these are valid once imported but the bare-name
    #       check can't distinguish "imported via a pattern we don't recognise"
    #       from "genuinely not in scope".
    #   (b) The spec's conservative rule ("not 100% sure → no block") applies.
    #       Missing-attribute access on local classes is the high-confidence case;
    #       undefined bare names have too many legitimate import patterns to be safe.
    #
    # TODO(PR-γ-followup): re-enable once we have full import-resolution tracking.
    for path, tree in trees.items():
        try:
            _check_missing_attr_access(tree, path, violations)
        except Exception:  # noqa: BLE001
            logger.debug("preflight_ast: _check_missing_attr_access failed on %s", path.name)

        try:
            _check_local_import_from(tree, path, code_dir, violations)
        except Exception:  # noqa: BLE001
            logger.debug("preflight_ast: _check_local_import_from failed on %s", path.name)

    # SDAR teacher/student env interface contract — needs its OWN recursive walk
    # (``rglob``), independent of ``py_files`` above which only globs one level.
    # The SDAR envs live at ``code/sdar/envs/*.py``, which is how the 2026-05-31
    # AttributeError slipped past pre-flight.
    try:
        _check_env_interface_contract(code_dir, violations)
    except Exception:  # noqa: BLE001 — never raise from pre-flight
        logger.debug("preflight_ast: _check_env_interface_contract failed")

    return violations


__all__ = [
    "PreflightViolation",
    "scan_code_dir",
]
