"""Change #5 — EXECUTE mode owns its dependencies (OPENRESEARCH_EXECUTE_OWNS_DEPS).

Default-OFF / byte-identical: unset AND repro mode != execute => the helper
returns False, so the harness's local cu121 pip bootstrap in
`_execute_in_sandbox` fires exactly as before. Only in repo-first EXECUTE mode
(mode == "execute" in rlm_state/repo_spec.json) — or an explicit override —
does the authors' conda env take ownership of deps, in which case the
bootstrap block is skipped entirely at the gate site
(`primitives.py::_execute_in_sandbox`, ~line 3978:
``if "local" in _mode_str and requirements_path.exists() and not
_execute_owns_deps(code_path):``). Exercising the full async sandbox coroutine
is out of scope for this pure-unit test; the gate-site comment/diff is the
verification for that half.
"""
from __future__ import annotations

import json

import pytest

from backend.agents.rlm.primitives import _execute_owns_deps


def _write_repo_spec(tmp_path, mode: str | None) -> str:
    """Lay out <tmp_path>/code + <tmp_path>/rlm_state/repo_spec.json and return
    code_path (the string the real caller passes)."""
    project_dir = tmp_path
    (project_dir / "rlm_state").mkdir(parents=True, exist_ok=True)
    code_dir = project_dir / "code"
    code_dir.mkdir(exist_ok=True)
    if mode is not None:
        (project_dir / "rlm_state" / "repo_spec.json").write_text(
            json.dumps({"mode": mode}), encoding="utf-8"
        )
    return str(code_dir)


def test_execute_mode_defaults_to_owns_deps(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_EXECUTE_OWNS_DEPS", raising=False)
    code_path = _write_repo_spec(tmp_path, "execute")
    assert _execute_owns_deps(code_path) is True


def test_adapt_mode_defaults_to_false(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_EXECUTE_OWNS_DEPS", raising=False)
    code_path = _write_repo_spec(tmp_path, "adapt")
    assert _execute_owns_deps(code_path) is False


def test_missing_repo_spec_defaults_to_false(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_EXECUTE_OWNS_DEPS", raising=False)
    code_path = _write_repo_spec(tmp_path, None)
    assert _execute_owns_deps(code_path) is False


@pytest.mark.parametrize("truthy", ["1", "true", "yes", "on", "True", "ON"])
def test_explicit_true_overrides_adapt_mode(tmp_path, monkeypatch, truthy):
    monkeypatch.setenv("OPENRESEARCH_EXECUTE_OWNS_DEPS", truthy)
    code_path = _write_repo_spec(tmp_path, "adapt")
    assert _execute_owns_deps(code_path) is True


@pytest.mark.parametrize("falsy", ["0", "false", "no", "off", "False", "OFF"])
def test_explicit_false_overrides_execute_mode(tmp_path, monkeypatch, falsy):
    monkeypatch.setenv("OPENRESEARCH_EXECUTE_OWNS_DEPS", falsy)
    code_path = _write_repo_spec(tmp_path, "execute")
    assert _execute_owns_deps(code_path) is False


def test_nonexistent_code_path_fails_soft_to_false(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_EXECUTE_OWNS_DEPS", raising=False)
    missing = str(tmp_path / "does-not-exist" / "code")
    assert _execute_owns_deps(missing) is False


def test_malformed_repo_spec_fails_soft_to_false(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENRESEARCH_EXECUTE_OWNS_DEPS", raising=False)
    project_dir = tmp_path
    (project_dir / "rlm_state").mkdir(parents=True, exist_ok=True)
    code_dir = project_dir / "code"
    code_dir.mkdir(exist_ok=True)
    (project_dir / "rlm_state" / "repo_spec.json").write_text("{not json", encoding="utf-8")
    assert _execute_owns_deps(str(code_dir)) is False
