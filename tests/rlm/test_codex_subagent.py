from __future__ import annotations

import builtins
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import backend.agents.rlm.codex_subagent as codex_mod
from backend.agents.rlm.codex_subagent import CodexSubagentResult, run_codex_subagent
from backend.agents.rlm.primitives import codex_repair
from backend.config import get_settings


def _enable_codex(monkeypatch: pytest.MonkeyPatch, *, max_calls: str = "3") -> None:
    monkeypatch.setenv("REPROLAB_CODEX_SUBAGENT", "1")
    monkeypatch.setenv("REPROLAB_CODEX_TIMEOUT_S", "9")
    monkeypatch.setenv("REPROLAB_CODEX_MAX_CALLS_PER_RUN", max_calls)
    monkeypatch.setenv("REPROLAB_CODEX_MAX_OUTPUT_CHARS", "120")
    monkeypatch.setenv("REPROLAB_CODEX_PROFILE", "reprolab-readwrite")
    monkeypatch.setenv(
        "REPROLAB_CODEX_ALLOWED_TASKS",
        "implementation_repair,test_debugging,dockerfile_repair,requirements_repair",
    )
    get_settings(_force_reload=True)


def test_flag_off_returns_disabled(make_context, tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_CODEX_SUBAGENT", "0")
    get_settings(_force_reload=True)

    result = codex_repair(
        "implementation_repair",
        "fix train.py syntax",
        "python -m py_compile code/train.py",
        ["code/train.py"],
        failure_class="syntax_error",
        ctx=make_context(tmp_path),
    )

    assert result["ok"] is False
    assert result["disabled"] is True
    assert result["error_type"] == "disabled"


def test_codex_unavailable_returns_typed_unavailable(tmp_path, monkeypatch):
    monkeypatch.delenv("REPROLAB_CODEX_CLI_PATH", raising=False)
    monkeypatch.setattr(codex_mod.shutil, "which", lambda name: None)

    result = run_codex_subagent(
        "fix it",
        tmp_path,
        timeout_s=1,
        profile=None,
        readonly=False,
    )

    assert result.ok is False
    assert result.error_type == "unavailable"
    assert result.exit_code is None


def test_timeout_kills_process_and_returns_timed_out(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_CODEX_CLI_PATH", "/usr/bin/codex")

    def fake_run(args, **kwargs):
        if args[1:3] == ["login", "status"]:
            return SimpleNamespace(returncode=0, stdout="logged in", stderr="")
        raise subprocess.TimeoutExpired(args, kwargs["timeout"], output="out" * 100)

    monkeypatch.setattr(codex_mod.subprocess, "run", fake_run)
    result = run_codex_subagent("fix it", tmp_path, timeout_s=1, profile=None, readonly=False)

    assert result.ok is False
    assert result.timed_out is True
    assert result.error_type == "timeout"
    assert result.exit_code is None


def test_output_truncation(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_CODEX_CLI_PATH", "/usr/bin/codex")

    def fake_run(args, **kwargs):
        if args[1:3] == ["login", "status"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="x" * 50, stderr="y" * 40)

    monkeypatch.setattr(codex_mod.subprocess, "run", fake_run)
    result = run_codex_subagent(
        "fix it",
        tmp_path,
        timeout_s=5,
        profile=None,
        readonly=False,
        task_type="traceback_explanation",
        max_output_chars=12,
    )

    assert result.ok is True
    assert result.stdout_tail == "x" * 12
    assert result.stderr_tail == "y" * 12
    assert result.stdout_tail_truncated is True
    assert result.stderr_tail_truncated is True


def test_allowed_task_passes(make_context, tmp_path, monkeypatch):
    _enable_codex(monkeypatch)
    captured = {}

    def fake_run_codex_subagent(**kwargs):
        captured.update(kwargs)
        return CodexSubagentResult(
            ok=True,
            timed_out=False,
            exit_code=0,
            stdout_tail="changed code/train.py",
            stderr_tail="",
            changed_files=["code/train.py"],
            duration_s=0.2,
            error_type=None,
        )

    monkeypatch.setattr(codex_mod, "run_codex_subagent", fake_run_codex_subagent)
    ctx = make_context(tmp_path)

    result = codex_repair(
        "implementation_repair",
        "Fix the SyntaxError in code/train.py.",
        "python -m py_compile code/train.py",
        ["code/train.py"],
        repair_context={"success": False, "failure_class": "syntax_error"},
        ctx=ctx,
    )

    assert result["ok"] is True
    assert captured["task_type"] == "implementation_repair"
    prompt = captured["prompt"]
    assert "Exact task:" in prompt
    assert "Allowed files or workspace scope:" in prompt
    assert "Test command to run:" in prompt
    assert "Max time budget: 9 seconds." in prompt
    assert "Do not print secrets" in prompt
    assert "Stop after the targeted fix" in prompt


def test_disallowed_task_rejected(make_context, tmp_path, monkeypatch):
    _enable_codex(monkeypatch)

    result = codex_repair(
        "paper_summary",
        "summarize the paper",
        "true",
        [],
        failure_class="syntax_error",
        ctx=make_context(tmp_path),
    )

    assert result["ok"] is False
    assert result["error_type"] == "task_type_rejected"


def test_max_calls_per_run_enforced(make_context, tmp_path, monkeypatch):
    _enable_codex(monkeypatch, max_calls="1")
    monkeypatch.setattr(
        codex_mod,
        "run_codex_subagent",
        lambda **kwargs: CodexSubagentResult(
            ok=True,
            timed_out=False,
            exit_code=0,
            stdout_tail="",
            stderr_tail="",
            changed_files=[],
            duration_s=0.1,
            error_type=None,
        ),
    )
    ctx = make_context(tmp_path)

    first = codex_repair(
        "implementation_repair",
        "fix syntax",
        "python -m py_compile code/train.py",
        ["code/train.py"],
        failure_class="syntax_error",
        ctx=ctx,
    )
    second = codex_repair(
        "implementation_repair",
        "fix syntax",
        "python -m py_compile code/train.py",
        ["code/train.py"],
        failure_class="syntax_error",
        ctx=ctx,
    )

    assert first["ok"] is True
    assert second["ok"] is False
    assert second["error_type"] == "max_calls_exceeded"


def test_no_auth_file_contents_are_read_or_logged(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_CODEX_CLI_PATH", "/usr/bin/codex")
    auth_path = tmp_path / ".codex" / "auth.json"
    auth_path.parent.mkdir()
    auth_path.write_text("SECRET_AUTH_TOKEN", encoding="utf-8")
    opened_auth = {"value": False}
    real_open = builtins.open

    def guarded_open(file, *args, **kwargs):
        if Path(file) == auth_path:
            opened_auth["value"] = True
            raise AssertionError("auth file must not be read")
        return real_open(file, *args, **kwargs)

    def fake_run(args, **kwargs):
        if args[1:3] == ["login", "status"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(codex_mod.subprocess, "run", fake_run)

    result = run_codex_subagent(
        "explain it",
        tmp_path,
        timeout_s=3,
        profile=None,
        readonly=False,
        task_type="traceback_explanation",
    )

    assert result.ok is True
    assert opened_auth["value"] is False
    assert "SECRET_AUTH_TOKEN" not in result.stdout_tail
    assert "SECRET_AUTH_TOKEN" not in result.stderr_tail


def test_changed_files_parsed_if_possible(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_CODEX_CLI_PATH", "/usr/bin/codex")
    target = tmp_path / "bug.py"
    target.write_text("x =\n", encoding="utf-8")

    def fake_run(args, **kwargs):
        if args[1:3] == ["login", "status"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        target.write_text("x = 1\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="fixed", stderr="")

    monkeypatch.setattr(codex_mod.subprocess, "run", fake_run)
    result = run_codex_subagent("fix bug.py", tmp_path, timeout_s=3, profile=None, readonly=False)

    assert result.ok is True
    assert result.changed_files == ["bug.py"]


def test_subprocess_failure_fail_softs(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_CODEX_CLI_PATH", "/usr/bin/codex")

    def fake_run(args, **kwargs):
        if args[1:3] == ["login", "status"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=2, stdout="", stderr="boom")

    monkeypatch.setattr(codex_mod.subprocess, "run", fake_run)
    result = run_codex_subagent("fix it", tmp_path, timeout_s=3, profile=None, readonly=False)

    assert result.ok is False
    assert result.error_type == "subprocess_failed"
    assert result.exit_code == 2
    assert result.stderr_tail == "boom"


def test_event_emission_does_not_crash_run(tmp_path, monkeypatch):
    monkeypatch.setenv("REPROLAB_CODEX_CLI_PATH", "/usr/bin/codex")

    def fake_run(args, **kwargs):
        if args[1:3] == ["login", "status"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(codex_mod.subprocess, "run", fake_run)

    result = run_codex_subagent(
        "fix it",
        tmp_path,
        timeout_s=3,
        profile=None,
        readonly=False,
        task_type="traceback_explanation",
        event_sink=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("emit failed")),
    )

    assert result.ok is True
