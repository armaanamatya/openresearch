"""Guard tests for the run_experiment environment / observability bugs.

Bug A: _execute_in_sandbox built `logs` from r.stdout only. A failed command
    writes its traceback to stderr, so every failure produced logs="" — the
    failure was undiagnosable on disk and the RLM repair loop got nothing to
    act on (repair_context carried an empty log).

Bug B: run_experiment ran the image built by detect_environment, which runs
    before any baseline code exists and routinely under-specifies dependencies
    (it missed `transformers` for the DPO-toxicity paper). run_experiment now
    rebuilds from ctx.project_dir/Dockerfile — the file the code agent keeps in
    step with the baseline's real imports — so the experiment runs against an
    environment that matches its own code.
"""

from __future__ import annotations

import json

import backend.agents.rlm.primitives as primitives
from backend.agents.rlm.primitives import _combine_command_output, run_experiment


class _FakeResult:
    """Minimal stand-in for a sandbox ExecResult (stdout / stderr only)."""

    def __init__(self, stdout: str = "", stderr: str = "") -> None:
        self.stdout = stdout
        self.stderr = stderr


# --- Bug A: a failed command's stderr must reach `logs` --------------------

def test_combine_command_output_includes_stderr():
    """A traceback written only to stderr must survive into the joined log."""
    results = [_FakeResult(
        stdout="",
        stderr="Traceback (most recent call last):\n"
               "ModuleNotFoundError: No module named 'transformers'",
    )]
    out = _combine_command_output(results)
    assert "ModuleNotFoundError" in out
    assert "transformers" in out


def test_combine_command_output_keeps_both_streams_ordered():
    """stdout and stderr are both kept; stdout precedes stderr for a command."""
    results = [_FakeResult(stdout="step ok\n", stderr="deprecation warning\n")]
    out = _combine_command_output(results)
    assert "step ok" in out and "deprecation warning" in out
    assert out.index("step ok") < out.index("deprecation warning")


def test_combine_command_output_empty_when_no_output():
    """No output on either stream → empty string, no stray separators."""
    assert _combine_command_output([_FakeResult(), _FakeResult()]) == ""


# --- Bug B: run_experiment rebuilds from the project Dockerfile ------------

def _code_dir_with_commands(tmp_path):
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "commands.json").write_text(json.dumps(["python train.py"]))
    return code_dir


def test_run_experiment_rebuilds_image_from_project_dockerfile(
    make_context, tmp_path, monkeypatch
):
    """run_experiment rebuilds from ctx.project_dir/Dockerfile and runs THAT image."""
    ctx = make_context(tmp_path)
    code_dir = _code_dir_with_commands(tmp_path)
    dockerfile_text = "FROM python:3.11-slim\nRUN pip install transformers\n"
    (ctx.project_dir / "Dockerfile").write_text(dockerfile_text)

    seen: dict = {}

    def fake_build_environment(env_spec, *, ctx):
        seen["dockerfile"] = env_spec.get("dockerfile")
        return {"ok": True, "image_tag": "reprolab/test:env-REBUILT",
                "error": "", "attempts": 1}

    async def fake_exec(code_path, env_id, commands, *, project_id, run_id,
                        sandbox_mode=None, run_budget=None):
        seen["env_id"] = env_id
        return {"metrics": {}, "success": True, "logs": ""}

    monkeypatch.setattr(primitives, "build_environment", fake_build_environment)
    monkeypatch.setattr(primitives, "_execute_in_sandbox", fake_exec)

    run_experiment(str(code_dir), "reprolab/test:env-STALE", ctx=ctx)

    assert seen["dockerfile"] == dockerfile_text
    assert seen["env_id"] == "reprolab/test:env-REBUILT", (
        "the experiment must run the rebuilt image, not the stale env_id arg"
    )


def test_run_experiment_falls_back_to_env_id_without_dockerfile(
    make_context, tmp_path, monkeypatch
):
    """No Dockerfile on disk → run_experiment uses the passed env_id (back-compat)."""
    ctx = make_context(tmp_path)
    code_dir = _code_dir_with_commands(tmp_path)

    def fake_build_environment(env_spec, *, ctx):  # pragma: no cover
        raise AssertionError("build_environment called without a Dockerfile on disk")

    seen: dict = {}

    async def fake_exec(code_path, env_id, commands, *, project_id, run_id,
                        sandbox_mode=None, run_budget=None):
        seen["env_id"] = env_id
        return {"metrics": {}, "success": True, "logs": ""}

    monkeypatch.setattr(primitives, "build_environment", fake_build_environment)
    monkeypatch.setattr(primitives, "_execute_in_sandbox", fake_exec)

    run_experiment(str(code_dir), "reprolab/test:env-check", ctx=ctx)
    assert seen["env_id"] == "reprolab/test:env-check"


def test_run_experiment_fails_soft_on_rebuild_failure(
    make_context, tmp_path, monkeypatch
):
    """A failed rebuild fails soft — the experiment never runs a stale image."""
    ctx = make_context(tmp_path)
    code_dir = _code_dir_with_commands(tmp_path)
    (ctx.project_dir / "Dockerfile").write_text("FROM nonexistent-base\n")

    def fake_build_environment(env_spec, *, ctx):
        return {"ok": False, "image_tag": "",
                "error": "docker build failed: unknown base image", "attempts": 2}

    def fake_exec(*a, **k):  # pragma: no cover
        raise AssertionError("_execute_in_sandbox ran despite a failed rebuild")

    monkeypatch.setattr(primitives, "build_environment", fake_build_environment)
    monkeypatch.setattr(primitives, "_execute_in_sandbox", fake_exec)

    result = run_experiment(str(code_dir), "reprolab/test:env-STALE", ctx=ctx)
    assert result["success"] is False
    assert "rebuild" in result["error"]
    assert "unknown base image" in result["error"]
