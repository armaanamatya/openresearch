import asyncio
import logging
from datetime import datetime, timezone

import backend.agents.rlm.primitives as primitives
from backend.services.runtime.interface import ExecResult, Sandbox


def _exec_result(command: str) -> ExecResult:
    now = datetime.now(timezone.utc)
    return ExecResult(
        command=command,
        exit_code=0,
        stdout="ok",
        stderr="",
        started_at=now,
        finished_at=now,
        duration_seconds=0.01,
    )


def _install_fake_runtime(monkeypatch, captured: dict) -> None:
    def fake_backend_for(mode, *, run_budget=None, gpu_plan=None):
        captured.setdefault("modes", []).append(mode)

        class _FakeBackend:
            pass

        return _FakeBackend()

    class _FakeService:
        def __init__(self, backend):
            self.backend = backend

        async def create_sandbox(self, cmd):
            captured["image"] = cmd.config.image
            return Sandbox(
                sandbox_id="fake",
                name="fake",
                image=cmd.config.image,
                config=cmd.config,
            )

        async def execute(self, cmd):
            captured.setdefault("commands", []).append(cmd.command)
            return _exec_result(cmd.command)

        async def destroy(self, cmd):
            captured["destroyed"] = True

    monkeypatch.setattr(primitives, "_backend_for_sandbox_mode", fake_backend_for)
    monkeypatch.setattr(primitives, "RuntimeAppService", _FakeService)


def _run_execute(tmp_path, *, env_id: str, sandbox_mode: str, monkeypatch, captured: dict):
    code_dir = tmp_path / f"code-{env_id}-{sandbox_mode}"
    code_dir.mkdir()
    _install_fake_runtime(monkeypatch, captured)
    return asyncio.run(
        primitives._execute_in_sandbox(
            str(code_dir),
            env_id,
            ["echo hi"],
            project_id="proj",
            run_id=f"run-{env_id}-{sandbox_mode}",
            sandbox_mode=sandbox_mode,
        )
    )


def test_env_id_local_with_runpod_mode_uses_runpod_and_warns(tmp_path, monkeypatch, caplog):
    captured: dict = {}
    caplog.set_level(logging.WARNING)

    result = _run_execute(
        tmp_path,
        env_id="local",
        sandbox_mode="runpod",
        monkeypatch=monkeypatch,
        captured=captured,
    )

    assert result["success"] is True
    assert captured["modes"] == ["runpod"]
    assert any("env_id='local' looks like a backend hint" in rec.message for rec in caplog.records)


def test_env_id_runpod_with_local_mode_uses_local_and_warns(tmp_path, monkeypatch, caplog):
    captured: dict = {}
    caplog.set_level(logging.WARNING)

    result = _run_execute(
        tmp_path,
        env_id="runpod",
        sandbox_mode="local",
        monkeypatch=monkeypatch,
        captured=captured,
    )

    assert result["success"] is True
    assert captured["modes"] == ["local"]
    assert any("env_id='runpod' looks like a backend hint" in rec.message for rec in caplog.records)


def test_matching_env_id_and_sandbox_mode_does_not_warn(tmp_path, monkeypatch, caplog):
    captured: dict = {}
    caplog.set_level(logging.WARNING)

    result = _run_execute(
        tmp_path,
        env_id="runpod",
        sandbox_mode="runpod",
        monkeypatch=monkeypatch,
        captured=captured,
    )

    assert result["success"] is True
    assert captured["modes"] == ["runpod"]
    assert not any("looks like a backend hint" in rec.message for rec in caplog.records)
