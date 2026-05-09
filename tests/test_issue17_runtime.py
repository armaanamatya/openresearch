"""Tests for runtime sandbox contracts and LocalDockerBackend."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import anyio

from backend.services.runtime import (
    CommandExecuted,
    CreateSandbox,
    DestroySandbox,
    ExecuteCommand,
    ExecResult,
    LocalDockerBackend,
    RuntimeAppService,
    RuntimeBackend,
    SandboxAggregate,
    SandboxConfig,
    SandboxState,
)


def test_runtime_contract_exports_expected_types(tmp_path: Path) -> None:
    config = SandboxConfig(
        project_id="prj_test",
        run_id="baseline",
        image="python:3.11-slim",
        project_root=tmp_path,
    )
    assert isinstance(LocalDockerBackend(client=FakeDockerClient()), RuntimeBackend)
    assert config.resolved_artifact_root() == tmp_path / "artifacts"


def test_sandbox_aggregate_tracks_lifecycle(tmp_path: Path) -> None:
    config = SandboxConfig(
        project_id="prj_test",
        run_id="baseline",
        image="python:3.11-slim",
        project_root=tmp_path,
    )
    agg = SandboxAggregate.empty("prj_test", "baseline")
    requested = list(agg.handle_request(config))
    agg.apply_all(requested)
    assert agg.state == SandboxState.REQUESTED

    created = list(agg.handle_created("ctr_1", "python:3.11-slim"))
    agg.apply_all(created)
    assert agg.state == SandboxState.CREATED
    assert agg.sandbox_id == "ctr_1"

    result = _exec_result("python --version", exit_code=0)
    command_events = list(agg.handle_command_result(result))
    assert isinstance(command_events[0], CommandExecuted)
    agg.apply_all(command_events)
    assert agg.state == SandboxState.RUNNING


def test_local_docker_backend_delegates_lifecycle(tmp_path: Path) -> None:
    async def scenario() -> None:
        client = FakeDockerClient()
        backend = LocalDockerBackend(client=client)
        config = SandboxConfig(
            project_id="prj_test",
            run_id="baseline",
            image="python:3.11-slim",
            project_root=tmp_path,
            artifact_root=tmp_path / "baseline",
        )

        sandbox = await backend.create_sandbox(config)
        assert sandbox.sandbox_id == "ctr_1"
        assert client.containers.created_kwargs["volumes"][str(tmp_path)]["bind"] == "/work"
        assert client.containers.created_kwargs["network_mode"] == "none"

        result = await backend.exec(sandbox, "python train.py", timeout=30)
        assert result.succeeded
        assert result.stdout == "ok"
        assert client.containers.container.exec_calls == ["python train.py"]

        await backend.copy_in(sandbox, "/work/config.json", b"{}")
        assert client.containers.container.put_paths == ["/work"]

        copied = await backend.copy_out(sandbox, "/artifacts/metrics.json")
        assert copied == b'{"mean_reward": 500}'

        await backend.destroy(sandbox)
        assert client.containers.container.stopped
        assert client.containers.container.removed

    anyio.run(scenario)


def test_runtime_service_records_events_without_store(tmp_path: Path) -> None:
    async def scenario() -> None:
        backend = LocalDockerBackend(client=FakeDockerClient())
        service = RuntimeAppService(backend)
        config = SandboxConfig(
            project_id="prj_test",
            run_id="baseline",
            image="python:3.11-slim",
            project_root=tmp_path,
        )
        sandbox = await service.create_sandbox(CreateSandbox(config=config))
        result = await service.execute(
            ExecuteCommand(sandbox=sandbox, command="python train.py", timeout=30)
        )
        await service.destroy(DestroySandbox(sandbox=sandbox))
        assert result.succeeded

    anyio.run(scenario)


def _exec_result(command: str, exit_code: int) -> ExecResult:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    return ExecResult(
        command=command,
        exit_code=exit_code,
        stdout="ok" if exit_code == 0 else "",
        stderr="" if exit_code == 0 else "failed",
        started_at=now,
        finished_at=now,
        duration_seconds=0.0,
    )


class FakeImages:
    def __init__(self) -> None:
        self.build_calls: list[dict] = []

    def build(self, **kwargs):
        self.build_calls.append(kwargs)
        return object(), []


class FakeExecRaw:
    exit_code = 0
    output = (b"ok", b"")


class FakeContainer:
    id = "ctr_1"

    def __init__(self) -> None:
        self.exec_calls: list[str] = []
        self.put_paths: list[str] = []
        self.stopped = False
        self.removed = False

    def exec_run(self, command, **_kwargs):
        self.exec_calls.append(command)
        return FakeExecRaw()

    def put_archive(self, path, data):
        assert data
        self.put_paths.append(path)
        return True

    def get_archive(self, _path):
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w") as tar:
            payload = b'{"mean_reward": 500}'
            info = tarfile.TarInfo(name="metrics.json")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        return [data.getvalue()], {}

    def stop(self, **_kwargs):
        self.stopped = True

    def remove(self, **_kwargs):
        self.removed = True


class FakeContainers:
    def __init__(self) -> None:
        self.container = FakeContainer()
        self.created_kwargs: dict = {}

    def run(self, image, **kwargs):
        assert image == "python:3.11-slim"
        self.created_kwargs = kwargs
        return self.container

    def get(self, sandbox_id):
        assert sandbox_id == "ctr_1"
        return self.container


class FakeDockerClient:
    def __init__(self) -> None:
        self.containers = FakeContainers()
        self.images = FakeImages()
