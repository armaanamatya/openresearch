"""Tests for Issue #26: Experiment Runner Agent."""

from pathlib import Path
import json
from datetime import datetime, timezone

import anyio

from backend.agents.experiment_runner import (
    _normalize_commands,
    run_offline,
    run_offline_failure,
    run_with_local_process,
    run_with_runtime,
)
from backend.agents.schemas import BaselineResult, ExperimentArtifacts
from backend.services.runtime import (
    ExecResult,
    RuntimeBackend,
    RuntimeAppService,
    Sandbox,
    SandboxConfig,
)


def _baseline():
    return BaselineResult(
        mode="implement_from_paper",
        code_path="/code",
        dockerfile_path="/code/Dockerfile",
        commands_to_run=["python train.py"],
        assumptions_applied=["A001"],
    )


class TestRunOffline:
    def test_produces_artifacts(self, tmp_path: Path):
        result = run_offline("prj_ppo", tmp_path, _baseline())
        assert isinstance(result, ExperimentArtifacts)
        assert result.success is True

    def test_writes_metrics_json(self, tmp_path: Path):
        run_offline("prj_ppo", tmp_path, _baseline())
        metrics_path = tmp_path / "prj_ppo" / "baseline" / "metrics.json"
        assert metrics_path.exists()
        metrics = json.loads(metrics_path.read_text())
        assert "mean_reward" in metrics

    def test_writes_log(self, tmp_path: Path):
        run_offline("prj_ppo", tmp_path, _baseline())
        log_path = tmp_path / "prj_ppo" / "baseline" / "logs" / "run.log"
        assert log_path.exists()
        assert len(log_path.read_text()) > 0

    def test_writes_commands_log(self, tmp_path: Path):
        run_offline("prj_ppo", tmp_path, _baseline())
        cmd_path = tmp_path / "prj_ppo" / "baseline" / "commands.log"
        assert cmd_path.exists()
        payload = json.loads(cmd_path.read_text().splitlines()[0])
        assert payload["command"] == "python train.py"
        assert payload["status"] == "succeeded"

    def test_writes_provenance(self, tmp_path: Path):
        run_offline("prj_ppo", tmp_path, _baseline())
        prov_path = tmp_path / "prj_ppo" / "baseline" / "provenance.json"
        assert prov_path.exists()
        prov = json.loads(prov_path.read_text())
        assert prov["project_id"] == "prj_ppo"

    def test_writes_plot(self, tmp_path: Path):
        run_offline("prj_ppo", tmp_path, _baseline())
        plot_path = tmp_path / "prj_ppo" / "baseline" / "plots" / "reward_curve.png"
        assert plot_path.exists()
        # Check it's a valid PNG (starts with PNG signature)
        assert plot_path.read_bytes()[:4] == b"\x89PNG"

    def test_custom_metrics(self, tmp_path: Path):
        result = run_offline(
            "prj_ppo", tmp_path, _baseline(),
            simulate_metrics={"mean_reward": 500.0, "custom": True},
        )
        assert result.metrics["mean_reward"] == 500.0
        assert result.metrics["custom"] is True

    def test_artifact_paths_are_valid(self, tmp_path: Path):
        result = run_offline("prj_ppo", tmp_path, _baseline())
        assert Path(result.log_path).exists()
        assert Path(result.commands_log_path).exists()
        assert Path(result.provenance_path).exists()


class TestRunOfflineFailure:
    def test_failure_mode(self, tmp_path: Path):
        result = run_offline_failure("prj_fail", tmp_path, _baseline())
        assert result.success is False
        assert result.error_message != ""

    def test_failure_still_writes_log(self, tmp_path: Path):
        result = run_offline_failure("prj_fail", tmp_path, _baseline())
        assert Path(result.log_path).exists()
        log_content = Path(result.log_path).read_text()
        assert "ERROR" in log_content

    def test_failure_has_no_metrics(self, tmp_path: Path):
        result = run_offline_failure("prj_fail", tmp_path, _baseline())
        assert result.metrics == {}


class TestRunWithRuntime:
    def test_executes_commands_and_writes_provenance(self, tmp_path: Path):
        async def scenario() -> ExperimentArtifacts:
            code_dir = tmp_path / "code"
            code_dir.mkdir()
            (code_dir / "Dockerfile").write_text("FROM python:3.11-slim\n")
            baseline = BaselineResult(
                mode="implement_from_paper",
                code_path=str(code_dir),
                dockerfile_path=str(code_dir / "Dockerfile"),
                commands_to_run=["python train.py"],
                assumptions_applied=["A001"],
            )
            runtime = RuntimeAppService(FakeRuntimeBackend())
            return await run_with_runtime(
                "prj_runtime",
                tmp_path,
                baseline,
                runtime=runtime,
            )

        result = anyio.run(scenario)
        assert result.success is True
        assert result.metrics["mean_reward"] == 501.0

        run_dir = tmp_path / "prj_runtime" / "baseline"
        command_payload = json.loads((run_dir / "commands.log").read_text().splitlines()[0])
        assert command_payload["command"] == "python train.py"
        assert command_payload["exit_code"] == 0

        provenance = json.loads((run_dir / "provenance.json").read_text())
        assert provenance["sandbox_id"] == "fake_sandbox"
        assert provenance["success"] is True
        assert provenance["runtime_kind"] == "docker"
        assert provenance["sandbox_config"]["network_disabled"] is True

    def test_resolves_generated_code_dir_when_agent_returns_container_path(self, tmp_path: Path):
        async def scenario() -> tuple[ExperimentArtifacts, FakeRuntimeBackend]:
            code_dir = tmp_path / "prj_runtime" / "code"
            code_dir.mkdir(parents=True)
            (code_dir / "Dockerfile").write_text("FROM python:3.11-slim\n")
            backend = FakeRuntimeBackend()
            baseline = BaselineResult(
                mode="implement_from_paper",
                code_path="/code",
                dockerfile_path="/code/Dockerfile",
                commands_to_run=["python train.py"],
                assumptions_applied=["A001"],
            )
            artifacts = await run_with_runtime(
                "prj_runtime",
                tmp_path,
                baseline,
                runtime=RuntimeAppService(backend),
                network_disabled=False,
                memory_limit="2g",
                cpus=1.5,
                platform="linux/amd64",
            )
            return artifacts, backend

        result, backend = anyio.run(scenario)
        assert result.success is True
        assert backend.config is not None
        assert backend.config.project_root == (tmp_path / "prj_runtime" / "code").resolve()
        assert backend.config.network_disabled is False
        assert backend.config.memory_limit == "2g"
        assert backend.config.cpus == 1.5
        assert backend.config.platform == "linux/amd64"

    def test_local_process_mode_runs_without_dockerfile(self, tmp_path: Path):
        async def scenario() -> ExperimentArtifacts:
            code_dir = tmp_path / "prj_local" / "code"
            code_dir.mkdir(parents=True)
            script = (
                "import json, os, pathlib\n"
                "out = pathlib.Path(os.environ['OUTPUT_DIR'])\n"
                "out.mkdir(parents=True, exist_ok=True)\n"
                "(out / 'metrics.json').write_text(json.dumps({'mean_reward': 12.5}))\n"
                "print('local done')\n"
            )
            (code_dir / "train.py").write_text(script)
            baseline = BaselineResult(
                mode="implement_from_paper",
                code_path=str(code_dir),
                dockerfile_path="",
                commands_to_run=["python train.py"],
                assumptions_applied=["A001"],
            )
            return await run_with_local_process(
                "prj_local",
                tmp_path,
                baseline,
            )

        result = anyio.run(scenario)
        assert result.success is True
        assert result.metrics["mean_reward"] == 12.5
        provenance = json.loads((tmp_path / "prj_local" / "baseline" / "provenance.json").read_text())
        assert provenance["runtime_kind"] == "local_process"


class FakeRuntimeBackend(RuntimeBackend):
    def __init__(self) -> None:
        self.config: SandboxConfig | None = None

    async def create_sandbox(self, config: SandboxConfig) -> Sandbox:
        self.config = config
        return Sandbox(
            sandbox_id="fake_sandbox",
            name="fake",
            image=config.image,
            config=config,
        )

    async def exec(self, sandbox: Sandbox, command: str, timeout: int) -> ExecResult:
        (sandbox.config.resolved_artifact_root() / "metrics.json").write_text(
            json.dumps({"mean_reward": 501.0, "target_met": True})
        )
        now = datetime.now(timezone.utc)
        return ExecResult(
            command=command,
            exit_code=0,
            stdout="done\n",
            started_at=now,
            finished_at=now,
            duration_seconds=0.0,
        )

    async def copy_out(self, sandbox: Sandbox, path: str) -> bytes:
        return b""

    async def copy_in(self, sandbox: Sandbox, path: str, data: bytes) -> None:
        return None

    async def destroy(self, sandbox: Sandbox) -> None:
        return None


class TestNormalizeCommands:
    """_normalize_commands strips repo-root / absolute prefixes so commands
    resolve inside the docker sandbox, which mounts code_dir at /work (cwd /work)."""

    def test_strips_repo_root_relative_prefix(self, tmp_path: Path):
        code_dir = tmp_path / "runs" / "prj_abc123" / "code"
        out = _normalize_commands(
            ["python runs/prj_abc123/code/train.py --smoke-test"], code_dir
        )
        assert out == ["python train.py --smoke-test"]

    def test_strips_absolute_code_dir_prefix(self, tmp_path: Path):
        code_dir = tmp_path / "runs" / "prj_abc123" / "code"
        code_dir.mkdir(parents=True)
        abs_cmd = f"python {code_dir.resolve()}/train.py"
        assert _normalize_commands([abs_cmd], code_dir) == ["python train.py"]

    def test_leaves_already_relative_commands_unchanged(self, tmp_path: Path):
        code_dir = tmp_path / "runs" / "prj_abc123" / "code"
        cmds = [
            "bash run_smoke.sh",
            "python train.py --env CartPole-v1",
            "python evaluate.py",
        ]
        assert _normalize_commands(cmds, code_dir) == cmds

    def test_strips_prefix_from_every_arg_path(self, tmp_path: Path):
        code_dir = tmp_path / "runs" / "prj_xyz" / "code"
        out = _normalize_commands(
            [
                "python runs/prj_xyz/code/evaluate.py "
                "--config runs/prj_xyz/code/config.json"
            ],
            code_dir,
        )
        assert out == ["python evaluate.py --config config.json"]
