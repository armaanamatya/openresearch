"""Experiment Runner Agent — executes code and captures artifacts.

Provides:
  - ``run_offline()`` — simulates experiment execution for tests/CI
  - ``run_with_sdk()`` — LLM-driven experiment planning and artifact synthesis
  - ``run_with_runtime()`` — real sandboxed command execution via RuntimeBackend
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from backend.agents.schemas import BaselineResult, ExperimentArtifacts, ReproductionContract
from backend.services.runtime import (
    CommandLogEntry,
    CreateSandbox,
    DestroySandbox,
    ExecuteCommand,
    LocalDockerBackend,
    RuntimeAppService,
    SandboxConfig,
    append_command_log,
    initialize_run_artifacts,
    utc_now_iso,
    write_json,
    write_metrics,
    write_provenance,
)

logger = logging.getLogger(__name__)


def run_offline(
    project_id: str,
    runs_root: Path,
    baseline_result: BaselineResult,
    reproduction_contract: ReproductionContract | None = None,
    *,
    simulate_metrics: dict[str, Any] | None = None,
) -> ExperimentArtifacts:
    """Simulate experiment execution without Docker (for tests/CI).

    Generates realistic artifact directory structure and metrics.
    """
    baseline_dir = initialize_run_artifacts(Path(runs_root) / project_id / "baseline")

    # Default simulation metrics (PPO CartPole-v1 success)
    metrics = simulate_metrics or {
        "mean_reward": 487.3,
        "eval_episodes": 100,
        "total_timesteps": 500000,
        "elapsed_seconds": 245.7,
        "target_met": True,
    }

    # Write metrics.json
    write_metrics(baseline_dir, metrics)

    # Write logs
    log_content = (
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Experiment started\n"
        f"[INFO] Environment: CartPole-v1\n"
        f"[INFO] Total timesteps: {metrics.get('total_timesteps', 500000)}\n"
        f"[INFO] Training complete\n"
        f"[INFO] Mean reward: {metrics.get('mean_reward', 0)}\n"
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Experiment completed\n"
    )
    (baseline_dir / "logs" / "run.log").write_text(log_content)

    # Write structured commands.log (JSONL)
    commands = baseline_result.commands_to_run or ["python train.py"]
    started_at = utc_now_iso()
    for command in commands:
        append_command_log(
            baseline_dir,
            CommandLogEntry(
                command=command,
                phase="offline_simulation",
                status="succeeded",
                started_at=started_at,
                finished_at=utc_now_iso(),
                duration_seconds=0.0,
                exit_code=0,
            ),
        )

    # Write provenance.json
    provenance = {
        "project_id": project_id,
        "code_path": baseline_result.code_path,
        "dockerfile_path": baseline_result.dockerfile_path,
        "commands": commands,
        "mode": baseline_result.mode,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "assumptions_applied": baseline_result.assumptions_applied,
    }
    write_provenance(baseline_dir, provenance)

    # Write a simple plot placeholder
    _write_placeholder_plot(baseline_dir / "plots" / "reward_curve.png")

    artifacts = ExperimentArtifacts(
        metrics=metrics,
        plots=[str(baseline_dir / "plots" / "reward_curve.png")],
        log_path=str(baseline_dir / "logs" / "run.log"),
        commands_log_path=str(baseline_dir / "commands.log"),
        provenance_path=str(baseline_dir / "provenance.json"),
        success=True,
    )

    # Write artifacts summary
    write_json(baseline_dir / "artifacts.json", artifacts.model_dump(mode="json"))
    logger.info("Experiment artifacts written to %s", baseline_dir)
    return artifacts


def run_offline_failure(
    project_id: str,
    runs_root: Path,
    baseline_result: BaselineResult,
    error_message: str = "Training diverged: NaN loss at step 1000",
) -> ExperimentArtifacts:
    """Simulate a failed experiment for testing verification logic."""
    baseline_dir = initialize_run_artifacts(Path(runs_root) / project_id / "baseline")

    # Write partial log
    (baseline_dir / "logs" / "run.log").write_text(
        f"[ERROR] {error_message}\n"
    )
    for command in baseline_result.commands_to_run or ["python train.py"]:
        append_command_log(
            baseline_dir,
            CommandLogEntry(
                command=command,
                phase="offline_simulation",
                status="failed",
                started_at=utc_now_iso(),
                finished_at=utc_now_iso(),
                duration_seconds=0.0,
                exit_code=1,
                cause_kind="simulated_failure",
            ),
        )

    return ExperimentArtifacts(
        metrics={},
        plots=[],
        log_path=str(baseline_dir / "logs" / "run.log"),
        commands_log_path=str(baseline_dir / "commands.log"),
        provenance_path="",
        success=False,
        error_message=error_message,
    )


async def run_with_runtime(
    project_id: str,
    runs_root: Path,
    baseline_result: BaselineResult,
    reproduction_contract: ReproductionContract | None = None,
    *,
    runtime: RuntimeAppService | None = None,
    command_timeout: int = 3600,
) -> ExperimentArtifacts:
    """Execute baseline commands in a real RuntimeBackend sandbox."""

    runs = Path(runs_root)
    project_dir = runs / project_id
    baseline_dir = initialize_run_artifacts(project_dir / "baseline")
    logs_dir = baseline_dir / "logs"
    code_dir = Path(baseline_result.code_path) if baseline_result.code_path else project_dir / "code"
    dockerfile_path = (
        Path(baseline_result.dockerfile_path)
        if baseline_result.dockerfile_path
        else code_dir / "Dockerfile"
    )
    commands = baseline_result.commands_to_run or ["python train.py"]

    service = runtime or RuntimeAppService(LocalDockerBackend())
    config = SandboxConfig(
        project_id=project_id,
        run_id="baseline",
        image=f"reprolab/{project_id}:baseline",
        project_root=code_dir,
        artifact_root=baseline_dir,
        dockerfile_path=dockerfile_path if dockerfile_path.exists() else None,
        build_context=code_dir if code_dir.exists() else None,
        readonly_project=True,
        environment={"OUTPUT_DIR": "/artifacts"},
    )

    run_started_at = utc_now_iso()
    sandbox = None
    command_results: list[dict[str, Any]] = []
    run_log_path = logs_dir / "run.log"
    run_log_path.write_text("")
    try:
        sandbox = await service.create_sandbox(CreateSandbox(config=config))
        for idx, command in enumerate(commands, start=1):
            result = await service.execute(
                ExecuteCommand(
                    sandbox=sandbox,
                    command=command,
                    timeout=command_timeout,
                )
            )
            stdout_path = logs_dir / f"command_{idx:03d}.stdout.log"
            stderr_path = logs_dir / f"command_{idx:03d}.stderr.log"
            stdout_path.write_text(result.stdout)
            stderr_path.write_text(result.stderr)
            with run_log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"$ {command}\n")
                if result.stdout:
                    handle.write(result.stdout)
                    if not result.stdout.endswith("\n"):
                        handle.write("\n")
                if result.stderr:
                    handle.write(result.stderr)
                    if not result.stderr.endswith("\n"):
                        handle.write("\n")

            status = "succeeded" if result.succeeded else "failed"
            append_command_log(
                baseline_dir,
                CommandLogEntry(
                    command=command,
                    phase="experiment_runner",
                    status=status,
                    started_at=result.started_at.isoformat(),
                    finished_at=result.finished_at.isoformat(),
                    duration_seconds=result.duration_seconds,
                    exit_code=result.exit_code,
                    stdout_path=str(stdout_path),
                    stderr_path=str(stderr_path),
                    cause_kind=result.cause_kind.value if result.cause_kind else "",
                ),
            )
            command_results.append(result.model_dump(mode="json"))
            if not result.succeeded:
                return _runtime_failure_artifacts(
                    project_id,
                    baseline_dir,
                    baseline_result,
                    reproduction_contract,
                    command_results,
                    error_message=result.stderr or result.stdout or "Command failed",
                )

        metrics_path = baseline_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        plots = sorted(str(path) for path in (baseline_dir / "plots").glob("*") if path.is_file())
        provenance = _provenance_payload(
            project_id,
            baseline_result,
            reproduction_contract,
            run_started_at,
            utc_now_iso(),
            sandbox_id=sandbox.sandbox_id,
            image=sandbox.image,
            command_results=command_results,
            success=True,
        )
        write_provenance(baseline_dir, provenance)
        artifacts = ExperimentArtifacts(
            metrics=metrics,
            plots=plots,
            log_path=str(run_log_path),
            commands_log_path=str(baseline_dir / "commands.log"),
            provenance_path=str(baseline_dir / "provenance.json"),
            success=True,
        )
        write_json(baseline_dir / "artifacts.json", artifacts.model_dump(mode="json"))
        return artifacts
    except Exception as exc:
        return _runtime_failure_artifacts(
            project_id,
            baseline_dir,
            baseline_result,
            reproduction_contract,
            command_results,
            error_message=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if sandbox is not None:
            await service.destroy(DestroySandbox(sandbox=sandbox))


async def run_with_sdk(
    project_id: str,
    runs_root: Path,
    baseline_result: BaselineResult,
    reproduction_contract: ReproductionContract | None = None,
    *,
    model: str | None = None,
) -> ExperimentArtifacts:
    """Ask the Claude Agent SDK to plan/synthesize experiment artifacts.

    This path does not execute Docker. Real command execution is handled by
    ``run_with_runtime`` once a runtime backend is available.
    """
    from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, query
    from backend.agents.prompts.experiment_runner import EXPERIMENT_RUNNER_PROMPT

    project_dir = Path(runs_root) / project_id
    baseline_dir = project_dir / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    context = {
        "baseline_result": baseline_result.model_dump(),
        "reproduction_contract": reproduction_contract.model_dump() if reproduction_contract else {},
    }

    prompt = (
        f"Execute the baseline experiment for project {project_id}.\n"
        f"Write artifacts to {baseline_dir}\n"
        f"Context:\n```json\n{json.dumps(context, indent=2)}\n```"
    )

    options = ClaudeAgentOptions(
        model=model,
        system_prompt=EXPERIMENT_RUNNER_PROMPT,
        permission_mode="bypassPermissions",
        max_turns=30,
        cwd=str(project_dir),
    )

    collected: list[str] = []
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text"):
                    collected.append(block.text)
        elif isinstance(message, ResultMessage):
            pass

    # Try to read artifacts
    artifacts_path = baseline_dir / "artifacts.json"
    if artifacts_path.exists():
        return ExperimentArtifacts(**json.loads(artifacts_path.read_text()))

    metrics_path = baseline_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        return ExperimentArtifacts(
            metrics=metrics,
            log_path=str(baseline_dir / "logs" / "run.log"),
            commands_log_path=str(baseline_dir / "commands.log"),
            provenance_path=str(baseline_dir / "provenance.json"),
            success=True,
        )

    return ExperimentArtifacts(success=False, error_message="No artifacts produced")


def _runtime_failure_artifacts(
    project_id: str,
    baseline_dir: Path,
    baseline_result: BaselineResult,
    reproduction_contract: ReproductionContract | None,
    command_results: list[dict[str, Any]],
    *,
    error_message: str,
) -> ExperimentArtifacts:
    provenance = _provenance_payload(
        project_id,
        baseline_result,
        reproduction_contract,
        run_started_at="",
        run_finished_at=utc_now_iso(),
        sandbox_id="",
        image="",
        command_results=command_results,
        success=False,
        error_message=error_message,
    )
    write_provenance(baseline_dir, provenance)
    log_path = baseline_dir / "logs" / "run.log"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[ERROR] {error_message}\n")
    artifacts = ExperimentArtifacts(
        metrics={},
        plots=[],
        log_path=str(log_path),
        commands_log_path=str(baseline_dir / "commands.log"),
        provenance_path=str(baseline_dir / "provenance.json"),
        success=False,
        error_message=error_message,
    )
    write_json(baseline_dir / "artifacts.json", artifacts.model_dump(mode="json"))
    return artifacts


def _provenance_payload(
    project_id: str,
    baseline_result: BaselineResult,
    reproduction_contract: ReproductionContract | None,
    run_started_at: str,
    run_finished_at: str,
    *,
    sandbox_id: str,
    image: str,
    command_results: list[dict[str, Any]],
    success: bool,
    error_message: str = "",
) -> dict[str, Any]:
    return {
        "project_id": project_id,
        "mode": baseline_result.mode,
        "code_path": baseline_result.code_path,
        "dockerfile_path": baseline_result.dockerfile_path,
        "sandbox_id": sandbox_id,
        "image": image,
        "started_at": run_started_at,
        "finished_at": run_finished_at,
        "success": success,
        "error_message": error_message,
        "commands": baseline_result.commands_to_run,
        "command_results": command_results,
        "assumptions_applied": baseline_result.assumptions_applied,
        "reproduction_contract": (
            reproduction_contract.model_dump(mode="json")
            if reproduction_contract is not None
            else {}
        ),
    }


def _write_placeholder_plot(path: Path) -> None:
    """Write a minimal valid PNG file as a placeholder."""
    # Minimal 1x1 pixel PNG
    import struct
    import zlib

    def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk = chunk_type + data
        return struct.pack(">I", len(data)) + chunk + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    raw_data = b"\x00\xff\xff\xff"  # filter byte + RGB
    idat = zlib.compress(raw_data)

    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", ihdr)
    png += _png_chunk(b"IDAT", idat)
    png += _png_chunk(b"IEND", b"")

    path.write_bytes(png)
