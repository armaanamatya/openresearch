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
    baseline_dir = Path(runs_root) / project_id / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    (baseline_dir / "plots").mkdir(exist_ok=True)
    (baseline_dir / "logs").mkdir(exist_ok=True)

    # Default simulation metrics (PPO CartPole-v1 success)
    metrics = simulate_metrics or {
        "mean_reward": 487.3,
        "eval_episodes": 100,
        "total_timesteps": 500000,
        "elapsed_seconds": 245.7,
        "target_met": True,
    }

    # Write metrics.json
    (baseline_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

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

    # Write commands.log
    commands = baseline_result.commands_to_run or ["python train.py"]
    (baseline_dir / "commands.log").write_text("\n".join(commands))

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
    (baseline_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

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
    (baseline_dir / "artifacts.json").write_text(artifacts.model_dump_json(indent=2))
    logger.info("Experiment artifacts written to %s", baseline_dir)
    return artifacts


def run_offline_failure(
    project_id: str,
    runs_root: Path,
    baseline_result: BaselineResult,
    error_message: str = "Training diverged: NaN loss at step 1000",
) -> ExperimentArtifacts:
    """Simulate a failed experiment for testing verification logic."""
    baseline_dir = Path(runs_root) / project_id / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    (baseline_dir / "logs").mkdir(exist_ok=True)

    # Write partial log
    (baseline_dir / "logs" / "run.log").write_text(
        f"[ERROR] {error_message}\n"
    )
    (baseline_dir / "commands.log").write_text(
        "\n".join(baseline_result.commands_to_run or ["python train.py"])
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
