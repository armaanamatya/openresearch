"""In-Pod entrypoint for the durable controller (WS3).

Runs inside the controller Job Pod: acquires the run's drive-lease, renews it on
a heartbeat cadence (so an expired lease reliably means "controller gone" —
which is what makes :func:`controller_cluster.sweep_orphaned_controllers` safe),
exports the stable fence into the env the campaign reads to fence its cell Jobs,
and drives the campaign to a terminal state. On losing the lease (superseded by
a takeover) it stops, so two controller generations never write concurrently.

The heavy cluster/campaign wiring here executes only inside a real Pod and is
exercised at drill time; :func:`heartbeat_loop` is the pure, unit-tested core.

Design: ``docs/history/specs/2026-07-12-cloud-native-durable-and-cpu-lane-design.md`` §3.6, §7.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

LEASE_LOST_EXIT = 75

_SECRET_FILES: dict[str, str] = {
    "anthropic-api-key": "ANTHROPIC_API_KEY",
    "azure-openai-api-key": "AZURE_OPENAI_API_KEY",
    "azure-foundry-api-key": "AZURE_FOUNDRY_API_KEY",
    "claude-code-oauth-token": "CLAUDE_CODE_OAUTH_TOKEN",
}

_CAMPAIGN_BUDGET_FLAGS: tuple[tuple[str, str], ...] = (
    ("OPENRESEARCH_CAMPAIGN_MAX_LLM_USD", "--max-llm-usd"),
    ("OPENRESEARCH_CAMPAIGN_MAX_GPU_USD", "--max-gpu-usd"),
    ("OPENRESEARCH_CAMPAIGN_MAX_GPU_HOURS", "--max-gpu-hours"),
)


def heartbeat_loop(
    *,
    lease: Any,
    token: Any,
    interval_s: float,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
    is_running: Callable[[], bool],
    on_lost: Callable[[], None],
) -> None:
    """Renew ``token`` every ``interval_s`` while ``is_running()`` holds.

    Fail-closed on supersede: if ``lease.renew`` returns ``None`` the lease was
    taken over, so ``on_lost`` is called (the caller stops the campaign) and the
    loop exits. Clock and sleep are injected so the loop is deterministic under
    test — nothing here calls ``time`` directly.
    """
    while is_running():
        sleep(interval_s)
        if not is_running():
            return
        token = lease.renew(token, clock())
        if token is None:
            on_lost()
            return


def load_mounted_secrets(secret_dir: Path, env: dict[str, str]) -> None:
    """Load CSI-mounted provider credentials without putting values in Job specs."""
    for file_name, env_name in _SECRET_FILES.items():
        path = secret_dir / file_name
        if not path.is_file() or env.get(env_name):
            continue
        value = path.read_text(encoding="utf-8").strip()
        if value:
            env[env_name] = value


def _terminate_process_group(process: Any) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except Exception:
        try:
            process.terminate()
        except Exception:
            return
    try:
        process.wait(timeout=10)
        return
    except Exception:
        pass
    try:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except Exception:
        pass


def drive_controller(
    *,
    lease: Any,
    project_id: str,
    owner_id: str,
    command: list[str],
    env: dict[str, str],
    heartbeat_s: float,
    clock: Callable[[], float] = time.time,
    popen_factory: Callable[..., Any] = subprocess.Popen,
    checkpoint: Callable[[], None] | None = None,
    on_acquired: Callable[[Any], None] | None = None,
) -> int:
    """Acquire, drive, and heartbeat one campaign until it exits or loses lease."""
    token = lease.acquire(project_id, owner_id, clock())
    if token is None:
        logger.warning("controller %s is already owned; stale Job exiting", project_id)
        return 0

    if on_acquired is not None:
        try:
            on_acquired(token)
        except Exception:
            logger.exception("controller %s stale-job reap failed", project_id)

    child_env = dict(env)
    child_env["OPENRESEARCH_DURABLE_CONTROLLER"] = "1"
    child_env["OPENRESEARCH_CELL_FENCE_EPOCH"] = str(token.fence_epoch)
    process = popen_factory(command, env=child_env, start_new_session=(os.name != "nt"))

    previous_handlers: dict[int, Any] = {}
    if os.name != "nt":
        def _forward_shutdown(signum: int, _frame: Any) -> None:
            logger.warning("controller %s received signal %s", project_id, signum)
            _terminate_process_group(process)

        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _forward_shutdown)

    def _checkpoint_if_current(*, final: bool = False) -> None:
        if checkpoint is None:
            return
        try:
            if not lease.is_current(token):
                logger.warning(
                    "controller %s skipped %scheckpoint after supersede",
                    project_id,
                    "final " if final else "",
                )
                return
            checkpoint()
        except Exception:
            logger.exception(
                "controller %s %scheckpoint failed",
                project_id,
                "final " if final else "",
            )

    try:
        while True:
            try:
                exit_code = int(process.wait(timeout=heartbeat_s))
                return exit_code
            except subprocess.TimeoutExpired:
                pass
            try:
                renewed = lease.renew(token, clock())
            except Exception:
                logger.exception("controller %s lease renewal errored; stopping", project_id)
                renewed = None
            if renewed is None:
                logger.error("controller %s lost its lease; stopping campaign", project_id)
                _terminate_process_group(process)
                return LEASE_LOST_EXIT
            token = renewed
            _checkpoint_if_current()
    finally:
        _checkpoint_if_current(final=True)
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)


def _download_controller_input(
    cloud: str,
    *,
    blob_name: str,
    destination: Path,
) -> str:
    """Download one launcher-staged input with the pod's workload identity."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    from backend.config import get_settings

    settings = get_settings()
    if cloud == "gcp":
        from backend.services.runtime import gcs_blob

        gcs_blob.download_artifact(
            blob_name,
            destination,
            bucket=settings.gcp_gcs_bucket,
            project=settings.gcp_project or None,
        )
    else:
        from backend.services.runtime import azure_blob

        azure_blob.download_artifact(
            blob_name,
            destination,
            account_name=settings.azure_storage_account,
            container_name=settings.azure_blob_container,
        )
    return str(destination)


def _materialize_paper(cloud: str, project_id: str, runs_root: Path) -> str | None:
    blob_name = os.environ.get("OPENRESEARCH_CONTROLLER_PAPER_BLOB", "").strip()
    if not blob_name:
        return None
    return _download_controller_input(
        cloud,
        blob_name=blob_name,
        destination=runs_root / "_inputs" / f"{project_id}.pdf",
    )


def _materialize_run_spec(
    cloud: str, project_id: str, runs_root: Path
) -> str | None:
    blob_name = os.environ.get("OPENRESEARCH_CONTROLLER_RUN_SPEC_BLOB", "").strip()
    if not blob_name:
        return None
    return _download_controller_input(
        cloud,
        blob_name=blob_name,
        destination=runs_root / "_inputs" / f"{project_id}-run-spec.json",
    )


def _campaign_budget_args(env: dict[str, str]) -> list[str]:
    args: list[str] = []
    for env_name, flag in _CAMPAIGN_BUDGET_FLAGS:
        raw = env.get(env_name, "").strip()
        try:
            value = float(raw)
        except ValueError as exc:
            raise RuntimeError(f"durable controller requires numeric {env_name}") from exc
        if value <= 0:
            raise RuntimeError(f"durable controller requires positive {env_name}")
        args.extend([flag, raw])
    return args


def _heartbeat_interval(env: dict[str, str]) -> float:
    from backend.services.runtime.blob_lease import LEASE_TTL_S

    raw = env.get("OPENRESEARCH_CONTROLLER_HEARTBEAT_S", "").strip() or "60"
    try:
        heartbeat_s = float(raw)
    except ValueError as exc:
        raise RuntimeError(
            "OPENRESEARCH_CONTROLLER_HEARTBEAT_S must be numeric"
        ) from exc
    if heartbeat_s <= 0 or heartbeat_s > LEASE_TTL_S / 3:
        raise RuntimeError(
            "OPENRESEARCH_CONTROLLER_HEARTBEAT_S must be positive and no "
            f"greater than one third of the {LEASE_TTL_S}s lease TTL"
        )
    return heartbeat_s


def _campaign_command(
    *,
    paper: str,
    project_id: str,
    cloud: str,
    runs_root: Path,
    run_spec: str | None,
    root_model: str | None = None,
    execution_mode: str = "max",
    gpu_mode: str = "auto",
    minimize_compute: bool = False,
    env: dict[str, str] | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "backend.cli",
        "--runs-root",
        str(runs_root),
        "campaign",
        paper,
        *_campaign_budget_args(dict(os.environ) if env is None else env),
        "--project-id",
        project_id,
        "--resume",
        "--sandbox",
        cloud,
        "--billing-sandbox",
        cloud,
        "--execution-mode",
        execution_mode,
        "--gpu-mode",
        gpu_mode,
    ]
    if minimize_compute:
        command.append("--minimize-compute")
    if run_spec:
        command.extend(["--run-spec", run_spec])
    if root_model:
        command.extend(["--root-model", root_model])
    return command


def _publish_controller_state(cloud: str, project_id: str, runs_root: Path) -> None:
    """Publish small operator-facing state files to the cloud artifact store."""
    project_dir = runs_root / project_id
    if not project_dir.is_dir():
        return
    candidates = [
        project_dir / "demo_status.json",
        project_dir / "dashboard_events.jsonl",
        project_dir / "final_report.json",
        project_dir / "final_report.md",
        project_dir / "cost_ledger.jsonl",
        project_dir / "experiment_runs.jsonl",
        project_dir / "tokens_total.json",
        project_dir / "campaign" / "campaign.json",
        project_dir / "campaign" / "attempts.jsonl",
    ]
    from backend.config import get_settings

    settings = get_settings()
    prefix = f"runs/{project_id}/controller-state"
    for path in candidates:
        if not path.is_file():
            continue
        blob_name = f"{prefix}/{path.relative_to(project_dir).as_posix()}"
        data = path.read_bytes()
        if cloud == "gcp":
            from backend.services.runtime import gcs_blob

            gcs_blob.upload_bytes(
                data,
                blob_name=blob_name,
                bucket=settings.gcp_gcs_bucket,
                project=settings.gcp_project or None,
            )
        else:
            from backend.services.runtime import azure_blob

            azure_blob.upload_bytes(
                data,
                blob_name=blob_name,
                account_name=settings.azure_storage_account,
                container_name=settings.azure_blob_container,
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="openresearch-controller")
    parser.add_argument("--cloud", choices=("gcp", "azure"), required=True)
    parser.add_argument("--paper", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--run-spec", default=None)
    parser.add_argument("--root-model", default=None)
    parser.add_argument(
        "--execution-mode", choices=("efficient", "max"), default="max"
    )
    parser.add_argument(
        "--gpu-mode", choices=("off", "auto", "prefer", "max"), default="auto"
    )
    parser.add_argument("--minimize-compute", action="store_true")
    args = parser.parse_args(argv)

    secret_dir = Path(
        os.environ.get(
            "OPENRESEARCH_CONTROLLER_SECRET_DIR", "/mnt/orchestrator-secrets"
        )
    )
    load_mounted_secrets(secret_dir, os.environ)

    from backend.agents.rlm.controller_cluster import build_default_cluster

    cluster = build_default_cluster(args.cloud)
    runs_root = Path(args.runs_root)
    paper = _materialize_paper(args.cloud, args.project_id, runs_root) or args.paper
    run_spec = (
        _materialize_run_spec(args.cloud, args.project_id, runs_root)
        or args.run_spec
    )
    command = _campaign_command(
        paper=paper,
        project_id=args.project_id,
        cloud=args.cloud,
        runs_root=runs_root,
        run_spec=run_spec,
        root_model=args.root_model,
        execution_mode=args.execution_mode,
        gpu_mode=args.gpu_mode,
        minimize_compute=args.minimize_compute,
        env=dict(os.environ),
    )
    heartbeat_s = _heartbeat_interval(dict(os.environ))
    exit_code = drive_controller(
        lease=cluster,
        project_id=args.project_id,
        owner_id=os.environ["OPENRESEARCH_CONTROLLER_OWNER_ID"],
        command=command,
        env=dict(os.environ),
        heartbeat_s=heartbeat_s,
        checkpoint=lambda: _publish_controller_state(
            args.cloud, args.project_id, runs_root
        ),
        on_acquired=lambda token: cluster.reap(args.project_id, token),
    )
    from backend.agents.rlm.run_controller import controller_job_exit_code

    return controller_job_exit_code(exit_code)


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
