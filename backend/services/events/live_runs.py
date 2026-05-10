"""Live run process management and SSE event streaming."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from backend.config import get_settings

RunMode = Literal["offline", "sdk"]
Provider = Literal["anthropic", "openai"]
ExecutionMode = Literal["efficient", "max"]
SandboxMode = Literal["auto", "docker", "local", "runpod"]
GpuMode = Literal["off", "auto", "prefer", "max"]
RunStatus = Literal["queued", "running", "stopped", "completed", "failed"]


class StartRunRequest(BaseModel):
    mode: RunMode = "offline"
    provider: Provider = "anthropic"
    verificationProvider: Provider | None = None
    executionMode: ExecutionMode = "efficient"
    sandbox: SandboxMode = "auto"
    gpuMode: GpuMode = "auto"


class LiveRunState(BaseModel):
    projectId: str
    outputDir: str
    runMode: RunMode
    llmProvider: Provider | None = None
    verificationProvider: Provider | None = None
    executionMode: ExecutionMode | None = None
    sandboxMode: SandboxMode | None = None
    gpuMode: GpuMode | None = None
    status: RunStatus
    sourceKind: Literal["workspace_fixture", "uploaded_pdf"] | None = None
    sourceLabel: str | None = None
    sourceNote: str | None = None
    startedAt: str | None = None
    updatedAt: str | None = None
    completedAt: str | None = None
    error: str | None = None
    pid: int | None = None
    payload: Any | None = None
    log: str = ""


def sse_event(event: str, data: Any, *, event_id: str | None = None) -> str:
    prefix = f"id: {event_id}\n" if event_id else ""
    return f"{prefix}event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


class FileLiveRunService:
    """Runs pipelines in subprocesses and exposes their file-backed state."""

    def __init__(
        self,
        *,
        runs_root: Path | None = None,
        repo_root: Path | None = None,
        python_bin: str | None = None,
    ) -> None:
        self.repo_root = (repo_root or Path(__file__).resolve().parents[3]).resolve()
        self.runs_root = (runs_root or self.repo_root / "runs").resolve()
        self.python_bin = python_bin or sys.executable

    async def start_run(self, request: StartRunRequest) -> LiveRunState:
        project_id = _fixture_project_id(request)
        return await self._start_python_run(
            request,
            project_id=project_id,
            uploaded_paper=None,
        )

    async def start_uploaded_run(
        self,
        request: StartRunRequest,
        *,
        file_name: str,
        content: bytes,
    ) -> LiveRunState:
        staged = await asyncio.to_thread(self._stage_upload, file_name, content)
        project_id = project_id_for_pdf_path(staged)
        return await self._start_python_run(
            request,
            project_id=project_id,
            uploaded_paper={"path": str(staged), "fileName": file_name},
        )

    async def get_run(self, project_id: str) -> LiveRunState | None:
        return await asyncio.to_thread(self._load_run, project_id)

    async def latest_run(
        self,
        *,
        mode: str | None = None,
        provider: str | None = None,
        execution_mode: str | None = None,
        sandbox: str | None = None,
        verification_provider: str | None = None,
        gpu_mode: str | None = None,
    ) -> LiveRunState | None:
        return await asyncio.to_thread(
            self._latest_run,
            mode,
            provider,
            execution_mode,
            sandbox,
            verification_provider,
            gpu_mode,
        )

    async def stop_run(self, project_id: str) -> LiveRunState | None:
        status = await asyncio.to_thread(self._read_status, project_id)
        if status is None:
            return None
        pid = status.get("pid")
        if isinstance(pid, int) and pid > 0:
            await asyncio.to_thread(_terminate_pid, pid)
        stopped_at = _now()
        status.update(
            {
                "status": "stopped",
                "updatedAt": stopped_at,
                "completedAt": stopped_at,
                "error": "Stopped by user",
            }
        )
        await asyncio.to_thread(self._write_status, project_id, status)
        return await self.get_run(project_id)

    async def stream_events(self, project_id: str) -> AsyncIterator[str]:
        last_log_len = 0
        last_status_json = ""
        counter = 0
        state = await self.get_run(project_id)
        if state is None:
            yield sse_event("agent_failed", {"projectId": project_id, "error": "Run not found"})
            return

        yield sse_event("run_state", state.model_dump(mode="json"), event_id=str(counter))
        last_status_json = state.model_dump_json()
        if state.log:
            last_log_len = len(state.log)
            yield sse_event(
                "agent_log",
                {"projectId": project_id, "text": state.log[-12000:], "log": state.log},
            )

        while state.status in {"queued", "running"}:
            await asyncio.sleep(1)
            counter += 1
            state = await self.get_run(project_id)
            if state is None:
                yield sse_event("agent_failed", {"projectId": project_id, "error": "Run not found"})
                return
            state_json = state.model_dump_json()
            if state_json != last_status_json:
                last_status_json = state_json
                yield sse_event("run_state", state.model_dump(mode="json"), event_id=str(counter))
            if len(state.log) > last_log_len:
                delta = state.log[last_log_len:]
                last_log_len = len(state.log)
                yield sse_event(
                    "agent_log",
                    {"projectId": project_id, "text": delta, "log": state.log},
                    event_id=f"log-{counter}",
                )
            if counter % 15 == 0:
                yield sse_event("heartbeat", {"projectId": project_id, "status": state.status})

    async def _start_python_run(
        self,
        request: StartRunRequest,
        *,
        project_id: str,
        uploaded_paper: dict[str, str] | None,
    ) -> LiveRunState:
        existing = await self.get_run(project_id)
        if existing and existing.status in {"queued", "running"} and _pid_exists(existing.pid):
            return existing

        output_dir = self.runs_root / project_id
        output_dir.mkdir(parents=True, exist_ok=True)
        meta = _initial_status(
            request,
            project_id=project_id,
            output_dir=output_dir,
            uploaded_paper=uploaded_paper,
        )
        await asyncio.to_thread(self._write_status, project_id, meta)

        stderr = (output_dir / "runner.stderr.log").open("a", encoding="utf-8")
        stdout = (output_dir / "runner.stdout.log").open("a", encoding="utf-8")
        try:
            process = subprocess.Popen(
                [
                    self.python_bin,
                    "-u",
                    "-c",
                    _python_script(
                        request,
                        project_id=project_id,
                        runs_root=self.runs_root,
                        uploaded_paper=uploaded_paper,
                    ),
                ],
                cwd=self.repo_root,
                stdout=stdout,
                stderr=stderr,
                stdin=subprocess.DEVNULL,
                env={
                    **os.environ,
                    "REPROLAB_GPU_MODE": request.gpuMode,
                    **({"REPROLAB_LLM_PROVIDER": request.provider} if request.mode == "sdk" else {}),
                    **(
                        {"REPROLAB_VERIFICATION_PROVIDER": request.verificationProvider}
                        if request.verificationProvider
                        else {}
                    ),
                },
            )
        finally:
            stderr.close()
            stdout.close()

        meta.update({"pid": process.pid, "updatedAt": _now()})
        await asyncio.to_thread(self._write_status, project_id, meta)
        return (await self.get_run(project_id)) or LiveRunState(**meta, payload=None, log="")

    def _load_run(self, project_id: str) -> LiveRunState | None:
        status = self._read_status(project_id)
        if status is None:
            return None
        if status.get("status") in {"queued", "running"} and not _pid_exists(status.get("pid")):
            status = {
                **status,
                "status": "failed",
                "updatedAt": _now(),
                "completedAt": _now(),
                "error": _summarize_failure(self._read_log(project_id)),
            }
            self._write_status(project_id, status)
        status.setdefault("payload", None)
        status["log"] = self._read_log(project_id)
        return LiveRunState(**status)

    def _latest_run(
        self,
        mode: str | None,
        provider: str | None,
        execution_mode: str | None,
        sandbox: str | None,
        verification_provider: str | None,
        gpu_mode: str | None,
    ) -> LiveRunState | None:
        candidates: list[tuple[float, str]] = []
        if not self.runs_root.exists():
            return None
        for status_path in self.runs_root.glob("*/demo_status.json"):
            status = _read_json(status_path)
            if not status:
                continue
            if mode and status.get("runMode") != mode:
                continue
            if provider and (status.get("llmProvider") or _provider_from_project_id(status_path.parent.name)) != provider:
                continue
            if execution_mode and (status.get("executionMode") or "efficient") != execution_mode:
                continue
            if sandbox and (status.get("sandboxMode") or "runpod") != sandbox:
                continue
            if verification_provider and status.get("verificationProvider") != verification_provider:
                continue
            if gpu_mode and (status.get("gpuMode") or "auto") != gpu_mode:
                continue
            timestamp = _parse_time(status.get("updatedAt") or status.get("startedAt"))
            candidates.append((timestamp, status_path.parent.name))
        if not candidates:
            return None
        _, project_id = sorted(candidates, reverse=True)[0]
        return self._load_run(project_id)

    def _read_status(self, project_id: str) -> dict[str, Any] | None:
        return _read_json(self.runs_root / project_id / "demo_status.json")

    def _write_status(self, project_id: str, status: dict[str, Any]) -> None:
        run_dir = self.runs_root / project_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "demo_status.json").write_text(
            json.dumps(status, indent=2),
            encoding="utf-8",
        )

    def _read_log(self, project_id: str, max_chars: int = 12000) -> str:
        path = self.runs_root / project_id / "runner.stderr.log"
        if not path.exists():
            return ""
        value = path.read_text(encoding="utf-8", errors="replace")
        return value[-max_chars:]

    def _stage_upload(self, file_name: str, content: bytes) -> Path:
        uploads_root = self.runs_root / ".lab_uploads"
        uploads_root.mkdir(parents=True, exist_ok=True)
        ext = Path(file_name).suffix or ".pdf"
        safe_base = "".join(
            char if char.isalnum() or char in "._-" else "-"
            for char in Path(file_name).stem
        ).strip("-") or "paper"
        target = uploads_root / f"{int(datetime.now().timestamp() * 1000)}-{uuid4()}-{safe_base}{ext}"
        target.write_bytes(content)
        return target


def project_id_for_pdf_path(file_path: Path) -> str:
    digest = sha256(f"pdf_path:{file_path.resolve()}".encode("utf-8")).hexdigest()
    return f"prj_{digest[:16]}"


def _fixture_project_id(request: StartRunRequest) -> str:
    review = request.verificationProvider or "same"
    if request.mode == "sdk":
        return f"ui_sdk_{request.provider}_review_{review}_demo_{int(datetime.now().timestamp() * 1000)}"
    return f"ui_demo_{int(datetime.now().timestamp() * 1000)}"


def _initial_status(
    request: StartRunRequest,
    *,
    project_id: str,
    output_dir: Path,
    uploaded_paper: dict[str, str] | None,
) -> dict[str, Any]:
    now = _now()
    status: dict[str, Any] = {
        "projectId": project_id,
        "outputDir": str(output_dir),
        "runMode": request.mode,
        "executionMode": request.executionMode,
        "sandboxMode": request.sandbox,
        "gpuMode": request.gpuMode,
        "status": "queued",
        "startedAt": now,
        "updatedAt": now,
    }
    if request.mode == "sdk":
        status["llmProvider"] = request.provider
        if request.verificationProvider:
            status["verificationProvider"] = request.verificationProvider
    if uploaded_paper:
        status.update(
            {
                "sourceKind": "uploaded_pdf",
                "sourceLabel": uploaded_paper["fileName"],
                "sourceNote": (
                    "This run started from a PDF uploaded directly in the lab. "
                    "The backend routed it through the repo's paper ingestion pipeline before running reproduction."
                ),
            }
        )
    else:
        status.update(
            {
                "sourceKind": "workspace_fixture",
                "sourceLabel": "In-repo PPO workspace fixture",
                "sourceNote": (
                    "This UI demo uses the deterministic PPO workspace fixture that already drives "
                    "the end-to-end pipeline tests."
                ),
            }
        )
    return status


def _python_script(
    request: StartRunRequest,
    *,
    project_id: str,
    runs_root: Path,
    uploaded_paper: dict[str, str] | None,
) -> str:
    common = {
        "project_id": project_id,
        "run_mode": request.mode,
        "provider": request.provider,
        "verification_provider": request.verificationProvider,
        "execution_mode": request.executionMode,
        "sandbox": request.sandbox,
        "gpu_mode": request.gpuMode,
        "runs_root": str(runs_root),
        "database_url": get_settings().database_url,
        "uploaded_paper": uploaded_paper,
    }
    return f"""
import asyncio
import json
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

from backend.agents.execution import ExecutionProfile, SandboxMode
from backend.agents.pipeline import run_pipeline_offline, run_pipeline_sdk
from backend.cli import cmd_reproduce

config = json.loads({json.dumps(json.dumps(common))})
project_id = config["project_id"]
runs_root = Path(config["runs_root"])
output_dir = runs_root / project_id
output_dir.mkdir(parents=True, exist_ok=True)
status_path = output_dir / "demo_status.json"

DEMO_WORKSPACE = {{
    "project_id": "prj_e2e_test",
    "entries": [
        {{"source_id": "src_1", "title": "Abstract", "excerpt": "We propose a new family of policy gradient methods for reinforcement learning, which alternate between sampling data and optimizing a surrogate objective."}},
        {{"source_id": "src_2", "title": "Experiments", "excerpt": "We test on CartPole-v1 environment. We use Adam optimizer with learning rate 3e-4 and batch size 64. We report mean reward over 100 episodes after 500000 timesteps."}},
        {{"source_id": "src_3", "title": "Conclusion", "excerpt": "We have introduced proximal policy optimization, a family of methods that use multiple epochs of stochastic gradient ascent."}},
    ],
}}

def now():
    return datetime.now(timezone.utc).isoformat()

def write_status(status, error=None, completed_at=None):
    existing = {{}}
    if status_path.exists():
        try:
            existing = json.loads(status_path.read_text())
        except Exception:
            existing = {{}}
    payload = {{
        **existing,
        "status": status,
        "updatedAt": now(),
    }}
    if completed_at:
        payload["completedAt"] = completed_at
    if error:
        payload["error"] = error
    status_path.write_text(json.dumps(payload, indent=2))

write_status("running")

try:
    if config["uploaded_paper"]:
        exit_code = cmd_reproduce(Namespace(
            source=config["uploaded_paper"]["path"],
            source_kind="pdf_path",
            agent="default",
            mode=config["run_mode"],
            model=None,
            provider=config["provider"] if config["run_mode"] == "sdk" else None,
            verification_provider=config["verification_provider"] if config["run_mode"] == "sdk" else None,
            execution_mode=config["execution_mode"],
            sandbox=config["sandbox"],
            gpu_mode=config["gpu_mode"],
            command_timeout=None,
            allow_sandbox_network=False,
            sandbox_platform=None,
            sandbox_memory=None,
            sandbox_cpus=None,
            hints="Keep this as a lightweight smoke test",
            n_paths=1,
            runs_root=config["runs_root"],
            database_url=config["database_url"],
        ))
        if exit_code != 0:
            raise RuntimeError(f"Pipeline exited with status {{exit_code}}")
    else:
        profile = ExecutionProfile.from_mode(config["execution_mode"], gpu_mode=config["gpu_mode"])
        if config["run_mode"] == "sdk":
            asyncio.run(run_pipeline_sdk(
                project_id,
                runs_root,
                DEMO_WORKSPACE,
                provider=config["provider"],
                verification_provider=config["verification_provider"],
                user_hints=["Keep this as a lightweight smoke test"],
                n_improvement_paths=1,
                execution_profile=profile,
                sandbox_mode=SandboxMode(config["sandbox"]),
            ))
        else:
            run_pipeline_offline(
                project_id,
                runs_root,
                DEMO_WORKSPACE,
                execution_profile=profile,
                sandbox_mode=SandboxMode(config["sandbox"]),
            )
    write_status("completed", completed_at=now())
except Exception as exc:
    write_status("failed", error=f"{{type(exc).__name__}}: {{exc}}", completed_at=now())
    raise
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _parse_time(value: Any) -> float:
    if not isinstance(value, str):
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _provider_from_project_id(project_id: str) -> str | None:
    if project_id.startswith("ui_sdk_openai_"):
        return "openai"
    if project_id.startswith("ui_sdk_anthropic_") or project_id.startswith("ui_sdk_demo_"):
        return "anthropic"
    return None


def _pid_exists(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _terminate_pid(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, capture_output=True)
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass


def _summarize_failure(log: str) -> str:
    lines = [line.strip() for line in log.splitlines() if line.strip()]
    for line in reversed(lines):
        if "error" in line.lower() or "exception" in line.lower():
            return line
    return lines[-1] if lines else "Demo runner stopped before completion"


__all__ = [
    "FileLiveRunService",
    "LiveRunState",
    "StartRunRequest",
    "project_id_for_pdf_path",
    "sse_event",
]

