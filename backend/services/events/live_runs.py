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
    sandbox: SandboxMode = "runpod"
    gpuMode: GpuMode = "auto"


class TelemetryRecordPublic(BaseModel):
    agent_id: str | None = None
    model: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    message_count: int | None = None
    output_chars: int | None = None
    success: bool | None = None
    error_message: str | None = None
    tool_calls: list[str] = Field(default_factory=list)


PIPELINE_STAGE_ORDER = [
    "ingested",
    "paper_understood",
    "artifacts_discovered",
    "environment_built",
    "plan_created",
    "gate_1_passed",
    "baseline_implemented",
    "baseline_run",
    "gate_2_passed",
    "improvements_selected",
    "improvements_run",
    "gate_3_passed",
    "research_map_generated",
    "complete",
]


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
    telemetry: list[TelemetryRecordPublic] = Field(default_factory=list)


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
        last_dash_count = 0
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

        # Flush any dashboard events already written before streaming started
        initial_dash = await asyncio.to_thread(self._read_dashboard_events, project_id, 0)
        for dash_event in initial_dash:
            yield sse_event("dashboard_event", dash_event, event_id=f"dash-{last_dash_count}")
            last_dash_count += 1

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
            # Stream new dashboard events
            new_dash = await asyncio.to_thread(self._read_dashboard_events, project_id, last_dash_count)
            for dash_event in new_dash:
                yield sse_event("dashboard_event", dash_event, event_id=f"dash-{last_dash_count}")
                last_dash_count += 1
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
        # On Windows, detach the child from the parent's process group so its
        # lifecycle doesn't trigger uvicorn's shutdown handler.
        creation_flags = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            if sys.platform == "win32"
            else 0
        )
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
                creationflags=creation_flags,
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
        pid = status.get("pid")
        if (
            status.get("status") in {"queued", "running"}
            and isinstance(pid, int)
            and pid > 0
            and not _pid_exists(pid)
        ):
            status = {
                **status,
                "status": "failed",
                "updatedAt": _now(),
                "completedAt": _now(),
                "error": _summarize_failure(self._read_log(project_id)),
            }
            self._write_status(project_id, status)
        status["log"] = self._read_log(project_id)
        status["telemetry"] = self._read_telemetry(project_id)
        status["payload"] = self._build_payload(project_id, status)
        status["llmProvider"] = _public_provider(status.get("llmProvider"))
        status["verificationProvider"] = _public_provider(status.get("verificationProvider"))
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
        path = run_dir / "demo_status.json"
        # Atomic write via tempfile + os.replace: a crash mid-write
        # leaves either the previous valid JSON or the new one — never
        # a half-written file that breaks _read_status downstream.
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(status, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def _read_telemetry(self, project_id: str, max_records: int = 50) -> list[TelemetryRecordPublic]:
        path = self.runs_root / project_id / "agent_telemetry.jsonl"
        if not path.exists():
            return []
        records: list[TelemetryRecordPublic] = []
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in lines[-max_records:]:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    records.append(TelemetryRecordPublic(
                        agent_id=data.get("agent_id"),
                        model=data.get("model"),
                        started_at=data.get("started_at"),
                        finished_at=data.get("finished_at"),
                        duration_seconds=data.get("duration_seconds"),
                        message_count=data.get("message_count"),
                        output_chars=data.get("output_chars"),
                        success=data.get("success"),
                        error_message=data.get("error_message") or None,
                        tool_calls=data.get("tool_calls", []),
                    ))
                except (json.JSONDecodeError, Exception):
                    continue
        except OSError:
            pass
        return records

    def _read_dashboard_events(self, project_id: str, offset: int = 0) -> list[dict[str, Any]]:
        path = self.runs_root / project_id / "dashboard_events.jsonl"
        if not path.exists():
            return []
        events: list[dict[str, Any]] = []
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in lines[offset:]:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        except OSError:
            pass
        return events

    def _read_pipeline_state(self, project_id: str) -> dict[str, Any] | None:
        return _read_json(self.runs_root / project_id / "pipeline_state.json")

    def _build_payload(self, project_id: str, status: dict[str, Any]) -> dict[str, Any]:
        pipeline_state = self._read_pipeline_state(project_id)
        dashboard_events = self._read_dashboard_events(project_id)
        log = str(status.get("log") or "")
        stage = _infer_stage(self.runs_root / project_id, pipeline_state, dashboard_events)
        meta = {
            "projectId": project_id,
            "outputDir": status.get("outputDir") or str(self.runs_root / project_id),
            "sourceKind": status.get("sourceKind") or "workspace_fixture",
            "runMode": status.get("runMode") or "offline",
            "llmProvider": status.get("llmProvider"),
            "verificationProvider": status.get("verificationProvider"),
            "executionMode": status.get("executionMode"),
            "sandboxMode": status.get("sandboxMode"),
            "gpuMode": status.get("gpuMode"),
            "sourceLabel": status.get("sourceLabel") or project_id,
            "sourceNote": status.get("sourceNote") or "",
        }
        return {
            **meta,
            "generatedAt": _now(),
            "log": log,
            "pipelineState": pipeline_state,
            "initialSnapshot": _snapshot_from_dashboard_events(dashboard_events),
            "events": dashboard_events,
            "summary": {
                "stage": stage,
                "meanReward": _mean_reward(pipeline_state),
                "improvementCount": len((pipeline_state or {}).get("path_results") or []),
                "runModeLabel": _run_mode_label(meta["runMode"], meta.get("llmProvider")),
                "llmProvider": meta.get("llmProvider"),
                "verificationProvider": meta.get("verificationProvider"),
                "executionMode": meta.get("executionMode"),
                "sandboxMode": meta.get("sandboxMode"),
                "gpuMode": meta.get("gpuMode"),
                "sourceLabel": meta["sourceLabel"],
            },
        }

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


def _public_provider(provider: Any) -> Provider | None:
    if provider in {"anthropic", "openai"}:
        return provider
    return None


def _infer_stage(
    run_dir: Path,
    pipeline_state: dict[str, Any] | None,
    dashboard_events: list[dict[str, Any]],
) -> str:
    stage = pipeline_state.get("stage") if pipeline_state else None
    if isinstance(stage, str) and stage:
        return stage

    completed_agents = {
        str(event.get("agentId") or event.get("agent", {}).get("id"))
        for event in dashboard_events
        if event.get("event") == "agent_completed"
    }
    running_agents = {
        str(event.get("agentId") or event.get("agent", {}).get("id"))
        for event in dashboard_events
        if event.get("event") == "agent_started"
    }

    artifact_files = {
        "paper_understood": "paper_claim_map.json",
        "artifacts_discovered": "artifact_index.json",
        "environment_built": "environment_spec.json",
        "plan_created": "reproduction_contract.json",
        "baseline_implemented": "baseline_result.json",
    }
    # Artifact files appear before the first full checkpoint in some runs, so
    # they provide a durable stage hint during the live part of the pipeline.
    for inferred_stage, file_name in reversed(list(artifact_files.items())):
        if (run_dir / file_name).exists():
            return inferred_stage

    if "baseline-implementation" in completed_agents:
        return "baseline_implemented"
    if "reproduction-planner" in completed_agents:
        return "plan_created"
    if "environment-detective" in completed_agents:
        return "environment_built"
    if "artifact-discovery" in completed_agents:
        return "artifacts_discovered"
    if "paper-understanding" in completed_agents:
        return "paper_understood"
    if "paper-understanding" in running_agents:
        return "ingested"
    return "ingested"


def _snapshot_from_dashboard_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    agents: dict[str, dict[str, Any]] = {}
    reasoning: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    progress: list[dict[str, Any]] = []
    data_panels: list[dict[str, Any]] = []
    hermes_panel: dict[str, Any] | None = None
    concept_card: dict[str, Any] | None = None

    for event in events:
        event_type = event.get("event")
        agent = event.get("agent")
        if event_type in {"agent_started", "agent_completed", "agent_failed"} and isinstance(agent, dict):
            agent_id = str(agent.get("id") or event.get("agentId") or "")
            if agent_id:
                agents[agent_id] = agent
        elif event_type == "agent_reasoning_step":
            reasoning.append({
                "id": f"{event.get('agentId', 'agent')}-{len(reasoning)}",
                "agentId": event.get("agentId") or "",
                "agentLabel": event.get("agentLabel") or event.get("agentId") or "Agent",
                "title": event.get("title") or "Reasoning update",
                "detail": event.get("detail") or "",
                "stepType": event.get("stepType") or "analysis",
                "timestamp": event.get("timestamp") or _now(),
                "citations": event.get("citations") or [],
            })
        elif event_type == "shared_state_updated":
            messages.append({
                "id": f"message-{len(messages)}",
                "fromAgentId": event.get("fromAgentId") or event.get("agentId") or "agent",
                "toAgentId": event.get("toAgentId") or "root-orchestrator",
                "summary": event.get("title") or "Shared state updated",
                "detail": event.get("detail") or "",
                "timestamp": event.get("timestamp") or _now(),
            })
        elif event_type == "verification_gate_result":
            progress.append({
                "stage": event.get("stage") or "plan",
                "status": event.get("status") or "pending",
                "detail": event.get("detail") or "",
            })
        elif event_type == "context_enrichment":
            data_panels.append({
                "id": f"context-{event.get('variableName', len(data_panels))}",
                "title": str(event.get("variableName") or "Context"),
                "summary": str(event.get("summary") or ""),
                "items": [str(event.get("agentId") or "")],
            })
        elif event_type == "hermes_check_updated" and isinstance(event.get("panel"), dict):
            hermes_panel = event["panel"]
        elif event_type == "concept_card_updated" and isinstance(event.get("card"), dict):
            concept_card = event["card"]

    return {
        "agents": list(agents.values()),
        "reasoning": reasoning,
        "messages": messages,
        "citations": [],
        "approvals": [],
        "progress": progress,
        "dataPanels": data_panels,
        "hermesPanel": hermes_panel,
        "conceptCard": concept_card,
    }


def _mean_reward(pipeline_state: dict[str, Any] | None) -> float | int | None:
    metrics = ((pipeline_state or {}).get("experiment_artifacts") or {}).get("metrics") or {}
    value = metrics.get("mean_reward")
    return value if isinstance(value, (float, int)) else None


def _run_mode_label(run_mode: Any, provider: Any) -> str:
    if run_mode != "sdk":
        return "Offline"
    if provider == "openai":
        return "SDK: OpenAI"
    if provider == "anthropic":
        return "SDK: Anthropic"
    return "SDK"


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
