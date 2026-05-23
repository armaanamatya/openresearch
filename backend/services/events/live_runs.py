"""Live run process management and SSE event streaming."""

from __future__ import annotations

import asyncio
import collections
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

# ---------------------------------------------------------------------------
# Watchdog configuration
# ---------------------------------------------------------------------------

_ACLOSE_PATTERN = "RuntimeError: aclose(): asynchronous generator is already running"
_WATCHDOG_THRESHOLD = 3          # occurrences required to trigger
_WATCHDOG_WINDOW_SECONDS = 30    # rolling window in seconds
_WATCHDOG_POLL_INTERVAL = 2.0    # seconds between stderr re-reads

from pydantic import BaseModel, Field

from backend.config import get_settings

RunMode = Literal["rlm", "rdr", "rlm-pure"]
Provider = Literal["anthropic", "openai"]
ExecutionMode = Literal["efficient", "max"]
SandboxMode = Literal["auto", "docker", "local", "runpod"]
GpuMode = Literal["off", "auto", "prefer", "max"]
ModelChoice = Literal["sonnet", "opus"]
RunStatus = Literal["queued", "running", "stopped", "completed", "failed"]

_MODEL_IDS: dict[str, str] = {
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-7",
}


class StartRunRequest(BaseModel):
    mode: RunMode = "rlm"
    provider: Provider = "anthropic"
    verificationProvider: Provider | None = None
    executionMode: ExecutionMode = "efficient"
    sandbox: SandboxMode = "runpod"
    gpuMode: GpuMode = "auto"
    model: ModelChoice = "sonnet"
    # rdr-specific: PaperBench bundle paper_id (directory name under
    # third_party/paperbench/ or an absolute path). Only required when
    # mode='rdr'; ignored for other modes.
    paper_id: str | None = None


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


class SourcePdfArtifact(BaseModel):
    fileName: str
    title: str
    sizeBytes: int
    sha256: str
    pageCount: int | None = None
    runPath: str
    codePath: str


class BenchmarkSummary(BaseModel):
    benchmarkName: str
    paperbenchTaskId: str
    overallScore: float
    targetMetric: str
    targetValue: float
    reproducedValue: float
    deltaValue: float
    verdict: str
    reportPath: str
    comparisonPath: str
    logPath: str
    # Track 3 — rubric-verifier comparison. All optional/defaulted so old
    # demo_status.json files (and offline-demo runs) still parse — and, crucially,
    # so these survive the LiveRunState(**status) round-trip instead of being
    # dropped by pydantic's default extra="ignore" before they reach the UI.
    paperbenchBaseline: dict[str, Any] | None = None
    ourRubricScore: float | None = None
    verificationDelta: float | None = None
    improvementIterations: int = 0
    meetsTarget: bool | None = None
    comparisonSummary: str = ""
    rubricAreas: list[dict[str, Any]] = Field(default_factory=list)
    baselineRubricAreas: list[dict[str, Any]] = Field(default_factory=list)


class LiveRunState(BaseModel):
    projectId: str
    outputDir: str
    runMode: RunMode
    llmProvider: Provider | None = None
    verificationProvider: Provider | None = None
    executionMode: ExecutionMode | None = None
    sandboxMode: SandboxMode | None = None
    gpuMode: GpuMode | None = None
    model: ModelChoice | None = None
    status: RunStatus
    sourceKind: Literal["workspace_fixture", "uploaded_pdf"] | None = None
    sourceLabel: str | None = None
    sourceNote: str | None = None
    sourcePdf: SourcePdfArtifact | None = None
    benchmark: BenchmarkSummary | None = None
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


async def _stderr_watchdog(
    project_id: str,
    run_dir: Path,
    pid: int,
) -> None:
    """Tail runner.stderr.log; flag the run as degraded on repeated aclose errors.

    Reads the last N bytes of the stderr log every _WATCHDOG_POLL_INTERVAL
    seconds. Maintains a rolling deque of timestamps for each occurrence of
    _ACLOSE_PATTERN. When _WATCHDOG_THRESHOLD occurrences land within
    _WATCHDOG_WINDOW_SECONDS the run is flagged:
      - demo_status.json: degraded=True, degraded_reason set
      - dashboard_events.jsonl: run_warning event appended

    The watchdog exits when the subprocess terminates (pid gone) or the task
    is cancelled (normal cleanup on run completion).
    """
    stderr_path = run_dir / "runner.stderr.log"
    events_path = run_dir / "dashboard_events.jsonl"
    status_path = run_dir / "demo_status.json"
    timestamps: collections.deque[float] = collections.deque()
    flagged = False
    # Track how many bytes we have already scanned so each pass we only look
    # at the new tail (not the entire file from scratch).
    last_size: int = 0

    while True:
        await asyncio.sleep(_WATCHDOG_POLL_INTERVAL)

        # Exit once the subprocess is gone.
        if not _pid_exists(pid):
            return

        if not stderr_path.exists():
            continue

        try:
            current_size = stderr_path.stat().st_size
        except OSError:
            continue

        if current_size <= last_size:
            continue

        # Read only the newly appended bytes.
        try:
            with stderr_path.open("rb") as fh:
                fh.seek(last_size)
                chunk = fh.read(current_size - last_size).decode("utf-8", errors="replace")
        except OSError:
            continue

        last_size = current_size

        # Count new occurrences in the fresh chunk.
        now = time.monotonic()
        count_in_chunk = chunk.count(_ACLOSE_PATTERN)
        for _ in range(count_in_chunk):
            timestamps.append(now)

        # Evict timestamps older than the window.
        cutoff = now - _WATCHDOG_WINDOW_SECONDS
        while timestamps and timestamps[0] < cutoff:
            timestamps.popleft()

        if flagged or len(timestamps) < _WATCHDOG_THRESHOLD:
            continue

        # Threshold crossed — flag once.
        flagged = True
        iso_now = datetime.now(timezone.utc).isoformat()

        # Update demo_status.json atomically.
        try:
            existing: dict[str, Any] = {}
            if status_path.exists():
                existing = json.loads(status_path.read_text(encoding="utf-8"))
            existing["degraded"] = True
            existing["degraded_reason"] = "SDK aclose loop detected"
            existing["updatedAt"] = iso_now
            tmp = status_path.with_suffix(status_path.suffix + ".tmp")
            tmp.write_text(json.dumps(existing, indent=2), encoding="utf-8")
            os.replace(tmp, status_path)
        except Exception:  # noqa: BLE001
            pass

        # Append run_warning event to dashboard_events.jsonl.
        warning_event: dict[str, Any] = {
            "event": "run_warning",
            "timestamp": iso_now,
            "level": "warn",
            "code": "sdk_aclose_loop",
            "message": (
                "Backend SDK aclose deadlock detected"
                " — run is degraded; consider restarting"
            ),
        }
        try:
            with events_path.open("a", encoding="utf-8") as ef:
                ef.write(json.dumps(warning_event) + "\n")
        except Exception:  # noqa: BLE001
            pass


def apply_sandbox_override(request: StartRunRequest, force_sandbox: str) -> StartRunRequest:
    """Force the sandbox mode for all runs when REPROLAB_FORCE_SANDBOX is set.

    Deployments without a GPU or Docker daemon (e.g. Railway) must pin every
    run to ``local`` regardless of what the client requested. An empty force
    value leaves the request unchanged.
    """
    if not force_sandbox:
        return request
    return request.model_copy(update={"sandbox": force_sandbox})


def apply_provider_override(request: StartRunRequest, force_provider: str) -> StartRunRequest:
    """Force the LLM provider for all runs when REPROLAB_FORCE_LLM_PROVIDER is set.

    The UI hard-codes ``provider="anthropic"`` in its start-run requests; on
    deployments where the operator only has credentials for the other side,
    rewriting at the request edge avoids an unconfigured-provider failure
    mid-pipeline. Mirrors apply_sandbox_override. An empty or invalid value
    leaves the request unchanged.
    """
    if force_provider not in ("anthropic", "openai"):
        return request
    return request.model_copy(update={"provider": force_provider})


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

    async def resume_run(
        self,
        project_id: str,
        *,
        request_overrides: dict[str, Any] | None = None,
    ) -> LiveRunState | None:
        """Re-spawn an orchestrator subprocess for an existing project_id.

        The orchestrator's ``run(resume=True)`` is the default and auto-
        resumes from the persisted pipeline_state.json checkpoint, so the
        subprocess picks up at the last completed stage rather than
        restarting from scratch. The original run config is read back from
        the existing demo_status.json; ``request_overrides`` lets callers
        bump knobs like ``executionMode=max`` to push past whatever caused
        the original failure (e.g. a wall-clock timeout on baseline-
        implementation).

        Returns None if the project doesn't exist. Refuses to re-spawn
        if the original process is still alive.
        """
        existing = await self.get_run(project_id)
        if existing is None:
            return None
        if _pid_exists(existing.pid):
            return existing
        status = await asyncio.to_thread(self._read_status, project_id)
        if status is None:
            return None
        merged = {
            "mode": status.get("runMode", "rlm"),
            "provider": status.get("llmProvider", "anthropic"),
            "verificationProvider": status.get("verificationProvider"),
            "executionMode": status.get("executionMode", "efficient"),
            "sandbox": status.get("sandboxMode", get_settings().default_sandbox),
            "gpuMode": status.get("gpuMode", "auto"),
            "model": status.get("model", "sonnet"),
        }
        if request_overrides:
            for key in (
                "mode", "provider", "verificationProvider",
                "executionMode", "sandbox", "gpuMode", "model",
            ):
                value = request_overrides.get(key)
                if value is not None:
                    merged[key] = value
        request = StartRunRequest(**{k: v for k, v in merged.items() if v is not None})
        # Re-derive uploaded_paper from the persisted source-pdf record so
        # _start_python_run can hand the same artifact back to the orchestrator
        # without re-staging the upload.
        uploaded_paper: dict[str, str] | None = None
        src_pdf = status.get("sourcePdf") or {}
        if status.get("sourceKind") == "uploaded_pdf" and src_pdf.get("runPath"):
            uploaded_paper = {
                "path": str(src_pdf["runPath"]),
                "fileName": str(src_pdf.get("fileName") or "paper.pdf"),
            }
        return await self._start_python_run(
            request,
            project_id=project_id,
            uploaded_paper=uploaded_paper,
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

    async def get_source_pdf_path(self, project_id: str) -> Path | None:
        return await asyncio.to_thread(self._source_pdf_path, project_id)

    async def get_final_report_path(self, project_id: str) -> Path | None:
        return await asyncio.to_thread(self._final_report_path, project_id)

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

    async def list_runs(
        self,
        *,
        limit: int = 10,
        status: str | None = None,
        q: str | None = None,
        order_by: str = "updated_at",
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._list_runs, limit, status, q, order_by)

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

        # A4-10: mild backoff after the first 60 s in a running state —
        # 1 s for the first minute, then grows towards ~5 s to avoid
        # hammering the file system on long-running jobs.
        _running_secs: float = 0.0
        while state.status in {"queued", "running"}:
            _poll_interval = 1.0 if _running_secs < 60.0 else min(5.0, 1.0 + (_running_secs - 60.0) / 60.0)
            await asyncio.sleep(_poll_interval)
            if state.status == "running":
                _running_secs += _poll_interval
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
        _s = get_settings()
        request = apply_sandbox_override(request, _s.force_sandbox)
        request = apply_provider_override(request, _s.force_llm_provider)
        existing = await self.get_run(project_id)
        if existing and existing.status in {"queued", "running"} and _pid_exists(existing.pid):
            return existing

        output_dir = self.runs_root / project_id
        output_dir.mkdir(parents=True, exist_ok=True)
        source_pdf, benchmark = await asyncio.to_thread(
            self._prepare_source_artifacts,
            request,
            project_id,
            output_dir,
            uploaded_paper,
        )
        meta = _initial_status(
            request,
            project_id=project_id,
            output_dir=output_dir,
            uploaded_paper=uploaded_paper,
            source_pdf=source_pdf,
            benchmark=benchmark,
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
                    "REPROLAB_LLM_PROVIDER": request.provider,
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

        # Launch the stderr watchdog as a fire-and-forget asyncio task.
        # It self-terminates when the subprocess exits (pid gone) and is also
        # cancelled by the event loop on normal shutdown — no extra cleanup needed.
        asyncio.get_event_loop().create_task(
            _stderr_watchdog(project_id, output_dir, process.pid),
            name=f"stderr-watchdog-{project_id}",
        )

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

    def _list_runs(
        self,
        limit: int,
        status_filter: str | None,
        q: str | None = None,
        order_by: str = "updated_at",
    ) -> list[dict[str, Any]]:
        """Walk runs_root for demo_status.json files; return at most `limit`,
        newest first by the chosen timestamp with file mtime as a tiebreaker.

        `q` is a case-insensitive substring match against `sourceLabel`.
        Rows whose `sourceLabel` is missing or empty are excluded when `q`
        is provided — an absent label cannot contain the query.

        `order_by` selects the sort key:
          - "updated_at" (default): `updatedAt` → `startedAt` → mtime
          - "completed_at": `completedAt` → 0 (still-running rows sort last)
        Any other value falls back to "updated_at"."""
        if not self.runs_root.exists():
            return []
        order_key = order_by if order_by in ("updated_at", "completed_at") else "updated_at"
        needle = q.lower().strip() if isinstance(q, str) and q.strip() else None
        entries: list[tuple[float, dict[str, Any]]] = []
        for status_path in self.runs_root.glob("*/demo_status.json"):
            data = _read_json(status_path)
            if not data:
                continue
            if status_filter is not None and data.get("status") != status_filter:
                continue
            if needle is not None:
                label = data.get("sourceLabel")
                if not isinstance(label, str) or needle not in label.lower():
                    continue
            if order_key == "completed_at":
                # Treat missing completedAt as 0 → those rows sort last
                # (matches the "still-running goes to the bottom" intent).
                timestamp = _parse_time(data.get("completedAt"))
            else:
                timestamp = _parse_time(data.get("updatedAt") or data.get("startedAt"))
                if timestamp == 0.0:
                    try:
                        timestamp = status_path.stat().st_mtime
                    except OSError:
                        pass
            entries.append((timestamp, data))
        entries.sort(key=lambda item: item[0], reverse=True)
        return [data for _, data in entries[: max(0, limit)]]

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

    def _prepare_source_artifacts(
        self,
        request: StartRunRequest,
        project_id: str,
        output_dir: Path,
        uploaded_paper: dict[str, str] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        code_dir = output_dir / "code"
        logs_dir = code_dir / "logs"
        code_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        code_pdf = code_dir / "paper.pdf"
        raw_pdf = output_dir / "raw_paper.pdf"
        if uploaded_paper:
            source_path = Path(uploaded_paper["path"])
            display_name = uploaded_paper["fileName"]
            title = Path(display_name).stem.replace("_", " ").replace("-", " ").strip() or "Uploaded paper"
            title = title[:1].upper() + title[1:] if title else "Uploaded paper"
            if source_path.exists():
                shutil.copyfile(source_path, code_pdf)
                shutil.copyfile(source_path, raw_pdf)
            else:
                _write_minimal_pdf(code_pdf, title=title)
                shutil.copyfile(code_pdf, raw_pdf)
        else:
            display_name = "reprolab-demo-paper.pdf"
            title = "ReproLab PPO Reproducibility Demo"
            fixture_pdf = self._fixture_pdf_path()
            if fixture_pdf is not None:
                shutil.copyfile(fixture_pdf, code_pdf)
                shutil.copyfile(fixture_pdf, raw_pdf)
            else:
                _write_minimal_pdf(code_pdf, title=title)
                shutil.copyfile(code_pdf, raw_pdf)

        source_pdf = {
            "fileName": display_name,
            "title": title,
            "sizeBytes": code_pdf.stat().st_size,
            "sha256": _file_sha256(code_pdf),
            "pageCount": _pdf_page_count(code_pdf),
            "runPath": str(raw_pdf.resolve()),
            "codePath": str(code_pdf.resolve()),
        }
        benchmark = self._write_demo_codebase_artifacts(
            request=request,
            project_id=project_id,
            code_dir=code_dir,
            source_pdf=source_pdf,
            uploaded=uploaded_paper is not None,
        )
        return source_pdf, benchmark

    def _fixture_pdf_path(self) -> Path | None:
        for name in ("demo_paper.pdf", "paperbench1.pdf"):
            candidate = self.repo_root / name
            if candidate.exists() and candidate.is_file():
                return candidate
        return None

    def _write_demo_codebase_artifacts(
        self,
        *,
        request: StartRunRequest,
        project_id: str,
        code_dir: Path,
        source_pdf: dict[str, Any],
        uploaded: bool,
    ) -> dict[str, Any]:
        logs_dir = code_dir / "logs"
        report_path = code_dir / "final_benchmark_report.md"
        comparison_path = code_dir / "paperbench_comparison.json"
        log_path = logs_dir / "paperbench_eval.log"
        manifest_path = code_dir / "reprolab_manifest.json"
        readme_path = code_dir / "README.md"

        benchmark = {
            "benchmarkName": "PaperBench-style final benchmark",
            "paperbenchTaskId": "reprolab-demo/ppo-cartpole-v1",
            "overallScore": 91.4 if not uploaded else 0.0,
            "targetMetric": "mean_reward",
            "targetValue": 475.0,
            "reproducedValue": 492.3 if not uploaded else 0.0,
            "deltaValue": 17.3 if not uploaded else 0.0,
            "verdict": "reproduced_with_caveats" if not uploaded else "pending_pipeline_result",
            "reportPath": str(report_path.resolve()),
            "comparisonPath": str(comparison_path.resolve()),
            "logPath": str(log_path.resolve()),
        }

        comparison = {
            "project_id": project_id,
            "benchmark": benchmark["benchmarkName"],
            "paperbench_task_id": benchmark["paperbenchTaskId"],
            "run_mode": request.mode,
            "execution_profile": request.executionMode,
            "source": source_pdf,
            "claim": {
                "metric": "mean_reward",
                "target": 475.0,
                "environment": "CartPole-v1",
                "evaluation_protocol": "100 deterministic evaluation episodes after PPO training",
            },
            "result": {
                "metric": "mean_reward",
                "value": benchmark["reproducedValue"],
                "delta_vs_target": benchmark["deltaValue"],
                "status": benchmark["verdict"],
            },
            "rubric": [
                {"area": "paper_understanding", "score": 0.96, "evidence": "paper_claim_map.json"},
                {"area": "environment_reconstruction", "score": 0.92, "evidence": "Dockerfile"},
                {"area": "baseline_implementation", "score": 0.91, "evidence": "train.py"},
                {"area": "execution_artifacts", "score": 0.88, "evidence": "metrics.json, commands.log, provenance.json"},
                {"area": "comparison_quality", "score": 0.90, "evidence": "final_benchmark_report.md"},
            ],
            "artifact_expectations": [
                "paper.pdf",
                "train.py",
                "config.json",
                "Dockerfile",
                "commands.log",
                "metrics.json",
                "provenance.json",
                "final_benchmark_report.md",
            ],
        }

        comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        report_path.write_text(_benchmark_report_markdown(project_id, benchmark, source_pdf, uploaded), encoding="utf-8")
        log_path.write_text(_paperbench_log(project_id, benchmark, uploaded), encoding="utf-8")
        readme_path.write_text(_generated_codebase_readme(project_id, source_pdf, uploaded), encoding="utf-8")
        manifest_path.write_text(
            json.dumps(
                {
                    "project_id": project_id,
                    "source_pdf": source_pdf,
                    "benchmark": benchmark,
                    "root_files": [
                        "paper.pdf",
                        "README.md",
                        "reprolab_manifest.json",
                        "paperbench_comparison.json",
                        "final_benchmark_report.md",
                    ],
                    "log_files": ["logs/paperbench_eval.log"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return benchmark

    def _source_pdf_path(self, project_id: str) -> Path | None:
        run_dir = (self.runs_root / project_id).resolve()
        status = self._read_status(project_id) or {}
        candidates: list[Path] = []
        source_pdf = status.get("sourcePdf")
        if isinstance(source_pdf, dict):
            for key in ("codePath", "runPath"):
                value = source_pdf.get(key)
                if isinstance(value, str):
                    candidates.append(Path(value))
        candidates.extend([run_dir / "code" / "paper.pdf", run_dir / "raw_paper.pdf"])

        for candidate in candidates:
            try:
                resolved = candidate.expanduser().resolve()
            except OSError:
                continue
            if not _is_relative_to(resolved, run_dir):
                continue
            if resolved.exists() and resolved.is_file():
                return resolved
        return None

    def _final_report_path(self, project_id: str) -> Path | None:
        run_dir = (self.runs_root / project_id).resolve()
        status = self._read_status(project_id) or {}
        candidates: list[Path] = []
        benchmark = status.get("benchmark")
        if isinstance(benchmark, dict):
            value = benchmark.get("reportPath")
            if isinstance(value, str):
                candidates.append(Path(value))
        candidates.extend([
            run_dir / "code" / "final_benchmark_report.md",
            run_dir / "final_report.md",
        ])

        for candidate in candidates:
            try:
                resolved = candidate.expanduser().resolve()
            except OSError:
                continue
            if not _is_relative_to(resolved, run_dir):
                continue
            if resolved.exists() and resolved.is_file():
                return resolved
        return None

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
            "runMode": status.get("runMode") or "rlm",
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
                "runModeLabel": _run_mode_label(meta.get("llmProvider")),
                "llmProvider": meta.get("llmProvider"),
                "verificationProvider": meta.get("verificationProvider"),
                "executionMode": meta.get("executionMode"),
                "sandboxMode": meta.get("sandboxMode"),
                "gpuMode": meta.get("gpuMode"),
                "sourceLabel": meta["sourceLabel"],
            },
        }

    def _read_log(self, project_id: str, max_chars: int = 12000) -> str:
        stdout_path = self.runs_root / project_id / "runner.stdout.log"
        stderr_path = self.runs_root / project_id / "runner.stderr.log"
        stdout_text = (
            stdout_path.read_text(encoding="utf-8", errors="replace")
            if stdout_path.exists()
            else None
        )
        stderr_text = (
            stderr_path.read_text(encoding="utf-8", errors="replace")
            if stderr_path.exists()
            else None
        )
        if stdout_text is None and stderr_text is None:
            return ""
        if stdout_text is None:
            return stderr_text[-max_chars:]
        if stderr_text is None:
            return stdout_text[-max_chars:]
        # Tail-cap each stream independently so a noisy stderr cannot
        # evict the agent's stdout entirely — surfacing stdout is the
        # whole point of reading both files.
        half = max_chars // 2
        return (
            stdout_text[-half:]
            + "\n--- runner.stderr.log ---\n"
            + stderr_text[-half:]
        )

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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _pdf_page_count(path: Path) -> int | None:
    try:
        import fitz  # type: ignore[import-not-found]

        with fitz.open(path) as doc:
            return int(doc.page_count)
    except Exception:
        return None


def _write_minimal_pdf(path: Path, *, title: str) -> None:
    escaped_title = title.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 18 Tf 72 720 Td ({escaped_title}) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(stream.encode('latin-1'))} >>\nstream\n{stream}\nendstream".encode("latin-1"),
    ]
    body = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(body))
        body.extend(f"{index} 0 obj\n".encode("ascii"))
        body.extend(obj)
        body.extend(b"\nendobj\n")
    xref_offset = len(body)
    body.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    body.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        body.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    body.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(bytes(body))


def _benchmark_report_markdown(
    project_id: str,
    benchmark: dict[str, Any],
    source_pdf: dict[str, Any],
    uploaded: bool,
) -> str:
    status_note = (
        "This uploaded-paper run is staged for the live pipeline; the comparison file will be "
        "replaced by measured values once the run completes."
        if uploaded
        else "The hardcoded demo ships with a deterministic PaperBench-style comparison so the "
        "UI has a realistic final benchmark surface before a long live run finishes."
    )
    delta = benchmark["deltaValue"]
    delta_text = f"{delta:+.1f}"
    return f"""# Final Benchmark Report

**Project:** `{project_id}`  
**Benchmark:** {benchmark["benchmarkName"]}  
**Task:** `{benchmark["paperbenchTaskId"]}`  
**Verdict:** `{benchmark["verdict"]}`

{status_note}

## Source Artifact

| Field | Value |
| --- | --- |
| PDF | `{source_pdf["fileName"]}` |
| Stored in generated code root | `paper.pdf` |
| Pages | {source_pdf.get("pageCount") or "unknown"} |
| Size | {source_pdf["sizeBytes"]} bytes |
| SHA256 | `{source_pdf["sha256"]}` |

## Final Metric Comparison

| Metric | Paper target | Reproduced value | Delta |
| --- | ---: | ---: | ---: |
| {benchmark["targetMetric"]} | {benchmark["targetValue"]:.1f} | {benchmark["reproducedValue"]:.1f} | {delta_text} |

## PaperBench-Style Rubric

| Area | Score | Evidence |
| --- | ---: | --- |
| Paper understanding | 0.96 | `paper_claim_map.json` |
| Environment reconstruction | 0.92 | `Dockerfile` |
| Baseline implementation | 0.91 | `train.py` |
| Execution artifacts | 0.88 | `metrics.json`, `commands.log`, `provenance.json` |
| Comparison quality | 0.90 | `final_benchmark_report.md` |

## Generated Codebase Root

The generated code root is designed to be inspectable without the dashboard:

```text
code/
  paper.pdf
  README.md
  Dockerfile
  train.py
  config.json
  commands.log
  paperbench_comparison.json
  final_benchmark_report.md
  logs/paperbench_eval.log
```
"""


def _paperbench_log(project_id: str, benchmark: dict[str, Any], uploaded: bool) -> str:
    if uploaded:
        result_line = "pending measured result; waiting for pipeline artifacts"
    else:
        result_line = (
            f"mean_reward={benchmark['reproducedValue']:.1f}, "
            f"target={benchmark['targetValue']:.1f}, delta={benchmark['deltaValue']:+.1f}"
        )
    return "\n".join(
        [
            "2026-05-10T09:30:12Z paperbench-eval INFO loaded task reprolab-demo/ppo-cartpole-v1",
            f"2026-05-10T09:30:13Z paperbench-eval INFO project={project_id}",
            "2026-05-10T09:30:14Z paperbench-eval INFO validating source artifact code/paper.pdf",
            "2026-05-10T09:30:15Z paperbench-eval INFO checking generated code root manifest",
            "2026-05-10T09:30:16Z paperbench-eval INFO replaying command log and provenance refs",
            f"2026-05-10T09:30:19Z paperbench-eval INFO final metric comparison: {result_line}",
            f"2026-05-10T09:30:20Z paperbench-eval INFO verdict={benchmark['verdict']}",
            "",
        ]
    )


def _generated_codebase_readme(
    project_id: str,
    source_pdf: dict[str, Any],
    uploaded: bool,
) -> str:
    source_mode = "uploaded lab paper" if uploaded else "built-in ReproLab demo paper"
    return f"""# Generated Reproduction Codebase

This directory is the generated code root for `{project_id}`.

## Source

- Source mode: {source_mode}
- Paper: `{source_pdf["fileName"]}`
- Stable source copy: `paper.pdf`
- SHA256: `{source_pdf["sha256"]}`

## Run Surface

The pipeline writes implementation files, benchmark comparisons, logs, and provenance into this
directory so the run can be reviewed outside the UI.

```bash
python train.py
```

## Review Artifacts

- `paperbench_comparison.json` - structured benchmark comparison
- `final_benchmark_report.md` - human-readable benchmark report
- `logs/paperbench_eval.log` - PaperBench-style evaluator log
- `reprolab_manifest.json` - source and artifact manifest
"""


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def finalize_benchmark(run_dir: Path) -> dict[str, Any]:
    """Build the benchmark summary from final_report.json, RLM- or SDK-schema-aware.

    Returns ``{"benchmark": <dict>}`` on success or ``{"benchmark": None}`` when
    the report is missing or unparseable.  The two schemas are auto-detected:

    * **RLM** — presence of ``"baseline_metrics"`` or ``"verdict"`` (without the
      SDK-only ``"rubric_overall_score"`` float key).
    * **SDK** — presence of ``"rubric_overall_score"`` or other SDK-only keys.
    """
    report_path = run_dir / "final_report.json"
    if not report_path.exists():
        return {"benchmark": None}
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError):
        return {"benchmark": None}
    if not isinstance(report, dict):
        return {"benchmark": None}

    # RLM schema detector: has "verdict" string + "rubric" dict but NOT the
    # SDK-only "rubric_overall_score" float key.
    is_rlm = (
        isinstance(report.get("verdict"), str)
        and isinstance(report.get("rubric"), dict)
        and "rubric_overall_score" not in report
    ) or "baseline_metrics" in report

    if is_rlm:
        rubric = report.get("rubric") or {}
        cost = report.get("cost") or {}
        return {
            "benchmark": {
                "verdict": report.get("verdict", ""),
                "rubric_score": rubric.get("overall_score"),
                "metrics": report.get("baseline_metrics") or {},
                "cost_usd": cost.get("llm_usd"),
            }
        }

    # Legacy SDK schema — preserve existing behaviour.
    rv = report.get("rubric_verification") or {}
    base_rv = report.get("baseline_rubric_verification") or {}
    return {
        "benchmark": {
            "overallScore": round((report.get("rubric_overall_score") or 0.0) * 100, 1),
            "targetMetric": report.get("primary_metric"),
            "targetValue": report.get("paper_primary_target"),
            "reproducedValue": report.get("reproduction_primary_value"),
            "deltaValue": report.get("reproduction_delta_vs_paper"),
            "verdict": report.get("reproduction_status"),
            "reproductionScore": report.get("reproduction_score"),
            "rubricOverallScore": report.get("rubric_overall_score"),
            "bestPathId": report.get("best_path_id"),
            "bestImprovementPct": report.get("best_overall_improvement_pct"),
            "paperbenchBaseline": report.get("paperbench_baseline"),
            "ourRubricScore": rv.get("overall_score"),
            "verificationDelta": report.get("verification_delta"),
            "improvementIterations": report.get("improvement_iterations") or 0,
            "meetsTarget": rv.get("meets_target"),
            "comparisonSummary": report.get("comparison_summary") or "",
            "rubricAreas": rv.get("areas") or [],
            "baselineRubricAreas": base_rv.get("areas") or [],
            "source": "computed_final_report",
        }
    }


def _fixture_project_id(request: StartRunRequest) -> str:
    # A4-4: rlm runs get a distinct prefix so the provider is recoverable
    # from the project_id and IDs cannot collide with legacy run IDs.
    return f"ui_rlm_{request.provider}_{int(datetime.now().timestamp() * 1000)}"


def _initial_status(
    request: StartRunRequest,
    *,
    project_id: str,
    output_dir: Path,
    uploaded_paper: dict[str, str] | None,
    source_pdf: dict[str, Any],
    benchmark: dict[str, Any],
) -> dict[str, Any]:
    now = _now()
    status: dict[str, Any] = {
        "projectId": project_id,
        "outputDir": str(output_dir),
        "runMode": request.mode,
        "executionMode": request.executionMode,
        "sandboxMode": request.sandbox,
        "gpuMode": request.gpuMode,
        "model": request.model,
        "sourcePdf": source_pdf,
        "benchmark": benchmark,
        "status": "queued",
        "startedAt": now,
        "updatedAt": now,
    }
    if uploaded_paper:
        status.update(
            {
                "sourceKind": "uploaded_pdf",
                "sourceLabel": uploaded_paper["fileName"],
                "sourceNote": (
                    "This run started from a PDF uploaded directly in the lab. "
                    "The backend copied it into the generated code root as paper.pdf before running reproduction."
                ),
            }
        )
    else:
        status.update(
            {
                "sourceKind": "workspace_fixture",
                "sourceLabel": "ReproLab PPO demo paper",
                "sourceNote": (
                    "This demo uses a checked-in PPO-style paper PDF, a deterministic generated "
                    "codebase, and a PaperBench-style final benchmark comparison."
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
    _settings = get_settings()
    if request.provider == "openai":
        _model_id = (
            _settings.openai_reasoning_model
            if request.model == "opus"
            else _settings.openai_default_model
        )
    else:
        _model_id = (
            _settings.anthropic_reasoning_model
            if request.model == "opus"
            else _settings.anthropic_default_model
        )
    common = {
        "project_id": project_id,
        "run_mode": request.mode,
        "provider": request.provider,
        "verification_provider": request.verificationProvider,
        "execution_mode": request.executionMode,
        "sandbox": request.sandbox,
        "gpu_mode": request.gpuMode,
        "model": _model_id,
        "runs_root": str(runs_root),
        "database_url": _settings.database_url,
        "uploaded_paper": uploaded_paper,
        # A4-7: budget fields threaded through so an API-set budget is honored
        # in the non-uploaded rlm path (run_pipeline_rlm call below).
        "max_usd": None,
        "max_wall_clock": None,
        "max_pod_seconds": None,
        "max_invocations": None,
        # rdr-specific: PaperBench bundle identifier (only used when mode='rdr').
        "paper_id": request.paper_id if request.mode == "rdr" else None,
        "max_repair_iterations": 2,
        "repair_target": 0.6,
    }
    return f"""
import asyncio
import json
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

from backend.agents.execution import ExecutionProfile, SandboxMode
from backend.agents.rlm.run import run_pipeline_rlm
from backend.cli import (
    cmd_reproduce,
    _REPRODUCE_DEFAULTS,
    _max_invocations_from_arg,
    _resolve_max_pod_seconds,
)

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
        {{"source_id": "src_2", "title": "Experiments", "excerpt": "We test on CartPole-v1 environment. We use Adam optimizer with learning rate 3e-4 and batch size 64. We report a mean reward of 475.0 over 100 episodes after 500000 timesteps."}},
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
    # A4-1: atomic write — crash mid-write leaves either old or new JSON,
    # never a half-written file that permanently breaks status reads.
    import os as _os
    _tmp = status_path.with_suffix(status_path.suffix + ".tmp")
    _tmp.write_text(json.dumps(payload, indent=2))
    _os.replace(_tmp, status_path)

def _is_rlm_report(fr):
    # RLM-shaped: has "verdict" string and "rubric" dict but NOT the SDK-only
    # "rubric_overall_score" key (which is a float, not a nested dict).
    return (
        isinstance(fr.get("verdict"), str)
        and isinstance(fr.get("rubric"), dict)
        and "rubric_overall_score" not in fr
    )

def finalize_benchmark():
    # Replace the staged benchmark placeholder with the measured values from the
    # pipeline's computed final_report.json (rubric, statistics, paper deltas).
    try:
        if config["uploaded_paper"]:
            from backend.services.events.live_runs import project_id_for_pdf_path
            report_dir = runs_root / project_id_for_pdf_path(Path(config["uploaded_paper"]["path"]))
        else:
            report_dir = output_dir
        report_json = report_dir / "final_report.json"
        report_md = report_dir / "final_report.md"
        if not report_json.exists():
            return
        fr = json.loads(report_json.read_text())
        # Mirror the canonical report into the demo run dir so the report viewer
        # and the backend's _final_report_path resolve the measured version.
        if report_dir != output_dir:
            (output_dir / "final_report.json").write_text(report_json.read_text())
            if report_md.exists():
                (output_dir / "final_report.md").write_text(report_md.read_text())
            # Also bridge dashboard events written by the RLM run so streaming
            # consumers see them from the demo run dir.
            src_dash = report_dir / "dashboard_events.jsonl"
            if src_dash.exists():
                import shutil as _shutil
                _shutil.copy2(src_dash, output_dir / "dashboard_events.jsonl")
        existing = json.loads(status_path.read_text()) if status_path.exists() else {{}}
        bench = dict(existing.get("benchmark") or {{}})
        if _is_rlm_report(fr):
            # RLM-shaped report: fields are verdict, rubric.overall_score,
            # baseline_metrics, reproduction_summary, cost, iterations.
            rubric = fr.get("rubric") or {{}}
            overall_score = rubric.get("overall_score") or 0.0
            baseline = fr.get("baseline_metrics") or {{}}
            # Derive a single representative metric value if any metrics exist.
            first_metric_key = next(iter(baseline), None)
            first_metric_val = baseline.get(first_metric_key) if first_metric_key else None
            bench.update({{
                "overallScore": round(overall_score * 100, 1),
                "verdict": fr.get("verdict") or bench.get("verdict"),
                "targetMetric": first_metric_key or bench.get("targetMetric"),
                "reproducedValue": first_metric_val,
                "reportPath": str((output_dir / "final_report.md").resolve()),
                "comparisonPath": str((output_dir / "final_report.json").resolve()),
                "source": "computed_final_report_rlm",
                "ourRubricScore": overall_score,
                "meetsTarget": rubric.get("meets_target"),
                "rubricAreas": rubric.get("areas") or [],
                "improvementIterations": fr.get("iterations") or 0,
                "comparisonSummary": fr.get("reproduction_summary") or "",
            }})
        else:
            # SDK-shaped report: original field mapping.
            rv = fr.get("rubric_verification") or {{}}
            base_rv = fr.get("baseline_rubric_verification") or {{}}
            bench.update({{
                "overallScore": round((fr.get("rubric_overall_score") or 0.0) * 100, 1),
                "targetMetric": fr.get("primary_metric") or bench.get("targetMetric"),
                "targetValue": fr.get("paper_primary_target"),
                "reproducedValue": fr.get("reproduction_primary_value"),
                "deltaValue": fr.get("reproduction_delta_vs_paper"),
                "verdict": fr.get("reproduction_status") or bench.get("verdict"),
                "reproductionScore": fr.get("reproduction_score"),
                "rubricOverallScore": fr.get("rubric_overall_score"),
                "bestPathId": fr.get("best_path_id"),
                "bestImprovementPct": fr.get("best_overall_improvement_pct"),
                "reportPath": str((output_dir / "final_report.md").resolve()),
                "comparisonPath": str((output_dir / "final_report.json").resolve()),
                "source": "computed_final_report",
                "paperbenchBaseline": fr.get("paperbench_baseline"),
                "ourRubricScore": rv.get("overall_score"),
                "verificationDelta": fr.get("verification_delta"),
                "improvementIterations": fr.get("improvement_iterations") or 0,
                "meetsTarget": rv.get("meets_target"),
                "comparisonSummary": fr.get("comparison_summary") or "",
                "rubricAreas": rv.get("areas") or [],
                "baselineRubricAreas": base_rv.get("areas") or [],
            }})
        existing["benchmark"] = bench
        # A4-1: atomic write — same guard as write_status above.
        import os as _os
        _tmp = status_path.with_suffix(status_path.suffix + ".tmp")
        _tmp.write_text(json.dumps(existing, indent=2))
        _os.replace(_tmp, status_path)
    except Exception:
        pass

write_status("running")

try:
    if config["uploaded_paper"]:
        # A4-5: build from the full _REPRODUCE_DEFAULTS set so a missing key
        # cannot AttributeError at runtime when cmd_reproduce accesses it.
        exit_code = cmd_reproduce(Namespace(**{{
            **_REPRODUCE_DEFAULTS,
            "source": config["uploaded_paper"]["path"],
            "source_kind": "pdf_path",
            "agent": "default",
            "mode": config["run_mode"],
            "model": config["model"],
            "provider": config["provider"],
            "verification_provider": None,
            "execution_mode": config["execution_mode"],
            "sandbox": config["sandbox"],
            "gpu_mode": config["gpu_mode"],
            "hints": "Keep this as a lightweight smoke test",
            "n_paths": 1,
            "runs_root": config["runs_root"],
            "database_url": config["database_url"],
            "max_usd": config["max_usd"],
            "max_wall_clock": config["max_wall_clock"],
            "max_pod_seconds": config["max_pod_seconds"],
            "max_invocations": config["max_invocations"],
        }}))
        if exit_code != 0:
            raise RuntimeError(f"Pipeline exited with status {{exit_code}}")
    else:
        profile = ExecutionProfile.from_mode(config["execution_mode"], gpu_mode=config["gpu_mode"])
        # A4-7: build run_budget once from threaded-through config fields so
        # an API-set budget is honored on both the rlm and rdr paths.
        _run_budget = None
        _max_pod_seconds = _resolve_max_pod_seconds(config["max_pod_seconds"])
        _max_invocations = _max_invocations_from_arg(config["max_invocations"])
        if (
            config["max_usd"] is not None
            or config["max_wall_clock"] is not None
            or _max_pod_seconds is not None
            or _max_invocations
        ):
            from backend.agents.resilience import RunBudget
            _run_budget = RunBudget(
                max_usd=config["max_usd"],
                max_wall_clock_seconds=config["max_wall_clock"],
                max_pod_seconds=_max_pod_seconds,
                max_invocations_per_agent=_max_invocations,
            )
        if config["run_mode"] == "rlm":
            # Default: hybrid Phase 1 (RDR) + Phase 2 (RLM adaptive repair).
            from backend.agents.hybrid.controller import run_pipeline_hybrid
            asyncio.run(run_pipeline_hybrid(
                project_id,
                runs_root,
                DEMO_WORKSPACE,
                provider=config["provider"],
                model=config["model"],
                execution_profile=profile,
                sandbox_mode=SandboxMode(config["sandbox"]),
                run_budget=_run_budget,
            ))
        elif config["run_mode"] == "rlm-pure":
            # Escape hatch: pure RLM, no rubric decomposition.
            asyncio.run(run_pipeline_rlm(
                project_id,
                runs_root,
                DEMO_WORKSPACE,
                provider=config["provider"],
                model=config["model"],
                execution_profile=profile,
                sandbox_mode=SandboxMode(config["sandbox"]),
                run_budget=_run_budget,
            ))
        elif config["run_mode"] == "rdr":
            # rdr: rubric-driven harness on a PaperBench bundle.
            # paper_id is the bundle directory name (e.g. "sequential-neural-score-estimation")
            # or an absolute path to the bundle.
            _paper_id = config.get("paper_id") or ""
            if not _paper_id:
                raise ValueError("mode='rdr' requires paper_id in the run request")
            from backend.agents.rdr.run import run_pipeline_rdr
            from backend.agents.execution import resolve_sandbox_mode
            _sandbox = resolve_sandbox_mode(config["sandbox"], pipeline_mode="rdr")
            asyncio.run(run_pipeline_rdr(
                project_id,
                runs_root,
                paper_id=_paper_id,
                provider=config["provider"],
                model=config["model"],
                sandbox_mode=_sandbox,
                max_repair_iterations=config.get("max_repair_iterations") or 2,
                repair_target=config.get("repair_target") or 0.6,
                run_budget=_run_budget,
            ))
        else:
            raise ValueError(
                f"run_mode={{config['run_mode']!r}} is not supported. "
                "Use 'rlm', 'rlm-pure', or 'rdr'."
            )
    # 2026-05-23: write "completed" BEFORE finalize_benchmark so a downstream
    # hang (claude-agent-sdk atexit subprocess.wait on WSL2 — Defect 2 in
    # docs/superpowers/specs/2026-05-22-sdk-aclose-investigation.md) cannot
    # leave the UI stuck on "running" with a fully-written final_report.json
    # already on disk. The prj_6b9acbfd8afcd789 bug: pipeline returned cleanly,
    # rubric 0.244, but demo_status stayed "running" because finalize_benchmark
    # or atexit cleanup wedged afterward.
    write_status("completed", completed_at=now())
    try:
        finalize_benchmark()
    except Exception:
        import traceback as _tb_ok
        _tb_ok.print_exc()
except Exception as exc:
    write_status("failed", error=f"{{type(exc).__name__}}: {{exc}}", completed_at=now())
    import traceback as _tb_err
    _tb_err.print_exc()
finally:
    # Bypass atexit hooks — claude-agent-sdk's subprocess.wait() can hang in
    # futex_wait_queue on WSL2 after SIGKILL. status is already on disk via the
    # branches above, so os._exit is safe and prevents indefinite "running" UI.
    import os as _os_exit, sys as _sys_exit, json as _json_exit
    _sys_exit.stdout.flush()
    _sys_exit.stderr.flush()
    _final_status = "failed"
    try:
        _final_status = _json_exit.loads(status_path.read_text()).get("status", "failed")
    except Exception:
        pass
    _os_exit._exit(0 if _final_status == "completed" else 1)
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
    # A4-4: rlm runs use ui_rlm_<provider>_<ms> — extract provider from the prefix.
    if project_id.startswith("ui_rlm_openai_"):
        return "openai"
    if project_id.startswith("ui_rlm_anthropic_"):
        return "anthropic"
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


def _run_mode_label(provider: Any) -> str:
    if provider == "openai":
        return "RLM: OpenAI"
    if provider == "anthropic":
        return "RLM: Anthropic"
    return "RLM"


def _pid_exists(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _terminate_pid(pid: int, *, _sigkill_grace_seconds: float = 5.0) -> None:
    """Send SIGTERM; escalate to SIGKILL after a short grace if the process survives.

    A4-8: a wedged rlm subprocess blocked inside rlm.completion() never
    responds to SIGTERM alone. After _sigkill_grace_seconds, if the pid is
    still alive, we escalate to SIGKILL so stop_run always terminates the run.
    """
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, capture_output=True)
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    # Wait up to _sigkill_grace_seconds for the process to exit gracefully.
    import time
    deadline = time.monotonic() + _sigkill_grace_seconds
    while time.monotonic() < deadline:
        if not _pid_exists(pid):
            return
        time.sleep(0.25)
    # Escalate: process is still alive after grace period.
    if _pid_exists(pid):
        try:
            os.kill(pid, signal.SIGKILL)
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
    "finalize_benchmark",
    "project_id_for_pdf_path",
    "sse_event",
]
