"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import re

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend import __version__
from backend.config import get_settings
from backend.persistence.database import Database
from backend.services.approval import ApprovalAction, ApprovalService, ApprovalState
from backend.services.context.graph import KnowledgeGraphService
from backend.services.context.memory import CrossProjectMemoryService, MemoryKind
from backend.services.datasets import DatasetCacheService
from backend.services.diagnostics import FailureDiagnosisService
from backend.services.events.live_runs import (
    FileLiveRunService,
    ProviderCredentials,
    StartRunRequest,
)
from backend.services.research_workspace import ResearchWorkspaceService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# rdr introspection helpers
# ---------------------------------------------------------------------------

_PAPER_FULL_KEYWORDS = frozenset(["paper_full", "paper_text", "raw_paper", "corpus"])
_MAX_JUSTIFICATION_CHARS = 1000


def _redact_corpus_keys(d: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of *d* with corpus-text keys stripped.

    Never include raw paper text in API responses — corpus-leak redaction.
    Applies to the top-level dict of a cluster or leaf-score record.
    """
    return {k: v for k, v in d.items() if k.lower() not in _PAPER_FULL_KEYWORDS}


def _truncate_justification(text: str) -> str:
    if len(text) <= _MAX_JUSTIFICATION_CHARS:
        return text
    return text[:_MAX_JUSTIFICATION_CHARS] + "…"


def _runs_root() -> Path:
    """Resolve the runs root, mirroring the logic in ``create_app``."""
    import os as _os
    from backend.config import get_settings as _gs
    s = _gs()
    env_val = _os.environ.get("REPROLAB_RUNS_ROOT")
    if s.runs_root is not None:
        return Path(s.runs_root)
    if env_val:
        return Path(env_val)
    return Path(__file__).resolve().parents[1] / "runs"


def _read_rdr_clusters(project_id: str) -> dict[str, Any] | None:
    """Read per-cluster status from ``runs/<id>/iterations/``.

    Returns None when the run directory does not exist; returns the
    response dict (clusters list possibly empty) when the dir exists.
    """
    run_dir = _runs_root() / project_id
    if not run_dir.is_dir():
        return None
    iterations_dir = run_dir / "iterations"
    if not iterations_dir.is_dir():
        return {"project_id": project_id, "clusters": []}

    # Index cluster checkpoints by cluster_id; accumulate repair history.
    # cluster_<index>_<uuid>.json → primary entry
    # repair_<n>_cluster_<uuid>.json → appended to repair_history
    cluster_map: dict[str, dict[str, Any]] = {}
    repair_map: dict[str, list[dict[str, Any]]] = {}

    for path in sorted(iterations_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        payload = _redact_corpus_keys(payload)
        cluster_id = payload.get("cluster_id", "")
        if path.name.startswith("cluster_"):
            # Parse index from cluster_<index>_<uuid>.json
            parts = path.stem.split("_", 2)
            try:
                index = int(parts[1])
            except (IndexError, ValueError):
                index = -1
            cluster_map[cluster_id] = {
                "index": index,
                "cluster_id": cluster_id,
                "title": payload.get("cluster_title", ""),
                "leaf_ids": payload.get("leaf_ids", []),
                "failed": payload.get("failed", False),
                "file_count": payload.get("file_count", 0),
                "repair_history": [],
            }
        elif path.name.startswith("repair_"):
            rep_n = payload.get("repair_pass", 0)
            repair_map.setdefault(cluster_id, []).append({
                "pass": rep_n,
                "failed": payload.get("failed", False),
                "file_count": payload.get("file_count", 0),
            })

    # Merge repair history into cluster entries
    for cid, repairs in repair_map.items():
        if cid in cluster_map:
            cluster_map[cid]["repair_history"] = sorted(repairs, key=lambda r: r["pass"])
        else:
            # Repair without initial cluster checkpoint (partial run) — create stub
            cluster_map[cid] = {
                "index": -1,
                "cluster_id": cid,
                "title": "",
                "leaf_ids": [],
                "failed": None,
                "file_count": 0,
                "repair_history": sorted(repairs, key=lambda r: r["pass"]),
            }

    clusters = sorted(cluster_map.values(), key=lambda c: c["index"])
    return {"project_id": project_id, "clusters": clusters}


def _read_rdr_repair_iterations(project_id: str) -> dict[str, Any] | None:
    """Summarize repair passes from ``runs/<id>/iterations/repair_*.json``."""
    run_dir = _runs_root() / project_id
    if not run_dir.is_dir():
        return None
    iterations_dir = run_dir / "iterations"
    if not iterations_dir.is_dir():
        return {"project_id": project_id, "passes": []}

    # Group repair checkpoints by pass number.
    by_pass: dict[int, list[dict[str, Any]]] = {}
    for path in iterations_dir.glob("repair_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rep_n = int(payload.get("repair_pass", 0))
        by_pass.setdefault(rep_n, []).append(payload)

    passes = []
    for rep_n in sorted(by_pass):
        entries = by_pass[rep_n]
        passes.append({
            "pass": rep_n,
            "cluster_count": len(entries),
            "failed_count": sum(1 for e in entries if e.get("failed", False)),
        })
    return {"project_id": project_id, "passes": passes}


def _read_rdr_leaf_scores(project_id: str) -> dict[str, Any] | None:
    """Read per-leaf scores from ``runs/<id>/final_report.json``.

    Returns None when the run dir does not exist or ``final_report.json`` is absent.
    """
    run_dir = _runs_root() / project_id
    if not run_dir.is_dir():
        return None
    report_path = run_dir / "final_report.json"
    if not report_path.exists():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    rubric = report.get("rubric") or {}
    overall_score = float(rubric.get("overall_score") or 0.0)
    raw_leaf_scores = rubric.get("leaf_scores") or []

    leaf_scores = []
    for entry in raw_leaf_scores:
        if not isinstance(entry, dict):
            continue
        leaf_id = str(entry.get("id") or entry.get("leaf_id") or "")
        score = float(entry.get("score") or 0.0)
        justification = _truncate_justification(str(entry.get("justification") or ""))
        leaf_scores.append({"id": leaf_id, "score": score, "justification": justification})

    return {
        "project_id": project_id,
        "overall_score": overall_score,
        "leaf_scores": leaf_scores,
    }


def _safe_runpod_name_part(value: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")
    return safe[:48] or "run"


def _read_dashboard_events(project_id: str) -> list[dict[str, Any]]:
    path = _runs_root() / project_id / "dashboard_events.jsonl"
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    events.append(event)
    except OSError:
        return []
    return events


def _coerce_runpod_pods(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [p for p in payload if isinstance(p, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("pods", "data", "results", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return [p for p in value if isinstance(p, dict)]
    return []


def _runpod_event_status(project_id: str, sandbox_mode: str | None) -> dict[str, Any]:
    events = _read_dashboard_events(project_id)
    run_experiment_events = [
        event
        for event in events
        if event.get("event") == "primitive_call"
        and event.get("primitive") == "run_experiment"
    ]
    last = run_experiment_events[-1] if run_experiment_events else None
    built_environment = any(
        event.get("event") == "primitive_call"
        and event.get("primitive") == "build_environment"
        and event.get("status") == "ok"
        for event in events
    )

    if sandbox_mode and sandbox_mode != "runpod":
        status = "not_runpod"
        label = f"sandbox: {sandbox_mode}"
        detail = f"This run uses the {sandbox_mode} sandbox; no RunPod pod will be created."
    elif last and last.get("status") == "start":
        status = "executing"
        label = "runpod: executing"
        detail = "run_experiment has started. The RunPod pod should be provisioning or executing commands."
    elif last and last.get("status") == "ok":
        status = "destroyed"
        label = "runpod: experiment complete"
        detail = "run_experiment completed; the runtime cleanup path should have destroyed the pod."
    elif last and last.get("status") == "error":
        status = "error"
        label = "runpod: last experiment failed"
        detail = "run_experiment failed; the root REPL can still repair and retry."
    elif built_environment:
        status = "not_yet"
        label = "runpod: ready at experiment"
        detail = "Environment is built. Pods are created lazily when run_experiment starts."
    else:
        status = "not_yet"
        label = "runpod: not yet"
        detail = "Pods are created lazily at run_experiment, after paper understanding, planning, and baseline implementation."

    return {
        "project_id": project_id,
        "sandbox_mode": sandbox_mode,
        "status": status,
        "label": label,
        "detail": detail,
        "source": "events",
        "pod": None,
        "updated_at": last.get("timestamp") if last else None,
    }


async def _query_runpod_status(project_id: str, sandbox_mode: str | None, settings: Any) -> dict[str, Any]:
    derived = _runpod_event_status(project_id, sandbox_mode)
    if sandbox_mode and sandbox_mode != "runpod":
        return derived
    if derived.get("status") == "not_yet" and not str(getattr(settings, "runpod_pod_id", "") or "").strip():
        return derived
    api_key = str(getattr(settings, "runpod_api_key", "") or "").strip()
    if not api_key:
        return derived

    headers = {"Authorization": f"Bearer {api_key}"}
    base_url = str(getattr(settings, "runpod_api_base_url", "https://rest.runpod.io/v1")).rstrip("/")
    persistent_pod_id = str(getattr(settings, "runpod_pod_id", "") or "").strip()
    try:
        async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=8) as client:
            if persistent_pod_id:
                response = await client.get(f"/pods/{persistent_pod_id}")
                response.raise_for_status()
                pods = [response.json()]
            else:
                response = await client.get("/pods")
                response.raise_for_status()
                pods = _coerce_runpod_pods(response.json())
    except Exception as exc:
        return {
            **derived,
            "source": "events",
            "api_error": str(exc),
        }

    prefix = f"reprolab-{_safe_runpod_name_part(project_id)}-"
    matching = [
        pod
        for pod in pods
        if persistent_pod_id
        or str(pod.get("name") or "").startswith(prefix)
    ]
    if not matching:
        return derived

    pod = matching[0]
    pod_id = str(pod.get("id") or pod.get("podId") or "")
    desired = str(pod.get("desiredStatus") or "").upper()
    current = str(pod.get("currentStatus") or "").upper()
    raw_status = current or desired or "UNKNOWN"
    if desired in {"EXITED", "FAILED", "DEAD"} or current in {"EXITED", "FAILED", "DEAD"}:
        status = "destroyed" if current == "EXITED" else "error"
        label = "runpod: destroyed" if status == "destroyed" else "runpod: pod error"
    elif desired in {"STOPPED", "TERMINATED"}:
        status = "destroyed"
        label = "runpod: destroyed"
    elif desired in {"STOPPING", "TERMINATING"} or current in {"STOPPING", "TERMINATING"}:
        status = "stopping"
        label = "runpod: stopping"
    elif current == "RUNNING":
        if derived.get("status") == "executing":
            status = "executing"
            label = f"runpod: executing {pod_id}" if pod_id else "runpod: executing"
        else:
            status = "ready"
            label = f"runpod: ready {pod_id}" if pod_id else "runpod: ready"
    else:
        status = "provisioning"
        label = "runpod: provisioning"

    return {
        **derived,
        "status": status,
        "label": label,
        "detail": f"RunPod API reports pod status {raw_status}.",
        "source": "runpod_api",
        "pod": {
            "id": pod_id or None,
            "name": pod.get("name"),
            "desiredStatus": desired or None,
            "currentStatus": current or None,
        },
    }


def _enforce_demo_gate(provided_secret: str | None, configured_secret: str) -> None:
    """Require a matching X-Demo-Secret header on the run-start endpoints.

    When ``configured_secret`` is empty the gate is disabled (local dev).
    When set, the caller must present a matching secret; a mismatch or a
    missing secret raises 401. The comparison is constant-time.
    """
    if not configured_secret:
        return
    if not provided_secret or not hmac.compare_digest(provided_secret, configured_secret):
        raise HTTPException(status_code=401, detail="A valid demo access secret is required.")


def _make_lifespan():
    """Build the FastAPI lifespan context manager with pod-sweep startup + periodic sweep.

    Fail-soft: startup sweep and scheduler errors are logged but never block
    the backend from starting — the typical local-dev case has no RUNPOD key.
    """
    from backend.services.runtime.pod_sweep_scheduler import PodSweepScheduler
    from backend.services.runtime.pod_sweeper import sweep_stale_pods

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        _pod_sweep_enabled = (
            bool(os.environ.get("REPROLAB_RUNPOD_API_KEY"))
            and os.environ.get("REPROLAB_POD_SWEEP_ENABLED", "true").lower()
            not in {"false", "0", "no", "off"}
        )
        scheduler = PodSweepScheduler()
        if _pod_sweep_enabled:
            try:
                max_age = int(os.environ.get("REPROLAB_POD_SWEEP_MAX_AGE_S", "7200"))
                summary = await asyncio.to_thread(
                    sweep_stale_pods,
                    max_age_seconds=max_age,
                    dry_run=False,
                )
                logger.info("startup pod sweep: %s", summary)
            except Exception as exc:
                logger.warning("startup pod sweep failed (non-fatal): %s", exc)
        try:
            await scheduler.start()
        except Exception as exc:
            logger.warning("pod_sweep_scheduler start failed (non-fatal): %s", exc)

        yield

        # Shutdown
        try:
            await scheduler.stop()
        except Exception:
            pass

    return lifespan


def create_app(*, run_service: Any | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    import os as _os
    import sys as _sys
    from pathlib import Path as _Path
    settings = get_settings()
    # Tier 2a — install pipeline.log + pipeline.jsonl on the root logger when
    # REPROLAB_LOG_DIR / REPROLAB_RUNS_ROOT is set. No-op otherwise.
    from backend.observability.run_logging import configure_root_logger
    configure_root_logger()

    # Resolve runs_root with a direct os.environ.get fallback. The Settings
    # singleton (`_settings_cache`) is per-process; under uvicorn --reload on
    # Windows the reloader and the worker are SEPARATE processes and the
    # worker's get_settings() apparently returned None earlier despite the
    # env var being baked into the cmd shim. Reading os.environ directly
    # here removes Settings as a possible failure point, and we still benefit
    # from Settings when env is unset (e.g. tests).
    env_runs_root = _os.environ.get("REPROLAB_RUNS_ROOT")
    effective_runs_root = settings.runs_root
    if effective_runs_root is None and env_runs_root:
        effective_runs_root = _Path(env_runs_root)

    # Diagnostic + marker. The print goes to backend.log; the marker file
    # captures the FULL story per-process so we can compare reloader vs
    # worker. backend.log only seems to capture stdout from one of them.
    print(
        f"[reprolab] runs_root: settings={settings.runs_root!r} "
        f"env={env_runs_root!r} effective={effective_runs_root!r} "
        f"pid={_os.getpid()} cwd={_os.getcwd()!r}",
        flush=True,
    )
    try:
        marker_root = (
            effective_runs_root if effective_runs_root else _Path("logs") / "_no_runs_root"
        )
        marker_root.mkdir(parents=True, exist_ok=True)
        (marker_root / f"_create_app_pid{_os.getpid()}.txt").write_text(
            f"settings.runs_root={settings.runs_root!r}\n"
            f"env REPROLAB_RUNS_ROOT={env_runs_root!r}\n"
            f"effective={effective_runs_root!r}\n"
            f"cwd={_os.getcwd()}\n"
            f"argv={_sys.argv}\n",
            encoding="utf-8",
        )
    except Exception:
        pass

    # Honor REPROLAB_RUNS_ROOT so dev.ps1 / dev.sh actually colocate pipeline
    # workspaces with the launch's server logs. When unset, FileLiveRunService
    # falls back to <repo>/runs as before.
    service = run_service or FileLiveRunService(runs_root=effective_runs_root)

    app = FastAPI(
        title="ReproLab Agent",
        version=__version__,
        debug=settings.debug,
        lifespan=_make_lifespan(),
    )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logging.getLogger(__name__).exception(
            "Unhandled route error: %s %s",
            request.method,
            request.url.path,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc) or "Internal Server Error"},
        )

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/auth-status")
    async def auth_status() -> dict:
        """Return which LLM providers have working credentials on this server.

        Used by the upload-view provider picker to enable/disable radio buttons
        without any user-visible credential input. Response shape: D1.
        No demo gate — this is purely a capability probe, not a run-start.
        """
        import asyncio as _asyncio
        from backend.agents.runtime.factory import aggregate_auth_status
        return await _asyncio.to_thread(aggregate_auth_status)

    # ------------------------------------------------------------------ #
    # Live-run API + SSE event stream (origin/main)
    # ------------------------------------------------------------------ #

    @app.post("/runs", status_code=202)
    async def start_run(request: StartRunRequest, x_demo_secret: str | None = Header(default=None)):
        _enforce_demo_gate(x_demo_secret, settings.demo_secret)
        return await service.start_run(request)

    @app.post("/runs/{project_id}/resume", status_code=202)
    async def resume_run(
        project_id: str,
        request: ResumeRunRequest | None = None,
        x_demo_secret: str | None = Header(default=None),
    ):
        """Re-spawn the orchestrator subprocess for an existing project.

        The orchestrator's resume-from-checkpoint logic picks up at the
        last completed stage. ``request_overrides`` (optional body) lets
        callers bump e.g. executionMode=max to push past a wall-clock
        failure without losing the work already done.
        """
        _enforce_demo_gate(x_demo_secret, settings.demo_secret)
        overrides: dict[str, Any] | None = None
        if request is not None:
            overrides = request.model_dump(exclude_none=True) or None
        state = await service.resume_run(project_id, request_overrides=overrides)
        if state is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return state

    @app.post("/runs/arxiv", status_code=202)
    async def start_arxiv_run(request: StartArxivRunRequest, x_demo_secret: str | None = Header(default=None)):
        """Fetch a paper from a URL (arXiv/openreview/etc) and start a run.

        Server-side fetch sidesteps browser CORS and the multipart upload
        gymnastics that the file path requires. The bytes are handed to the
        same ``start_uploaded_run`` service as a real upload — no second code
        path to keep in sync.
        """
        _enforce_demo_gate(x_demo_secret, settings.demo_secret)
        normalized_url = (request.url or "").strip()
        if not normalized_url:
            raise HTTPException(status_code=400, detail="An arXiv (or other paper) URL is required.")
        if not re.match(r"^https?://", normalized_url, re.IGNORECASE):
            raise HTTPException(status_code=400, detail="URL must start with http:// or https://.")
        # arxiv.org/abs/1234.5678 → arxiv.org/pdf/1234.5678 so the response is
        # the PDF rather than the HTML abstract page.
        fetch_url = re.sub(r"^(https?://arxiv\.org)/abs/", r"\1/pdf/", normalized_url, flags=re.IGNORECASE)
        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=httpx.Timeout(30.0, connect=10.0),
                headers={"user-agent": "ReproLab/0.1 (+https://github.com/anthropics/openresearch)"},
            ) as client:
                response = await client.get(fetch_url)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Could not fetch paper from {fetch_url!r}: {exc}") from exc
        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"Upstream returned HTTP {response.status_code} for {fetch_url!r}.",
            )
        content = response.content
        if not content:
            raise HTTPException(status_code=502, detail="Upstream returned an empty body.")
        if len(content) > 50 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Fetched paper exceeds 50 MB limit.")
        # Validate it's actually a PDF — content-type or magic bytes. arxiv
        # serves application/pdf; we also accept the %PDF- header as a
        # secondary check because some mirrors advertise octet-stream.
        looks_like_pdf = content[:5] == b"%PDF-" or "pdf" in (response.headers.get("content-type") or "").lower()
        if not looks_like_pdf:
            raise HTTPException(
                status_code=415,
                detail="Fetched content does not appear to be a PDF (no %PDF- header and content-type is not pdf-ish).",
            )
        # Derive a stable, safe filename from the URL's last path segment.
        # `2512.24601` → `arxiv_2512_24601.pdf` so the manifest renders nicely.
        last_segment = re.split(r"[/?#]", normalized_url.rstrip("/"))[-1] or "paper"
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", last_segment).strip("._-") or "paper"
        if "arxiv.org" in normalized_url.lower() and not safe_stem.lower().startswith("arxiv"):
            safe_stem = f"arxiv_{safe_stem}"
        if not safe_stem.lower().endswith(".pdf"):
            safe_stem = f"{safe_stem}.pdf"
        run_request = StartRunRequest(
            mode=request.mode or "rlm",
            provider=request.provider or "anthropic",
            verificationProvider=request.verificationProvider,
            executionMode=request.executionMode or "max",
            sandbox=request.sandbox or settings.default_sandbox,
            gpuMode=request.gpuMode or "auto",
            model=request.model or "sonnet",
            minimize_compute=request.minimize_compute,
            provider_credentials=request.provider_credentials,
            estimate_id=request.estimate_id,
        )
        return await service.start_uploaded_run(
            run_request,
            file_name=safe_stem,
            content=content,
        )

    @app.post("/runs/upload", status_code=202)
    async def start_uploaded_run(request: Request, x_demo_secret: str | None = Header(default=None)):
        _enforce_demo_gate(x_demo_secret, settings.demo_secret)
        form = await request.form()
        paper = form.get("paper")
        if paper is None or not hasattr(paper, "read"):
            raise HTTPException(status_code=400, detail="Upload a PDF before starting a lab run.")
        # Normalize the reported filename to a bare basename. A client may
        # send a path-qualified name — Windows browsers/tools can include
        # `C:\\...\\file.pdf` or backslash separators — and downstream code
        # (_stage_upload) treats it as a POSIX path. Strip both separators
        # so staging is platform-agnostic regardless of the upload source.
        raw_name = str(getattr(paper, "filename", "") or "paper.pdf")
        file_name = raw_name.replace("\\", "/").rsplit("/", 1)[-1].strip() or "paper.pdf"
        if not file_name.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")
        content = await paper.read()
        if not content:
            raise HTTPException(status_code=400, detail="Upload a PDF before starting a lab run.")
        run_request = StartRunRequest(
            mode=_form_value(form, "mode", "rlm"),
            provider=_form_value(form, "provider", "anthropic"),
            verificationProvider=_optional_form_value(form, "verificationProvider"),
            executionMode=_form_value(form, "executionMode", "max"),
            sandbox=_form_value(form, "sandbox", settings.default_sandbox),
            gpuMode=_form_value(form, "gpuMode", "auto"),
            model=_form_value(form, "model", "sonnet"),
            # Lane Q parity fix (codex review 2026-05-25): the multipart upload
            # path was dropping minimize_compute silently. /runs/arxiv forwards
            # it (line ~564); this path now matches.
            minimize_compute=_optional_form_bool(form, "minimizeCompute"),
            provider_credentials=_optional_form_provider_credentials(form),
            estimate_id=_optional_form_value(form, "estimateId"),
        )
        return await service.start_uploaded_run(
            run_request,
            file_name=file_name,
            content=content,
        )

    @app.get("/runs")
    async def list_runs(
        limit: int = 10,
        status: str | None = None,
        q: str | None = None,
        order_by: str = "updated_at",
    ) -> list[dict]:
        return await service.list_runs(
            limit=limit,
            status=status,
            q=q,
            order_by=order_by,
        )

    @app.get("/runs/latest")
    async def latest_run(
        mode: str | None = None,
        provider: str | None = None,
        executionMode: str | None = None,
        sandbox: str | None = None,
        verificationProvider: str | None = None,
        gpuMode: str | None = None,
    ):
        state = await service.latest_run(
            mode=mode,
            provider=provider,
            execution_mode=executionMode,
            sandbox=sandbox,
            verification_provider=verificationProvider,
            gpu_mode=gpuMode,
        )
        if state is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return state

    @app.get("/runs/{project_id}")
    async def get_run(project_id: str):
        state = await service.get_run(project_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return state

    @app.get("/runs/{project_id}/source-pdf")
    async def get_source_pdf(project_id: str):
        getter = getattr(service, "get_source_pdf_path", None)
        if not callable(getter):
            raise HTTPException(status_code=404, detail="Source PDF not found")
        path = await getter(project_id)
        if path is None:
            raise HTTPException(status_code=404, detail="Source PDF not found")
        return FileResponse(
            path,
            media_type="application/pdf",
            filename="paper.pdf",
            content_disposition_type="inline",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/runs/{project_id}/final-report")
    async def get_final_report(project_id: str):
        getter = getattr(service, "get_final_report_path", None)
        if not callable(getter):
            raise HTTPException(status_code=404, detail="Final report not found")
        path = await getter(project_id)
        if path is None:
            raise HTTPException(status_code=404, detail="Final report not found")
        return FileResponse(
            path,
            media_type="text/markdown; charset=utf-8",
            filename="final_benchmark_report.md",
            content_disposition_type="inline",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/runs/{project_id}/rerun", status_code=202)
    async def rerun(
        project_id: str,
        x_demo_secret: str | None = Header(default=None),
    ):
        """Start a fresh run using the same paper source as an existing run.

        Reads ``runs/<project_id>/demo_status.json`` to discover the original
        PDF path (``sourcePdf.runPath``).  The PDF bytes are passed directly to
        ``start_uploaded_run`` — which stages them under a *new* project_id and
        spawns a fresh orchestrator — so the old run's state is never mutated.

        Returns the new run's LiveRunState (same shape as ``/runs/upload``).
        404 if the project does not exist; 422 if the source PDF is gone.
        """
        _enforce_demo_gate(x_demo_secret, settings.demo_secret)
        _read_status = getattr(service, "_read_status", None)
        if not callable(_read_status):
            raise HTTPException(status_code=500, detail="Service does not support rerun.")
        status = await asyncio.to_thread(_read_status, project_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Run not found")

        src_pdf = (status.get("sourcePdf") or {}) if isinstance(status, dict) else {}
        run_path = src_pdf.get("runPath") if isinstance(src_pdf, dict) else None
        file_name = (src_pdf.get("fileName") or "paper.pdf") if isinstance(src_pdf, dict) else "paper.pdf"

        if not run_path:
            raise HTTPException(
                status_code=422,
                detail="Source PDF location not recorded in this run's status — cannot rerun."
            )

        pdf_path = Path(run_path)
        if not pdf_path.exists():
            raise HTTPException(
                status_code=422,
                detail=f"Source PDF is no longer on disk ({run_path!r}) — cannot rerun."
            )

        content = await asyncio.to_thread(pdf_path.read_bytes)
        run_request = StartRunRequest(
            mode=status.get("runMode", "rlm"),
            provider=status.get("llmProvider", "anthropic"),
            verificationProvider=status.get("verificationProvider"),
            executionMode=status.get("executionMode", "efficient"),
            sandbox=status.get("sandboxMode", settings.default_sandbox),
            gpuMode=status.get("gpuMode", "auto"),
            model=status.get("model", "sonnet"),
        )
        return await service.start_uploaded_run(
            run_request,
            file_name=str(file_name),
            content=content,
        )

    @app.delete("/runs/{project_id}")
    async def stop_run(project_id: str, x_demo_secret: str | None = Header(default=None)):
        _enforce_demo_gate(x_demo_secret, settings.demo_secret)
        state = await service.stop_run(project_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return state

    @app.get("/runs/{project_id}/events")
    async def stream_run_events(project_id: str):
        return StreamingResponse(
            service.stream_events(project_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/runs/{project_id}/runpod-status")
    async def get_runpod_status(project_id: str):
        state = await service.get_run(project_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return await _query_runpod_status(project_id, state.sandboxMode, settings)

    # ------------------------------------------------------------------ #
    # rdr-specific introspection endpoints
    # ------------------------------------------------------------------ #

    @app.get("/runs/{project_id}/clusters")
    async def get_rdr_clusters(project_id: str) -> dict:
        """Per-cluster status for an rdr run.

        Reads ``runs/<id>/iterations/cluster_*.json`` and ``repair_*.json``.
        Returns 404 when the run directory does not exist; 200 + empty list
        when the iterations directory is absent or empty.
        Corpus-leak redaction: raw paper-text keys are stripped before response.
        """
        result = await asyncio.to_thread(_read_rdr_clusters, project_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return result

    @app.get("/runs/{project_id}/repair-iterations")
    async def get_rdr_repair_iterations(project_id: str) -> dict:
        """Repair-pass summary for an rdr run.

        Returns 404 when the run directory does not exist.
        """
        result = await asyncio.to_thread(_read_rdr_repair_iterations, project_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return result

    @app.get("/runs/{project_id}/leaf-scores")
    async def get_rdr_leaf_scores(project_id: str) -> dict:
        """Per-leaf scores from the rdr run's ``final_report.json``.

        Returns 404 when the run dir or final_report.json do not exist yet.
        Justification strings are capped at 1000 characters to bound payload size.
        """
        result = await asyncio.to_thread(_read_rdr_leaf_scores, project_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Run not found or scoring not complete")
        return result

    # ------------------------------------------------------------------ #
    # Models (LLM choices surfaced in the upload-view dropdown)
    # ------------------------------------------------------------------ #

    @app.get("/models")
    async def list_models() -> list[dict[str, Any]]:
        from backend.agents.rlm.models import list_root_model_choices

        return list_root_model_choices()

    # ------------------------------------------------------------------ #
    # Phase 2 workspace services (HEAD)
    # ------------------------------------------------------------------ #

    @app.get("/phase2/projects/{project_id}/summary")
    async def phase2_summary(project_id: str, memory_query: str = "") -> dict:
        db = _database(settings.database_url)
        try:
            summary = ResearchWorkspaceService(db).summarize_project(
                project_id,
                memory_query=memory_query,
            )
            return summary.model_dump(mode="json")
        finally:
            db.close()

    @app.get("/phase2/projects/{project_id}/graph")
    async def phase2_graph_query(
        project_id: str,
        entity_type: str = "function",
        calls: str | None = None,
        imports: str | None = None,
        name: str | None = None,
        path_contains: str | None = None,
    ) -> dict:
        db = _database(settings.database_url)
        try:
            result = KnowledgeGraphService(db).query(
                entity_type,
                project_id=project_id,
                calls=calls,
                imports=imports,
                name=name,
                path_contains=path_contains,
            )
            return result.model_dump(mode="json")
        finally:
            db.close()

    @app.get("/phase2/memory/search")
    async def phase2_memory_search(
        query: str,
        kind: MemoryKind | None = None,
        limit: int = 5,
    ) -> dict:
        db = _database(settings.database_url)
        try:
            results = CrossProjectMemoryService(db).search(query, kind=kind, limit=limit)
            return {"results": [item.model_dump(mode="json") for item in results]}
        finally:
            db.close()

    @app.post("/phase2/approvals/evaluate")
    async def phase2_approval_evaluate(
        request: ApprovalEvaluateRequest,
        x_demo_secret: str | None = Header(default=None),
    ) -> dict:
        _enforce_demo_gate(x_demo_secret, settings.demo_secret)
        db = _database(settings.database_url)
        try:
            approval_service = ApprovalService(db)
            evaluation = approval_service.evaluate(
                action=request.action,
                dataset_size_gb=request.dataset_size_gb,
                runtime_minutes=request.runtime_minutes,
                gpu_cost_usd=request.gpu_cost_usd,
                repo_trust_level=request.repo_trust_level,
                license_state=request.license_state,
                network_stage=request.network_stage,
                assumption_risk=request.assumption_risk,
                external_data=request.external_data,
                metadata=request.metadata,
            )
            approval = approval_service.request_if_needed(
                project_id=request.project_id,
                label=request.label or request.action.replace("_", " ").title(),
                evaluation=evaluation,
            )
            return {
                "evaluation": evaluation.model_dump(mode="json"),
                "approval": approval.model_dump(mode="json") if approval else None,
            }
        finally:
            db.close()

    @app.post("/phase2/approvals/{approval_id}/resolve")
    async def phase2_approval_resolve(
        approval_id: str,
        request: ApprovalResolveRequest,
        x_demo_secret: str | None = Header(default=None),
    ) -> dict:
        _enforce_demo_gate(x_demo_secret, settings.demo_secret)
        db = _database(settings.database_url)
        try:
            try:
                resolved = ApprovalService(db).resolve(
                    approval_id,
                    state=request.state,
                    resolved_by=request.resolved_by,
                    note=request.note,
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="Approval not found") from exc
            return resolved.model_dump(mode="json")
        finally:
            db.close()

    @app.post("/phase2/datasets/plan")
    async def phase2_dataset_plan(
        request: DatasetPlanRequest,
        x_demo_secret: str | None = Header(default=None),
    ) -> dict:
        _enforce_demo_gate(x_demo_secret, settings.demo_secret)
        db = _database(settings.database_url)
        try:
            entry = DatasetCacheService(db).plan(
                name=request.name,
                source_url=request.source_url,
                version=request.version,
                checksum=request.checksum,
                size_bytes=request.size_bytes,
                source_project_id=request.project_id,
                metadata=request.metadata,
            )
            return entry.model_dump(mode="json")
        finally:
            db.close()

    @app.post("/phase2/failures/diagnose")
    async def phase2_failure_diagnose(
        request: FailureDiagnoseRequest,
        x_demo_secret: str | None = Header(default=None),
    ) -> dict:
        _enforce_demo_gate(x_demo_secret, settings.demo_secret)
        db = _database(settings.database_url)
        try:
            event = FailureDiagnosisService(db).diagnose(
                project_id=request.project_id,
                stage=request.stage,
                command=request.command,
                exit_code=request.exit_code,
                stdout=request.stdout,
                stderr=request.stderr,
                timed_out=request.timed_out,
                cause_kind=request.cause_kind,
                artifact_refs=tuple(request.artifact_refs),
            )
            return event.model_dump(mode="json")
        finally:
            db.close()

    # Leaderboard route — read-only ranking of completed runs across models.
    # Mounted via include_router because it lives in its own module
    # (backend/routes/leaderboard.py); spec
    # docs/superpowers/specs/2026-05-23-rubric-climb-leaderboard.md §4.4.
    # No demo-gate; reads are public.
    from backend.routes.leaderboard import router as leaderboard_router
    app.include_router(leaderboard_router)

    # Chat-steering route — POST /runs/{project_id}/messages.
    from backend.routes.messages import router as messages_router
    app.include_router(messages_router)

    # Worker reports route — GET /runs/{project_id}/reports.
    from backend.routes.reports import router as reports_router
    app.include_router(reports_router)

    # Budget estimation route — POST /paper/estimate.
    from backend.routes.estimate import router as estimate_router
    app.include_router(estimate_router)

    # Codex I3 fix: audit freshness was defined but never invoked. Run at
    # startup so the operator sees a stale-pricing WARNING in logs the
    # moment the process boots. Non-blocking on failure.
    try:
        from backend.services.pricing import check_audit_freshness
        stale = check_audit_freshness()
        if stale:
            logger.warning(
                "pricing: %d catalog entr%s past the 90-day audit window: %s. "
                "Refresh docs/runbooks/pricing-audit-YYYY-QN.md and bump "
                "CATALOG_SCHEMA_VERSION.",
                len(stale),
                "y is" if len(stale) == 1 else "ies are",
                ", ".join(sorted(stale)[:8]),
            )
    except Exception:  # noqa: BLE001 — startup hook must never block app boot
        logger.exception("pricing: audit freshness check raised at startup")

    return app


# --------------------------------------------------------------------------- #
# Phase 2 request models (HEAD)
# --------------------------------------------------------------------------- #

class ResumeRunRequest(BaseModel):
    """Body for ``POST /runs/{project_id}/resume``.

    All fields optional — when None, the resumed run inherits the
    original run's config (read from demo_status.json). Set fields
    override that inheritance per-key.
    """

    mode: str | None = None
    provider: str | None = None
    verificationProvider: str | None = None
    executionMode: str | None = None
    sandbox: str | None = None
    gpuMode: str | None = None
    model: str | None = None


class StartArxivRunRequest(BaseModel):
    """Body for ``POST /runs/arxiv``.

    All run-config fields are optional so the client only has to send the URL.
    Defaults are resolved server-side to mirror what the multipart upload path
    provides (mode=rlm, provider=anthropic, sandbox=<settings default>, …).
    """

    url: str = ""
    mode: str | None = None
    provider: str | None = None
    verificationProvider: str | None = None
    executionMode: str | None = None
    sandbox: str | None = None
    gpuMode: str | None = None
    model: str | None = None
    # Lane Q — "reproduce the CLAIM, not the recipe" mode. Forwarded to
    # StartRunRequest.minimize_compute when the lab UI passes it via
    # /api/demo/arxiv.
    minimize_compute: bool | None = None
    # BYO LLM credentials — when present, override env-var defaults for
    # this run's subprocess. Never persisted; see live_runs.ProviderCredentials.
    provider_credentials: ProviderCredentials | None = None
    # Budget estimate coupling — the cached estimate's p90 numbers are used
    # as default budget caps for the actual run.
    estimate_id: str | None = None


class ApprovalEvaluateRequest(BaseModel):
    project_id: str
    action: ApprovalAction
    label: str = ""
    dataset_size_gb: float | None = None
    runtime_minutes: float | None = None
    gpu_cost_usd: float | None = None
    repo_trust_level: str = ""
    license_state: str = ""
    network_stage: str = ""
    assumption_risk: str = ""
    external_data: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ApprovalResolveRequest(BaseModel):
    state: ApprovalState
    resolved_by: str = ""
    note: str = ""


class DatasetPlanRequest(BaseModel):
    project_id: str
    name: str
    source_url: str = ""
    version: str = ""
    checksum: str = ""
    size_bytes: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FailureDiagnoseRequest(BaseModel):
    project_id: str
    stage: str
    command: str = ""
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    cause_kind: str = ""
    artifact_refs: list[str] = Field(default_factory=list)


def _database(database_url: str) -> Database:
    db = Database(database_url)
    db.initialize()
    return db


# --------------------------------------------------------------------------- #
# Form helpers (origin/main, used by /runs/upload)
# --------------------------------------------------------------------------- #

def _form_value(form: Any, key: str, default: str) -> str:
    value = form.get(key)
    return str(value) if value not in (None, "") else default


def _optional_form_value(form: Any, key: str) -> str | None:
    value = form.get(key)
    if value in (None, "", "same"):
        return None
    return str(value)


def _optional_form_bool(form: Any, key: str) -> bool | None:
    """Parse a multipart form field as an optional bool.

    Accepts "true"/"1"/"yes" → True, "false"/"0"/"no" → False, missing/empty → None.
    """
    value = form.get(key)
    if value in (None, ""):
        return None
    v = str(value).strip().lower()
    if v in {"true", "1", "yes", "on"}:
        return True
    if v in {"false", "0", "no", "off"}:
        return False
    return None


def _optional_form_provider_credentials(form: Any) -> ProviderCredentials | None:
    """Parse the optional ``providerCredentials`` form field as JSON.

    Multipart form fields are strings, so the BYO-key dict has to be
    JSON-stringified on the client. Empty or missing → None. Malformed
    JSON or invalid pydantic shape → HTTP 400 with the validator's
    actual error message (so the user sees "Azure config incomplete"
    rather than a generic 500).
    """
    raw = form.get("providerCredentials")
    if raw in (None, ""):
        return None
    try:
        import json as _json
        data = _json.loads(str(raw))
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"providerCredentials field is not valid JSON: {exc}",
        ) from exc
    if not isinstance(data, dict):
        raise HTTPException(
            status_code=400,
            detail="providerCredentials field must be a JSON object.",
        )
    if not data:
        return None
    try:
        return ProviderCredentials(**data)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"providerCredentials validation failed: {exc}",
        ) from exc
