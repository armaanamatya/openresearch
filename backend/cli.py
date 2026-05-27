"""ReproLab CLI — drives ingestion, inspection, and reproduction.

  $ python -m backend.cli ingest <pdf-path>
      project_id=prj_..., parsed=N sections, sources=N, chunks=N,
      workspace=ws_..., variables=['claim_map']

  $ python -m backend.cli inspect <project_id> [--variable VAR]
      Prints the materialized workspace state.

  $ python -m backend.cli reproduce <pdf-path> [--mode rlm]
      Full pipeline: ingest paper -> build workspace -> run agent pipeline.

This is a thin sequential composer: it wires Intake -> Parser ->
Indexer -> Workspace through a shared SqliteEventStore. The reproduce
command extends the pipeline into the agent layer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from backend.agents.execution import DEFAULT_SANDBOX_MODE
from backend.config import get_settings
from backend.eventstore.sqlite_store import SqliteEventStore
from backend.services.context.indexer import (
    IndexerAppService,
    SourcesProjection,
    StartIndexing,
)
from backend.services.context.workspace import (
    BuildWorkspace,
    WorkspaceAppService,
)
from backend.services.ingestion.discovery import (
    ArtifactDiscoveryAppService,
    DiscoverArtifacts,
    RegexArtifactDiscoveryAdapter,
)
from backend.services.ingestion.intake import (
    ArxivId,
    DoiRef,
    FetchPaper,
    IntakeAppService,
    PdfPath,
    RegisterProject,
)
from backend.services.ingestion.intake.service import project_id_for
from backend.services.ingestion.intake.fetchers.arxiv import ArxivFetcher
from backend.services.ingestion.intake.fetchers.doi import DoiFetcher
from backend.services.ingestion.intake.fetchers.pdf_path import PdfPathFetcher
from backend.services.ingestion.parser import (
    ParserAppService,
    StartParsing,
)
from backend.services.ingestion.parser.extractor import extractor_from_settings
from backend.services.ingestion.parser.resolving_parser import ResolvingParser

# Force-import event modules so all @register_event decorators run.
import backend.services.context.indexer.events  # noqa: F401
import backend.services.context.workspace.events  # noqa: F401
import backend.services.ingestion.discovery.events  # noqa: F401
import backend.services.ingestion.intake.events  # noqa: F401
import backend.services.ingestion.parser.events  # noqa: F401


_ARXIV_RE = re.compile(
    r"(?:arxiv:|arxiv\.org/(?:abs|pdf)/)?(?P<id>\d{4}\.\d{4,5}(?:v\d+)?)(?:\.pdf)?$",
    re.IGNORECASE,
)


def _build_workspace_claim_map(
    variables: dict, project_id: str, runs_root: Path | None = None
) -> dict:
    """Build the workspace claim map handed to a run.

    `variables` is the workspace view's `{name: Cited}` dict.

    The RLM orchestrator is the only pipeline. It offloads the paper whole
    into the REPL `context` variable, never a prompt (rlm-pivot-brief.md §7).
    Sources the corpus in order: (1) `parsed_full_text.txt` — the parser's
    direct full-text output, the clean source of truth; (2) the workspace
    `paper_text` variable; (3) one entry per variable. The workspace variable
    is reassembled from indexed chunks and has been observed to lose content,
    so the parser blob is preferred.
    """
    def _value_str(cited: Any) -> str:
        return (
            cited.value if isinstance(cited.value, str)
            else json.dumps(cited.value) if cited.value is not None
            else ""
        )

    def _one_entry(text: str) -> dict:
        return {
            "project_id": project_id,
            "entries": [
                {"source_id": project_id, "title": "paper_text", "excerpt": text}
            ],
        }

    # Phase 6: RLM is the only run mode; drop the `if mode == "rlm":` wrapper
    # and use the same 3-tier resolution unconditionally. Keep main's parser-
    # failure warnings (post-honesty improvement, not present in 295ab4e).
    # 1. parsed_full_text.txt — the parser's clean, complete output.
    if runs_root is not None:
        blob = Path(runs_root) / project_id / "parsed_full_text.txt"
        if not blob.exists():
            logger.warning(
                "parsed_full_text.txt missing — parser likely failed; "
                "falling back to workspace variable (lossy)"
            )
            blob_text = ""
        else:
            try:
                blob_text = blob.read_text(encoding="utf-8", errors="replace")
            except OSError:
                blob_text = ""
            if not blob_text.strip():
                logger.warning(
                    "parsed_full_text.txt is empty — parser likely failed"
                )
        if blob_text.strip():
            return _one_entry(blob_text)
    # 2. The workspace `paper_text` variable.
    paper_cited = variables.get("paper_text")
    if paper_cited is not None:
        val = paper_cited.value
        if isinstance(val, dict) and isinstance(val.get("text"), str):
            full_text = val["text"]
        elif isinstance(val, str):
            full_text = val
        else:
            full_text = None
        if full_text is not None:
            return _one_entry(full_text)
    # 3. Fallback: one entry per variable, un-truncated.
    return {
        "project_id": project_id,
        "entries": [
            {"source_id": name, "title": name, "excerpt": _value_str(cited)}
            for name, cited in variables.items()
        ],
    }


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write ``data`` to ``path`` atomically via tempfile + os.replace.

    Prevents a half-written demo_status.json if the process is killed
    mid-flush — readers see either the old contents or the new contents,
    never a truncated/empty file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _mark_demo_status_stopped(
    runs_root: Path,
    project_id: str,
    *,
    reason: str = "Pipeline interrupted",
) -> None:
    """Best-effort: flip demo_status.json to status=stopped on graceful exit.

    Called on KeyboardInterrupt/CancelledError in the CLI. Reads the
    existing status, preserves all fields, sets status=stopped + a
    completedAt timestamp + an error string the dashboard can render.
    Silent on failure — never let status bookkeeping mask the original
    interrupt cause.
    """
    try:
        path = runs_root / project_id / "demo_status.json"
        existing: dict[str, Any] = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
        now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        merged = {
            **existing,
            "status": "stopped",
            "updatedAt": now_iso,
            "completedAt": now_iso,
            "error": reason,
        }
        _atomic_write_json(path, merged)
    except Exception:
        return


def _mark_demo_status_failed(
    runs_root: Path,
    project_id: str,
    *,
    reason: str = "Pipeline crashed",
) -> None:
    """PR-ν.3 / P3 — flip demo_status.json to status=failed on uncaught exception.

    Distinct from ``_mark_demo_status_stopped`` (which fires on graceful
    KeyboardInterrupt / Ctrl-C and writes ``status=stopped``). This is for
    the crash path: an Exception escaped ``run_pipeline_rlm`` (or any
    setup-phase or post-finalize cleanup) before the terminal status was
    written. Without it the dashboard shows the run as ``running`` forever.

    Defensive against double-write: if the file is already in a terminal
    state (``completed`` / ``failed`` / ``stopped``), leave it alone. The
    happy path's ``_finalize`` already wrote the correct terminal status;
    we only fire when that path was never reached.

    Silent on failure — best-effort status bookkeeping must never mask the
    original exception that triggered it.
    """
    try:
        path = runs_root / project_id / "demo_status.json"
        # If the run never registered (no demo_status.json) skip — writing a
        # fresh "failed" record would fabricate UI state for a run the lab
        # never tracked. Diverges from `_mark_demo_status_stopped` which
        # creates the file; that helper fires on Ctrl-C of an in-flight run,
        # so the file usually exists. This one fires on an exception that may
        # have happened before any pipeline state was written.
        if not path.exists():
            return
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict):
                existing = {}
        except Exception:
            existing = {}  # corrupt; overwrite with a fresh failed payload
        if existing.get("status") in ("completed", "failed", "stopped"):
            return  # something else (probably _finalize) already wrote terminal status
        now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        merged = {
            **existing,
            "status": "failed",
            "updatedAt": now_iso,
            "completedAt": now_iso,
            "error": reason,
        }
        _atomic_write_json(path, merged)
    except Exception:
        return


def _make_services(
    database_url: str, runs_root: Path
) -> tuple[
    SqliteEventStore,
    IntakeAppService,
    ParserAppService,
    ArtifactDiscoveryAppService,
    IndexerAppService,
    WorkspaceAppService,
]:
    store = SqliteEventStore(database_url)
    intake = IntakeAppService(
        store=store,
        fetchers={
            "pdf_path": PdfPathFetcher(runs_root=runs_root),
            "arxiv": ArxivFetcher(runs_root=runs_root),
            "doi": DoiFetcher(runs_root=runs_root),
        },
    )
    parser = ParserAppService(
        store=store,
        parser=ResolvingParser(),
        runs_root=runs_root,
        extractor=extractor_from_settings(get_settings()),
    )
    discovery = ArtifactDiscoveryAppService(
        store=store,
        adapters=[RegexArtifactDiscoveryAdapter()],
    )
    indexer = IndexerAppService(store=store)

    # Auto-wire Chroma embedding store if chromadb is available.
    embedding_store = None
    try:
        from backend.services.context.semantic.store import try_create_chroma_store
        embedding_store = try_create_chroma_store()
    except Exception:
        pass

    workspace = WorkspaceAppService(
        store=store,
        indexer=indexer,
        discovery=discovery,
        embedding_store=embedding_store,
    )
    return store, intake, parser, discovery, indexer, workspace


def cmd_ingest(args: argparse.Namespace) -> int:
    runs_root = Path(args.runs_root)
    store, intake, parser, discovery, indexer, workspace = _make_services(
        args.database_url, runs_root
    )

    source = _source_from_cli(args.source, args.source_kind)
    print(f"[1/6] Registering project for {args.source}", file=sys.stderr)
    project_id = intake.register_project(RegisterProject(source=source))
    print(f"      project_id={project_id}", file=sys.stderr)

    print("[2/6] Fetching paper", file=sys.stderr)
    if not intake.fetch_paper(FetchPaper(project_id=project_id)):
        print("      FAILED — see paper_fetch_failed event", file=sys.stderr)
        return 1

    print("[3/6] Parsing", file=sys.stderr)
    if not parser.start_parsing(StartParsing(project_id=project_id)):
        print("      FAILED — see parsing_failed event", file=sys.stderr)
        return 1

    print("[4/6] Discovering external artifacts", file=sys.stderr)
    if not discovery.discover(DiscoverArtifacts(project_id=project_id)):
        print("      FAILED — see discovery_failed event", file=sys.stderr)
        return 1

    print("[5/6] Indexing", file=sys.stderr)
    if not indexer.start_indexing(StartIndexing(project_id=project_id)):
        print("      FAILED — see indexing_failed event", file=sys.stderr)
        return 1

    print("[6/6] Building workspace", file=sys.stderr)
    workspace_id = workspace.build_workspace(
        BuildWorkspace(project_id=project_id, agent_name=args.agent)
    )

    sources = SourcesProjection()
    indexer.project_into_projection(project_id, sources)
    view = workspace.materialize_view(workspace_id)
    summary = {
        "project_id": project_id,
        "workspace_id": workspace_id,
        "workspace_ready": view.is_ready,
        "discovered_artifacts": len(discovery.list_artifacts(project_id)),
        "sources": sources.source_count,
        "chunks": sources.chunk_count,
        "variables": sorted(view.variables.keys()),
        "variable_count": view.variable_count,
    }
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    store.close()
    return 0


def cmd_regenerate_report(args: argparse.Namespace) -> int:
    """Regenerate ``final_report.md`` from existing ``final_report.json`` +
    ``tokens_total.json`` + ``timing.json`` sidecars.

    Use case: the renderer added a new section (e.g., PR-ν.2 added Token Usage
    + Per-Step Timing) and you want to refresh existing runs without re-running
    them. Idempotent — pure read+rewrite of the markdown; the JSON is untouched.
    """
    from backend.agents.rlm.report import (
        RLMFinalReport,
        _atomic_write,
        _render_markdown,
    )

    runs_root = Path(args.runs_root)
    project_dir = runs_root / args.project_id
    fr_path = project_dir / "final_report.json"
    md_path = project_dir / "final_report.md"

    if not fr_path.exists():
        sys.stderr.write(f"ERROR: {fr_path} does not exist\n")
        return 1

    try:
        report_data = json.loads(fr_path.read_text(encoding="utf-8"))
        report = RLMFinalReport(**report_data)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"ERROR: failed to load final_report.json: {exc}\n")
        return 2

    md = _render_markdown(report, project_dir=project_dir)
    _atomic_write(md_path, md)
    sys.stdout.write(f"Regenerated {md_path} ({len(md)} chars)\n")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    runs_root = Path(args.runs_root)
    store, _intake, _parser, _discovery, indexer, workspace = _make_services(
        args.database_url, runs_root
    )

    # Resolve workspace_id deterministically from project_id + agent.
    from backend.services.context.workspace.service import _workspace_id_for

    wsid = _workspace_id_for(args.project_id, args.agent)

    view = workspace.materialize_view(wsid)
    if not view.is_ready and view.variable_count == 0:
        print(
            f"No workspace found for project {args.project_id!r}. "
            f"Run `ingest` first.",
            file=sys.stderr,
        )
        return 2

    if args.variable is not None:
        cited = view.get(args.variable)
        if cited is None:
            print(f"Variable {args.variable!r} not in workspace", file=sys.stderr)
            return 3
        out = {
            "name": args.variable,
            "value": cited.value,
            "citations": [c.model_dump() for c in cited.citations],
        }
    else:
        out = {
            "workspace_id": view.workspace_id,
            "is_ready": view.is_ready,
            "variables": {
                name: {
                    "value_summary": _summarize_value(cited.value),
                    "citation_count": len(cited.citations),
                }
                for name, cited in view.variables.items()
            },
        }
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    store.close()
    return 0


def _summarize_value(value: object) -> object:
    """Compact view: keep dict keys, replace long lists/strings with len.
    Just for the inspect summary; the per-variable view is full fidelity."""
    if isinstance(value, dict):
        return {
            k: _summarize_value(v) for k, v in value.items() if not k.startswith("_")
        }
    if isinstance(value, list):
        if len(value) > 5:
            return {"_list_len": len(value), "head": value[:2]}
        return [_summarize_value(v) for v in value]
    if isinstance(value, str) and len(value) > 200:
        return value[:200] + "…"
    return value



_REPRODUCE_DEFAULTS = {
    "database_url": get_settings().database_url,
    # Honor REPROLAB_RUNS_ROOT via Settings — see backend/config.py.
    "runs_root": str(get_settings().runs_root) if get_settings().runs_root else "runs",
    "source_kind": "auto",
    "agent": "default",
    "mode": "rlm",
    "model": None,
    "provider": None,
    "verification_provider": None,
    "hints": None,
    "n_paths": 3,
    # Default flipped 2026-05-25: "max" removes the per-agent turn / tool-call /
    # 20-min wall-clock caps that bound the "efficient" profile. Run-level
    # budgets (--max-wall-clock, --max-usd) still bind. User explicitly asked
    # for "no compute limitations" as the default; turn caps were biting the
    # paper-faithful reproductions.
    "execution_mode": "max",
    "sandbox": DEFAULT_SANDBOX_MODE.value,
    "gpu_mode": "auto",
    "command_timeout": None,
    "allow_sandbox_network": False,
    "sandbox_platform": None,
    "sandbox_memory": None,
    "sandbox_cpus": None,
    "max_usd": None,
    "max_wall_clock": None,
    "max_pod_seconds": None,
    "max_rlm_iterations": None,
    "max_invocations": None,
    "seed": None,
    "attempt_id": None,
    "run_group_id": None,
    "blacklist": None,
    "project_id": None,
    "paper_hint": None,
    "scope_spec": None,
    # Lane Q — defaults to False (strict reproduction). Set to True via
    # --minimize-compute or the lab UI "Minimize compute" checkbox.
    "minimize_compute": False,
}


def _with_reproduce_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Backfill argparse defaults for generated Namespace callers."""
    for name, value in _REPRODUCE_DEFAULTS.items():
        if not hasattr(args, name):
            setattr(args, name, value)
    return args


def _find_latest_rdr_project_dir(runs_root: Path, paper_id: str) -> str | None:
    """Find the most-recently-modified ``pb_<paper_id>_*`` directory under *runs_root*.

    Returns the directory *name* (not the full path) so the caller can reuse it
    as a project_id, or ``None`` when no match is found.
    """
    import re as _re

    safe_pid = _re.sub(r"[^a-zA-Z0-9_\-]", "_", paper_id)
    prefix = f"pb_{safe_pid}_"
    candidates = [
        d for d in runs_root.iterdir()
        if d.is_dir() and d.name.startswith(prefix)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime).name


def _cmd_reproduce_rdr(args: argparse.Namespace, runs_root: Path) -> int:
    """Dispatch ``--mode rdr``: run the rubric-driven harness on a PaperBench bundle.

    The positional ``source`` arg is the bundle paper_id (directory name under
    ``third_party/paperbench/``) or an absolute path to the bundle directory.
    Bypasses the standard ingest pipeline — the bundle carries its own paper.md
    and rubric.json.
    """
    import re
    import time

    from backend.agents.rdr.run import run_pipeline_rdr

    paper_id = args.source
    resume: bool = getattr(args, "resume", False)
    project_id_override: str | None = getattr(args, "project_id", None)

    def _safe_dir_name(s: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_\-]", "_", s)

    if resume:
        # Reuse an existing run directory: explicit --project-id wins, then
        # most-recently-modified pb_<paper_id>_* dir, then a fresh timestamped one.
        if project_id_override:
            project_id = project_id_override
        else:
            found = _find_latest_rdr_project_dir(runs_root, paper_id)
            if found:
                project_id = found
            else:
                logger.warning(
                    "[rdr] --resume: no prior run dir found for %s — starting fresh", paper_id
                )
                raw_id = f"pb_{_safe_dir_name(paper_id)}_{int(time.time())}"
                project_id = raw_id[:80]
    else:
        if project_id_override:
            project_id = project_id_override
        else:
            raw_id = f"pb_{_safe_dir_name(paper_id)}_{int(time.time())}"
            project_id = raw_id[:80]

    max_repair_iterations: int = getattr(args, "max_repair_iterations", 2)
    repair_target: float = getattr(args, "repair_target", 0.6)

    from backend.agents.execution import resolve_sandbox_mode
    from backend.agents.resilience import RunBudget

    _max_pod_seconds = _resolve_max_pod_seconds(getattr(args, "max_pod_seconds", None))
    _max_invocations = _max_invocations_from_arg(getattr(args, "max_invocations", None))
    run_budget = None
    if (
        getattr(args, "max_usd", None) is not None
        or getattr(args, "max_wall_clock", None) is not None
        or _max_pod_seconds is not None
        or _max_invocations
    ):
        run_budget = RunBudget(
            max_usd=getattr(args, "max_usd", None),
            max_wall_clock_seconds=getattr(args, "max_wall_clock", None),
            max_pod_seconds=_max_pod_seconds,
            max_invocations_per_agent=_max_invocations,
        )
    sandbox_mode = resolve_sandbox_mode(args.sandbox, pipeline_mode="rdr")

    print(f"[rdr] paper_id  : {paper_id}", file=sys.stderr)
    print(f"[rdr] project_id: {project_id}", file=sys.stderr)
    print(f"[rdr] runs_root : {runs_root}", file=sys.stderr)
    print(f"[rdr] sandbox   : {sandbox_mode.value}", file=sys.stderr)
    if resume:
        print(f"[rdr] resume    : True", file=sys.stderr)

    try:
        rdr_result = asyncio.run(
            run_pipeline_rdr(
                project_id,
                runs_root,
                paper_id=paper_id,
                provider=getattr(args, "provider", None),
                model=getattr(args, "model", None),
                sandbox_mode=sandbox_mode,
                max_repair_iterations=max_repair_iterations,
                repair_target=repair_target,
                resume=resume,
                run_budget=run_budget,
                cluster_concurrency=getattr(args, "cluster_concurrency", None),
            )
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        print(
            "\n[reprolab] RDR pipeline interrupted (Ctrl-C). Exiting.",
            file=sys.stderr,
            flush=True,
        )
        return 130

    result = {
        "project_id": rdr_result.project_id,
        "status": rdr_result.status,
        "rubric_score": rdr_result.rubric_score,
        "clusters_total": rdr_result.clusters_total,
        "clusters_failed": rdr_result.clusters_failed,
        "repair_iterations": rdr_result.repair_iterations,
        "final_report_path": rdr_result.final_report_path,
        "cost_usd": rdr_result.cost_usd,
        "output_dir": str(runs_root / rdr_result.project_id),
    }
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if rdr_result.status in ("completed", "partial") else 3


def _is_paperbench_bundle_id(source: str, runs_root: Path) -> bool:  # noqa: ARG001
    """Return True when *source* names a vendored PaperBench bundle directory.

    The source is treated as a bundle ID — not an arXiv/DOI/PDF — when it
    matches a subdirectory inside ``third_party/paperbench/`` relative to the
    repo root.  Absolute paths, arXiv IDs, and DOIs all fail this check and
    fall through to the normal ingest pipeline.
    """
    # Repo root = the parent of the ``backend/`` package (two levels up from cli.py).
    repo_root = Path(__file__).resolve().parent.parent
    bundles_root = repo_root / "third_party" / "paperbench"
    return (bundles_root / source).is_dir()


def _cmd_reproduce_rlm_paperbench(args: argparse.Namespace, runs_root: Path) -> int:
    """Dispatch ``--mode rlm`` on a vendored PaperBench bundle.

    Mirror of ``_cmd_reproduce_rdr`` for the RLM path: the source arg is a
    bundle paper_id, not an arXiv/DOI.  This lets callers use:

        python -m backend.cli reproduce sequential-neural-score-estimation \\
            --mode rlm --model claude-oauth --sandbox local

    without needing the arxiv ID or a full ``scripts/rlm_paperbench.py`` invocation.
    """
    import re
    import time

    paper_id = args.source
    project_id_override: str | None = getattr(args, "project_id", None)

    def _safe(s: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_\-]", "_", s)

    if project_id_override:
        project_id = project_id_override
    else:
        project_id = f"pb_{_safe(paper_id)}_{int(time.time())}"[:80]

    # Load bundle from the canonical third_party location.
    repo_root = Path(__file__).resolve().parent.parent
    bundles_root = repo_root / "third_party" / "paperbench"

    from backend.evals.paperbench.bundle import load_paperbench_bundle, PaperBenchBundleError
    try:
        bundle = load_paperbench_bundle(bundles_root, paper_id)
    except PaperBenchBundleError as exc:
        print(f"[rlm] PaperBench bundle error: {exc}", file=sys.stderr)
        return 2

    from backend.services.ingestion.paperbench import bundle_to_workspace_claim_map
    workspace_claim_map = bundle_to_workspace_claim_map(bundle)
    # Override the project_id so artifacts land in the right directory.
    workspace_claim_map["project_id"] = project_id
    # Pass the bundle's rubric into the claim map so run_pipeline_rlm skips
    # the rubric-generation step (which would try to call the LLM unnecessarily
    # when the bundle already ships a complete rubric.json).
    workspace_claim_map["rubric_spec"] = bundle.rubric()

    from backend.agents.execution import (
        ExecutionProfile,
        ensure_sandbox_mode_available,
        resolve_sandbox_mode,
    )
    from backend.agents.resilience import RunBudget
    from backend.services.runtime import SandboxRuntimeError

    execution_profile = ExecutionProfile.from_mode(
        getattr(args, "execution_mode", "max"),
        command_timeout_seconds=getattr(args, "command_timeout", None),
        sandbox_network_disabled=not getattr(args, "allow_sandbox_network", False),
        sandbox_memory_limit=getattr(args, "sandbox_memory", None),
        sandbox_cpus=getattr(args, "sandbox_cpus", None),
        sandbox_platform=getattr(args, "sandbox_platform", None),
        gpu_mode=getattr(args, "gpu_mode", "auto"),
    )
    # PR-μ.2: thread execution_mode for resolve_experiment_timeout_s (see other call site).
    os.environ["REPROLAB_EXECUTION_MODE"] = execution_profile.mode.value
    run_budget = None
    _max_pod_seconds = _resolve_max_pod_seconds(getattr(args, "max_pod_seconds", None))
    _max_invocations = _max_invocations_from_arg(getattr(args, "max_invocations", None))
    if (
        getattr(args, "max_usd", None) is not None
        or getattr(args, "max_wall_clock", None) is not None
        or _max_pod_seconds is not None
        or _max_invocations
    ):
        run_budget = RunBudget(
            max_usd=getattr(args, "max_usd", None),
            max_wall_clock_seconds=getattr(args, "max_wall_clock", None),
            max_pod_seconds=_max_pod_seconds,
            max_invocations_per_agent=_max_invocations,
        )
    sandbox_mode = resolve_sandbox_mode(getattr(args, "sandbox", "auto"), pipeline_mode="rlm")

    print(f"[rlm] paper_id  : {paper_id}", file=sys.stderr)
    print(f"[rlm] project_id: {project_id}", file=sys.stderr)
    print(f"[rlm] runs_root : {runs_root}", file=sys.stderr)
    print(f"[rlm] sandbox   : {sandbox_mode.value}", file=sys.stderr)

    try:
        ensure_sandbox_mode_available(sandbox_mode)
    except SandboxRuntimeError as exc:
        print(f"[rlm] Sandbox preflight failed: {exc}", file=sys.stderr)
        return 2

    # Route by mode: default 'rlm' → hybrid; 'rlm-pure' → pure RLM.
    mode = getattr(args, "mode", "rlm")
    if mode == "rlm-pure":
        print(f"[rlm-pure] paper_id  : {paper_id}", file=sys.stderr)
        print(f"[rlm-pure] project_id: {project_id}", file=sys.stderr)
        _runner_label = "rlm-pure"
    else:
        print(f"[hybrid] paper_id  : {paper_id}", file=sys.stderr)
        print(f"[hybrid] project_id: {project_id}", file=sys.stderr)
        _runner_label = "hybrid"

    try:
        if mode == "rlm-pure":
            from backend.agents.rlm.run import run_pipeline_rlm
            rlm_result = asyncio.run(run_pipeline_rlm(
                project_id,
                runs_root,
                workspace_claim_map,
                model=getattr(args, "model", None),
                provider=getattr(args, "provider", None),
                run_budget=run_budget,
                sandbox_mode=sandbox_mode,
                seed=getattr(args, "seed", None),
                execution_profile=execution_profile,
                attempt_id=getattr(args, "attempt_id", None),
                run_group_id=getattr(args, "run_group_id", None),
            ))
        else:
            # Default: hybrid (Phase 1 RDR + Phase 2 RLM repair)
            from backend.agents.hybrid.controller import run_pipeline_hybrid
            rlm_result = asyncio.run(run_pipeline_hybrid(
                project_id,
                runs_root,
                workspace_claim_map,
                model=getattr(args, "model", None),
                provider=getattr(args, "provider", None),
                run_budget=run_budget,
                sandbox_mode=sandbox_mode,
                seed=getattr(args, "seed", None),
                execution_profile=execution_profile,
                attempt_id=getattr(args, "attempt_id", None),
                run_group_id=getattr(args, "run_group_id", None),
                cluster_concurrency=getattr(args, "cluster_concurrency", None),
            ))
    except (KeyboardInterrupt, asyncio.CancelledError):
        print(
            f"\n[reprolab] {_runner_label} pipeline interrupted (Ctrl-C). Exiting.",
            file=sys.stderr,
            flush=True,
        )
        # Mirror the RLM/hybrid path: write status=stopped so the dashboard
        # doesn't show this as "running" indefinitely.
        _mark_demo_status_stopped(
            runs_root, project_id,
            reason=f"{_runner_label} pipeline interrupted (Ctrl-C)",
        )
        return 130
    except Exception as exc:
        from backend.agents.resilience import BudgetExhausted
        if isinstance(exc, BudgetExhausted):
            print(f"[{_runner_label}] Pipeline budget exhausted: {exc}", file=sys.stderr)
            _mark_demo_status_failed(
                runs_root, project_id,
                reason=f"{_runner_label} pipeline budget exhausted: {exc}",
            )
            return 3
        # PR-ν.3 / P3 — see RLM/hybrid path for rationale. Mirror the same
        # guard so a PaperBench-mode crash also reaches a terminal status.
        _mark_demo_status_failed(
            runs_root, project_id,
            reason=f"{_runner_label} pipeline crashed: {type(exc).__name__}: {exc}",
        )
        raise

    result = {
        "project_id": rlm_result.project_id,
        "status": rlm_result.status,
        "output_dir": str(runs_root / rlm_result.project_id),
        "iterations": rlm_result.iterations,
        "rubric_score": rlm_result.rubric_score,
        "cost_usd": rlm_result.cost_usd,
        "final_report_path": rlm_result.final_report_path,
    }
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if rlm_result.status in ("completed", "partial") else 3


def cmd_reproduce(args: argparse.Namespace) -> int:
    """Full pipeline: ingest a paper, build workspace, run agent pipeline."""
    args = _with_reproduce_defaults(args)
    # Cross-platform path normalization — converts Windows paths to WSL mount
    # paths, strips surrounding quotes, expands tilde-home, etc.
    # Identity-preserving for non-path inputs (arXiv IDs, URLs, DOIs).
    from backend.services.paths import normalize_path_input
    args.source = normalize_path_input(args.source)
    # Tier 2a — wire pipeline.log/jsonl on the root logger before any agent
    # module gets a chance to emit. This is the *subprocess* hot path
    # (live_runs.py spawns `python -c "from backend.cli import cmd_reproduce; ..."`),
    # so configuring here ensures the agent logs land in logs/<TS>/ alongside
    # the server logs. No-op when REPROLAB_LOG_DIR / REPROLAB_RUNS_ROOT unset.
    from backend.observability.run_logging import configure_root_logger
    configure_root_logger()
    runs_root = Path(args.runs_root)

    # Dynamic GPU CLI overrides: set env vars BEFORE any Settings construction so
    # pydantic-settings picks them up. Non-None CLI values override env defaults.
    import os as _os
    if getattr(args, "dynamic_gpu", None) is not None:
        _os.environ["REPROLAB_DYNAMIC_GPU"] = "true" if args.dynamic_gpu else "false"
    if getattr(args, "force_single_gpu", None) is not None:
        _os.environ["REPROLAB_FORCE_SINGLE_GPU"] = "true" if args.force_single_gpu else "false"
    if getattr(args, "max_gpu_usd_per_hour", None) is not None:
        _os.environ["REPROLAB_MAX_GPU_USD_PER_HOUR"] = str(args.max_gpu_usd_per_hour)
    if getattr(args, "max_run_gpu_usd", None) is not None:
        _os.environ["REPROLAB_MAX_RUN_GPU_USD"] = str(args.max_run_gpu_usd)
    if getattr(args, "dynamic_gpu_headroom", None) is not None:
        _os.environ["REPROLAB_DYNAMIC_GPU_HEADROOM"] = str(args.dynamic_gpu_headroom)
    if getattr(args, "vram_gb", None) is not None:
        _os.environ["REPROLAB_VRAM_OVERRIDE_GB"] = str(args.vram_gb)
    _max_rlm_iter = getattr(args, "max_rlm_iterations", None)
    if _max_rlm_iter is not None and _max_rlm_iter > 0:
        _os.environ["REPROLAB_MAX_RLM_ITERATIONS"] = str(_max_rlm_iter)
    elif _max_rlm_iter == 0:
        # Explicit 0 disables the cap; clear any inherited env var.
        _os.environ.pop("REPROLAB_MAX_RLM_ITERATIONS", None)

    # Paper-hint + operator scope-spec composition. The two flags are independent —
    # either, both, or neither may be set. Result is persisted via env vars so the
    # spawned subprocess (cmd_reproduce → run_pipeline_hybrid/_rlm → RunContext)
    # picks them up uniformly. This mirrors the dynamic-GPU env-var pattern above.
    from backend.agents.prompts.paper_hints import lookup_paper_hint as _lookup_hint
    _paper_hint_obj = _lookup_hint(getattr(args, "paper_hint", None))
    _operator_scope = _load_scope_spec_arg(getattr(args, "scope_spec", None))
    _effective_scope = _operator_scope.merge_with_paper_default(
        _paper_hint_obj.default_scope if _paper_hint_obj is not None else None
    )
    _os.environ["REPROLAB_SCOPE_SPEC_JSON"] = _effective_scope.model_dump_json()

    if _paper_hint_obj is not None and _paper_hint_obj.guidance:
        _existing_guidance = _os.environ.get("REPROLAB_BASELINE_EXTRA_GUIDANCE", "").strip()
        _hint_id = args.paper_hint
        _hint_text = f"[paper-hint {_hint_id}] {_paper_hint_obj.guidance}"
        _os.environ["REPROLAB_BASELINE_EXTRA_GUIDANCE"] = (
            f"{_hint_text}\n\n{_existing_guidance}" if _existing_guidance else _hint_text
        )
        print(
            f"[paper-hint] Applied {_hint_id} ({len(_paper_hint_obj.guidance)} chars guidance, "
            f"{len(_paper_hint_obj.invariants)} invariants, "
            f"{'scope' if _paper_hint_obj.default_scope else 'no scope'}).",
            file=sys.stderr,
        )
    elif getattr(args, "paper_hint", None):
        print(
            f"[paper-hint] No built-in hint for {args.paper_hint!r}; continuing without one.",
            file=sys.stderr,
        )
    if getattr(args, "scope_spec", None):
        print(
            f"[scope] Effective scope: models={_effective_scope.models or '∅'}, "
            f"datasets={_effective_scope.dataset_ids() or '∅'}, "
            f"seeds={_effective_scope.seeds or '∅'}.",
            file=sys.stderr,
        )

    # rdr mode: rubric-driven harness on a vendored PaperBench bundle.
    # Bypasses the ingest pipeline entirely — the positional `source` arg is
    # treated as a bundle paper_id (or absolute path), not a PDF/arXiv/DOI.
    if args.mode == "rdr":
        return _cmd_reproduce_rdr(args, runs_root)

    # rlm (default hybrid) or rlm-pure with a PaperBench bundle ID:
    # bypass ingest, load the bundle directly.
    if args.mode in ("rlm", "rlm-pure") and _is_paperbench_bundle_id(args.source, runs_root):
        return _cmd_reproduce_rlm_paperbench(args, runs_root)

    provider = getattr(args, "provider", None)
    verification_provider = getattr(args, "verification_provider", None)

    # --- Phase 1: Ingest ---
    store, intake, parser, discovery, indexer, workspace = _make_services(
        args.database_url, runs_root
    )

    source = _source_from_cli(args.source, args.source_kind)

    # --fresh: purge the prior run before registering so the first append
    # does not hit a ConcurrencyError on the existing aggregate.
    if getattr(args, "fresh", False):
        fresh_pid = project_id_for(source)
        run_dir = runs_root / fresh_pid
        shutil.rmtree(run_dir, ignore_errors=True)
        purged = store.purge_project_aggregates(fresh_pid)
        print(
            f"[--fresh] Purged runs/{fresh_pid}/ and {purged} event-store rows.",
            file=sys.stderr,
        )
    else:
        # PR-π Module D — resume offer: check for a prior interrupted run BEFORE
        # archiving so rlm_state/ is still readable for the iteration count.
        _presumed_pid_early = getattr(args, "project_id", None) or project_id_for(source)
        _prior_project_dir = runs_root / _presumed_pid_early
        if not getattr(args, "resume", False) and _offer_resume(_prior_project_dir):
            args.resume = True

        # Archive prior-attempt artifacts (final_report.*, experiment_runs.jsonl,
        # cost_ledger.jsonl, dashboard_events.jsonl, rlm_state/, etc.) under
        # runs/<id>/attempts/<ts>/ so the new attempt does not commingle with
        # an older one in the UI or the final report. The ingested paper is
        # preserved so this does NOT trigger a re-fetch / re-parse.
        from backend.services.runs.archive import archive_run_artifacts
        presumed_pid = _presumed_pid_early
        archived = archive_run_artifacts(presumed_pid, runs_root)
        if archived:
            print(
                f"[archive] Moved {len(archived['moved'])} prior-attempt artifact(s) "
                f"to {archived['attempt_dir']}",
                file=sys.stderr,
            )

    print(f"[ingest 1/6] Registering project for {args.source}", file=sys.stderr)
    project_id = intake.register_project(RegisterProject(source=source))
    # T15 / handoff P1-I8: when the REST API spawns the CLI it passes --project-id
    # so the CLI writes to the same runs/<id>/ directory the API watches.  The
    # override replaces the source-derived id *after* registration so the event-
    # store aggregate (keyed by the source-derived id) is still created correctly.
    if getattr(args, "project_id", None):
        project_id = args.project_id
    print(f"             project_id={project_id}", file=sys.stderr)

    print("[ingest 2/6] Fetching paper", file=sys.stderr)
    if not intake.fetch_paper(FetchPaper(project_id=project_id)):
        print("             FAILED — see paper_fetch_failed event", file=sys.stderr)
        return 1

    print("[ingest 3/6] Parsing", file=sys.stderr)
    if not parser.start_parsing(StartParsing(project_id=project_id)):
        print("             FAILED — see parsing_failed event", file=sys.stderr)
        return 1

    print("[ingest 4/6] Discovering external artifacts", file=sys.stderr)
    if not discovery.discover(DiscoverArtifacts(project_id=project_id)):
        print("             FAILED — see discovery_failed event", file=sys.stderr)
        return 1

    print("[ingest 5/6] Indexing", file=sys.stderr)
    if not indexer.start_indexing(StartIndexing(project_id=project_id)):
        print("             FAILED — see indexing_failed event", file=sys.stderr)
        return 1

    print("[ingest 6/6] Building workspace", file=sys.stderr)
    workspace_id = workspace.build_workspace(
        BuildWorkspace(project_id=project_id, agent_name=args.agent)
    )
    view = workspace.materialize_view(workspace_id)

    workspace_claim_map = _build_workspace_claim_map(
        view.variables, project_id, runs_root
    )

    # Lane U — write paper identity to demo_status.json so the lab UI
    # doesn't render "Untitled Paper". The CLI path (cmd_reproduce) was
    # missing this; only the API-side /runs/upload path was populating
    # paperTitle. Derive paperId from the source kind: arxiv → bare id,
    # pdf path → filename stem, doi → the doi string. Derive paperTitle
    # from the parser if it stored a title-like first section; otherwise
    # fall back to a readable variant of paperId (better than "Untitled").
    try:
        _ds_path = runs_root / project_id / "demo_status.json"
        _existing = {}
        if _ds_path.exists():
            try:
                _existing = json.loads(_ds_path.read_text(encoding="utf-8")) or {}
            except (json.JSONDecodeError, OSError):
                _existing = {}
        _paper_id = ""
        _paper_title = ""
        if isinstance(source, ArxivId):
            _paper_id = source.arxiv_id
            _paper_title = f"arXiv:{source.arxiv_id}"
        elif isinstance(source, PdfPath):
            _stem = Path(source.path).stem
            _paper_id = _stem
            _paper_title = _stem.replace("_", " ").replace("-", " ").strip() or _stem
        elif isinstance(source, DoiRef):
            _paper_id = source.doi
            _paper_title = f"doi:{source.doi}"
        # Try to upgrade _paper_title from the workspace claim map (the
        # parser may have extracted the real title into the first section).
        _entries = (workspace_claim_map or {}).get("entries") or []
        _first_title = (_entries[0].get("title") if _entries else "") or ""
        # Reject the noise titles the bundle path produces ("Abstract",
        # "Introduction", "1 Introduction"). Anything else is probably real.
        _is_noise = (not _first_title) or _first_title.strip().lower() in {
            "abstract", "introduction", "1 introduction", "1. introduction",
            "summary", "overview",
        }
        if not _is_noise:
            _paper_title = _first_title.strip()
        _existing.update({
            "paperId": _paper_id,
            "paperTitle": _paper_title,
            "paper": {"id": _paper_id, "title": _paper_title},
        })
        # Atomic write so a crash mid-write leaves either old or new JSON.
        _tmp = _ds_path.with_suffix(_ds_path.suffix + ".tmp")
        _tmp.parent.mkdir(parents=True, exist_ok=True)
        _tmp.write_text(json.dumps(_existing, indent=2), encoding="utf-8")
        os.replace(_tmp, _ds_path)
    except Exception:  # noqa: BLE001 — title rendering must not block the run
        pass

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Workspace ready — {len(view.variables)} variables", file=sys.stderr)

    provider_note = f", provider={provider}" if provider else ""
    verification_note = (
        f", verification_provider={verification_provider}"
        if verification_provider
        else ""
    )
    print(f"Starting agent pipeline ({args.mode} mode{provider_note}{verification_note})...", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    # --- Phase 2: Agent Pipeline ---
    user_hints = [h.strip() for h in args.hints.split(",")] if args.hints else None
    blacklist_terms = _blacklist_entries_from_arg(args.blacklist)
    from backend.agents.execution import (
        ExecutionProfile,
        ensure_sandbox_mode_available,
        resolve_sandbox_mode,
    )
    from backend.services.runtime import SandboxRuntimeError

    execution_profile = ExecutionProfile.from_mode(
        args.execution_mode,
        command_timeout_seconds=args.command_timeout,
        sandbox_network_disabled=not args.allow_sandbox_network,
        sandbox_memory_limit=args.sandbox_memory,
        sandbox_cpus=args.sandbox_cpus,
        sandbox_platform=args.sandbox_platform,
        gpu_mode=getattr(args, "gpu_mode", "auto"),
        minimize_compute=getattr(args, "minimize_compute", False),
    )
    # PR-μ.2: thread execution_mode through to resolve_experiment_timeout_s.
    # RunContext doesn't currently carry execution_mode, so the resolver in
    # primitives.py falls back to this env var when ctx.execution_mode is None.
    # Without this, max-mode capped at the 7200s default instead of the
    # EXPERIMENT_TIMEOUT_BY_MODE["max"]=21600s the user requested.
    os.environ["REPROLAB_EXECUTION_MODE"] = execution_profile.mode.value
    run_budget = None
    _max_pod_seconds = _resolve_max_pod_seconds(args.max_pod_seconds)
    _max_run_gpu_usd = getattr(args, "max_run_gpu_usd", None)
    if (
        args.max_usd is not None
        or args.max_wall_clock is not None
        or args.max_invocations
        or _max_pod_seconds is not None
        or _max_run_gpu_usd is not None
    ):
        from backend.agents.resilience import RunBudget

        run_budget = RunBudget(
            max_usd=args.max_usd,
            max_wall_clock_seconds=args.max_wall_clock,
            max_pod_seconds=_max_pod_seconds,
            max_invocations_per_agent=_max_invocations_from_arg(args.max_invocations),
            max_run_gpu_usd=_max_run_gpu_usd,
        )
    sandbox_mode = resolve_sandbox_mode(args.sandbox, pipeline_mode=args.mode)
    print(
        f"Execution profile: {execution_profile.mode.value}; sandbox: {sandbox_mode.value}",
        file=sys.stderr,
    )
    try:
        ensure_sandbox_mode_available(sandbox_mode)
    except SandboxRuntimeError as exc:
        print(f"Sandbox preflight failed: {exc}", file=sys.stderr)
        store.close()
        return 2

    try:
        if args.mode == "rlm-pure":
            # Escape hatch: pure RLM, no rubric decomposition.
            from backend.agents.rlm.run import run_pipeline_rlm

            rlm_result = asyncio.run(run_pipeline_rlm(
                project_id, runs_root, workspace_claim_map,
                model=args.model,
                provider=provider,
                run_budget=run_budget,
                sandbox_mode=sandbox_mode,
                seed=args.seed,
                execution_profile=execution_profile,
                attempt_id=args.attempt_id,
                run_group_id=args.run_group_id,
                workspace_service=workspace,
                workspace_id=workspace_id,
            ))
        elif args.mode in ("rlm", None):
            # Default: hybrid Phase 1 (RDR) + Phase 2 (RLM repair).
            from backend.agents.hybrid.controller import run_pipeline_hybrid

            rlm_result = asyncio.run(run_pipeline_hybrid(
                project_id, runs_root, workspace_claim_map,
                model=args.model,
                provider=provider,
                run_budget=run_budget,
                sandbox_mode=sandbox_mode,
                seed=args.seed,
                execution_profile=execution_profile,
                attempt_id=args.attempt_id,
                run_group_id=args.run_group_id,
                workspace_service=workspace,
                workspace_id=workspace_id,
                cluster_concurrency=getattr(args, "cluster_concurrency", None),
            ))
        else:
            print(
                f"Error: --mode {args.mode!r} is not supported here. "
                "Use --mode rlm (default hybrid), --mode rlm-pure (pure RLM), "
                "or --mode rdr instead.",
                file=sys.stderr,
            )
            store.close()
            return 1
    except (KeyboardInterrupt, asyncio.CancelledError):
        # Graceful interrupt: don't dump a stack trace. Flip
        # demo_status.json to status=stopped so the dashboard reflects
        # the right state instead of waiting for live_runs reconciliation
        # to mark the run "failed" via _pid_exists. Conventional SIGINT
        # exit code is 130.
        print(
            "\n[reprolab] Pipeline interrupted (Ctrl-C). "
            "Marking run as stopped and exiting.",
            file=sys.stderr,
            flush=True,
        )
        _mark_demo_status_stopped(
            runs_root,
            project_id,
            reason="Pipeline interrupted (Ctrl-C)",
        )
        return 130
    except Exception as exc:
        from backend.agents.resilience import BudgetExhausted

        if isinstance(exc, BudgetExhausted):
            # Budget exhaustion already writes a terminal status inside the
            # pipeline; just surface the message and return non-zero.
            print(f"Pipeline budget exhausted: {exc}", file=sys.stderr)
            _mark_demo_status_failed(
                runs_root, project_id,
                reason=f"Pipeline budget exhausted: {exc}",
            )
            return 3
        # PR-ν.3 / P3 — uncaught exception escaped the pipeline; ensure
        # demo_status.json is marked failed before we let it propagate.
        # Without this the dashboard shows the run as "running" forever
        # (audit pattern P3: 4 zombie-status runs in the 3-day window).
        _mark_demo_status_failed(
            runs_root, project_id,
            reason=f"Pipeline crashed: {type(exc).__name__}: {exc}",
        )
        raise
    finally:
        store.close()

    # Print final summary
    out_dir = runs_root / project_id
    # RLMRunResult — no PipelineState fields
    result = {
        "project_id": rlm_result.project_id,
        "status": rlm_result.status,
        "output_dir": str(out_dir),
        "iterations": rlm_result.iterations,
        "rubric_score": rlm_result.rubric_score,
        "cost_usd": rlm_result.cost_usd,
        "final_report_path": rlm_result.final_report_path,
        "execution_mode": execution_profile.mode.value,
        "sandbox": sandbox_mode.value,
    }
    # A4-6: a budget-exhausted (or otherwise failed) rlm run must exit
    # non-zero. run_pipeline_rlm never raises on budget breach — it
    # returns status="failed" (set by Batch O). Exit code 3 signals
    # budget exhaustion to callers (same as BudgetExhausted exceptions).
    if rlm_result.status == "failed":
        json.dump(result, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 3
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the reprolab argument parser (testable without calling main)."""
    parser = argparse.ArgumentParser(prog="reprolab")
    parser.add_argument(
        "--database-url",
        default=get_settings().database_url,
        help="SQLite URL for the event store (defaults to REPROLAB_DATABASE_URL).",
    )
    parser.add_argument(
        "--runs-root",
        default=str(get_settings().runs_root) if get_settings().runs_root else "runs",
        help="Per-project blob directory root (defaults to REPROLAB_RUNS_ROOT or ./runs).",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    ingest = sub.add_parser("ingest", help="Ingest a paper end-to-end.")
    ingest.add_argument("source", help="PDF path, arXiv id/URL, or DOI/doi.org URL.")
    ingest.add_argument(
        "--source-kind",
        choices=("auto", "pdf_path", "arxiv", "doi"),
        default="auto",
        help="How to interpret SOURCE (default: auto).",
    )
    ingest.add_argument("--agent", default="default", help="Agent name for the workspace.")
    ingest.set_defaults(func=cmd_ingest)

    inspect = sub.add_parser("inspect", help="Inspect a built workspace.")
    inspect.add_argument("project_id")
    inspect.add_argument("--agent", default="default")
    inspect.add_argument("--variable", default=None, help="Print one variable's full payload.")
    inspect.set_defaults(func=cmd_inspect)

    regen = sub.add_parser(
        "regenerate-report",
        help="Regenerate final_report.md from existing final_report.json + sidecars.",
    )
    regen.add_argument("project_id", help="Run project id (e.g., prj_03271ba130d423fe).")
    regen.set_defaults(func=cmd_regenerate_report)

    reproduce = sub.add_parser("reproduce", help="Full pipeline: ingest + agent pipeline.")
    reproduce.add_argument("source", help="PDF path, arXiv id/URL, or DOI/doi.org URL.")
    reproduce.add_argument(
        "--source-kind",
        choices=("auto", "pdf_path", "arxiv", "doi"),
        default="auto",
    )
    reproduce.add_argument("--agent", default="default", help="Agent name for the workspace.")
    reproduce.add_argument(
        "--mode", choices=("rlm", "rdr", "rlm-pure"), default="rlm",
        help=(
            "Pipeline mode: "
            "'rlm' (default) — hybrid Phase 1 RDR + Phase 2 RLM adaptive repair; "
            "'rdr' — pure rubric-driven harness, predictable cost ceiling (PaperBench bundles only); "
            "'rlm-pure' — pure RLM, no rubric decomposition (debug/escape hatch)."
        ),
    )
    reproduce.add_argument("--model", default=None, help="Model override for the RLM orchestrator.")
    reproduce.add_argument(
        "--provider",
        choices=("anthropic", "openai"),
        default=None,
        help="SDK provider override (defaults to REPROLAB_LLM_PROVIDER).",
    )
    reproduce.add_argument(
        "--verification-provider",
        choices=("anthropic", "openai"),
        default=None,
        help="Optional SDK provider for supervisor verification/review agents.",
    )
    reproduce.add_argument("--hints", default=None, help="Comma-separated user hints for improvement.")
    reproduce.add_argument("--n-paths", type=int, default=3, help="Number of improvement paths.")
    reproduce.add_argument(
        "--execution-mode",
        choices=("efficient", "max"),
        default="max",
        help=(
            "Execution profile (default `max` since 2026-05-25): "
            "`max` removes per-agent turn / tool-call / 20-min wall-clock caps; "
            "`efficient` re-enables them for cost-sensitive runs. Run-level "
            "budgets (--max-wall-clock, --max-usd) still bind in both modes."
        ),
    )
    reproduce.add_argument(
        "--sandbox",
        choices=("auto", "local", "docker", "runpod"),
        default=DEFAULT_SANDBOX_MODE.value,
        help=(
            f"Experiment backend (default: {DEFAULT_SANDBOX_MODE.value}). "
            "runpod uses a remote GPU Pod; docker is isolated local Docker; "
            "local runs commands on the host; auto resolves to the configured default."
        ),
    )
    reproduce.add_argument(
        "--gpu-mode",
        choices=("off", "auto", "prefer", "max"),
        default="auto",
        help=(
            "GPU policy for experiment sandboxes: off disables CUDA visibility; "
            "auto records intent without forcing GPUs; prefer/max request Docker GPUs when available."
        ),
    )
    reproduce.add_argument(
        "--command-timeout",
        type=int,
        default=None,
        help="Override per-command sandbox timeout in seconds.",
    )
    reproduce.add_argument(
        "--allow-sandbox-network",
        action="store_true",
        help="Allow network access inside Docker sandbox containers.",
    )
    reproduce.add_argument(
        "--sandbox-platform",
        default=None,
        help="Optional Docker platform, e.g. linux/amd64 for cross-architecture runs.",
    )
    reproduce.add_argument(
        "--sandbox-memory",
        default=None,
        help="Docker memory limit override, e.g. 4g or 8192m.",
    )
    reproduce.add_argument(
        "--sandbox-cpus",
        type=float,
        default=None,
        help="Docker CPU limit override.",
    )
    reproduce.add_argument(
        "--max-usd",
        type=float,
        default=None,
        help="Maximum estimated provider spend before blocking the next SDK invocation.",
    )
    reproduce.add_argument(
        "--max-wall-clock",
        type=float,
        default=None,
        help="Maximum whole-run wall-clock seconds before blocking the next SDK invocation.",
    )
    reproduce.add_argument(
        "--max-pod-seconds",
        type=float,
        default=None,
        help=(
            "Maximum elapsed seconds a RunPod pod may run AFTER SSH connect "
            "(not from POST /pods — boot time is not budgeted) before the next "
            "exec() raises BudgetExhausted and the pod is force-destroyed. "
            "Persistent pods (REPROLAB_RUNPOD_POD_ID) are NOT auto-deleted; "
            "an ERROR log is emitted and manual cleanup is required. "
            "Also read from REPROLAB_MAX_POD_SECONDS env var."
        ),
    )
    reproduce.add_argument(
        "--dynamic-gpu",
        dest="dynamic_gpu",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable dynamic GPU SKU selection from paper hardware clues (default: from REPROLAB_DYNAMIC_GPU).",
    )
    reproduce.add_argument(
        "--force-single-gpu",
        dest="force_single_gpu",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="When dynamic-gpu is on, cap GPU count at 1 (default: from REPROLAB_FORCE_SINGLE_GPU).",
    )
    reproduce.add_argument(
        "--max-gpu-usd-per-hour",
        dest="max_gpu_usd_per_hour",
        type=float,
        default=None,
        help="Per-GPU $/hr cap for SKU selection (default: from REPROLAB_MAX_GPU_USD_PER_HOUR=10.0).",
    )
    reproduce.add_argument(
        "--max-run-gpu-usd",
        dest="max_run_gpu_usd",
        type=float,
        default=None,
        help="Total RunPod USD cap per run (default: from REPROLAB_MAX_RUN_GPU_USD=10.0).",
    )
    reproduce.add_argument(
        "--dynamic-gpu-headroom",
        dest="dynamic_gpu_headroom",
        type=float,
        default=None,
        help="Multiplier on LLM VRAM estimate before tier-up (default: from REPROLAB_DYNAMIC_GPU_HEADROOM=1.25).",
    )
    reproduce.add_argument(
        "--vram-gb",
        dest="vram_gb",
        type=int,
        default=None,
        help="Manual VRAM override; bypasses LLM estimate but headroom multiplier still applies.",
    )
    reproduce.add_argument(
        "--max-invocations",
        default=None,
        help=(
            "Comma-separated per-agent invocation caps, e.g. "
            "paper-understanding=3,artifact-discovery=5."
        ),
    )
    reproduce.add_argument(
        "--no-cache",
        dest="no_cache",
        action="store_true",
        default=False,
        help=(
            "Disable the primitive_cache for this run (debugging aid). "
            "Equivalent to REPROLAB_PRIMITIVE_CACHE=disabled. Forces every "
            "cacheable primitive (understand_section, plan_reproduction, "
            "implement_baseline, verify_against_rubric, etc.) to recompute. "
            "Cache hits still validate via schema-on-hit even when enabled, "
            "so use this only when you suspect poisoning or want a clean baseline."
        ),
    )
    reproduce.add_argument(
        "--minimize-compute",
        dest="minimize_compute",
        action="store_true",
        default=False,
        help=(
            "Reproduce the paper's CLAIM, not its recipe. The agent gets a "
            "prompt block instructing it to swap slow paper schedules "
            "(SGD+linear-decay-from-10 over 3000 epochs) for modern fast "
            "equivalents (Adam@lr=0.001 over 200-500 epochs), and to record "
            "each substitution in scope.declared_reductions so the "
            "scope-adjusted rubric scores the metric match (paper's reported "
            "test error / accuracy / etc.) rather than the recipe-step count. "
            "Use when budget is tight or the paper's training schedule is "
            "obviously a historical artefact (slow optimizers, excess epochs). "
            "Off by default — strict reproduction is the safer baseline."
        ),
    )
    reproduce.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed to pass through prompts, state, and generated experiment configs.",
    )
    reproduce.add_argument(
        "--attempt-id",
        default=None,
        help="Stable identifier for one seeded pipeline attempt.",
    )
    reproduce.add_argument(
        "--run-group-id",
        default=None,
        help="Identifier shared by multiple seeded attempts.",
    )
    reproduce.add_argument(
        "--blacklist",
        default=None,
        help=(
            "Comma-separated blocked URLs/terms, or a path to a newline-delimited "
            "PaperBench blacklist file."
        ),
    )
    reproduce.add_argument(
        "--fresh",
        action="store_true",
        default=False,
        help=(
            "Purge any prior run of the same paper before starting. "
            "Deletes runs/<project_id>/ and all event-store rows for the project "
            "so the re-run starts with a clean slate."
        ),
    )
    reproduce.add_argument(
        "--max-rlm-iterations",
        dest="max_rlm_iterations",
        type=int,
        default=None,
        help=(
            "(rlm mode) hard cap on the total number of RLM root-loop iterations. "
            "When the root reaches this count, FINAL_VAR is accepted unconditionally "
            "and the best partial report is shipped. Default: from "
            "REPROLAB_MAX_RLM_ITERATIONS env var (default 5). "
            "Set to 0 to disable the cap."
        ),
    )
    reproduce.add_argument(
        "--max-repair-iterations",
        dest="max_repair_iterations",
        type=int,
        default=2,
        help="(rdr mode) maximum repair-pass iterations after initial scoring.",
    )
    reproduce.add_argument(
        "--repair-target",
        dest="repair_target",
        type=float,
        default=0.6,
        help="(rdr mode) cluster score threshold below which a cluster is queued for repair.",
    )
    reproduce.add_argument(
        "--cluster-concurrency",
        dest="cluster_concurrency",
        type=int,
        default=None,
        help=(
            "(rdr/rlm modes) maximum number of Code Development clusters to dispatch "
            "concurrently. Code Execution and Result Analysis clusters always run "
            "sequentially (they depend on Code Dev outputs). Default 8 — also "
            "configurable via RDR_CLUSTER_CONCURRENCY env var. Pass 1 to force "
            "fully sequential execution."
        ),
    )
    reproduce.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help=(
            "(rdr mode) resume from the most-recently-modified run directory for the "
            "same paper_id, skipping clusters that have existing checkpoints. "
            "Use after a watchdog kill (exit 124) to continue without restarting from scratch."
        ),
    )
    reproduce.add_argument(
        "--project-id",
        dest="project_id",
        default=None,
        help=(
            "Override the project id (writes to runs/<project-id>/). When unset, "
            "an id is derived from the paper source. Used by the REST API so the "
            "spawned CLI writes to the directory the API watches (T15 / P1-I8). "
            "In rdr mode, also used with --resume to target a specific prior run."
        ),
    )
    reproduce.add_argument(
        "--paper-hint",
        dest="paper_hint",
        default=None,
        help=(
            "Paper-specific hint id (typically an arXiv id, e.g. 2605.15155). "
            "Looks up PaperHint from backend.agents.prompts.paper_hints.PAPER_HINTS. "
            "Composes three independent layers: appends .guidance to "
            "REPROLAB_BASELINE_EXTRA_GUIDANCE (with [paper-hint <id>] prefix); "
            "merges .default_scope under any operator --scope-spec via "
            "ScopeSpec.merge_with_paper_default; .invariants ride along to PR D's "
            "rubric scorer. Unknown ids are silently ignored — the run continues."
        ),
    )
    reproduce.add_argument(
        "--scope-spec",
        dest="scope_spec",
        default=None,
        help=(
            "Operator-stated reproduction scope. Accepts EITHER an inline JSON "
            "object (e.g. '{\"models\":[\"Qwen3-1.7B\"],\"seeds\":[42]}') OR a "
            "path to a JSON file. Detection rule: a value starting with '{' is "
            "treated as inline JSON; anything else is read as a filesystem path. "
            "Merges under any --paper-hint default_scope via "
            "ScopeSpec.merge_with_paper_default — operator fields win, absences "
            "fall back to paper defaults."
        ),
    )
    reproduce.set_defaults(func=cmd_reproduce)

    from backend.cli_paperbench import add_paperbench_subparser
    add_paperbench_subparser(sub)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    # --no-cache: disable primitive_cache as early as possible so no module
    # that imports primitive_cache caches a stale ``is_enabled()`` read.
    if getattr(args, "no_cache", False):
        os.environ["REPROLAB_PRIMITIVE_CACHE"] = "disabled"
    return int(args.func(args))


# ---------------------------------------------------------------------------
# PR-π Module D helpers — orphan sweep + resume offer
# ---------------------------------------------------------------------------


def _count_iterations(project_dir: Path) -> int:
    """Count completed iterations from rlm_state/iterations.jsonl."""
    iters_path = project_dir / "rlm_state" / "iterations.jsonl"
    if not iters_path.exists():
        return 0
    count = 0
    try:
        for line in iters_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                count += 1
    except OSError:
        pass
    return count


def _read_last_rubric(project_dir: Path) -> float:
    """Read the last rubric overall_score from rlm_state/ or final_report.json."""
    # Try final_report.json first (may exist from a prior partial run).
    for p in (project_dir / "final_report.json", project_dir / "final_report.json"):
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                score = (data.get("rubric") or {}).get("overall_score")
                if score is not None:
                    return float(score)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass
    # Try dashboard_events.jsonl for the last rubric_score event.
    events_path = project_dir / "dashboard_events.jsonl"
    if events_path.exists():
        last_score: float | None = None
        try:
            for line in events_path.read_text(encoding="utf-8").splitlines():
                if '"rubric_score"' not in line:
                    continue
                try:
                    ev = json.loads(line)
                    if ev.get("event") == "rubric_score":
                        s = ev.get("overall_score")
                        if s is not None:
                            last_score = float(s)
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
        except OSError:
            pass
        if last_score is not None:
            return last_score
    return 0.0


def _offer_resume(project_dir: Path) -> bool:
    """Check for an interrupted prior run and offer to resume.

    Returns True if the user agreed to resume (or non-interactively the
    run is skipped). Returns False if there is no prior interrupted run or
    the user declined. Only prompts when stdin is a TTY.
    """
    import sys

    status_path = project_dir / "demo_status.json"
    if not status_path.exists():
        return False
    try:
        prior = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if prior.get("status") != "interrupted":
        return False

    last_iter = _count_iterations(project_dir)
    last_rubric = _read_last_rubric(project_dir)
    print(
        f"Detected interrupted prior run for {project_dir.name} "
        f"(iter={last_iter}, last_rubric={last_rubric:.2f})."
    )

    if not sys.stdin.isatty():
        return False

    try:
        answer = input("Resume from last checkpoint? [Y/n] ").strip().lower()
    except EOFError:
        return False
    return answer in {"", "y", "yes"}


def _module_main(argv: list[str] | None = None) -> None:
    """Entrypoint for ``python -m backend.cli``.

    Reproduction runs may leave SDK cleanup threads behind after a bounded
    timeout. Once ``main`` has closed stores and printed its result, bypass
    interpreter atexit cleanup for that subcommand so the CLI itself remains
    bounded.

    PR-π Module D: sweep orphaned runs at startup so stale status=running
    entries are converted to status=interrupted before the new run starts.
    """
    selected_argv = list(sys.argv[1:] if argv is None else argv)

    # Orphan sweep — best-effort; never crashes the CLI.
    try:
        from backend.services.events.run_liveness import sweep_orphaned_runs
        _settings = get_settings()
        _runs_root = Path(_settings.runs_root) if _settings.runs_root else Path("runs")
        if _runs_root.exists():
            orphans = sweep_orphaned_runs(_runs_root)
            if orphans:
                print(f"[orphan-sweep] marked {len(orphans)} interrupted run(s):")
                for o in orphans:
                    print(f"  {o.project_id}  ({o.reason})")
    except Exception:  # noqa: BLE001 — never crash the CLI due to sweep failure
        logger.debug("_module_main: orphan sweep failed (non-fatal)", exc_info=True)

    code = main(argv)
    if selected_argv and selected_argv[0] == "reproduce":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    raise SystemExit(code)


def _source_from_cli(raw: str, source_kind: str):
    if source_kind == "pdf_path":
        return PdfPath(path=str(Path(raw).expanduser().resolve()))
    if source_kind == "arxiv":
        return ArxivId(arxiv_id=raw)
    if source_kind == "doi":
        return DoiRef(doi=raw)

    path = Path(raw).expanduser()
    if path.exists():
        return PdfPath(path=str(path.resolve()))
    if _ARXIV_RE.search(raw.strip()):
        return ArxivId(arxiv_id=raw)
    return DoiRef(doi=raw)


def _blacklist_entries_from_arg(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    candidate = Path(raw).expanduser()
    if candidate.exists() and candidate.is_file():
        return tuple(
            line.strip()
            for line in candidate.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _load_scope_spec_arg(raw: str | None):
    """Parse --scope-spec value (inline JSON or filesystem path) into a ScopeSpec.

    Inline detection: a value starting with '{' is parsed as JSON directly;
    anything else is treated as a path. Missing paths raise FileNotFoundError
    so a typo never silently produces an empty scope.
    """
    from backend.agents.schemas import ScopeSpec
    if not raw or not raw.strip():
        return ScopeSpec()
    text = raw.strip()
    if text.startswith("{"):
        return ScopeSpec.model_validate_json(text)
    path = Path(text).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"--scope-spec: {path} does not exist")
    return ScopeSpec.model_validate_json(path.read_text(encoding="utf-8"))


def _resolve_max_pod_seconds(cli_value: float | None) -> float | None:
    """CLI flag wins; falls back to REPROLAB_MAX_POD_SECONDS env var.

    Explicit None from the CLI (flag unset) triggers env fallback;
    an explicit float value (including 0.0, the kill-switch) is honored
    as-is. This is `is None`-checked rather than truthy-checked precisely
    so that ``--max-pod-seconds 0`` does not silently fall through to env.
    """
    if cli_value is not None:
        return cli_value
    env_value = os.environ.get("REPROLAB_MAX_POD_SECONDS")
    if env_value:
        return float(env_value)
    return None


def _max_invocations_from_arg(raw: str | None) -> dict[str, int]:
    if not raw:
        return {}
    caps: dict[str, int] = {}
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                "--max-invocations entries must be agent_id=count, "
                f"got {item!r}"
            )
        name, value = item.split("=", 1)
        caps[name.strip()] = int(value)
    return caps


if __name__ == "__main__":
    _module_main()
