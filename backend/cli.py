"""ReproLab CLI — drives ingestion, inspection, and reproduction.

  $ python -m backend.cli ingest <pdf-path>
      project_id=prj_..., parsed=N sections, sources=N, chunks=N,
      workspace=ws_..., variables=['claim_map']

  $ python -m backend.cli inspect <project_id> [--variable VAR]
      Prints the materialized workspace state.

  $ python -m backend.cli reproduce <pdf-path> [--mode offline|sdk]
      Full pipeline: ingest paper -> build workspace -> run agent pipeline.

This is a thin sequential composer: it wires Intake -> Parser ->
Indexer -> Workspace through a shared SqliteEventStore. The reproduce
command extends the pipeline into the agent layer.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

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
from backend.services.ingestion.intake.fetchers.arxiv import ArxivFetcher
from backend.services.ingestion.intake.fetchers.doi import DoiFetcher
from backend.services.ingestion.intake.fetchers.pdf_path import PdfPathFetcher
from backend.services.ingestion.parser import (
    ParserAppService,
    StartParsing,
)
from backend.services.ingestion.parser.pymupdf_parser import PyMuPdfParser

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
        store=store, parser=PyMuPdfParser(), runs_root=runs_root
    )
    discovery = ArtifactDiscoveryAppService(
        store=store,
        adapters=[RegexArtifactDiscoveryAdapter()],
    )
    indexer = IndexerAppService(store=store)
    workspace = WorkspaceAppService(store=store, indexer=indexer)
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


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate a completed pipeline run (reproduction + innovation)."""
    from backend.agents.orchestrator import PipelineState
    from backend.evals import EvalRunner, EvalStore
    from backend.evals.runner import print_innovation_report, print_reproduction_report

    runs_root = Path(args.runs_root)
    state = PipelineState.load_checkpoint(runs_root, args.project_id)
    if state is None:
        print(f"No pipeline state found for {args.project_id}", file=sys.stderr)
        return 2

    store = EvalStore(args.db)
    runner = EvalRunner(store=store)

    # Parse paper metrics
    paper_metrics: dict[str, float] = {}
    if args.paper_metrics:
        paper_metrics = json.loads(args.paper_metrics)
    elif state.experiment_artifacts and state.experiment_artifacts.metrics:
        # Default: use experiment's own metrics as ground truth (self-comparison)
        paper_metrics = {
            k: v for k, v in state.experiment_artifacts.metrics.items()
            if isinstance(v, (int, float))
        }

    # Run reproduction eval
    repro = runner.evaluate_reproduction(
        state, paper_metrics, version=args.version, paper_id=args.project_id,
    )
    print_reproduction_report(repro)

    # Run innovation eval if improvements exist
    if state.improvement_hypotheses and state.research_map:
        innov = runner.evaluate_innovation(state, version=args.version, paper_id=args.project_id)
        print_innovation_report(innov)

    store.close()

    # JSON output
    result = {
        "project_id": args.project_id,
        "version": args.version,
        "reproduction_composite": repro.composite_score(),
        "innovation_hypothesis_quality": (
            innov.mean_hypothesis_quality() if state.improvement_hypotheses else None
        ),
        "innovation_integrity_pass_rate": (
            innov.integrity_pass_rate() if state.improvement_hypotheses else None
        ),
    }
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def _resolve_sdk_providers(
    args: argparse.Namespace,
) -> tuple[str | None, str | None]:
    if getattr(args, "mode", None) != "sdk":
        return None, None

    from backend.agents.runtime import selected_provider, validate_provider_credentials

    provider = selected_provider(getattr(args, "provider", None))
    requested_verification_provider = getattr(args, "verification_provider", None)
    verification_provider = (
        selected_provider(requested_verification_provider)
        if requested_verification_provider
        else None
    )
    if provider == "openai":
        validate_provider_credentials(provider)
    if verification_provider == "openai":
        validate_provider_credentials(verification_provider)
    return provider, verification_provider


_REPRODUCE_DEFAULTS = {
    "database_url": get_settings().database_url,
    "runs_root": "runs",
    "source_kind": "auto",
    "agent": "default",
    "mode": "sdk",
    "model": None,
    "provider": None,
    "verification_provider": None,
    "hints": None,
    "n_paths": 3,
    "execution_mode": "efficient",
    "sandbox": "auto",
    "gpu_mode": "auto",
    "command_timeout": None,
    "allow_sandbox_network": False,
    "sandbox_platform": None,
    "sandbox_memory": None,
    "sandbox_cpus": None,
}


def _with_reproduce_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Backfill argparse defaults for generated Namespace callers."""
    for name, value in _REPRODUCE_DEFAULTS.items():
        if not hasattr(args, name):
            setattr(args, name, value)
    return args


def cmd_reproduce(args: argparse.Namespace) -> int:
    """Full pipeline: ingest a paper, build workspace, run agent pipeline."""
    import asyncio

    args = _with_reproduce_defaults(args)
    runs_root = Path(args.runs_root)

    # --- Phase 1: Ingest ---
    store, intake, parser, discovery, indexer, workspace = _make_services(
        args.database_url, runs_root
    )

    source = _source_from_cli(args.source, args.source_kind)
    print(f"[ingest 1/6] Registering project for {args.source}", file=sys.stderr)
    project_id = intake.register_project(RegisterProject(source=source))
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

    # Build workspace claim map for the agent pipeline
    workspace_claim_map = {
        "project_id": project_id,
        "entries": [
            {
                "source_id": name,
                "title": name,
                "excerpt": (
                    cited.value if isinstance(cited.value, str)
                    else json.dumps(cited.value) if cited.value is not None
                    else ""
                ),
            }
            for name, cited in view.variables.items()
        ],
    }

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Workspace ready — {len(view.variables)} variables", file=sys.stderr)
    provider, verification_provider = _resolve_sdk_providers(args)

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
    from backend.agents.execution import (
        ExecutionProfile,
        resolve_sandbox_mode,
    )

    execution_profile = ExecutionProfile.from_mode(
        args.execution_mode,
        command_timeout_seconds=args.command_timeout,
        sandbox_network_disabled=not args.allow_sandbox_network,
        sandbox_memory_limit=args.sandbox_memory,
        sandbox_cpus=args.sandbox_cpus,
        sandbox_platform=args.sandbox_platform,
        gpu_mode=getattr(args, "gpu_mode", "auto"),
    )
    sandbox_mode = resolve_sandbox_mode(args.sandbox, pipeline_mode=args.mode)
    print(
        f"Execution profile: {execution_profile.mode.value}; sandbox: {sandbox_mode.value}",
        file=sys.stderr,
    )

    try:
        if args.mode == "offline":
            from backend.agents.pipeline import run_pipeline_offline

            state = run_pipeline_offline(
                project_id, runs_root, workspace_claim_map,
                user_hints=user_hints,
                n_improvement_paths=args.n_paths,
                execution_profile=execution_profile,
                sandbox_mode=sandbox_mode,
                workspace_service=workspace,
                workspace_id=workspace_id,
            )
        else:
            from backend.agents.pipeline import run_pipeline_sdk

            state = asyncio.run(run_pipeline_sdk(
                project_id, runs_root, workspace_claim_map,
                model=args.model,
                provider=provider,
                verification_provider=verification_provider,
                user_hints=user_hints,
                n_improvement_paths=args.n_paths,
                execution_profile=execution_profile,
                sandbox_mode=sandbox_mode,
                workspace_service=workspace,
                workspace_id=workspace_id,
            ))
    finally:
        store.close()

    # Print final summary
    out_dir = runs_root / project_id
    result = {
        "project_id": project_id,
        "stage": state.stage.value,
        "output_dir": str(out_dir),
        "gates": {
            "gate_1": state.gate_1.passed if state.gate_1 else None,
            "gate_2": state.gate_2.passed if state.gate_2 else None,
            "gate_3": state.gate_3.passed if state.gate_3 else None,
        },
        "assumptions": len(state.assumption_ledger),
        "improvement_paths": len(state.path_results),
        "research_map": state.research_map is not None,
        "execution_mode": execution_profile.mode.value,
        "sandbox": sandbox_mode.value,
    }
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reprolab")
    parser.add_argument(
        "--database-url",
        default=get_settings().database_url,
        help="SQLite URL for the event store (defaults to REPROLAB_DATABASE_URL).",
    )
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Per-project blob directory root (default: ./runs).",
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

    evaluate = sub.add_parser("eval", help="Evaluate a completed pipeline run.")
    evaluate.add_argument("project_id", help="Project ID to evaluate.")
    evaluate.add_argument("--version", default="dev", help="Agent version label.")
    evaluate.add_argument(
        "--paper-metrics", default=None,
        help="JSON string of paper's reported metrics (e.g. '{\"mean_reward\": 500}').",
    )
    evaluate.add_argument("--db", default="evals.db", help="Eval store database path.")
    evaluate.set_defaults(func=cmd_eval)

    reproduce = sub.add_parser("reproduce", help="Full pipeline: ingest + agent pipeline.")
    reproduce.add_argument("source", help="PDF path, arXiv id/URL, or DOI/doi.org URL.")
    reproduce.add_argument(
        "--source-kind",
        choices=("auto", "pdf_path", "arxiv", "doi"),
        default="auto",
    )
    reproduce.add_argument("--agent", default="default", help="Agent name for the workspace.")
    reproduce.add_argument(
        "--mode", choices=("offline", "sdk"), default="sdk",
        help="Pipeline mode: 'sdk' uses LLM (default), 'offline' is deterministic.",
    )
    reproduce.add_argument("--model", default=None, help="Model override for SDK mode.")
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
        default="efficient",
        help="Execution profile: efficient keeps current bounded defaults; max raises budgets.",
    )
    reproduce.add_argument(
        "--sandbox",
        choices=("auto", "local", "docker", "runpod"),
        default="auto",
        help=(
            "Experiment backend: local runs commands on the host; docker is isolated; runpod uses a remote GPU Pod; "
            "auto selects Docker and never falls back to host-local execution."
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
    reproduce.set_defaults(func=cmd_reproduce)

    args = parser.parse_args(argv)
    return int(args.func(args))


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


if __name__ == "__main__":
    raise SystemExit(main())
