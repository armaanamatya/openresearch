"""rdr_paperbench.py — run the RDR rubric-driven harness on a vendored PaperBench bundle.

Usage example:
    .venv/bin/python scripts/rdr_paperbench.py sequential-neural-score-estimation \\
        --provider anthropic --sandbox docker --max-repair-iterations 3
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load .env so OPENAI_API_KEY / ANTHROPIC_API_KEY reach os.environ — the RDR
# primitive LLM client reads the process env, and nothing else in the repo
# loads .env for a local (non-docker) run.
load_dotenv()

# Surface logger.* output from the harness in the run log so a run is
# debuggable post-hoc. HTTP libraries are quieted to keep signal high.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
for _noisy in ("httpx", "httpcore", "openai", "urllib3", "docker"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


def _safe_dir_name(s: str) -> str:
    """Replace non-alphanumeric characters with underscores."""
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", s)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the RDR rubric-driven harness on a vendored PaperBench bundle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "paper_id",
        help="Directory name under --bundles-root (e.g. 'sequential-neural-score-estimation').",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model label stored on RunContext. None = library default.",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default=None,
        help="LLM provider for primitive calls.",
    )
    parser.add_argument(
        "--sandbox",
        default="docker",
        help="Sandbox backend: local | docker | runpod.",
    )
    parser.add_argument(
        "--max-repair-iterations",
        type=int,
        default=2,
        help="Maximum repair loops after initial cluster scoring.",
    )
    parser.add_argument(
        "--repair-target",
        type=float,
        default=0.6,
        help="Cluster-level score threshold below which a cluster is queued for repair.",
    )
    parser.add_argument(
        "--bundles-root",
        default="third_party/paperbench",
        help="Root directory that contains per-paper bundle subdirectories.",
    )
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Root directory where run artefacts are written.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help=(
            "Resume from the most-recently-modified run directory for the same paper_id, "
            "skipping clusters that have existing checkpoints. "
            "Passed automatically by rdr_paperbench_retry.sh after a watchdog kill (exit 124)."
        ),
    )
    parser.add_argument(
        "--project-id",
        dest="project_id",
        default=None,
        help="Explicit run directory name to use or resume (overrides auto-generated name).",
    )

    args = parser.parse_args()

    # ---- 1. Resolve bundle (validate it exists) --------------------------------
    from backend.evals.paperbench.bundle import load_paperbench_bundle

    bundle = load_paperbench_bundle(args.bundles_root, args.paper_id)

    meta = bundle.metadata()
    paper_title = meta.get("title", bundle.paper_id)

    # ---- 2. Build project_id ---------------------------------------------------
    runs_root = Path(args.runs_root).resolve()
    resume: bool = args.resume
    project_id_override: str | None = getattr(args, "project_id", None)

    if project_id_override:
        project_id = project_id_override
    elif resume:
        # Find most-recently-modified pb_rdr_<paper_id>_* directory.
        safe_pid = _safe_dir_name(bundle.paper_id)
        prefix = f"pb_rdr_{safe_pid}_"
        candidates = [
            d for d in runs_root.iterdir()
            if d.is_dir() and d.name.startswith(prefix)
        ] if runs_root.is_dir() else []
        if candidates:
            project_id = max(candidates, key=lambda d: d.stat().st_mtime).name
            print(f"[rdr_paperbench] --resume: reusing run dir {project_id}", file=sys.stderr)
        else:
            raw_id = f"pb_rdr_{bundle.paper_id}_{int(time.time())}"
            project_id = _safe_dir_name(raw_id)
            print(
                f"[rdr_paperbench] --resume: no prior dir found, starting fresh as {project_id}",
                file=sys.stderr,
            )
    else:
        raw_id = f"pb_rdr_{bundle.paper_id}_{int(time.time())}"
        project_id = _safe_dir_name(raw_id)

    print(f"project_id : {project_id}", file=sys.stderr)
    print(f"paper      : {paper_title}", file=sys.stderr)
    print(f"bundle     : {bundle.root}", file=sys.stderr)

    # ---- 3. SandboxMode --------------------------------------------------------
    from backend.agents.execution import (
        ensure_sandbox_mode_available,
        resolve_sandbox_mode,
    )
    from backend.services.runtime import SandboxRuntimeError

    sandbox_mode = resolve_sandbox_mode(args.sandbox, pipeline_mode="rdr")

    print(f"sandbox    : {sandbox_mode.value}", file=sys.stderr)

    try:
        ensure_sandbox_mode_available(sandbox_mode)
    except SandboxRuntimeError as exc:
        print(f"Sandbox preflight failed: {exc}", file=sys.stderr)
        return 2

    # ---- 4. Run ----------------------------------------------------------------
    from backend.agents.rdr.run import run_pipeline_rdr

    rdr_result = asyncio.run(
        run_pipeline_rdr(
            project_id,
            runs_root,
            paper_id=args.paper_id,
            provider=args.provider,
            model=args.model,
            sandbox_mode=sandbox_mode,
            max_repair_iterations=args.max_repair_iterations,
            repair_target=args.repair_target,
            bundles_root=str(bundle.root.parent),
            resume=resume,
        )
    )

    # ---- 5. Print result -------------------------------------------------------
    print("\n=== RDR Run Result ===")
    print(f"  project_id        : {rdr_result.project_id}")
    print(f"  status            : {rdr_result.status}")
    print(f"  rubric_score      : {rdr_result.rubric_score}")
    print(f"  clusters_total    : {rdr_result.clusters_total}")
    print(f"  clusters_failed   : {rdr_result.clusters_failed}")
    print(f"  repair_iterations : {rdr_result.repair_iterations}")
    print(f"  cost_usd          : {rdr_result.cost_usd}")
    print(f"  final_report_path : {rdr_result.final_report_path}")

    return 0 if rdr_result.status in ("completed", "partial") else 1


if __name__ == "__main__":
    sys.exit(main())
