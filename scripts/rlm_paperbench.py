"""rlm_paperbench.py — run the RLM orchestrator on a vendored PaperBench bundle.

Usage example:
    .venv/bin/python scripts/rlm_paperbench.py sequential-neural-score-estimation \\
        --provider anthropic --sandbox docker --max-wall-clock 3600
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

# Load .env so OPENAI_API_KEY / ANTHROPIC_API_KEY reach os.environ — the rlm
# root-model client and the primitive LLM client both read the process env,
# and nothing else in the repo loads .env for a local (non-docker) run.
load_dotenv()

# Surface logger.* output from the orchestrator and primitives in the run log
# so a run is debuggable post-hoc — without this only bare print()s appear, and
# diagnosing a failure means re-deriving it by hand. HTTP libraries are quieted
# to keep the signal high (httpx logs every LLM request at INFO).
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
        description="Run the RLM orchestrator on a vendored PaperBench bundle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "paper_id",
        help="Directory name under --bundles-root (e.g. 'sequential-neural-score-estimation').",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Root model key (e.g. 'claude-opus-4-5'). None = library default.",
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
        "--gpu-mode",
        default="prefer",
        help="GPU scheduling hint: auto | prefer | max | none.",
    )
    parser.add_argument(
        "--max-usd",
        type=float,
        default=None,
        help="Hard budget cap in USD.",
    )
    parser.add_argument(
        "--max-wall-clock",
        type=int,
        default=3600,
        help="Wall-clock timeout in seconds.",
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

    args = parser.parse_args()

    # ---- 1. Load the bundle --------------------------------------------------
    from backend.evals.paperbench.bundle import load_paperbench_bundle

    bundle = load_paperbench_bundle(args.bundles_root, args.paper_id)

    # Derive a human-readable title from config.yaml metadata (falls back to paper_id).
    meta = bundle.metadata()
    paper_title = meta.get("title", bundle.paper_id)

    # ---- 2. Build project_id ------------------------------------------------
    raw_id = f"pb_{bundle.paper_id}_{int(time.time())}"
    project_id = _safe_dir_name(raw_id)

    print(f"project_id : {project_id}", file=sys.stderr)
    print(f"paper      : {paper_title}", file=sys.stderr)
    print(f"bundle     : {bundle.root}", file=sys.stderr)

    # ---- 3. Workspace claim map ---------------------------------------------
    paper_text = bundle.paper_md_path.read_text(encoding="utf-8")
    workspace_claim_map: dict = {
        "project_id": project_id,
        "entries": [
            {
                "source_id": bundle.paper_id,
                "title": paper_title,
                "excerpt": paper_text,
            }
        ],
        "rubric_spec": bundle.rubric(),
    }

    # ---- 4. RunBudget -------------------------------------------------------
    run_budget = None
    if args.max_usd is not None or args.max_wall_clock is not None:
        from backend.agents.resilience import RunBudget

        run_budget = RunBudget(
            max_usd=args.max_usd,
            max_wall_clock_seconds=args.max_wall_clock,
        )

    # ---- 5. ExecutionProfile + SandboxMode ----------------------------------
    from backend.agents.execution import (
        ExecutionProfile,
        ensure_sandbox_mode_available,
        resolve_sandbox_mode,
    )
    from backend.services.runtime import SandboxRuntimeError

    execution_profile = ExecutionProfile.from_mode(
        "efficient",
        gpu_mode=args.gpu_mode,
    )
    sandbox_mode = resolve_sandbox_mode(args.sandbox, pipeline_mode="rlm")

    print(
        f"execution  : {execution_profile.mode.value}; sandbox: {sandbox_mode.value}",
        file=sys.stderr,
    )

    try:
        ensure_sandbox_mode_available(sandbox_mode)
    except SandboxRuntimeError as exc:
        print(f"Sandbox preflight failed: {exc}", file=sys.stderr)
        return 2

    # ---- 6. Run -------------------------------------------------------------
    from backend.agents.rlm.run import run_pipeline_rlm

    rlm_result = asyncio.run(
        run_pipeline_rlm(
            project_id,
            Path(args.runs_root),
            workspace_claim_map,
            model=args.model,
            provider=args.provider,
            run_budget=run_budget,
            sandbox_mode=sandbox_mode,
            execution_profile=execution_profile,
        )
    )

    # ---- 7. Print result ----------------------------------------------------
    print("\n=== RLM Run Result ===")
    print(f"  project_id        : {rlm_result.project_id}")
    print(f"  status            : {rlm_result.status}")
    print(f"  iterations        : {rlm_result.iterations}")
    print(f"  rubric_score      : {rlm_result.rubric_score}")
    print(f"  cost_usd          : {rlm_result.cost_usd}")
    print(f"  final_report_path : {rlm_result.final_report_path}")

    return 0 if rlm_result.status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
