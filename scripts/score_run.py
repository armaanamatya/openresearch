#!/usr/bin/env python3
"""Score a completed RLM reproduction run against its PaperBench rubric.

Runs the post-run leaf scorer (``backend.evals.paperbench.leaf_scorer``) — the
*authoritative* PaperBench score — and writes the result back into the run's
``final_report.json``. The in-loop ``verify_against_rubric`` primitive is
fail-soft; this leaf scorer (flatten leaves -> batched LLM grading -> weighted
roll-up) is the score of record.

Usage:
    python scripts/score_run.py <run_dir> <paper_id>

Example:
    python scripts/score_run.py \\
        runs/pb_sequential-neural-score-estimation_1779390764 \\
        sequential-neural-score-estimation

Grading uses the Featherless Qwen root model — the same backend the RLM run
uses — so no extra API key is required beyond ``FEATHERLESS_API_KEY``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Score a completed RLM run against its PaperBench rubric.",
    )
    parser.add_argument("run_dir", help="runs/<id> directory of a completed run")
    parser.add_argument(
        "paper_id",
        help="PaperBench bundle id (directory under --bundles-root)",
    )
    parser.add_argument("--bundles-root", default="third_party/paperbench")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    rubric_path = Path(args.bundles_root) / args.paper_id / "rubric.json"

    if not (run_dir / "final_report.json").exists():
        print(f"error: no final_report.json in {run_dir}", file=sys.stderr)
        return 1
    if not rubric_path.exists():
        print(f"error: rubric not found: {rubric_path}", file=sys.stderr)
        return 1

    rubric_tree = json.loads(rubric_path.read_text(encoding="utf-8"))

    # Grading client — the Featherless Qwen root, same backend the RLM run uses.
    from backend.agents.rlm.models import resolve_root_model
    from backend.services.context.workspace.tools.openai_client import OpenAILlmClient

    root = resolve_root_model("qwen3-coder-featherless")
    bk = root.backend_kwargs
    llm_client = OpenAILlmClient(
        model=bk["model_name"], api_key=bk["api_key"], base_url=bk["base_url"]
    )

    from backend.evals.paperbench.leaf_scorer import (
        amend_final_report,
        score_reproduction,
    )

    print(f"scoring {run_dir.name}")
    print(f"  rubric : {rubric_path}")
    score = score_reproduction(rubric_tree, run_dir, llm_client)
    amend_final_report(run_dir, score)

    print(f"  overall_score : {score['overall_score']:.4f}")
    print(f"  leaves graded : {score['graded']}/{score['leaf_count']}")
    print(f"  written to    : {run_dir / 'final_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
