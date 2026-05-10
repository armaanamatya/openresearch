"""Adapt PaperBench bundles into the existing agent pipeline input shape."""

from __future__ import annotations

import json
from typing import Any

from backend.evals.paperbench.bundle import PaperBenchBundle
from backend.evals.paperbench.score import summarize_rubric


def bundle_to_workspace_claim_map(
    bundle: PaperBenchBundle,
    *,
    max_excerpt_chars: int = 120_000,
) -> dict[str, Any]:
    """Return the workspace-claim-map shape consumed by ``run_pipeline_sdk``.

    This keeps PaperBench intake outside the general arXiv/DOI/PDF source model
    while still feeding the pipeline the same high-level ``entries`` surface.
    """

    rubric = bundle.rubric()
    summary = summarize_rubric(rubric).to_dict()
    metadata = bundle.metadata()
    entries = [
        _entry("paper.md", "PaperBench paper markdown", bundle.read_paper_markdown(), max_excerpt_chars),
        _entry("addendum.md", "PaperBench addendum", bundle.read_addendum(), max_excerpt_chars),
        _entry(
            "task_instructions",
            "PaperBench task instructions",
            bundle.read_task_instructions(),
            max_excerpt_chars,
        ),
        _entry(
            "rubric_summary",
            "PaperBench rubric weighted summary",
            json.dumps(summary, indent=2),
            max_excerpt_chars,
        ),
        _entry(
            "rubric.json",
            "PaperBench rubric",
            json.dumps(rubric, indent=2),
            max_excerpt_chars,
        ),
    ]
    return {
        "project_id": f"paperbench_{bundle.paper_id}",
        "paperbench": {
            "paper_id": bundle.paper_id,
            "bundle_root": str(bundle.root),
            "metadata": metadata,
            "blacklist_entries": list(bundle.blacklist_entries()),
            "rubric_summary": summary,
        },
        "entries": [entry for entry in entries if entry["excerpt"]],
    }


def _entry(source_id: str, title: str, text: str, max_excerpt_chars: int) -> dict[str, str]:
    excerpt = text[:max_excerpt_chars]
    if len(text) > max_excerpt_chars:
        excerpt += "\n\n[TRUNCATED]"
    return {"source_id": source_id, "title": title, "excerpt": excerpt}
