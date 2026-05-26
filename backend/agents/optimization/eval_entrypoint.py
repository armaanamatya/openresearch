"""Subprocess entrypoint for one GEPA candidate × paper evaluation.

Invoked by ``ReproLabGEPAAdapter._run_one_eval`` via
``python -m backend.agents.optimization.eval_entrypoint``. Inputs come from
env vars (subprocess-safe; no argv parsing surface for injection):

- ``REPROLAB_GEPA_CANDIDATE_JSON`` — JSON dict of {component_id: text}
- ``REPROLAB_GEPA_PAPER_ID`` — arxiv id or local fixture path
- ``REPROLAB_GEPA_RUN_DIR`` — destination dir for run artifacts
- ``REPROLAB_GEPA_TASK_LM`` — model id passed to ``run_pipeline_rlm``
- ``REPROLAB_PRIMITIVE_CACHE_SURFACE_SALT`` — passthrough; the cache layer
  reads it via ``primitive_cache._active_surface_salt``
- ``REPROLAB_PRIMITIVE_CACHE`` — set to ``disabled`` for validation runs

Re-establishes ``PROMPT_OVERRIDES.use(candidate)`` for the duration of the
``run_pipeline_rlm`` call, then exits.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path


def _resolve_paper(paper_id: str) -> tuple[str, dict]:
    """Map ``paper_id`` to (project_id, workspace_claim_map)."""
    # arxiv id → use existing arxiv ingestion path
    if paper_id.replace(".", "").isdigit() or paper_id.startswith(("arxiv:", "arXiv:")):
        norm = paper_id.removeprefix("arxiv:").removeprefix("arXiv:")
        return f"gepa-{norm.replace('.', '-')}", {"arxiv_id": norm}
    # local path
    return f"gepa-{Path(paper_id).stem}", {"local_path": paper_id}


async def _main() -> int:
    candidate_json = os.environ.get("REPROLAB_GEPA_CANDIDATE_JSON")
    paper_id = os.environ.get("REPROLAB_GEPA_PAPER_ID")
    run_dir = os.environ.get("REPROLAB_GEPA_RUN_DIR")
    task_lm = os.environ.get("REPROLAB_GEPA_TASK_LM", "gpt-5")
    if not candidate_json or not paper_id or not run_dir:
        print(
            "eval_entrypoint: missing REPROLAB_GEPA_{CANDIDATE_JSON,PAPER_ID,RUN_DIR}",
            file=sys.stderr,
        )
        return 2

    candidate = json.loads(candidate_json)
    runs_root = Path(run_dir).parent  # one level up; project_id is the leaf
    project_id = Path(run_dir).name

    from backend.agents.optimization.prompt_overrides import PROMPT_OVERRIDES
    from backend.agents.rlm.run import run_pipeline_rlm

    logging.basicConfig(level=logging.INFO)
    _proj, claim_map = _resolve_paper(paper_id)
    with PROMPT_OVERRIDES.use(candidate):
        try:
            await run_pipeline_rlm(
                project_id=project_id,
                runs_root=runs_root,
                workspace_claim_map=claim_map,
                model=task_lm,
            )
        except Exception:
            logging.exception("eval_entrypoint: run_pipeline_rlm raised")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
